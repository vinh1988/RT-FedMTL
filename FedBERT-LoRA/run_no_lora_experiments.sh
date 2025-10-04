#!/bin/bash

# Non-LoRA Federated Learning Experiments
# Focus: Non-IID data metrics and client participation analysis (2-10 clients)
# Output: CSV metrics for research analysis

set -e

# Configuration
VENV_PATH="/home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA/venv"
BASE_PORT=8771
EXPERIMENT_TYPE=${1:-"scalability"}  # scalability, non_iid, participation
MAX_CLIENTS=${2:-10}
ROUNDS=${3:-22}
SAMPLES=${4:-1000}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_DIR="no_lora_logs"
mkdir -p $LOG_DIR

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Cleanup function
cleanup() {
    log "Cleaning up background processes..."
    
    # Kill processes on research ports
    for port in $(seq $BASE_PORT $((BASE_PORT + 10))); do
        if lsof -ti:$port > /dev/null 2>&1; then
            log "Killing processes on port $port"
            kill $(lsof -ti:$port) 2>/dev/null || true
        fi
    done
    
    # Wait for cleanup
    sleep 2
    log "Cleanup completed"
}

# Set trap for cleanup
trap cleanup EXIT

# Activate virtual environment
activate_venv() {
    log "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    
    # Verify Python environment
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
}

# Check GPU availability
check_gpu() {
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        log "GPU available: $(python3 -c "import torch; print(torch.cuda.get_device_name())")"
    else
        warn "No GPU available, using CPU"
    fi
}

# Run single experiment
run_no_lora_experiment() {
    local exp_name=$1
    local num_clients=$2
    local port=$3
    local rounds=$4
    local samples=$5
    local distribution=$6
    local alpha=$7
    
    log "Starting Non-LoRA experiment: $exp_name"
    log "Configuration: Clients=$num_clients, Rounds=$rounds, Samples=$samples, Distribution=$distribution"
    
    # Start server
    local server_log="$LOG_DIR/${exp_name}_server.log"
    log "Starting server on port $port (log: $server_log)"
    
    timeout 1200 python3 no_lora_federated_system.py \
        --mode server \
        --port $port \
        --rounds $rounds \
        --samples $samples \
        --total_clients $num_clients \
        --distribution $distribution \
        --alpha $alpha > "$server_log" 2>&1 &
    
    local server_pid=$!
    sleep 5
    
    # Check if server started successfully
    if ! kill -0 $server_pid 2>/dev/null; then
        error "Server failed to start for experiment $exp_name"
        return 1
    fi
    
    # Start clients
    local client_pids=()
    local tasks=("sst2" "qqp" "stsb")
    
    for i in $(seq 1 $num_clients); do
        local task=${tasks[$((($i - 1) % 3))]}  # Cycle through tasks
        local client_id="no_lora_client_${i}"
        local client_log="$LOG_DIR/${exp_name}_client_${i}.log"
        
        log "Starting client $client_id with task $task (log: $client_log)"
        
        sleep 2  # Stagger client starts
        
        python3 no_lora_federated_system.py \
            --mode client \
            --client_id "$client_id" \
            --task "$task" \
            --port $port \
            --samples $samples \
            --total_clients $num_clients \
            --distribution $distribution \
            --alpha $alpha > "$client_log" 2>&1 &
        
        client_pids+=($!)
    done
    
    log "All $num_clients clients started for experiment $exp_name"
    
    # Wait for server to complete
    log "Waiting for experiment $exp_name to complete..."
    wait $server_pid
    local server_exit_code=$?
    
    # Clean up client processes
    for pid in "${client_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
        fi
    done
    
    if [ $server_exit_code -eq 0 ]; then
        log "Experiment $exp_name completed successfully"
    else
        warn "Experiment $exp_name completed with exit code $server_exit_code"
    fi
    
    # Brief pause between experiments
    sleep 5
    
    return $server_exit_code
}

# Run scalability analysis (2-10 clients)
run_scalability_analysis() {
    local max_clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== CLIENT SCALABILITY ANALYSIS (2-$max_clients clients) ==="
    log "Testing Non-IID federated learning scalability"
    
    for clients in $(seq 2 $max_clients); do
        log "--- Testing with $clients clients ---"
        
        # Use unique port for each client configuration to avoid conflicts
        local port=$((8800 + clients))  # Simple port assignment: 8802, 8803, 8804, 8805
        
        # Extra cleanup before each experiment
        cleanup
        sleep 3
        
        # Non-IID scalability test
        run_no_lora_experiment "scalability_${clients}c" $clients $port $rounds $samples "non_iid" 0.5
        
        if [ $? -eq 0 ]; then
            log "Scalability test with $clients clients completed"
        else
            error "Scalability test with $clients clients failed"
        fi
        
        # Cleanup after each experiment
        cleanup
        sleep 2
    done
    
    log "Scalability analysis completed"
}

# Run Non-IID parameter sweep
run_non_iid_analysis() {
    local clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== NON-IID PARAMETER ANALYSIS ==="
    log "Testing different Non-IID parameters with $clients clients"
    
    # Different alpha values for Dirichlet distribution
    local alphas=(0.1 0.3 0.5 1.0 2.0)
    
    for alpha in "${alphas[@]}"; do
        log "--- Testing Non-IID alpha=$alpha ---"
        
        run_no_lora_experiment "non_iid_alpha_${alpha}" $clients $BASE_PORT $rounds $samples "non_iid" $alpha
    done
    
    # Test pathological Non-IID
    log "--- Testing Pathological Non-IID ---"
    run_no_lora_experiment "pathological_non_iid" $clients $BASE_PORT $rounds $samples "pathological" 0.5
    
    # Test IID for comparison
    log "--- Testing IID (baseline) ---"
    run_no_lora_experiment "iid_baseline" $clients $BASE_PORT $rounds $samples "iid" 0.5
    
    log "Non-IID parameter analysis completed"
}

# Run participation analysis
run_participation_analysis() {
    local max_clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== CLIENT PARTICIPATION ANALYSIS ==="
    log "Analyzing client participation patterns with varying client counts"
    
    # Test with different client counts and longer rounds for participation patterns
    local client_counts=(3 5 7 10)
    local participation_rounds=$((rounds + 10))  # More rounds to see participation patterns
    
    for clients in "${client_counts[@]}"; do
        log "--- Participation analysis with $clients clients ---"
        
        # Non-IID participation test
        run_no_lora_experiment "participation_${clients}c" $clients $BASE_PORT $participation_rounds $samples "non_iid" 0.3
    done
    
    log "Participation analysis completed"
}

# Run comprehensive Non-IID study
run_comprehensive_study() {
    local max_clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== COMPREHENSIVE NON-IID FEDERATED LEARNING STUDY ==="
    log "Running complete analysis: scalability + Non-IID + participation"
    
    # Scalability analysis (reduced rounds for time)
    log "Phase 1: Scalability Analysis"
    run_scalability_analysis $max_clients $((rounds - 5)) $((samples / 2))
    
    # Non-IID parameter analysis
    log "Phase 2: Non-IID Parameter Analysis"
    run_non_iid_analysis 5 $rounds $samples
    
    # Participation analysis
    log "Phase 3: Client Participation Analysis"
    run_participation_analysis $max_clients $rounds $samples
    
    log "Comprehensive study completed"
}

# Generate experiment summary
generate_summary() {
    log "=== EXPERIMENT SUMMARY ==="
    
    # Count log files
    local total_experiments=$(ls $LOG_DIR/*_server.log 2>/dev/null | wc -l)
    log "Total experiments conducted: $total_experiments"
    
    # Check for CSV results
    if [ -d "no_lora_results" ]; then
        local csv_files=$(ls no_lora_results/*.csv 2>/dev/null | wc -l)
        log "CSV result files generated: $csv_files"
        
        if [ $csv_files -gt 0 ]; then
            log "CSV files generated:"
            ls no_lora_results/*.csv 2>/dev/null | while read file; do
                log "  - $(basename $file)"
            done
        fi
    fi
    
    # Check for errors
    local error_count=$(grep -l "ERROR\|Exception\|Traceback" $LOG_DIR/*.log 2>/dev/null | wc -l)
    if [ $error_count -gt 0 ]; then
        warn "Found errors in $error_count log files"
        log "Check the following files for errors:"
        grep -l "ERROR\|Exception\|Traceback" $LOG_DIR/*.log 2>/dev/null || true
    else
        log "No errors detected in log files"
    fi
    
    log "Experiment logs saved in: $LOG_DIR"
    log "CSV results saved in: no_lora_results/"
}

# Main execution
main() {
    log "Starting Non-LoRA Federated Learning Experiments"
    log "Experiment type: $EXPERIMENT_TYPE"
    log "Max clients: $MAX_CLIENTS"
    log "Number of rounds: $ROUNDS"
    log "Samples per client: $SAMPLES"
    
    # Setup
    activate_venv
    check_gpu
    cleanup  # Clean any existing processes
    
    # Run experiments based on type
    case $EXPERIMENT_TYPE in
        "scalability")
            run_scalability_analysis $MAX_CLIENTS $ROUNDS $SAMPLES
            ;;
        "non_iid")
            run_non_iid_analysis 5 $ROUNDS $SAMPLES
            ;;
        "participation")
            run_participation_analysis $MAX_CLIENTS $ROUNDS $SAMPLES
            ;;
        "comprehensive")
            run_comprehensive_study $MAX_CLIENTS $ROUNDS $SAMPLES
            ;;
        *)
            error "Unknown experiment type: $EXPERIMENT_TYPE"
            echo "Available types: scalability, non_iid, participation, comprehensive"
            exit 1
            ;;
    esac
    
    # Generate summary
    generate_summary
    
    log "All Non-LoRA experiments completed successfully!"
}

# Help function
show_help() {
    echo "Non-LoRA Federated Learning Experiment Runner"
    echo ""
    echo "Usage: $0 [experiment_type] [max_clients] [rounds] [samples]"
    echo ""
    echo "Experiment Types:"
    echo "  scalability    - Test scalability from 2 to max_clients"
    echo "  non_iid        - Non-IID parameter analysis (alpha sweep)"
    echo "  participation  - Client participation pattern analysis"
    echo "  comprehensive  - Run all experiments (complete study)"
    echo ""
    echo "Parameters:"
    echo "  max_clients    - Maximum number of clients (default: 10)"
    echo "  rounds         - Number of federated rounds (default: 22)"
    echo "  samples        - Data samples per client (default: 1000)"
    echo ""
    echo "Examples:"
    echo "  $0 scalability 10 22 1000    # Test 2-10 clients"
    echo "  $0 non_iid 5 22 1000         # Non-IID parameter sweep"
    echo "  $0 participation 8 25 800    # Participation analysis"
    echo "  $0 comprehensive 10 20 500   # Complete study"
    echo ""
    echo "Output:"
    echo "  - no_lora_logs/              (experiment logs)"
    echo "  - no_lora_results/           (CSV metrics files)"
    echo ""
    echo "Key Metrics Generated:"
    echo "  - Client participation rates and patterns"
    echo "  - Non-IID data distribution analysis"
    echo "  - Scalability performance (2-10 clients)"
    echo "  - Heterogeneity measures (KL divergence, JS divergence)"
    echo "  - Communication and computational efficiency"
}

# Check for help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Run main function
main
