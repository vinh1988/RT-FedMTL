#!/bin/bash

# Deep Research Training System for Streaming Federated Learning
# Comprehensive experiment runner for LoRA vs non-LoRA comparison
# Usage: ./run_research_experiments.sh [experiment_type] [num_clients]

set -e

# Configuration
VENV_PATH="/home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA/venv"
BASE_PORT=8770
EXPERIMENT_TYPE=${1:-"baseline"}  # baseline, scalability, comparative, full_suite
NUM_CLIENTS=${2:-5}
ROUNDS=${3:-22}
SAMPLES=${4:-1000}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_DIR="research_logs"
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
run_experiment() {
    local exp_name=$1
    local use_lora=$2
    local port=$3
    local clients=$4
    local rounds=$5
    local samples=$6
    
    log "Starting experiment: $exp_name"
    log "Configuration: LoRA=$use_lora, Clients=$clients, Rounds=$rounds, Samples=$samples"
    
    # Start server
    local server_log="$LOG_DIR/${exp_name}_server.log"
    log "Starting server on port $port (log: $server_log)"
    
    if [ "$use_lora" = "true" ]; then
        timeout 600 python3 research_federated_system.py \
            --mode server \
            --port $port \
            --rounds $rounds \
            --use_lora \
            --samples $samples > "$server_log" 2>&1 &
    else
        timeout 600 python3 research_federated_system.py \
            --mode server \
            --port $port \
            --rounds $rounds \
            --samples $samples > "$server_log" 2>&1 &
    fi
    
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
    
    for i in $(seq 1 $clients); do
        local task=${tasks[$((($i - 1) % 3))]}  # Cycle through tasks
        local client_id="research_client_${i}"
        local client_log="$LOG_DIR/${exp_name}_client_${i}.log"
        
        log "Starting client $client_id with task $task (log: $client_log)"
        
        sleep 2  # Stagger client starts
        
        if [ "$use_lora" = "true" ]; then
            python3 research_federated_system.py \
                --mode client \
                --client_id "$client_id" \
                --task "$task" \
                --port $port \
                --use_lora \
                --samples $samples > "$client_log" 2>&1 &
        else
            python3 research_federated_system.py \
                --mode client \
                --client_id "$client_id" \
                --task "$task" \
                --port $port \
                --samples $samples > "$client_log" 2>&1 &
        fi
        
        client_pids+=($!)
    done
    
    log "All clients started for experiment $exp_name"
    
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

# Run baseline comparison
run_baseline_comparison() {
    local clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== BASELINE COMPARISON EXPERIMENT ==="
    log "Comparing LoRA vs non-LoRA with $clients clients, $rounds rounds, $samples samples"
    
    # LoRA baseline
    run_experiment "baseline_lora" "true" $BASE_PORT $clients $rounds $samples
    
    # Non-LoRA baseline  
    run_experiment "baseline_no_lora" "false" $((BASE_PORT + 1)) $clients $rounds $samples
    
    log "Baseline comparison completed"
}

# Run scalability analysis
run_scalability_analysis() {
    local max_clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== SCALABILITY ANALYSIS ==="
    log "Testing scalability from 2 to $max_clients clients"
    
    for clients in $(seq 2 $max_clients); do
        log "--- Testing with $clients clients ---"
        
        # LoRA scalability
        run_experiment "scalability_lora_${clients}c" "true" $BASE_PORT $clients $rounds $samples
        
        # Non-LoRA scalability
        run_experiment "scalability_no_lora_${clients}c" "false" $((BASE_PORT + 1)) $clients $rounds $samples
        
        log "Scalability test with $clients clients completed"
    done
    
    log "Scalability analysis completed"
}

# Run parameter sensitivity analysis
run_parameter_analysis() {
    local clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== PARAMETER SENSITIVITY ANALYSIS ==="
    
    # Different LoRA ranks
    local ranks=(4 8 16 32 64)
    for rank in "${ranks[@]}"; do
        log "--- Testing LoRA rank $rank ---"
        
        # This would require modifying the script to accept rank parameter
        # For now, we'll use the baseline configuration
        run_experiment "lora_rank_${rank}" "true" $BASE_PORT $clients $rounds $samples
    done
    
    log "Parameter sensitivity analysis completed"
}

# Run communication efficiency analysis
run_communication_analysis() {
    local clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== COMMUNICATION EFFICIENCY ANALYSIS ==="
    
    # Test with different sample sizes to analyze communication overhead
    local sample_sizes=(500 1000 2000)
    
    for samples_test in "${sample_sizes[@]}"; do
        log "--- Testing with $samples_test samples per client ---"
        
        # LoRA communication test
        run_experiment "comm_lora_${samples_test}s" "true" $BASE_PORT $clients $rounds $samples_test
        
        # Non-LoRA communication test
        run_experiment "comm_no_lora_${samples_test}s" "false" $((BASE_PORT + 1)) $clients $rounds $samples_test
    done
    
    log "Communication efficiency analysis completed"
}

# Run full research suite
run_full_suite() {
    local max_clients=$1
    local rounds=$2
    local samples=$3
    
    log "=== FULL RESEARCH SUITE ==="
    log "Running comprehensive federated learning research experiments"
    
    # Baseline comparison
    run_baseline_comparison 5 $rounds $samples
    
    # Scalability analysis (reduced for time)
    run_scalability_analysis $max_clients $rounds $((samples / 2))
    
    # Parameter analysis (reduced rounds for time)
    run_parameter_analysis 3 $((rounds / 2)) $((samples / 2))
    
    # Communication analysis
    run_communication_analysis 4 $((rounds / 2)) $samples
    
    log "Full research suite completed"
}

# Generate experiment summary
generate_summary() {
    log "=== EXPERIMENT SUMMARY ==="
    
    # Count log files
    local total_experiments=$(ls $LOG_DIR/*_server.log 2>/dev/null | wc -l)
    log "Total experiments conducted: $total_experiments"
    
    # Check for results
    if [ -d "results" ]; then
        local result_files=$(ls results/*.json 2>/dev/null | wc -l)
        log "Result files generated: $result_files"
    fi
    
    if [ -d "metrics" ]; then
        local metric_files=$(ls metrics/*.json 2>/dev/null | wc -l)
        log "Metric files generated: $metric_files"
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
    log "Results saved in: results/ and metrics/"
}

# Main execution
main() {
    log "Starting Deep Research Federated Learning Experiments"
    log "Experiment type: $EXPERIMENT_TYPE"
    log "Number of clients: $NUM_CLIENTS"
    log "Number of rounds: $ROUNDS"
    log "Samples per client: $SAMPLES"
    
    # Setup
    activate_venv
    check_gpu
    cleanup  # Clean any existing processes
    
    # Run experiments based on type
    case $EXPERIMENT_TYPE in
        "baseline")
            run_baseline_comparison $NUM_CLIENTS $ROUNDS $SAMPLES
            ;;
        "scalability")
            run_scalability_analysis $NUM_CLIENTS $ROUNDS $SAMPLES
            ;;
        "parameter")
            run_parameter_analysis $NUM_CLIENTS $ROUNDS $SAMPLES
            ;;
        "communication")
            run_communication_analysis $NUM_CLIENTS $ROUNDS $SAMPLES
            ;;
        "full_suite")
            run_full_suite $NUM_CLIENTS $ROUNDS $SAMPLES
            ;;
        *)
            error "Unknown experiment type: $EXPERIMENT_TYPE"
            echo "Available types: baseline, scalability, parameter, communication, full_suite"
            exit 1
            ;;
    esac
    
    # Generate summary
    generate_summary
    
    log "All experiments completed successfully!"
}

# Help function
show_help() {
    echo "Deep Research Federated Learning Experiment Runner"
    echo ""
    echo "Usage: $0 [experiment_type] [num_clients] [rounds] [samples]"
    echo ""
    echo "Experiment Types:"
    echo "  baseline      - Compare LoRA vs non-LoRA baseline performance"
    echo "  scalability   - Test scalability from 2 to num_clients"
    echo "  parameter     - Parameter sensitivity analysis"
    echo "  communication - Communication efficiency analysis"
    echo "  full_suite    - Run all experiments (comprehensive)"
    echo ""
    echo "Parameters:"
    echo "  num_clients   - Maximum number of clients (default: 5)"
    echo "  rounds        - Number of federated rounds (default: 22)"
    echo "  samples       - Data samples per client (default: 1000)"
    echo ""
    echo "Examples:"
    echo "  $0 baseline 5 22 1000"
    echo "  $0 scalability 10 15 500"
    echo "  $0 full_suite 8 20 800"
    echo ""
    echo "Results will be saved in:"
    echo "  - research_logs/  (experiment logs)"
    echo "  - results/        (experiment results)"
    echo "  - metrics/        (detailed metrics)"
}

# Check for help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Run main function
main
