#!/bin/bash
# Convenient script to run federated BERT experiments

set -e

echo "🚀 FedBERT-LoRA Experiment Runner"
echo "================================="

# Default parameters
NUM_CLIENTS=${1:-10}
NUM_ROUNDS=${2:-20}
TASK=${3:-"sst2"}
DEVICE=${4:-"cpu"}

echo "Configuration:"
echo "  - Task: $TASK"
echo "  - Clients: $NUM_CLIENTS"
echo "  - Rounds: $NUM_ROUNDS"
echo "  - Device: $DEVICE"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider running: source venv/bin/activate"
    echo ""
fi

# Run the experiment
echo "🏃 Starting experiment..."
python examples/run_glue_experiment.py \
    --task $TASK \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --seed 42 \
    --log_level INFO

echo ""
echo "✅ Experiment completed!"
echo ""
echo "📊 To run more experiments:"
echo "  ./run_experiment.sh 15 30 cola    # 15 clients, 30 rounds, CoLA task"
echo "  ./run_experiment.sh 5 10 mrpc     # 5 clients, 10 rounds, MRPC task"
