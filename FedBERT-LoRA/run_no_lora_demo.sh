#!/bin/bash

# Streaming Federated Learning WITHOUT LoRA Demo
# Pure Knowledge Distillation + Cross-Architecture Learning

echo "🚀 Starting Streaming Federated Learning WITHOUT LoRA"
echo "=============================================="
echo "Features:"
echo "- No parameter-efficient fine-tuning (full model training)"
echo "- Pure knowledge distillation for cross-architecture transfer"
echo "- Real-time WebSocket streaming"
echo "- Multi-task learning (SST-2, QQP, STS-B)"
echo "- Heterogeneous models: BERT-base (server) ↔ Tiny-BERT (clients)"
echo ""

# Configuration
PORT=8768
ROUNDS=3
EPOCHS=2
SAMPLES=100

# Kill any existing processes on the port
echo "🧹 Cleaning up existing processes..."
kill $(lsof -t -i:$PORT) 2>/dev/null || true
sleep 2

# Change to the correct directory
cd /home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Start server in background
echo "🌐 Starting server on port $PORT..."
timeout 180 python3 streaming_no_lora.py \
    --mode server \
    --port $PORT \
    --rounds $ROUNDS \
    --epochs $EPOCHS \
    --samples $SAMPLES &

# Wait for server to start
sleep 5

# Start SST-2 client
echo "👤 Starting SST-2 client..."
python3 streaming_no_lora.py \
    --mode client \
    --client_id client_sst2 \
    --task sst2 \
    --port $PORT \
    --rounds $ROUNDS \
    --epochs $EPOCHS \
    --samples $SAMPLES &

# Wait a bit before starting next client
sleep 3

# Start QQP client  
echo "👤 Starting QQP client..."
python3 streaming_no_lora.py \
    --mode client \
    --client_id client_qqp \
    --task qqp \
    --port $PORT \
    --rounds $ROUNDS \
    --epochs $EPOCHS \
    --samples $SAMPLES &

# Wait a bit before starting next client
sleep 3

# Start STS-B client
echo "👤 Starting STS-B client..."
python3 streaming_no_lora.py \
    --mode client \
    --client_id client_stsb \
    --task stsb \
    --port $PORT \
    --rounds $ROUNDS \
    --epochs $EPOCHS \
    --samples $SAMPLES &

echo ""
echo "🎯 All clients started!"
echo "📊 Monitor the training progress in real-time"
echo "⏱️  Training will complete in ~3-5 minutes"
echo ""
echo "🔍 What you'll see:"
echo "├── WebSocket connections establishing"
echo "├── Knowledge distillation between BERT-base ↔ Tiny-BERT"
echo "├── Multi-task learning across SST-2, QQP, STS-B"
echo "├── Real-time parameter updates (full model training)"
echo "└── Performance metrics for each task"
echo ""
echo "💡 Key Differences from LoRA version:"
echo "├── ✅ Full parameter training (no parameter efficiency)"
echo "├── ✅ Pure knowledge distillation (no LoRA complications)"
echo "├── ✅ Direct cross-architecture learning"
echo "└── ⚠️  Higher memory usage and training time"
echo ""

# Wait for all background processes
wait

echo ""
echo "🎉 No-LoRA Streaming Federated Learning Demo Complete!"
echo "📈 Check the training logs above for results"
