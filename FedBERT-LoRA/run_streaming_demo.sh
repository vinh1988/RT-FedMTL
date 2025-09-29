#!/bin/bash

# Streaming GLUE Federated Learning Demo Script
# This script starts the server and all three clients automatically

echo "🎯 Starting Streaming GLUE Federated Learning Demo"
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

# Kill any existing processes on port 8765
echo "🧹 Cleaning up existing processes..."
lsof -ti:8765 | xargs kill -9 2>/dev/null || true
sleep 2

# Start server in background
echo "🌐 Starting federated server..."
python3 streaming_glue_federated.py --mode server --rounds 3 &
SERVER_PID=$!
sleep 5

# Start clients in background
echo "👤 Starting SST-2 client..."
python3 streaming_glue_federated.py --mode client --client_id client_sst2 --task sst2 &
CLIENT1_PID=$!
sleep 2

echo "👤 Starting QQP client..."
python3 streaming_glue_federated.py --mode client --client_id client_qqp --task qqp &
CLIENT2_PID=$!
sleep 2

echo "👤 Starting STS-B client..."
python3 streaming_glue_federated.py --mode client --client_id client_stsb --task stsb &
CLIENT3_PID=$!

echo ""
echo "🚀 All components started!"
echo "   Server PID: $SERVER_PID"
echo "   SST-2 Client PID: $CLIENT1_PID"
echo "   QQP Client PID: $CLIENT2_PID"
echo "   STS-B Client PID: $CLIENT3_PID"
echo ""
echo "📡 Watch the streaming federated learning in action!"
echo "   Press Ctrl+C to stop all processes"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    kill $SERVER_PID $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID 2>/dev/null
    wait
    echo "✅ All processes stopped"
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Wait for all background processes
wait
