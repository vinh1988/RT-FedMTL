#!/bin/bash

# Fixed Streaming GLUE Federated Learning Demo
echo "🎯 Starting FIXED Streaming GLUE Federated Learning Demo"
echo "======================================================="

# Activate environment
source venv/bin/activate

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
lsof -ti:8766 | xargs kill -9 2>/dev/null || true
sleep 2

# Start server in background
echo "🌐 Starting fixed federated server..."
python3 fixed_streaming_glue.py --mode server --port 8766 --rounds 3 &
SERVER_PID=$!
sleep 5

# Start clients
echo "👤 Starting SST-2 client..."
python3 fixed_streaming_glue.py --mode client --client_id client_sst2 --task sst2 --port 8766 &
CLIENT1_PID=$!
sleep 2

echo "👤 Starting QQP client..."
python3 fixed_streaming_glue.py --mode client --client_id client_qqp --task qqp --port 8766 &
CLIENT2_PID=$!
sleep 2

echo "👤 Starting STS-B client..."
python3 fixed_streaming_glue.py --mode client --client_id client_stsb --task stsb --port 8766 &
CLIENT3_PID=$!

echo ""
echo "🚀 All components started!"
echo "   Server PID: $SERVER_PID"
echo "   SST-2 Client PID: $CLIENT1_PID"
echo "   QQP Client PID: $CLIENT2_PID"
echo "   STS-B Client PID: $CLIENT3_PID"
echo ""
echo "📡 Watch the fixed streaming federated learning!"
echo "   Press Ctrl+C to stop all processes"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    kill $SERVER_PID $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID 2>/dev/null
    wait
    echo "✅ All processes stopped"
}

# Set trap for cleanup
trap cleanup EXIT

# Wait for processes
wait
