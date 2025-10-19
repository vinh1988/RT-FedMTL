#!/bin/bash

# 🚀 Quick Launcher for Optimized MTL Federated Learning
# This script helps launch the complete system with proper sequencing

set -e  # Exit on any error

echo "🚀 Quick MTL Federated Learning Launcher"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if optimized files exist
if [ ! -f "optimized_mtl_federated.py" ]; then
    echo "❌ Optimized system not found!"
    echo "   Please ensure optimized_mtl_federated.py exists"
    exit 1
fi

echo "✅ Environment ready"

# Function to start server in background
start_server() {
    echo "🖥️ Starting MTL Federated Server..."
    python optimized_mtl_federated.py --mode server --rounds 3 --total_clients 3 &
    SERVER_PID=$!
    echo "✅ Server started (PID: $SERVER_PID)"
    echo "   Server logs: tail -f optimized_mtl_federated.log"
}

# Function to start a client
start_client() {
    local client_id=$1
    local tasks=$2

    echo "🤖 Starting Client: $client_id ($tasks)..."

    # Start client in background
    python optimized_mtl_federated.py --mode client --client_id "$client_id" --tasks $tasks &
    CLIENT_PID=$!
    echo "✅ Client $client_id started (PID: $CLIENT_PID)"
}

# Main menu
echo ""
echo "📋 Launch Options:"
echo "1) Start complete system (Server + 3 Clients)"
echo "2) Start server only"
echo "3) Start clients only"
echo "4) Show all commands"
echo "5) Exit"
echo ""

read -p "Select option (1-5): " choice

case $choice in
    1)
        echo "🚀 Launching complete MTL Federated Learning System..."
        echo ""

        # Start server
        start_server

        # Wait longer for server to fully initialize
        echo "⏳ Waiting for server to fully initialize..."
        sleep 5  # Increased from 3 to 5 seconds

        # Start clients one by one with delays
        start_client "client_1" "sst2 qqp stsb"
        sleep 2  # Wait between clients
        start_client "client_2" "sst2 qqp stsb"
        sleep 2  # Wait between clients
        start_client "client_3" "sst2 qqp stsb"

        echo ""
        echo "🎉 Complete system launched!"
        echo ""
        echo "📊 Monitoring:"
        echo "   - Server logs: tail -f optimized_mtl_federated.log"
        echo "   - System status: ps aux | grep optimized_mtl"
        echo ""
        echo "🔧 To stop all processes:"
        echo "   kill $SERVER_PID $(pgrep -f 'optimized_mtl_federated.py' | grep -v $$)"
        echo ""
        echo "✅ System is running! Check the logs above for progress."
        ;;

    2)
        start_server
        echo ""
        echo "✅ Server started! Now start clients manually:"
        echo "   python optimized_mtl_federated.py --mode client --client_id client_1 --tasks sst2 qqp stsb"
        ;;

    3)
        echo "🤖 Starting all clients..."
        start_client "client_1" "sst2 qqp stsb"
        sleep 1
        start_client "client_2" "sst2 qqp stsb"
        sleep 1
        start_client "client_3" "sst2 qqp stsb"
        echo ""
        echo "✅ All clients started! Make sure server is running."
        ;;

    4)
        echo "📋 All Available Commands:"
        echo ""
        cat run_optimized_mtl.sh
        ;;

    5)
        echo "👋 Goodbye!"
        exit 0
        ;;

    *)
        echo "❌ Invalid option. Please select 1-5."
        exit 1
        ;;
esac
