#!/bin/bash

# 🚀 Distributed MTL System Launcher
# Launches server and clients for distributed multi-task learning

echo "🚀 Distributed MTL System Launcher"
echo "=================================="

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Distributed MTL System:${NC}"
echo "  - Server: BERT-Base model for coordination"
echo "  - Clients: Each trains on specific dataset with MTL"
echo "  - No Federated Learning: Independent training with transfer learning"
echo ""

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Function to start server
start_server() {
    echo -e "${BLUE}🖥️ Starting Distributed MTL Server...${NC}"
    python distributed_mtl_system.py --mode server --rounds 1 &
    SERVER_PID=$!
    echo "✅ Server started (PID: $SERVER_PID)"
    echo "   Server logs: tail -f distributed_mtl.log"
}

# Function to start a client
start_client() {
    local client_id=$1
    local dataset=$2

    echo -e "${GREEN}🤖 Starting Client: $client_id (dataset: $dataset)...${NC}"

    # Start client in background
    python distributed_mtl_system.py --mode client --client_id "$client_id" --dataset "$dataset" &
    CLIENT_PID=$!
    echo "✅ Client $client_id started (PID: $CLIENT_PID)"
}

# Main menu
echo ""
echo -e "${YELLOW}📋 Launch Options:${NC}"
echo "1) Start complete system (Server + 3 Clients)"
echo "2) Start server only"
echo "3) Start clients only"
echo "4) Show all commands"
echo "5) Exit"
echo ""

read -p "Select option (1-5): " choice

case $choice in
    1)
        echo -e "${GREEN}🚀 Launching complete Distributed MTL System...${NC}"
        echo ""

        # Start server
        start_server

        # Wait for server to initialize
        echo "⏳ Waiting for server to initialize..."
        sleep 3

        # Start clients with different datasets
        start_client "client_sst2" "sst2"
        sleep 1
        start_client "client_qqp" "qqp"
        sleep 1
        start_client "client_stsb" "stsb"

        echo ""
        echo -e "${GREEN}🎉 Complete system launched!${NC}"
        echo ""
        echo -e "${YELLOW}📊 Monitoring:${NC}"
        echo "   - Server logs: tail -f distributed_mtl.log"
        echo "   - System status: ps aux | grep distributed_mtl"
        echo ""
        echo -e "${YELLOW}📈 Results:${NC}"
        echo "   - CSV metrics: ls -la distributed_mtl_results/"
        echo ""
        echo -e "${YELLOW}🔧 To stop all processes:${NC}"
        echo "   kill \$(ps aux | grep 'distributed_mtl_system.py' | grep -v grep | awk '{print \$2}')"
        echo ""
        echo -e "${GREEN}✅ System is running! Check the logs above for progress.${NC}"
        ;;

    2)
        start_server
        echo ""
        echo -e "${BLUE}✅ Server started! Now start clients manually:${NC}"
        echo "   python distributed_mtl_system.py --mode client --client_id client_sst2 --dataset sst2"
        echo "   python distributed_mtl_system.py --mode client --client_id client_qqp --dataset qqp"
        echo "   python distributed_mtl_system.py --mode client --client_id client_stsb --dataset stsb"
        ;;

    3)
        echo -e "${GREEN}🤖 Starting all clients...${NC}"
        start_client "client_sst2" "sst2"
        sleep 1
        start_client "client_qqp" "qqp"
        sleep 1
        start_client "client_stsb" "stsb"
        echo ""
        echo -e "${YELLOW}✅ All clients started! Make sure server is running.${NC}"
        ;;

    4)
        echo -e "${BLUE}📋 All Available Commands:${NC}"
        echo ""
        echo -e "${GREEN}Server Commands:${NC}"
        echo "   python distributed_mtl_system.py --mode server --rounds 1"
        echo ""
        echo -e "${GREEN}Client Commands:${NC}"
        echo "   python distributed_mtl_system.py --mode client --client_id client_sst2 --dataset sst2"
        echo "   python distributed_mtl_system.py --mode client --client_id client_qqp --dataset qqp"
        echo "   python distributed_mtl_system.py --mode client --client_id client_stsb --dataset stsb"
        echo ""
        echo -e "${YELLOW}Alternative Datasets:${NC}"
        echo "   - sst2: Sentiment Analysis (Classification)"
        echo "   - qqp: Question Pair Classification"
        echo "   - stsb: Semantic Similarity (Regression)"
        echo ""
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
