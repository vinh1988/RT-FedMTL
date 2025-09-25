#!/bin/bash
# Complete environment setup script for FedBERT-LoRA (Ubuntu optimized)

set -e

echo "🔧 FedBERT-LoRA Environment Setup (Ubuntu)"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect Ubuntu and install system dependencies if needed
echo -e "${PURPLE}Checking Ubuntu system...${NC}"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        echo "✅ Ubuntu detected: $VERSION"
        
        # Check for required system packages
        echo -e "${BLUE}Checking system dependencies...${NC}"
        REQUIRED_PACKAGES="python3 python3-pip python3-venv python3-dev build-essential"
        MISSING_PACKAGES=""
        
        for package in $REQUIRED_PACKAGES; do
            if ! dpkg -l | grep -q "^ii  $package "; then
                MISSING_PACKAGES="$MISSING_PACKAGES $package"
            fi
        done
        
        if [ ! -z "$MISSING_PACKAGES" ]; then
            echo -e "${YELLOW}Installing missing system packages:$MISSING_PACKAGES${NC}"
            echo "This requires sudo access..."
            sudo apt update && sudo apt install -y $MISSING_PACKAGES
            echo -e "${GREEN}✅ System packages installed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Not Ubuntu, but will try to continue...${NC}"
    fi
fi

# Check Python version with better Ubuntu handling
echo -e "\n${BLUE}Checking Python version...${NC}"
if command_exists python3; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Found Python $python_version"
    
    # Check if Python version is >= 3.8
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        echo -e "${GREEN}✅ Python version is compatible${NC}"
        PYTHON_CMD="python3"
    else
        echo -e "${YELLOW}⚠️  Python 3.8+ recommended. Trying to install Python 3.9...${NC}"
        if command_exists apt; then
            sudo apt install -y python3.9 python3.9-venv python3.9-dev
            if command_exists python3.9; then
                PYTHON_CMD="python3.9"
                echo -e "${GREEN}✅ Python 3.9 installed${NC}"
            else
                echo -e "${RED}❌ Could not install Python 3.9. Please install manually.${NC}"
                exit 1
            fi
        else
            echo -e "${RED}❌ Python 3.8+ required. Please upgrade Python.${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}❌ Python3 not found. Installing...${NC}"
    sudo apt install -y python3 python3-pip python3-venv python3-dev
    PYTHON_CMD="python3"
fi

# Set default if not set
PYTHON_CMD=${PYTHON_CMD:-python3}

# Create virtual environment
echo -e "\n${BLUE}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        rm -rf venv
        $PYTHON_CMD -m venv venv
        echo -e "${GREEN}✅ Virtual environment recreated${NC}"
    else
        echo -e "${YELLOW}Using existing virtual environment${NC}"
    fi
else
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip and setuptools
echo -e "\n${BLUE}Upgrading pip and setuptools...${NC}"
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch - detect CUDA for Ubuntu
echo -e "\n${BLUE}Detecting CUDA availability...${NC}"
if command_exists nvidia-smi; then
    echo -e "${GREEN}CUDA detected! Installing PyTorch with CUDA support...${NC}"
    # Check CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2 || echo "11.8")
    echo "CUDA Version: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo -e "${YELLOW}No CUDA detected, installing CPU version...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo -e "\n${BLUE}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Install project in development mode
echo -e "\n${BLUE}Installing project in development mode...${NC}"
pip install -e .

# Create necessary directories
echo -e "\n${BLUE}Creating project directories...${NC}"
mkdir -p logs data checkpoints outputs cache

# Set up Hugging Face cache for Ubuntu
echo -e "\n${BLUE}Setting up Hugging Face cache...${NC}"
export HF_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
echo "export HF_HOME=\"./cache\"" >> venv/bin/activate
echo "export TRANSFORMERS_CACHE=\"./cache\"" >> venv/bin/activate

# Test installation
echo -e "\n${BLUE}Testing installation...${NC}"
python -c "
import torch
import transformers
import flwr
import peft
import datasets
print('✅ All core dependencies imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'Flower version: {flwr.__version__}')
print(f'PEFT version: {peft.__version__}')
"

# Test PyTorch and CUDA
echo -e "\n${BLUE}Testing PyTorch and CUDA...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')

if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Available GPUs: {torch.cuda.device_count()}')
    
    # Test CUDA tensor
    try:
        x = torch.randn(2, 2).cuda()
        y = torch.randn(2, 2).cuda()
        z = torch.mm(x, y)
        print('✅ CUDA tensor operations work')
    except Exception as e:
        print(f'⚠️  CUDA test failed: {e}')
else:
    print('ℹ️  CUDA not available - using CPU mode')
    # Test CPU operations
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    z = torch.mm(x, y)
    print('✅ CPU tensor operations work')
"

# Create activation helper script for Ubuntu
echo -e "\n${BLUE}Creating Ubuntu activation helper...${NC}"
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Ubuntu activation helper for FedBERT-LoRA

echo "🐧 Activating FedBERT-LoRA environment on Ubuntu..."
source venv/bin/activate

echo "✅ Environment activated!"
echo "System: $(lsb_release -d | cut -f2)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo "Device: CPU"
fi

echo ""
echo "Quick start commands:"
echo "  python examples/run_simple_experiment.py"
echo "  python examples/run_glue_experiment.py --task sst2"
echo "  ./run_experiment.sh 5 10 sst2"
EOF
chmod +x activate_env.sh

# Display system information
echo -e "\n${PURPLE}Ubuntu System Information:${NC}"
if command_exists lsb_release; then
    echo "OS: $(lsb_release -d | cut -f2)"
fi
echo "Kernel: $(uname -r)"
echo "Python: $($PYTHON_CMD --version)"
if command_exists nvidia-smi; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "CPU Cores: $(nproc)"

echo -e "\n${GREEN}🎉 Ubuntu environment setup completed successfully!${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Activate the environment: source venv/bin/activate"
echo "   OR use the Ubuntu helper: ./activate_env.sh"
echo "2. Run a simple test: python examples/run_simple_experiment.py"
echo "3. Run GLUE experiment: python examples/run_glue_experiment.py --task sst2"
echo "4. Use the convenience script: ./run_experiment.sh"
echo ""
echo -e "${BLUE}Ubuntu-specific features:${NC}"
echo "• System packages auto-installed via apt"
echo "• CUDA auto-detected and configured"
echo "• Hugging Face cache optimized"
echo "• Ubuntu activation helper created"
echo ""
echo -e "${GREEN}Happy federated learning on Ubuntu! 🐧🚀${NC}"
