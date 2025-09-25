#!/bin/bash
# Ubuntu-optimized environment setup script for FedBERT-LoRA

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

# Function to install system packages
install_system_deps() {
    echo -e "${BLUE}Checking system dependencies...${NC}"
    
    # Check if we need to update package list
    if [ ! -f /var/lib/apt/periodic/update-success-stamp ] || [ $(find /var/lib/apt/periodic/update-success-stamp -mtime +1) ]; then
        echo -e "${YELLOW}Updating package list...${NC}"
        sudo apt update
    fi
    
    # Required system packages
    REQUIRED_PACKAGES="python3 python3-pip python3-venv python3-dev build-essential git curl"
    MISSING_PACKAGES=""
    
    for package in $REQUIRED_PACKAGES; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            MISSING_PACKAGES="$MISSING_PACKAGES $package"
        fi
    done
    
    if [ ! -z "$MISSING_PACKAGES" ]; then
        echo -e "${YELLOW}Installing missing system packages:$MISSING_PACKAGES${NC}"
        sudo apt install -y $MISSING_PACKAGES
        echo -e "${GREEN}✅ System packages installed${NC}"
    else
        echo -e "${GREEN}✅ All system packages already installed${NC}"
    fi
}

# Detect Ubuntu version
echo -e "${PURPLE}Detecting Ubuntu version...${NC}"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $NAME $VERSION"
    UBUNTU_VERSION=$VERSION_ID
else
    echo -e "${RED}❌ Cannot detect Ubuntu version${NC}"
    exit 1
fi

# Check if running on Ubuntu
if [[ "$ID" != "ubuntu" ]]; then
    echo -e "${YELLOW}⚠️  This script is optimized for Ubuntu, but will try to continue...${NC}"
fi

# Install system dependencies
echo -e "\n${BLUE}Would you like to install/update system dependencies? (recommended)${NC}"
read -p "Install system packages? (Y/n): " install_sys
if [[ $install_sys =~ ^[Nn]$ ]]; then
    echo -e "${YELLOW}Skipping system package installation${NC}"
else
    install_system_deps
fi

# Check Python version
echo -e "\n${BLUE}Checking Python version...${NC}"
if command_exists python3; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Found Python $python_version"
    
    # Check if Python version is >= 3.8
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        echo -e "${GREEN}✅ Python version is compatible${NC}"
    else
        echo -e "${RED}❌ Python 3.8+ required. Current version: $python_version${NC}"
        echo -e "${YELLOW}Installing Python 3.9...${NC}"
        sudo apt install -y python3.9 python3.9-venv python3.9-dev
        # Create symlink if needed
        if [ ! -f /usr/bin/python3.9 ]; then
            echo -e "${RED}❌ Python 3.9 installation failed${NC}"
            exit 1
        fi
        PYTHON_CMD="python3.9"
    fi
else
    echo -e "${RED}❌ Python3 not found. Installing...${NC}"
    sudo apt install -y python3 python3-pip python3-venv python3-dev
    PYTHON_CMD="python3"
fi

# Set Python command
PYTHON_CMD=${PYTHON_CMD:-python3}

# Check if venv module is available
echo -e "\n${BLUE}Checking Python venv module...${NC}"
if ! $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
    echo -e "${YELLOW}Installing python3-venv...${NC}"
    sudo apt install -y python3-venv
fi

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

# Install PyTorch - detect if CUDA is available
echo -e "\n${BLUE}Detecting CUDA availability...${NC}"
if command_exists nvidia-smi; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "CUDA detected: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "11.8" ]] || [[ "$CUDA_VERSION" > "11.8" ]]; then
        echo -e "${GREEN}Installing PyTorch with CUDA 11.8 support...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" == "12."* ]]; then
        echo -e "${GREEN}Installing PyTorch with CUDA 12.1 support...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo -e "${YELLOW}CUDA version not optimal, installing CPU version...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo -e "${YELLOW}No CUDA detected, installing CPU version...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo -e "\n${BLUE}Installing project dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

# Install project in development mode
echo -e "\n${BLUE}Installing project in development mode...${NC}"
if [ -f "setup.py" ]; then
    pip install -e .
else
    echo -e "${YELLOW}⚠️  setup.py not found, skipping development install${NC}"
fi

# Create necessary directories
echo -e "\n${BLUE}Creating project directories...${NC}"
mkdir -p logs data checkpoints outputs cache

# Set up cache directories for Hugging Face
echo -e "\n${BLUE}Setting up Hugging Face cache...${NC}"
export HF_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
echo "export HF_HOME=\"./cache\"" >> venv/bin/activate
echo "export TRANSFORMERS_CACHE=\"./cache\"" >> venv/bin/activate

# Test installation
echo -e "\n${BLUE}Testing installation...${NC}"
python -c "
try:
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
    print(f'Datasets version: {datasets.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test PyTorch functionality
echo -e "\n${BLUE}Testing PyTorch functionality...${NC}"
python -c "
import torch
import numpy as np

# Test basic tensor operations
x = torch.randn(2, 3)
y = torch.randn(3, 2)
z = torch.mm(x, y)
print(f'✅ PyTorch tensor operations work')

# Check CUDA
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Available GPUs: {torch.cuda.device_count()}')
    
    # Test CUDA tensor
    try:
        cuda_tensor = torch.randn(2, 2).cuda()
        print(f'✅ CUDA tensor operations work')
    except Exception as e:
        print(f'⚠️  CUDA tensor test failed: {e}')
else:
    print('ℹ️  CUDA not available - using CPU mode')
"

# Test model loading
echo -e "\n${BLUE}Testing model loading...${NC}"
python -c "
try:
    from transformers import AutoTokenizer
    
    print('Testing tokenizer loading...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print('✅ BERT tokenizer loaded successfully')
    
    # Test tokenization
    text = 'Hello, this is a test.'
    tokens = tokenizer(text, return_tensors='pt')
    print(f'✅ Tokenization works: {tokens[\"input_ids\"].shape}')
    
except Exception as e:
    print(f'⚠️  Model loading test failed: {e}')
    print('This might be due to network issues. Models will download on first use.')
"

# Create activation script
echo -e "\n${BLUE}Creating activation helper script...${NC}"
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation helper for FedBERT-LoRA environment

echo "🚀 Activating FedBERT-LoRA environment..."
source venv/bin/activate

echo "✅ Environment activated!"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo ""
echo "Quick commands:"
echo "  python examples/run_simple_experiment.py"
echo "  python examples/run_glue_experiment.py --task sst2"
echo "  ./run_experiment.sh 5 10 sst2"
echo ""
EOF
chmod +x activate_env.sh

# Final system info
echo -e "\n${PURPLE}System Information:${NC}"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Python: $($PYTHON_CMD --version)"
echo "Pip: $(pip --version)"
if command_exists nvidia-smi; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "CPU: $(nproc) cores"

echo -e "\n${GREEN}🎉 Ubuntu environment setup completed successfully!${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Activate the environment: source venv/bin/activate"
echo "   OR use the helper: ./activate_env.sh"
echo "2. Run a simple test: python examples/run_simple_experiment.py"
echo "3. Run GLUE experiment: python examples/run_glue_experiment.py --task sst2"
echo "4. Use the convenience script: ./run_experiment.sh"
echo ""
echo -e "${BLUE}Ubuntu-specific notes:${NC}"
echo "• System packages installed via apt"
echo "• CUDA auto-detected and configured"
echo "• Cache directories configured for Hugging Face models"
echo "• Virtual environment with proper Python version"
echo ""
echo -e "${GREEN}Happy federated learning on Ubuntu! 🐧🚀${NC}"
