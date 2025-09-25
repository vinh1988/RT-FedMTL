# Ubuntu Setup Guide for FedBERT-LoRA 🐧

## ✅ Ubuntu-Optimized Setup

Your `setup_environment.sh` script is now **Ubuntu-optimized** with the following features:

### 🔧 **Ubuntu-Specific Features Added:**

1. **System Package Detection & Installation**
   - Auto-detects Ubuntu version
   - Installs required system packages via `apt`:
     - `python3`, `python3-pip`, `python3-venv`
     - `python3-dev`, `build-essential`
   - Handles missing dependencies automatically

2. **Smart Python Version Handling**
   - Detects current Python version
   - Auto-installs Python 3.9 if current version < 3.8
   - Uses appropriate Python command (`python3` or `python3.9`)

3. **CUDA Auto-Detection**
   - Detects NVIDIA GPU via `nvidia-smi`
   - Auto-installs appropriate PyTorch version:
     - CUDA 12.x → PyTorch with cu121
     - CUDA 11.8 → PyTorch with cu118  
     - No GPU → CPU-only PyTorch

4. **Ubuntu System Information**
   - Shows OS version, kernel, memory, CPU cores
   - Displays GPU information if available
   - Creates Ubuntu-specific activation helper

5. **Optimized Cache Setup**
   - Configures Hugging Face cache directory
   - Sets up environment variables in activation script

## 🚀 **How to Use (Ubuntu)**

### **Option 1: Quick Setup**
```bash
cd /home/pqvinh/Documents/LABs/FedAvgLS/FedBERT-LoRA

# Make executable and run
chmod +x setup_environment.sh
./setup_environment.sh
```

### **Option 2: Full Ubuntu Setup** 
```bash
# Use the comprehensive Ubuntu script
chmod +x setup_environment_ubuntu.sh
./setup_environment_ubuntu.sh
```

## 🔍 **What the Ubuntu Script Does:**

### **System Check & Setup:**
```bash
🐧 Checking Ubuntu system...
✅ Ubuntu detected: 22.04.3 LTS
🔧 Checking system dependencies...
📦 Installing missing system packages: python3-dev build-essential
```

### **Python Detection:**
```bash
🔍 Checking Python version...
Found Python 3.10.12
✅ Python version is compatible
```

### **CUDA Detection:**
```bash
🔍 Detecting CUDA availability...
✅ CUDA detected! Installing PyTorch with CUDA support...
CUDA Version: 11.8
```

### **Environment Creation:**
```bash
🔧 Creating virtual environment...
✅ Virtual environment created
🔧 Activating virtual environment...
⬆️ Upgrading pip and setuptools...
```

## 🎯 **Ubuntu-Specific Commands:**

### **Activation (Ubuntu Helper):**
```bash
# Use the Ubuntu-optimized activation helper
./activate_env.sh

# Output:
🐧 Activating FedBERT-LoRA environment on Ubuntu...
✅ Environment activated!
System: Ubuntu 22.04.3 LTS
Python: Python 3.10.12
PyTorch: 2.1.0+cu118
GPU: NVIDIA GeForce RTX 3080
```

### **Quick Test:**
```bash
# Activate environment
source venv/bin/activate

# Test on Ubuntu
python examples/run_simple_experiment.py
```

### **System Requirements Check:**
```bash
# Check if you have required packages
dpkg -l | grep -E "python3|python3-pip|python3-venv|python3-dev|build-essential"

# Check CUDA (if you have GPU)
nvidia-smi

# Check system resources
free -h && nproc
```

## 🛠️ **Manual Ubuntu Setup (if needed):**

### **Install System Dependencies:**
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential git curl
```

### **Install Python 3.9+ (if needed):**
```bash
sudo apt install -y python3.9 python3.9-venv python3.9-dev
```

### **Install CUDA (optional, for GPU):**
```bash
# Check if CUDA is already installed
nvidia-smi

# If not, install CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit
```

## ⚠️ **Common Ubuntu Issues & Solutions:**

### **1. Permission Errors:**
```bash
# If you get permission errors
sudo chown -R $USER:$USER /home/pqvinh/Documents/LABs/FedAvgLS/FedBERT-LoRA
```

### **2. Python venv Issues:**
```bash
# If venv creation fails
sudo apt install -y python3-venv
# Or try with specific version
python3.9 -m venv venv
```

### **3. Package Installation Fails:**
```bash
# Update package list and upgrade
sudo apt update && sudo apt upgrade
# Install build tools
sudo apt install -y build-essential python3-dev
```

### **4. CUDA Issues:**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# If CUDA not working, reinstall drivers
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit
```

### **5. Memory Issues:**
```bash
# Check available memory
free -h

# If low memory, reduce batch size in configs
# Edit configs/config.yaml:
data:
  train_batch_size: 8  # Reduce from 16
```

## ✅ **Verification Commands:**

### **Test Ubuntu Setup:**
```bash
cd /home/pqvinh/Documents/LABs/FedAvgLS/FedBERT-LoRA
source venv/bin/activate

# Check Python and packages
python --version
pip list | grep -E "torch|transformers|flwr|peft"

# Check CUDA (if available)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test basic functionality
python -c "
from src.models.federated_bert import FederatedBERTConfig
print('✅ FedBERT imports work')
"
```

## 🎉 **Ready to Go!**

Your Ubuntu setup is now optimized for:
- ✅ **System package management** via apt
- ✅ **CUDA auto-detection** and configuration  
- ✅ **Python version handling** (3.8+ support)
- ✅ **Memory optimization** for Ubuntu systems
- ✅ **Easy activation** with Ubuntu helper script

### **Start Your First Experiment:**
```bash
./activate_env.sh
python examples/run_simple_experiment.py
```

**Happy federated learning on Ubuntu! 🐧🚀**
