# Installation Guide

This guide will help you set up the FedBERT-LoRA environment from scratch.

## Prerequisites

- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **Git** (for cloning the repository)
- **4GB+ RAM** (for running experiments)
- **Optional**: CUDA-capable GPU for faster training

## Quick Setup (Recommended)

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd FedBERT-LoRA
```

### 2. Run Automated Setup
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

This script will:
- ✅ Check Python version compatibility
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Set up the project structure
- ✅ Test the installation

### 3. Activate Environment
```bash
source venv/bin/activate
```

### 4. Test Installation
```bash
# Quick test
python examples/run_simple_experiment.py

# GLUE task test
python examples/run_glue_experiment.py --task sst2 --num_clients 5 --num_rounds 5
```

## Manual Setup

If you prefer to set up manually or the automated script doesn't work:

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Upgrade pip
```bash
pip install --upgrade pip
```

### 3. Install PyTorch
```bash
# CPU version (recommended for development)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU version (if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install Project
```bash
pip install -e .
```

### 6. Create Directories
```bash
mkdir -p logs data checkpoints outputs
```

## Verification

### Check Core Dependencies
```python
import torch
import transformers
import flwr
import peft
import datasets

print("✅ All dependencies installed successfully!")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Flower: {flwr.__version__}")
print(f"PEFT: {peft.__version__}")
```

### Test Model Loading
```python
from transformers import AutoModel, AutoTokenizer

# Test server model
server_model = AutoModel.from_pretrained("bert-base-uncased")
print(f"✅ BERT-base loaded: {server_model.config.hidden_size} dimensions")

# Test client model  
client_model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
print(f"✅ TinyBERT loaded: {client_model.config.hidden_size} dimensions")
```

### Test Federated Components
```python
from src.models.federated_bert import create_federated_bert_models, FederatedBERTConfig

config = FederatedBERTConfig()
server, client = create_federated_bert_models(config)
print("✅ Federated models created successfully")
```

## Common Issues and Solutions

### 1. Python Version Error
**Error**: `Python 3.8+ required`
**Solution**: 
```bash
# Check your Python version
python3 --version

# Install Python 3.8+ if needed (Ubuntu/Debian)
sudo apt update
sudo apt install python3.9 python3.9-venv

# Use specific Python version
python3.9 -m venv venv
```

### 2. CUDA Issues
**Error**: CUDA version mismatch
**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (safer option)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Memory Issues
**Error**: Out of memory during model loading
**Solution**:
- Use CPU mode: Set `device: "cpu"` in configs
- Reduce batch size: Set `batch_size: 8` or lower
- Reduce number of clients: Start with `num_clients: 3`

### 4. Network Issues
**Error**: Can't download models/datasets
**Solution**:
```bash
# Set up cache directory
export HF_HOME=./cache
export TRANSFORMERS_CACHE=./cache

# Use offline mode if models are already downloaded
export TRANSFORMERS_OFFLINE=1
```

### 5. Permission Issues
**Error**: Permission denied when running scripts
**Solution**:
```bash
chmod +x setup_environment.sh
chmod +x run_experiment.sh
```

## Development Setup

For contributors and developers:

### 1. Install Development Dependencies
```bash
pip install -e ".[dev]"
```

### 2. Install Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### 3. Run Tests
```bash
pytest tests/
```

### 4. Code Formatting
```bash
black src/ examples/
isort src/ examples/
```

## Docker Setup (Alternative)

If you prefer using Docker:

### 1. Build Docker Image
```bash
docker build -t fedbert-lora .
```

### 2. Run Container
```bash
docker run -it --rm -v $(pwd):/workspace fedbert-lora
```

### 3. Run Experiments in Container
```bash
python examples/run_simple_experiment.py
```

## Performance Optimization

### For CPU Training
```yaml
# In configs/config.yaml
experiment:
  device: "cpu"

data:
  num_workers: 4  # Adjust based on CPU cores
  train_batch_size: 8  # Smaller batch size for CPU

federated:
  num_clients: 5  # Start with fewer clients
```

### For GPU Training
```yaml
# In configs/config.yaml
experiment:
  device: "cuda"

data:
  num_workers: 2  # Fewer workers when using GPU
  train_batch_size: 16  # Larger batch size for GPU

federated:
  num_clients: 10  # Can handle more clients with GPU
```

## Next Steps

After successful installation:

1. **Read the README**: Understand the project structure and features
2. **Run Examples**: Start with simple experiments to verify everything works
3. **Explore Configurations**: Modify configs to suit your needs
4. **Add Your Data**: Follow the data utilities guide to add custom datasets
5. **Contribute**: Check out the contributing guidelines

## Getting Help

- **GitHub Issues**: Report bugs and ask questions
- **Documentation**: Check the `/docs` directory for detailed guides
- **Examples**: Look at `/examples` for working code samples

Happy federated learning! 🚀
