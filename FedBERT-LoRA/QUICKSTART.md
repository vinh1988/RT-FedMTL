# Quick Start Guide

Get FedBERT-LoRA running in 5 minutes! 🚀

## Step 1: Navigate to Project Directory
```bash
cd /home/pqvinh/Documents/LABs/FedAvgLS/FedBERT-LoRA
```

## Step 2: Run Ubuntu-Optimized Setup
```bash
# Make setup script executable
chmod +x setup_environment.sh

# Run the Ubuntu-optimized setup (this will take 2-3 minutes)
./setup_environment.sh
```

The Ubuntu setup script will:
- ✅ **Auto-detect Ubuntu** and install system packages via apt
- ✅ **Check Python version** (installs 3.9 if needed)
- ✅ **Detect CUDA** and install appropriate PyTorch version
- ✅ **Create virtual environment** with proper Python
- ✅ **Install dependencies** and set up cache
- ✅ **Test everything works** including CUDA if available

## Step 3: Activate Environment
```bash
# Option 1: Standard activation
source venv/bin/activate

# Option 2: Use Ubuntu helper (shows system info)
./activate_env.sh
```

You should see `(venv)` in your terminal prompt and Ubuntu system information.

## Step 4: Run Your First Experiment

### Option A: Super Simple Test (30 seconds)
```bash
python examples/run_simple_experiment.py
```

### Option B: Real GLUE Task (2-3 minutes)
```bash
python examples/run_glue_experiment.py --task sst2 --num_clients 5 --num_rounds 5
```

### Option C: Using the Convenience Script
```bash
./run_experiment.sh 5 10 sst2
# 5 clients, 10 rounds, SST-2 task
```

## Expected Output

You should see something like:
```
🐧 Activating FedBERT-LoRA environment on Ubuntu...
✅ Environment activated!
System: Ubuntu 22.04.3 LTS
Python: Python 3.10.12
PyTorch: 2.1.0+cu118
GPU: NVIDIA GeForce RTX 3080 (or "Device: CPU")

🚀 FedBERT-LoRA Experiment Runner
=================================
Configuration:
  - Task: sst2
  - Clients: 5
  - Rounds: 10
  - Device: cuda (or cpu)

🏃 Starting experiment...
INFO - Starting federated BERT experiment on SST2
INFO - Configuration: 5 clients, 10 rounds
INFO - Client 0 starting training round 0
...
✅ Experiment completed!
```

## What's Happening?

1. **Server**: Runs BERT-base model (768 dimensions) with LoRA adapters
2. **Clients**: Each runs TinyBERT (312 dimensions) with LoRA adapters  
3. **Knowledge Transfer**: Progressive transfer from server to clients
4. **Aggregation**: FedAvg aggregates only LoRA parameters (super efficient!)
5. **Communication**: Only ~1% of full model parameters are shared

## Troubleshooting

### If Setup Fails:
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Manual setup
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### If Experiments Fail:
```bash
# Try with fewer clients and rounds
python examples/run_glue_experiment.py --task sst2 --num_clients 3 --num_rounds 3

# Check if environment is activated
echo $VIRTUAL_ENV  # Should show path to venv
```

### Memory Issues (Ubuntu):
```bash
# Check available memory
free -h

# Edit configs/config.yaml if needed:
data:
  train_batch_size: 8  # Reduce from 16
  
federated:
  num_clients: 3  # Reduce from 10

# Or use CPU mode
experiment:
  device: "cpu"
```

## Next Steps

Once everything works:

1. **Explore Configurations**: Check `configs/` directory
2. **Try Different Tasks**: `cola`, `mrpc`, `qqp`, `rte`
3. **Modify Parameters**: Change LoRA rank, transfer weights, etc.
4. **Add Your Data**: Follow data utilities guide
5. **Scale Up**: Try more clients and rounds

## Key Files to Know

- `main.py` - Main entry point with Hydra configs
- `examples/` - Ready-to-run example scripts
- `configs/` - All configuration files
- `src/models/` - BERT server and TinyBERT client models
- `src/server/` - Flower server implementation
- `src/clients/` - Flower client implementation

## Quick Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Simple test
python examples/run_simple_experiment.py

# GLUE experiments
python examples/run_glue_experiment.py --task sst2 --num_clients 10 --num_rounds 20
python examples/run_glue_experiment.py --task cola --num_clients 8 --num_rounds 15

# Using Hydra configs
python main.py experiment.name=my_test federated.num_clients=5

# Convenience script
./run_experiment.sh 10 25 mrpc  # 10 clients, 25 rounds, MRPC task
```

That's it! You now have a fully functional heterogeneous federated learning system with BERT-base server and TinyBERT clients! 🎉
