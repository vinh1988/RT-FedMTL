#!/bin/bash

echo "========================================"
echo "Centralized Training - Multi-Task Learning"
echo "Mini BERT with All Tasks (SST2, QQP, STSB)"
echo "========================================"

# ===== Project paths =====
PROJECT_ROOT="/home/vinh/Documents/code/FedAvgLS_windows/FedAvgLS"
WORK_DIR="$PROJECT_ROOT/experiment_new_solution/models/mini-bert/centralized-mtl-all-tasks"

# ===== Virtual Environment =====
VENV_PATH="$PROJECT_ROOT/venv"

export PYTHONPATH="$PROJECT_ROOT"

echo "Project Root: $PROJECT_ROOT"
echo "Work Directory: $WORK_DIR"
echo "Virtual Environment: $VENV_PATH"

# Check if virtual environment exists
echo "Checking virtual environment..."
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment '$VENV_PATH' not found!"
    echo "Please create it first:"
    echo "cd $PROJECT_ROOT"
    echo "python -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install pytorch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Environment found: $VENV_PATH"
echo

# ===== Start Centralized MTL Training =====
echo "Starting Centralized MTL Training..."

# Activate virtual environment and run training
source "$VENV_PATH/bin/activate"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0
python centralized_main.py

echo
echo "========================================"
echo "Centralized MTL training started!"
echo "========================================"
echo
echo "Model: prajjwal1/bert-mini"
echo "Tasks: SST2 + QQP + STSB"
echo "Training Type: Centralized Multi-Task Learning"
echo
echo "SST2: 66,477 train + 872 val samples (Classification)"
echo "QQP: 323,415 train + 40,431 val samples (Classification)"
echo "STSB: 4,249 train + 1,500 val samples (Regression)"
echo
echo "Total: 394,141 training samples across all tasks"
echo "Single model learns all tasks simultaneously"
echo
echo "Check training window for progress"
echo "Results will be saved to: centralized_mtl_results"
echo
read -p "Press Enter to exit..."
