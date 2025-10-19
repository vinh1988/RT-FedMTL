# STS-B Training Scripts

This directory contains standalone scripts for training models on the Semantic Textual Similarity Benchmark (STS-B) dataset with comprehensive metrics tracking.

## Files

- `train_stsb.py` - Main training script with full configuration options
- `quick_train.py` - Simple script for quick training runs
- `requirements.txt` - Python dependencies needed

## Features

- ✅ **Comprehensive Metrics Tracking** - Training/validation loss and Pearson correlation
- ✅ **CSV Output** - All metrics saved to timestamped CSV files
- ✅ **Multiple Model Support** - Works with any HuggingFace transformer model
- ✅ **Flexible Configuration** - Configurable epochs, batch size, learning rate, etc.
- ✅ **GPU Support** - Automatic CUDA detection and utilization
- ✅ **Progress Logging** - Detailed training progress and final results

## Usage

### Quick Start (Recommended)

```bash
# Navigate to the stsb directory
cd /home/pc/Documents/LAB/FedAvgLS/stsb

# Run quick training (uses dummy data for demonstration)
python3 quick_train.py
```

### Full Training Script

```bash
# For more control over training parameters
python3 train_stsb.py \
    --model "bert-base-uncased" \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --output_dir "/home/pc/Documents/LAB/FedAvgLS/stsb/results"
```

### With Real STS-B Data

```bash
# If you have actual STS-B dataset files
python3 train_stsb.py \
    --train_data "/path/to/stsb/train.tsv" \
    --val_data "/path/to/stsb/val.tsv" \
    --model "bert-base-uncased" \
    --epochs 10 \
    --batch_size 32
```

## Output

The scripts will create:

1. **Model Files** - Saved HuggingFace model in `stsb_model_YYYYMMDD_HHMMSS/`
2. **Metrics CSV** - Training metrics in `stsb_training_metrics_YYYYMMDD_HHMMSS.csv`
3. **Console Output** - Real-time training progress and final results

### Sample CSV Output

```csv
epoch,train_loss,val_loss,train_correlation,val_correlation
1,2.3456,2.1234,0.4567,0.5234
2,1.9876,1.8765,0.6234,0.6789
3,1.6543,1.5432,0.7345,0.7890
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `"bert-base-uncased"` | HuggingFace model name |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `16` | Training batch size |
| `--learning_rate` | `2e-5` | Learning rate for optimizer |
| `--max_length` | `128` | Maximum sequence length |
| `--train_data` | `None` | Path to training data file |
| `--val_data` | `None` | Path to validation data file |
| `--output_dir` | `/home/pc/Documents/LAB/FedAvgLS/stsb` | Output directory |
| `--device` | `auto` | Device to use (`cpu`, `cuda`, or `auto`) |

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch transformers numpy scikit-learn
```

## Expected Results

- **Training Time**: ~5-15 minutes per epoch (depending on model size and hardware)
- **Memory Usage**: ~2-8GB GPU memory (depending on model and batch size)
- **Output Metrics**:
  - Training Loss: Decreases from ~2.5 to ~0.5
  - Validation Loss: Similar pattern to training loss
  - Pearson Correlation: Increases from ~0.4 to ~0.8+

## Troubleshooting

1. **Out of Memory**: Reduce batch size or use a smaller model
2. **Slow Training**: Enable GPU acceleration if available
3. **Poor Performance**: Try different learning rates or model architectures

## Integration with Federated Learning

This standalone training script can be used to:
- Pre-train models before federated learning
- Evaluate model performance on STS-B task
- Compare different model architectures
- Generate baseline metrics for federated learning experiments
