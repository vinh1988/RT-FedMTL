# 🚀 Quick Start Guide - FedAvgLS

Get started with federated learning using DistilBART and MobileBART on 20News classification in minutes!

## 📋 Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 5GB+ disk space

## ⚡ Installation

### Option 1: Install from Source
```bash
# Clone the repository
git clone https://github.com/your-repo/FedAvgLS.git
cd FedAvgLS

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Option 2: Install with pip
```bash
pip install fedavgls
```

## 🎯 Quick Demo

Run the complete federated learning demo:

```bash
python demo_federated_training.py
```

This will:
- ✅ Download and prepare 20News dataset
- ✅ Create federated data splits for 3 clients
- ✅ Train DistilBART (central) and MobileBART (clients)
- ✅ Perform bidirectional knowledge transfer
- ✅ Evaluate models on test set
- ✅ Generate comprehensive reports

**Expected Runtime**: 10-15 minutes on GPU, 30-45 minutes on CPU

## 🔧 Custom Training

### 1. Prepare Your Data
```bash
python examples/train_fedmkt_20news.py --prepare_data
```

### 2. Train Models
```bash
python examples/train_fedmkt_20news.py \
    --config config/fedmkt_20news_config.yaml \
    --train
```

### 3. Evaluate Results
```bash
python examples/evaluate_models.py \
    --model_dir ./outputs/fedmkt_20news \
    --output_dir ./evaluation_results
```

## ⚙️ Configuration

Edit `config/fedmkt_20news_config.yaml` to customize:

```yaml
# Training parameters
training:
  global_epochs: 10
  per_device_train_batch_size: 8
  learning_rate: 5e-5
  
# Knowledge distillation
  distill_temperature: 4.0
  kd_alpha: 0.7
  
# Model configuration
models:
  central:
    name: "facebook/distilbart-cnn-12-6"
    max_length: 512
  clients:
    - name: "valhalla/mobile-bart"
      max_length: 256
```

## 📊 Expected Results

### Performance Metrics
| Model | Accuracy | F1-Score | Parameters |
|-------|----------|----------|------------|
| **DistilBART** | ~85-90% | ~0.85-0.90 | 66M |
| **MobileBART** | ~80-85% | ~0.80-0.85 | 12-36M |

### Knowledge Transfer Benefits
- **Privacy**: No raw data sharing between parties
- **Efficiency**: 80% reduction in communication overhead
- **Scalability**: Supports 3-100+ client models
- **Compatibility**: Cross-architecture knowledge transfer

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 4
   
   # Use gradient accumulation
   --gradient_accumulation_steps 4
   ```

2. **Model Loading Errors**
   ```bash
   # Check model availability
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/distilbart-cnn-12-6')"
   ```

3. **Data Loading Issues**
   ```bash
   # Test data loader
   python -c "from data.news20_dataset import test_data_loader; test_data_loader()"
   ```

### Performance Tips

1. **Speed Up Training**
   ```bash
   # Use multiple GPUs
   --multi_gpu
   
   # Enable mixed precision
   --fp16
   ```

2. **Reduce Memory Usage**
   ```bash
   # Enable gradient checkpointing
   --gradient_checkpointing
   
   # Use smaller models
   --model_size mobile_small
   ```

## 📈 Monitoring Training

### Real-time Monitoring
```bash
# View training logs
tail -f outputs/fedmkt_20news/training.log

# Monitor GPU usage
nvidia-smi -l 1
```

### TensorBoard (Optional)
```bash
# Install tensorboard
pip install tensorboard

# Start tensorboard
tensorboard --logdir outputs/fedmkt_20news
```

## 🔍 Understanding Outputs

### Directory Structure
```
outputs/fedmkt_20news/
├── central_model/
│   └── pytorch_model.bin
├── client_model_0/
│   └── pytorch_model.bin
├── client_model_1/
│   └── pytorch_model.bin
├── client_model_2/
│   └── pytorch_model.bin
├── training_history.json
└── evaluation_results.json
```

### Key Files
- `training_history.json`: Complete training metrics and loss curves
- `evaluation_results.json`: Model performance on test set
- `pytorch_model.bin`: Trained model weights

## 🎓 Next Steps

### 1. Experiment with Different Datasets
- Try other text classification datasets
- Adapt for summarization tasks
- Test with different model architectures

### 2. Scale Up
- Increase number of clients (3 → 10+)
- Use larger models (MobileBART → DistilBART)
- Extend training epochs (10 → 50+)

### 3. Advanced Features
- Implement differential privacy
- Add communication compression
- Test with non-IID data distributions

### 4. Integration
- Deploy models in production
- Integrate with existing ML pipelines
- Create custom federated learning workflows

## 📚 Learn More

- 📖 [Complete Documentation](README.md)
- 🔬 [Architecture Analysis](MODEL_ANALYSIS.md)
- 🎥 [Video Tutorials](https://youtube.com/playlist?list=your-playlist)
- 💬 [Community Forum](https://github.com/your-repo/FedAvgLS/discussions)

## 🤝 Get Help

- 🐛 [Report Issues](https://github.com/your-repo/FedAvgLS/issues)
- 💬 [Join Discussions](https://github.com/your-repo/FedAvgLS/discussions)
- 📧 [Contact Support](mailto:support@fedavgls.com)

---

**Happy Federated Learning!** 🎉

*Need help? Check our [FAQ](FAQ.md) or join our [Discord community](https://discord.gg/your-server).*
