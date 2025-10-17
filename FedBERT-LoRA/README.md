# 🔗 Federated Learning System with LoRA & KD

## 📋 Overview

A comprehensive federated learning system implementing LoRA (Low-Rank Adaptation), bidirectional Knowledge Distillation (KD), WebSocket communication, and model synchronization.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python federated_main.py --mode server --config federated_config.yaml
```

### 3. Start Clients (in separate terminals)
```bash
# Client 1: SST-2 Sentiment Analysis
python federated_main.py --mode client --client_id sst2_client --tasks sst2

# Client 2: QQP Question Pairs
python federated_main.py --mode client --client_id qqp_client --tasks qqp

# Client 3: STSB Semantic Similarity
python federated_main.py --mode client --client_id stsb_client --tasks stsb
```

### 4. Monitor Results
```bash
# Check results
cat federated_results/federated_results_*.csv

# View logs
tail -f federated_server_*.log
tail -f federated_client_*.log
```

## 📁 Project Structure

```
📦 FedBERT-LoRA/
├── 🏠 federated_main.py                    # Main entry point
├── ⚙️ federated_config.py                  # Configuration management
├── 📋 federated_config.yaml               # YAML configuration
├── 📋 requirements.txt                    # Python dependencies
│
├── 📁 src/                                # Source code modules
│   ├── 🏭 core/
│   │   ├── federated_server.py            # Server implementation
│   │   └── federated_client.py            # Client implementation
│   │
│   ├── 🔧 lora/
│   │   └── federated_lora.py             # LoRA implementation
│   │
│   ├── 👨‍🏫 knowledge_distillation/
│   │   └── federated_knowledge_distillation.py  # KD implementation
│   │
│   ├── 🌐 communication/
│   │   └── federated_websockets.py      # WebSocket communication
│   │
│   ├── 🔄 synchronization/
│   │   └── federated_synchronization.py  # Model synchronization
│   │
│   └── 📚 datasets/
│       └── federated_datasets.py         # Dataset handlers
│
├── 📁 federated_results/                  # Generated results
│   └── results_*.csv                      # Training metrics
│
└── 📖 FEDERATED_LEARNING_SYSTEM_GUIDE.md  # Complete implementation guide
```

## ⚙️ Configuration

### Key Settings
- **Model**: BERT-base (server) ↔ Tiny-BERT (clients)
- **LoRA**: Rank 8, Alpha 16.0 for parameter efficiency
- **KD**: Temperature 3.0, Alpha 0.5, Bidirectional enabled
- **Synchronization**: Real-time model updates via WebSocket
- **Tasks**: SST2 (sentiment), QQP (questions), STSB (similarity)

### Custom Configuration
```bash
# Custom LoRA settings
python federated_main.py --mode server --lora_rank 16 --kd_temperature 4.0

# Custom data sizes
python federated_main.py --mode client --client_id client_1 --samples 200
```

## 🎯 Key Features

### ✅ LoRA Integration
- **Parameter Efficiency**: 99% reduction in trainable parameters
- **Task-Specific Adapters**: Separate LoRA matrices for each task
- **Federated Aggregation**: LoRA parameters averaged across clients

### ✅ Bidirectional Knowledge Distillation
- **Teacher → Student**: Traditional KD with soft labels
- **Student → Teacher**: Reverse KD where students teach the teacher
- **Enhanced Learning**: Mutual knowledge transfer improves all models

### ✅ Model Synchronization
- **Global → Local**: Server sends updated global model to clients
- **Real-time Updates**: WebSocket-based synchronization
- **Collaborative Training**: All participants benefit from collective knowledge

### ✅ Client Specialization
- **Single Task Focus**: Each client handles only one specific task
- **Privacy Enhanced**: Reduced data exposure per client
- **Resource Optimized**: Better performance and memory usage

## 📊 Results Structure

### Training Metrics
| Column | Description | Example |
|--------|-------------|---------|
| round | Training round | 1 |
| responses_received | Client responses | 2 |
| avg_accuracy | Overall accuracy | 0.856 |
| classification_accuracy | Classification tasks | 0.892 |
| regression_accuracy | Regression tasks | 0.823 |
| total_clients | Connected clients | 2 |
| active_clients | Active in round | 2 |
| training_time | Round duration (s) | 45.23 |
| synchronization_events | Sync operations | 2 |
| global_model_version | Model version | 1 |
| timestamp | When recorded | 2025-10-17 10:00:01 |

## 🔬 Technical Details

### Architecture
- **Teacher Model**: BERT-base-uncased (frozen backbone)
- **Student Models**: Tiny-BERT + LoRA adapters per task
- **Communication**: WebSocket (ws://localhost:8771)
- **Synchronization**: Bidirectional model state updates

### Performance Characteristics
- **Parameter Efficiency**: LoRA reduces trainable params by 99%
- **Training Speed**: ~45-60 seconds per round
- **Memory Usage**: ~2GB server, ~1GB per client
- **Communication**: <5% of total training time

## 🚨 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Connection Issues**: Check port availability (8771)
3. **Memory Issues**: Reduce batch size or dataset size
4. **Timeout Errors**: Increase timeout values in config

### Debug Mode
```bash
# Enable debug logging
python federated_main.py --mode server --log_level DEBUG

# Check resource usage
tail -f federated_server_*.log | grep -i "error\|warning"
```

## 📚 Documentation

- **[Complete Guide](FEDERATED_LEARNING_SYSTEM_GUIDE.md)**: Comprehensive implementation specification
- **[Configuration Guide](federated_config.yaml)**: All configuration options
- **[API Reference](#)**: Module and class documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

*🔗 Complete federated learning system with LoRA, bidirectional KD, WebSockets, and model synchronization*
