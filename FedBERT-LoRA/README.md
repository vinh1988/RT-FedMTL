# FedBERT-LoRA: Heterogeneous Federated Learning with BERT variants

A comprehensive implementation of heterogeneous federated learning using BERT-base on the server and TinyBERT on clients, enhanced with LoRA (Low-Rank Adaptation) for parameter efficiency and advanced knowledge transfer mechanisms.

## 🚀 Key Features

### Core Setup (Must-Have)
- ✅ **BERT-base Server**: Strong global model acting as central aggregator
- ✅ **TinyBERT Clients**: Lightweight models for local fine-tuning
- ✅ **LoRA Fine-tuning**: Train only low-rank adapters to reduce communication
- ✅ **Projection Layers**: Bridge hidden dimension mismatch (TinyBERT 312 → BERT-base 768)
- ✅ **Progressive Transfer**: Gradually transfer client knowledge into BERT-base
- ✅ **Dynamic Alignment**: Align client-server knowledge to prevent drift
- ✅ **FedAvg Aggregation**: Classic federated averaging with LoRA-aware aggregation
- ✅ **Single-terminal Simulation**: Run all clients + server in one process using Flower

### Advanced Features
- 🔄 **Knowledge Transfer**: Logits and hidden states alignment with temperature scaling
- 📊 **Multiple Aggregation Strategies**: Data-size weighted, uniform, loss-based
- 🎯 **GLUE Task Support**: Ready-to-use configurations for common NLP benchmarks
- 📈 **Comprehensive Logging**: Detailed metrics tracking and visualization support
- ⚙️ **Flexible Configuration**: Hydra-based configuration management
- 🔧 **Extensible Architecture**: Easy to add new models, aggregation methods, and tasks

## 📁 Project Structure

```
FedBERT-LoRA/
├── src/
│   ├── models/
│   │   ├── federated_bert.py      # BERT-base server & TinyBERT client models
│   │   └── knowledge_transfer.py  # Progressive transfer & dynamic alignment
│   ├── server/
│   │   └── flower_server.py       # Flower-based federated server
│   ├── clients/
│   │   └── flower_client.py       # Flower-based federated clients
│   ├── aggregation/
│   │   └── fedavg.py              # LoRA-aware FedAvg implementation
│   └── utils/
│       ├── data_utils.py          # Data loading and partitioning
│       └── training_utils.py      # Training utilities and helpers
├── configs/
│   ├── config.yaml                # Main configuration file
│   ├── model/bert_tiny_fed.yaml   # Model-specific configurations
│   ├── training/default.yaml      # Training configurations
│   └── federated/fedavg.yaml      # Federated learning configurations
├── examples/
│   ├── run_simple_experiment.py   # Simple experiment runner
│   └── run_glue_experiment.py     # GLUE task experiment runner
├── main.py                        # Main entry point with Hydra
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd FedBERT-LoRA
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install in development mode**
```bash
pip install -e .
```

## 🚀 Quick Start

### Simple Experiment
Run a basic federated learning experiment with default settings:

```bash
# Simple simulation with 5 clients, 10 rounds
python examples/run_simple_experiment.py
```

### GLUE Task Experiment
Run federated learning on GLUE tasks with real data:

```bash
# SST-2 sentiment classification
python examples/run_glue_experiment.py --task sst2 --num_clients 10 --num_rounds 20

# CoLA grammatical acceptability
python examples/run_glue_experiment.py --task cola --num_clients 8 --num_rounds 15

# MRPC paraphrase detection
python examples/run_glue_experiment.py --task mrpc --num_clients 6 --num_rounds 25
```

### Advanced Configuration
Use Hydra for complex configurations:

```bash
# Run with custom configuration
python main.py experiment.name=my_experiment federated.num_clients=20 federated.num_rounds=50

# Override specific parameters
python main.py data.task_name=cola lora.r=32 knowledge_transfer.progressive_transfer.enabled=true

# Use different configuration files
python main.py --config-path configs --config-name custom_config
```

## ⚙️ Configuration

### Model Configuration
```yaml
# Server model (BERT-base)
server_model:
  name: "bert-base-uncased"
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 12

# Client model (TinyBERT)
client_model:
  name: "huawei-noah/TinyBERT_General_4L_312D"
  hidden_size: 312
  num_attention_heads: 12
  num_hidden_layers: 4
```

### LoRA Configuration
```yaml
lora:
  r: 16                    # Rank of adaptation
  alpha: 32               # Scaling parameter
  dropout: 0.1            # LoRA dropout
  target_modules:         # Target modules for LoRA
    - "query"
    - "value" 
    - "key"
    - "dense"
```

### Knowledge Transfer Configuration
```yaml
knowledge_transfer:
  progressive_transfer:
    enabled: true
    warmup_rounds: 5      # Rounds to ramp up transfer
    transfer_weight: 0.5  # Maximum transfer weight
  
  dynamic_alignment:
    enabled: true
    logits_weight: 0.7    # Weight for logits alignment
    hidden_weight: 0.3    # Weight for hidden states alignment
    temperature: 4.0      # Temperature for knowledge distillation
```

### Federated Learning Configuration
```yaml
federated:
  num_clients: 10         # Total number of clients
  clients_per_round: 5    # Clients participating per round
  num_rounds: 50          # Total communication rounds
  local_epochs: 3         # Local training epochs per round
```

## 🧠 Architecture Details

### Heterogeneous Model Setup
- **Server**: BERT-base-uncased (768 hidden dimensions, 12 layers)
- **Clients**: TinyBERT (312 hidden dimensions, 4 layers)
- **Alignment**: Projection layers bridge dimensional differences

### LoRA Integration
- Applied to attention layers (query, key, value, dense)
- Only LoRA parameters are communicated and aggregated
- Dramatically reduces communication overhead (~1% of full model)

### Knowledge Transfer Mechanisms

1. **Progressive Transfer**
   - Gradually increases knowledge transfer weight during warmup
   - Exponentially decays after warmup to prevent overfitting
   - Configurable warmup rounds and decay rates

2. **Dynamic Alignment**
   - Aligns both logits and hidden states between server and clients
   - Temperature-scaled knowledge distillation
   - Weighted combination of alignment losses

3. **Projection-based Communication**
   - Server projects BERT-base features to TinyBERT dimensions
   - Clients project TinyBERT features for server integration
   - Bidirectional knowledge flow

### Aggregation Strategy
- **LoRA-aware FedAvg**: Only aggregates LoRA parameters
- **Multiple weighting schemes**: Uniform, data-size based, loss-based
- **Gradient clipping**: Optional gradient norm clipping
- **Server momentum**: Optional server-side momentum for stability

## 📊 Supported Tasks

### GLUE Benchmark
- **SST-2**: Stanford Sentiment Treebank (sentiment classification)
- **CoLA**: Corpus of Linguistic Acceptability (grammatical acceptability)
- **MRPC**: Microsoft Research Paraphrase Corpus (paraphrase detection)
- **QQP**: Quora Question Pairs (question similarity)
- **RTE**: Recognizing Textual Entailment
- **MNLI**: Multi-Genre Natural Language Inference

### Data Partitioning Strategies
- **IID**: Independent and identically distributed
- **Non-IID Dirichlet**: Dirichlet distribution-based partitioning
- **Non-IID Shards**: Shard-based partitioning (2 shards per client)

## 🔬 Experimental Results

### Communication Efficiency
- **LoRA Parameters**: ~1% of full model parameters
- **Projection Overhead**: Minimal additional communication
- **Knowledge Transfer**: Improves convergence with minimal overhead

### Model Performance
- **Server Model**: Full BERT-base capacity for complex reasoning
- **Client Models**: Efficient TinyBERT for local processing
- **Knowledge Transfer**: Bridges performance gap between models

### Scalability
- **Single-terminal Simulation**: Supports 10-100+ clients
- **Resource Efficiency**: CPU-friendly for development and testing
- **GPU Scaling**: Supports GPU acceleration for production

## 🛠️ Development

### Adding New Models
1. Extend `FederatedBERTServer` or `FederatedBERTClient`
2. Implement LoRA parameter extraction/setting methods
3. Add projection layers for dimensional alignment
4. Update configuration files

### Adding New Aggregation Methods
1. Extend `FedAvgAggregator` base class
2. Implement custom parameter aggregation logic
3. Add configuration options
4. Register in server strategy

### Adding New Tasks
1. Extend `GLUEDataset` for new data formats
2. Add task-specific preprocessing
3. Update configuration with task parameters
4. Add example scripts

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Test individual components
python src/models/federated_bert.py
python src/models/knowledge_transfer.py
python src/aggregation/fedavg.py
```

## 📈 Monitoring and Logging

### Built-in Metrics
- **Training Loss/Accuracy**: Per client and aggregated
- **Evaluation Metrics**: Validation performance tracking
- **Transfer Metrics**: Knowledge transfer effectiveness
- **Communication Metrics**: Parameter transfer statistics

### Logging Integration
- **TensorBoard**: Built-in TensorBoard logging support
- **Weights & Biases**: Optional W&B integration
- **Custom Logging**: Configurable logging levels and formats

### Visualization
```python
# Example: Plot training history
import matplotlib.pyplot as plt

def plot_federated_metrics(history):
    rounds = [r for r, _ in history.metrics_distributed["eval_accuracy"]]
    accuracies = [acc for _, acc in history.metrics_distributed["eval_accuracy"]]
    
    plt.plot(rounds, accuracies)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning Progress")
    plt.show()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. **Federated Learning**: McMahan, B., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS 2017.

2. **LoRA**: Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.

3. **BERT**: Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.

4. **TinyBERT**: Jiao, X., et al. "TinyBERT: Distilling BERT for Natural Language Understanding." Findings of EMNLP 2020.

5. **Flower Framework**: Beutel, D. J., et al. "Flower: A Friendly Federated Learning Research Framework." arXiv preprint arXiv:2007.14390 2020.

## 🙋‍♂️ Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Documentation**: Comprehensive docs available in `/docs` directory
- **Examples**: Working examples in `/examples` directory

---

**Built with ❤️ for the federated learning community**
