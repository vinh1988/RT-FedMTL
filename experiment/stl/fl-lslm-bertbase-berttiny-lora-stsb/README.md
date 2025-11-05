# Federated Learning for STS-B with BERT-base and BERT-tiny + LoRA

This experiment implements a federated learning setup for the STS-B (Semantic Textual Similarity Benchmark) task using BERT-base as the teacher model and BERT-tiny with LoRA as the student model.

## Experiment Details

- **Model Type**: Single-Task Learning (STL)
- **Teacher Model**: BERT-base-uncased
- **Student Model**: prajjwal1/bert-tiny with LoRA
- **Task**: Semantic Textual Similarity (STS-B)
- **Knowledge Distillation**: Enabled with temperature=3.0
- **LoRA Configuration**:
  - Rank: 16
  - Alpha: 64.0
  - Dropout: 0.1
  - Unfrozen layers: 2

## Setup

1. **Environment**
   ```bash
   # Create and activate virtual environment (if not already done)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install required packages
   pip install -r requirements.txt
   ```

2. **Directory Structure**
   ```
   fl-lslm-bertbase-berttiny-lora-stsb/
   ├── src/                    # Source code
   ├── federated_config.py     # Configuration loader
   ├── federated_main.py       # Main entry point
   ├── federated_config.yaml   # Experiment configuration
   └── README.md               # This file
   ```

## Running the Experiment

### 1. Start the Federated Learning Server

In one terminal window:
```bash
python federated_main.py --mode server --config federated_config.yaml
```

### 2. Start the Client

In a separate terminal window:
```bash
python federated_main.py --mode client --config federated_config.yaml --client_id stsb_client --task stsb
```

## Output Directory Structure

After running the experiment, the following directory structure will be created:

```
fl-lslm-bertbase-berttiny-lora-stsb/
├── federated_results/
│   ├── global_model/           # Global model checkpoints
│   │   ├── round_1/
│   │   ├── round_2/
│   │   └── ...
│   ├── client_models/          # Client model checkpoints
│   │   └── stsb_client/
│   │       ├── round_1/
│   │       ├── round_2/
│   │       └── ...
│   ├── metrics/                # Training and evaluation metrics
│   │   ├── train_metrics.json
│   │   ├── eval_metrics.json
│   │   └── test_metrics.json
│   └── logs/                   # Log files
│       ├── server.log
│       └── client_stsb_client.log
├── federated_stsb_client.log   # Client log file
└── federated_server.log        # Server log file
```

## Monitoring

- **Server Logs**: `federated_server.log`
- **Client Logs**: `federated_stsb_client.log`
- **Metrics**: Check `federated_results/metrics/` for detailed metrics

## Stopping the Experiment

Press `Ctrl+C` in both terminal windows to stop the server and client processes gracefully.

## Notes

- The experiment uses early stopping with a patience of 5 rounds based on the Pearson correlation metric.
- Model checkpoints are saved after each round in the respective directories.
- The best model based on validation performance is saved separately in `federated_results/best_model/`.
