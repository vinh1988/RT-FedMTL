# FedAvgLS Experiments

This document outlines the structure and organization of experiments for the FedAvgLS project.

## Experiment Structure

### Directory Naming Convention
```
{ID:02d}_{framework}_{task_type}_{model_type}_{model_details}_{dataset}
```
- `ID`: 02-25 (unique identifier)
- `framework`: `fed` (Federated) or `cen` (Centralized)
- `task_type`: `stl` (Single Task) or `mtl` (Multi-Task)
- `model_type`: `llm` (Large), `slm` (Small), or `lslm` (Large-Small)
- `model_details`: e.g., `bert-base-uncased`, `bert-tiny`, `bert-base-tiny`
- `dataset`: `sst2`, `qqp`, `stsb`, or `multi`

### Federated Learning - Single Task Learning (STL)
| ID | Model Type | Global Model | Client Model | Dataset | Metrics | Directory Name |
|----|------------|--------------|--------------|---------|---------|----------------|
| 02 | LLM | BERT-base-uncased | BERT-base-uncased | SST-2 | Accuracy, F1 | `02_fed_stl_llm_bert-base-uncased_sst2` |
| 03 | LLM | BERT-base-uncased | BERT-base-uncased | QQP | Accuracy, F1 | `03_fed_stl_llm_bert-base-uncased_qqp` |
| 04 | LLM | BERT-base-uncased | BERT-base-uncased | STS-B | Pearson, Spearman | `04_fed_stl_llm_bert-base-uncased_stsb` |
| 05 | SLM | prajjwal1/bert-tiny | prajjwal1/bert-tiny | SST-2 | Accuracy, F1 | `05_fed_stl_slm_bert-tiny_sst2` |
| 06 | SLM | prajjwal1/bert-tiny | prajjwal1/bert-tiny | QQP | Accuracy, F1 | `06_fed_stl_slm_bert-tiny_qqp` |
| 07 | SLM | prajjwal1/bert-tiny | prajjwal1/bert-tiny | STS-B | Pearson, Spearman | `07_fed_stl_slm_bert-tiny_stsb` |
| 08 | L-SLM | BERT-base-uncased | prajjwal1/bert-tiny | SST-2 | Accuracy, F1 | `08_fed_stl_lslm_bert-base-tiny_sst2` |
| 09 | L-SLM | BERT-base-uncased | prajjwal1/bert-tiny | QQP | Accuracy, F1 | `09_fed_stl_lslm_bert-base-tiny_qqp` |
| 10 | L-SLM | BERT-base-uncased | prajjwal1/bert-tiny | STS-B | Pearson, Spearman | `10_fed_stl_lslm_bert-base-tiny_stsb` |
| 11 | L-SLM (LoRA) | BERT-base-uncased | prajjwal1/bert-tiny + LoRA | SST-2 | Accuracy, F1 | `11_fed_stl_lslm_lora_bert-base-tiny_sst2` |
| 12 | L-SLM (LoRA) | BERT-base-uncased | prajjwal1/bert-tiny + LoRA | QQP | Accuracy, F1 | `12_fed_stl_lslm_lora_bert-base-tiny_qqp` |
| 13 | L-SLM (LoRA) | BERT-base-uncased | prajjwal1/bert-tiny + LoRA | STS-B | Pearson, Spearman | `13_fed_stl_lslm_lora_bert-base-tiny_stsb` |

### Centralized - Single Task Learning (STL)
| ID | Model Type | Global Model | Client Model | Dataset | Metrics | Directory Name |
|----|------------|--------------|--------------|---------|---------|----------------|
| 14 | LLM | BERT-base-uncased | - | SST-2 | Accuracy, F1 | `14_cen_stl_llm_bert-base-uncased_sst2` |
| 15 | LLM | BERT-base-uncased | - | QQP | Accuracy, F1 | `15_cen_stl_llm_bert-base-uncased_qqp` |
| 16 | LLM | BERT-base-uncased | - | STS-B | Pearson, Spearman | `16_cen_stl_llm_bert-base-uncased_stsb` |
| 17 | SLM | prajjwal1/bert-tiny | - | SST-2 | Accuracy, F1 | `17_cen_stl_slm_bert-tiny_sst2` |
| 18 | SLM | prajjwal1/bert-tiny | - | QQP | Accuracy, F1 | `18_cen_stl_slm_bert-tiny_qqp` |
| 19 | SLM | prajjwal1/bert-tiny | - | STS-B | Pearson, Spearman | `19_cen_stl_slm_bert-tiny_stsb` |

### Federated Learning - Multi-Task Learning (MTL)
| ID | Model Type | Global Model | Client Model | Datasets | Metrics | Directory Name |
|----|------------|--------------|--------------|----------|---------|----------------|
| 20 | LLM | BERT-base-uncased | BERT-base-uncased | SST-2, QQP, STS-B | SST-2: Acc, F1; QQP: Acc, F1; STS-B: Pearson, Spearman | `20_fed_mtl_llm_bert-base-uncased_multi` |
| 21 | SLM | prajjwal1/bert-tiny | prajjwal1/bert-tiny | SST-2, QQP, STS-B | SST-2: Acc, F1; QQP: Acc, F1; STS-B: Pearson, Spearman | `21_fed_mtl_slm_bert-tiny_multi` |
| 22 | L-SLM | BERT-base-uncased | prajjwal1/bert-tiny | SST-2, QQP, STS-B | SST-2: Acc, F1; QQP: Acc, F1; STS-B: Pearson, Spearman | `22_fed_mtl_lslm_bert-base-tiny_multi` |
| 23 | L-SLM (LoRA) | BERT-base-uncased | prajjwal1/bert-tiny + LoRA | SST-2, QQP, STS-B | SST-2: Acc, F1; QQP: Acc, F1; STS-B: Pearson, Spearman | `23_fed_mtl_lslm_lora_bert-base-tiny_multi` |

### Centralized - Multi-Task Learning (MTL)
| ID | Model Type | Global Model | Client Model | Datasets | Metrics | Directory Name |
|----|------------|--------------|--------------|----------|---------|----------------|
| 24 | LLM | BERT-base-uncased | - | SST-2, QQP, STS-B | SST-2: Acc, F1; QQP: Acc, F1; STS-B: Pearson, Spearman | `24_cen_mtl_llm_bert-base-uncased_multi` |
| 25 | SLM | prajjwal1/bert-tiny | - | SST-2, QQP, STS-B | SST-2: Acc, F1; QQP: Acc, F1; STS-B: Pearson, Spearman | `25_cen_mtl_slm_bert-tiny_multi` |

## Directory Structure

```
experiment/FATE-LLM/
├── experiments/           # All experiment directories (02-25)
└── shared/               # Shared resources
    ├── configs/          # Configuration templates
    ├── scripts/          # Training/evaluation scripts
    └── utils/            # Utility functions
```

## Getting Started

1. **Setup Environment**
   ```bash
   # Navigate to the experiment directory
   cd experiments/02_fed_stl_llm_bert-base-sst2
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r ../../shared/requirements.txt
   ```

2. **Run an Experiment**
   ```bash
   # Example: Run federated STL with BERT-base on SST-2
   python ../../shared/scripts/run_fed_stl.py --config config.yaml
   ```

## Adding New Experiments

1. Create a new directory following the naming convention: `{ID}_{framework}_{task_type}_{model_type}_{details}`
2. Copy configuration templates from `shared/configs/`
3. Update the configuration files as needed
4. Add experiment details to this document

## Contributing

Please follow the project's coding standards and document any new features or changes.
