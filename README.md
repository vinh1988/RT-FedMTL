# Federated Learning Experiment Plan

## Model Configurations

### Single Task Learning
| Model Type | Federated Learning | Centralized |
|------------|-------------------|-------------|
| LLMs | Global & Client: BERT-base-uncased | BERT-base-uncased |
| SLMs | Global & Client: prajjwal1/bert-tiny | prajjwal1/bert-tiny |
| L-SLMs (Our method) | • Global: BERT-base-uncased<br>• Client: prajjwal1/bert-tiny | |
| L-SLMs (Our method) | • Global: BERT-base-uncased<br>• Client: prajjwal1/bert-tiny + LoRA | |

### Multi-Task Learning
| Model Type | Federated Learning | Centralized |
|------------|-------------------|-------------|
| LLMs | Global & Client: BERT-base-uncased | BERT-base-uncased |
| SLMs | Global & Client: prajjwal1/bert-tiny | prajjwal1/bert-tiny |
| L-SLMs (Our method) | • Global: BERT-base-uncased<br>• Client: prajjwal1/bert-tiny | |
| L-SLMs (Our method) | • Global: BERT-base-uncased<br>• Client: prajjwal1/bert-tiny + LoRA | |

## Experiment Details

### Single Task Learning (STL)
| Framework | Model Type | Global Model | Client Model | Datasets | Validation Metrics | Total Experiments |
|-----------|------------|--------------|--------------|----------|-------------------|------------------|
| Federated Learning | LLMs | BERT-base-uncased | BERT-base-uncased | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 3 |
|  | SLMs | prajjwal1/bert-tiny | prajjwal1/bert-tiny | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 3 |
|  | L-SLMs (Our method) | BERT-base-uncased | prajjwal1/bert-tiny | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 3 |
|  | L-SLMs (Our method) | BERT-base-uncased | prajjwal1/bert-tiny + LoRA | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 3 |
| Centralized | LLMs | BERT-base-uncased | — | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 3 |
|  | SLMs | prajjwal1/bert-tiny | — | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 3 |

### Multi-Task Learning (MTL)
| Framework | Model Type | Global Model | Client Model | Datasets | Validation Metrics | Total Experiments |
|-----------|------------|--------------|--------------|----------|-------------------|------------------|
| Federated Learning | LLMs | BERT-base-uncased | BERT-base-uncased | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 1 |
|  | SLMs | prajjwal1/bert-tiny | prajjwal1/bert-tiny | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 1 |
|  | L-SLMs (Our method) | BERT-base-uncased | prajjwal1/bert-tiny | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 1 |
|  | L-SLMs (Our method) | BERT-base-uncased | prajjwal1/bert-tiny + LoRA | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 1 |
| Centralized | LLMs | BERT-base-uncased | — | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 1 |
|  | SLMs | prajjwal1/bert-tiny | — | SST-2, QQP, STS-B | F1, Precision, Recall, Pearson | 1 |

## Notes
- **LLMs**: Large Language Models
- **SLMs**: Small Language Models
- **L-SLMs**: Large-Small Language Models (Our method)
- **LoRA**: Low-Rank Adaptation
- **Datasets**: All experiments use SST-2, QQP, and STS-B datasets
- **Validation Metrics**: All experiments use F1, Precision, Recall, and Pearson correlation metrics
