# Master Model Multi-Dimensional Comparison

This report provides a comprehensive analysis of BERT models across Paradigms, Task Types, and Data Distributions.

## Comparison Tables Overview

The following benchmark tables are included in this report:

1.  **Paradigm Performance**: Centralized vs. Federated (MTL)
    *   **Attributes**: `Model`, `Paradigm`, `SST-2 Acc`, `QQP F1`, `STSB Pear`, `Res (GB)`
2.  **Data Distribution Impact**: IID vs. Non-IID (FL-MTL)
    *   **Attributes**: `Model`, `Distribution`, `SST-2 Acc`, `QQP F1`, `STSB Pear`, `Res (GB)`
3.  **Multi-Task Learning Gain**: Single-Task vs. MTL (FL)
    *   **Attributes**: `Model`, `Task Type`, `SST-2 Acc`, `QQP F1`, `STSB Pear`, `Res (GB)`

## Training Summary

The following table lists the models and training methods included in the source data:

| Dimension | Scope / Values |
| --- | --- |
| **Model Families** | `tiny_bert`, `mini-bert`, `medium-bert`, `distil-bert`, `mini-lm` |
| **Paradigms** | `Centralized`, `Federated Learning (FL)` |
| **Task Types** | `Single-Task (SST-2, QQP, STS-B)`, `Multi-Task (MTL)` |
| **Distributions** | `IID`, `Non-IID (3-client partition)` |
| **Optimization** | `Full Fine-Tuning`, `PEFT (LoRA)` |

## 1. Paradigm Performance: Centralized vs. Federated (MTL)
Comparing best Multi-Task Learning (MTL) results for each paradigm.

````carousel
![Paradigm Comparison: SST-2](./plots/comp_paradigm_val_sst2_acc.png)
<!-- slide -->
![Paradigm Comparison: QQP](./plots/comp_paradigm_val_qqp_f1.png)
<!-- slide -->
![Paradigm Comparison: STS-B](./plots/comp_paradigm_val_stsb_pearson.png)
<!-- slide -->
![Resource Usage: Centralized vs FL](./plots/comp_resource.png)
````

| Model | Paradigm | SST-2 Acc | QQP F1 | STSB Pear | Res (GB) |
| --- | --- | --- | --- | --- | --- |
| TINY_BERT | Centralized | 0.8211 | 0.8101 | 0.6474 | 0.3440 |
| TINY_BERT | FL | 0.9014 | 0.7494 | 0.5212 | 0.0881 |
| | | | | | |
| MINI-BERT | Centralized | 0.8589 | 0.8580 | 0.7989 | 2.4651 |
| MINI-BERT | FL | 0.8234 | 0.6668 | 0.4009 | 0.1969 |
| | | | | | |
| MEDIUM-BERT | Centralized | 0.8865 | 0.8777 | 0.8615 | 2.6671 |
| MEDIUM-BERT | FL | 0.8933 | 0.7441 | 0.4219 | 0.6790 |
| | | | | | |
| DISTIL-BERT | Centralized | 0.9037 | 0.8778 | 0.8635 | 4.3214 |
| DISTIL-BERT | FL | 0.8257 | 0.6754 | 0.5108 | 0.2852 |
| | | | | | |
| MINI-LM | Centralized | 0.8796 | 0.8649 | 0.8580 | 1.5045 |
| MINI-LM | FL | 0.9369 | 0.8662 | 0.8384 | 0.3835 |
| | | | | | |

## 2. Data Distribution Impact: IID vs. Non-IID (FL-MTL)
Comparing the effect of Non-IID data on Federated Multi-Task Learning.

````carousel
![Distribution Impact: SST-2](./plots/comp_distribution_val_sst2_acc.png)
<!-- slide -->
![Distribution Impact: QQP](./plots/comp_distribution_val_qqp_f1.png)
<!-- slide -->
![Distribution Impact: STS-B](./plots/comp_distribution_val_stsb_pearson.png)
````

| Model | Distribution | SST-2 Acc | QQP F1 | STSB Pear | Res (GB) |
| --- | --- | --- | --- | --- | --- |
| TINY_BERT | IID | 0.9014 | 0.7494 | 0.5212 | 0.0881 |
| TINY_BERT | Non-IID | 0.7810 | 0.4733 | 0.0081 | 0.0881 |
| | | | | | |
| MINI-BERT | IID | 0.7649 | 0.5602 | 0.2824 | 0.0630 |
| MINI-BERT | Non-IID | 0.8234 | 0.6668 | 0.4009 | 0.1969 |
| | | | | | |
| MEDIUM-BERT | IID | 0.9392 | 0.8655 | 0.8258 | 0.6807 |
| MEDIUM-BERT | Non-IID | 0.8933 | 0.7441 | 0.4219 | 0.6790 |
| | | | | | |
| DISTIL-BERT | IID | 0.8257 | 0.6754 | 0.5108 | 0.2852 |
| DISTIL-BERT | Non-IID | 0.9232 | 0.8054 | 0.6963 | 1.0968 |
| | | | | | |
| MINI-LM | IID | 0.9369 | 0.8662 | 0.8384 | 0.3835 |
| MINI-LM | Non-IID | 0.8888 | 0.8150 | 0.7003 | 0.3846 |
| | | | | | |

## 3. Multi-Task Learning Gain: Single-Task vs. MTL (FL)
Evaluating the performance delta between Single-Task and Multi-Task Federated Learning.

````carousel
![Task Type Comparison: SST-2](./plots/comp_tasktype_val_sst2_acc.png)
<!-- slide -->
![Task Type Comparison: QQP](./plots/comp_tasktype_val_qqp_f1.png)
<!-- slide -->
![Task Type Comparison: STS-B](./plots/comp_tasktype_val_stsb_pearson.png)
````

| Model | Task Type | SST-2 Acc | QQP F1 | STSB Pear | Res (GB) |
| --- | --- | --- | --- | --- | --- |
| TINY_BERT | Single-Task | 0.9060 | 0.3887 | 0.2186 | 0.0881 |
| TINY_BERT | Multi-Task (MTL) | 0.9014 | 0.7494 | 0.5212 | 0.0706 |
| | | | | | |
| MINI-BERT | Single-Task | 0.9151 | 0.6201 | 0.6504 | 0.1969 |
| MINI-BERT | Multi-Task (MTL) | 0.9266 | 0.7981 | 0.6940 | 0.1522 |
| | | | | | |
| MEDIUM-BERT | Single-Task | 0.9300 | 0.6852 | 0.6521 | 0.6795 |
| MEDIUM-BERT | Multi-Task (MTL) | 0.9392 | 0.8655 | 0.8258 | 0.5146 |
| | | | | | |
| DISTIL-BERT | Single-Task | 0.9346 | 0.7724 | 0.7753 | 1.0972 |
| DISTIL-BERT | Multi-Task (MTL) | 0.9461 | 0.8610 | 0.8284 | 0.8239 |
| | | | | | |
| MINI-LM | Single-Task | 0.9243 | 0.7644 | 0.7877 | 0.3839 |
| MINI-LM | Multi-Task (MTL) | 0.9369 | 0.8662 | 0.8384 | 0.2923 |
| | | | | | |
