# IID vs Non-IID: Stacked Performance Comparison

![IID vs Non-IID: Stacked Performance Comparison](iid_vs_noniid_stacked_performance.png)

## Description
Stacked performance comparison between IID and Non-IID data distributions. Each bar shows both distributions stacked together, sorted by total combined performance.

## Key Insights
- **Performance Hierarchy**: Clear ranking of model-task combinations by total performance
- **Distribution Impact**: Visual representation of each distribution's contribution to total
- **Robustness Patterns**: Height ratios indicate model robustness to distribution shifts
- **Task Sensitivity**: Different tasks show different IID vs Non-IID balance

## Metrics Data

| Model | Task | IID | Non-IID | Total | Degradation | Percent_Degrad |
|---|---|---|---|---|---|---|
| distil-bert | SST2 | 0.8939 | 0.9289 | 1.8228 | -0.0350 | -3.9125 |
| medium-bert | SST2 | 0.8830 | 0.9117 | 1.7947 | -0.0286 | -3.2412 |
| mini-lm | SST2 | 0.8741 | 0.9065 | 1.7807 | -0.0324 | -3.7075 |
| mini-bert | SST2 | 0.8558 | 0.8693 | 1.7250 | -0.0135 | -1.5731 |
| distil-bert | QQP | 0.8260 | 0.8494 | 1.6754 | -0.0233 | -2.8233 |
| tiny_bert | SST2 | 0.8277 | 0.8435 | 1.6712 | -0.0158 | -1.9082 |
| mini-lm | QQP | 0.8036 | 0.8525 | 1.6561 | -0.0489 | -6.0856 |
| medium-bert | QQP | 0.8260 | 0.8086 | 1.6346 | 0.0174 | 2.1100 |
| mini-bert | QQP | 0.7767 | 0.7711 | 1.5478 | 0.0056 | 0.7204 |
| distil-bert | STSB | 0.7685 | 0.7358 | 1.5043 | 0.0327 | 4.2521 |
| tiny_bert | QQP | 0.7276 | 0.6952 | 1.4228 | 0.0323 | 4.4437 |
| mini-lm | STSB | 0.6572 | 0.7440 | 1.4012 | -0.0868 | -13.2136 |
| medium-bert | STSB | 0.8092 | 0.5370 | 1.3462 | 0.2722 | 33.6406 |
| mini-bert | STSB | 0.6517 | 0.5256 | 1.1774 | 0.1261 | 19.3443 |
| tiny_bert | STSB | 0.5648 | 0.1133 | 0.6782 | 0.4515 | 79.9312 |


## Data Source
- **File**: master_model_comparison.csv
- **Total Experiments**: 50
- **Models**: distil-bert, medium-bert, mini-bert, mini-lm, tiny_bert
- **Paradigms**: Centralized, FL
- **Task Types**: Single-Task, Multi-Task (MTL)
- **Distributions**: IID, Non-IID

---
