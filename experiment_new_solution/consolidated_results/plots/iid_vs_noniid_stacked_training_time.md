# IID vs Non-IID: Stacked Training Time Comparison

![IID vs Non-IID: Stacked Training Time Comparison](iid_vs_noniid_stacked_training_time.png)

## Description
Stacked training time comparison between IID and Non-IID data distributions. Each bar shows both distributions stacked together, sorted by total combined training time.

## Key Insights
- **Convergence Patterns**: Different models show different convergence requirements
- **Distribution Impact**: Visual representation of Non-IID training overhead
- **Model Adaptation**: Training time scaling with distribution complexity
- **Efficiency Patterns**: Some models handle Non-IID more efficiently

## Metrics Data

| Model | IID | Non-IID | Total | Ratio | Difference |
|---|---|---|---|---|---|
| distil-bert | 21586.0992 | 31711.8425 | 53297.9417 | 1.4691 | 10125.7433 |
| medium-bert | 16246.8858 | 20739.0875 | 36985.9733 | 1.2765 | 4492.2017 |
| mini-lm | 12098.9610 | 12179.0750 | 24278.0360 | 1.0066 | 80.1140 |
| mini-bert | 4210.5836 | 6764.9975 | 10975.5811 | 1.6067 | 2554.4139 |
| tiny_bert | 4846.0188 | 4224.5000 | 9070.5188 | 0.8717 | -621.5188 |


## Data Source
- **File**: master_model_comparison.csv
- **Total Experiments**: 50
- **Models**: distil-bert, medium-bert, mini-bert, mini-lm, tiny_bert
- **Paradigms**: Centralized, FL
- **Task Types**: Single-Task, Multi-Task (MTL)
- **Distributions**: IID, Non-IID

---
