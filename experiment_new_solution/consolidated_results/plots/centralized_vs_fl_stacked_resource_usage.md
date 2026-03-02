# Centralized vs FL: Stacked Resource Usage Comparison

![Centralized vs FL: Stacked Resource Usage Comparison](centralized_vs_fl_stacked_resource_usage.png)

## Description
Stacked resource usage comparison between Centralized and Federated Learning (FL) paradigms. Each bar shows both paradigms stacked together, sorted by total combined resource usage.

## Key Insights
- **Resource Hierarchy**: Clear ranking of models by total resource requirements
- **FL Efficiency**: Visual representation of FL's distributed resource efficiency
- **Model Scaling**: Resource usage patterns across different model sizes
- **Deployment Patterns**: Different resource requirements for each paradigm

## Metrics Data

| Model | Centralized | FL | Total | Ratio | Difference |
|---|---|---|---|---|---|
| distil-bert | 4.2673 | 0.9605 | 5.2278 | 0.2251 | -3.3067 |
| medium-bert | 2.6579 | 0.5970 | 3.2549 | 0.2246 | -2.0609 |
| mini-bert | 2.4528 | 0.1745 | 2.6274 | 0.0712 | -2.2783 |
| mini-lm | 1.4998 | 0.3381 | 1.8379 | 0.2255 | -1.1616 |
| tiny_bert | 0.3431 | 0.0793 | 0.4224 | 0.2313 | -0.2637 |


## Data Source
- **File**: master_model_comparison.csv
- **Total Experiments**: 50
- **Models**: distil-bert, medium-bert, mini-bert, mini-lm, tiny_bert
- **Paradigms**: Centralized, FL
- **Task Types**: Single-Task, Multi-Task (MTL)
- **Distributions**: IID, Non-IID

---
