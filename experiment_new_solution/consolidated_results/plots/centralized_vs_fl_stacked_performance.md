# Centralized vs FL: Stacked Performance Comparison

![Centralized vs FL: Stacked Performance Comparison](centralized_vs_fl_stacked_performance.png)

## Description
Stacked performance comparison between Centralized and Federated Learning (FL) paradigms. Each bar shows both paradigms stacked together, sorted by total combined performance.

## Key Insights
- **Performance Hierarchy**: Clear ranking of model-task combinations by total performance
- **Paradigm Contribution**: Visual representation of each paradigm's contribution to total
- **Top Performers**: Best configurations show strong performance from both paradigms
- **Task Patterns**: Different tasks show different paradigm dominance patterns

## Metrics Data

| Model | Task | Centralized | FL | Total | Difference | Percent_Diff |
|---|---|---|---|---|---|---|
| distil-bert | SST2 | 0.9019 | 0.9074 | 1.8093 | 0.0055 | 0.6043 |
| medium-bert | SST2 | 0.8899 | 0.8939 | 1.7838 | 0.0040 | 0.4486 |
| mini-lm | SST2 | 0.8905 | 0.8822 | 1.7727 | -0.0083 | -0.9328 |
| mini-bert | SST2 | 0.8658 | 0.8575 | 1.7233 | -0.0083 | -0.9616 |
| distil-bert | QQP | 0.8297 | 0.8359 | 1.6655 | 0.0062 | 0.7495 |
| tiny_bert | SST2 | 0.8263 | 0.8363 | 1.6626 | 0.0101 | 1.2180 |
| medium-bert | QQP | 0.8346 | 0.8130 | 1.6476 | -0.0215 | -2.5773 |
| mini-lm | QQP | 0.7953 | 0.8322 | 1.6275 | 0.0368 | 4.6309 |
| distil-bert | STSB | 0.8674 | 0.7027 | 1.5701 | -0.1647 | -18.9834 |
| mini-bert | QQP | 0.7880 | 0.7682 | 1.5562 | -0.0197 | -2.5051 |
| medium-bert | STSB | 0.8644 | 0.6455 | 1.5099 | -0.2189 | -25.3219 |
| mini-lm | STSB | 0.8600 | 0.5992 | 1.4592 | -0.2608 | -30.3269 |
| tiny_bert | QQP | 0.7194 | 0.7155 | 1.4349 | -0.0038 | -0.5335 |
| mini-bert | STSB | 0.8152 | 0.5069 | 1.3222 | -0.3083 | -37.8191 |
| tiny_bert | STSB | 0.6872 | 0.2779 | 0.9651 | -0.4093 | -59.5649 |


## Data Source
- **File**: master_model_comparison.csv
- **Total Experiments**: 50
- **Models**: distil-bert, medium-bert, mini-bert, mini-lm, tiny_bert
- **Paradigms**: Centralized, FL
- **Task Types**: Single-Task, Multi-Task (MTL)
- **Distributions**: IID, Non-IID

---
