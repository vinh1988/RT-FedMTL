# Single vs Multi-Task: Resource Usage Comparison

![Single vs Multi-Task: Resource Usage Comparison](single_vs_multitask_stacked_resource_usage_balanced.png)

## Description
Resource usage comparison between Single-Task and Multi-Task Learning approaches. All text and numbers are 1.5x larger for optimal readability.

## Key Insights
- **Resource Efficiency**: Clear ranking of models by total resource consumption
- **Multi-Task Benefits**: Visual representation of shared parameter efficiency
- **Model Patterns**: Different models show different resource scaling
- **Deployment Considerations**: Resource requirements for different task configurations

## Metrics Data

| Model | Single | Multi | Total | Ratio | Difference |
|---|---|---|---|---|---|
| DistilBERT | 2.6732 | 1.6983 | 4.3715 | 0.6353 | -0.9750 |
| BERT-Medium | 1.6672 | 1.0527 | 2.7199 | 0.6314 | -0.6145 |
| BERT-Mini | 1.3228 | 0.7304 | 2.0532 | 0.5522 | -0.5923 |
| MiniLM | 0.9411 | 0.5954 | 1.5364 | 0.6326 | -0.3457 |
| TinyBERT | 0.2154 | 0.1390 | 0.3544 | 0.6451 | -0.0764 |


## Data Source
- **File**: master_model_comparison.csv
- **Total Experiments**: 50
- **Models**: DistilBERT, BERT-Medium, BERT-Mini, MiniLM, TinyBERT
- **Paradigms**: Centralized, FL
- **Task Types**: Single-Task, Multi-Task
- **Distributions**: IID, Non-IID

---
