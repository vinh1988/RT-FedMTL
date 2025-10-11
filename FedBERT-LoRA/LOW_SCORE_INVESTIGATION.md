# Low Score Investigation Report

## Overview
This document investigates the low performance observed in the federated learning experiments, particularly focusing on the STSB (Semantic Textual Similarity Benchmark) task which showed significantly lower R² scores compared to other tasks.

## Affected Components
- **Task**: STSB (Semantic Textual Similarity Benchmark)
- **Client**: Client 3
- **Metric**: R² Score (Regression)
- **Current Best R²**: 0.153
- **Current Final R²**: 0.048

## Root Cause Analysis

### 1. Task Complexity
- STSB is a regression task (predicting similarity scores from 0-5)
- More challenging than classification tasks (SST-2, QQP)
- Requires fine-grained understanding of semantic relationships
- **Pearson Correlation**: Current score of 0.241 indicates weak linear relationship between predictions and true scores

### 2. Model Limitations
- Using Tiny-BERT (pretrained on general text)
- May lack capacity for nuanced semantic understanding
- Shared architecture across heterogeneous tasks may lead to task interference

### 3. Data Issues
- Limited training samples (400 per client)
- Imbalanced distribution of similarity scores
- Potential label noise in the STSB dataset

### 4. Training Dynamics
- Fixed learning rate may not be optimal for regression
- No task-specific adaptation
- Aggregation might be dominated by classification tasks

## Evidence from Logs

### STSB Client Logs
```
[STSB Client] Round 15 - Best R²: 0.153 (RMSE: 0.287, MAE: 0.241, Pearson: 0.241)
[STSB Client] Round 22 - Final R²: 0.048 (RMSE: 0.302, MAE: 0.251, Pearson: 0.241)
```

### Pearson Correlation Analysis
- **Current Pearson**: 0.241 (weak positive correlation)
- **Interpretation**: 
  - 0.00-0.19: Very weak
  - 0.20-0.39: Weak
  - 0.40-0.59: Moderate
  - 0.60-0.79: Strong
  - 0.80-1.00: Very strong
- **Implications**: 
  - The model struggles to capture the linear relationship between input pairs and similarity scores
  - Suggests the need for better feature representation or model capacity

### System Metrics
- High CPU utilization (94%)
- Stable memory usage (~2.3GB)
- Consistent throughput (~750 samples/second)

## Recommended Actions

### Immediate Fixes
1. **Task-Specific Learning Rates**
   ```python
   # Current
   optimizer = AdamW(model.parameters(), lr=2e-5)
   
   # Proposed
   optimizer = AdamW([
       {'params': base_model.parameters(), 'lr': 2e-5},
       {'params': regression_head.parameters(), 'lr': 5e-5}
   ])
   ```

2. **Loss Function Modification**
   ```python
   # Current: MSE Loss
   loss_fn = nn.MSELoss()
   
   # Proposed: Custom loss combining MSE and negative Pearson
   class CombinedLoss(nn.Module):
       def __init__(self, alpha=0.5):
           super().__init__()
           self.alpha = alpha
           self.mse = nn.MSELoss()
           
       def pearson_loss(self, preds, targets):
           preds_centered = preds - preds.mean()
           targets_centered = targets - targets.mean()
           cov = (preds_centered * targets_centered).mean()
           preds_std = torch.sqrt((preds_centered ** 2).mean() + 1e-8)
           targets_std = torch.sqrt((targets_centered ** 2).mean() + 1e-8)
           pearson = cov / (preds_std * targets_std + 1e-8)
           return 1 - pearson  # Minimize this
           
       def forward(self, preds, targets):
           return self.alpha * self.mse(preds, targets) + \
                 (1 - self.alpha) * self.pearson_loss(preds, targets)
   
   loss_fn = CombinedLoss(alpha=0.7)
   ```

### Medium-Term Improvements
1. **Model Architecture**
   - Consider task-specific adapters
   - Add residual connections for regression head
   - Implement gradient reversal for domain adaptation

2. **Data Augmentation**
   - Back-translation for text pairs
   - Synonym replacement
   - Random masking

3. **Federated Learning Strategy**
   - Implement FedProx for better convergence
   - Add differential privacy for robustness
   - Implement client selection based on task performance

### Long-Term Solutions
1. **Pre-training**
   - Domain-adaptive pre-training on similar text pairs
   - Multi-task pre-training objective

2. **Architecture Search**
   - Neural architecture search for optimal model size
   - Task-specific model scaling

3. **Advanced Techniques**
   - Knowledge distillation from larger models
   - Meta-learning for few-shot adaptation
   - Contrastive learning for better representations

## Testing Plan
1. **A/B Testing**
   - Test with different loss functions
   - Compare learning rate schedules
   - Evaluate impact of batch size

2. **Validation Strategy**
   - Implement k-fold cross-validation
   - Add early stopping based on validation metrics
   - Monitor gradient norms and weight updates

## Success Metrics
- **Primary**: 
  - R² > 0.4 on STSB test set
  - Pearson correlation > 0.6 (moderate to strong relationship)
- **Secondary**: Training stability (reduced variance across rounds)
- **Tertiary**: Convergence speed (reduced training time)

## Timeline
1. **Week 1**: Implement and test immediate fixes
2. **Week 2-3**: Deploy medium-term improvements
3. **Month 2**: Evaluate and implement long-term solutions

## Monitoring
- Track metrics per task
- Monitor gradient norms and updates
- Log detailed training dynamics

## Conclusion
The low scores in the STSB task appear to stem from a combination of model capacity, task difficulty, and training dynamics. The proposed changes should help improve performance while maintaining the federated learning setup. Regular monitoring and iterative improvements will be key to achieving the target metrics.
