# Training and Validation Split Implementation

## Overview

The system now properly separates training and validation data, with both metrics reported in the results CSV.

## Configuration

In `federated_config.yaml`:
```yaml
stsb:
  train_samples: 2000    # Used for training (model learns from these)
  val_samples: 400       # Used for validation (model evaluates on these)
```

## How It Works

### 1. **Data Loading** (`src/datasets/federated_datasets.py`)
- Loads total of 2400 samples (2000 + 400)
- Splits into:
  - **Training**: First 2000 samples → `texts` and `labels`
  - **Validation**: Next 400 samples → `val_texts` and `val_labels`

### 2. **Training** (`src/core/federated_client.py`)
- **Training Phase**:
  - Model learns from 2000 training samples
  - Parameters are updated via backpropagation
  - Loss and accuracy calculated on training data
  
- **Validation Phase**:
  - Model evaluates on 400 validation samples
  - NO parameter updates (evaluation mode)
  - Provides unbiased performance estimate

### 3. **Metrics Reported**

#### Training Metrics:
- `accuracy`: Model performance on training data
- `loss`: Training loss
- `samples_processed`: 2000 (training samples)
- `correct_predictions`: Count within tolerance (training)

#### Validation Metrics:
- `val_accuracy`: Model performance on unseen validation data
- `val_loss`: Validation loss  
- `val_samples`: 400 (validation samples)
- For STSB: `val_correlation`, `val_mae`

## CSV Output Format

```csv
round,client_id,task,accuracy,loss,samples_processed,correct_predictions,val_accuracy,val_loss,val_samples,timestamp
1,stsb_client,stsb,0.45,0.32,2000,900,0.38,0.35,400,2025-10-19 22:00:00
```

**Reading the Results:**
- **Training**: 45% accuracy, 900/2000 within tolerance
- **Validation**: 38% accuracy on 400 unseen samples
- **Interpretation**: Model is learning (training acc > val acc is normal)

## Why This Matters

### Without Validation:
```
Round 1: Training acc = 0.50
Round 2: Training acc = 0.80  ← Might be overfitting!
```

### With Validation:
```
Round 1: Training acc = 0.50, Val acc = 0.45  Learning
Round 2: Training acc = 0.80, Val acc = 0.50  Still learning
Round 3: Training acc = 0.95, Val acc = 0.48  Overfitting!
```

## Key Differences

| Aspect | Training Data | Validation Data |
|--------|--------------|----------------|
| **Purpose** | Learn patterns | Evaluate performance |
| **Updates** | Parameters updated | No updates (eval mode) |
| **Samples** | 2000 | 400 |
| **Metrics** | `accuracy`, `loss` | `val_accuracy`, `val_loss` |
| **When** | Every training batch | After training complete |

## Expected Behavior

### Good Training:
```
Round 1: Train=0.45, Val=0.40 (small gap)
Round 2: Train=0.55, Val=0.52 (both improving)
Round 3: Train=0.65, Val=0.60 (consistent gap)
```

### Overfitting:
```
Round 1: Train=0.45, Val=0.40
Round 2: Train=0.75, Val=0.42 (gap increasing)
Round 3: Train=0.95, Val=0.38 (val getting worse!)
```

## Code Flow

```python
# 1. Load data with split
task_data = dataset.prepare_data()
# Returns: {texts, labels, val_texts, val_labels}

# 2. Create separate dataloaders
train_dataloader = get_dataloader(texts, labels)
val_dataloader = get_dataloader(val_texts, val_labels)

# 3. Training phase
model.train()
for batch in train_dataloader:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. Validation phase
model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        # Evaluate only, no updates
        val_loss = calculate_loss()

# 5. Return both metrics
return {
    'accuracy': train_acc,
    'val_accuracy': val_acc
}
```

## Answering Your Question

**Q: "Why 446 correct predictions when I set 400 samples for validation?"**

**A:** The 446 is from **2000 training samples**, NOT the 400 validation samples!

- `samples_processed: 2000` ← Training samples
- `correct_predictions: 446` ← 22.3% of training samples within tolerance
- `val_samples: 400` ← Validation samples (separate column)

The 446 and 400 are completely different things:
- **446**: How many training predictions are "correct" (within 0.1 tolerance)
- **400**: How many samples are in the validation set

---

**Date**: October 19, 2025  
**Status**:  Training/Validation Split Implemented
