# STSB Accuracy Fix: Sigmoid Activation for Regression

## Problem Summary

The STSB (Semantic Textual Similarity Benchmark) task was showing very low accuracy (0-13%) despite the model training. 

### Root Cause Analysis

**From the logs (`federated_client_stsb_client.log`):**
```
Predictions: [0.0055602, -0.00101179, -0.00307259, ...]
Labels:      [0.6,       0.8,         0.16,      ...]
Pred mean: -0.0004, std: 0.0037
Label mean: 0.5143, std: 0.3094
```

**The Issue:**
- **Model predictions**: Values near 0 (range: -0.005 to 0.006)
- **Actual labels**: Normalized to 0-1 range (mean: 0.514)
- **Result**: MAE = 0.514 (essentially predicting 0 for all samples)

### Why This Happened

1. **Label Normalization**: STSB labels are normalized from 0-5 scale to 0-1 scale:
   ```python
   # In src/datasets/federated_datasets.py
   labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to 0-1
   ```

2. **Missing Activation**: The LoRA model was outputting **raw logits** (unbounded) for all tasks, including STSB regression:
   ```python
   # Before fix - no activation for regression
   combined_logits = lora_output  # Can be any value!
   return combined_logits
   ```

3. **Scale Mismatch**: Without sigmoid activation, the model predicts values around 0, while labels are 0-1. This causes:
   - Very high MAE (≈ 0.514)
   - Very low correlation (0-43%)
   - Near-zero accuracy

## The Fix

**File**: `src/lora/federated_lora.py`, lines 158-160

```python
combined_logits = lora_output

# Apply sigmoid activation for regression tasks to constrain output to 0-1
if task_name == 'stsb':
    combined_logits = torch.sigmoid(combined_logits)

return combined_logits
```

### What Sigmoid Does

Sigmoid function: `σ(x) = 1 / (1 + e^(-x))`

- **Input (raw logits)**: Any value (-∞ to +∞)
- **Output**: Constrained to (0, 1) range
- **Properties**:
  - `σ(0) = 0.5` (center point)
  - `σ(-5) ≈ 0.007` (near 0)
  - `σ(+5) ≈ 0.993` (near 1)

This matches the normalized STSB label range perfectly!

## Verification

**Test Results** (`test_sigmoid_fix.py`):
```
 SST2 (classification) output range: [-0.0559, -0.0221] (raw logits)
 QQP (classification) output range: [-0.0058, 0.0154] (raw logits)
 STSB (regression) output range: [0.4922, 0.4944] (sigmoid-activated)
    ALL VALUES IN 0-1 RANGE!
```

## Expected Results After Fix

### Before Fix:
```csv
round,client_id,task,accuracy,mae,correlation
1,stsb_client,stsb,0.0,0.514,0.0
2,stsb_client,stsb,0.027,0.515,0.027
3,stsb_client,stsb,0.132,0.514,0.132
```

### After Fix (Expected):
```csv
round,client_id,task,accuracy,mae,correlation
1,stsb_client,stsb,0.25,0.15,0.25
2,stsb_client,stsb,0.40,0.12,0.40
3,stsb_client,stsb,0.55,0.09,0.55
```

**Key Improvements:**
-  Predictions now in 0-1 range (matching labels)
-  MAE should drop from ~0.514 to ~0.10-0.15
-  Correlation-based accuracy should reach 40-60%
-  Model can actually learn the regression task

## Why This Is Critical

1. **Scale Matching**: Neural networks learn best when inputs and outputs are on similar scales
2. **Gradient Flow**: Sigmoid provides smooth gradients for backpropagation
3. **Meaningful Loss**: MSE loss now measures actual prediction error, not scale mismatch
4. **Interpretability**: Predictions directly represent similarity scores (0 = dissimilar, 1 = identical)

## Classification vs Regression

| Aspect | Classification (SST2, QQP) | Regression (STSB) |
|--------|---------------------------|-------------------|
| **Output** | Raw logits | Sigmoid-activated |
| **Range** | (-∞, +∞) | (0, 1) |
| **Loss** | Cross-Entropy | MSE |
| **Accuracy** | Correct/Total | Correlation |

## Next Steps

1. **Run Training**: Test with the sigmoid activation enabled
2. **Monitor Metrics**: Watch for:
   - STSB predictions in 0-1 range
   - MAE decreasing over rounds
   - Correlation increasing (target: 40-60%)
3. **Compare Results**: Previous runs vs. new runs

## Technical Notes

- The fix only affects the **forward pass** during training and inference
- No changes needed to loss functions (MSE is already correct)
- No changes needed to labels (already normalized to 0-1)
- Classification tasks (SST2, QQP) are unaffected

---

**Date**: October 19, 2025  
**Status**:  Fixed - Ready for Testing
