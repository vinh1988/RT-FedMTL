# Training Analysis: Client 3 - SCENARIO_1_scalability_3c

## Issue Summary
The training log shows **all classification metrics (Accuracy, Precision, Recall, F1) are 0.0000**, which is INCORRECT and indicates a serious problem.

## Root Cause Analysis

### **CRITICAL ISSUE: Severe Data Imbalance**

Comparing the 3 clients:
- **Client 1 (SST-2)**: 5,748 samples ✅
- **Client 2 (QQP)**: 5,748 samples ✅  
- **Client 3 (STSB)**: **480 samples** ❌ (12x fewer!)

### 1. **Why Client 3 Has So Few Samples**

The configuration requested:
- `samples_per_client = 5749`
- But STSB dataset is much smaller than SST-2 and QQP

**STSB dataset size**: Only ~5,749 samples total in training set
- When split among clients with non-IID distribution, Client 3 only got 480 samples
- This is only **8.3%** of the requested amount

### 2. **Task Type: Regression (Correct)**
- STSB is a regression task (predicting similarity scores 0-5)
- Code correctly identifies this: `no_lora_federated_system.py:409`
  ```python
  self.task_type = "regression" if task_name == "stsb" else "classification"
  ```

### 3. **Why Metrics Are 0.0**
In `no_lora_federated_system.py:563-597`:

```python
if self.task_type == "classification" and len(all_predictions) > 0:
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(...)
else:
    # Fallback for regression or empty predictions
    accuracy = 0.0
    precision = recall = f1 = 0.0
```

**Since STSB is regression, classification metrics are intentionally set to 0.0**

### 4. **Data Distribution Analysis**
```
Client 3 (stsb): 480 samples, distribution: {'bin_0': 0, 'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 480}
```

This distribution shows **extreme skew**:
- All 480 samples fall into `bin_4` (highest similarity scores: 4.0-5.0 range)
- This is due to non-IID split on an already small dataset
- Creates severe data heterogeneity
- Code reference: `no_lora_federated_system.py:293-296`

### 5. **Impact on Federated Aggregation**

Looking at the CSV results, the **average metrics are dominated by Clients 1 & 2**:
- Round 1: avg_accuracy = 0.354 (average of 0.707 and 0.0)
- Round 3: avg_accuracy = 0.518 (average of 0.891, 0.664, and 0.0)
- Client 3 contributes **0.0 to all classification metrics**

This creates **misleading aggregate statistics** because:
1. Client 3 is doing regression (different task type)
2. Client 3 has 12x fewer samples
3. Averaging classification metrics with 0.0 from regression task is invalid

### 6. **Loss Values Show Learning (But Limited)**
- Loss decreases from ~0.0055 to ~0.0041 over 22 rounds
- MSE loss for regression: `F.mse_loss(student_logits.squeeze(), labels)`
- But with only 480 samples, generalization is limited

## Conclusion

**The results ARE problematic for several reasons:**

❌ **Problems Identified:**
1. **Severe data imbalance**: Client 3 has 12x fewer samples (480 vs 5,748)
2. **Invalid metric aggregation**: Mixing classification (Clients 1&2) with regression (Client 3)
3. **Misleading CSV results**: Average metrics don't represent true federated performance
4. **Dataset size mismatch**: STSB is too small for the requested `samples_per_client=5749`
5. **Extreme distribution skew**: All samples in one bin reduces learning diversity

## Recommendations

### **Option 1: Fix Data Sampling (Recommended)**

Modify the data loading to ensure balanced samples across all clients:

```python
# In no_lora_federated_system.py, around line 195-200
if task_name == "stsb":
    dataset = load_dataset("glue", "stsb")[split]
    texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
    labels = [float(item["label"]) / 5.0 for item in dataset]
    
    # NEW: Limit samples_per_client based on available data
    max_available = len(texts) // total_clients
    actual_samples = min(samples_per_client, max_available)
    logger.warning(f"STSB has limited data. Requesting {actual_samples} samples instead of {samples_per_client}")
```

### **Option 2: Use Different Dataset Split**

Modify `experiment_config.ini`:
```ini
[SCENARIO_1]
tasks = sst2,qqp,mnli  # Replace stsb with mnli (larger dataset)
# OR reduce samples_per_client to match STSB size
samples_per_client = 400  # Match smallest dataset
```

### **Option 3: Separate Task-Specific Metrics**

Update aggregation logic to handle mixed task types:

```python
# In server aggregation code
classification_clients = [c for c in clients if c.task_type == "classification"]
regression_clients = [c for c in clients if c.task_type == "regression"]

# Calculate separate averages
if classification_clients:
    avg_classification_metrics = calculate_avg(classification_clients)
if regression_clients:
    avg_regression_metrics = calculate_avg(regression_clients)
```

### **Option 4: Add Regression-Specific Metrics**

In `no_lora_federated_system.py:560-640`:

```python
if self.task_type == "regression" and len(all_predictions) > 0:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    logger.info(f"Client {self.client_id} training complete: "
               f"Loss={avg_loss:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, "
               f"MAE={mae:.4f}, R²={r2:.4f}, Pearson={pearson_corr:.4f}")
```

## Immediate Action Required

**Your experiment has a fundamental design flaw:**

1. ❌ **Multi-task learning with mixed task types** (classification + regression) requires separate metric tracking
2. ❌ **STSB dataset is too small** for the requested 5,749 samples per client
3. ❌ **Current CSV results are misleading** - they average incompatible metrics

**Choose one:**
- **A)** Replace STSB with a larger classification task (MNLI, QNLI)
- **B)** Reduce `samples_per_client` to 400 to match STSB size
- **C)** Implement separate metric tracking for regression vs classification
- **D)** Use only classification tasks for SCENARIO_1

## Files to Modify

1. **`experiment_config.ini`** (Line 33):
   ```ini
   # Current (problematic):
   tasks = sst2,qqp,stsb
   
   # Option A - All classification:
   tasks = sst2,qqp,mnli
   
   # Option B - Reduce samples:
   samples_per_client = 400
   ```

2. **`no_lora_federated_system.py`** (Lines 560-640):
   - Add regression-specific metrics calculation
   - Separate logging for regression vs classification

3. **Server aggregation logic**:
   - Don't average metrics across different task types
   - Track classification and regression separately
