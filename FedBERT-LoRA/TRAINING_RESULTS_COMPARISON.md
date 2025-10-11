# Training Results Comparison - Before vs After Fix

## Configuration Changes

### Before (Problematic):
```ini
[FULL_SCALE]
samples_per_client = 5749
[SCALABILITY_STUDY]
client_ranges = 2,3,5,7,10
```

### After (Fixed):
```ini
[FULL_SCALE]
samples_per_client = 400
[SCALABILITY_STUDY]
client_ranges = 3,5,7,10  # Removed 2 clients minimum
```

---

## Results Comparison - 3 Clients Configuration

### ✅ FIXED: Data Distribution Now Balanced!

#### Before (samples_per_client = 5749):
```
Client 1 (SST-2): 5,748 samples ✅
Client 2 (QQP):   5,748 samples ✅
Client 3 (STSB):    480 samples ❌ (12x fewer!)
```

#### After (samples_per_client = 400):
```
Client 1 (SST-2): 399 samples ✅
Client 2 (QQP):   399 samples ✅
Client 3 (STSB):  400 samples ✅ (BALANCED!)
```

**Improvement:** Client 3 now has the same amount of data as other clients!

---

## Training Metrics Comparison

### Client 1 (SST-2 - Classification)
**Round 1:**
- Before: Loss=0.5643, Acc=0.7077
- After:  Loss=0.6773, Acc=0.5731

**Round 3:**
- Before: Loss=0.2736, Acc=0.8914
- After:  Loss=0.6352, Acc=0.6374

**Analysis:** Lower accuracy with 400 samples is expected (less training data), but more realistic for federated learning scenarios.

---

### Client 2 (QQP - Classification)
**Round 1:**
- After: Loss=0.6724, Acc=0.6040

**Round 3:**
- After: Loss=0.6242, Acc=0.6533

**Analysis:** Consistent performance with Client 1, showing balanced learning.

---

### Client 3 (STSB - Regression)
**Before (480 samples):**
```
Round 1: Loss=0.0055, Acc=0.0000 (regression task)
Round 22: Loss=0.0046, Acc=0.0000
Distribution: {'bin_0': 0, 'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 480}
```

**After (400 samples):**
```
Round 1: Loss=0.3252, Acc=0.0000 (regression task)
Round 3: Loss=0.0071, Acc=0.0000
Round 5: Loss=0.0057, Acc=0.0000
Distribution: {'bin_0': 0, 'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 400}
```

**Analysis:**
- ✅ Sample count now balanced (400 vs 399)
- ✅ Loss values are learning properly (0.3252 → 0.0057)
- ⚠️ Still shows Acc=0.0 (expected for regression tasks)
- ⚠️ Distribution still all in bin_4 (data heterogeneity issue remains)

---

## Aggregate Metrics (From CSV)

### Round 5 (3 clients):
```
average_accuracy: 0.4676
accuracy_std: 0.3381
worst_client_accuracy: 0.0 (Client 3 - regression)
best_client_accuracy: 0.7878 (Client 1 or 2)
```

### Round 10 (3 clients):
```
average_accuracy: 0.5940
accuracy_std: 0.4245
worst_client_accuracy: 0.0
best_client_accuracy: 0.9666
```

**Note:** These averages still include Client 3's 0.0 (regression task), which skews the results.

---

## Key Improvements ✅

1. **✅ Balanced Sample Distribution**
   - All clients now have ~400 samples
   - No more 12x data imbalance

2. **✅ Scalability Starts at 3 Clients**
   - Removed 2-client test
   - More realistic federated learning scenario

3. **✅ Faster Training**
   - 400 samples per client vs 5,749
   - Experiment completed in ~5 minutes vs ~8 minutes

4. **✅ More Realistic Scenario**
   - Matches real-world federated learning constraints
   - Better for testing heterogeneous multi-task learning

---

## Remaining Issues ⚠️

### 1. **STSB Distribution Still Skewed**
```
{'bin_0': 0, 'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 400}
```
- All samples still in highest similarity range
- This is a fundamental issue with STSB's non-IID split for regression
- **Recommendation:** Consider using a different regression task or adjusting non-IID alpha

### 2. **Mixed Task Type Metrics**
- CSV still averages classification metrics with regression 0.0 values
- `average_accuracy` includes Client 3's 0.0, making it misleading
- **Recommendation:** Implement separate metric tracking for classification vs regression

### 3. **Classification Metrics for Regression**
- Client 3 logs: `Acc=0.0000, P=0.0000, R=0.0000, F1=0.0000`
- These are meaningless for regression tasks
- **Recommendation:** Add regression-specific metrics (MSE, RMSE, MAE, R², Pearson)

---

## Recommendations for Next Steps

### Option A: Replace STSB with Classification Task
```ini
[SCENARIO_1]
tasks = sst2,qqp,mnli  # All classification
```
**Pros:** Consistent metrics, easier to interpret
**Cons:** Loses regression task diversity

### Option B: Implement Separate Metric Tracking
```python
# In server aggregation
classification_metrics = [m for m in metrics if m.task_type == "classification"]
regression_metrics = [m for m in metrics if m.task_type == "regression"]
```
**Pros:** Proper handling of mixed tasks
**Cons:** Requires code changes

### Option C: Add Regression Metrics
```python
if task_type == "regression":
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
```
**Pros:** Better evaluation of regression performance
**Cons:** Requires code changes

### Option D: Adjust Non-IID Alpha for Better Distribution
```ini
[SCENARIO_1]
non_iid_alpha = 1.0  # Increase from 0.5 for more diversity
```
**Pros:** May improve bin distribution
**Cons:** May not fully solve the issue

---

## Conclusion

**The main issue has been FIXED! ✅**

The data imbalance problem (480 vs 5,748 samples) has been resolved. All clients now train on balanced datasets (~400 samples each).

**However, two design issues remain:**
1. STSB regression metrics showing as 0.0 (by design, but confusing)
2. All STSB samples in one distribution bin (data heterogeneity issue)

For a production-ready experiment, consider implementing **Option B** (separate metric tracking) or **Option A** (use only classification tasks).
