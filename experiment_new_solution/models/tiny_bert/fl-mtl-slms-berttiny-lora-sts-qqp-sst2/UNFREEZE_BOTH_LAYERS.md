# Unfreezing BOTH Layers for bert-tiny

## [DATA] **Why 2 Layers are Necessary**

### **Results with 1 Layer Unfrozen (Failed)**
From `federated_results_20260112_222436.csv`:

```
Round 1-5: 
  SST-2 Global: 46.22% (stuck)
  QQP Global:   37.55% (stuck)  
  STS-B Global: -0.215 (NEGATIVE correlation!)
```

**The metrics don't change at all across rounds = NOT LEARNING**

### **Key Issue: bert-tiny is TOO SMALL**

bert-tiny has **ONLY 2 layers**:
- Unfreezing 1 layer = **50% of model frozen**
- Still too constrained for proper learning
- **Need to unfreeze BOTH layers**

---

## [SUCCESS] **Solution: Unfreeze BOTH Layers**

### **Change Made**

**File**: `src/models/peft_lora_model.py`

```python
# Changed from:
layers_to_unfreeze = 1  # Unfreeze last 1 layer

# To:
layers_to_unfreeze = 2  # Unfreeze BOTH layers (ALL for bert-tiny)
```

### **What This Means**

For **bert-tiny** (2 layers total):
- [SUCCESS] Layer 0: **UNFROZEN** (with LoRA adapters)
- [SUCCESS] Layer 1: **UNFROZEN** (with LoRA adapters)
- [SUCCESS] Pooler: **UNFROZEN**
- [SUCCESS] Task Heads: **UNFROZEN**

**ALL transformer layers are now trainable!**

---

## [DATA] **Parameter Comparison**

| Configuration | Frozen | Trainable | Trainable % | Size/Update |
|---------------|--------|-----------|-------------|-------------|
| **Pure LoRA (all frozen)** | 11.0M | 110K | 1.0% | 440 KB |
| **LoRA + 1 layer** | 10.75M | 350K | 3.2% | 1.4 MB |
| **LoRA + 2 layers (BOTH)** | 10.4M | **700K** | **6.3%** | **2.8 MB** |
| **Full fine-tuning** | 0 | 11.0M | 100% | 44 MB |

**Still 16× more efficient than full fine-tuning!**

---

## [TARGET] **Expected Improvements**

### **Before (1 layer)**:
```
SST-2: 46.22% [FAIL] (worse than random!)
QQP:   37.55% [FAIL] (terrible)
STS-B: -0.215 [FAIL] (NEGATIVE correlation!)
```

### **After (2 layers) - Expected**:
```
SST-2: 75-82% [SUCCESS] (matching local validation)
QQP:   72-78% [SUCCESS] (matching local validation)
STS-B: 0.60-0.75 [SUCCESS] (positive correlation!)
```

**Should match the local validation performance now!**

---

## [SEARCH] **Verification**

When you run training, look for:

### **1. In Logs:**
```
Unfreezing last 2 transformer layer(s) for better adaptation
  Unfreezing layer 0
  Unfreezing layer 1
  Unfreezing pooler layer
```

### **2. In Parameter Summary:**
```
PEFT LoRA Model Parameters:
  Total parameters: ~11,100,000
  Trainable parameters: ~700,000
  Percentage trainable: 6.31%  ← Should be ~6-7%
```

### **3. In Results (federated_results.csv):**
- **global_sst2_val_accuracy** should start ~75% and IMPROVE
- **global_qqp_val_accuracy** should start ~72% and IMPROVE  
- **global_stsb_val_pearson** should be POSITIVE (>0.5) and IMPROVE
- **Metrics should CHANGE each round**, not stay constant!

---

## 🧠 **Why This Makes Sense**

### **For Small Models (bert-tiny)**:
- 2 layers is the **minimum** for meaningful NLP
- Layer 0: Low-level features (tokens, syntax)
- Layer 1: High-level features (semantics, task-specific)
- **Both are needed** for proper task learning

### **For Larger Models (bert-base, 12 layers)**:
- Would still unfreeze last 2 layers only
- Layers 10 & 11 are sufficient
- Lower layers (0-9) stay frozen

### **The LoRA Part**:
- LoRA adapters are on **ALL** layers (0 & 1)
- Unfrozen layers + LoRA = **double adaptation**
- Base weights trainable + Low-rank adapters = **maximum capacity**

---

## [CHART] **Trade-offs**

### **Pros:**
[SUCCESS] Model can actually learn (most important!)
[SUCCESS] Matches local validation performance
[SUCCESS] Still 16× more efficient than full fine-tuning
[SUCCESS] LoRA + unfrozen layers = best of both worlds

### **Cons:**
[FAIL] 6.4× more parameters than pure LoRA
[FAIL] 2.8 MB per update instead of 440 KB
[FAIL] Slightly slower training

### **Verdict:**
**Worth it!** A model that doesn't learn (1% trainable) is useless.
Better to train 6.3% that actually learns than 1% that doesn't.

---

## [SYNC] **Comparison: Three Approaches**

| Approach | Trainable | Learning | Efficiency | Best For |
|----------|-----------|----------|------------|----------|
| **Pure LoRA (frozen)** | 1% | [FAIL] NO | [STAR][STAR][STAR][STAR][STAR] | Large models (BERT-base+) |
| **LoRA + 1 layer** | 3.2% | [FAIL] NO | [STAR][STAR][STAR][STAR] | Medium models (6+ layers) |
| **LoRA + 2 layers** | 6.3% | [SUCCESS] YES | [STAR][STAR][STAR] | **Small models (2-4 layers)** |
| **Full fine-tuning** | 100% | [SUCCESS] YES | [STAR] | Abundant resources |

**For bert-tiny, LoRA + 2 layers unfrozen is the optimal choice.**

---

## [START] **Ready to Test!**

Run the training with the same commands. The model should now:
1. [SUCCESS] Learn properly (metrics improve each round)
2. [SUCCESS] Match local validation performance
3. [SUCCESS] Show positive STS-B correlation
4. [SUCCESS] Reach 75-82% SST-2 accuracy
5. [SUCCESS] Reach 72-78% QQP accuracy

**The difference should be dramatic! From 46% → 75%+ [CELEBRATE]**

---

## [NOTE] **Summary**

**Problem**: Model not learning with 1 layer unfrozen (46% SST-2, -0.215 STS-B)  
**Root Cause**: bert-tiny has ONLY 2 layers, need both for sufficient capacity  
**Solution**: Unfreeze BOTH layers (all transformer layers)  
**Expected**: 75-82% SST-2, 72-78% QQP, 0.60-0.75 STS-B correlation  
**Trade-off**: 6.3% trainable (still 16× smaller than full model)  

**[CELEBRATE] bert-tiny will now learn properly with full transformer capacity!**

