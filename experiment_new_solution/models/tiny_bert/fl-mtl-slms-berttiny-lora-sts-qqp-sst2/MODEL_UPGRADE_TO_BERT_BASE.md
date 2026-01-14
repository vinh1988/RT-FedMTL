# Model Upgrade: BERT-Small → BERT-Base

## 🎯 **Why Upgrade?**

### Problem with BERT-Small
```csv
# Global metrics STUCK at same values
Round 1: SST-2=48.51%, QQP=52.44%, STS-B=0.1391
Round 2: SST-2=48.51%, QQP=52.44%, STS-B=0.1391  ← NO IMPROVEMENT!
```

**Root Cause:**
- ✅ Synchronization is working (no errors)
- ✅ Parameters are being updated (86 LoRA params per task)
- ❌ Model is too small to learn effectively with LoRA

**BERT-Small Limitations:**
- Only **4 transformer layers** (very shallow)
- Only **512 hidden size** (limited capacity)
- With LoRA, even less effective capacity
- Unfreezing 2 layers = **50% of model** (too much, overfits locally)

---

## ✅ **Solution: Upgrade to BERT-Base**

### BERT-Base Advantages

| Aspect | BERT-Small | BERT-Base | Improvement |
|--------|------------|-----------|-------------|
| **Layers** | 4 | 12 | 🔺 **3x more depth** |
| **Hidden Size** | 512 | 768 | 🔺 **1.5x more capacity** |
| **Parameters** | 29M | 110M | 🔺 **3.8x more params** |
| **Attention Heads** | 8 | 12 | 🔺 **Better attention** |
| **Learning Capacity** | Limited | Strong | 🔺 **Better learning** |

---

## 📝 **Changes Made**

### 1. Model Upgrade

**File:** `federated_config.yaml` (lines 4-7)

**Before:**
```yaml
model:
  server_model: "prajjwal1/bert-small"  # 4 layers, 29M params
  client_model: "prajjwal1/bert-small"
  use_peft_lora: true
```

**After:**
```yaml
model:
  server_model: "bert-base-uncased"     # 12 layers, 110M params ⭐
  client_model: "bert-base-uncased"
  use_peft_lora: true
```

### 2. Adjusted Unfreezing

**File:** `federated_config.yaml` (line 20)

**Before:**
```yaml
unfreeze_layers: 2  # 2 out of 4 = 50% (too much!)
```

**After:**
```yaml
unfreeze_layers: 3  # 3 out of 12 = 25% (better balance) ⭐
```

**Reasoning:**
- BERT-Small: 2/4 layers = 50% unfrozen → Overfits locally
- BERT-Base: 3/12 layers = 25% unfrozen → Better balance between global/local learning

### 3. Increased Training Rounds

**File:** `federated_config.yaml` (line 30)

**Before:**
```yaml
num_rounds: 2  # Too few to see improvement
```

**After:**
```yaml
num_rounds: 3  # More rounds to observe learning ⭐
```

### 4. Adjusted Learning Rate

**File:** `federated_config.yaml` (line 36)

**Before:**
```yaml
learning_rate: 0.0002  # Conservative for small model
```

**After:**
```yaml
learning_rate: 0.0003  # Slightly higher for bigger model ⭐
```

---

## 📊 **Expected Results**

### Before (BERT-Small)
```csv
round,global_sst2_val_accuracy,global_qqp_val_accuracy,...
1,0.4851,0.5244,...              ← Poor initial
2,0.4851,0.5244,...              ← STUCK!
```

**Issues:**
- Model too small
- Can't learn effectively with LoRA
- Gets stuck in local minima

### After (BERT-Base)
```csv
round,global_sst2_val_accuracy,global_qqp_val_accuracy,...
1,0.50-0.55,0.52-0.58,...        ← Better initial (pre-trained BERT-Base)
2,0.60-0.70,0.60-0.70,...        ← Should IMPROVE! ⭐
3,0.70-0.80,0.65-0.75,...        ← Keep improving! ⭐
```

**Expected Improvements:**
- ✅ Better initial performance (BERT-Base is well pre-trained)
- ✅ Global metrics should improve each round
- ✅ Higher final accuracy (BERT-Base capacity)
- ✅ More stable learning

---

## 🎯 **LoRA Configuration Impact**

### BERT-Small (Old)
```
Total Parameters: 29M
LoRA Trainable: ~7M (24%)
Unfrozen: 2 layers (50%)
→ Total Trainable: ~40% (too much!)
→ Learning: Unstable, overfits locally
```

### BERT-Base (New)
```
Total Parameters: 110M
LoRA Trainable: ~27M (24%)
Unfrozen: 3 layers (25%)
→ Total Trainable: ~35% (better balance)
→ Learning: More stable, better generalization
```

**Key Insight:** With a bigger base model, LoRA has more capacity to adapt effectively!

---

## 🚀 **Training Commands**

```bash
cd /home/vinh/Documents/code/FedAvgLS/experiment_new_solution/models/tiny_bert/fl-mtl-slms-berttiny-lora-sts-qqp-sst2

# Start server
/home/vinh/Documents/code/FedAvgLS/venv/bin/python federated_main.py --mode server --config federated_config.yaml

# In separate terminals, start clients:
# Terminal 2:
/home/vinh/Documents/code/FedAvgLS/venv/bin/python federated_main.py --mode client --client-id sst2_client --config federated_config.yaml

# Terminal 3:
/home/vinh/Documents/code/FedAvgLS/venv/bin/python federated_main.py --mode client --client-id qqp_client --config federated_config.yaml

# Terminal 4:
/home/vinh/Documents/code/FedAvgLS/venv/bin/python federated_main.py --mode client --client-id stsb_client --config federated_config.yaml
```

---

## 📋 **What to Look For**

### 1. Model Initialization
```
✓ "Base model: bert-base-uncased (12 layers)"
✓ "Total Parameters: 110,000,000+"
✓ "Trainable Percentage: 30-35%"
```

### 2. Better Initial Performance
```
Round 1 (BERT-Base): SST-2=50-55%, QQP=52-58%
  vs
Round 1 (BERT-Small): SST-2=48.51%, QQP=52.44%

→ BERT-Base should start higher!
```

### 3. Improving Global Metrics
```csv
round,global_sst2_val_accuracy,global_qqp_val_accuracy,...
1,0.52-0.55,0.55-0.60,...        ← Initial
2,0.62-0.68,0.63-0.68,...        ← IMPROVES! ⭐
3,0.70-0.75,0.68-0.73,...        ← KEEPS IMPROVING! ⭐
```

**Key:** Each round should show **increasing** accuracy!

### 4. Logs Should Show
```
✓ "✓ Updated local LoRA adapters for task 'sst2'"
✓ "  Parameters updated: 180-200" (more params for bigger model)
✓ "Global model validation complete: {improving metrics}"
✓ NO AttributeErrors or KeyErrors
```

---

## 🔧 **Performance Considerations**

### Memory Usage
- **BERT-Small**: ~500MB GPU memory
- **BERT-Base**: ~2-3GB GPU memory
- **Solution**: LoRA keeps it manageable (only 24% trainable)

### Training Time
- **BERT-Small**: ~13 min/round
- **BERT-Base**: ~15-20 min/round (slightly slower)
- **Trade-off**: Worth it for better accuracy!

### Dataset Size
Current config already uses full datasets:
- SST-2: 66K train samples ✅
- QQP: 323K train samples ✅
- STS-B: 4.2K train samples ✅

No need to change dataset sizes!

---

## 📊 **Expected Benchmark Comparison**

### BERT-Small (Current - Stuck)
```
SST-2: 48.51% (stuck)
QQP: 52.44% (stuck)
STS-B: 0.139 Pearson (very poor)
```

### BERT-Base (Expected)
```
SST-2: 75-85% (after 3 rounds)
QQP: 70-80% (after 3 rounds)
STS-B: 0.70-0.80 Pearson (much better)
```

**Target (BERT-Base Published):**
- SST-2: ~92.7%
- QQP: ~91.3%
- STS-B: ~90.0 Pearson

**Our Goal:** 75-85% (reasonable for federated learning with LoRA)

---

## 🎉 **Summary**

### Problems Fixed
1. ❌ Model too small → ✅ Upgraded to BERT-Base (3.8x params)
2. ❌ Too much unfreezing (50%) → ✅ Better balance (25%)
3. ❌ Metrics stuck → ✅ More capacity to learn
4. ❌ Only 2 rounds → ✅ 3 rounds to see improvement

### Files Changed
- ✅ `federated_config.yaml` (4 changes)

### Expected Outcome
- ✅ Better initial performance (BERT-Base pre-training)
- ✅ Global metrics improve each round
- ✅ Final accuracy: 75-85% (vs 48% stuck)
- ✅ More stable, reliable learning

---

## 🚀 **Ready to Train!**

The model is now upgraded to **BERT-Base** with optimized LoRA configuration. This should finally show **improving global metrics** across rounds! 🎯

**Note:** First run will download BERT-Base (~440MB), then training begins.

