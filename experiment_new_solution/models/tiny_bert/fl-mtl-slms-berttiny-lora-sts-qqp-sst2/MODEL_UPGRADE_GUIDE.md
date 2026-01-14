# Model Upgrade Guide: TinyBERT → BERT-Small

## 🎯 **Why Upgrade?**

**TinyBERT (2 layers)** is too restrictive:
- Unfreezing 2 layers = **100% encoder trainable** ❌
- No frozen layers for parameter efficiency
- Limited model capacity → poor learning

**BERT-Small (4 layers)** provides better balance:
- Unfreezing 2 layers = **50% encoder frozen** ✅
- Better parameter efficiency for federated learning
- 7x more parameters → better learning capacity

---

## 📊 **Model Comparison**

| Metric | TinyBERT (Old) | BERT-Small (New) | Improvement |
|--------|----------------|------------------|-------------|
| **Layers** | 2 | 4 | +100% |
| **Hidden Size** | 128 | 512 | +300% |
| **Total Params** | 4.4M | 29M | +559% |
| **Unfrozen Layers** | 2 (100%) | 2 (50%) | Better efficiency |
| **Trainable %** | ~6.3% | ~12-15% | +2x capacity |

---

## 🚀 **What Changed**

### Configuration Update
```yaml
model:
  server_model: "prajjwal1/bert-small"  # Was: prajjwal1/bert-tiny
  client_model: "prajjwal1/bert-small"  # Was: prajjwal1/bert-tiny
```

### LoRA Configuration (Unchanged)
```yaml
peft_lora:
  rank: 16              # LoRA rank
  alpha: 32             # Scaling factor
  dropout: 0.1          # Dropout
  target_modules:       # Apply LoRA to:
    - "query"           #   Query attention
    - "key"             #   Key attention
    - "value"           #   Value attention
    - "dense"           #   FFN dense layers
  unfreeze_layers: 2    # Unfreeze last 2 layers
```

---

## 📈 **Expected Results**

### TinyBERT Results (2 layers)
```
SST-2: 56.08% (target: 75%+)  ❌ Below target
QQP:   48.86% (target: 70%+)  ❌ Below target
STS-B: 0.090  (target: 0.4+)  ❌ Below target
```

### BERT-Small Expected (4 layers)
```
SST-2: 75-80%  ✅ Should reach target
QQP:   70-75%  ✅ Should reach target
STS-B: 0.4-0.6 ✅ Should reach target
```

**Why Better?**
- ✅ **More capacity**: 29M vs 4.4M parameters
- ✅ **Better representations**: 512 vs 128 hidden size
- ✅ **Proper layer freezing**: 50% frozen vs 0% frozen
- ✅ **More LoRA adapters**: Applied to 4 layers instead of 2

---

## ⚙️ **Training Considerations**

### Memory Usage
- **TinyBERT**: ~1-2GB GPU memory
- **BERT-Small**: ~3-4GB GPU memory
- **Increase**: +2GB (still reasonable)

### Training Speed
- **TinyBERT**: ~4.5 minutes/round
- **BERT-Small**: ~6-8 minutes/round
- **Increase**: +30-50% time (acceptable)

### Communication Cost
- **TinyBERT LoRA**: ~1-2MB per update
- **BERT-Small LoRA**: ~3-5MB per update
- **Increase**: +2-3MB (minimal)

---

## 🎯 **Training Checklist**

When you run training, verify:

### 1. Model Initialization
```
✓ Loading BERT-Small (4 layers)
✓ Hidden size: 512
✓ Total parameters: ~29M
```

### 2. LoRA Application
```
✓ Applying LoRA to: query, key, value, dense
✓ LoRA rank: 16, alpha: 32
✓ Target modules found in all 4 layers
```

### 3. Layer Unfreezing
```
✓ Unfreezing layer 2 (3rd layer)
✓ Unfreezing layer 3 (4th layer)
✓ Unfreezing pooler layer
✓ Trainable parameters: ~12-15%
```

### 4. Validation Metrics (After Round 1)
```
✓ SST-2: Should be 60%+ (not 56%)
✓ QQP: Should be 55%+ (not 48%)
✓ STS-B: Should be 0.15+ (not 0.09)
```

### 5. Learning Progress (After Round 2+)
```
✓ Metrics IMPROVE each round (not stay constant!)
✓ Global validation metrics UPDATE (not stuck at Round 0)
✓ Local validation > Global validation (expected gap)
```

---

## 🔧 **Alternative Models**

If BERT-Small doesn't fit your needs:

### Option 1: DistilBERT (6 layers)
- **Pros**: Even better capacity, well-known model
- **Cons**: 2.3x larger than BERT-Small, slower training
- **Config**: `server_model: "distilbert-base-uncased"`

### Option 2: BERT-Base (12 layers)
- **Pros**: Full BERT, best performance
- **Cons**: 3.8x larger than BERT-Small, much slower
- **Config**: `server_model: "bert-base-uncased"`

### Option 3: Stay with TinyBERT (2 layers)
- **Pros**: Fastest, smallest
- **Cons**: Poor learning (as observed)
- **Fix**: Increase LoRA rank to 32, unfreeze all layers
- **Config**: Keep current + update `rank: 32`

---

## 📝 **Summary**

| Aspect | Change |
|--------|--------|
| **Model** | TinyBERT → BERT-Small |
| **Layers** | 2 → 4 |
| **Params** | 4.4M → 29M |
| **Training Time** | +30-50% |
| **Memory** | +2GB |
| **Expected Accuracy** | +15-20% improvement |

---

## 🚀 **Next Steps**

1. ✅ **Config updated** to BERT-Small
2. 🔄 **Run training**:
   ```bash
   cd /home/vinh/Documents/code/FedAvgLS/experiment_new_solution/models/tiny_bert/fl-mtl-slms-berttiny-lora-sts-qqp-sst2
   /home/vinh/Documents/code/FedAvgLS/venv/bin/python federated_main.py --mode server --config federated_config.yaml 2>&1 &
   ```
3. 📊 **Monitor results** in `federated_results/`
4. 🎯 **Verify improvements** (SST-2 should reach 75%+)

---

**Expected Outcome**: With BERT-Small's 4 layers and proper LoRA configuration, the model should learn much more effectively than TinyBERT!

