# LoRA Layer Unfreezing Bug Fix

## 🐛 **The Bug**

### Symptom
```
Expected trainable parameters: 12-15% (with 2 layers unfrozen)
Actual trainable parameters:   2.06% (only LoRA adapters)
```

**Result**: Model couldn't learn effectively because only LoRA adapters were trainable, not the unfrozen layers!

---

## 🔍 **Root Cause**

In `src/models/peft_lora_model.py`, the unfreezing logic was applied to `self.bert`, but then **new models were created for each task**:

```python
# Lines 57-81: Unfreeze layers on self.bert
for param in self.bert.parameters():
    param.requires_grad = False

# ... unfreezing logic on self.bert ...

# Line 100: Create NEW model for each task ❌
for task in tasks:
    task_model = AutoModel.from_pretrained(base_model_name)  # ← BUG!
    task_peft_model = get_peft_model(task_model, peft_config)
    self.task_adapters[task] = task_peft_model
```

**Problem**: 
- Unfreezing happened on `self.bert` (which was never used)
- Task models were created fresh from `AutoModel.from_pretrained()` with all layers frozen
- The unfreezing never carried over to the task models!

---

## ✅ **The Fix**

### 1. Remove Unused `self.bert`
```python
# OLD: Created self.bert and unfroze its layers (unused!)
self.bert = AutoModel.from_pretrained(base_model_name)
# ... unfreezing logic ...

# NEW: Just get config, create task models later
config = AutoConfig.from_pretrained(base_model_name)
self.hidden_size = config.hidden_size
```

### 2. Unfreeze Layers in Each Task Model
```python
for task in tasks:
    # Create task model with LoRA
    task_model = AutoModel.from_pretrained(base_model_name)
    task_peft_model = get_peft_model(task_model, peft_config)
    
    # CRITICAL FIX: Unfreeze layers in THIS task model
    if hasattr(task_peft_model.base_model.model, 'encoder'):
        base_bert = task_peft_model.base_model.model  # Access BERT inside PEFT wrapper
        total_layers = len(base_bert.encoder.layer)
        start_unfreeze = max(0, total_layers - self.unfreeze_layers)
        
        # Unfreeze last N layers
        for layer_idx in range(start_unfreeze, total_layers):
            for param in base_bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = True
        
        # Unfreeze pooler
        if hasattr(base_bert, 'pooler') and base_bert.pooler is not None:
            for param in base_bert.pooler.parameters():
                param.requires_grad = True
    
    self.task_adapters[task] = task_peft_model
```

**Key Points**:
- Access the actual BERT model inside the PEFT wrapper: `task_peft_model.base_model.model`
- Unfreeze layers AFTER applying PEFT/LoRA
- Do this for EACH task model independently

---

## 📊 **Expected Results**

### Before Fix (BERT-Small)
```
Total Parameters:      88,109,568
Trainable Parameters:   1,818,624
Trainable Percentage:   2.06%  ❌ Too low!

Round 1: SST-2=54.36%, QQP=62.70%, STS-B=-0.156
Round 2: SST-2=54.36%, QQP=62.70%, STS-B=-0.156  ← STUCK!
```

### After Fix (BERT-Small, 4 layers, unfreeze 2)
```
Total Parameters:      88,109,568
Trainable Parameters:  ~11-13M
Trainable Percentage:  12-15%  ✅ Much better!

Expected improvement:
- SST-2: 75-80%
- QQP:   70-75%
- STS-B: 0.4-0.6
- Metrics should IMPROVE each round!
```

---

## 🎯 **Verification Checklist**

When training runs, look for these in the logs:

### 1. Model Initialization
```
✓ Initializing PEFT LoRA MTL Model:
✓   Base model: prajjwal1/bert-small (4 layers)
✓   Will unfreeze last 2 layers in each task model
```

### 2. Task Model Creation
```
✓ Task 'sst2': Unfreezing last 2 layers (layers 2-3)
✓ Task 'qqp': Unfreezing last 2 layers (layers 2-3)
✓ Task 'stsb': Unfreezing last 2 layers (layers 2-3)
```

### 3. Trainable Parameters
```
✓ Trainable Percentage: 12-15%  ← Should be 12-15%, NOT 2%!
```

### 4. Training Progress
```
✓ Validation metrics IMPROVE each round
✓ Global metrics UPDATE (not stuck at same values)
✓ SST-2 reaches 75%+
✓ STS-B becomes positive (0.4+)
```

---

## 📝 **Files Changed**

### `src/models/peft_lora_model.py`
1. **Removed unused `self.bert`** (lines 53-81)
   - Replaced with just loading config
   
2. **Added layer unfreezing in task loop** (lines ~100-120)
   - Unfreeze layers in each task model after PEFT application
   
3. **Updated imports** (line 9)
   - Added `AutoConfig` import

---

## 🚀 **Summary**

| Aspect | Before | After |
|--------|--------|-------|
| **Bug Location** | Lines 53-81 (unused) + Line 100 | Lines 100-120 (fixed) |
| **Trainable %** | 2.06% ❌ | 12-15% ✅ |
| **Learning** | Stuck / Poor | Should improve ✓ |
| **Code Clarity** | Confusing (unused self.bert) | Clear (unfreeze per task) |

**Impact**: Model can now properly train with unfrozen layers, not just LoRA adapters!

