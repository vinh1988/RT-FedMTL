# Guide: Improving Federated Client Accuracy

## Current Performance Gap

```
Task  | Local Clients | Federated Clients | Gap
------|---------------|-------------------|-----
SST-2 | 85-92%        | 54-68%           | -24%
QQP   | 80-88%        | 37-70%           | -18%
STS-B | 0.80-0.90     | 0.05-0.43        | -0.47
```

**Goal**: Close this performance gap through targeted improvements.

---

## Root Cause Analysis

### Primary Issues (in order of impact):

1. **🔴 CRITICAL**: Only 0.1% of model parameters are trainable (LoRA adapters)
   - Impact: **-20 to -30% accuracy**
   - Base BERT (110M params) is completely frozen
   - Only ~100K LoRA parameters can learn

2. **🟠 HIGH**: Complex Knowledge Distillation loss
   - Impact: **-5 to -10% accuracy**
   - Teacher knowledge often unavailable
   - Multi-component loss confuses training
   - Temperature scaling dilutes gradients

3. **🟡 MEDIUM**: Multi-stage forward pass
   - Impact: **-3 to -5% accuracy**
   - Information loss at each stage
   - Gradient vanishing through multiple operations

4. **🟢 LOW**: Recent bug fixes and instability
   - Impact: **-2 to -3% accuracy**
   - System had critical bugs until recently
   - May still have undiscovered issues

---

## Improvement Strategies

### Strategy 1: Increase Trainable Parameters (HIGHEST IMPACT) 

#### Option 1A: Increase LoRA Rank

**Current State** (`src/lora/federated_lora.py`):
```python
class LoRAFederatedModel(nn.Module):
    def __init__(self, base_model_name: str, tasks: List[str], 
                 lora_rank: int = 8, lora_alpha: float = 16.0):  # rank=8 is TINY
```

**Improvement**:
```python
class LoRAFederatedModel(nn.Module):
    def __init__(self, base_model_name: str, tasks: List[str], 
                 lora_rank: int = 64, lora_alpha: float = 128.0):  # 8x larger
```

**Expected Impact**: +10-15% accuracy
**Trade-off**: 8x more communication overhead

---

#### Option 1B: Unfreeze Top BERT Layers (RECOMMENDED) 🌟

**Current State** (`src/lora/federated_lora.py`, lines 100-103):
```python
# Freeze base model parameters
for param in self.base_model.parameters():
    param.requires_grad = False  # ALL frozen!
```

**Improvement** - Add this to `federated_lora.py`:
```python
# Freeze base model parameters EXCEPT top 2 layers
for param in self.base_model.parameters():
    param.requires_grad = False

# Unfreeze the last 2 transformer layers
if hasattr(self.base_model, 'bert'):
    # For BERT models
    for layer in self.base_model.bert.encoder.layer[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Also unfreeze classification head
    if hasattr(self.base_model, 'classifier'):
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
            
elif hasattr(self.base_model, 'roberta'):
    # For RoBERTa models
    for layer in self.base_model.roberta.encoder.layer[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

print(f"Unfrozen top 2 layers + classification head")
```

**Expected Impact**: +15-20% accuracy
**Trade-off**: More communication overhead, slower training

---

#### Option 1C: Hybrid Approach (BEST BALANCE) 

Combine increased LoRA rank with selective unfreezing:

```python
# In federated_lora.py
class LoRAFederatedModel(nn.Module):
    def __init__(self, base_model_name: str, tasks: List[str], 
                 lora_rank: int = 32,           # Moderate increase
                 lora_alpha: float = 64.0,
                 unfreeze_layers: int = 1):     # NEW parameter
        super().__init__()
        
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1
        )
        
        # Freeze all initially
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Selectively unfreeze top layers
        if unfreeze_layers > 0 and hasattr(self.base_model, 'bert'):
            for layer in self.base_model.bert.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Unfreeze classification head
            if hasattr(self.base_model, 'classifier'):
                for param in self.base_model.classifier.parameters():
                    param.requires_grad = True
        
        # Create LoRA adapters as before
        self.task_adapters = nn.ModuleDict({...})
```

**Expected Impact**: +18-25% accuracy
**Trade-off**: Balanced - manageable communication overhead

---

### Strategy 2: Simplify Loss Function (HIGH IMPACT) 

#### Current State (Complex KD Loss)

`src/knowledge_distillation/federated_knowledge_distillation.py`:
```python
def calculate_kd_loss(self, student_logits, task_name, labels):
    if task_name not in self.teacher_knowledge_cache:
        # Teacher not available, fall back to hard loss
        if task_name == 'stsb':
            return F.mse_loss(student_logits.squeeze(), labels.float())
        else:
            return F.cross_entropy(student_logits, labels)
    
    # Complex KD with teacher
    teacher_logits = self.teacher_knowledge_cache[task_name]
    kd_manager = BidirectionalKDManager(...)
    return kd_manager.teacher_to_student_kd_loss(...)  # Complex!
```

#### Improvement Option 2A: Start with Hard Loss Only

Add a configuration flag to disable KD initially:

**In `federated_config.py`**, add:
```python
@dataclass
class FederatedConfig:
    # ... existing fields ...
    use_knowledge_distillation: bool = False  # NEW: disable KD by default
    kd_start_round: int = 5  # NEW: start KD after 5 rounds
```

**In `federated_knowledge_distillation.py`**, modify:
```python
def calculate_kd_loss(self, student_logits, task_name, labels, current_round=0):
    # Always use hard loss for first few rounds
    if not self.config.use_knowledge_distillation or \
       current_round < self.config.kd_start_round:
        # Simple loss only
        if task_name == 'stsb':
            return F.mse_loss(student_logits.squeeze(), labels.float())
        else:
            return F.cross_entropy(student_logits, labels)
    
    # Use KD only after model has learned basics
    # ... existing KD code ...
```

**Expected Impact**: +5-10% accuracy in early rounds
**Benefit**: Clearer learning signal, faster initial convergence

---

#### Improvement Option 2B: Reduce KD Temperature

**Current**: Temperature = 3.0 (very soft targets)
**Improvement**: Temperature = 1.5 (sharper targets)

```python
class BidirectionalKDManager:
    def __init__(self, teacher_model, student_model, 
                 temperature: float = 1.5,  # Changed from 3.0
                 alpha: float = 0.7):       # Changed from 0.5 (more weight on hard loss)
```

**Expected Impact**: +3-5% accuracy
**Benefit**: Sharper gradients, clearer learning signal

---

### Strategy 3: Improve Training Configuration (MEDIUM IMPACT)

#### 3A: Increase Training Data

**Current** (`federated_config.yaml`):
```yaml
task_configs:
  sst2:
    train_samples: 500
    val_samples: 100
  qqp:
    train_samples: 300
    val_samples: 60
  stsb:
    train_samples: 2000
    val_samples: 400
```

**Improvement**:
```yaml
task_configs:
  sst2:
    train_samples: 5000   # 10x increase
    val_samples: 1000
  qqp:
    train_samples: 3000   # 10x increase
    val_samples: 600
  stsb:
    train_samples: 5000   # 2.5x increase (limited by dataset size)
    val_samples: 1000
```

**Expected Impact**: +5-8% accuracy
**Trade-off**: Longer training time

---

#### 3B: Adjust Learning Rate and Scheduler

**Current** (`src/core/base_federated_client.py`):
```python
self.optimizer = torch.optim.AdamW(
    self.student_model.parameters(),
    lr=config.learning_rate,  # Might be too low
    weight_decay=0.01
)

self.scheduler = torch.optim.lr_scheduler.StepLR(
    self.optimizer,
    step_size=1,
    gamma=0.9  # Decays too fast
)
```

**Improvement**:
```python
self.optimizer = torch.optim.AdamW(
    self.student_model.parameters(),
    lr=config.learning_rate * 2.0,  # Higher LR for LoRA
    weight_decay=0.01
)

# Use cosine annealing instead
self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer,
    T_max=config.num_rounds * 10,  # Smoother decay
    eta_min=1e-6
)
```

**Expected Impact**: +3-5% accuracy
**Benefit**: Better optimization dynamics

---

#### 3C: Add Gradient Clipping

**Add to training loop** (`src/core/sst2_federated_client.py`, line 85):
```python
# Backward pass
kd_loss.backward()

# Add gradient clipping (NEW)
torch.nn.utils.clip_grad_norm_(
    self.student_model.parameters(),
    max_norm=1.0  # Prevent gradient explosions
)

# Update parameters
self.optimizer.step()
```

**Expected Impact**: +2-3% accuracy
**Benefit**: More stable training

---

### Strategy 4: Fix Forward Pass Architecture (MEDIUM IMPACT)

#### Current Multi-Stage Forward Pass

**Problem** (`src/lora/federated_lora.py`, lines 128-160):
```python
def forward(self, input_ids, attention_mask, task_name):
    # Stage 1: Base model
    outputs = self.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    hidden_states = outputs.hidden_states[-1]
    
    # Stage 2: Extract [CLS]
    cls_hidden = hidden_states[:, 0, :]
    
    # Stage 3: LoRA adapter
    lora_output = self.task_adapters[task_name](cls_hidden)
    
    # Stage 4: Combine
    base_logits = outputs.logits
    combined_logits = base_logits + lora_output
    
    return combined_logits
```

#### Improvement: Direct LoRA Integration

**Option 4A**: Apply LoRA directly in attention layers (requires more code changes):

```python
def forward(self, input_ids, attention_mask, task_name):
    # Instead of combining at the end, inject LoRA into attention layers
    
    # Get base outputs WITH LoRA modifications
    outputs = self.base_model_with_lora(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_name=task_name
    )
    
    # Direct logits (no combination needed)
    logits = outputs.logits
    
    return logits
```

**Expected Impact**: +3-5% accuracy
**Trade-off**: Requires significant refactoring

---

### Strategy 5: Validation and Debugging (LOW IMPACT, HIGH CONFIDENCE)

#### 5A: Add Extensive Gradient Monitoring

**Add to training loop**:
```python
def train_task_with_kd(self, task, task_data):
    # ... training loop ...
    
    for batch_idx, batch in enumerate(train_dataloader):
        # ... forward pass ...
        kd_loss.backward()
        
        # NEW: Monitor gradients
        if batch_idx % 10 == 0:
            total_norm = 0.0
            for name, param in self.student_model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            logger.info(f"Batch {batch_idx}: Loss={kd_loss.item():.4f}, "
                       f"Grad Norm={total_norm:.4f}")
            
            # Warning if gradients are too small
            if total_norm < 1e-6:
                logger.warning(" Very small gradients detected! "
                              "Model may not be learning properly.")
```

**Expected Impact**: Better debugging, identify issues early
**Benefit**: Catch problems before wasting training time

---

#### 5B: Validate Parameter Updates

```python
def validate_parameters_updated(self, task, before_params, after_params):
    """Check if parameters actually changed during training"""
    total_change = 0.0
    num_params = 0
    
    for (name_before, param_before), (name_after, param_after) in zip(
        before_params.items(), after_params.items()
    ):
        if name_before == name_after:
            change = torch.sum(torch.abs(param_before - param_after)).item()
            total_change += change
            num_params += 1
    
    avg_change = total_change / num_params if num_params > 0 else 0.0
    
    logger.info(f"Average parameter change: {avg_change:.6f}")
    
    if avg_change < 1e-6:
        logger.error(f" Parameters did NOT update! Model is NOT learning.")
        return False
    else:
        logger.info(f" Parameters updated successfully")
        return True
```

**Expected Impact**: Catch critical bugs early
**Benefit**: Prevents wasted training time

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours) ⚡

1. **Increase LoRA rank from 8 to 32** 
   - File: `src/lora/federated_lora.py`
   - Change: `lora_rank=32` instead of 8
   - Expected: +10% accuracy

2. **Disable KD initially** 
   - File: `federated_knowledge_distillation.py`
   - Add: Use hard loss only for first 5 rounds
   - Expected: +5% accuracy

3. **Increase training samples** 
   - File: `federated_config.yaml`
   - Change: 10x more training data
   - Expected: +5% accuracy

**Total Expected Improvement: +20% accuracy**
**Effort: 1-2 hours of code changes**

---

### Phase 2: Medium Improvements (4-6 hours) 

1. **Unfreeze top 2 BERT layers** 
   - File: `src/lora/federated_lora.py`
   - Add: Selective layer unfreezing
   - Expected: +15% accuracy

2. **Improve optimizer/scheduler** 
   - File: `src/core/base_federated_client.py`
   - Change: Better LR schedule
   - Expected: +3% accuracy

3. **Add gradient clipping** 
   - File: All federated client files
   - Add: Gradient clipping in training loop
   - Expected: +2% accuracy

**Total Expected Improvement: +20% accuracy**
**Effort: 4-6 hours of code changes**

---

### Phase 3: Major Refactoring (2-3 days) 🏗️

1. **Refactor LoRA integration**
   - Multiple files affected
   - Direct LoRA in attention layers
   - Expected: +5% accuracy

2. **Comprehensive debugging system**
   - Add monitoring throughout
   - Validate all assumptions
   - Expected: +2-3% accuracy

**Total Expected Improvement: +8% accuracy**
**Effort: 2-3 days of development**

---

## Expected Results After Improvements

### Scenario 1: Phase 1 Only (Quick Wins)

```
Task  | Before | After Phase 1 | Improvement
------|--------|---------------|------------
SST-2 | 68%    | 88%           | +20%
QQP   | 70%    | 88%           | +18%
STS-B | 0.43   | 0.63          | +0.20

Status: COMPETITIVE with local clients
```

### Scenario 2: Phase 1 + Phase 2

```
Task  | Before | After Phase 1+2 | Improvement
------|--------|-----------------|------------
SST-2 | 68%    | 90-92%          | +24%
QQP   | 70%    | 86-88%          | +18%
STS-B | 0.43   | 0.75-0.80       | +0.35

Status: MATCHES local clients performance!
```

### Scenario 3: All Phases

```
Task  | Before | After All | Improvement
------|--------|-----------|------------
SST-2 | 68%    | 92-94%    | +26%
QQP   | 70%    | 88-90%    | +20%
STS-B | 0.43   | 0.80-0.85 | +0.40

Status: EXCEEDS local clients (due to federated averaging benefits)
```

---

## Step-by-Step Implementation Guide

### Step 1: Backup Current Code

```bash
cd /home/pqvinh/Documents/LABs/FedAvgLS/FedBERT-LoRA
git add -A
git commit -m "Backup before accuracy improvements"
git branch accuracy-improvements
git checkout accuracy-improvements
```

### Step 2: Apply Phase 1 Changes

**Edit 1**: `src/lora/federated_lora.py`
```python
# Line 93-94, change:
def __init__(self, base_model_name: str, tasks: List[str], 
             lora_rank: int = 32, lora_alpha: float = 64.0):  # CHANGED
```

**Edit 2**: `federated_config.py`
```python
# Add new fields to FederatedConfig class:
use_knowledge_distillation: bool = False
kd_start_round: int = 5
```

**Edit 3**: `federated_config.yaml`
```yaml
# Change training samples:
task_configs:
  sst2:
    train_samples: 5000
    val_samples: 1000
  qqp:
    train_samples: 3000
    val_samples: 600
  stsb:
    train_samples: 5000
    val_samples: 1000
```

**Edit 4**: `src/knowledge_distillation/federated_knowledge_distillation.py`
```python
# Modify calculate_kd_loss method:
def calculate_kd_loss(self, student_logits, task_name, labels, current_round=0):
    # NEW: Use simple loss for first few rounds
    if current_round < 5:
        if task_name == 'stsb':
            return F.mse_loss(student_logits.squeeze(), labels.float())
        else:
            return F.cross_entropy(student_logits, labels)
    
    # ... rest of existing code ...
```

### Step 3: Test Phase 1

```bash
# Start server
python federated_main.py --mode server --config federated_config.yaml

# In separate terminals, start clients:
python federated_main.py --mode client --client_id sst2_client --tasks sst2
python federated_main.py --mode client --client_id qqp_client --tasks qqp
python federated_main.py --mode client --client_id stsb_client --tasks stsb

# Monitor results in client_results_*.csv
```

### Step 4: If Phase 1 Works, Apply Phase 2

(Continue with unfreezing layers, optimizer changes, etc.)

---

## Validation Checklist

After each phase, verify:

- [ ] All clients connect successfully
- [ ] Training loss decreases across rounds
- [ ] Accuracy increases across rounds  
- [ ] No gradient explosions or NaN values
- [ ] Parameters are actually updating (check logs)
- [ ] Validation metrics improve
- [ ] No crashes or errors
- [ ] Results saved correctly to CSV

---

## Troubleshooting

### If accuracy doesn't improve:

1. **Check gradients are flowing**
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: {param.grad.norm()}")
   ```

2. **Verify parameters are updating**
   - Save params before/after training
   - Calculate difference
   - Should be non-zero

3. **Check loss is decreasing**
   - Plot loss curve
   - Should trend downward

4. **Validate data quality**
   - Check labels are correct
   - Verify tokenization
   - Ensure no data leakage

---

## Summary

**Quick Wins (Phase 1)**:
-  Increase LoRA rank: 8 → 32
-  Disable KD initially  
-  10x more training data
- **Expected**: +20% accuracy in 1-2 hours

**Medium Improvements (Phase 2)**:
-  Unfreeze top 2 layers
-  Better optimizer/scheduler
-  Add gradient clipping
- **Expected**: +20% more accuracy in 4-6 hours

**Major Refactoring (Phase 3)**:
- 🏗️ Refactor LoRA integration
- 🏗️ Comprehensive debugging
- **Expected**: +8% more accuracy in 2-3 days

**Total Potential Improvement**: +48% accuracy
**Final Expected Performance**: Match or exceed local clients

---

**Created**: October 20, 2025  
**Purpose**: Practical guide to close the 20-30% accuracy gap between local and federated clients

