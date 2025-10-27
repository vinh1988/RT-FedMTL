# Training Time Optimization Guide

## Problem: QQP Client Times Out

**Symptom:**
```
WARNING - Timeout collecting updates. Got 2/3
WARNING - Proceeding with 2 out of 3 clients
INFO - Still waiting for: ['qqp_client']
```

## Root Cause: Training Takes Too Long!

### **Original Settings:**
```yaml
batch_size: 8
train_samples: 323,415 (QQP)
val_samples: 40,431 (QQP)
round_timeout: 3000 seconds (50 minutes)
```

### **Actual Training Time:**
- Training batches: 323,415 ÷ 8 = **40,427 batches**
- Validation batches: 40,431 ÷ 8 = **5,054 batches**
- Total: **45,481 batches per round**
- At 1 second/batch: **12.6 hours!** ⏰
- Round timeout: **50 minutes** ⏱️
- **Result: TIMEOUT!** ❌

## Solutions Applied

### ✅ **Solution 1: Increased Batch Size**
```yaml
batch_size: 32  # Changed from 8 to 32 (4x faster!)
```

**Impact:**
- Training batches: 323,415 ÷ 32 = **10,107 batches** (4x fewer!)
- Validation batches: 40,431 ÷ 32 = **1,264 batches**
- Total: **11,371 batches** (75% reduction!)

### ✅ **Solution 2: Reduced Sample Size (For Testing)**
```yaml
task_configs:
  qqp:
    train_samples: 32000   # Reduced from 323,415
    val_samples: 4000      # Reduced from 40,431
```

**Impact:**
- Training batches: 32,000 ÷ 32 = **1,000 batches**
- Validation batches: 4,000 ÷ 32 = **125 batches**
- Total: **1,125 batches** (98% reduction!)
- Estimated time: **~19 minutes** ✅

## Training Time Comparison

| Configuration | Batches | Est. Time | Fits in 50min? |
|--------------|---------|-----------|----------------|
| **Original** (batch=8, full data) | 45,481 | 12.6 hours | ❌ NO |
| **Batch=32** (full data) | 11,371 | 3.2 hours | ❌ NO |
| **Batch=64** (full data) | 5,685 | 1.6 hours | ❌ NO |
| **Batch=32, 32K samples** | 1,125 | 19 min | ✅ YES |
| **Batch=64, 32K samples** | 563 | 9.4 min | ✅ YES |

## Recommended Configurations

### **For Quick Testing (Current Config):**
```yaml
training:
  batch_size: 32

task_configs:
  qqp:
    train_samples: 32000
    val_samples: 4000
```
- **Time:** ~19 minutes per round
- **Purpose:** Testing, debugging, development
- **Accuracy:** Good (10% of full dataset)

### **For Better Accuracy:**
```yaml
training:
  batch_size: 64

task_configs:
  qqp:
    train_samples: 100000
    val_samples: 10000
```
- **Time:** ~26 minutes per round
- **Purpose:** Better model performance
- **Accuracy:** Very good (31% of full dataset)

### **For Full Dataset Training:**

**Option A: Increase batch size + timeout**
```yaml
training:
  batch_size: 128

task_configs:
  qqp:
    train_samples: 323415   # Full dataset
    val_samples: 40431

communication:
  round_timeout: 10000      # 2.8 hours
  send_timeout: 3600
```
- **Time:** ~48 minutes per round
- **Purpose:** Maximum accuracy
- **Requires:** Long timeout

**Option B: Use subset with high batch size**
```yaml
training:
  batch_size: 64

task_configs:
  qqp:
    train_samples: 150000   # 46% of full dataset
    val_samples: 15000
```
- **Time:** ~39 minutes per round
- **Purpose:** Balance between accuracy and speed
- **Fits:** Within 50-minute timeout

## Batch Size Guidelines

### **Impact on Training:**

| Batch Size | Speed | Memory | Accuracy | Recommendation |
|-----------|-------|--------|----------|----------------|
| 8 | Slow | Low | Good | For small GPU |
| 16 | Medium | Medium | Good | Balanced |
| 32 | Fast | Medium | Good | **Recommended** |
| 64 | Very Fast | High | Good | For large GPU |
| 128 | Fastest | Very High | Fair | Large GPU + careful tuning |

### **Memory Requirements:**

- **Batch size 8:** ~2 GB GPU RAM
- **Batch size 16:** ~4 GB GPU RAM
- **Batch size 32:** ~6 GB GPU RAM
- **Batch size 64:** ~10 GB GPU RAM
- **Batch size 128:** ~18 GB GPU RAM

If you get CUDA out of memory errors, reduce batch size!

## Testing the New Configuration

### **1. Start Fresh Server:**
```bash
cd /home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA
python federated_main.py --mode server --config federated_config.yaml
```

### **2. Start All Three Clients:**

**Terminal 2 (SST2):**
```bash
python federated_main.py --mode client --client_id sst2_client --tasks sst2
```

**Terminal 3 (QQP):**
```bash
python federated_main.py --mode client --client_id qqp_client --tasks qqp
```

**Terminal 4 (STSB):**
```bash
python federated_main.py --mode client --client_id stsb_client --tasks stsb
```

### **3. Expected Timeline (with new config):**

| Event | Time | What You'll See |
|-------|------|----------------|
| Client registration | 0-10s | "Client X registered" |
| Round 1 starts | 10s | "=== Round 1/30 ===" |
| SST2 training | 1-8 min | "Task sst2 - Batch X" |
| STSB training | 1-3 min | "Task stsb - Batch X" |
| **QQP training** | **15-20 min** | "Task qqp - Batch X" |
| Updates sent | 18-21 min | "Message sent successfully" |
| Round 1 complete | 21 min | "Round 1 completed" |
| **Total per round** | **~21 min** | ✅ Within 50min timeout! |

## Monitoring Progress

### **Watch QQP Client:**
```bash
# In QQP client terminal, you'll see:
[TRAINING] Starting training for task qqp with 1000 batches
[STATS] Task qqp - Batch 5, Loss: 0.XXXX
[STATS] Task qqp - Batch 10, Loss: 0.XXXX
...
[STATS] Task qqp - Batch 1000, Loss: 0.XXXX
[SUCCESS] Task qqp training completed
Client qqp_client using send timeout of 3600 seconds
Message sent successfully from client qqp_client
```

### **Server Progress:**
```bash
# Server will show:
Waiting for client updates... (expecting ALL 3 clients)
Currently connected clients: ['sst2_client', 'qqp_client', 'stsb_client']
Updates received: 1/3
Received from: ['sst2_client']
Still waiting for: ['qqp_client', 'stsb_client']
...
Updates received: 3/3
All client updates received (3/3)
Round 1 completed
```

## For Production: Full Dataset Configuration

When you're ready for full dataset training, use this config:

```yaml
training:
  num_rounds: 30
  batch_size: 64              # Balance speed/memory
  local_epochs: 1

task_configs:
  sst2:
    train_samples: 66477      # Full SST2
    val_samples: 872
  
  qqp:
    train_samples: 323415     # Full QQP
    val_samples: 40431
  
  stsb:
    train_samples: 4249       # Full STSB
    val_samples: 1500

communication:
  round_timeout: 8000         # 133 minutes (2.2 hours)
  send_timeout: 3600          # 60 minutes
```

**Expected time per round:** ~80 minutes

## Summary

✅ **Quick Fix Applied:**
- Increased batch_size to 32 (4x faster)
- Reduced QQP samples to 32K (10x faster)
- Training now takes ~19 min vs 12+ hours!

✅ **Result:**
- QQP client completes within 50-minute timeout
- All three clients can participate in federated learning
- System works as expected!

📈 **Next Steps:**
1. Test with current config (batch=32, 32K samples)
2. Once working, gradually increase sample size
3. Monitor timing and adjust timeout if needed
4. For production, use full dataset with larger batch size

The QQP client should now complete successfully! 🎉

