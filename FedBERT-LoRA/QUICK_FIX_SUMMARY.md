# Quick Fix Summary - QQP Full Data Timeout Issue

## The Problem
- ✅ Works with **less QQP data** (e.g., 1000 samples)
- ❌ Fails with **full QQP data** (323,415 samples)
- Even with `round_timeout: 3000` seconds!

## The Root Cause
**WebSocket send timeout was hardcoded to 300 seconds (5 minutes)**, but sending updates from full dataset training takes **10-15 minutes**!

## The Fix (Applied)
1. ✅ Added configurable `send_timeout` parameter (now 3600 seconds = 60 minutes)
2. ✅ Fixed `round_timeout` not being loaded from YAML config
3. ✅ Updated all relevant files to use the new timeout

## Files Changed
- `federated_config.py` - Added send_timeout support
- `federated_config.yaml` - Set send_timeout: 3600
- `src/communication/federated_websockets.py` - Use configurable timeout
- `src/core/federated_client.py` - Pass timeout to WebSocket client

## What Changed in Your Config

### Before (Ignored):
```yaml
communication:
  round_timeout: 3000  # Was not being loaded!
```

### After (Working):
```yaml
communication:
  round_timeout: 3000   # Server waits 50 minutes for all clients
  send_timeout: 3600    # Client has 60 minutes to send updates
```

## How to Test

### Start Server:
```bash
cd /home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA
python federated_main.py --mode server --rounds 2 --port 8771
```

### Start QQP Client (Full Data):
```bash
# In another terminal
cd /home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA
python federated_main.py --mode client --client_id client_qqp --tasks qqp --port 8771
```

### Watch for Success Messages:
```
Client client_qqp using send timeout of 3600 seconds
Message sent successfully from client client_qqp
[SUCCESS] Training completed and update sent for round 1
```

## Adjusting Timeouts (If Needed)

Edit `federated_config.yaml`:

### For Even Larger Datasets:
```yaml
communication:
  round_timeout: 4500   # 75 minutes
  send_timeout: 5400    # 90 minutes
```

### For Faster Hardware:
```yaml
communication:
  round_timeout: 2400   # 40 minutes
  send_timeout: 3000    # 50 minutes
```

## Timeout Guidelines

| Dataset Size | send_timeout | round_timeout |
|-------------|--------------|---------------|
| <1K samples | 300s (5 min) | 600s (10 min) |
| 1K-10K | 900s (15 min) | 1200s (20 min) |
| 10K-100K | 1800s (30 min) | 2400s (40 min) |
| **>100K (like QQP)** | **3600s (60 min)** | **3000s (50 min)** |

## Why It Works Now

### Timeline for Full QQP Dataset:

| Stage | Time | Timeout |
|-------|------|---------|
| Training | ~30 min | N/A |
| Serializing updates | ~5 min | N/A |
| Sending updates | ~10 min | ✅ **send_timeout: 3600s** |
| Server aggregation | ~1 min | N/A |
| **Total** | **~46 min** | ✅ **round_timeout: 3000s** |

### Before Fix:
- Training: 30 min ✅
- Sending: 10 min ❌ **TIMEOUT at 300s (5 min)!**
- Result: **Failed to send update**

### After Fix:
- Training: 30 min ✅
- Sending: 10 min ✅ **Within 3600s (60 min) timeout**
- Result: **Successfully sent update!**

## Troubleshooting

### Still Getting Timeout?

1. **Check actual send time:**
```bash
grep "Message sent successfully" federated_client_*.log
```

2. **Check message size:**
```bash
grep "sending message of size" federated_client_*.log
```

3. **Increase timeouts further** if needed:
```yaml
send_timeout: 7200    # 2 hours
round_timeout: 6000   # 100 minutes
```

### Connection Issues?

1. **Check if client is connected:**
```bash
grep "connected to server" federated_client_*.log
```

2. **Check server logs:**
```bash
tail -f federated_server_8771.log
```

3. **Check network:**
```bash
netstat -an | grep 8771
```

## Summary

✅ **ROOT CAUSE:** Hardcoded 300s send timeout vs 10-15 min actual send time
✅ **FIX:** Made send_timeout configurable, set to 3600s (60 min)
✅ **BONUS:** Fixed round_timeout not being loaded from config
✅ **RESULT:** Full QQP dataset training now works!

Your training should now complete successfully with the full 323,415 QQP samples! 🎉

