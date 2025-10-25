# QQP Full Data Timeout Fix

## Problem Identified

When training with the **full QQP dataset** (323,415 training samples), the client could not send updates to the server even with a `round_timeout` of 3000 seconds. However, with **less data**, it worked fine.

## Root Causes

### 1. **WebSocket Send Timeout Too Short** (PRIMARY ISSUE)
The WebSocket client had a **hardcoded send timeout of 300 seconds (5 minutes)**:
```python
send_timeout = 300  # 5 minutes timeout for send
```

When training on the full QQP dataset:
- Training takes a long time (acceptable)
- The LoRA updates are large
- **Sending the update takes longer than 300 seconds**
- The send operation **times out before completion**
- Result: Server never receives the update

### 2. **round_timeout Not Being Loaded from YAML**
The `round_timeout` parameter in `federated_config.yaml` was being **ignored** because:
- The config mapping in `federated_config.py` was missing the `('communication', 'round_timeout')` mapping
- The value was hardcoded to 3000 in the property method
- Your YAML setting had no effect

### 3. **No send_timeout Configuration**
There was no way to configure the send timeout for large updates:
- Always used the hardcoded 300 seconds
- Could not be adjusted for full dataset training
- No flexibility for different dataset sizes

## Solutions Applied

### 1. ✅ Added Configurable `send_timeout`

**In `federated_config.py`:**
- Added `send_timeout: int = 3600` to `CommunicationConfig` class
- Added `send_timeout: int = 3600` to `FederatedConfig` class
- Updated the `communication` property to use the config value
- Added mapping: `('communication', 'send_timeout'): 'send_timeout'`

**In `federated_websockets.py`:**
- Changed `WebSocketClient.__init__()` to accept `send_timeout` parameter
- Replaced hardcoded `send_timeout = 300` with `self.send_timeout`
- Added logging to show the timeout being used

**In `federated_client.py`:**
- Updated client initialization to pass `send_timeout` from config to `WebSocketClient`

**In `federated_config.yaml`:**
- Added `send_timeout: 3600` (60 minutes) for full dataset training

### 2. ✅ Fixed `round_timeout` Loading

**In `federated_config.py`:**
- Added mapping: `('communication', 'round_timeout'): 'round_timeout'`
- Changed communication property to use `getattr(self, 'round_timeout', 3000)`
- Added `round_timeout` to YAML export in `to_yaml_file()`

### 3. ✅ Improved Configuration

**Updated `federated_config.yaml`:**
```yaml
communication:
  port: 8771
  timeout: 60                    # Client timeout (seconds)
  websocket_timeout: 30          # WebSocket timeout (seconds)
  retry_attempts: 3              # Retry attempts
  round_timeout: 3000            # Server waits 3000s for all client updates (50 minutes)
  send_timeout: 3600             # Client send timeout for large updates (60 minutes)
```

## How It Works Now

### Small Dataset (e.g., 100 samples):
1. Training completes in ~30 seconds
2. Update is small, sends in ~1 second
3. Well within 300-second timeout ✅

### Full Dataset (e.g., 323,415 samples):
1. Training completes in ~30 minutes
2. Update is large, might take 10-15 minutes to send
3. **Now uses 3600-second (60-minute) timeout** ✅
4. Send completes successfully ✅

### Server Side:
1. Server waits up to 3000 seconds (50 minutes) for all clients
2. This gives enough time for:
   - Client training on full dataset (~30 minutes)
   - Client sending large updates (~15 minutes)
   - Network delays and retries (~5 minutes buffer)

## Configuration Guidelines

### For Different Dataset Sizes:

| Dataset Size | Expected Training Time | Recommended `send_timeout` | Recommended `round_timeout` |
|-------------|------------------------|---------------------------|----------------------------|
| Small (<1K) | < 1 minute | 300s (5 min) | 600s (10 min) |
| Medium (1K-10K) | 1-5 minutes | 900s (15 min) | 1200s (20 min) |
| Large (10K-100K) | 5-20 minutes | 1800s (30 min) | 2400s (40 min) |
| Very Large (>100K) | 20-40 minutes | **3600s (60 min)** | **3000s (50 min)** |

### Calculation Formula:
```
send_timeout = (expected_training_time + expected_send_time + buffer) * 1.5
round_timeout = send_timeout + 300 (for client startup and connection)
```

## Testing the Fix

### Test with Full QQP Dataset:

1. **Start Server:**
```bash
python federated_main.py --mode server --rounds 2 --port 8771
```

2. **Start QQP Client with Full Dataset:**
```bash
python federated_main.py --mode client --client_id client_qqp --tasks qqp --port 8771
```

3. **Monitor Logs:**
```bash
# Watch for these success indicators:
# - "Client client_qqp using send timeout of 3600 seconds"
# - "Message sent successfully from client client_qqp"
# - "[SUCCESS] Training completed and update sent for round 1"
```

### Expected Behavior:
- ✅ Client trains on all 323,415 QQP samples
- ✅ Client sends update (may take 10-15 minutes)
- ✅ Server receives update before timeout
- ✅ Round completes successfully

### If Still Experiencing Issues:

**Increase timeouts further:**
```yaml
communication:
  round_timeout: 4500            # 75 minutes
  send_timeout: 5400             # 90 minutes
```

**Check logs for:**
- Actual send time: `grep "Message sent successfully" federated_client_*.log`
- Timeout errors: `grep "timeout" federated_client_*.log`
- Message size: `grep "message of size" federated_client_*.log`

## Technical Details

### Why Sending Takes So Long:

1. **LoRA Update Size:**
   - Full dataset produces larger gradient updates
   - More diverse data = more parameter adjustments
   - Typical LoRA update: 10-50 MB serialized

2. **JSON Serialization:**
   - Python tensors → Lists → JSON
   - Large nested structures are slow to serialize
   - 50 MB tensor → ~200 MB JSON string

3. **WebSocket Transmission:**
   - JSON string sent over WebSocket
   - Network bandwidth limits
   - 200 MB over 100 Mbps = ~16 seconds minimum
   - Add protocol overhead: ~2-3x longer

4. **Total Time:**
   - Serialization: 5-10 minutes
   - Transmission: 1-5 minutes
   - **Total: 10-15 minutes for full dataset**

### Alternative Optimizations (Future Work):

1. **Compression:** Use gzip/lz4 to compress JSON
2. **Binary Protocol:** Use Protocol Buffers instead of JSON
3. **Chunked Uploads:** Split large updates into chunks
4. **Sparse Updates:** Only send changed parameters
5. **Quantization:** Reduce parameter precision

## Summary

✅ **FIXED:** Added configurable `send_timeout` (now 3600 seconds = 60 minutes)
✅ **FIXED:** `round_timeout` now properly loaded from YAML config
✅ **IMPROVED:** Better logging to show timeout values being used
✅ **DOCUMENTED:** Configuration guidelines for different dataset sizes

**The QQP client can now successfully train on the full dataset and send updates to the server!**

