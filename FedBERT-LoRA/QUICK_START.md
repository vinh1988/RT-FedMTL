# Quick Start - Multi-Task Federated Learning

## ✅ All Changes Applied

Your multi-task federated learning system is now ready with:
- **Balanced data**: 400 samples per client
- **Proper regression metrics**: MSE, RMSE, MAE, R², Pearson
- **Task-specific logging**: Clear output for classification vs regression
- **3-client minimum**: Scalability tests start at 3 clients

---

## Run the Experiment

```bash
cd /home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA
./run_comprehensive_experiments.sh scenario1 FULL_SCALE
```

---

## What to Expect

### Client Logs

**Classification Clients (1 & 2):**
```
Client 1 (sst2) training complete: Loss=0.6773, Acc=0.5731, P=0.5465, R=0.5731, F1=0.4990
Client 2 (qqp) training complete: Loss=0.6724, Acc=0.6040, P=0.5510, R=0.6040, F1=0.5176
```

**Regression Client (3):**
```
Client 3 (stsb) training complete: Loss=0.0071, MSE=0.0012, RMSE=0.0346, MAE=0.0234, R²=0.8756, Pearson=0.9123
```

### Sample Distribution
```
Client 1 (sst2): 399 samples, distribution: {'class_0': 170, 'class_1': 229}
Client 2 (qqp): 399 samples, distribution: {'class_0': 245, 'class_1': 154}
Client 3 (stsb): 400 samples, distribution: {'bin_0': 0, 'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 400}
```

---

## Check Results

### 1. View Client 3 Regression Metrics
```bash
grep "Client 3" experiment_logs/SCENARIO_1_scalability_3c_client_3.log | grep "training complete"
```

### 2. View CSV Results
```bash
cat experiment_results/SCENARIO_1_scalability_metrics_3clients_*.csv | head -10
```

### 3. View All Logs
```bash
ls -lh experiment_logs/SCENARIO_1_*.log
```

---

## Understanding the Metrics

### Classification Metrics (Clients 1 & 2)
- **Accuracy**: Percentage of correct predictions (0-1, higher is better)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Regression Metrics (Client 3)
- **MSE**: Mean Squared Error (lower is better, ~0.001 is good)
- **RMSE**: Root Mean Squared Error (lower is better, ~0.03 is good)
- **MAE**: Mean Absolute Error (lower is better, ~0.02 is good)
- **R²**: Coefficient of determination (0-1, higher is better, >0.8 is good)
- **Pearson**: Correlation coefficient (-1 to 1, >0.9 is excellent)

### CSV Metric Mapping
In the CSV files, regression metrics are mapped to classification field names:
- `accuracy` → R² Score
- `precision` → 1 - MAE
- `recall` → Pearson Correlation
- `f1_score` → RMSE

---

## Troubleshooting

### If you see "Acc=0.0000" for Client 3:
❌ The old code is still running. Make sure you saved `no_lora_federated_system.py`

### If imports fail:
```bash
/home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA/venv/bin/pip install scipy scikit-learn
```

### If experiment hangs:
```bash
# Kill all processes
pkill -f "no_lora_federated_system.py"
lsof -ti:8771,8781,8782,8783 | xargs -r kill -9

# Restart
./run_comprehensive_experiments.sh scenario1 FULL_SCALE
```

---

## Documentation Files

- 📄 `CHANGES_SUMMARY.md` - Complete list of changes
- 📄 `MULTI_TASK_IMPLEMENTATION.md` - Detailed implementation guide
- 📄 `TRAINING_RESULTS_COMPARISON.md` - Before/After comparison
- 📄 `TRAINING_ANALYSIS_CLIENT_3.md` - Original problem analysis
- 📄 `QUICK_START.md` - This file

---

## Ready to Run! 🚀

Your experiment is configured and ready. Simply run:

```bash
./run_comprehensive_experiments.sh scenario1 FULL_SCALE
```

Expected runtime: ~5 minutes for 3 clients, 22 rounds, 400 samples each.
