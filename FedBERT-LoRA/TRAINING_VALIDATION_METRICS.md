# Training vs Validation Metrics Explanation

## The Issue
Previously, the CSV output was confusing because it mixed training and validation metrics:
- `samples_processed` (training samples: 20, 50, 30)
- `correct_predictions` (training correct: 2, 25, 17)
- `val_samples` (validation samples: 4, 10, 6)

This made it unclear that `correct_predictions` was from **training**, not validation.

## The Solution
We now have separate columns for training and validation correct predictions:

### CSV Columns (client_results_*.csv)

| Column | Description | Example |
|--------|-------------|---------|
| `samples_processed` | Number of training samples | 20 |
| `correct_predictions` | Correct predictions on training data | 2 |
| `accuracy` | Training accuracy | 0.10 (10%) |
| `loss` | Training loss | 0.35 |
| `val_samples` | Number of validation samples | 4 |
| `val_correct_predictions` | Correct predictions on validation data | 1 |
| `val_accuracy` | Validation accuracy | 0.25 (25%) |
| `val_loss` | Validation loss | 0.08 |

## Example Output

```csv
round,client_id,task,accuracy,loss,samples_processed,correct_predictions,val_accuracy,val_loss,val_samples,val_correct_predictions,timestamp
1,stsb_client,stsb,0.46,0.08,20,2,0.36,0.08,4,1,2025-10-19 23:20:52
1,sst2_client,sst2,0.50,0.35,50,25,0.40,0.35,10,4,2025-10-19 23:20:54
1,qqp_client,qqp,0.57,0.35,30,17,0.50,0.35,6,3,2025-10-19 23:21:11
```

### Interpretation

**STSB Client (Round 1):**
- **Training**: 2 correct out of 20 samples = 10% tolerance-based correct
- **Validation**: 1 correct out of 4 samples = 25% tolerance-based correct
- Training accuracy: 0.46 (correlation-based for regression)
- Validation accuracy: 0.36 (correlation-based for regression)

**SST2 Client (Round 1):**
- **Training**: 25 correct out of 50 samples = 50%
- **Validation**: 4 correct out of 10 samples = 40%

**QQP Client (Round 1):**
- **Training**: 17 correct out of 30 samples = 57%
- **Validation**: 3 correct out of 6 samples = 50%

## Key Points

1. **Training Metrics** (`samples_processed`, `correct_predictions`, `accuracy`, `loss`)
   - Calculated during the training loop
   - Model is learning from this data
   - May show overfitting if much better than validation

2. **Validation Metrics** (`val_samples`, `val_correct_predictions`, `val_accuracy`, `val_loss`)
   - Calculated on held-out data the model hasn't trained on
   - Better indicator of true model performance
   - Should be used to assess generalization

3. **For Regression (STSB)**:
   - `accuracy` = correlation-based (0 to 1, where 1 is perfect correlation)
   - `correct_predictions` = tolerance-based count (within 0.1 of true value)
   - Also reports MAE, MSE, RMSE, and correlation

4. **For Classification (SST2, QQP)**:
   - `accuracy` = percentage of correct predictions
   - `correct_predictions` = count of samples where prediction == label

## Configuration

To use the full dataset sizes configured in `federated_config.yaml`:

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

**Important**: Don't use `--samples` command-line argument, as it overrides the YAML configuration!

Run clients without `--samples`:
```bash
# Correct way (uses YAML config)
python federated_main.py --mode client --client_id stsb_client --tasks stsb

# Wrong way (overrides YAML config)
python federated_main.py --mode client --client_id stsb_client --tasks stsb --samples 20
```


