"""
Metrics utilities for centralized training
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from typing import List, Dict

def compute_qqp_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute QQP metrics (binary classification)"""
    
    # Convert to numpy arrays
    y_true = np.array(labels)
    y_pred = np.array(predictions)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc)
    }

def compute_sst2_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute SST2 metrics (binary classification)"""
    # Same as QQP for binary classification
    return compute_qqp_metrics(predictions, labels)

def compute_stsb_metrics(predictions: List[float], labels: List[float]) -> Dict[str, float]:
    """Compute STSB metrics (regression)"""
    
    # Convert to numpy arrays
    y_true = np.array(labels)
    y_pred = np.array(predictions)
    
    # Regression metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Pearson correlation with validation
    pearson_corr = 0.0
    if len(y_true) > 1:
        try:
            # Only check for NaN/inf in std calculation
            std_true = np.std(y_true)
            std_pred = np.std(y_pred)
            
            # Allow very small std values (close to 0 but not exactly 0)
            if not (np.isnan(std_true) or np.isnan(std_pred) or 
                     np.isinf(std_true) or np.isinf(std_pred) or
                     (std_true == 0 and std_pred == 0)):
                pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
                if np.isnan(pearson_corr):
                    pearson_corr = 0.0
        except Exception as e:
            print(f"Pearson calculation error: {e}")
            pearson_corr = 0.0
    
    # Spearman correlation with validation
    spearman_corr = 0.0
    if len(y_true) > 1:
        try:
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(y_true, y_pred)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        except Exception as e:
            print(f"Spearman calculation error: {e}")
            spearman_corr = 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson': float(pearson_corr),
        'spearman': float(spearman_corr)
    }
