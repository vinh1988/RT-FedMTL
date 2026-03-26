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
    
    # Pearson correlation
    pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Spearman correlation (rank correlation)
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
    }
