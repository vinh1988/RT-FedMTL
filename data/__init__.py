"""
Data processing and dataset utilities for FedAvgLS.
"""

from .news20_dataset import (
    News20Dataset,
    News20DataLoader,
    News20Config,
    create_20news_federated_data
)

__all__ = [
    "News20Dataset",
    "News20DataLoader",
    "News20Config", 
    "create_20news_federated_data"
]
