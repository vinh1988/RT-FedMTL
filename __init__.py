"""
FedAvgLS: Federated Learning with DistilBART and MobileBART
A comprehensive federated learning framework for cross-architecture knowledge transfer.
"""

__version__ = "1.0.0"
__author__ = "FedAvgLS Team"
__email__ = "fedavgls@example.com"

# Core imports
from .data.news20_dataset import News20Dataset, News20DataLoader, News20Config
from .models.bart_classification import (
    DistilBARTClassifier, 
    MobileBARTClassifier, 
    BARTClassificationHead,
    create_distilbart_model,
    create_mobilebart_model,
    create_lora_config
)
from .training.fedmkt_trainer import (
    FedMKTTrainer,
    FedMKTTrainingConfig,
    KnowledgeDistillationLoss
)

__all__ = [
    # Data
    "News20Dataset",
    "News20DataLoader", 
    "News20Config",
    
    # Models
    "DistilBARTClassifier",
    "MobileBARTClassifier",
    "BARTClassificationHead",
    "create_distilbart_model",
    "create_mobilebart_model",
    "create_lora_config",
    
    # Training
    "FedMKTTrainer",
    "FedMKTTrainingConfig", 
    "KnowledgeDistillationLoss",
]
