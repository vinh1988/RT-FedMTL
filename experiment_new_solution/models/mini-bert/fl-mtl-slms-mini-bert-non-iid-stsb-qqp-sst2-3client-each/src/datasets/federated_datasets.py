#!/usr/bin/env python3
"""
Federated Learning Dataset Handlers
Task-specific dataset loading with flexible sizing
"""

import logging
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from datasets import load_dataset
import torch

logger = logging.getLogger(__name__)


def extract_client_number(client_id: str) -> int:
    """
    Extract numeric client ID from client_id string
    
    Args:
        client_id: String like 'sst2_client_1', 'qqp_client_2', etc.
    
    Returns:
        int: Numeric client ID (0-based)
    """
    try:
        # Extract the number after the last underscore
        return int(client_id.split('_')[-1]) - 1
    except (ValueError, IndexError):
        logger.warning(f"Could not extract client number from {client_id}, using 0")
        return 0


def load_non_iid_split(texts: List[str], labels: List, num_clients: int, alpha: float = 0.5, random_seed: int = 42) -> List[Tuple[List[str], List]]:
    """
    Implement Non-IID splitting using Dirichlet distribution
    
    Args:
        texts: List of text samples
        labels: List of corresponding labels
        num_clients: Number of clients to split across
        alpha: Dirichlet alpha parameter (lower = more heterogeneity)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of tuples (client_texts, client_labels) for each client
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    num_samples = len(texts)
    
    # Get unique labels and create label distribution
    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)
    
    # Generate Dirichlet distribution over labels
    dirichlet_dist = np.random.dirichlet([alpha] * num_labels, num_clients)
    
    # Initialize client data
    client_data = [[] for _ in range(num_clients)]
    
    # Distribute samples based on Dirichlet distribution
    for label_idx, label in enumerate(unique_labels):
        # Get indices of samples with this label
        label_indices = [i for i, l in enumerate(labels) if l == label]
        
        # Shuffle label indices
        np.random.shuffle(label_indices)
        
        # Distribute this label's samples according to Dirichlet proportions
        label_dist = dirichlet_dist[:, label_idx]
        label_dist = label_dist / label_dist.sum()  # Normalize
        
        start_idx = 0
        for client_idx in range(num_clients):
            # Calculate number of samples for this client
            if client_idx < num_clients - 1:
                num_samples_for_client = int(len(label_indices) * label_dist[client_idx])
            else:
                # Last client gets remaining samples
                num_samples_for_client = len(label_indices) - start_idx
            
            # Assign samples to client
            end_idx = start_idx + num_samples_for_client
            client_samples = label_indices[start_idx:end_idx]
            
            for sample_idx in client_samples:
                client_data[client_idx].append(sample_idx)
            
            start_idx = end_idx
    
    # Convert indices back to texts and labels
    result = []
    for client_idx in range(num_clients):
        client_indices = client_data[client_idx]
        client_texts = [texts[i] for i in client_indices]
        client_labels = [labels[i] for i in client_indices]
        result.append((client_texts, client_labels))
        
        logger.info(f"Client {client_idx + 1}: {len(client_texts)} samples, "
                   f"label distribution: {dict(zip(*np.unique(client_labels, return_counts=True)))}")
    
    return result

@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    task_name: str
    train_samples: int = None  # None means use all available
    val_samples: int = None     # None means use all available
    train_val_split: float = 0.8
    random_seed: int = 42

class BaseDatasetHandler(ABC):
    """Base class for dataset handlers"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.task_name = config.task_name

    @abstractmethod
    def load_raw_dataset(self) -> Tuple[List[str], List]:
        """Load raw dataset - implement in subclasses"""
        pass

    def prepare_data(self, num_clients: int = 1, client_id: int = 0, use_non_iid: bool = False, non_iid_alpha: float = 0.5) -> Dict:
        """Prepare training and validation data"""
        logger.info(f"Loading {self.task_name} dataset...")

        # Load raw data
        texts, labels = self.load_raw_dataset()

        logger.info(f"Loaded {len(texts)} samples for {self.task_name}")

        # Apply sample limits if specified
        if self.config.train_samples is not None or self.config.val_samples is not None:
            total_needed = (self.config.train_samples or 0) + (self.config.val_samples or 0)
            if len(texts) > total_needed:
                # Sample from dataset
                indices = list(range(len(texts)))
                random.seed(self.config.random_seed)
                random.shuffle(indices)

                selected_indices = indices[:total_needed]
                texts = [texts[i] for i in selected_indices]
                labels = [labels[i] for i in selected_indices]

        total_samples = len(texts)
        
        # Calculate train_size for both Non-IID and IID cases
        train_size = self.config.train_samples or int(total_samples * self.config.train_val_split)

        # Apply Non-IID splitting if requested and multiple clients
        if use_non_iid and num_clients > 1:
            # Use Non-IID splitting for training data
            all_texts = texts[:train_size]
            all_labels = labels[:train_size]
            
            client_splits = load_non_iid_split(
                texts=all_texts,
                labels=all_labels,
                num_clients=num_clients,
                alpha=non_iid_alpha,
                random_seed=self.config.random_seed
            )
            
            # Get this client's data
            if client_id < len(client_splits):
                train_texts, train_labels = client_splits[client_id]
            else:
                # Fallback to simple split
                train_texts = all_texts
                train_labels = all_labels
        else:
            # Simple train/val split (IID)
            indices = list(range(total_samples))
            random.seed(self.config.random_seed)
            random.shuffle(indices)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_texts = [texts[i] for i in train_indices]
            train_labels = [labels[i] for i in train_indices]

        # Handle validation data (always use simple split)
        val_size = self.config.val_samples or int(total_samples * (1 - self.config.train_val_split))
        val_start = train_size
        val_end = min(val_start + val_size, total_samples)
        val_texts = texts[val_start:val_end]
        val_labels = labels[val_start:val_end]

        logger.info(f"Task {self.task_name}: Train={len(train_texts)}, Validation={len(val_texts)}")

        return {
            'texts': train_texts,
            'labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels,
            'task_type': self.get_task_type(),
            'distribution': {
                'data': len(train_labels),
                'validation': len(val_labels)
            }
        }

    @abstractmethod
    def get_task_type(self) -> str:
        """Return task type: 'classification' or 'regression'"""
        pass

class SST2DatasetHandler(BaseDatasetHandler):
    """Handler for SST-2 sentiment analysis dataset"""

    def load_raw_dataset(self) -> Tuple[List[str], List]:
        """Load SST-2 dataset"""
        dataset = load_dataset("glue", "sst2", split="train")
        texts = [item["sentence"] for item in dataset]
        labels = [item["label"] for item in dataset]
        return texts, labels

    def get_task_type(self) -> str:
        return "classification"

class QQPDatasetHandler(BaseDatasetHandler):
    """Handler for QQP question pair classification dataset"""

    def load_raw_dataset(self) -> Tuple[List[str], List]:
        """Load QQP dataset"""
        dataset = load_dataset("glue", "qqp", split="train")
        texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
        labels = [item["label"] for item in dataset]
        return texts, labels

    def get_task_type(self) -> str:
        return "classification"

class STSBDatasetHandler(BaseDatasetHandler):
    """Handler for STSB semantic similarity dataset"""

    def load_raw_dataset(self) -> Tuple[List[str], List]:
        """Load STSB dataset"""
        dataset = load_dataset("glue", "stsb", split="train")
        texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
        labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to 0-1
        return texts, labels

    def get_task_type(self) -> str:
        return "regression"

class DatasetFactory:
    """Factory for creating dataset handlers"""

    _handlers = {
        'sst2': SST2DatasetHandler,
        'qqp': QQPDatasetHandler,
        'stsb': STSBDatasetHandler,
    }

    @classmethod
    def create_handler(cls, task_name: str, config: DatasetConfig):
        """Create appropriate dataset handler"""
        if task_name not in cls._handlers:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(cls._handlers.keys())}")

        return cls._handlers[task_name](config)

    @classmethod
    def get_available_tasks(cls) -> List[str]:
        """Get list of available tasks"""
        return list(cls._handlers.keys())
