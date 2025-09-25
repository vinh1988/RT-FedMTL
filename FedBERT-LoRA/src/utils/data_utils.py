"""
Data utilities for federated BERT learning.
Handles data loading, preprocessing, and partitioning.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset, load_dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GLUEDataset(Dataset):
    """PyTorch Dataset for GLUE tasks"""
    
    def __init__(self, data: HFDataset, tokenizer: AutoTokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different GLUE task formats
        if "sentence1" in item and "sentence2" in item:
            # Sentence pair tasks (MRPC, RTE, etc.)
            text1 = item["sentence1"]
            text2 = item["sentence2"]
            encoding = self.tokenizer(
                text1, text2,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        elif "sentence" in item:
            # Single sentence tasks (SST-2, CoLA)
            text = item["sentence"]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length", 
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            # Fallback for other formats
            text = str(item.get("text", item.get("premise", "")))
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


def create_data_loader(data: HFDataset, 
                      tokenizer: AutoTokenizer,
                      batch_size: int = 16,
                      max_length: int = 128,
                      shuffle: bool = True,
                      num_workers: int = 0) -> DataLoader:
    """Create a PyTorch DataLoader from HuggingFace dataset"""
    
    dataset = GLUEDataset(data, tokenizer, max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


def prepare_glue_data(task_name: str = "sst2", 
                     cache_dir: Optional[str] = None) -> Tuple[HFDataset, HFDataset, int]:
    """Load and prepare GLUE dataset"""
    
    logger.info(f"Loading GLUE task: {task_name}")
    
    # Load dataset
    if task_name.lower() == "mnli":
        dataset = load_dataset("glue", "mnli", cache_dir=cache_dir)
        train_data = dataset["train"]
        # Use matched validation for MNLI
        val_data = dataset["validation_matched"]
    else:
        dataset = load_dataset("glue", task_name, cache_dir=cache_dir)
        train_data = dataset["train"]
        val_data = dataset["validation"]
    
    # Get number of labels
    if task_name.lower() in ["sst2", "cola", "mrpc", "qqp", "rte"]:
        num_labels = 2
    elif task_name.lower() in ["mnli"]:
        num_labels = 3
    elif task_name.lower() in ["sts-b"]:
        num_labels = 1  # Regression task
    else:
        # Try to infer from data
        labels = set(train_data["label"])
        num_labels = len(labels)
    
    logger.info(f"Dataset loaded: {len(train_data)} train, {len(val_data)} validation samples")
    logger.info(f"Number of labels: {num_labels}")
    
    return train_data, val_data, num_labels


def partition_data(data: HFDataset, 
                  num_clients: int,
                  partition_strategy: str = "iid",
                  alpha: float = 0.5) -> List[HFDataset]:
    """Partition data among federated clients"""
    
    logger.info(f"Partitioning data for {num_clients} clients using {partition_strategy} strategy")
    
    if partition_strategy == "iid":
        return _partition_iid(data, num_clients)
    elif partition_strategy == "non_iid_dirichlet":
        return _partition_non_iid_dirichlet(data, num_clients, alpha)
    elif partition_strategy == "non_iid_shards":
        return _partition_non_iid_shards(data, num_clients)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")


def _partition_iid(data: HFDataset, num_clients: int) -> List[HFDataset]:
    """Partition data in IID manner"""
    
    # Shuffle data
    shuffled_data = data.shuffle(seed=42)
    
    # Split into equal chunks
    chunk_size = len(shuffled_data) // num_clients
    partitions = []
    
    for i in range(num_clients):
        start_idx = i * chunk_size
        if i == num_clients - 1:
            # Last client gets remaining data
            end_idx = len(shuffled_data)
        else:
            end_idx = (i + 1) * chunk_size
        
        client_data = shuffled_data.select(range(start_idx, end_idx))
        partitions.append(client_data)
    
    logger.info(f"IID partitioning completed: {[len(p) for p in partitions]} samples per client")
    return partitions


def _partition_non_iid_dirichlet(data: HFDataset, 
                                num_clients: int, 
                                alpha: float = 0.5) -> List[HFDataset]:
    """Partition data using Dirichlet distribution for non-IID split"""
    
    # Get labels
    labels = np.array(data["label"])
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    
    # Generate Dirichlet distribution for each client
    label_distributions = np.random.dirichlet([alpha] * num_labels, num_clients)
    
    # Calculate number of samples per client per label
    total_samples = len(data)
    samples_per_client = total_samples // num_clients
    
    partitions = [[] for _ in range(num_clients)]
    
    for label in unique_labels:
        # Get indices for this label
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        
        # Distribute samples according to Dirichlet distribution
        start_idx = 0
        for client_id in range(num_clients):
            # Calculate number of samples for this client and label
            proportion = label_distributions[client_id][label]
            num_samples = int(proportion * samples_per_client)
            
            # Ensure we don't exceed available samples
            end_idx = min(start_idx + num_samples, len(label_indices))
            
            if start_idx < len(label_indices):
                client_indices = label_indices[start_idx:end_idx]
                partitions[client_id].extend(client_indices.tolist())
                start_idx = end_idx
    
    # Convert to HuggingFace datasets
    hf_partitions = []
    for partition_indices in partitions:
        if partition_indices:
            client_data = data.select(partition_indices)
            hf_partitions.append(client_data)
        else:
            # Create empty dataset with same structure
            empty_data = data.select([0])  # Take first sample as template
            empty_data = empty_data.select([])  # Make it empty
            hf_partitions.append(empty_data)
    
    logger.info(f"Non-IID Dirichlet partitioning completed: {[len(p) for p in hf_partitions]} samples per client")
    return hf_partitions


def _partition_non_iid_shards(data: HFDataset, num_clients: int) -> List[HFDataset]:
    """Partition data into shards for non-IID split (each client gets 2 shards)"""
    
    # Sort data by labels
    sorted_indices = np.argsort(data["label"])
    
    # Create shards (2 shards per client)
    num_shards = num_clients * 2
    shard_size = len(data) // num_shards
    
    shards = []
    for i in range(num_shards):
        start_idx = i * shard_size
        if i == num_shards - 1:
            end_idx = len(data)
        else:
            end_idx = (i + 1) * shard_size
        
        shard_indices = sorted_indices[start_idx:end_idx]
        shards.append(shard_indices)
    
    # Randomly assign 2 shards to each client
    np.random.shuffle(shards)
    partitions = []
    
    for i in range(num_clients):
        # Each client gets 2 consecutive shards
        client_indices = np.concatenate([shards[i*2], shards[i*2 + 1]])
        client_data = data.select(client_indices.tolist())
        partitions.append(client_data)
    
    logger.info(f"Non-IID shards partitioning completed: {[len(p) for p in partitions]} samples per client")
    return partitions


def create_federated_datasets(task_name: str = "sst2",
                             num_clients: int = 10,
                             partition_strategy: str = "iid",
                             alpha: float = 0.5,
                             test_split: float = 0.2,
                             cache_dir: Optional[str] = None) -> Tuple[List[HFDataset], List[HFDataset], int]:
    """Create federated datasets for training and testing"""
    
    # Load GLUE data
    train_data, val_data, num_labels = prepare_glue_data(task_name, cache_dir)
    
    # Partition training data
    train_partitions = partition_data(train_data, num_clients, partition_strategy, alpha)
    
    # Create test partitions (smaller, for local evaluation)
    test_partitions = []
    for train_partition in train_partitions:
        # Take a small portion of training data for local testing
        test_size = max(1, int(len(train_partition) * test_split))
        test_partition = train_partition.select(range(min(test_size, len(train_partition))))
        test_partitions.append(test_partition)
    
    return train_partitions, test_partitions, num_labels


if __name__ == "__main__":
    # Test data utilities
    from transformers import AutoTokenizer
    
    # Test GLUE data loading
    train_data, val_data, num_labels = prepare_glue_data("sst2")
    print(f"Loaded SST-2: {len(train_data)} train, {len(val_data)} val, {num_labels} labels")
    
    # Test data partitioning
    partitions = partition_data(train_data, num_clients=5, partition_strategy="iid")
    print(f"Created {len(partitions)} partitions: {[len(p) for p in partitions]} samples")
    
    # Test data loader
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    data_loader = create_data_loader(partitions[0], tokenizer, batch_size=8)
    print(f"Created data loader with {len(data_loader)} batches")
    
    # Test a batch
    batch = next(iter(data_loader))
    print(f"Batch shape: {batch['input_ids'].shape}")
    print("Data utilities test completed successfully")
