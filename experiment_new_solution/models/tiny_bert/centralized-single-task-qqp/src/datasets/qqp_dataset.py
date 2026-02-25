"""
QQP Dataset Handler for Centralized Training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Tuple
import random

class QQPDataset(Dataset):
    """QQP dataset for centralized training"""
    
    def __init__(self, max_length: int = 128, cache_dir: str = "cache"):
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Load dataset
        self.dataset = load_dataset("glue", "mrpc", cache_dir=cache_dir)  # Using MRPC as QQP proxy
        
    def __len__(self) -> int:
        return len(self.dataset["train"])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset["train"][idx]
        return {
            "sentence1": item["sentence1"],
            "sentence2": item["sentence2"],
            "label": torch.tensor(item["label"], dtype=torch.long)
        }
    
    def prepare_data(self, train_samples: int, val_samples: int, 
                    test_samples: int, batch_size: int = 16, 
                    random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders"""
        
        # Set random seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Split dataset
        full_train = self.dataset["train"]
        full_val = self.dataset["validation"]
        full_test = self.dataset["test"]
        
        # Sample training data
        train_indices = random.sample(range(len(full_train)), min(train_samples, len(full_train)))
        train_data = full_train.select(train_indices)
        
        # Sample validation data
        val_indices = random.sample(range(len(full_val)), min(val_samples, len(full_val)))
        val_data = full_val.select(val_indices)
        
        # Sample test data
        test_indices = random.sample(range(len(full_test)), min(test_samples, len(full_test)))
        test_data = full_test.select(test_indices)
        
        # Create datasets
        train_dataset = QQPDatasetWrapper(train_data, self.max_length)
        val_dataset = QQPDatasetWrapper(val_data, self.max_length)
        test_dataset = QQPDatasetWrapper(test_data, self.max_length)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

class QQPDatasetWrapper(Dataset):
    """Wrapper for QQP dataset with tokenization"""
    
    def __init__(self, dataset, max_length: int = 128):
        self.dataset = dataset
        self.max_length = max_length
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            item["sentence1"],
            item["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }
