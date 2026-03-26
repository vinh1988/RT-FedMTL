"""
Multi-Task Dataset Handler for Centralized Training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Tuple, Union
import random

class MTLDataset:
    """Multi-task dataset for centralized training"""
    
    def __init__(self, task_configs: Dict, max_length: int = 128, cache_dir: str = "cache"):
        self.task_configs = task_configs
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Load datasets
        self.datasets = {}
        self.load_datasets()
        
    def load_datasets(self):
        """Load all task datasets"""
        # Load SST2
        if 'sst2' in self.task_configs:
            self.datasets['sst2'] = load_dataset("glue", "sst2", cache_dir=self.cache_dir)
        
        # Load QQP
        if 'qqp' in self.task_configs:
            self.datasets['qqp'] = load_dataset("glue", "mrpc", cache_dir=self.cache_dir)  # Using MRPC as QQP proxy
        
        # Load STSB
        if 'stsb' in self.task_configs:
            self.datasets['stsb'] = load_dataset("glue", "stsb", cache_dir=self.cache_dir)
    
    def prepare_data(self, batch_size: int = 16, random_seed: int = 42) -> Tuple[Dict, Dict, Dict]:
        """Prepare data loaders for all tasks"""
        
        # Set random seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        train_loaders = {}
        val_loaders = {}
        test_loaders = {}
        
        for task_name, task_config in self.task_configs.items():
            # Get dataset
            dataset = self.datasets[task_name]
            
            # Sample training data
            train_samples = task_config['train_samples']
            val_samples = task_config['val_samples']
            test_samples = task_config.get('test_samples', val_samples)
            
            # Sample indices
            train_indices = random.sample(range(len(dataset["train"])), min(train_samples, len(dataset["train"])))
            train_data = dataset["train"].select(train_indices)
            
            val_indices = random.sample(range(len(dataset["validation"])), min(val_samples, len(dataset["validation"])))
            val_data = dataset["validation"].select(val_indices)
            
            # Create wrapped datasets
            train_dataset = MTLDatasetWrapper(train_data, task_name, task_config, self.max_length)
            val_dataset = MTLDatasetWrapper(val_data, task_name, task_config, self.max_length)
            
            # Create data loaders
            train_loaders[task_name] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loaders[task_name] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Test loader (same as validation)
            test_loaders[task_name] = val_loaders[task_name]
        
        return train_loaders, val_loaders, test_loaders

class MTLDatasetWrapper(Dataset):
    """Wrapper for MTL dataset with tokenization"""
    
    def __init__(self, dataset, task_name: str, task_config: Dict, max_length: int = 128):
        self.dataset = dataset
        self.task_name = task_name
        self.task_config = task_config
        self.max_length = max_length
        self.task_type = task_config['task_type']
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Tokenize based on task type
        if self.task_name == 'sst2':
            # Single sentence task
            encoded = self.tokenizer(
                item["sentence"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            label = torch.tensor(item["label"], dtype=torch.long)  # Classification
            
        elif self.task_name in ['qqp', 'stsb']:
            # Sentence pair task
            encoded = self.tokenizer(
                item["sentence1"],
                item["sentence2"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            if self.task_type == 'classification':
                label = torch.tensor(item["label"], dtype=torch.long)  # Classification
            else:  # regression
                label = torch.tensor(item["label"], dtype=torch.float)  # Regression
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": label,
            "task_name": self.task_name
        }
