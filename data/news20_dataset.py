#!/usr/bin/env python3
"""
20News Dataset Loader for Federated Learning
Handles dataset preparation, splitting, and preprocessing for classification tasks.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, AutoTokenizer
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class News20Config:
    """Configuration for 20News dataset"""
    data_dir: str = "./data/20news"
    max_length: int = 512
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_clients: int = 3
    random_seed: int = 42
    subset: str = "all"  # "train", "test", "all"
    remove_headers: bool = True
    remove_footers: bool = True
    remove_quotes: bool = True


class News20Dataset(Dataset):
    """20News Dataset for classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class News20DataLoader:
    """Data loader for 20News federated learning"""
    
    def __init__(self, config: News20Config):
        self.config = config
        self.tokenizer = None
        self.num_classes = 20
        self.class_names = [
            'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
            'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
            'sci.space', 'soc.religion.christian', 'talk.politics.guns',
            'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
        ]
        
    def load_tokenizer(self, model_name: str = "facebook/bart-base"):
        """Load tokenizer for the specified model"""
        try:
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer for {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_raw_data(self) -> Tuple[List[str], List[int]]:
        """Load raw 20News data"""
        logger.info("Loading 20News dataset...")
        
        try:
            # Load dataset
            newsgroups = fetch_20newsgroups(
                subset=self.config.subset,
                remove=('headers', 'footers', 'quotes') if self.config.remove_headers else ()
            )
            
            texts = newsgroups.data
            labels = newsgroups.target
            
            logger.info(f"Loaded {len(texts)} documents with {len(set(labels))} classes")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Failed to load 20News data: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better classification"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove email addresses
        import re
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '[URL]', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '[NUM]', text)
        
        return text
    
    def split_data(self, texts: List[str], labels: List[int]) -> Dict[str, Tuple[List[str], List[int]]]:
        """Split data into train/val/test sets"""
        logger.info("Splitting data into train/validation/test sets...")
        
        # First split: train vs (val + test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels,
            test_size=(1 - self.config.train_split),
            random_state=self.config.random_seed,
            stratify=labels
        )
        
        # Second split: val vs test
        val_size = self.config.val_split / (self.config.val_split + self.config.test_split)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=(1 - val_size),
            random_state=self.config.random_seed,
            stratify=temp_labels
        )
        
        logger.info(f"Data split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        return {
            "train": (train_texts, train_labels),
            "val": (val_texts, val_labels),
            "test": (test_texts, test_labels)
        }
    
    def create_federated_splits(self, texts: List[str], labels: List[int]) -> Dict[str, Tuple[List[str], List[int]]]:
        """Create federated data splits for multiple clients"""
        logger.info(f"Creating federated splits for {self.config.num_clients} clients...")
        
        # Sort by label to ensure balanced distribution
        sorted_indices = np.argsort(labels)
        sorted_texts = [texts[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        # Create balanced splits
        client_data = {}
        samples_per_client = len(sorted_texts) // self.config.num_clients
        
        for client_id in range(self.config.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            
            # Last client gets remaining samples
            if client_id == self.config.num_clients - 1:
                end_idx = len(sorted_texts)
            
            client_texts = sorted_texts[start_idx:end_idx]
            client_labels = sorted_labels[start_idx:end_idx]
            
            client_data[f"client_{client_id}"] = (client_texts, client_labels)
            
            logger.info(f"Client {client_id}: {len(client_texts)} samples")
        
        return client_data
    
    def create_datasets(self, model_name: str = "facebook/bart-base") -> Dict[str, Dict[str, Dataset]]:
        """Create all datasets for federated learning"""
        if self.tokenizer is None:
            self.load_tokenizer(model_name)
        
        # Load raw data
        texts, labels = self.load_raw_data()
        
        # Preprocess texts
        logger.info("Preprocessing texts...")
        texts = [self.preprocess_text(text) for text in texts]
        
        # Split data
        split_data = self.split_data(texts, labels)
        
        # Create federated splits for training data
        train_texts, train_labels = split_data["train"]
        federated_splits = self.create_federated_splits(train_texts, train_labels)
        
        # Create datasets
        datasets = {}
        
        # Federated training datasets
        for client_id, (client_texts, client_labels) in federated_splits.items():
            datasets[client_id] = {
                "train": News20Dataset(
                    client_texts, client_labels, self.tokenizer, self.config.max_length
                )
            }
        
        # Validation and test datasets (shared)
        val_texts, val_labels = split_data["val"]
        test_texts, test_labels = split_data["test"]
        
        val_dataset = News20Dataset(val_texts, val_labels, self.tokenizer, self.config.max_length)
        test_dataset = News20Dataset(test_texts, test_labels, self.tokenizer, self.config.max_length)
        
        # Add shared datasets to each client
        for client_id in datasets:
            datasets[client_id]["val"] = val_dataset
            datasets[client_id]["test"] = test_dataset
        
        logger.info("Created all datasets successfully")
        return datasets
    
    def create_data_loaders(self, datasets: Dict[str, Dict[str, Dataset]], 
                          batch_size: int = 8, num_workers: int = 4) -> Dict[str, Dict[str, DataLoader]]:
        """Create data loaders for all datasets"""
        data_loaders = {}
        
        for client_id, client_datasets in datasets.items():
            data_loaders[client_id] = {}
            
            for split, dataset in client_datasets.items():
                shuffle = (split == "train")
                data_loaders[client_id][split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=True
                )
        
        return data_loaders
    
    def save_data_info(self, datasets: Dict[str, Dict[str, Dataset]], save_path: str):
        """Save dataset information for analysis"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data_info = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "num_clients": self.config.num_clients,
            "dataset_sizes": {},
            "class_distributions": {}
        }
        
        for client_id, client_datasets in datasets.items():
            data_info["dataset_sizes"][client_id] = {}
            data_info["class_distributions"][client_id] = {}
            
            for split, dataset in client_datasets.items():
                data_info["dataset_sizes"][client_id][split] = len(dataset)
                
                # Calculate class distribution
                labels = [dataset.labels[i] for i in range(len(dataset))]
                class_counts = np.bincount(labels, minlength=self.num_classes)
                data_info["class_distributions"][client_id][split] = class_counts.tolist()
        
        with open(save_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        logger.info(f"Saved data info to {save_path}")


def create_20news_federated_data(config: News20Config, 
                                model_name: str = "facebook/bart-base",
                                save_info: bool = True) -> Dict[str, Dict[str, DataLoader]]:
    """Main function to create federated 20News data"""
    
    # Create data loader
    data_loader = News20DataLoader(config)
    
    # Create datasets
    datasets = data_loader.create_datasets(model_name)
    
    # Create data loaders
    data_loaders = data_loader.create_data_loaders(datasets)
    
    # Save data info if requested
    if save_info:
        info_path = os.path.join(config.data_dir, "data_info.json")
        data_loader.save_data_info(datasets, info_path)
    
    return data_loaders


def test_data_loader():
    """Test function for the data loader"""
    config = News20Config(
        data_dir="./data/20news",
        max_length=256,
        num_clients=3
    )
    
    try:
        # Create federated data
        data_loaders = create_20news_federated_data(config)
        
        # Test loading
        for client_id, client_loaders in data_loaders.items():
            print(f"\n{client_id}:")
            for split, loader in client_loaders.items():
                print(f"  {split}: {len(loader)} batches")
                
                # Test one batch
                for batch in loader:
                    print(f"    Batch shape: {batch['input_ids'].shape}")
                    print(f"    Labels: {batch['labels'][:5].tolist()}")
                    break
        
        print("\nData loader test completed successfully!")
        
    except Exception as e:
        print(f"Data loader test failed: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the data loader
    test_data_loader()
