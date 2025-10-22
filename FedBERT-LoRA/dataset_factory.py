#!/usr/bin/env python3
"""
Dataset Factory for Local Training
Creates dataset handlers for different tasks (FL components removed)
"""

import logging
import torch
from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """Simple dataset wrapper"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32 if isinstance(self.labels[idx], float) else torch.long)
        }

class SimpleDatasetHandler:
    """Simple dataset handler for local training"""

    def __init__(self, task: str, config, train_texts=None, train_labels=None, val_texts=None, val_labels=None):
        self.task = task
        self.config = config
        self.train_texts = train_texts or []
        self.train_labels = train_labels or []
        self.val_texts = val_texts or []
        self.val_labels = val_labels or []

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model)

        # Create datasets
        self.train_dataset = SimpleDataset(self.train_texts, self.train_labels, self.tokenizer)
        self.val_dataset = SimpleDataset(self.val_texts, self.val_labels, self.tokenizer)

    def get_dataloader(self, batch_size=None):
        """Get training dataloader"""
        batch_size = batch_size or self.config.batch_size
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True  # Enable faster GPU transfer if using GPU
        )

    def get_val_dataloader(self, batch_size=None):
        """Get validation dataloader"""
        batch_size = batch_size or self.config.batch_size
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True  # Enable faster GPU transfer if using GPU
        )

class DatasetFactory:
    """Factory for creating dataset handlers"""

    @staticmethod
    def create_handler(task: str, config) -> SimpleDatasetHandler:
        """Create a dataset handler for the specified task"""

        # Generate dummy data for testing (replace with real data loading)
        if task == "sst2":
            # Dummy SST-2 data (sentiment analysis)
            train_texts = [
                "This movie is absolutely fantastic",
                "I really enjoyed this film",
                "The acting was superb",
                "A masterpiece of cinema",
                "This movie is terrible",
                "I hated every minute",
                "Poor acting and bad story",
                "Waste of time",
                "Great movie, highly recommended",
                "Amazing cinematography",
                "Boring and predictable",
                "Not worth watching"
            ] * 8  # Multiply for more data

            train_labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0] * 8

            val_texts = [
                "Excellent movie with great plot",
                "Disappointing and poorly made",
                "Outstanding performances",
                "Mediocre at best"
            ]

            val_labels = [1, 0, 1, 0]

        elif task == "qqp":
            # Dummy QQP data (question pair classification)
            train_texts = [
                "What is the capital of France? Paris is the capital of France.",
                "How to cook pasta? Boil water and add pasta.",
                "What is machine learning? ML is a subset of AI.",
                "How does photosynthesis work? Plants convert light to energy.",
                "What is the weather today? I don't know the weather.",
                "How to bake a cake? Mix ingredients and bake.",
                "What is quantum computing? Computers using quantum mechanics.",
                "How to lose weight? Diet and exercise."
            ] * 6

            train_labels = [1, 1, 1, 1, 0, 1, 1, 1] * 6

            val_texts = [
                "What is AI? Artificial Intelligence.",
                "How to program? I don't know programming."
            ]

            val_labels = [1, 0]

        elif task == "stsb":
            # Dummy STSB data (semantic similarity)
            train_texts = [
                "The cat sits on the mat. A cat is resting on a mat.",
                "Machine learning is fascinating. AI is interesting.",
                "The weather is nice today. Today has beautiful weather.",
                "I love pizza. Pizza is my favorite food.",
                "The car is red. The vehicle has a crimson color.",
                "She runs quickly. He walks slowly.",
                "The sun is bright. The moon is dim.",
                "Water is wet. Fire is hot."
            ] * 6

            # Similarity scores (0-1 scale for STSB)
            train_labels = [0.9, 0.3, 0.8, 0.7, 0.6, 0.2, 0.1, 0.1] * 6

            val_texts = [
                "Dogs are friendly. Cats are mean.",
                "The sky is blue. The ocean is deep."
            ]

            val_labels = [0.2, 0.1]

        else:
            raise ValueError(f"Unsupported task: {task}")

        return SimpleDatasetHandler(
            task=task,
            config=config,
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels
        )
