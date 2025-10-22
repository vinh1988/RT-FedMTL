#!/usr/bin/env python3
"""
Simple test without KD to verify LoRA training works
"""

import sys
import os
import torch
from federated_config import FederatedConfig
from src.lora.federated_lora import LoRAFederatedModel
from dataset_factory import DatasetFactory

def test_lora_training():
    """Test LoRA training without KD"""
    try:
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {device}")

        config = FederatedConfig()
        config.client_model = 'prajjwal1/bert-tiny'
        config.lora_rank = 8
        config.lora_alpha = 16.0
        config.learning_rate = 2e-5
        config.batch_size = 4

        # Create model
        model = LoRAFederatedModel(
            base_model_name=config.client_model,
            tasks=['sst2'],
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha
        )
        model.to(device)

        # Create dataset
        handler = DatasetFactory.create_handler('sst2', config)
        dataloader = handler.get_dataloader()

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if 'lora' in n.lower()],
            lr=config.learning_rate
        )

        # Setup loss
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Just test first 2 batches
                break

            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).long()

            # Forward pass
            outputs = model(input_ids, attention_mask, 'sst2')
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

        print("✅ LoRA training test passed!")
        return True

    except Exception as e:
        print(f"❌ LoRA training test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing LoRA Training (without KD)")
    print("=" * 40)
    success = test_lora_training()
    if success:
        print("\n🎉 Basic LoRA training works correctly!")
    else:
        print("\n❌ Issues found with basic LoRA training.")
