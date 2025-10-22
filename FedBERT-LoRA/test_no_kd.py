#!/usr/bin/env python3
"""
Test version without KD to isolate device issues
"""

import argparse
import sys
import torch
import logging
from federated_config import FederatedConfig, load_config
from src.lora.federated_lora import LoRAFederatedModel
from dataset_factory import DatasetFactory

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Test LoRA without KD")

    parser.add_argument("--tasks", nargs='+', choices=["sst2", "qqp", "stsb"],
                       help="Task names for training (space-separated)")
    parser.add_argument("--rounds", type=int, default=1, help="Number of training rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per round")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")

    return parser

def main():
    """Test LoRA training without KD"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")

    # Load config
    config = FederatedConfig()
    config.client_model = "prajjwal1/bert-tiny"
    config.lora_rank = args.lora_rank
    config.lora_alpha = 16.0
    config.learning_rate = 2e-5
    config.batch_size = 4

    print("🔧 Configuration:")
    print(f"  Model: {config.client_model}")
    print(f"  LoRA Rank: {config.lora_rank}")
    print(f"  Device: {device}")

    # Initialize student model
    print("🚀 Initializing model...")
    student_model = LoRAFederatedModel(
        base_model_name=config.client_model,
        tasks=args.tasks,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha
    )
    student_model.to(device)

    # Initialize dataset handlers
    dataset_handlers = {}
    for task in args.tasks:
        handler = DatasetFactory.create_handler(task, config)
        dataset_handlers[task] = handler

    # Setup optimizer
    lora_params = [p for n, p in student_model.named_parameters() if 'lora' in n.lower()]
    optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate)

    # Setup loss functions
    task_criterion = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss()

    print(f"🏃 Starting training for {len(args.tasks)} tasks...")

    # Training loop
    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")

        for task, handler in dataset_handlers.items():
            print(f"  Training on {task}...")

            dataloader = handler.get_dataloader()
            task_loss = 0
            num_batches = 0

            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Handle different task types
                if task in ['sst2', 'qqp']:
                    labels = labels.long()
                    criterion = task_criterion
                else:  # STSB regression
                    criterion = mse_criterion

                # Forward pass
                student_model.train()
                outputs = student_model(input_ids, attention_mask, task)

                # Calculate loss
                loss = criterion(outputs.squeeze() if task == 'stsb' else outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                task_loss += loss.item()
                num_batches += 1

                if num_batches >= 2:  # Just test first 2 batches
                    break

            avg_loss = task_loss / num_batches if num_batches > 0 else 0
            print(f"    {task} loss: {avg_loss:.4f}")

    print("✅ LoRA training without KD completed successfully!")

if __name__ == "__main__":
    main()
