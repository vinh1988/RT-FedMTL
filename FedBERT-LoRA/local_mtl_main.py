#!/usr/bin/env python3
"""
Multi-Task Learning with LoRA & Knowledge Distillation
Local training mode (Federated Learning components removed)
"""

import argparse
import sys
from typing import List

# Import from the modular structure (FL components removed)
from federated_config import FederatedConfig, load_config
from src.lora.federated_lora import LoRAFederatedModel
from src.knowledge_distillation.federated_knowledge_distillation import BidirectionalKDManager
from dataset_factory import DatasetFactory
from src.training.local_trainer import LocalTrainer
import torch
import logging

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Multi-Task Learning with LoRA & KD")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--config-file", type=str, default="federated_config.yaml",
                       help="Configuration file name")

    # Training arguments
    parser.add_argument("--tasks", nargs='+', choices=["sst2", "qqp", "stsb"],
                       help="Task names for training (space-separated)")
    parser.add_argument("--rounds", type=int, default=2, help="Number of training rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per round")

    # Model arguments
    parser.add_argument("--teacher_model", type=str, default="bert-base-uncased",
                       help="Teacher model name")
    parser.add_argument("--student_model", type=str, default="prajjwal1/bert-tiny",
                       help="Student model name")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")

    # KD arguments
    parser.add_argument("--kd_temperature", type=float, default=3.0, help="KD temperature")
    parser.add_argument("--kd_alpha", type=float, default=0.5, help="KD alpha")

    # Logging arguments
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    return parser

def main():
    """Main entry point for local MTL training"""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args)

        # Override config with command line arguments
        if args.rounds != 2:
            config.num_rounds = args.rounds
        if args.epochs != 1:
            config.local_epochs = args.epochs
        if args.log_level != "INFO":
            config.log_level = args.log_level
        if args.lora_rank != 8:
            config.lora_rank = args.lora_rank
        if args.lora_alpha != 16.0:
            config.lora_alpha = args.lora_alpha
        if args.kd_temperature != 3.0:
            config.kd_temperature = args.kd_temperature
        if args.kd_alpha != 0.5:
            config.kd_alpha = args.kd_alpha

        # Set up logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        logger = logging.getLogger(__name__)

        # Print configuration summary
        print("🔧 Multi-Task Learning Configuration")
        print("=" * 50)
        print(f"📊 Teacher: {config.server_model}, Student: {config.client_model}")
        print(f"🔧 LoRA: Rank={config.lora_rank}, Alpha={config.lora_alpha}")
        print(f"👨‍🏫 KD: T={config.kd_temperature}, α={config.kd_alpha}")
        print(f"🎯 Training: {config.num_rounds} rounds, {config.local_epochs} epochs")
        print(f"📚 Tasks: {', '.join(args.tasks) if args.tasks else 'All tasks'}")
        print("=" * 50)

        # Check if tasks are specified
        if not args.tasks:
            print("❌ Error: Tasks are required for local training")
            sys.exit(1)

        # Initialize models and components
        print("🚀 Initializing models and components...")

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {device}")

        # Initialize teacher model (frozen)
        teacher_model = None
        if True:  # Always use teacher for KD when available
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            teacher_model = AutoModelForSequenceClassification.from_pretrained(
                config.server_model, num_labels=2  # Will be overridden per task
            )
            teacher_model.eval()
            teacher_model.to(device)  # Move to same device as student model

        # Initialize student model with LoRA
        student_model = LoRAFederatedModel(
            base_model_name=config.client_model,
            tasks=args.tasks,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha
        )

        # Initialize KD manager
        kd_manager = BidirectionalKDManager(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=config.kd_temperature,
            alpha=config.kd_alpha
        )

        # Move KD manager models to device
        if teacher_model is not None:
            teacher_model.to(device)
        student_model.to(device)

        # Initialize dataset handlers
        dataset_handlers = {}
        for task in args.tasks:
            handler = DatasetFactory.create_handler(task, config)
            dataset_handlers[task] = handler

        # Initialize local trainer
        trainer = LocalTrainer(
            student_model=student_model,
            kd_manager=kd_manager,
            dataset_handlers=dataset_handlers,
            config=config
        )

        # Run local training
        print(f"🏃 Starting local multi-task training for {len(args.tasks)} tasks...")
        results = trainer.train_local_mtl(args.rounds)

        # Print final results
        print("\n📊 Training Results:")
        print("=" * 50)
        for task, metrics in results.items():
            print(f"🎯 {task.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        print("=" * 50)

        print("✅ Local multi-task training completed successfully!")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
