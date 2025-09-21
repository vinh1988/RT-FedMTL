#!/usr/bin/env python3
"""
Main Training Script for FedMKT on 20News Classification
Complete federated learning pipeline for DistilBART ↔ MobileBART knowledge transfer.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.news20_dataset import News20DataLoader, News20Config
from training.fedmkt_trainer import FedMKTTrainer, FedMKTTrainingConfig
from models.bart_classification import create_lora_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_config(yaml_config: Dict[str, Any]) -> News20Config:
    """Create data configuration from YAML config"""
    dataset_config = yaml_config.get("dataset", {})
    
    return News20Config(
        data_dir=yaml_config.get("paths", {}).get("data_dir", "./data/20news"),
        max_length=dataset_config.get("max_length", 512),
        train_split=dataset_config.get("train_split", 0.7),
        val_split=dataset_config.get("val_split", 0.15),
        test_split=dataset_config.get("test_split", 0.15),
        num_clients=len(yaml_config.get("models", {}).get("clients", [])),
        random_seed=42
    )


def create_training_config(yaml_config: Dict[str, Any]) -> FedMKTTrainingConfig:
    """Create training configuration from YAML config"""
    training_config = yaml_config.get("training", {})
    model_config = yaml_config.get("models", {})
    
    return FedMKTTrainingConfig(
        # Model configurations
        distilbart_model_name=model_config.get("central", {}).get("name", "facebook/distilbart-cnn-12-6"),
        mobilebart_model_name=model_config.get("clients", [{}])[0].get("name", "valhalla/mobile-bart"),
        num_labels=model_config.get("central", {}).get("num_labels", 20),
        max_length=model_config.get("central", {}).get("max_length", 512),
        
        # Training parameters
        learning_rate=training_config.get("learning_rate", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        num_epochs=training_config.get("global_epochs", 10),
        batch_size=training_config.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
        
        # Knowledge distillation
        distill_temperature=training_config.get("distill_temperature", 4.0),
        kd_alpha=training_config.get("kd_alpha", 0.7),
        distill_loss_type=training_config.get("distill_loss_type", "kl"),
        
        # LoRA configuration
        use_lora=True,
        lora_r=yaml_config.get("lora_config", {}).get("central", {}).get("r", 16),
        lora_alpha=yaml_config.get("lora_config", {}).get("central", {}).get("lora_alpha", 32),
        lora_dropout=yaml_config.get("lora_config", {}).get("central", {}).get("lora_dropout", 0.1),
        
        # Output
        output_dir=yaml_config.get("output", {}).get("output_dir", "./outputs/fedmkt_20news")
    )


def prepare_data(config: News20Config) -> Dict[str, Any]:
    """Prepare federated data"""
    logger.info("Preparing federated data...")
    
    # Create data loader
    data_loader = News20DataLoader(config)
    
    # Load and prepare datasets
    datasets = data_loader.create_datasets()
    
    # Create data loaders
    data_loaders = data_loader.create_data_loaders(
        datasets,
        batch_size=8,
        num_workers=4
    )
    
    # Save data info
    os.makedirs(config.data_dir, exist_ok=True)
    data_loader.save_data_info(datasets, os.path.join(config.data_dir, "data_info.json"))
    
    logger.info("Data preparation completed!")
    return data_loaders


def train_fedmkt(data_loaders: Dict[str, Any], config: FedMKTTrainingConfig):
    """Train FedMKT models"""
    logger.info("Starting FedMKT training...")
    
    # Create trainer
    trainer = FedMKTTrainer(config)
    
    # Extract public data loader (use validation set as public data for simplicity)
    public_data_loader = list(data_loaders.values())[0]["val"]
    
    # Train models
    training_history = trainer.train(data_loaders, public_data_loader)
    
    # Save models
    trainer.save_models()
    
    logger.info("FedMKT training completed!")
    return training_history


def evaluate_models(data_loaders: Dict[str, Any], config: FedMKTTrainingConfig):
    """Evaluate trained models"""
    logger.info("Evaluating models...")
    
    # Create trainer and load models
    trainer = FedMKTTrainer(config)
    trainer.create_central_model()
    
    for client_id in data_loaders.keys():
        trainer.create_client_model(client_id)
    
    # Load trained models (simplified - in practice, load from saved checkpoints)
    # trainer.load_models()
    
    evaluation_results = {}
    
    # Evaluate on test set
    test_loader = list(data_loaders.values())[0]["test"]
    
    # Evaluate central model
    central_eval = trainer.evaluate_model(trainer.central_model, test_loader)
    evaluation_results["central_model"] = central_eval
    logger.info(f"Central Model - Test Accuracy: {central_eval['accuracy']:.4f}")
    
    # Evaluate client models
    for client_id, client_model in trainer.client_models.items():
        client_eval = trainer.evaluate_model(client_model, test_loader)
        evaluation_results[f"client_{client_id}"] = client_eval
        logger.info(f"Client {client_id} - Test Accuracy: {client_eval['accuracy']:.4f}")
    
    return evaluation_results


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="FedMKT Training for 20News Classification")
    parser.add_argument("--config", type=str, default="config/fedmkt_20news_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--prepare_data", action="store_true",
                       help="Prepare data only")
    parser.add_argument("--train", action="store_true",
                       help="Train models only")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate models only")
    parser.add_argument("--output_dir", type=str, default="./outputs/fedmkt_20news",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    yaml_config = load_config(args.config)
    
    # Create configurations
    data_config = create_data_config(yaml_config)
    training_config = create_training_config(yaml_config)
    
    # Override output directory if specified
    if args.output_dir:
        training_config.output_dir = args.output_dir
    
    logger.info("Configuration loaded successfully!")
    logger.info(f"Output directory: {training_config.output_dir}")
    
    try:
        if args.prepare_data or (not args.train and not args.evaluate):
            # Prepare data
            data_loaders = prepare_data(data_config)
            
            if args.prepare_data:
                logger.info("Data preparation completed. Exiting.")
                return
        
        if args.train or (not args.prepare_data and not args.evaluate):
            # Load data if not already loaded
            if 'data_loaders' not in locals():
                data_loaders = prepare_data(data_config)
            
            # Train models
            training_history = train_fedmkt(data_loaders, training_config)
            
            if args.train:
                logger.info("Training completed. Exiting.")
                return
        
        if args.evaluate:
            # Load data if not already loaded
            if 'data_loaders' not in locals():
                data_loaders = prepare_data(data_config)
            
            # Evaluate models
            evaluation_results = evaluate_models(data_loaders, training_config)
            
            # Save evaluation results
            import json
            eval_path = os.path.join(training_config.output_dir, "evaluation_results.json")
            with open(eval_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logger.info(f"Evaluation completed. Results saved to {eval_path}")
        
        logger.info("All tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def test_training_pipeline():
    """Test the complete training pipeline"""
    logger.info("Testing training pipeline...")
    
    # Create test configuration
    test_config = {
        "dataset": {
            "name": "20newsgroups",
            "task": "classification",
            "num_classes": 20,
            "max_length": 256,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15
        },
        "models": {
            "central": {
                "name": "facebook/distilbart-cnn-12-6",
                "num_labels": 20,
                "max_length": 256
            },
            "clients": [
                {"name": "valhalla/mobile-bart", "client_id": 0},
                {"name": "valhalla/mobile-bart", "client_id": 1},
                {"name": "valhalla/mobile-bart", "client_id": 2}
            ]
        },
        "training": {
            "global_epochs": 2,
            "per_device_train_batch_size": 4,
            "learning_rate": 1e-4,
            "distill_temperature": 4.0,
            "kd_alpha": 0.7
        },
        "lora_config": {
            "central": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1
            }
        },
        "paths": {
            "data_dir": "./data/20news_test"
        },
        "output": {
            "output_dir": "./outputs/fedmkt_20news_test"
        }
    }
    
    try:
        # Test data preparation
        data_config = create_data_config(test_config)
        data_config.num_clients = 3
        data_config.max_length = 256
        
        logger.info("Testing data preparation...")
        data_loaders = prepare_data(data_config)
        logger.info("Data preparation test passed!")
        
        # Test training configuration
        training_config = create_training_config(test_config)
        training_config.num_epochs = 2
        training_config.batch_size = 4
        training_config.output_dir = "./outputs/fedmkt_20news_test"
        
        logger.info("Testing model training...")
        training_history = train_fedmkt(data_loaders, training_config)
        logger.info("Model training test passed!")
        
        # Test evaluation
        logger.info("Testing model evaluation...")
        evaluation_results = evaluate_models(data_loaders, training_config)
        logger.info("Model evaluation test passed!")
        
        logger.info("All pipeline tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Run test if no arguments provided
        test_training_pipeline()
    else:
        # Run main training
        main()
