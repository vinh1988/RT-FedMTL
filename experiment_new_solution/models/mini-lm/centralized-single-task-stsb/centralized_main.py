#!/usr/bin/env python3
"""
Centralized Training Script for Single Task STSB
MiniLM-L6-H384-uncased with Full Fine-Tuning
"""

import os
import sys
import yaml
import torch
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.minilm_model import MiniLMModel
from src.datasets.stsb_dataset import STSBDataset
from src.training.trainer import CentralizedTrainer
from src.utils.logger import setup_logger
from src.utils.metrics import compute_stsb_metrics

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main training function"""
    # Setup
    config_path = "centralized_config.yaml"
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logger(
        name="centralized_stsb",
        level=config["output"]["log_level"],
        log_dir=config["output"]["results_dir"]
    )
    
    logger.info("=" * 60)
    logger.info("CENTRALIZED TRAINING - STSB SINGLE TASK")
    logger.info("=" * 60)
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Dataset: {config['training']['dataset']}")
    logger.info(f"Training Samples: {config['training']['train_samples']}")
    logger.info(f"Validation Samples: {config['training']['val_samples']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    
    # Set random seed
    torch.manual_seed(config['training']['random_seed'])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = MiniLMModel(
        model_name=config['model']['model_name'],
        num_labels=1,  # STSB is regression (1 output)
        max_length=config['data']['max_length']
    )
    model.to(device)
    
    # Initialize dataset
    logger.info("Loading dataset...")
    dataset = STSBDataset(
        max_length=config['data']['max_length'],
        cache_dir=config['paths']['cache_dir']
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = dataset.prepare_data(
        train_samples=config['training']['train_samples'],
        val_samples=config['training']['val_samples'],
        test_samples=config['training']['test_samples'],
        batch_size=config['training']['batch_size'],
        random_seed=config['training']['random_seed']
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = CentralizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=config,
        metrics_fn=compute_stsb_metrics,
        logger=logger
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        # Final evaluation
        logger.info("Running final evaluation...")
        test_results = trainer.evaluate()
        
        # Log results
        end_time = datetime.now()
        training_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total Training Time: {training_time}")
        logger.info(f"Test Results: {test_results}")
        
        # Save final results
        results_path = Path(config['output']['results_dir']) / "final_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump({
                'config': config,
                'training_time': str(training_time),
                'test_results': test_results,
                'timestamp': datetime.now().isoformat()
            }, f, default_flow_style=False)
        
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("Centralized STSB training completed successfully!")

if __name__ == "__main__":
    main()
