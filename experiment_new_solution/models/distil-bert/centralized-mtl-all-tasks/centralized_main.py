#!/usr/bin/env python3
"""
Centralized Training Script for Multi-Task Learning
MiniLM-L6-H384-uncased with Full Fine-Tuning - SST2, QQP, STSB
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

from src.models.distilbert_mtl_model import DistilBERTMTLModel
from src.datasets.mtl_dataset import MTLDataset
from src.training.mtl_trainer import CentralizedMTLTrainer
from src.utils.logger import setup_logger
from src.utils.metrics import compute_sst2_metrics, compute_qqp_metrics, compute_stsb_metrics

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
        name="centralized_mtl",
        level=config["output"]["log_level"],
        log_dir=config["output"]["results_dir"]
    )
    
    logger.info("=" * 60)
    logger.info("CENTRALIZED TRAINING - MULTI-TASK LEARNING")
    logger.info("=" * 60)
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Tasks: SST2, QQP, STSB")
    
    for task, task_config in config['task_configs'].items():
        logger.info(f"{task.upper()}: {task_config['train_samples']} train + {task_config['val_samples']} val samples")
    
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    
    # Set random seed
    torch.manual_seed(config['training']['random_seed'])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Additional CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA not available - training will use CPU!")
        logger.warning("This will be much slower than GPU training")
    
    # Initialize model
    logger.info("Initializing MTL model...")
    model = DistilBERTMTLModel(
        model_name=config['model']['model_name'],
        task_configs=config['task_configs'],
        max_length=config['data']['max_length']
    )
    model.to(device)
    
    # Initialize dataset
    logger.info("Loading datasets...")
    dataset = MTLDataset(
        task_configs=config['task_configs'],
        max_length=config['data']['max_length'],
        cache_dir=config['paths']['cache_dir']
    )
    
    # Prepare data
    train_loaders, val_loaders, test_loaders = dataset.prepare_data(
        batch_size=config['training']['batch_size'],
        random_seed=config['training']['random_seed']
    )
    
    # Initialize trainer
    logger.info("Initializing MTL trainer...")
    metrics_fns = {
        'sst2': compute_sst2_metrics,
        'qqp': compute_qqp_metrics,
        'stsb': compute_stsb_metrics
    }
    
    trainer = CentralizedMTLTrainer(
        model=model,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        device=device,
        config=config,
        metrics_fns=metrics_fns,
        logger=logger
    )
    
    # Start training
    logger.info("Starting MTL training...")
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
        logger.info("MTL TRAINING COMPLETED")
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
        logger.error(f"MTL training failed: {e}")
        raise
    
    logger.info("Centralized MTL training completed successfully!")

if __name__ == "__main__":
    main()
