"""
Training utilities for federated BERT learning.
"""

import torch
import numpy as np
import random
import logging
import os
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed}")


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for training"""
    
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logging.info(f"CUDA available: Using GPU {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logging.info("CUDA not available: Using CPU")
    
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count total and trainable parameters in a model"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """Print model information"""
    
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n{model_name} Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   round_num: int,
                   metrics: dict,
                   checkpoint_path: str):
    """Save training checkpoint"""
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "round": round_num,
        "metrics": metrics
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """Load training checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics for logging"""
    
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{prefix}{key}: {value:.4f}")
        else:
            formatted.append(f"{prefix}{key}: {value}")
    
    return ", ".join(formatted)


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_model_size(model: torch.nn.Module) -> float:
    """Calculate model size in MB"""
    
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


if __name__ == "__main__":
    # Test utilities
    setup_logging("INFO")
    set_seed(42)
    device = get_device("auto")
    
    print(f"Device: {device}")
    print("Training utilities test completed successfully")
