"""
Logger utility for centralized training
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
