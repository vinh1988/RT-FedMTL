#!/usr/bin/env python3
"""
Simple example script to run federated BERT learning.
This demonstrates the basic usage without complex configuration.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.server.flower_server import create_flower_server
from src.clients.flower_client import client_fn
from src.utils.training_utils import setup_logging, set_seed


def main():
    """Run a simple federated learning experiment"""
    
    # Setup
    setup_logging("INFO")
    set_seed(42)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting simple federated BERT experiment")
    
    # Configuration
    NUM_CLIENTS = 3
    NUM_ROUNDS = 2
    ENABLE_KNOWLEDGE_TRANSFER = False  # Disable for simple test
    
    # Create server
    server = create_flower_server(
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        enable_knowledge_transfer=ENABLE_KNOWLEDGE_TRANSFER
    )
    
    logger.info(f"Created server with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds")
    logger.info(f"Knowledge transfer: {'Enabled' if ENABLE_KNOWLEDGE_TRANSFER else 'Disabled'}")
    
    # Run simulation
    try:
        history = server.simulate_federated_learning(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS
        )
        
        logger.info("Experiment completed successfully!")
        
        # Print basic results
        if history and hasattr(history, 'metrics_distributed'):
            logger.info("Training completed. Check logs for detailed metrics.")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
