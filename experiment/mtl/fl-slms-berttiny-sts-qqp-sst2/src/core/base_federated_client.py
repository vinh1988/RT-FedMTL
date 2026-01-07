#!/usr/bin/env python3
"""
Base Federated Learning Client Implementation
Contains shared functionality for all specialized federated clients
(Standard FL - No LoRA, No KD)
"""

import asyncio
import json
import logging
import time
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from federated_config import FederatedConfig
from src.models.standard_bert import StandardBERTModel
from src.communication.federated_websockets import WebSocketClient, MessageProtocol
from src.synchronization.federated_synchronization import ClientModelSynchronizer
from src.datasets.federated_datasets import DatasetFactory, DatasetConfig

logger = logging.getLogger(__name__)

class BaseFederatedClient(ABC):
    """Base class for federated learning clients with shared functionality"""

    def __init__(self, client_id: str, task: str, config: FederatedConfig):
        self.client_id = client_id
        self.task = task  # Single task for specialized clients
        self.tasks = [task]  # Keep as list for compatibility
        self.config = config
        self.device = self.get_device()

        # Initialize standard BERT model (full fine-tuning)
        self.model = StandardBERTModel(
            base_model_name=config.client_model,
            tasks=self.tasks
        )
        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize optimizer for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=0.9
        )

        # Initialize synchronization (with task for MTL)
        self.websocket_client = WebSocketClient(
            f"ws://localhost:{config.port}",
            client_id
        )
        self.model_synchronizer = ClientModelSynchronizer(
            self.model, self.websocket_client, self.task
        )

        # Initialize datasets
        self.dataset_handler = self.initialize_dataset_handler()

        # Setup logging
        self.setup_logging()

        # Training state
        self.is_training = False
        self.current_round = 0

    def get_device(self):
        """Get available device (GPU/CPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'federated_client_{self.client_id}.log'),
                logging.StreamHandler()
            ]
        )

    def initialize_dataset_handler(self):
        """Initialize dataset handler for the specific task"""
        if self.task in self.config.task_configs:
            config = self.config.task_configs[self.task]
            dataset_config = DatasetConfig(
                task_name=self.task,
                train_samples=config.get('train_samples'),
                val_samples=config.get('val_samples'),
                random_seed=config.get('random_seed', 42)
            )
            return DatasetFactory.create_handler(self.task, dataset_config)
        return None

    async def connect_and_register(self):
        """Connect to server and register"""
        await self.websocket_client.connect()

        # Register with server
        registration_message = MessageProtocol.create_registration_message(
            self.client_id,
            self.tasks,
            {
                "model": self.config.client_model,
                "supported_tasks": self.tasks
            }
        )

        await self.websocket_client.send(registration_message)
        logger.info(f"Client {self.client_id} registered with task: {self.task}")

    async def run_client(self):
        """Main client execution loop"""
        try:
            # Connect and register
            await self.connect_and_register()

            # Setup message handlers
            self.setup_message_handlers()

            # Keep connection alive
            while self.websocket_client.is_connected:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Client shutting down...")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            await self.websocket_client.disconnect()

    def setup_message_handlers(self):
        """Setup WebSocket message handlers"""
        self.websocket_client.register_message_handler(
            "global_model_sync", self.handle_global_model_sync
        )
        self.websocket_client.register_message_handler(
            "training_request", self.handle_training_request
        )
        self.websocket_client.register_message_handler(
            "registration_ack", self.handle_registration_ack
        )
        self.websocket_client.register_message_handler(
            "heartbeat", self.handle_heartbeat
        )

    async def handle_global_model_sync(self, data: Dict):
        """Handle incoming global model synchronization"""
        logger.info(f"Received global model synchronization")

        # Update local model with global knowledge
        sync_result = await self.model_synchronizer.synchronize_with_global_model(
            data["global_model_state"]
        )

        # Send acknowledgment
        await self.model_synchronizer.send_synchronization_acknowledgment(sync_result)

    async def handle_registration_ack(self, data: Dict):
        """Handle registration acknowledgment from server"""
        accepted = data.get("accepted", False)
        message = data.get("message", "")

        if accepted:
            logger.info(f"Registration acknowledged: {message}")
        else:
            logger.warning(f"Registration rejected: {message}")

    async def handle_heartbeat(self, data: Dict):
        """Handle heartbeat messages to keep connection alive"""
        # Send heartbeat response to keep connection alive
        heartbeat_response = MessageProtocol.create_heartbeat_message(self.client_id)
        await self.websocket_client.send(heartbeat_response)

    async def handle_training_request(self, data: Dict):
        """Handle training request from server"""
        round_num = data["round"]
        global_params = data.get("global_params", {})

        logger.info(f"Received training request for round {round_num}")
        logger.info(f"Starting training for client {self.client_id} with task: {self.task}")

        # Perform local training
        self.is_training = True
        self.current_round = round_num

        try:
            local_metrics = await self.perform_local_training()

            # Extract model parameters
            model_updates = self.model.get_all_parameters()

            # Send update to server (include task label for MTL aggregation)
            update_message = MessageProtocol.create_client_update_message(
                self.client_id,
                round_num,
                model_updates,
                local_metrics,
                {}
            )
            # Add task label for MTL-aware aggregation
            update_message['task'] = self.task

            # Use retry logic for sending updates
            success = await self.websocket_client.send(update_message, max_retries=5)
            if success:
                logger.info(f"[SUCCESS] Training completed and update sent for round {round_num}")
                logger.info(f"[METRICS] Client {self.client_id} metrics: {local_metrics}")
            else:
                logger.error(f"[ERROR] Failed to send update for round {round_num} after retries")

        except Exception as e:
            logger.error(f"Error in local training for round {round_num}: {e}")
        finally:
            self.is_training = False

    async def perform_local_training(self) -> Dict[str, float]:
        """Perform local training"""
        if not self.dataset_handler:
            logger.error(f"No dataset handler available for task {self.task}")
            return {}

        # Get data for this task
        task_data = self.dataset_handler.prepare_data()

        # Train on this task
        task_metrics = await self.train_task(self.task, task_data)
        return {self.task: task_metrics}

    @abstractmethod
    async def train_task(self, task: str, task_data: Dict) -> Dict[str, float]:
        """Train on a specific task - implemented by specialized clients"""
        pass

    def extract_model_updates(self) -> Dict[str, Dict]:
        """Extract model parameters for federated aggregation"""
        return self.model.get_all_parameters()

    def get_client_status(self) -> Dict:
        """Get current client status"""
        return {
            "client_id": self.client_id,
            "task": self.task,
            "is_connected": self.websocket_client.is_connected,
            "is_training": self.is_training,
            "current_round": self.current_round,
            "model_synchronized": self.model_synchronizer.is_synchronized,
            "dataset_available": self.dataset_handler is not None
        }

def run_base_client(client: BaseFederatedClient):
    """Run a federated learning client"""
    asyncio.run(client.run_client())
