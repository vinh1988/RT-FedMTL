#!/usr/bin/env python3
"""
Federated Learning Server Implementation
Orchestrates training, manages clients, and handles synchronization
"""

import asyncio
import json
import logging
import time
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch

from federated_config import FederatedConfig
from src.lora.federated_lora import LoRAAggregator
from src.peft.federated_peft_lora import PEFTAggregator
from src.knowledge_distillation.federated_knowledge_distillation import GlobalKDManager
from src.communication.federated_websockets import WebSocketServer, MessageProtocol
from src.synchronization.federated_synchronization import SynchronizationManager

logger = logging.getLogger(__name__)

class FederatedServer:
    """Federated Learning Server with LoRA, KD, and synchronization support"""

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.connected_clients: Dict[str, Dict] = {}
        self.client_updates: Dict[int, List[Dict]] = {}
        self.global_lora_params: Dict[str, Dict] = {}
        self.global_teacher_logits: Dict[str, torch.Tensor] = {}
        
        # Data loaders for teacher fine-tuning (will be set when available)
        self.teacher_train_loader = None
        self.teacher_eval_loader = None
        self.metric_fn = None  # Function to compute evaluation metric

        # Initialize components
        self.lora_aggregator = PEFTAggregator()
        
        # Initialize teacher model and KD manager
        teacher_model = None
        if hasattr(config, 'teacher_model_name'):
            from transformers import AutoModelForSequenceClassification
            try:
                teacher_model = AutoModelForSequenceClassification.from_pretrained(
                    config.teacher_model_name,
                    num_labels=config.num_labels if hasattr(config, 'num_labels') else 2,
                    output_attentions=False,
                    output_hidden_states=False
                )
                logger.info(f"Initialized teacher model: {config.teacher_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize teacher model: {e}")
        
        # Initialize KD manager with teacher model
        self.kd_manager = GlobalKDManager(teacher_model, config)
        
        # Setup WebSocket server and synchronization manager
        self.websocket_server = WebSocketServer(config.port)
        self.synchronization_manager = SynchronizationManager(self)

        # Setup results management and logging
        self.setup_results_management()
        self.setup_logging()
        
        logger.info("Federated server initialized with TeacherTrainer integration")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'federated_server_{self.config.port}.log'),
                logging.StreamHandler()
            ]
        )

    def setup_results_management(self):
        """Setup CSV files for results and model checkpoints"""
        # Create results directory if it doesn't exist
        results_dir = os.path.join(self.config.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create checkpoints directory for teacher model
        self.teacher_checkpoint_dir = os.path.join(self.config.output_dir, 'teacher_checkpoints')
        os.makedirs(self.teacher_checkpoint_dir, exist_ok=True)
        
        # Initialize client results CSV
        self.client_results_file = os.path.join(results_dir, 'client_results.csv')
        self._init_csv_file()
        
        # Initialize teacher metrics CSV
        self.teacher_metrics_file = os.path.join(results_dir, 'teacher_metrics.csv')
        self._init_teacher_metrics_file()
        
        # Initialize server metrics CSV
        self.server_metrics_file = os.path.join(results_dir, 'server_metrics.csv')
        self.csv_filename = self.server_metrics_file  # For backward compatibility
        self._init_server_metrics_file()
        
        # Set client CSV filename for logging
        self.client_csv_filename = self.client_results_file
        
        # Initialize CSV writer for server metrics
        self.csv_file = open(self.server_metrics_file, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Initialize client CSV writer
        self.client_csv_file = open(self.client_results_file, 'a', newline='')
        self.client_csv_writer = csv.writer(self.client_csv_file)
        
        # Write headers if files are empty
        if os.path.getsize(self.server_metrics_file) == 0:
            self.csv_writer.writerow(['round', 'num_clients', 'duration', 'model_version', 'timestamp'])
            self.csv_file.flush()
            
        if os.path.getsize(self.client_results_file) == 0:
            self.client_csv_writer.writerow([
                'round', 'client_id', 'task', 'accuracy', 'loss', 'samples_processed', 
                'correct_predictions', 'f1_score', 'pearson_correlation', 'spearman_correlation',
                'mae', 'mse', 'rmse', 'val_accuracy', 'val_loss', 'val_samples',
                'val_correct_predictions', 'val_f1_score', 'val_pearson_correlation',
                'val_spearman_correlation', 'val_mae', 'timestamp'
            ])
            self.client_csv_file.flush()
    
    def _init_server_metrics_file(self):
        """Initialize the server metrics CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.server_metrics_file) or os.path.getsize(self.server_metrics_file) == 0:
            with open(self.server_metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'round', 'num_clients', 'duration', 'model_version',
                    'avg_train_loss', 'avg_val_loss', 'avg_train_acc', 'avg_val_acc',
                    'num_samples', 'timestamp'
                ])
            logger.info("Initialized server metrics file with headers")
    
    def __del__(self):
        """Cleanup resources when the server is destroyed"""
        # Close server metrics file
        if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception as e:
                logger.error(f"Error closing server metrics CSV file: {e}")
                
        # Close client results file
        if hasattr(self, 'client_csv_file') and self.client_csv_file and not self.client_csv_file.closed:
            try:
                self.client_csv_file.flush()
                self.client_csv_file.close()
            except Exception as e:
                logger.error(f"Error closing client results CSV file: {e}")

    def _init_teacher_metrics_file(self):
        """Initialize CSV file for teacher training metrics"""
        if not os.path.exists(self.teacher_metrics_file):
            with open(self.teacher_metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'round', 'epoch', 'train_loss', 'eval_loss', 'metric_value',
                    'learning_rate', 'kd_alpha', 'temperature'
                ])

    def _init_csv_file(self):
        """Initialize CSV file for client results"""
        if not os.path.exists(self.client_results_file):
            with open(self.client_results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round", "client_id", "task", "accuracy", "loss", "samples_processed",
                    "correct_predictions", "f1_score", "pearson_correlation", "spearman_correlation",
                    "mae", "mse", "rmse",
                    "val_accuracy", "val_loss", "val_samples", "val_correct_predictions",
                    "val_f1_score", "val_pearson_correlation", "val_spearman_correlation", "val_mae",
                    "timestamp"
                ])

    def record_client_results(self, round_num: int, client_id: str, client_metrics: Dict):
        """Record individual client results"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Process each task's metrics for this client
        for task_name, task_metrics in client_metrics.items():
            # Skip non-task entries like student_knowledge
            if task_name == "student_knowledge":
                continue
                
            if isinstance(task_metrics, dict):
                row = [
                    round_num,
                    client_id,
                    task_name,
                    task_metrics.get('accuracy', 0.0),
                    task_metrics.get('loss', 0.0),
                    task_metrics.get('samples_processed', 0),
                    task_metrics.get('correct_predictions', 0),
                    # Classification metrics (F1)
                    task_metrics.get('f1_score', ''),
                    # Regression metrics (Pearson, Spearman, MAE, MSE, RMSE)
                    task_metrics.get('pearson_correlation', ''),
                    task_metrics.get('spearman_correlation', ''),
                    task_metrics.get('mae', ''),
                    task_metrics.get('mse', ''),
                    task_metrics.get('rmse', ''),
                    # Validation metrics
                    task_metrics.get('val_accuracy', 0.0),
                    task_metrics.get('val_loss', 0.0),
                    task_metrics.get('val_samples', 0),
                    task_metrics.get('val_correct_predictions', 0),
                    task_metrics.get('val_f1_score', ''),
                    task_metrics.get('val_pearson_correlation', ''),
                    task_metrics.get('val_spearman_correlation', ''),
                    task_metrics.get('val_mae', ''),
                    timestamp
                ]
                with open(self.client_results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
        
    async def client_handler(self, websocket):
        """Handle individual client connections"""
        client_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                # Deserialize tensors in received messages
                data = MessageProtocol.deserialize_tensors(data, device="cpu")
                message_type = data.get("type")

                if message_type == MessageProtocol.CLIENT_REGISTER:
                    client_id = await self.handle_client_registration(websocket, data)
                elif message_type == MessageProtocol.CLIENT_UPDATE:
                    await self.handle_client_update(websocket, data)
                elif message_type == MessageProtocol.SYNC_ACKNOWLEDGMENT:
                    await self.handle_sync_acknowledgment(websocket, data)
                elif message_type == MessageProtocol.HEARTBEAT:
                    await self.handle_heartbeat(websocket, data)

        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            if client_id and client_id in self.connected_clients:
                logger.info(f"Client {client_id} disconnected")
                del self.connected_clients[client_id]
                # Also remove from WebSocket server clients
                if client_id in self.websocket_server.clients:
                    del self.websocket_server.clients[client_id]

    async def handle_client_registration(self, websocket, data: Dict) -> str:
        """Handle client registration"""
        client_id = data["client_id"]
        tasks = data.get("tasks", [])
        capabilities = data.get("capabilities", {})

        # Register client in WebSocket server
        self.websocket_server.clients[client_id] = websocket

        # Register client in federated server
        self.connected_clients[client_id] = {
            "websocket": websocket,
            "tasks": tasks,
            "capabilities": capabilities,
            "last_seen": time.time(),
            "registered_at": datetime.now().isoformat()
        }

        logger.info(f"Client {client_id} registered with tasks: {tasks}")

        # Send acknowledgment
        response = {
            "type": "registration_ack",
            "client_id": client_id,
            "accepted": True,
            "message": f"Registered successfully with {len(tasks)} tasks"
        }
        await websocket.send(json.dumps(response))

        return client_id

    async def handle_client_update(self, websocket, data: Dict):
        """Handle client model updates"""
        client_id = data["client_id"]
        round_num = data["round"]
        lora_updates = data.get("lora_updates", {})
        metrics = data.get("metrics", {})
        student_knowledge = data.get("student_knowledge", {})

        # Store client update
        if round_num not in self.client_updates:
            self.client_updates[round_num] = []

        self.client_updates[round_num].append({
            "client_id": client_id,
            "lora_updates": lora_updates,
            "metrics": metrics,
            "student_knowledge": student_knowledge,
            "received_at": datetime.now().isoformat()
        })

        # Record individual client results
        self.record_client_results(round_num, client_id, metrics)

        # Update client last seen
        if client_id in self.connected_clients:
            self.connected_clients[client_id]["last_seen"] = time.time()

        logger.info(f"Received update from client {client_id} for round {round_num}")

    async def handle_sync_acknowledgment(self, websocket, data: Dict):
        """Handle synchronization acknowledgment from clients"""
        client_id = data["client_id"]
        synchronized = data.get("synchronized", False)

        logger.info(f"Client {client_id} synchronization acknowledgment: {synchronized}")

    async def handle_heartbeat(self, websocket, data: Dict):
        """Handle heartbeat messages"""
        client_id = data["client_id"]

        # Update client last seen
        if client_id in self.connected_clients:
            self.connected_clients[client_id]["last_seen"] = time.time()

    async def wait_for_clients(self) -> bool:
        """Wait for all expected clients to connect before starting training"""
        # For federated learning, we typically want all clients to participate
        # Use expected_clients from config, fallback to min_clients
        expected_clients = getattr(self.config, 'expected_clients', self.config.min_clients)
        
        # If we have fewer clients than expected, wait for more to connect
        # Only proceed if we've been waiting too long (timeout)
        if expected_clients > len(self.connected_clients):
            logger.info(f"Waiting for more clients to connect (have {len(self.connected_clients)}, need {expected_clients})")
        
        logger.info(f"Waiting for clients... (need {expected_clients}, currently {len(self.connected_clients)})")

        start_time = time.time()
        max_wait_time = 120  # Wait up to 2 minutes for all clients
        
        while len(self.connected_clients) < expected_clients:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.warning(f"Timeout waiting for all clients after {elapsed:.0f}s")
                logger.warning(f"Proceeding with {len(self.connected_clients)} clients: {list(self.connected_clients.keys())}")
                break

            logger.info(f"Waiting for clients... ({len(self.connected_clients)}/{expected_clients})")
            logger.info(f"Connected clients: {list(self.connected_clients.keys())}")
            await asyncio.sleep(2)

        # Only proceed if we have the expected number of clients or timeout occurred
        if len(self.connected_clients) >= expected_clients:
            logger.info(f"Starting training with {len(self.connected_clients)} clients: {list(self.connected_clients.keys())}")
            return True
        elif len(self.connected_clients) >= self.config.min_clients:
            logger.warning(f"Starting with {len(self.connected_clients)} clients (expected {expected_clients}): {list(self.connected_clients.keys())}")
            return True
        else:
            logger.error(f"Not enough clients connected ({len(self.connected_clients)}/{self.config.min_clients})")
            return False

    async def run_federated_training(self):
        """Main federated training loop with synchronization"""
        logger.info("Starting federated training with LoRA, KD, and synchronization")

        # Wait for clients
        if not await self.wait_for_clients():
            logger.error("Insufficient clients for training")
            return

        # Training rounds with bidirectional synchronization
        for round_num in range(1, self.config.num_rounds + 1):
            logger.info(f"=== Round {round_num}/{self.config.num_rounds} ===")
            round_start = time.time()

            try:
                # Step 1: Send current global model to clients for synchronization
                if self.config.enable_synchronization:
                    current_global_state = self.synchronization_manager.get_global_model_state()
                    await self.synchronization_manager.broadcast_global_model(current_global_state)

                    # Wait for clients to acknowledge synchronization
                    await self.wait_for_sync_acknowledgments(round_num)

                # Step 2: Generate teacher knowledge for this round
                sample_inputs = self.generate_sample_inputs_for_kd()
                teacher_knowledge = self.kd_manager.generate_teacher_knowledge(sample_inputs)

                # Convert teacher knowledge tensors to serializable format
                serializable_teacher_knowledge = {}
                for task, tensor in teacher_knowledge.items():
                    if isinstance(tensor, torch.Tensor):
                        serializable_teacher_knowledge[task] = tensor.tolist()
                    else:
                        serializable_teacher_knowledge[task] = tensor

                # Step 3: Send training request to clients
                training_request = MessageProtocol.create_training_request_message(
                    round_num, self.global_lora_params, serializable_teacher_knowledge
                )
                await self.websocket_server.broadcast(training_request)

                # Step 4: Wait for client updates
                await self.collect_client_updates(round_num)

                # Step 5: Aggregate LoRA updates from clients
                if round_num in self.client_updates and self.client_updates[round_num]:
                    client_updates = self.client_updates[round_num]
                    aggregated_lora = self.lora_aggregator.aggregate_lora_updates(client_updates)
                    self.synchronization_manager.update_global_model_from_aggregation(aggregated_lora)

                # Step 6: Update teacher model with student knowledge (reverse KD)
                if round_num in self.client_updates and self.client_updates[round_num]:
                    student_knowledge = [
                        update.get('student_knowledge', {})
                        for update in self.client_updates[round_num]
                    ]
                    
                    # Update teacher model with student knowledge
                    update_result = self.kd_manager.update_teacher_from_students(
                        student_knowledge_updates=student_knowledge,
                        train_loader=self.teacher_train_loader,
                        eval_loader=self.teacher_eval_loader,
                        metric_fn=self.metric_fn,
                        metric_name=self.config.metric_for_best_model
                    )
                    
                    if update_result.get('updated', False):
                        if update_result.get('fine_tuned', False):
                            logger.info(f"Teacher model fine-tuned with student knowledge. Best metric: {update_result.get('best_metric', 'N/A')}")
                        else:
                            logger.info(f"Teacher model updated with student knowledge. Avg loss: {update_result.get('avg_reverse_loss', 'N/A')}")
                    else:
                        logger.warning(f"Teacher model update failed: {update_result.get('reason', 'Unknown error')}")

                # Step 7: Record results for this round
                self.record_round_results(round_num, round_start)

                logger.info(f"Round {round_num} completed in {time.time() - round_start:.2f}s")

            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                continue

        # Finalize training
        self.finalize_training()

    async def wait_for_sync_acknowledgments(self, round_num: int, timeout: int = 120):
        """Wait for synchronization acknowledgments from clients"""
        start_time = time.time()
        acks_received = set()

        while time.time() - start_time < timeout:
            # Check for new acknowledgments (this would be implemented in the message handler)
            # For now, we'll assume clients acknowledge immediately
            acks_received.update(self.connected_clients.keys())

            if len(acks_received) >= len(self.connected_clients):
                logger.info("All clients synchronized")
                return True

            await asyncio.sleep(1)

        logger.warning(f"Timeout waiting for sync acknowledgments. Got {len(acks_received)}/{len(self.connected_clients)}")
        return False

    async def collect_client_updates(self, round_num: int, timeout: int = None):
        """Collect client updates for a round - wait for ALL connected clients"""
        if timeout is None:
            # Use config round_timeout, default to 3400 seconds (56.7 minutes) if not set
            timeout = getattr(self.config.communication, 'round_timeout', 3400)
        
        start_time = time.time()
        updates_received = 0
        
        # Use the expected clients from config, not current connected count
        expected_clients = getattr(self.config, 'expected_clients', len(self.connected_clients))
        
        if expected_clients == 0:
            logger.error("No clients expected!")
            return False

        logger.info(f"Waiting for client updates... (expecting ALL {expected_clients} clients)")
        logger.info(f"Currently connected clients: {list(self.connected_clients.keys())}")
        logger.info(f"Timeout set to {timeout} seconds")

        while time.time() - start_time < timeout:
            if round_num in self.client_updates:
                updates_received = len(self.client_updates[round_num])
                # Extract client IDs from the list of update dictionaries
                received_clients = [update.get('client_id', 'unknown') for update in self.client_updates[round_num]]
                missing_clients = [c for c in self.connected_clients.keys() if c not in received_clients]
                
                logger.info(f"Updates received: {updates_received}/{expected_clients}")
                logger.info(f"Received from: {received_clients}")
                if missing_clients:
                    logger.info(f"Still waiting for: {missing_clients}")

            # Wait for ALL connected clients to respond
            if updates_received >= expected_clients:
                logger.info(f"All client updates received ({updates_received}/{expected_clients})")
                return True

            # Check if any clients disconnected during waiting
            current_connected = len(self.connected_clients)
            if current_connected < expected_clients:
                logger.warning(f"Client count changed: {current_connected}/{expected_clients}")
                # Don't reduce expected_clients - we still want to wait for all originally expected clients
                if current_connected == 0:
                    logger.error("No clients connected!")
                    return False

            # Show progress every 30 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                logger.info(f"Still waiting... {elapsed:.0f}s elapsed, {updates_received}/{expected_clients} updates received")

            await asyncio.sleep(2)

        logger.warning(f"Timeout collecting updates. Got {updates_received}/{expected_clients}")
        if updates_received > 0:
            logger.warning(f"Proceeding with {updates_received} out of {expected_clients} clients")
            return True
        else:
            logger.error("No client updates received!")
            return False

    def generate_sample_inputs_for_kd(self):
        """
        Generate sample inputs for teacher knowledge generation.
        
        Returns:
            Dictionary mapping task names to sample input batches
        """
        sample_inputs = {}
        
        try:
            # Try to get sample inputs from the evaluation dataset if available
            if hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
                # Get a sample batch from the evaluation dataset
                sample = next(iter(self.teacher_eval_loader)) if self.teacher_eval_loader else None
                if sample:
                    for task in self.config.tasks:
                        if task in sample:
                            sample_inputs[task] = {
                                'input_ids': sample[task]['input_ids'][:1],  # Just take one example
                                'attention_mask': sample[task]['attention_mask'][:1],
                                'labels': sample[task]['labels'][:1] if 'labels' in sample[task] else None
                            }
                    
                    if sample_inputs:
                        return sample_inputs
            
            # Fallback to dummy data if no evaluation dataset is available
            logger.warning("No evaluation dataset available, using dummy data for KD")
            for task in self.config.tasks:
                # Create a dummy input with the expected format
                max_length = getattr(self.config, 'max_seq_length', 128)
                sample_inputs[task] = {
                    'input_ids': torch.randint(0, 1000, (1, max_length), dtype=torch.long),
                    'attention_mask': torch.ones((1, max_length), dtype=torch.long),
                    'labels': torch.zeros(1, dtype=torch.float32 if task == 'stsb' else torch.long)
                }
                
        except Exception as e:
            logger.error(f"Error generating sample inputs for KD: {e}")
            # Return minimal dummy data if anything goes wrong
            return {
                'dummy': {
                    'input_ids': torch.zeros((1, 32), dtype=torch.long),
                    'attention_mask': torch.ones((1, 32), dtype=torch.long),
                    'labels': torch.zeros(1, dtype=torch.float32)
                }
            }
            
        return sample_inputs

    def record_round_results(self, round_num: int, round_start: float):
        """Record results for a training round"""
        round_time = time.time() - round_start
        logger.info(f"Round {round_num} completed in {round_time:.2f} seconds")
        
        metrics_dict = {
            'round_time': round_time,
            'num_clients': len(self.connected_clients)
        }
        
        # Check and add teacher metrics if available
        if hasattr(self, 'kd_manager') and self.kd_manager is not None:
            if hasattr(self.kd_manager, 'teacher_trainer') and self.kd_manager.teacher_trainer is not None:
                if hasattr(self.kd_manager.teacher_trainer, 'latest_metrics'):
                    latest_metrics = self.kd_manager.teacher_trainer.latest_metrics
                    if latest_metrics:  # Only add if metrics exist
                        metrics_dict['teacher_metrics'] = latest_metrics
                        logger.info(f"[DEBUG] Added teacher_metrics to metrics_dict: {latest_metrics}")
                    else:
                        logger.warning("[DEBUG] Teacher trainer's latest_metrics is empty")
                else:
                    logger.warning("[DEBUG] Teacher trainer has no latest_metrics attribute")
            else:
                logger.warning("[DEBUG] Teacher trainer is not available in kd_manager")
        else:
            logger.warning("[DEBUG] kd_manager is not available")
        
        # Always log round metrics, with or without teacher metrics
        if hasattr(self, 'csv_writer') and self.csv_writer is not None:
            self.log_round_metrics(
                round_num=round_num,
                num_clients=len(self.connected_clients),
                duration=round_time,
                model_version=self.synchronization_manager.global_model_version,
                metrics=metrics_dict
            )
                    
    def log_round_metrics(self, round_num: int, num_clients: int, duration: float, model_version: str, metrics: Dict = None):
        """
        Log metrics for the current round
        
        Args:
            round_num: Current round number
            num_clients: Number of clients in this round
            duration: Duration of the round in seconds
            model_version: Current model version
            metrics: Dictionary containing additional metrics to log (optional)
        """
        if not hasattr(self, 'csv_writer') or not self.csv_writer:
            logger.warning("CSV writer not initialized, skipping metrics logging")
            return
            
        try:
            # Default values
            metrics = metrics or {}
            
            # Calculate average metrics from client updates if available
            if round_num in self.client_updates:
                updates = self.client_updates[round_num]
                if updates:
                    # Calculate average metrics
                    avg_train_loss = sum(update.get('metrics', {}).get('loss', 0) for update in updates) / len(updates)
                    avg_val_loss = sum(update.get('metrics', {}).get('val_loss', 0) for update in updates) / len(updates)
                    avg_train_acc = sum(update.get('metrics', {}).get('accuracy', 0) for update in updates) / len(updates)
                    avg_val_acc = sum(update.get('metrics', {}).get('val_accuracy', 0) for update in updates) / len(updates)
                    total_samples = sum(update.get('metrics', {}).get('samples_processed', 0) for update in updates)
                    
                    # Update metrics with calculated values
                    metrics.update({
                        'avg_train_loss': avg_train_loss,
                        'avg_val_loss': avg_val_loss,
                        'avg_train_acc': avg_train_acc,
                        'avg_val_acc': avg_val_acc,
                        'num_samples': total_samples
                    })
            
            # Prepare the row with metrics
            row = [
                round_num,  # Round number
                num_clients,  # Number of clients
                f"{duration:.2f}",  # Duration in seconds
                model_version,  # Model version
                metrics.get('avg_train_loss', 0.0),  # Average training loss
                metrics.get('avg_val_loss', 0.0),  # Average validation loss
                metrics.get('avg_train_acc', 0.0),  # Average training accuracy
                metrics.get('avg_val_acc', 0.0),  # Average validation accuracy
                metrics.get('num_samples', 0),  # Total number of samples
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Timestamp
            ]
            
            # Write the row to the CSV file
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            
            logger.info(f"Logged metrics for round {round_num} with {num_clients} clients")
            
            # Log teacher metrics if available
            if 'teacher_metrics' in metrics:
                self._log_teacher_metrics(round_num, metrics['teacher_metrics'])
                
        except Exception as e:
            logger.error(f"Error logging round metrics: {e}")
            
    def _log_teacher_metrics(self, round_num, metrics):
        """Log teacher model metrics with enhanced debugging"""
        logger.info(f"[DEBUG] Starting _log_teacher_metrics for round {round_num}")
        
        # Ensure metrics is a dictionary
        if not isinstance(metrics, dict):
            logger.error(f"[ERROR] Expected metrics to be a dict, got {type(metrics)}: {metrics}")
            return
            
        logger.info(f"[DEBUG] Metrics received: {metrics}")
        
        try:
            # Initialize file and writer if they don't exist
            if not hasattr(self, 'teacher_metrics_file') or not hasattr(self, 'teacher_metrics_writer'):
                logger.info("[DEBUG] Initializing teacher_metrics_writer")
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(self.teacher_metrics_file), exist_ok=True)
                    
                    # Open file in append mode
                    self.teacher_metrics_file = open(self.teacher_metrics_file, 'a', newline='')
                    self.teacher_metrics_writer = csv.writer(self.teacher_metrics_file)
                    logger.info(f"[DEBUG] Opened teacher_metrics_file at {self.teacher_metrics_file}")
                    
                    # Write header if file is empty
                    file_size = os.path.getsize(self.teacher_metrics_file.name) if os.path.exists(self.teacher_metrics_file.name) else 0
                    logger.info(f"[DEBUG] Teacher metrics file size: {file_size} bytes")
                    
                    if file_size == 0:
                        headers = [
                            'round', 'epoch', 'train_loss', 'eval_loss', 'metric_value',
                            'learning_rate', 'kd_alpha', 'temperature', 'samples_used',
                            'update_type', 'timestamp'
                        ]
                        self.teacher_metrics_writer.writerow(headers)
                        self.teacher_metrics_file.flush()
                        logger.info("[DEBUG] Wrote headers to teacher_metrics_file")
                    else:
                        logger.info("[DEBUG] Teacher metrics file already has content, skipping header")
                        
                except Exception as e:
                    logger.error(f"[ERROR] Failed to initialize teacher metrics writer: {e}", exc_info=True)
                    return
            
            # Prepare the metrics row with default values
            row = [
                round_num,
                metrics.get('epoch', 0),
                metrics.get('train_loss', 0.0),
                metrics.get('eval_loss', 0.0),
                metrics.get('metric_value', 0.0),
                metrics.get('learning_rate', 0.0),
                metrics.get('kd_alpha', getattr(self.config, 'kd_alpha', 0.5)),
                metrics.get('temperature', getattr(self.config, 'temperature', 3.0)),
                metrics.get('samples_used', 0),
                metrics.get('update_type', 'none'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
            
            logger.info(f"[DEBUG] Prepared teacher metrics row: {row}")
            
            # Write the row
            try:
                self.teacher_metrics_writer.writerow(row)
                self.teacher_metrics_file.flush()
                logger.info(f"[SUCCESS] Logged teacher metrics for round {round_num}")
                
                # Verify file was written
                if os.path.exists(self.teacher_metrics_file.name):
                    with open(self.teacher_metrics_file.name, 'r') as f:
                        lines = f.readlines()
                        logger.info(f"[DEBUG] Teacher metrics file now has {len(lines)} lines")
                        if len(lines) > 1:  # Header + at least one data row
                            logger.info(f"[DEBUG] Last line written: {lines[-1].strip()}")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to write teacher metrics: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in _log_teacher_metrics: {e}", exc_info=True)
            
        logger.info(f"[DEBUG] Completed _log_teacher_metrics for round {round_num}")

    def calculate_aggregated_metrics(self, updates: List[Dict]) -> Dict:
        """Calculate aggregated metrics across all clients"""
        if not updates:
            return {
                'avg_accuracy': 0.0,
                'classification_accuracy': 0.0,
                'regression_accuracy': 0.0,
                'total_clients': 0,
                'active_clients': 0,
                'training_time': 0.0
            }

        # Extract metrics from all clients
        all_accuracies = []
        classification_accuracies = []
        regression_accuracies = []

        for update in updates:
            client_metrics = update.get('metrics', {})
            
            # Process task-specific metrics
            for task_name, task_metrics in client_metrics.items():
                if isinstance(task_metrics, dict) and 'accuracy' in task_metrics:
                    accuracy = task_metrics['accuracy']
                    all_accuracies.append(accuracy)
                    
                    # Categorize by task type
                    if task_name in ['sst2', 'qqp']:
                        classification_accuracies.append(accuracy)
                    elif task_name == 'stsb':
                        regression_accuracies.append(accuracy)

        # Calculate averages
        avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        classification_accuracy = sum(classification_accuracies) / len(classification_accuracies) if classification_accuracies else 0.0
        regression_accuracy = sum(regression_accuracies) / len(regression_accuracies) if regression_accuracies else 0.0

        return {
            'avg_accuracy': avg_accuracy,
            'classification_accuracy': classification_accuracy,
            'regression_accuracy': regression_accuracy,
            'total_clients': len(self.connected_clients),
            'active_clients': len(updates),
            'training_time': 0.0  # Would calculate actual time
        }

    def finalize_training(self):
        """Finalize training and create summary"""
        try:
            # Close CSV files if they exist
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                self.csv_file.close()
                logger.info(f"Global results saved to {self.csv_filename}")
                
            if hasattr(self, 'client_csv_file') and not self.client_csv_file.closed:
                self.client_csv_file.close()
                logger.info(f"Client results saved to {self.client_csv_filename}")

            # Create summary
            summary_file = os.path.join(self.config.output_dir, "results", "training_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("=== Federated Learning Training Summary ===\n")
                f.write(f"Configuration: LoRA + KD + Synchronization\n")
                f.write(f"Total Rounds: {self.config.num_rounds}\n")
                f.write(f"Total Clients: {len(self.connected_clients)}\n")
                f.write(f"Global Results File: {os.path.basename(self.csv_filename)}\n")
                f.write(f"Client Results File: {os.path.basename(self.client_csv_filename)}\n")
                f.write(f"Teacher Metrics File: {os.path.basename(self.teacher_metrics_file)}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error finalizing training: {e}")
            raise

    async def start_server(self):
        """Start the federated server"""
        logger.info(f"Starting federated server on port {self.config.port}")

        # Setup message handlers
        self.websocket_server.register_message_handler(
            MessageProtocol.CLIENT_REGISTER,
            self.handle_client_registration
        )
        self.websocket_server.register_message_handler(
            MessageProtocol.CLIENT_UPDATE,
            self.handle_client_update
        )
        self.websocket_server.register_message_handler(
            MessageProtocol.HEARTBEAT,
            self.handle_heartbeat
        )

        # Start server
        await self.websocket_server.start_server(self.client_handler)

        # Start heartbeat task to keep connections alive
        asyncio.create_task(self._heartbeat_task())

        # Run training
        await self.run_federated_training()

    async def _heartbeat_task(self):
        """Send periodic heartbeats to keep connections alive"""
        while True:
            try:
                if self.connected_clients:
                    heartbeat_message = MessageProtocol.create_heartbeat_message("server")
                    await self.websocket_server.broadcast(heartbeat_message)
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(30)

async def run_server(config: FederatedConfig):
    """Run the federated server"""
    server = FederatedServer(config)
    await server.start_server()

if __name__ == "__main__":
    import argparse
    from federated_config import create_argument_parser, load_config

    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments for server mode
    if args.mode != "server":
        parser.error("This script is for server mode only.")

    config = load_config(args)
    config.print_summary()

    asyncio.run(run_server(config))
