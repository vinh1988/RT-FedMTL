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

        # Initialize components
        self.lora_aggregator = LoRAAggregator()
        self.kd_manager = GlobalKDManager(None, config)  # Teacher model will be set later
        self.websocket_server = WebSocketServer(config.port)
        self.synchronization_manager = SynchronizationManager(self)

        # Setup results management
        self.setup_results_management()

        # Setup logging
        self.setup_logging()

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
        """Setup CSV file for results"""
        os.makedirs(self.config.results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(
            self.config.results_dir,
            f"federated_results_{timestamp}.csv"
        )

        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write CSV headers
        headers = [
            "round", "responses_received", "avg_accuracy", "classification_accuracy",
            "regression_accuracy", "total_clients", "active_clients", "training_time",
            "synchronization_events", "global_model_version", "timestamp"
        ]
        self.csv_writer.writerow(headers)
        self.csv_file.flush()

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
        """Wait for minimum number of clients"""
        logger.info(f"Waiting for clients... (need {self.config.min_clients})")

        start_time = time.time()
        while len(self.connected_clients) < self.config.min_clients:
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout:
                logger.warning(f"Timeout waiting for clients after {elapsed:.0f}s")
                return False

            logger.info(f"Waiting for clients... ({len(self.connected_clients)}/{self.config.min_clients})")
            await asyncio.sleep(2)

        logger.info(f"Minimum clients reached ({len(self.connected_clients)}), starting training")
        return True

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
                    self.kd_manager.update_teacher_from_students(student_knowledge)

                # Step 7: Record results for this round
                self.record_round_results(round_num, round_start)

                logger.info(f"Round {round_num} completed in {time.time() - round_start:.2f}s")

            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                continue

        # Finalize training
        self.finalize_training()

    async def wait_for_sync_acknowledgments(self, round_num: int, timeout: int = 30):
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

    async def collect_client_updates(self, round_num: int, timeout: int = 60):
        """Collect client updates for a round"""
        start_time = time.time()
        updates_received = 0
        expected_clients = len(self.connected_clients)

        logger.info(f"Waiting for client updates... (expecting {expected_clients})")

        while time.time() - start_time < timeout:
            if round_num in self.client_updates:
                updates_received = len(self.client_updates[round_num])

            if updates_received >= expected_clients:
                logger.info(f"All client updates received ({updates_received}/{expected_clients})")
                return True

            await asyncio.sleep(2)

        logger.warning(f"Timeout collecting updates. Got {updates_received}/{expected_clients}")
        return False

    def generate_sample_inputs_for_kd(self) -> Dict[str, Dict]:
        """Generate sample inputs for teacher knowledge generation"""
        # This would typically use a small validation set or synthetic data
        # For demonstration, we'll create placeholder inputs
        sample_inputs = {}

        for task in ['sst2', 'qqp', 'stsb']:
            # Create dummy inputs for demonstration
            if task in ['sst2', 'qqp']:
                # Binary classification tasks
                sample_inputs[task] = {
                    'input_ids': torch.randint(0, 1000, (1, 128)),
                    'attention_mask': torch.ones(1, 128)
                }
            else:
                # Regression task
                sample_inputs[task] = {
                    'input_ids': torch.randint(0, 1000, (1, 128)),
                    'attention_mask': torch.ones(1, 128)
                }

        return sample_inputs

    def record_round_results(self, round_num: int, round_start: float):
        """Record results for a training round"""
        updates = self.client_updates.get(round_num, [])
        responses = len(updates)

        if responses > 0:
            # Calculate aggregated metrics
            metrics = self.calculate_aggregated_metrics(updates)

            # Record to CSV
            row = [
                round_num,
                responses,
                f"{metrics['avg_accuracy']:.4f}",
                f"{metrics['classification_accuracy']:.4f}",
                f"{metrics['regression_accuracy']:.4f}",
                len(self.connected_clients),
                len(self.connected_clients),
                f"{time.time() - round_start:.2f}",
                self.synchronization_manager.global_model_version,
                str(self.synchronization_manager.global_model_version),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]

            self.csv_writer.writerow(row)
            self.csv_file.flush()

            # Print progress
            print(f"🏃 Round {round_num} completed: {metrics['avg_accuracy']:.4f} avg accuracy")

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
            self.csv_file.close()
            logger.info(f"Results saved to {self.csv_filename}")

            # Create summary
            summary_file = os.path.join(self.config.results_dir, "training_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("Federated Learning Training Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Configuration: LoRA + KD + Synchronization\n")
                f.write(f"Total Rounds: {self.config.num_rounds}\n")
                f.write(f"Total Clients: {len(self.connected_clients)}\n")
                f.write(f"Results File: {self.csv_filename}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            logger.info(f"Summary saved to {summary_file}")

        except Exception as e:
            logger.error(f"Error finalizing training: {e}")

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

        # Start server
        await self.websocket_server.start_server(self.client_handler)

        # Run training
        await self.run_federated_training()

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
