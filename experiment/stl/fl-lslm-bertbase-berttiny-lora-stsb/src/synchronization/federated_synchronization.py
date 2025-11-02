#!/usr/bin/env python3
"""
Model Synchronization Implementation
Bidirectional model synchronization for federated learning
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
import torch
import logging

logger = logging.getLogger(__name__)

class SynchronizationManager:
    """Manages model synchronization between server and clients"""

    def __init__(self, server):
        self.server = server
        self.synchronization_history = []
        self.global_model_version = 0

    async def broadcast_global_model(self, global_state: Dict):
        """Broadcast updated global model to all clients"""
        sync_message = {
            "type": "global_model_sync",
            "global_model_state": global_state,
            "version": self.global_model_version,
            "timestamp": datetime.now().isoformat()
        }

        # Send to all connected clients
        await self.server.websocket_server.broadcast(sync_message)

        # Record synchronization
        self.synchronization_history.append({
            "type": "broadcast",
            "timestamp": datetime.now(),
            "clients_notified": len(self.server.connected_clients),
            "version": self.global_model_version
        })

        logger.info(f"Broadcasted global model version {self.global_model_version} to {len(self.server.connected_clients)} clients")

    def increment_model_version(self):
        """Increment global model version"""
        self.global_model_version += 1
        return self.global_model_version

    def get_global_model_state(self) -> Dict:
        """Get current global model state for synchronization"""
        # Convert tensors to JSON-serializable format
        serializable_teacher_logits = {}
        if hasattr(self.server, 'global_teacher_logits'):
            for task, tensor in self.server.global_teacher_logits.items():
                if isinstance(tensor, torch.Tensor):
                    serializable_teacher_logits[task] = tensor.tolist()
                else:
                    serializable_teacher_logits[task] = tensor

        serializable_lora_params = {}
        if hasattr(self.server, 'global_lora_params'):
            for task, params in self.server.global_lora_params.items():
                serializable_task_params = {}
                for param_name, param_value in params.items():
                    if isinstance(param_value, torch.Tensor):
                        serializable_task_params[param_name] = param_value.tolist()
                    else:
                        serializable_task_params[param_name] = param_value
                serializable_lora_params[task] = serializable_task_params

        return {
            "teacher_logits": serializable_teacher_logits,
            "global_lora_params": serializable_lora_params,
            "model_version": self.global_model_version,
            "aggregation_round": getattr(self.server, 'current_round', 0)
        }

    def update_global_model_from_aggregation(self, aggregated_lora: Dict):
        """Update global model with aggregated LoRA parameters"""
        # Update global LoRA parameters
        if not hasattr(self.server, 'global_lora_params'):
            self.server.global_lora_params = {}

        self.server.global_lora_params.update(aggregated_lora)

        # Increment version
        self.increment_model_version()

        logger.info(f"Updated global model to version {self.global_model_version}")

    def get_synchronization_summary(self) -> Dict:
        """Get summary of synchronization history"""
        if not self.synchronization_history:
            return {"total_sync_events": 0}

        return {
            "total_sync_events": len(self.synchronization_history),
            "current_version": self.global_model_version,
            "average_clients_per_sync": sum(
                event["clients_notified"] for event in self.synchronization_history
            ) / len(self.synchronization_history),
            "last_sync_timestamp": self.synchronization_history[-1]["timestamp"].isoformat() if self.synchronization_history else None
        }

class ClientModelSynchronizer:
    """Handles client-side model synchronization"""

    def __init__(self, local_model, websocket_client):
        self.local_model = local_model
        self.websocket_client = websocket_client
        self.global_model_cache = None
        self.synchronization_log = []
        self.is_synchronized = False

    async def synchronize_with_global_model(self, global_state: Dict) -> Dict:
        """Update local model with global knowledge"""
        try:
            # Update LoRA parameters with global aggregation
            global_lora_params = global_state.get("global_lora_params", {})
            if global_lora_params:
                await self.update_lora_with_global_params(global_lora_params)

            # Update local knowledge base with global teacher knowledge
            global_teacher_logits = global_state.get("teacher_logits", {})
            if global_teacher_logits:
                # Convert lists back to tensors
                tensor_logits = {}
                for task, logits_list in global_teacher_logits.items():
                    if isinstance(logits_list, list):
                        # Get device from the model's parameters
                        model_device = next(self.local_model.parameters()).device
                        tensor_logits[task] = torch.tensor(logits_list, device=model_device)
                    else:
                        # Ensure existing tensors are on the correct device
                        if isinstance(logits_list, torch.Tensor):
                            model_device = next(self.local_model.parameters()).device
                            tensor_logits[task] = logits_list.to(model_device)
                        else:
                            tensor_logits[task] = logits_list
                self.local_model.update_with_global_knowledge(tensor_logits)

            # Update cache
            self.global_model_cache = global_state

            # Mark as synchronized
            self.is_synchronized = True

            # Log synchronization
            sync_record = {
                "timestamp": datetime.now(),
                "global_knowledge_integrated": True,
                "lora_params_updated": bool(global_lora_params),
                "model_version": global_state.get("model_version", 0)
            }
            self.synchronization_log.append(sync_record)

            logger.info("Client model synchronized with global model")

            return {
                "synchronized": True,
                "global_knowledge_integrated": True,
                "lora_params_updated": bool(global_lora_params),
                "model_version": global_state.get("model_version", 0)
            }

        except Exception as e:
            logger.error(f"Error during model synchronization: {e}")
            return {
                "synchronized": False,
                "error": str(e)
            }

    async def update_lora_with_global_params(self, global_lora_params: Dict):
        """Update local LoRA parameters with global parameters"""
        for task, lora_params in global_lora_params.items():
            if task in self.local_model.task_adapters:
                # Convert lists back to tensors for LoRA parameters
                tensor_params = {}
                for param_name, param_value in lora_params.items():
                    if isinstance(param_value, list):
                        # Get device from the model's parameters
                        model_device = next(self.local_model.parameters()).device
                        tensor_params[param_name] = torch.tensor(param_value, device=model_device)
                    else:
                        # Ensure existing tensors are on the correct device
                        if isinstance(param_value, torch.Tensor):
                            model_device = next(self.local_model.parameters()).device
                            tensor_params[param_name] = param_value.to(model_device)
                        else:
                            tensor_params[param_name] = param_value

                # Update local LoRA with global LoRA
                self.local_model.task_adapters[task].load_lora_params(tensor_params)
                logger.debug(f"Updated LoRA parameters for task {task}")

    async def send_synchronization_acknowledgment(self, sync_result: Dict):
        """Send synchronization acknowledgment to server"""
        ack_message = {
            "type": "sync_acknowledgment",
            "client_id": self.websocket_client.client_id,
            "synchronized": sync_result["synchronized"],
            "model_version": sync_result.get("model_version", 0),
            "timestamp": datetime.now().isoformat()
        }

        await self.websocket_client.send(ack_message)

    def get_synchronization_status(self) -> Dict:
        """Get current synchronization status"""
        return {
            "is_synchronized": self.is_synchronized,
            "global_model_version": self.global_model_cache.get("model_version", 0) if self.global_model_cache else 0,
            "last_sync_timestamp": self.synchronization_log[-1]["timestamp"].isoformat() if self.synchronization_log else None,
            "sync_events_count": len(self.synchronization_log),
            "global_knowledge_cached": bool(self.global_model_cache)
        }

class ModelStateManager:
    """Manages model state for synchronization"""

    def __init__(self):
        self.model_states = {}
        self.state_history = []

    def save_model_state(self, model_id: str, state: Dict):
        """Save model state for synchronization"""
        self.model_states[model_id] = state
        self.state_history.append({
            "model_id": model_id,
            "timestamp": datetime.now(),
            "state_size": len(str(state))
        })

    def get_model_state(self, model_id: str) -> Dict:
        """Get saved model state"""
        return self.model_states.get(model_id, {})

    def get_state_summary(self) -> Dict:
        """Get summary of saved states"""
        return {
            "total_states": len(self.model_states),
            "state_history_length": len(self.state_history),
            "latest_state_timestamp": self.state_history[-1]["timestamp"].isoformat() if self.state_history else None,
            "model_ids": list(self.model_states.keys())
        }

class SynchronizationProtocol:
    """Defines synchronization protocols and message formats"""

    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    SYNC_BROADCAST = "sync_broadcast"
    SYNC_ACKNOWLEDGMENT = "sync_acknowledgment"

    @staticmethod
    def create_sync_request(client_id: str, requested_version: int = None) -> Dict:
        """Create synchronization request message"""
        return {
            "type": SynchronizationProtocol.SYNC_REQUEST,
            "client_id": client_id,
            "requested_version": requested_version,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def create_sync_response(sync_data: Dict, version: int) -> Dict:
        """Create synchronization response message"""
        return {
            "type": SynchronizationProtocol.SYNC_RESPONSE,
            "sync_data": sync_data,
            "version": version,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def create_sync_broadcast(global_state: Dict, version: int) -> Dict:
        """Create synchronization broadcast message"""
        return {
            "type": SynchronizationProtocol.SYNC_BROADCAST,
            "global_state": global_state,
            "version": version,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def create_sync_acknowledgment(client_id: str, version: int, success: bool) -> Dict:
        """Create synchronization acknowledgment message"""
        return {
            "type": SynchronizationProtocol.SYNC_ACKNOWLEDGMENT,
            "client_id": client_id,
            "version": version,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
