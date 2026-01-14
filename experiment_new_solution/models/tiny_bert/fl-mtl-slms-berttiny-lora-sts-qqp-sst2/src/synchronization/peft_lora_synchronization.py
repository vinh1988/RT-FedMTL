#!/usr/bin/env python3
"""
PEFT LoRA Synchronization Implementation
Manages efficient LoRA adapter synchronization for federated multi-task learning
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
import torch
import logging

logger = logging.getLogger(__name__)


class PEFTLoRASynchronizationManager:
    """Manages PEFT LoRA adapter synchronization between server and clients"""

    def __init__(self, server):
        self.server = server
        self.synchronization_history = []
        self.global_model_version = 0

    async def broadcast_lora_adapters(self, lora_state: Dict):
        """
        Broadcast LoRA adapter parameters to clients
        Much more efficient than full model broadcast
        """
        sync_message = {
            "type": "lora_sync",
            "lora_state": lora_state,
            "version": self.global_model_version,
            "timestamp": datetime.now().isoformat()
        }

        # Send to all connected clients
        await self.server.websocket_server.broadcast(sync_message)

        # Record synchronization
        self.synchronization_history.append({
            "type": "lora_broadcast",
            "timestamp": datetime.now(),
            "clients_notified": len(self.server.connected_clients),
            "version": self.global_model_version,
            "params_synced": len(lora_state.get("lora_parameters", {}))
        })

        logger.info(f"Broadcasted LoRA adapters v{self.global_model_version} to {len(self.server.connected_clients)} clients")
        logger.info(f"  LoRA parameters synced: {len(lora_state.get('lora_parameters', {}))}")

    async def broadcast_task_lora_adapters(self, task: str, lora_params: Dict):
        """
        Broadcast task-specific LoRA adapters to relevant clients
        """
        sync_message = {
            "type": "task_lora_sync",
            "task": task,
            "lora_parameters": self._serialize_params(lora_params),
            "version": self.global_model_version,
            "timestamp": datetime.now().isoformat()
        }

        # Send only to clients working on this task
        task_clients = [
            client for client in self.server.connected_clients.values()
            if client.get('task') == task
        ]

        if task_clients:
            for client in task_clients:
                websocket = client.get('websocket')
                if websocket:
                    await self.server.websocket_server.send_message(websocket, sync_message)

        logger.info(f"Broadcasted task '{task}' LoRA adapters to {len(task_clients)} clients")

    def increment_model_version(self):
        """Increment global model version"""
        self.global_model_version += 1
        return self.global_model_version

    def get_lora_state(self, task: str = None) -> Dict:
        """
        Get current LoRA adapter state for synchronization
        
        Args:
            task: If specified, get only that task's LoRA adapters
                  If None, get all tasks' LoRA adapters
        """
        if not hasattr(self.server, 'peft_lora_model'):
            logger.warning("PEFT LoRA model not available")
            return {}

        # Get LoRA parameters from server model
        # PEFTLoRAServerModel wraps PEFTLoRAMTLModel, so access the inner model
        if task:
            lora_params = self.server.peft_lora_model.mtl_model.get_lora_parameters(task=task)
        else:
            lora_params = self.server.peft_lora_model.get_global_lora_state()

        # Serialize parameters
        serializable_params = self._serialize_params(lora_params)

        return {
            "lora_parameters": serializable_params,
            "model_version": self.global_model_version,
            "aggregation_round": getattr(self.server, 'current_round', 0),
            "task": task if task else "all_tasks"
        }

    def _serialize_params(self, params: Dict) -> Dict:
        """Convert tensor parameters to JSON-serializable format"""
        serializable = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, torch.Tensor):
                serializable[param_name] = param_value.tolist()
            else:
                serializable[param_name] = param_value
        return serializable

    def _deserialize_params(self, params: Dict) -> Dict:
        """Convert JSON parameters back to tensors"""
        tensor_params = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, list):
                tensor_params[param_name] = torch.tensor(param_value, dtype=torch.float32)
            else:
                tensor_params[param_name] = param_value
        return tensor_params

    def update_lora_from_aggregation(self, aggregated_lora_params: Dict):
        """
        Update PEFT LoRA model with aggregated adapter parameters
        """
        if not hasattr(self.server, 'peft_lora_model'):
            logger.warning("PEFT LoRA model not available for update")
            return

        # Update server model with aggregated LoRA parameters
        self.server.peft_lora_model.update_from_aggregation(aggregated_lora_params)

        # Increment version
        self.increment_model_version()

        logger.info(f"Updated LoRA adapters to version {self.global_model_version}")
        logger.info(f"  Parameters updated: {len(aggregated_lora_params)}")

    def get_synchronization_summary(self) -> Dict:
        """Get summary of synchronization history"""
        if not self.synchronization_history:
            return {"total_sync_events": 0}

        return {
            "total_sync_events": len(self.synchronization_history),
            "current_version": self.global_model_version,
            "average_clients_per_sync": sum(
                event.get("clients_notified", 0) for event in self.synchronization_history
            ) / len(self.synchronization_history) if self.synchronization_history else 0,
            "last_sync_timestamp": self.synchronization_history[-1]["timestamp"].isoformat() if self.synchronization_history else None,
            "total_params_synced": sum(
                event.get("params_synced", 0) for event in self.synchronization_history
            )
        }


class ClientPEFTLoRASynchronizer:
    """Handles client-side PEFT LoRA synchronization"""

    def __init__(self, local_model, websocket_client, task: str):
        self.local_model = local_model  # Should be PEFTLoRAMTLModel
        self.websocket_client = websocket_client
        self.task = task
        self.lora_cache = None
        self.synchronization_log = []
        self.is_synchronized = False

        logger.info(f"ClientPEFTLoRASynchronizer initialized for task: {task}")

    async def request_lora_sync(self):
        """Request LoRA adapter synchronization from server"""
        sync_request = {
            "type": "request_lora_sync",
            "task": self.task,
            "timestamp": datetime.now().isoformat()
        }

        await self.websocket_client.send_message(sync_request)
        logger.info(f"Requested LoRA sync for task: {self.task}")

    async def synchronize_with_global_model(self, global_state: Dict):
        """
        Unified synchronization method that handles both LoRA and MTL formats
        
        Args:
            global_state: Can be:
                - Full message dict with "lora_state" key (LoRA format)
                - Just the lora_state dict itself
                - MTL format (for compatibility)
        
        Returns:
            Dict with synchronization results
        """
        # Extract lora_state if it's nested in the message
        if "lora_state" in global_state:
            lora_state = global_state["lora_state"]
        else:
            # Assume the entire dict is the lora_state
            lora_state = global_state
        
        # Call the existing update logic
        self.update_local_lora(lora_state)
        
        # Return acknowledgment result
        return {
            "success": self.is_synchronized,
            "task": self.task,
            "model_version": lora_state.get('model_version', 0),
            "timestamp": datetime.now().isoformat()
        }

    def update_local_lora(self, lora_state: Dict):
        """
        Update local model with synchronized LoRA adapters
        
        Args:
            lora_state: Dict containing 'lora_parameters', 'model_version', etc.
        """
        lora_params = lora_state.get('lora_parameters', {})

        if not lora_params:
            logger.warning("Received empty LoRA parameters")
            return

        # Deserialize parameters (convert lists to tensors)
        tensor_params = {}
        for param_name, param_value in lora_params.items():
            if isinstance(param_value, list):
                tensor_params[param_name] = torch.tensor(param_value, dtype=torch.float32)
            else:
                tensor_params[param_name] = param_value

        # Update local model
        if hasattr(self.local_model, 'set_lora_parameters'):
            self.local_model.set_lora_parameters(tensor_params, task=self.task)
            self.is_synchronized = True
            
            # Cache the synchronized state
            self.lora_cache = lora_state
            
            # Log synchronization
            self.synchronization_log.append({
                "timestamp": datetime.now(),
                "model_version": lora_state.get('model_version', 0),
                "params_updated": len(tensor_params)
            })

            logger.info(f"✓ Updated local LoRA adapters for task '{self.task}'")
            logger.info(f"  Parameters updated: {len(tensor_params)}")
            logger.info(f"  Model version: {lora_state.get('model_version', 0)}")
        else:
            logger.error("Local model does not support LoRA parameter updates")

    def get_local_lora_state(self) -> Dict:
        """Get current local LoRA adapter state"""
        if not hasattr(self.local_model, 'get_lora_parameters'):
            logger.warning("Local model does not support LoRA parameter extraction")
            return {}

        # Get LoRA parameters for this client's task
        lora_params = self.local_model.get_lora_parameters(task=self.task)

        return {
            "lora_parameters": lora_params,
            "task": self.task,
            "is_synchronized": self.is_synchronized
        }

    async def send_synchronization_acknowledgment(self, sync_result: Dict):
        """Send synchronization acknowledgment to server"""
        ack_message = {
            "type": "sync_acknowledgment",
            "client_id": getattr(self.websocket_client, 'client_id', 'unknown'),
            "synchronized": sync_result.get("success", False),
            "model_version": sync_result.get("model_version", 0),
            "task": self.task,
            "timestamp": datetime.now().isoformat()
        }

        await self.websocket_client.send_message(ack_message)
        logger.info(f"Sent synchronization acknowledgment for task: {self.task}")

    def get_sync_summary(self) -> Dict:
        """Get synchronization summary"""
        return {
            "task": self.task,
            "is_synchronized": self.is_synchronized,
            "total_syncs": len(self.synchronization_log),
            "last_sync": self.synchronization_log[-1] if self.synchronization_log else None
        }

