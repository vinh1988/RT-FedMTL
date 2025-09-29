#!/usr/bin/env python3
"""
Simple test for streaming WebSocket federated learning
Debug version to identify connection and training issues
"""

import asyncio
import websockets
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import logging
import argparse
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTestServer:
    """Simple test server for debugging"""
    
    def __init__(self, port=8765):
        self.port = port
        self.connected_clients = {}
        self.client_results = {}
        self.current_round = 0
        
    async def register_client(self, websocket, client_info):
        """Register a new client"""
        client_id = client_info['client_id']
        task_name = client_info.get('task_name', 'unknown')
        
        self.connected_clients[client_id] = {
            'websocket': websocket,
            'task_name': task_name,
            'status': 'connected'
        }
        
        logger.info(f"✅ Client {client_id} ({task_name}) registered. Total clients: {len(self.connected_clients)}")
        
        # Send welcome
        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': f'Welcome {client_id}! Server ready.',
            'client_count': len(self.connected_clients)
        }))
    
    async def broadcast_message(self, message):
        """Broadcast to all clients"""
        if not self.connected_clients:
            return
            
        disconnected = []
        for client_id, client_info in self.connected_clients.items():
            try:
                await client_info['websocket'].send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            del self.connected_clients[client_id]
            logger.warning(f"❌ Client {client_id} disconnected")
    
    async def start_training_round(self, round_num):
        """Start a training round"""
        self.current_round = round_num
        self.client_results = {}
        
        logger.info(f"🚀 Starting round {round_num} with {len(self.connected_clients)} clients")
        
        # Send training command
        message = {
            'type': 'start_training',
            'round': round_num,
            'message': f'Start training round {round_num}',
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_message(message)
    
    async def handle_client_result(self, client_id, result):
        """Handle training result from client"""
        self.client_results[client_id] = result
        
        logger.info(f"📊 Received result from {client_id}: loss={result.get('loss', 0):.4f}")
        
        # Check if round complete
        if len(self.client_results) >= len(self.connected_clients):
            await self.complete_round()
    
    async def complete_round(self):
        """Complete the current round"""
        avg_loss = np.mean([r.get('loss', 0) for r in self.client_results.values()])
        
        logger.info(f"✅ Round {self.current_round} completed. Average loss: {avg_loss:.4f}")
        
        # Broadcast completion
        await self.broadcast_message({
            'type': 'round_complete',
            'round': self.current_round,
            'avg_loss': avg_loss,
            'results': dict(self.client_results)
        })
    
    async def handle_message(self, websocket, message):
        """Handle client message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'register':
                await self.register_client(websocket, data)
            
            elif msg_type == 'training_result':
                client_id = data.get('client_id')
                result = data.get('result', {})
                await self.handle_client_result(client_id, result)
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def client_handler(self, websocket, path):
        """Handle client connection"""
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
    
    async def run_training_demo(self):
        """Run a simple training demo"""
        logger.info("⏳ Waiting for clients to connect...")
        
        # Wait for at least 2 clients
        while len(self.connected_clients) < 2:
            await asyncio.sleep(1)
            logger.info(f"Connected clients: {len(self.connected_clients)}")
        
        logger.info(f"✅ {len(self.connected_clients)} clients connected. Starting training...")
        
        # Run 2 training rounds
        for round_num in range(1, 3):
            await self.start_training_round(round_num)
            
            # Wait for results
            while len(self.client_results) < len(self.connected_clients):
                await asyncio.sleep(0.5)
            
            await asyncio.sleep(1)  # Brief pause
        
        logger.info("🎉 Training demo completed!")
    
    async def start_server(self):
        """Start the test server"""
        logger.info(f"🌐 Starting test server on port {self.port}")
        
        server = await websockets.serve(self.client_handler, "localhost", self.port)
        logger.info("📡 Server started, waiting for clients...")
        
        # Run training demo
        training_task = asyncio.create_task(self.run_training_demo())
        await training_task
        
        # Keep server alive briefly
        await asyncio.sleep(5)
        
        server.close()
        await server.wait_closed()


class SimpleTestClient:
    """Simple test client for debugging"""
    
    def __init__(self, client_id, task_name="test", port=8765):
        self.client_id = client_id
        self.task_name = task_name
        self.port = port
        self.websocket = None
    
    async def connect_to_server(self):
        """Connect to server"""
        uri = f"ws://localhost:{self.port}"
        
        try:
            self.websocket = await websockets.connect(uri)
            logger.info(f"🔗 {self.client_id} connected to server")
            
            # Register
            await self.websocket.send(json.dumps({
                'type': 'register',
                'client_id': self.client_id,
                'task_name': self.task_name
            }))
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise
    
    def simulate_training(self):
        """Simulate training and return fake results"""
        # Simulate some training time
        time.sleep(np.random.uniform(1, 3))
        
        # Return fake results
        return {
            'loss': np.random.uniform(0.1, 1.0),
            'accuracy': np.random.uniform(0.5, 0.9),
            'task': self.task_name
        }
    
    async def handle_server_message(self, message):
        """Handle server message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'welcome':
                logger.info(f"🎉 {self.client_id}: {data.get('message')}")
            
            elif msg_type == 'start_training':
                round_num = data.get('round')
                logger.info(f"🎯 {self.client_id}: Starting training round {round_num}")
                
                # Simulate training
                result = self.simulate_training()
                
                # Send result back
                await self.websocket.send(json.dumps({
                    'type': 'training_result',
                    'client_id': self.client_id,
                    'result': result
                }))
                
                logger.info(f"📊 {self.client_id}: Sent training result")
            
            elif msg_type == 'round_complete':
                round_num = data.get('round')
                avg_loss = data.get('avg_loss')
                logger.info(f"✅ {self.client_id}: Round {round_num} completed, avg loss: {avg_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def start_client(self):
        """Start the test client"""
        await self.connect_to_server()
        
        try:
            async for message in self.websocket:
                await self.handle_server_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 {self.client_id}: Disconnected from server")
        except Exception as e:
            logger.error(f"{self.client_id} error: {e}")


async def run_test_server():
    """Run test server"""
    server = SimpleTestServer()
    await server.start_server()


async def run_test_client(client_id, task_name):
    """Run test client"""
    client = SimpleTestClient(client_id, task_name)
    await client.start_client()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--client_id", type=str, help="Client ID")
    parser.add_argument("--task", type=str, default="test", help="Task name")
    
    args = parser.parse_args()
    
    if args.mode == "server":
        print("🌐 Starting test server...")
        asyncio.run(run_test_server())
    
    elif args.mode == "client":
        if not args.client_id:
            print("Error: --client_id required for client mode")
            return
        
        print(f"👤 Starting test client: {args.client_id}")
        asyncio.run(run_test_client(args.client_id, args.task))


if __name__ == "__main__":
    main()
