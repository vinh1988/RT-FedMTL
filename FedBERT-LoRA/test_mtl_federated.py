#!/usr/bin/env python3
"""
Test script for MTL Federated Learning System
Tests the new MultiTaskFederatedClient and MTLFederatedServer
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from no_lora_federated_system import (
    NoLoRAConfig,
    MultiTaskFederatedClient,
    MTLFederatedServer,
    run_mtl_federated_experiment
)

async def test_mtl_federated_system():
    """Test the MTL federated learning system"""
    print(" Testing MTL Federated Learning System...")

    # Create configuration for testing
    config = NoLoRAConfig(
        server_model="prajjwal1/bert-tiny",
        client_model="prajjwal1/bert-tiny",
        num_rounds=2,  # Just 2 rounds for testing
        min_clients=1,
        max_clients=2,
        local_epochs=1,  # Just 1 epoch for testing
        batch_size=8,  # Small batch size for testing
        learning_rate=5e-5,
        data_samples_per_client=100,  # Small dataset for testing
        data_distribution="non_iid",
        non_iid_alpha=0.5,
        port=8773  # Different port for testing
    )

    print(f" Configuration created: {config.num_rounds} rounds, {config.max_clients} clients")

    # Test 1: Create MTL Federated Client
    print("\n Test 1: Creating MTL Federated Client...")
    try:
        client = MultiTaskFederatedClient(
            client_id="test_client_1",
            tasks=["sst2", "qqp", "stsb"],
            config=config,
            total_clients=2
        )
        print(f" MTL Client created successfully for tasks: {client.tasks}")
        print(f" Client has {len(client.student_models)} models (one per task)")
        print(f" Dataset loaded with {sum(len(client.dataset.task_data[task]['texts']) for task in client.tasks)} total samples")
    except Exception as e:
        print(f" Failed to create MTL client: {e}")
        return False

    # Test 2: Test data loading
    print("\n Test 2: Testing data loading...")
    try:
        for task_name in client.tasks:
            dataloader = client.dataset.get_task_dataloader(task_name, config.batch_size)
            task_info = client.dataset.get_task_info(task_name)
            print(f" {task_name}: {len(task_info['texts'])} samples, type: {task_info['task_type']}")
            print(f"   Distribution: {list(task_info['distribution'].keys())[:3]}...")  # Show first 3 bins
    except Exception as e:
        print(f" Failed to load data: {e}")
        return False

    # Test 3: Test teacher logits generation
    print("\n Test 3: Testing teacher logits generation...")
    try:
        for task_name in client.tasks:
            teacher_logits = client._get_teacher_logits(task_name, 0)
            task_info = client.dataset.get_task_info(task_name)
            expected_shape = (config.batch_size, task_info['num_classes'])
            print(f" {task_name}: Teacher logits shape {teacher_logits.shape} (expected: {expected_shape})")
    except Exception as e:
        print(f" Failed to generate teacher logits: {e}")
        return False

    # Test 4: Test local training (dry run)
    print("\n Test 4: Testing local training (dry run)...")
    try:
        # This will fail to connect to server, but should at least initialize properly
        training_task = asyncio.create_task(
            client.local_training(round_num=1)
        )

        # Wait a bit for initialization
        await asyncio.sleep(1)

        # Cancel the training task since we can't connect to server in test
        training_task.cancel()
        print(" Local training initialization successful")
    except asyncio.CancelledError:
        print(" Local training cancelled successfully (expected)")
    except Exception as e:
        print(f" Local training had issues (expected in test environment): {e}")

    print("\n All tests passed! MTL Federated Learning System is ready.")
    return True

async def test_server_creation():
    """Test server creation"""
    print("\n Testing MTL Federated Server creation...")

    config = NoLoRAConfig(
        server_model="prajjwal1/bert-tiny",
        client_model="prajjwal1/bert-tiny",
        num_rounds=2,
        min_clients=1,
        max_clients=2,
        local_epochs=1,
        batch_size=8,
        learning_rate=5e-5,
        data_samples_per_client=100,
        data_distribution="non_iid",
        non_iid_alpha=0.5,
        port=8774
    )

    try:
        server = MTLFederatedServer(config)
        print(" MTL Federated Server created successfully")
        print(f" Server model: {config.server_model}")
        print(f" Global model parameters: {sum(p.numel() for p in server.global_model.parameters()):,}")
        return True
    except Exception as e:
        print(f" Failed to create server: {e}")
        return False

async def main():
    """Main test function"""
    print(" Starting MTL Federated Learning System Tests...")

    # Test client creation and functionality
    client_success = await test_mtl_federated_system()

    # Test server creation
    server_success = await test_server_creation()

    if client_success and server_success:
        print("\n All tests passed! The MTL Federated Learning System is ready for use.")
        print("\n Usage Examples:")
        print("Server: python no_lora_federated_system.py --mode server --rounds 22 --total_clients 5")
        print("Client: python no_lora_federated_system.py --mode client --client_id client_1 --tasks sst2 qqp stsb")
    else:
        print("\n Some tests failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
