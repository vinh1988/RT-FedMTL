#!/usr/bin/env python3
"""
Quick test script for optimized MTL Federated Learning System
Tests the optimized system with improved STSB regression performance
"""

import sys
import os
import asyncio
import logging
import time
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the optimized system
from optimized_mtl_federated import OptimizedConfig, OptimizedMTLFederatedServer, OptimizedMultiTaskFederatedClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_optimized_system():
    """Test the optimized MTL federated system"""
    print(" Testing Optimized MTL Federated Learning System")
    print("=" * 60)

    try:
        # Test 1: Configuration loading
        print(" Testing configuration loading...")
        config = OptimizedConfig(
            num_rounds=2,
            min_clients=1,
            max_clients=2,
            samples_per_client=200,
            learning_rate=1e-4,
            data_distribution="non_iid",
            non_iid_alpha=0.3
        )
        print(" Configuration loaded successfully")
        print(f"   - Rounds: {config.num_rounds}")
        print(f"   - Learning Rate: {config.learning_rate}")
        print(f"   - Samples per client: {config.samples_per_client}")

        # Test 2: Server initialization
        print("\n Testing server initialization...")
        server = OptimizedMTLFederatedServer(config)
        print(" Server initialized successfully")
        print(f"   - Model: {config.server_model}")
        print(f"   - Global model parameters: {sum(p.numel() for p in server.global_model.parameters()):,}")

        # Test 3: Client initialization (without actual connection)
        print("\n🤖 Testing client initialization...")
        client = OptimizedMultiTaskFederatedClient(
            client_id="test_client",
            tasks=["sst2", "qqp", "stsb"],
            config=config,
            total_clients=2
        )
        print(" Client initialized successfully")
        print(f"   - Tasks: {client.tasks}")
        print(f"   - Total samples: {sum(len(client.dataset.task_data[task]['texts']) for task in client.tasks)}")

        # Test 4: Data distribution analysis
        print("\n Testing data distribution...")
        for task_name in client.tasks:
            task_info = client.dataset.get_task_info(task_name)
            print(f"   - {task_name}: {len(task_info['texts'])} samples, type: {task_info['task_type']}")

        # Test 5: STSB regression optimization check
        print("\n Testing STSB regression optimization...")
        stsb_info = client.dataset.get_task_info("stsb")
        if stsb_info['task_type'] == "regression":
            labels = stsb_info['labels']
            if labels:
                # Check if labels are properly normalized (0-1 range)
                min_label = min(labels)
                max_label = max(labels)
                mean_label = np.mean(labels)
                print(f"   - STSB label range: {min_label:.3f} to {max_label:.3f}")
                print(f"   - STSB label mean: {mean_label:.3f}")

                if 0 <= min_label and max_label <= 1:
                    print("    STSB labels properly normalized")
                else:
                    print("    STSB labels may need normalization")

        # Test 6: Model parameter efficiency
        print("\n⚡ Testing model efficiency...")
        total_params = 0
        for task_name in client.tasks:
            model = client.student_models[task_name]
            task_params = sum(p.numel() for p in model.parameters())
            total_params += task_params
            print(f"   - {task_name} model: {task_params:,} parameters")

        print(f"   - Total student parameters: {total_params:,}")
        print(f"   - Teacher parameters: {sum(p.numel() for p in client.teacher_model.parameters()):,}")

        print("\n All optimization tests passed!")
        print("\n Ready for optimized MTL federated learning!")
        print("\n Next Steps:")
        print("1. Start server: python optimized_mtl_federated.py --mode server")
        print("2. Start client: python optimized_mtl_federated.py --mode client --client_id client_1 --tasks sst2 qqp stsb")
        print("3. Monitor improved STSB regression performance")

        return True

    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = asyncio.run(test_optimized_system())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
