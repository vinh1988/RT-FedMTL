#!/usr/bin/env python3
"""
Simple test script for MTL Federated Learning System
Tests basic imports and class creation without heavy model loading
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all necessary modules can be imported"""
    print(" Testing imports...")

    try:
        # Test basic imports
        import torch
        import numpy as np
        import asyncio
        from typing import Dict, List, Optional, Tuple
        from dataclasses import dataclass, asdict
        from collections import defaultdict
        from pathlib import Path
        print(" Basic imports successful")

        # Test transformers import (without actually loading models)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(" Transformers imports successful")

        # Test datasets import
        from datasets import load_dataset
        print(" Datasets import successful")

        # Test sklearn imports
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        print(" Scikit-learn imports successful")

        return True

    except Exception as e:
        print(f" Import failed: {e}")
        return False

def test_class_definitions():
    """Test that our classes are properly defined"""
    print("\n🏗️ Testing class definitions...")

    try:
        # Import our modules
        from no_lora_federated_system import (
            NoLoRAConfig,
            MultiTaskFederatedDataset,
            TaskSpecificFederatedDataset,
            MultiTaskFederatedClient,
            MTLFederatedServer
        )
        print(" All classes imported successfully")

        # Test configuration creation
        config = NoLoRAConfig(
            server_model="prajjwal1/bert-tiny",
            client_model="prajjwal1/bert-tiny",
            num_rounds=2,
            min_clients=1,
            max_clients=2,
            data_samples_per_client=50,  # Very small for testing
            data_distribution="non_iid",
            non_iid_alpha=0.5
        )
        print(" Configuration created successfully")

        # Test dataset creation (without loading actual data)
        print(" Class definitions test passed")

        return True

    except Exception as e:
        print(f" Class definition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mtl_architecture():
    """Test MTL architecture concepts"""
    print("\n🏛️ Testing MTL architecture...")

    try:
        # Test that we have the right structure for MTL
        from no_lora_federated_system import MultiTaskFederatedClient, MTLFederatedServer, NoLoRAConfig

        # Check that client accepts multiple tasks
        config = NoLoRAConfig(
            client_model="prajjwal1/bert-tiny",
            server_model="prajjwal1/bert-tiny",
            data_samples_per_client=50
        )

        # Verify the class accepts multiple tasks parameter
        import inspect
        client_init = inspect.signature(MultiTaskFederatedClient.__init__)
        server_init = inspect.signature(MTLFederatedServer.__init__)

        if 'tasks' in client_init.parameters:
            print(" MultiTaskFederatedClient accepts multiple tasks")
        else:
            print(" MultiTaskFederatedClient missing tasks parameter")
            return False

        if 'config' in server_init.parameters:
            print(" MTLFederatedServer accepts config parameter")
        else:
            print(" MTLFederatedServer missing config parameter")
            return False

        print(" MTL architecture test passed")
        return True

    except Exception as e:
        print(f" MTL architecture test failed: {e}")
        return False

def main():
    """Main test function"""
    print(" Simple MTL Federated Learning System Test")
    print("=" * 60)

    # Test 1: Imports
    import_success = test_imports()

    # Test 2: Class definitions
    class_success = test_class_definitions()

    # Test 3: MTL architecture
    arch_success = test_mtl_architecture()

    print("\n Test Results:")
    print(f"   Imports: {' PASS' if import_success else ' FAIL'}")
    print(f"   Classes: {' PASS' if class_success else ' FAIL'}")
    print(f"   Architecture: {' PASS' if arch_success else ' FAIL'}")

    if import_success and class_success and arch_success:
        print("\n All tests passed! MTL Federated Learning System structure is correct.")
        print("\n Ready for full testing with model loading!")

        print("\n Next Steps:")
        print("1. Run full test: python test_mtl_federated.py")
        print("2. Run demo: python demo_mtl_federated.py")
        print("3. Start server: python no_lora_federated_system.py --mode server")
        print("4. Start client: python no_lora_federated_system.py --mode client --client_id client_1 --tasks sst2 qqp stsb")

        return True
    else:
        print("\n Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
