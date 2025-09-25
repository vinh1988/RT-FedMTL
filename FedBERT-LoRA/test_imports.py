#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports"""
    
    print("🔍 Testing imports...")
    
    try:
        # Test core libraries
        import torch
        import transformers
        import flwr
        import peft
        import datasets
        print("✅ Core libraries imported successfully")
        
        # Test our modules
        from src.models.federated_bert import FederatedBERTConfig, create_federated_bert_models
        print("✅ FederatedBERT models imported")
        
        from src.models.knowledge_transfer import ProgressiveTransferConfig, DynamicAlignmentConfig
        print("✅ Knowledge transfer modules imported")
        
        from src.aggregation.fedavg import create_fedavg_aggregator
        print("✅ Aggregation modules imported")
        
        from src.server.flower_server import create_flower_server
        print("✅ Server modules imported")
        
        from src.clients.flower_client import client_fn
        print("✅ Client modules imported")
        
        from src.utils.training_utils import setup_logging, set_seed
        print("✅ Training utilities imported")
        
        from src.utils.data_utils import prepare_glue_data
        print("✅ Data utilities imported")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality"""
    
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test model creation
        from src.models.federated_bert import FederatedBERTConfig, create_federated_bert_models
        
        config = FederatedBERTConfig(num_labels=2)
        server_model, client_model = create_federated_bert_models(config)
        print("✅ Models created successfully")
        
        # Test parameter extraction
        server_params = server_model.get_lora_parameters()
        client_params = client_model.get_lora_parameters()
        print(f"✅ Parameter extraction works (server: {len(server_params)}, client: {len(client_params)})")
        
        # Test aggregator
        from src.aggregation.fedavg import create_fedavg_aggregator
        aggregator = create_fedavg_aggregator()
        print("✅ Aggregator created successfully")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 FedBERT-LoRA Import Test")
    print("=" * 30)
    
    success = test_imports()
    
    if success:
        success = test_basic_functionality()
    
    if success:
        print("\n✅ All tests passed! Your setup is working correctly.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        exit(1)
