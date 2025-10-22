#!/usr/bin/env python3
"""
Test script for the fixed local MTL training system
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports work correctly"""
    try:
        from federated_config import load_config
        print("✅ Config import works")

        from src.lora.federated_lora import LoRAFederatedModel
        print("✅ LoRA import works")

        from src.knowledge_distillation.federated_knowledge_distillation import BidirectionalKDManager
        print("✅ KD import works")

        from dataset_factory import DatasetFactory
        print("✅ DatasetFactory import works")

        from src.training.local_trainer import LocalTrainer
        print("✅ LocalTrainer import works")

        print("\n🎉 All imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_syntax():
    """Test that the main script has no syntax errors"""
    try:
        import local_mtl_main
        print("✅ Main script syntax is valid!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in main script: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error in main script: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without full training"""
    try:
        from federated_config import FederatedConfig
        from dataset_factory import DatasetFactory

        # Create a minimal config
        config = FederatedConfig()
        config.client_model = "prajjwal1/bert-tiny"
        config.lora_rank = 8
        config.lora_alpha = 16.0
        config.learning_rate = 2e-5
        config.kd_temperature = 3.0
        config.kd_alpha = 0.5

        # Test dataset factory
        handler = DatasetFactory.create_handler("sst2", config)
        dataloader = handler.get_dataloader()

        # Get one batch
        for batch in dataloader:
            print(f"✅ Dataset works - batch shape: {batch['input_ids'].shape}")
            break

        print("✅ Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Fixed Local MTL Training System")
    print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    # Test syntax
    syntax_ok = test_syntax()

    if imports_ok and syntax_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()

        if functionality_ok:
            print("\n🎉 All tests passed! System is ready for local MTL training.")
            print("\nTo run training:")
            print("source venv/bin/activate")
            print("python3 local_mtl_main.py --tasks sst2 qqp stsb --rounds 3")
        else:
            print("\n⚠️ Some functionality tests failed.")
    else:
        print("\n❌ Import or syntax tests failed. Please check the fixes.")
