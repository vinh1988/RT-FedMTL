#!/usr/bin/env python3
"""
Quick fix for FedBERT-LoRA import issues
"""

import os
import sys

def main():
    """Quick fix for import issues"""
    
    print("🔧 Quick Fix for FedBERT-LoRA")
    print("=" * 30)
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Add to Python path
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    print(f"📁 Working directory: {script_dir}")
    
    try:
        # Test basic imports
        print("🧪 Testing basic imports...")
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import flwr
        print(f"✅ Flower {flwr.__version__}")
        
        # Test our core modules
        print("\n🧪 Testing core modules...")
        
        from src.models.federated_bert import FederatedBERTConfig
        print("✅ FederatedBERT config")
        
        from src.aggregation.fedavg import LoRAFedAvgAggregator
        print("✅ LoRA aggregator")
        
        from src.server.flower_server import create_flower_server
        print("✅ Flower server")
        
        from src.clients.flower_client import client_fn
        print("✅ Flower client")
        
        print("\n🎉 All core imports working!")
        
        # Test simple model creation
        print("\n🧪 Testing model creation...")
        config = FederatedBERTConfig(num_labels=2)
        print("✅ Config created")
        
        # Test server creation
        print("\n🧪 Testing server creation...")
        server = create_flower_server(
            num_rounds=1,
            num_clients=2,
            enable_knowledge_transfer=False
        )
        print("✅ Server created")
        
        print("\n🎉 Quick fix successful! Your setup is working.")
        print("\nNow you can run:")
        print("  python examples/run_simple_experiment.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Suggested fixes:")
        print("1. Make sure you're in the virtual environment:")
        print("   source venv/bin/activate")
        print("2. Reinstall dependencies:")
        print("   pip install -r requirements.txt")
        print("3. Install in development mode:")
        print("   pip install -e .")
        
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
