#!/usr/bin/env python3
"""
Test script to validate multi-GPU configuration
"""
import os
# Ensure both GPUs are visible
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import sys
import torch
from federated_config import load_config
import argparse

def test_multi_gpu_config():
    """Test multi-GPU configuration loading and device assignment"""
    
    # Create minimal args for config loading
    args = argparse.Namespace()
    args.config = None
    args.config_file = "federated_config.yaml"
    
    try:
        # Load configuration
        config = load_config(args)
        print("✅ Configuration loaded successfully")
        
        # Check GPU config
        gpu_config = config.gpu_config
        print(f"✅ Multi-GPU enabled: {gpu_config.enable_multi_gpu}")
        print(f"✅ Assignment strategy: {gpu_config.gpu_assignment_strategy}")
        print(f"✅ Manual assignments: {gpu_config.manual_gpu_assignments}")
        
        # Test device assignment for different clients
        from src.core.federated_client import FederatedClient
        
        test_clients = ["sst2_client", "qqp_client", "stsb_client"]
        
        for client_id in test_clients:
            print(f"\n🔧 Testing device assignment for {client_id}:")
            
            # Create a mock client (we'll just test the device assignment logic)
            class MockClient:
                def __init__(self, client_id, config):
                    self.client_id = client_id
                    self.config = config
                    self.device = self.get_device()
                
                def get_device(self):
                    """Copy of the get_device method from FederatedClient"""
                    if not torch.cuda.is_available():
                        print(f"   ❌ CUDA not available - using CPU only")
                        return torch.device("cpu")
                    
                    # Check if multi-GPU is enabled
                    gpu_config = getattr(self.config, 'gpu_config', None)
                    if not gpu_config or not gpu_config.enable_multi_gpu:
                        # Default behavior: use GPU 0
                        device = torch.device("cuda:0")
                        print(f"   ✅ Using default GPU 0: {device}")
                        return device
                    
                    # Multi-GPU assignment logic
                    num_gpus = torch.cuda.device_count()
                    print(f"   📊 Found {num_gpus} GPUs")
                    
                    # Choose GPU based on strategy
                    if gpu_config.gpu_assignment_strategy == "manual":
                        gpu_id = self._get_manual_gpu_assignment(gpu_config)
                    elif gpu_config.gpu_assignment_strategy == "least_loaded":
                        gpu_id = self._get_least_loaded_gpu()
                    else:  # round_robin (default)
                        gpu_id = self._get_round_robin_gpu_assignment(num_gpus)
                    
                    # Validate GPU ID
                    if gpu_id >= num_gpus:
                        print(f"   ⚠️ GPU {gpu_id} not available. Using GPU 0 instead.")
                        gpu_id = 0
                    
                    device = torch.device(f"cuda:{gpu_id}")
                    print(f"   ✅ Assigned to GPU {gpu_id}: {device}")
                    return device
                
                def _get_manual_gpu_assignment(self, gpu_config) -> int:
                    """Get GPU ID from manual assignment configuration"""
                    if self.client_id in gpu_config.manual_gpu_assignments:
                        gpu_id = gpu_config.manual_gpu_assignments[self.client_id]
                        print(f"   📋 Manual assignment: {self.client_id} -> GPU {gpu_id}")
                        return gpu_id
                    else:
                        print(f"   ⚠️ No manual assignment for {self.client_id}, using round-robin")
                        return self._get_round_robin_gpu_assignment(torch.cuda.device_count())
                
                def _get_round_robin_gpu_assignment(self, num_gpus: int) -> int:
                    """Get GPU ID using round-robin assignment based on client_id hash"""
                    client_hash = hash(self.client_id) % num_gpus
                    gpu_id = abs(client_hash)  # Ensure positive
                    print(f"   🔄 Round-robin assignment: {self.client_id} -> GPU {gpu_id}")
                    return gpu_id
                
                def _get_least_loaded_gpu(self) -> int:
                    """Get GPU ID with least memory usage"""
                    min_memory_usage = float('inf')
                    best_gpu_id = 0
                    
                    for gpu_id in range(torch.cuda.device_count()):
                        memory_used = torch.cuda.memory_allocated(gpu_id) / 1e9  # GB
                        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9  # GB
                        memory_usage_percent = (memory_used / memory_total) * 100
                        
                        if memory_usage_percent < min_memory_usage:
                            min_memory_usage = memory_usage_percent
                            best_gpu_id = gpu_id
                    
                    print(f"   📊 Least-loaded assignment: {self.client_id} -> GPU {best_gpu_id} ({min_memory_usage:.1f}% used)")
                    return best_gpu_id
            
            client = MockClient(client_id, config)
            
            # Test tensor placement
            if client.device.type == 'cuda':
                test_tensor = torch.randn(1000, 1000).to(client.device)
                print(f"   ✅ Test tensor created on {client.device}, shape: {test_tensor.shape}")
                del test_tensor
                torch.cuda.empty_cache()
        
        print(f"\n🎉 Multi-GPU configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing multi-GPU configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_gpu_config()
    sys.exit(0 if success else 1)
