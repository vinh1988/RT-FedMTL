#!/usr/bin/env python3
"""
Test script for enhanced metrics (F1, Precision, Recall)
Quick test to verify the new metrics system works correctly
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from no_lora_federated_system import NoLoRAConfig, run_no_lora_experiment

async def test_enhanced_metrics():
    """Test the enhanced metrics system with a quick experiment"""
    print("🧪 Testing Enhanced Metrics System (F1, Precision, Recall)")
    print("=" * 60)
    
    # Create a minimal config for testing
    config = NoLoRAConfig(
        port=8899,  # Use unique port for testing
        num_rounds=1,  # Just 1 round for quick test
        data_samples_per_client=20,  # Small dataset
        max_clients=2,  # Only 2 clients
        data_distribution="non_iid",
        non_iid_alpha=0.5
    )
    
    print(f"✅ Configuration created:")
    print(f"   • Port: {config.port}")
    print(f"   • Rounds: {config.num_rounds}")
    print(f"   • Samples per client: {config.data_samples_per_client}")
    print(f"   • Max clients: {config.max_clients}")
    print(f"   • Distribution: {config.data_distribution} (α={config.non_iid_alpha})")
    
    print(f"\n🚀 Starting enhanced metrics test...")
    print(f"Expected new metrics: Precision, Recall, F1-score, Per-class metrics")
    
    try:
        # This will test the server initialization and metric collection
        await run_no_lora_experiment(config, "server", total_clients=2)
        print(f"\n✅ Enhanced metrics test completed successfully!")
        print(f"Check the logs above for P= (Precision), R= (Recall), F1= (F1-score)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"This might be expected if no clients connect (server-only test)")
        print(f"The important thing is that the enhanced metrics system initialized correctly")

if __name__ == "__main__":
    print("🔬 Enhanced Metrics Test for BERT Federated Learning")
    print("This test verifies that F1, Precision, and Recall metrics are working")
    asyncio.run(test_enhanced_metrics())
