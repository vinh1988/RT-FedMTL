#!/usr/bin/env python3
"""
Minimal test script to verify the federated learning setup works
"""

import os
import sys
import logging

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_minimal_federated_learning():
    """Test minimal federated learning setup"""
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting minimal federated learning test")
    
    try:
        # Import core components
        from src.models.federated_bert import FederatedBERTConfig, create_federated_bert_models
        from src.aggregation.fedavg import create_fedavg_aggregator
        
        logger.info("✅ Core imports successful")
        
        # Create models
        config = FederatedBERTConfig(num_labels=2)
        server_model, client_model = create_federated_bert_models(config)
        
        logger.info("✅ Models created")
        
        # Test parameter extraction
        server_params = server_model.get_lora_parameters()
        client_params = client_model.get_lora_parameters()
        
        logger.info(f"✅ Parameter extraction: Server={len(server_params)}, Client={len(client_params)}")
        
        # Create aggregator
        aggregator = create_fedavg_aggregator()
        
        logger.info("✅ Aggregator created")
        
        # Simulate simple aggregation
        client_info = [{"data_size": 100}, {"data_size": 150}]
        dummy_params1 = {name: param.clone() for name, param in client_params.items()}
        dummy_params2 = {name: param.clone() + 0.01 for name, param in client_params.items()}
        
        aggregated = aggregator.aggregate([dummy_params1, dummy_params2], client_info)
        
        logger.info(f"✅ Aggregation successful: {len(aggregated)} parameters")
        
        logger.info("🎉 Minimal federated learning test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🧪 Minimal FedBERT-LoRA Test")
    print("=" * 30)
    
    setup_logging()
    
    success = test_minimal_federated_learning()
    
    if success:
        print("\n✅ SUCCESS: Core federated learning components work!")
        print("\nYou can now try:")
        print("  python examples/run_simple_experiment.py")
    else:
        print("\n❌ FAILED: Check the errors above")
        print("\nTry running:")
        print("  python quick_fix.py")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
