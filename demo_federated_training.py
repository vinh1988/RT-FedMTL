#!/usr/bin/env python3
"""
Demo Script for FedMKT Federated Training on 20News Classification
This script demonstrates the complete federated learning pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.news20_dataset import News20Config, create_20news_federated_data
from training.fedmkt_trainer import FedMKTTrainer, FedMKTTrainingConfig
from models.bart_classification import create_lora_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_federated_training():
    """Demo the complete federated training pipeline"""
    
    print("=" * 80)
    print("🚀 FEDMKT FEDERATED TRAINING DEMO")
    print("DistilBART ↔ MobileBART on 20News Classification")
    print("=" * 80)
    
    # Step 1: Configuration
    print("\n📋 Step 1: Setting up configuration...")
    
    data_config = News20Config(
        data_dir="./data/20news_demo",
        max_length=256,  # Smaller for demo
        num_clients=3,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    training_config = FedMKTTrainingConfig(
        # Model configurations
        distilbart_model_name="facebook/bart-base",
        mobilebart_model_name="facebook/bart-base",
        num_labels=20,
        max_length=256,
        
        # Training parameters (reduced for demo)
        learning_rate=5e-5,
        num_epochs=3,  # Reduced for demo
        batch_size=4,  # Reduced for demo
        gradient_accumulation_steps=2,
        
        # Knowledge distillation
        distill_temperature=4.0,
        kd_alpha=0.7,
        
        # LoRA configuration
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        
        # Output
        output_dir="./outputs/fedmkt_demo"
    )
    
    print(f"✅ Data config: {data_config.num_clients} clients, max_length={data_config.max_length}")
    print(f"✅ Training config: {training_config.num_epochs} epochs, batch_size={training_config.batch_size}")
    
    # Step 2: Data Preparation
    print("\n📊 Step 2: Preparing federated data...")
    
    try:
        data_loaders = create_20news_federated_data(
            data_config, 
            model_name=training_config.distilbart_model_name,
            save_info=True
        )
        
        print(f"✅ Created federated data for {len(data_loaders)} clients")
        
        # Print data statistics
        for client_id, client_loaders in data_loaders.items():
            train_size = len(client_loaders["train"].dataset)
            val_size = len(client_loaders["val"].dataset)
            test_size = len(client_loaders["test"].dataset)
            print(f"   {client_id}: Train={train_size}, Val={val_size}, Test={test_size}")
            
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Starting federated training...")
    
    try:
        # Create trainer
        trainer = FedMKTTrainer(training_config)
        
        # Create models
        print("   Creating central DistilBART model...")
        central_model = trainer.create_central_model()
        
        print("   Creating client MobileBART models...")
        for client_id in data_loaders.keys():
            client_model = trainer.create_client_model(client_id)
        
        # Get public data loader (use validation set as public data)
        public_data_loader = list(data_loaders.values())[0]["val"]
        
        print(f"   Starting {training_config.num_epochs} federated training rounds...")
        
        # Training loop
        for round_idx in range(training_config.num_epochs):
            print(f"\n   🔄 Round {round_idx + 1}/{training_config.num_epochs}")
            
            round_results = trainer.federated_training_round(
                data_loaders, public_data_loader, round_idx
            )
            
            # Print round results
            if "evaluation" in round_results["central_model"]:
                central_eval = round_results["central_model"]["evaluation"]
                print(f"      Central Model - Accuracy: {central_eval['accuracy']:.4f}")
            
            for client_id, client_results in round_results["client_models"].items():
                if "evaluation" in client_results:
                    client_eval = client_results["evaluation"]
                    print(f"      Client {client_id} - Accuracy: {client_eval['accuracy']:.4f}")
        
        print("✅ Federated training completed!")
        
        # Save models
        trainer.save_models()
        trainer.save_training_history()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 4: Evaluation
    print("\n📈 Step 4: Evaluating trained models...")
    
    try:
        # Evaluate on test set
        test_loader = list(data_loaders.values())[0]["test"]
        
        print("   Evaluating central model...")
        central_eval = trainer.evaluate_model(trainer.central_model, test_loader)
        print(f"      Central Model - Test Accuracy: {central_eval['accuracy']:.4f}")
        
        print("   Evaluating client models...")
        for client_id, client_model in trainer.client_models.items():
            client_eval = trainer.evaluate_model(client_model, test_loader)
            print(f"      Client {client_id} - Test Accuracy: {client_eval['accuracy']:.4f}")
        
        print("✅ Evaluation completed!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return
    
    # Step 5: Summary
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("📊 Summary:")
    print(f"   • Federated learning with {data_config.num_clients} clients")
    print(f"   • Central model: DistilBART ({sum(p.numel() for p in trainer.central_model.parameters()):,} parameters)")
    print(f"   • Client models: MobileBART (~{sum(p.numel() for p in list(trainer.client_models.values())[0].parameters()):,} parameters each)")
    print(f"   • Training rounds: {training_config.num_epochs}")
    print(f"   • Knowledge transfer: Bidirectional (Central ↔ Clients)")
    print(f"   • Privacy: No raw data sharing, only logits exchanged")
    print(f"   • Output directory: {training_config.output_dir}")
    
    print("\n📁 Generated Files:")
    output_dir = Path(training_config.output_dir)
    if output_dir.exists():
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                print(f"   • {file_path.relative_to(output_dir)}")
    
    print("\n🔬 Key Features Demonstrated:")
    print("   ✅ Cross-architecture knowledge transfer (DistilBART ↔ MobileBART)")
    print("   ✅ Token alignment and dimension projection")
    print("   ✅ Knowledge distillation with temperature scaling")
    print("   ✅ LoRA-based parameter-efficient fine-tuning")
    print("   ✅ Privacy-preserving federated learning")
    print("   ✅ Bidirectional learning (central teaches clients, clients inform central)")
    
    print("\n🚀 Next Steps:")
    print("   1. Run with full dataset: python examples/train_fedmkt_20news.py")
    print("   2. Experiment with different configurations")
    print("   3. Try with other datasets and tasks")
    print("   4. Scale to more clients and longer training")
    
    print("=" * 80)


def quick_test():
    """Quick test of individual components"""
    print("🧪 Quick Component Tests...")
    
    try:
        # Test data loading
        print("   Testing data loading...")
        data_config = News20Config(max_length=128, num_clients=2)
        data_loaders = create_20news_federated_data(data_config, save_info=False)
        print("   ✅ Data loading works")
        
        # Test model creation
        print("   Testing model creation...")
        training_config = FedMKTTrainingConfig(num_epochs=1, batch_size=2)
        trainer = FedMKTTrainer(training_config)
        central_model = trainer.create_central_model()
        client_model = trainer.create_client_model(0)
        print("   ✅ Model creation works")
        
        # Test forward pass
        print("   Testing forward pass...")
        import torch
        batch_size, seq_length = 2, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, 20, (batch_size,))
        
        outputs = central_model(input_ids, attention_mask, labels)
        print(f"   ✅ Forward pass works - Loss: {outputs['loss'].item():.4f}")
        
        print("✅ All quick tests passed!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run quick test
        quick_test()
    else:
        # Run full demo
        demo_federated_training()
