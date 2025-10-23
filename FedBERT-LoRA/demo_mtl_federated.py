#!/usr/bin/env python3
"""
Demo script for MTL Federated Learning System
Shows how to use the new MultiTaskFederatedClient and MTLFederatedServer
"""

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from no_lora_federated_system import (
    NoLoRAConfig,
    MultiTaskFederatedClient,
    MTLFederatedServer,
    run_mtl_federated_experiment
)

def demo_configuration():
    """Show example configurations for MTL federated learning"""
    print(" MTL Federated Learning Configuration Examples")
    print("=" * 50)

    # Example 1: Basic MTL Federated Learning
    print("\n Example 1: Basic MTL Federated Learning")
    config1 = NoLoRAConfig(
        server_model="prajjwal1/bert-tiny",
        client_model="prajjwal1/bert-tiny",
        num_rounds=22,
        min_clients=2,
        max_clients=5,
        local_epochs=3,
        batch_size=16,
        learning_rate=5e-5,
        distillation_temperature=4.0,
        distillation_alpha=0.7,
        data_samples_per_client=2000,
        data_distribution="non_iid",
        non_iid_alpha=0.5,
        port=8771
    )
    print(f"   Rounds: {config1.num_rounds}")
    print(f"   Clients: {config1.min_clients}-{config1.max_clients}")
    print(f"   Learning Rate: {config1.learning_rate}")
    print(f"   Non-IID Alpha: {config1.non_iid_alpha}")

    # Example 2: High Heterogeneity MTL
    print("\n Example 2: High Heterogeneity MTL")
    config2 = NoLoRAConfig(
        server_model="prajjwal1/bert-tiny",
        client_model="prajjwal1/bert-tiny",
        num_rounds=30,
        min_clients=3,
        max_clients=8,
        local_epochs=2,
        batch_size=16,
        learning_rate=3e-5,
        distillation_temperature=3.0,
        distillation_alpha=0.8,
        data_samples_per_client=1500,
        data_distribution="non_iid",
        non_iid_alpha=1.0,  # High heterogeneity
        port=8772
    )
    print(f"   High Heterogeneity Alpha: {config2.non_iid_alpha}")
    print(f"   More Rounds: {config2.num_rounds}")
    print(f"   Higher KD Weight: {config2.distillation_alpha}")

    return config1, config2

def demo_client_usage():
    """Show how to use MTL Federated Client"""
    print("\n👤 MTL Federated Client Usage Examples")
    print("=" * 50)

    # Example client initialization
    print("\n Client Initialization:")
    print("```python")
    print("from no_lora_federated_system import MultiTaskFederatedClient, NoLoRAConfig")
    print("")
    print("# Create configuration")
    print("config = NoLoRAConfig(")
    print("    client_model='prajjwal1/bert-tiny',")
    print("    server_model='prajjwal1/bert-tiny',")
    print("    num_rounds=22,")
    print("    data_samples_per_client=2000,")
    print("    data_distribution='non_iid',")
    print("    non_iid_alpha=0.5")
    print(")")
    print("")
    print("# Create MTL client with multiple tasks")
    print("client = MultiTaskFederatedClient(")
    print("    client_id='client_1',")
    print("    tasks=['sst2', 'qqp', 'stsb'],")
    print("    config=config,")
    print("    total_clients=5")
    print(")")
    print("```")

    # Example command line usage
    print("\n Command Line Usage:")
    print("```bash")
    print("# Start server")
    print("python no_lora_federated_system.py --mode server --rounds 22 --total_clients 5")
    print("")
    print("# Start client")
    print("python no_lora_federated_system.py --mode client --client_id client_1 --tasks sst2 qqp stsb")
    print("```")

def demo_key_features():
    """Highlight key features of the MTL federated system"""
    print("\n✨ Key Features of MTL Federated Learning System")
    print("=" * 50)

    features = [
        " Multi-Task Learning: Each client handles multiple tasks simultaneously",
        " Federated Learning: Knowledge sharing across distributed clients",
        " Knowledge Distillation: Teacher-student learning between tasks",
        " Non-IID Distribution: Realistic heterogeneous data across clients",
        " Comprehensive Metrics: Detailed performance tracking per task",
        "⚡ Transfer Learning: Knowledge transfer between related tasks",
        " Flexible Configuration: Easy customization of all parameters",
        " Advanced Binning: Sophisticated data distribution for regression tasks"
    ]

    for feature in features:
        print(f"   {feature}")

def demo_architecture():
    """Show the architecture of the MTL federated system"""
    print("\n🏗️ MTL Federated Learning Architecture")
    print("=" * 50)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                    MTL Federated Learning System                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Client 1   │  │  Client 2   │  │  Client 3   │              │
│  │             │  │             │  │             │              │
│  │ SST2 QQP    │  │ SST2 STSB   │  │ QQP STSB    │              │
│  │ STSB        │  │ QQP         │  │ SST2        │              │
│  │             │  │             │  │             │              │
│  │ • Model     │  │ • Model     │  │ • Model     │              │
│  │ • Optimizer │  │ • Optimizer │  │ • Optimizer │              │
│  │ • Data      │  │ • Data      │  │ • Data      │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                 MTL Federated Server                        │  │
│  │                                                         │  │
│  │ • Global Model (Teacher)                                │  │
│  │ • Client Coordination                                   │  │
│  │ • Parameter Aggregation                                 │  │
│  │ • Metrics Collection                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Knowledge Distillation ←→ Transfer Learning ←→ Federation    │
└─────────────────────────────────────────────────────────────────┘
    """)

async def main():
    """Main demo function"""
    print(" MTL Federated Learning System Demo")
    print("=" * 60)

    # Show configuration examples
    demo_configuration()

    # Show client usage
    demo_client_usage()

    # Show key features
    demo_key_features()

    # Show architecture
    demo_architecture()

    print("\n Summary")
    print("=" * 60)
    print(" Multi-Task Learning: Each client learns multiple tasks")
    print(" Federated Learning: Distributed training across clients")
    print(" Transfer Learning: Knowledge sharing between tasks")
    print(" Knowledge Distillation: Teacher-student learning")
    print(" Non-IID Distribution: Realistic heterogeneous data")
    print(" Comprehensive Metrics: Detailed performance tracking")

    print("\n The system combines the benefits of:")
    print("   • Multi-task learning for better generalization")
    print("   • Federated learning for privacy-preserving distributed training")
    print("   • Transfer learning for knowledge sharing between tasks")
    print("   • Knowledge distillation for model compression and learning")

    print("\n For more information, see:")
    print("   • MULTI_TASK_IMPLEMENTATION.md")
    print("   • FEDERATED_LEARNING_COMPARISON.md")
    print("   • Run: python test_mtl_federated.py")

if __name__ == "__main__":
    asyncio.run(main())
