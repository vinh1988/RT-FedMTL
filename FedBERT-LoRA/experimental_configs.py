#!/usr/bin/env python3
"""
Experimental Configurations for Deep Research Federated Learning
Defines comprehensive experimental setups for LoRA vs non-LoRA comparison
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

@dataclass
class ExperimentConfig:
    """Base configuration for federated learning experiments"""
    
    # Experiment identification
    experiment_name: str
    experiment_type: str  # "lora" or "no_lora"
    description: str
    
    # Model configuration
    server_model: str = "bert-base-uncased"
    client_model: str = "prajjwal1/bert-tiny"
    
    # Federated learning parameters
    num_rounds: int = 22
    min_clients: int = 2
    max_clients: int = 10
    client_selection_strategy: str = "all"  # "all", "random", "stratified"
    
    # Training parameters
    local_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Data configuration
    tasks: List[str] = None  # ["sst2", "qqp", "stsb"]
    data_samples_per_client: int = 1000
    data_distribution: str = "iid"  # "iid", "non_iid", "pathological"
    
    # LoRA configuration (only for LoRA experiments)
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Knowledge Distillation configuration
    use_knowledge_distillation: bool = True
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    # Communication configuration
    compression_method: str = "none"  # "none", "quantization", "sparsification"
    communication_frequency: int = 1  # Communicate every N local epochs
    
    # Evaluation configuration
    evaluation_frequency: int = 1  # Evaluate every N rounds
    save_checkpoints: bool = True
    detailed_logging: bool = True
    
    # System configuration
    device: str = "auto"  # "auto", "cpu", "cuda"
    random_seed: int = 42
    port: int = 8769
    timeout: int = 300
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["sst2", "qqp", "stsb"]
        if self.lora_target_modules is None:
            self.lora_target_modules = ["query", "key", "value", "dense"]

# Predefined experimental configurations for research

# LoRA Experiments
LORA_BASELINE_CONFIG = ExperimentConfig(
    experiment_name="lora_baseline",
    experiment_type="lora",
    description="Baseline LoRA federated learning with standard parameters",
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    num_rounds=22,
    data_samples_per_client=1000
)

LORA_HIGH_RANK_CONFIG = ExperimentConfig(
    experiment_name="lora_high_rank",
    experiment_type="lora",
    description="LoRA with higher rank for better performance",
    use_lora=True,
    lora_rank=64,
    lora_alpha=128,
    num_rounds=22,
    data_samples_per_client=1000
)

LORA_LOW_RANK_CONFIG = ExperimentConfig(
    experiment_name="lora_low_rank",
    experiment_type="lora",
    description="LoRA with lower rank for maximum efficiency",
    use_lora=True,
    lora_rank=8,
    lora_alpha=16,
    num_rounds=22,
    data_samples_per_client=1000
)

LORA_SCALABILITY_CONFIG = ExperimentConfig(
    experiment_name="lora_scalability",
    experiment_type="lora",
    description="LoRA scalability test with varying client numbers",
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    num_rounds=22,
    min_clients=2,
    max_clients=10,
    data_samples_per_client=500  # Smaller datasets for scalability test
)

# Non-LoRA Experiments
NO_LORA_BASELINE_CONFIG = ExperimentConfig(
    experiment_name="no_lora_baseline",
    experiment_type="no_lora",
    description="Baseline federated learning without LoRA",
    use_lora=False,
    num_rounds=22,
    data_samples_per_client=1000
)

NO_LORA_FULL_TRAINING_CONFIG = ExperimentConfig(
    experiment_name="no_lora_full_training",
    experiment_type="no_lora",
    description="Full parameter training without LoRA",
    use_lora=False,
    local_epochs=5,  # More epochs to compensate for full training
    learning_rate=1e-5,  # Lower learning rate for stability
    num_rounds=22,
    data_samples_per_client=1000
)

NO_LORA_SCALABILITY_CONFIG = ExperimentConfig(
    experiment_name="no_lora_scalability",
    experiment_type="no_lora",
    description="Non-LoRA scalability test with varying client numbers",
    use_lora=False,
    num_rounds=22,
    min_clients=2,
    max_clients=10,
    data_samples_per_client=500
)

# Comparative Experiments
COMPARATIVE_EFFICIENCY_CONFIG = ExperimentConfig(
    experiment_name="comparative_efficiency",
    experiment_type="comparative",
    description="Direct comparison of LoRA vs non-LoRA efficiency",
    num_rounds=22,
    data_samples_per_client=1000,
    detailed_logging=True,
    save_checkpoints=True
)

COMPARATIVE_ACCURACY_CONFIG = ExperimentConfig(
    experiment_name="comparative_accuracy",
    experiment_type="comparative",
    description="Direct comparison of LoRA vs non-LoRA accuracy",
    num_rounds=22,
    local_epochs=5,  # More training for accuracy comparison
    data_samples_per_client=2000,  # More data for better accuracy assessment
    evaluation_frequency=1
)

# Knowledge Distillation Experiments
KD_TEMPERATURE_SWEEP_CONFIG = ExperimentConfig(
    experiment_name="kd_temperature_sweep",
    experiment_type="kd_analysis",
    description="Knowledge distillation temperature sensitivity analysis",
    use_lora=True,
    distillation_temperature=4.0,  # Will be varied in experiment
    num_rounds=15,  # Shorter for parameter sweep
    data_samples_per_client=500
)

KD_ALPHA_SWEEP_CONFIG = ExperimentConfig(
    experiment_name="kd_alpha_sweep",
    experiment_type="kd_analysis",
    description="Knowledge distillation alpha parameter sensitivity",
    use_lora=True,
    distillation_alpha=0.7,  # Will be varied in experiment
    num_rounds=15,
    data_samples_per_client=500
)

# Data Heterogeneity Experiments
NON_IID_LORA_CONFIG = ExperimentConfig(
    experiment_name="non_iid_lora",
    experiment_type="heterogeneity",
    description="LoRA performance under non-IID data distribution",
    use_lora=True,
    data_distribution="non_iid",
    num_rounds=25,  # More rounds for non-IID convergence
    data_samples_per_client=800
)

NON_IID_NO_LORA_CONFIG = ExperimentConfig(
    experiment_name="non_iid_no_lora",
    experiment_type="heterogeneity",
    description="Non-LoRA performance under non-IID data distribution",
    use_lora=False,
    data_distribution="non_iid",
    num_rounds=25,
    data_samples_per_client=800
)

# Communication Efficiency Experiments
COMMUNICATION_LORA_CONFIG = ExperimentConfig(
    experiment_name="communication_lora",
    experiment_type="communication",
    description="Communication efficiency analysis with LoRA",
    use_lora=True,
    communication_frequency=2,  # Communicate every 2 epochs
    num_rounds=30,  # More rounds to test communication patterns
    data_samples_per_client=600
)

COMMUNICATION_NO_LORA_CONFIG = ExperimentConfig(
    experiment_name="communication_no_lora",
    experiment_type="communication",
    description="Communication efficiency analysis without LoRA",
    use_lora=False,
    communication_frequency=2,
    num_rounds=30,
    data_samples_per_client=600
)

# Comprehensive Research Suite
RESEARCH_EXPERIMENT_SUITE = {
    "lora_experiments": [
        LORA_BASELINE_CONFIG,
        LORA_HIGH_RANK_CONFIG,
        LORA_LOW_RANK_CONFIG,
        LORA_SCALABILITY_CONFIG
    ],
    "no_lora_experiments": [
        NO_LORA_BASELINE_CONFIG,
        NO_LORA_FULL_TRAINING_CONFIG,
        NO_LORA_SCALABILITY_CONFIG
    ],
    "comparative_experiments": [
        COMPARATIVE_EFFICIENCY_CONFIG,
        COMPARATIVE_ACCURACY_CONFIG
    ],
    "kd_experiments": [
        KD_TEMPERATURE_SWEEP_CONFIG,
        KD_ALPHA_SWEEP_CONFIG
    ],
    "heterogeneity_experiments": [
        NON_IID_LORA_CONFIG,
        NON_IID_NO_LORA_CONFIG
    ],
    "communication_experiments": [
        COMMUNICATION_LORA_CONFIG,
        COMMUNICATION_NO_LORA_CONFIG
    ]
}

class ExperimentManager:
    """Manages experimental configurations and execution"""
    
    def __init__(self, save_dir: str = "experiments"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: ExperimentConfig) -> Path:
        """Save experiment configuration to file"""
        config_file = self.save_dir / f"{config.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        return config_file
    
    def load_config(self, config_file: Path) -> ExperimentConfig:
        """Load experiment configuration from file"""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return ExperimentConfig(**config_dict)
    
    def create_parameter_sweep(self, base_config: ExperimentConfig, 
                             parameter_name: str, values: List[Any]) -> List[ExperimentConfig]:
        """Create parameter sweep experiments"""
        sweep_configs = []
        
        for i, value in enumerate(values):
            config_dict = asdict(base_config)
            config_dict[parameter_name] = value
            config_dict["experiment_name"] = f"{base_config.experiment_name}_{parameter_name}_{i}"
            config_dict["description"] = f"{base_config.description} - {parameter_name}={value}"
            
            sweep_configs.append(ExperimentConfig(**config_dict))
        
        return sweep_configs
    
    def create_client_scalability_sweep(self, base_config: ExperimentConfig, 
                                      client_counts: List[int]) -> List[ExperimentConfig]:
        """Create client scalability experiments"""
        scalability_configs = []
        
        for client_count in client_counts:
            config_dict = asdict(base_config)
            config_dict["max_clients"] = client_count
            config_dict["min_clients"] = min(2, client_count)
            config_dict["experiment_name"] = f"{base_config.experiment_name}_clients_{client_count}"
            config_dict["description"] = f"{base_config.description} - {client_count} clients"
            
            scalability_configs.append(ExperimentConfig(**config_dict))
        
        return scalability_configs
    
    def get_comparative_configs(self) -> Dict[str, List[ExperimentConfig]]:
        """Get matched pairs of LoRA vs non-LoRA configurations for comparison"""
        
        comparative_pairs = {
            "baseline_comparison": [LORA_BASELINE_CONFIG, NO_LORA_BASELINE_CONFIG],
            "scalability_comparison": [LORA_SCALABILITY_CONFIG, NO_LORA_SCALABILITY_CONFIG],
            "communication_comparison": [COMMUNICATION_LORA_CONFIG, COMMUNICATION_NO_LORA_CONFIG],
            "heterogeneity_comparison": [NON_IID_LORA_CONFIG, NON_IID_NO_LORA_CONFIG]
        }
        
        return comparative_pairs
    
    def save_experiment_suite(self, suite_name: str = "research_suite"):
        """Save all experimental configurations"""
        suite_dir = self.save_dir / suite_name
        suite_dir.mkdir(exist_ok=True)
        
        saved_configs = {}
        
        for category, configs in RESEARCH_EXPERIMENT_SUITE.items():
            category_dir = suite_dir / category
            category_dir.mkdir(exist_ok=True)
            
            saved_configs[category] = []
            for config in configs:
                config_file = category_dir / f"{config.experiment_name}.json"
                with open(config_file, 'w') as f:
                    json.dump(asdict(config), f, indent=2)
                saved_configs[category].append(str(config_file))
        
        # Save suite index
        suite_index = suite_dir / "suite_index.json"
        with open(suite_index, 'w') as f:
            json.dump(saved_configs, f, indent=2)
        
        return suite_dir

# Suggested metrics for publication based on experiment type
PUBLICATION_METRICS_BY_EXPERIMENT = {
    "lora": [
        "parameter_efficiency_ratio",
        "communication_cost_reduction", 
        "training_time_speedup",
        "accuracy_retention_rate",
        "memory_usage_reduction"
    ],
    "no_lora": [
        "baseline_accuracy",
        "convergence_rate",
        "communication_overhead",
        "computational_cost",
        "scalability_performance"
    ],
    "comparative": [
        "accuracy_comparison",
        "efficiency_comparison", 
        "communication_comparison",
        "scalability_comparison",
        "convergence_comparison"
    ],
    "kd_analysis": [
        "knowledge_transfer_efficiency",
        "teacher_student_gap",
        "distillation_loss_analysis",
        "temperature_sensitivity",
        "alpha_parameter_impact"
    ],
    "heterogeneity": [
        "non_iid_performance",
        "data_heterogeneity_impact",
        "convergence_under_heterogeneity",
        "fairness_metrics",
        "robustness_analysis"
    ],
    "communication": [
        "communication_rounds_analysis",
        "bandwidth_utilization",
        "latency_impact",
        "compression_effectiveness",
        "communication_frequency_optimization"
    ]
}

def get_recommended_metrics(experiment_type: str) -> List[str]:
    """Get recommended metrics for publication based on experiment type"""
    return PUBLICATION_METRICS_BY_EXPERIMENT.get(experiment_type, [])

def create_full_research_suite() -> Dict[str, Any]:
    """Create complete research suite with all configurations"""
    
    manager = ExperimentManager()
    
    # Create parameter sweeps
    lora_rank_sweep = manager.create_parameter_sweep(
        LORA_BASELINE_CONFIG, "lora_rank", [4, 8, 16, 32, 64]
    )
    
    kd_temperature_sweep = manager.create_parameter_sweep(
        KD_TEMPERATURE_SWEEP_CONFIG, "distillation_temperature", [1.0, 2.0, 4.0, 8.0, 16.0]
    )
    
    kd_alpha_sweep = manager.create_parameter_sweep(
        KD_ALPHA_SWEEP_CONFIG, "distillation_alpha", [0.1, 0.3, 0.5, 0.7, 0.9]
    )
    
    # Create scalability sweeps
    client_scalability_lora = manager.create_client_scalability_sweep(
        LORA_SCALABILITY_CONFIG, [2, 4, 6, 8, 10]
    )
    
    client_scalability_no_lora = manager.create_client_scalability_sweep(
        NO_LORA_SCALABILITY_CONFIG, [2, 4, 6, 8, 10]
    )
    
    # Complete research suite
    full_suite = {
        **RESEARCH_EXPERIMENT_SUITE,
        "parameter_sweeps": {
            "lora_rank_sweep": lora_rank_sweep,
            "kd_temperature_sweep": kd_temperature_sweep,
            "kd_alpha_sweep": kd_alpha_sweep
        },
        "scalability_sweeps": {
            "lora_scalability": client_scalability_lora,
            "no_lora_scalability": client_scalability_no_lora
        }
    }
    
    return full_suite

if __name__ == "__main__":
    # Example usage
    manager = ExperimentManager()
    
    # Save individual configs
    manager.save_config(LORA_BASELINE_CONFIG)
    manager.save_config(NO_LORA_BASELINE_CONFIG)
    
    # Save complete suite
    suite_dir = manager.save_experiment_suite()
    print(f"Research suite saved to: {suite_dir}")
    
    # Create and save parameter sweeps
    full_suite = create_full_research_suite()
    print(f"Created {sum(len(configs) for configs in full_suite.values())} total configurations")
