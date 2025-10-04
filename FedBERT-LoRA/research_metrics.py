#!/usr/bin/env python3
"""
Comprehensive Metrics Collection System for Federated Learning Research
Tracks all essential metrics for academic publication
"""

import json
import csv
import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CommunicationMetrics:
    """Communication efficiency metrics"""
    round_num: int
    client_id: str
    
    # Parameter transmission
    parameters_sent_bytes: int
    parameters_received_bytes: int
    compression_ratio: float
    
    # Network metrics
    upload_time: float
    download_time: float
    total_communication_time: float
    network_latency: float
    
    # Efficiency metrics
    communication_rounds: int
    bandwidth_utilization: float

@dataclass
class ModelPerformanceMetrics:
    """Model performance and accuracy metrics"""
    round_num: int
    client_id: str
    task: str
    
    # Training performance
    train_accuracy: float
    train_loss: float
    train_f1_score: float
    train_precision: float
    train_recall: float
    
    # Validation performance (if available)
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_f1_score: Optional[float] = None
    
    # Task-specific metrics
    task_specific_metrics: Dict[str, float] = None
    
    # Model convergence
    loss_improvement: float = 0.0
    accuracy_improvement: float = 0.0
    convergence_indicator: float = 0.0

@dataclass
class ComputationalMetrics:
    """Computational efficiency and resource usage"""
    round_num: int
    client_id: str
    
    # Training time
    local_training_time: float
    forward_pass_time: float
    backward_pass_time: float
    optimization_time: float
    
    # Resource usage
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    gpu_utilization_percent: float
    
    # Model complexity
    model_parameters: int
    trainable_parameters: int
    model_size_mb: float
    flops_per_sample: Optional[int] = None

@dataclass
class FederatedLearningMetrics:
    """Federated learning specific metrics"""
    round_num: int
    
    # Participation
    total_clients: int
    participating_clients: int
    client_participation_rate: float
    
    # Aggregation
    aggregation_time: float
    parameter_diversity: float  # Variance in client parameters
    consensus_measure: float    # How similar client updates are
    
    # Global model performance
    global_accuracy: Dict[str, float]  # Per task
    global_loss: Dict[str, float]
    
    # Convergence
    convergence_rate: float
    stability_measure: float
    rounds_to_convergence: Optional[int] = None

@dataclass
class PrivacyMetrics:
    """Privacy and security metrics"""
    round_num: int
    client_id: str
    
    # Data privacy
    data_samples_used: int
    data_distribution_entropy: float
    local_data_heterogeneity: float
    
    # Model privacy
    parameter_noise_level: float
    gradient_norm: float
    information_leakage_risk: float

@dataclass
class LoRAMetrics:
    """LoRA-specific metrics"""
    round_num: int
    client_id: str
    
    # LoRA parameters
    lora_rank: int
    lora_parameters: int
    base_parameters: int
    parameter_efficiency: float  # LoRA params / Total params
    
    # Performance comparison
    lora_accuracy: float
    full_model_accuracy: Optional[float] = None
    accuracy_retention: Optional[float] = None  # LoRA acc / Full acc
    
    # Efficiency gains
    training_speedup: Optional[float] = None
    memory_reduction: Optional[float] = None
    communication_reduction: float = 0.0

@dataclass
class KnowledgeDistillationMetrics:
    """Knowledge distillation metrics"""
    round_num: int
    client_id: str
    
    # Distillation loss components
    kd_loss: float
    task_loss: float
    total_loss: float
    distillation_alpha: float
    
    # Teacher-student performance
    teacher_accuracy: Optional[float] = None
    student_accuracy: float = 0.0
    knowledge_transfer_efficiency: Optional[float] = None
    
    # Temperature effects
    temperature: float = 4.0
    soft_target_entropy: float = 0.0

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, experiment_name: str, save_dir: str = "metrics"):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.communication_metrics: List[CommunicationMetrics] = []
        self.performance_metrics: List[ModelPerformanceMetrics] = []
        self.computational_metrics: List[ComputationalMetrics] = []
        self.federated_metrics: List[FederatedLearningMetrics] = []
        self.privacy_metrics: List[PrivacyMetrics] = []
        self.lora_metrics: List[LoRAMetrics] = []
        self.kd_metrics: List[KnowledgeDistillationMetrics] = []
        
        # System monitoring
        self.process = psutil.Process()
        self.start_time = time.time()
        
        logger.info(f"Metrics collector initialized for experiment: {experiment_name}")
    
    def collect_communication_metrics(self, round_num: int, client_id: str, 
                                    params_sent: Dict, params_received: Dict,
                                    timing_info: Dict) -> CommunicationMetrics:
        """Collect communication efficiency metrics"""
        
        # Calculate parameter sizes
        sent_bytes = sum(np.array(v).nbytes for v in params_sent.values())
        received_bytes = sum(np.array(v).nbytes for v in params_received.values())
        
        # Compression ratio (if applicable)
        compression_ratio = received_bytes / sent_bytes if sent_bytes > 0 else 1.0
        
        metrics = CommunicationMetrics(
            round_num=round_num,
            client_id=client_id,
            parameters_sent_bytes=sent_bytes,
            parameters_received_bytes=received_bytes,
            compression_ratio=compression_ratio,
            upload_time=timing_info.get('upload_time', 0.0),
            download_time=timing_info.get('download_time', 0.0),
            total_communication_time=timing_info.get('total_comm_time', 0.0),
            network_latency=timing_info.get('latency', 0.0),
            communication_rounds=1,
            bandwidth_utilization=timing_info.get('bandwidth_util', 0.0)
        )
        
        self.communication_metrics.append(metrics)
        return metrics
    
    def collect_performance_metrics(self, round_num: int, client_id: str, task: str,
                                  train_results: Dict, val_results: Optional[Dict] = None) -> ModelPerformanceMetrics:
        """Collect model performance metrics"""
        
        # Calculate F1, precision, recall if not provided
        accuracy = train_results.get('accuracy', 0.0)
        loss = train_results.get('loss', 0.0)
        
        # Default values for classification metrics
        f1_score = train_results.get('f1_score', accuracy)  # Approximate
        precision = train_results.get('precision', accuracy)
        recall = train_results.get('recall', accuracy)
        
        metrics = ModelPerformanceMetrics(
            round_num=round_num,
            client_id=client_id,
            task=task,
            train_accuracy=accuracy,
            train_loss=loss,
            train_f1_score=f1_score,
            train_precision=precision,
            train_recall=recall,
            val_accuracy=val_results.get('accuracy') if val_results else None,
            val_loss=val_results.get('loss') if val_results else None,
            val_f1_score=val_results.get('f1_score') if val_results else None,
            task_specific_metrics=train_results.get('task_metrics', {}),
            loss_improvement=train_results.get('loss_improvement', 0.0),
            accuracy_improvement=train_results.get('accuracy_improvement', 0.0),
            convergence_indicator=train_results.get('convergence', 0.0)
        )
        
        self.performance_metrics.append(metrics)
        return metrics
    
    def collect_computational_metrics(self, round_num: int, client_id: str,
                                    timing_info: Dict, model_info: Dict) -> ComputationalMetrics:
        """Collect computational efficiency metrics"""
        
        # Get system resource usage
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # GPU metrics
        gpu_memory_mb = 0.0
        gpu_util_percent = 0.0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            # GPU utilization would need nvidia-ml-py for accurate measurement
            gpu_util_percent = 50.0  # Placeholder
        
        metrics = ComputationalMetrics(
            round_num=round_num,
            client_id=client_id,
            local_training_time=timing_info.get('training_time', 0.0),
            forward_pass_time=timing_info.get('forward_time', 0.0),
            backward_pass_time=timing_info.get('backward_time', 0.0),
            optimization_time=timing_info.get('optim_time', 0.0),
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            gpu_memory_usage_mb=gpu_memory_mb,
            gpu_utilization_percent=gpu_util_percent,
            model_parameters=model_info.get('total_params', 0),
            trainable_parameters=model_info.get('trainable_params', 0),
            model_size_mb=model_info.get('model_size_mb', 0.0),
            flops_per_sample=model_info.get('flops', None)
        )
        
        self.computational_metrics.append(metrics)
        return metrics
    
    def collect_federated_metrics(self, round_num: int, aggregation_info: Dict,
                                global_performance: Dict) -> FederatedLearningMetrics:
        """Collect federated learning specific metrics"""
        
        total_clients = aggregation_info.get('total_clients', 0)
        participating = aggregation_info.get('participating_clients', 0)
        
        metrics = FederatedLearningMetrics(
            round_num=round_num,
            total_clients=total_clients,
            participating_clients=participating,
            client_participation_rate=participating / total_clients if total_clients > 0 else 0.0,
            aggregation_time=aggregation_info.get('aggregation_time', 0.0),
            parameter_diversity=aggregation_info.get('param_variance', 0.0),
            consensus_measure=aggregation_info.get('consensus', 0.0),
            global_accuracy=global_performance.get('accuracy', {}),
            global_loss=global_performance.get('loss', {}),
            convergence_rate=global_performance.get('convergence_rate', 0.0),
            stability_measure=global_performance.get('stability', 0.0),
            rounds_to_convergence=global_performance.get('convergence_round')
        )
        
        self.federated_metrics.append(metrics)
        return metrics
    
    def collect_lora_metrics(self, round_num: int, client_id: str,
                           lora_info: Dict, performance_comparison: Dict) -> LoRAMetrics:
        """Collect LoRA-specific metrics"""
        
        lora_params = lora_info.get('lora_parameters', 0)
        base_params = lora_info.get('base_parameters', 1)
        
        metrics = LoRAMetrics(
            round_num=round_num,
            client_id=client_id,
            lora_rank=lora_info.get('rank', 16),
            lora_parameters=lora_params,
            base_parameters=base_params,
            parameter_efficiency=lora_params / (lora_params + base_params),
            lora_accuracy=performance_comparison.get('lora_accuracy', 0.0),
            full_model_accuracy=performance_comparison.get('full_accuracy'),
            accuracy_retention=performance_comparison.get('accuracy_retention'),
            training_speedup=performance_comparison.get('speedup'),
            memory_reduction=performance_comparison.get('memory_reduction'),
            communication_reduction=lora_params / base_params if base_params > 0 else 0.0
        )
        
        self.lora_metrics.append(metrics)
        return metrics
    
    def collect_kd_metrics(self, round_num: int, client_id: str,
                         kd_info: Dict, performance_info: Dict) -> KnowledgeDistillationMetrics:
        """Collect knowledge distillation metrics"""
        
        metrics = KnowledgeDistillationMetrics(
            round_num=round_num,
            client_id=client_id,
            kd_loss=kd_info.get('kd_loss', 0.0),
            task_loss=kd_info.get('task_loss', 0.0),
            total_loss=kd_info.get('total_loss', 0.0),
            distillation_alpha=kd_info.get('alpha', 0.7),
            teacher_accuracy=performance_info.get('teacher_accuracy'),
            student_accuracy=performance_info.get('student_accuracy', 0.0),
            knowledge_transfer_efficiency=performance_info.get('transfer_efficiency'),
            temperature=kd_info.get('temperature', 4.0),
            soft_target_entropy=kd_info.get('entropy', 0.0)
        )
        
        self.kd_metrics.append(metrics)
        return metrics
    
    def save_all_metrics(self, suffix: str = ""):
        """Save all collected metrics to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.experiment_name}_{timestamp}{suffix}"
        
        # Save as JSON
        all_metrics = {
            "experiment_info": {
                "name": self.experiment_name,
                "timestamp": timestamp,
                "duration": time.time() - self.start_time
            },
            "communication_metrics": [asdict(m) for m in self.communication_metrics],
            "performance_metrics": [asdict(m) for m in self.performance_metrics],
            "computational_metrics": [asdict(m) for m in self.computational_metrics],
            "federated_metrics": [asdict(m) for m in self.federated_metrics],
            "privacy_metrics": [asdict(m) for m in self.privacy_metrics],
            "lora_metrics": [asdict(m) for m in self.lora_metrics],
            "kd_metrics": [asdict(m) for m in self.kd_metrics]
        }
        
        json_file = self.save_dir / f"{base_name}.json"
        with open(json_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save individual CSV files for easy analysis
        self._save_csv_metrics(base_name)
        
        logger.info(f"All metrics saved to {json_file}")
        return json_file
    
    def _save_csv_metrics(self, base_name: str):
        """Save metrics as CSV files"""
        
        metrics_types = [
            ("communication", self.communication_metrics),
            ("performance", self.performance_metrics),
            ("computational", self.computational_metrics),
            ("federated", self.federated_metrics),
            ("lora", self.lora_metrics),
            ("kd", self.kd_metrics)
        ]
        
        for metric_type, metrics_list in metrics_types:
            if metrics_list:
                csv_file = self.save_dir / f"{base_name}_{metric_type}.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=asdict(metrics_list[0]).keys())
                    writer.writeheader()
                    for metrics in metrics_list:
                        writer.writerow(asdict(metrics))
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics"""
        
        summary = {
            "experiment_duration": time.time() - self.start_time,
            "total_rounds": len(set(m.round_num for m in self.performance_metrics)),
            "total_clients": len(set(m.client_id for m in self.performance_metrics)),
            "communication_stats": self._get_communication_summary(),
            "performance_stats": self._get_performance_summary(),
            "computational_stats": self._get_computational_summary(),
            "lora_stats": self._get_lora_summary(),
            "kd_stats": self._get_kd_summary()
        }
        
        return summary
    
    def _get_communication_summary(self) -> Dict[str, float]:
        """Get communication metrics summary"""
        if not self.communication_metrics:
            return {}
        
        total_bytes = sum(m.parameters_sent_bytes + m.parameters_received_bytes 
                         for m in self.communication_metrics)
        avg_comm_time = np.mean([m.total_communication_time for m in self.communication_metrics])
        avg_compression = np.mean([m.compression_ratio for m in self.communication_metrics])
        
        return {
            "total_communication_bytes": total_bytes,
            "average_communication_time": avg_comm_time,
            "average_compression_ratio": avg_compression,
            "total_communication_rounds": len(self.communication_metrics)
        }
    
    def _get_performance_summary(self) -> Dict[str, float]:
        """Get performance metrics summary"""
        if not self.performance_metrics:
            return {}
        
        final_accuracies = {}
        for task in set(m.task for m in self.performance_metrics):
            task_metrics = [m for m in self.performance_metrics if m.task == task]
            if task_metrics:
                final_accuracies[f"{task}_final_accuracy"] = task_metrics[-1].train_accuracy
                final_accuracies[f"{task}_avg_accuracy"] = np.mean([m.train_accuracy for m in task_metrics])
        
        return final_accuracies
    
    def _get_computational_summary(self) -> Dict[str, float]:
        """Get computational metrics summary"""
        if not self.computational_metrics:
            return {}
        
        return {
            "total_training_time": sum(m.local_training_time for m in self.computational_metrics),
            "average_memory_usage": np.mean([m.memory_usage_mb for m in self.computational_metrics]),
            "average_gpu_memory": np.mean([m.gpu_memory_usage_mb for m in self.computational_metrics]),
            "peak_memory_usage": max(m.memory_usage_mb for m in self.computational_metrics)
        }
    
    def _get_lora_summary(self) -> Dict[str, float]:
        """Get LoRA metrics summary"""
        if not self.lora_metrics:
            return {}
        
        return {
            "average_parameter_efficiency": np.mean([m.parameter_efficiency for m in self.lora_metrics]),
            "average_communication_reduction": np.mean([m.communication_reduction for m in self.lora_metrics]),
            "final_lora_accuracy": self.lora_metrics[-1].lora_accuracy if self.lora_metrics else 0.0
        }
    
    def _get_kd_summary(self) -> Dict[str, float]:
        """Get knowledge distillation summary"""
        if not self.kd_metrics:
            return {}
        
        return {
            "average_kd_loss": np.mean([m.kd_loss for m in self.kd_metrics]),
            "average_task_loss": np.mean([m.task_loss for m in self.kd_metrics]),
            "final_student_accuracy": self.kd_metrics[-1].student_accuracy if self.kd_metrics else 0.0,
            "knowledge_transfer_efficiency": np.mean([m.knowledge_transfer_efficiency for m in self.kd_metrics 
                                                    if m.knowledge_transfer_efficiency is not None])
        }

# Publication-ready metrics for academic papers
PUBLICATION_METRICS = {
    "accuracy_metrics": [
        "final_accuracy_per_task",
        "convergence_rate", 
        "accuracy_improvement_over_rounds"
    ],
    "efficiency_metrics": [
        "communication_cost_reduction",
        "parameter_efficiency_ratio",
        "training_time_speedup",
        "memory_usage_reduction"
    ],
    "federated_learning_metrics": [
        "client_participation_rate",
        "parameter_diversity",
        "consensus_measure",
        "scalability_analysis"
    ],
    "lora_specific_metrics": [
        "parameter_compression_ratio",
        "accuracy_retention_rate",
        "communication_overhead_reduction"
    ],
    "knowledge_distillation_metrics": [
        "knowledge_transfer_efficiency",
        "teacher_student_performance_gap",
        "distillation_loss_convergence"
    ],
    "privacy_metrics": [
        "data_heterogeneity_measure",
        "information_leakage_risk",
        "local_privacy_preservation"
    ]
}

def get_publication_ready_metrics(metrics_collector: MetricsCollector) -> Dict[str, Any]:
    """Extract publication-ready metrics from collected data"""
    
    summary = metrics_collector.get_summary_statistics()
    
    publication_metrics = {
        "experimental_setup": {
            "total_rounds": summary["total_rounds"],
            "total_clients": summary["total_clients"],
            "experiment_duration_hours": summary["experiment_duration"] / 3600
        },
        "accuracy_results": summary["performance_stats"],
        "efficiency_results": {
            **summary["communication_stats"],
            **summary["computational_stats"]
        },
        "lora_results": summary["lora_stats"],
        "knowledge_distillation_results": summary["kd_stats"]
    }
    
    return publication_metrics
