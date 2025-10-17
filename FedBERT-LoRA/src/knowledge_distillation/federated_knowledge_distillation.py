#!/usr/bin/env python3
"""
Knowledge Distillation Implementation
Bidirectional KD for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

class BidirectionalKDManager:
    """Manages bidirectional knowledge distillation"""

    def __init__(self, teacher_model, student_model, temperature: float = 3.0, alpha: float = 0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_history = []

    def teacher_to_student_kd_loss(self, student_logits, teacher_logits, labels):
        """Traditional KD: Teacher teaches student"""
        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        # KL divergence loss for soft targets
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

        # Hard loss from ground truth labels
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            # Combined loss
            total_loss = self.alpha * kd_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = kd_loss

        return total_loss

    def student_to_teacher_kd_loss(self, student_logits, teacher_logits):
        """Reverse KD: Student teaches teacher"""
        # Teacher learns to match student's predictions (MSE loss)
        return F.mse_loss(teacher_logits, student_logits)

    def bidirectional_kd_loss(self, student_logits, teacher_logits, labels, reverse_weight: float = 0.1):
        """Combined bidirectional KD loss"""
        # Forward KD (teacher → student)
        forward_loss = self.teacher_to_student_kd_loss(student_logits, teacher_logits, labels)

        # Reverse KD (student → teacher)
        reverse_loss = self.student_to_teacher_kd_loss(student_logits, teacher_logits)

        # Combined loss with weighting
        total_loss = forward_loss + reverse_weight * reverse_loss

        # Record distillation event
        self.distillation_history.append({
            'timestamp': torch.tensor([0.0]),
            'forward_loss': forward_loss.item(),
            'reverse_loss': reverse_loss.item(),
            'total_loss': total_loss.item(),
            'temperature': self.temperature,
            'alpha': self.alpha
        })

        return total_loss

    def get_distillation_summary(self) -> Dict:
        """Get summary of distillation history"""
        if not self.distillation_history:
            return {}

        losses = [event['total_loss'] for event in self.distillation_history]
        return {
            'total_distillation_events': len(self.distillation_history),
            'average_total_loss': sum(losses) / len(losses),
            'min_loss': min(losses),
            'max_loss': max(losses),
            'temperature_used': self.temperature,
            'alpha_used': self.alpha
        }

class LocalKDEngine:
    """Client-side KD engine for local training"""

    def __init__(self, student_model, tasks: List[str], config):
        self.student_model = student_model
        self.tasks = tasks
        self.config = config
        self.teacher_knowledge_cache = {}

    def update_teacher_knowledge(self, teacher_knowledge: Dict[str, torch.Tensor]):
        """Update cached teacher knowledge for KD"""
        self.teacher_knowledge_cache.update(teacher_knowledge)

    def calculate_kd_loss(self, student_logits: torch.Tensor, task_name: str, labels: torch.Tensor = None) -> torch.Tensor:
        """Calculate KD loss for a specific task"""
        if task_name not in self.teacher_knowledge_cache:
            # No teacher knowledge available, use only hard loss
            if labels is not None:
                return F.cross_entropy(student_logits, labels)
            return torch.tensor(0.0, device=student_logits.device)

        teacher_logits = self.teacher_knowledge_cache[task_name]

        # Create KD manager for this calculation
        kd_manager = BidirectionalKDManager(
            None, None,  # We don't need full models for loss calculation
            temperature=self.config.kd_temperature,
            alpha=self.config.kd_alpha
        )

        return kd_manager.teacher_to_student_kd_loss(student_logits, teacher_logits, labels)

    def prepare_student_knowledge_for_teacher(self, task_data: Dict) -> Dict[str, torch.Tensor]:
        """Prepare student knowledge to send back to teacher"""
        student_knowledge = {}

        for task_name in self.tasks:
            if task_name in task_data:
                # Generate student predictions for this task
                with torch.no_grad():
                    student_logits = self.student_model(
                        task_data[task_name]['input_ids'],
                        task_data[task_name]['attention_mask'],
                        task_name
                    )
                    student_knowledge[task_name] = student_logits

        return student_knowledge

class GlobalKDManager:
    """Server-side KD manager for global knowledge management"""

    def __init__(self, teacher_model, config):
        self.teacher_model = teacher_model
        self.config = config
        self.global_knowledge_base = {}
        self.kd_optimizer = torch.optim.AdamW(self.teacher_model.parameters(), lr=1e-4)

    def generate_teacher_knowledge(self, sample_inputs: Dict[str, Dict] = None) -> Dict[str, torch.Tensor]:
        """Generate teacher knowledge (soft labels) for all tasks"""
        teacher_knowledge = {}

        if sample_inputs:
            # Generate knowledge using sample inputs
            with torch.no_grad():
                for task_name, inputs in sample_inputs.items():
                    teacher_logits = self.teacher_model(
                        inputs['input_ids'],
                        inputs['attention_mask']
                    )
                    teacher_knowledge[task_name] = teacher_logits
        else:
            # Generate placeholder knowledge (can be enhanced with actual data)
            for task in ['sst2', 'qqp', 'stsb']:
                # Create dummy knowledge for demonstration
                if task in ['sst2', 'qqp']:
                    teacher_knowledge[task] = torch.randn(1, 2)  # Binary classification
                else:
                    teacher_knowledge[task] = torch.randn(1, 1)  # Regression

        # Cache for future use
        self.global_knowledge_base.update(teacher_knowledge)

        return teacher_knowledge

    def update_teacher_from_students(self, student_knowledge_updates: List[Dict]) -> Dict:
        """Update teacher model using student knowledge (reverse KD)"""
        if not student_knowledge_updates:
            return {"updated": False, "reason": "No student knowledge provided"}

        total_loss = 0.0
        num_updates = 0

        for update in student_knowledge_updates:
            student_knowledge = update.get('student_knowledge', {})

            for task_name, student_logits in student_knowledge.items():
                if task_name in self.global_knowledge_base:
                    # Teacher learns from student's predictions
                    teacher_logits = self.teacher_model(student_logits)

                    # Reverse KD loss
                    reverse_loss = F.mse_loss(teacher_logits, student_logits)

                    total_loss += reverse_loss.item()
                    num_updates += 1

        if num_updates > 0:
            # Update teacher model
            avg_loss = total_loss / num_updates
            self.kd_optimizer.step()
            self.kd_optimizer.zero_grad()

            return {
                "updated": True,
                "avg_reverse_loss": avg_loss,
                "num_updates": num_updates,
                "tasks_updated": list(self.global_knowledge_base.keys())
            }

        return {"updated": False, "reason": "No valid updates"}

    def get_teacher_knowledge_summary(self) -> Dict:
        """Get summary of teacher knowledge state"""
        return {
            'cached_tasks': list(self.global_knowledge_base.keys()),
            'knowledge_temperature': self.config.kd_temperature,
            'knowledge_alpha': self.config.kd_alpha,
            'teacher_model_frozen': all(not p.requires_grad for p in self.teacher_model.parameters())
        }
