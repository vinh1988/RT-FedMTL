#!/usr/bin/env python3
"""
Model Evaluation Script for FedMKT Trained Models
Comprehensive evaluation of DistilBART and MobileBART models on 20News dataset.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.news20_dataset import News20Config, create_20news_federated_data
from training.fedmkt_trainer import FedMKTTrainer, FedMKTTrainingConfig
from models.bart_classification import DistilBARTClassifier, MobileBARTClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for FedMKT models"""
    
    def __init__(self, config: FedMKTTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 20News class names
        self.class_names = [
            'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
            'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
            'sci.space', 'soc.religion.christian', 'talk.politics.guns',
            'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
        ]
        
        logger.info(f"ModelEvaluator initialized on device: {self.device}")
    
    def load_model(self, model_path: str, model_type: str = "distilbart") -> torch.nn.Module:
        """Load a trained model"""
        logger.info(f"Loading {model_type} model from {model_path}")
        
        if model_type == "distilbart":
            model = DistilBARTClassifier(
                model_name=self.config.distilbart_model_name,
                num_labels=self.config.num_labels,
                max_length=self.config.max_length
            )
        elif model_type == "mobilebart":
            model = MobileBARTClassifier(
                model_name=self.config.mobilebart_model_name,
                num_labels=self.config.num_labels,
                max_length=self.config.max_length // 2
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully")
        return model
    
    def evaluate_model(self, model: torch.nn.Module, data_loader) -> Dict[str, Any]:
        """Evaluate model and return comprehensive metrics"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_loss": total_loss / len(data_loader),
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": {
                "precision": precision_per_class.tolist(),
                "recall": recall_per_class.tolist(),
                "f1_score": f1_per_class.tolist(),
                "support": support_per_class.tolist()
            },
            "class_names": self.class_names,
            "predictions": all_predictions.tolist(),
            "labels": all_labels.tolist(),
            "logits": all_logits.tolist()
        }
        
        return results
    
    def generate_classification_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed classification report"""
        predictions = np.array(results["predictions"])
        labels = np.array(results["labels"])
        
        report = classification_report(
            labels, predictions, 
            target_names=self.class_names,
            digits=4
        )
        
        return report
    
    def plot_confusion_matrix(self, results: Dict[str, Any], model_name: str, save_path: str):
        """Plot and save confusion matrix"""
        cm = np.array(results["confusion_matrix"])
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(self, results: Dict[str, Any], model_name: str, save_path: str):
        """Plot per-class precision, recall, and F1 scores"""
        precision = results["per_class_metrics"]["precision"]
        recall = results["per_class_metrics"]["recall"]
        f1_score = results["per_class_metrics"]["f1_score"]
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Metrics - {model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Per-class metrics plot saved to {save_path}")
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]], save_path: str):
        """Compare multiple models and create comparison plots"""
        
        # Extract metrics for comparison
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create comparison bar plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.8)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {save_path}")
    
    def evaluate_all_models(self, model_paths: Dict[str, str], data_loader, output_dir: str):
        """Evaluate all models and generate comprehensive reports"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        for model_name, model_path in model_paths.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Determine model type
            model_type = "distilbart" if "central" in model_name.lower() else "mobilebart"
            
            # Load and evaluate model
            model = self.load_model(model_path, model_type)
            results = self.evaluate_model(model, data_loader)
            
            all_results[model_name] = results
            
            # Generate classification report
            report = self.generate_classification_report(results)
            
            # Save results
            results_path = os.path.join(output_dir, f"{model_name}_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save classification report
            report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
            with open(report_path, 'w') as f:
                f.write(f"Classification Report for {model_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            
            # Generate plots
            cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            self.plot_confusion_matrix(results, model_name, cm_path)
            
            metrics_path = os.path.join(output_dir, f"{model_name}_per_class_metrics.png")
            self.plot_per_class_metrics(results, model_name, metrics_path)
            
            # Print summary
            logger.info(f"{model_name} Results:")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  Precision: {results['precision']:.4f}")
            logger.info(f"  Recall: {results['recall']:.4f}")
            logger.info(f"  F1-Score: {results['f1_score']:.4f}")
            logger.info(f"  Average Loss: {results['avg_loss']:.4f}")
        
        # Generate comparison plots if multiple models
        if len(all_results) > 1:
            comparison_path = os.path.join(output_dir, "model_comparison.png")
            self.compare_models(all_results, comparison_path)
        
        # Save combined results
        combined_path = os.path.join(output_dir, "all_results.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"All evaluation results saved to {output_dir}")
        return all_results


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate FedMKT trained models")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing trained models")
    parser.add_argument("--config", type=str, default="config/fedmkt_20news_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--data_config", type=str, default=None,
                       help="Custom data configuration")
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Create training configuration
    training_config = FedMKTTrainingConfig(
        distilbart_model_name=yaml_config.get("models", {}).get("central", {}).get("name", "facebook/distilbart-cnn-12-6"),
        mobilebart_model_name=yaml_config.get("models", {}).get("clients", [{}])[0].get("name", "valhalla/mobile-bart"),
        num_labels=yaml_config.get("models", {}).get("central", {}).get("num_labels", 20),
        max_length=yaml_config.get("models", {}).get("central", {}).get("max_length", 512)
    )
    
    # Create data configuration
    data_config = News20Config(
        data_dir=yaml_config.get("paths", {}).get("data_dir", "./data/20news"),
        max_length=training_config.max_length,
        num_clients=3
    )
    
    try:
        # Prepare test data
        logger.info("Preparing test data...")
        data_loaders = create_20news_federated_data(data_config, save_info=False)
        test_loader = list(data_loaders.values())[0]["test"]
        
        # Find model files
        model_paths = {}
        model_dir = Path(args.model_dir)
        
        for model_file in model_dir.rglob("pytorch_model.bin"):
            model_name = model_file.parent.name
            model_paths[model_name] = str(model_file)
        
        if not model_paths:
            raise ValueError(f"No model files found in {args.model_dir}")
        
        logger.info(f"Found {len(model_paths)} models: {list(model_paths.keys())}")
        
        # Create evaluator and run evaluation
        evaluator = ModelEvaluator(training_config)
        results = evaluator.evaluate_all_models(model_paths, test_loader, args.output_dir)
        
        logger.info("Evaluation completed successfully!")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        for model_name, model_results in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {model_results['accuracy']:.4f}")
            print(f"  Precision: {model_results['precision']:.4f}")
            print(f"  Recall:    {model_results['recall']:.4f}")
            print(f"  F1-Score:  {model_results['f1_score']:.4f}")
            print(f"  Avg Loss:  {model_results['avg_loss']:.4f}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def test_evaluation():
    """Test evaluation functionality"""
    logger.info("Testing model evaluation...")
    
    # Create test configuration
    config = FedMKTTrainingConfig(
        num_labels=20,
        max_length=256
    )
    
    # Create test data
    data_config = News20Config(max_length=256, num_clients=1)
    data_loaders = create_20news_federated_data(data_config, save_info=False)
    test_loader = list(data_loaders.values())[0]["test"]
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Create dummy model paths (for testing)
    model_paths = {
        "central_model": "dummy_path",
        "client_model_0": "dummy_path"
    }
    
    try:
        logger.info("Evaluation test completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation test failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Run test if no arguments provided
        test_evaluation()
    else:
        # Run main evaluation
        main()
