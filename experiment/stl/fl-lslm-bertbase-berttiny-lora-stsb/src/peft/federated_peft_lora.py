#!/usr/bin/env python3
"""
PEFT LoRA (Low-Rank Adaptation) Implementation
Using HuggingFace PEFT library for parameter-efficient fine-tuning
"""

import logging
import os
import sys
from typing import Dict, List, Optional

# Debug: Print module loading
print(f"[DEBUG] Loading {__file__}")
print(f"[DEBUG] Current working directory: {os.getcwd()}")
print(f"[DEBUG] Python path: {sys.path}")

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    PreTrainedModel
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure debug level is set

# Add console handler if not already present
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.debug(f"[DEBUG] Initialized logger for {__name__}")

class PEFTLoRAModel(nn.Module):
    """Federated model with PEFT LoRA adapters for multi-task learning"""
    
    def __init__(
        self, 
        model_name: str, 
        tasks: List[str], 
        peft_config: Dict,
        unfreeze_layers: int = 0,
        is_teacher: bool = False
    ):
        super().__init__()
        self.tasks = tasks
        self.is_teacher = is_teacher
        self.unfreeze_layers = unfreeze_layers
        
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        # Load base model
        logger.info("="*80)
        logger.info(f"[PEFT LoRA] Loading base model: {model_name}")
        logger.info(f"[PEFT LoRA] Current working directory: {os.getcwd()}")
        logger.info(f"[PEFT LoRA] Python path: {sys.path}")
        
        # Load config
        logger.info("[PEFT LoRA] Loading model config...")
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=1,  # For regression tasks like STS-B
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Load base model
        logger.info("[PEFT LoRA] Loading pre-trained model...")
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )
        logger.info("[PEFT LoRA] Base model loaded successfully")
        
        # Configure PEFT LoRA
        logger.info("\n" + "="*80)
        logger.info("[PEFT LoRA] Configuring LoRA with the following settings:")
        logger.info(f"  - Rank (r): {peft_config.get('r', 16)}")
        logger.info(f"  - Alpha: {peft_config.get('lora_alpha', 64.0)}")
        logger.info(f"  - Dropout: {peft_config.get('lora_dropout', 0.1)}")
        logger.info(f"  - Target modules: {peft_config.get('target_modules', ['query', 'value'])}")
        logger.info(f"  - Modules to save: {peft_config.get('modules_to_save', ['classifier'])}")
        
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=peft_config.get('r', 16),
            lora_alpha=peft_config.get('lora_alpha', 64.0),
            lora_dropout=peft_config.get('lora_dropout', 0.1),
            bias=peft_config.get('bias', 'none'),
            target_modules=peft_config.get('target_modules', ['query', 'value']),
            modules_to_save=peft_config.get('modules_to_save', ['classifier'])
        )
        
        # Apply PEFT to the model
        logger.info("\n[PEFT LoRA] Applying PEFT to the model...")
        logger.info(f"[PEFT LoRA] Model class before PEFT: {type(self.base_model).__name__}")
        logger.info(f"[PEFT LoRA] PEFT config: {self.peft_config}")
        
        # Apply PEFT
        self.model = get_peft_model(self.base_model, self.peft_config)
        
        # Log model structure after PEFT
        logger.info(f"[PEFT LoRA] Model class after PEFT: {type(self.model).__name__}")
        logger.info("[PEFT LoRA] PEFT model created successfully")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"[PEFT LoRA] Total parameters: {total_params:,}")
        logger.info(f"[PEFT LoRA] Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Log LoRA-specific parameters
        lora_params = sum(p.numel() for n, p in self.model.named_parameters() if 'lora_' in n)
        logger.info(f"[PEFT LoRA] LoRA parameters: {lora_params:,}")
        
        # Log all parameter names that have requires_grad=True
        trainable_param_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        logger.info(f"[PEFT LoRA] Trainable parameters: {', '.join(trainable_param_names) or 'None'}")
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        logger.info("="*80)
        
        # Unfreeze top layers if specified
        if unfreeze_layers > 0:
            self._unfreeze_top_layers(unfreeze_layers)
        
        # Print trainable parameters
        self.print_trainable_parameters()
    
    def _unfreeze_top_layers(self, num_layers: int):
        """
        Unfreeze the pooler layers and optionally top transformer layers
        
        Args:
            num_layers: Number of top transformer layers to unfreeze (0 means only unfreeze pooler)
        """
        # Enable debug logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        logger.debug("\n" + "="*80)
        logger.debug("DEBUGGING _unfreeze_top_layers")
        logger.debug(f"Input num_layers: {num_layers}")
        
        if not hasattr(self.model, 'base_model'):
            logger.warning("Model does not have 'base_model' attribute!")
            logger.warning(f"Model attributes: {dir(self.model)}")
            return
            
        base_model = self.model.base_model
        logger.debug(f"Base model type: {type(base_model).__name__}")
        
        # Log all parameter names and their current requires_grad status
        logger.debug("\nAll parameters in base model (before unfreezing):")
        for name, param in base_model.named_parameters():
            logger.debug(f"{name}: requires_grad={param.requires_grad}, shape={tuple(param.shape) if param is not None else 'None'}")
        
        # Find and log all pooler-related parameters
        pooler_params = [(n, p) for n, p in base_model.named_parameters() if 'pooler' in n.lower()]
        
        if not pooler_params:
            logger.warning("No pooler layers found in the model!")
            logger.warning("Parameter names that were checked:")
            for name, _ in base_model.named_parameters():
                logger.warning(f"- {name}")
        
        # Unfreeze pooling layers in base model
        logger.debug("\nProcessing pooler layers:")
        pooler_found = False
        for name, param in base_model.named_parameters():
            if "pooler" in name.lower():
                old_value = param.requires_grad
                param.requires_grad = True
                pooler_found = True
                logger.debug(f"  - {name}: requires_grad changed from {old_value} to {param.requires_grad}")
        
        # Log all parameters after unfreezing
        if pooler_found:
            logger.info("\n" + "="*40)
            logger.info("Pooler parameters after unfreezing:")
            for name, param in base_model.named_parameters():
                if "pooler" in name.lower():
                    logger.info(f"  - {name}: requires_grad={param.requires_grad}")
            logger.info("="*40 + "\n")
        
        # Optionally unfreeze top transformer layers if num_layers > 0
        if num_layers > 0:
            logger.debug("\nHandling model architecture for transformer layers:")
            encoder = None
            model_type = None
            
            if hasattr(base_model, 'bert'):
                # BERT-style models
                encoder = base_model.bert.encoder
                model_type = 'BERT'
            elif hasattr(base_model, 'roberta'):
                # RoBERTa-style models
                encoder = base_model.roberta.encoder
                model_type = 'RoBERTa'
            
            if encoder is not None and hasattr(encoder, 'layer'):
                total_layers = len(encoder.layer)
                layers_to_unfreeze = min(num_layers, total_layers)
                
                # Unfreeze the specified number of top layers
                for layer in encoder.layer[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                logger.info(f"Unfroze top {layers_to_unfreeze} {model_type} layers")
    
    def print_trainable_parameters(self):
        """Prints the number of trainable parameters in the model."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    
    def forward(self, input_ids, attention_mask, task_name=None):
        """Forward pass with task-specific processing"""
        # For teacher model, we might not have task_name
        if task_name is None or task_name not in self.tasks:
            task_name = self.tasks[0]  # Default to first task
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }
    
    def get_peft_config(self) -> Dict:
        """Get the PEFT configuration"""
        return {
            'peft_type': 'LORA',
            'task_type': 'SEQ_CLS',
            'inference_mode': False,
            'r': self.peft_config.r,
            'lora_alpha': self.peft_config.lora_alpha,
            'lora_dropout': self.peft_config.lora_dropout,
            'target_modules': self.peft_config.target_modules,
            'modules_to_save': self.peft_config.modules_to_save
        }
    
    def get_trainable_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get trainable parameters for federated learning
        
        Returns:
            Dict mapping task names to parameter dictionaries with 'lora_A' and 'lora_B' keys
        """
        params = {}
        
        # Initialize with default task
        task = 'stsb'
        params[task] = {'lora_A': None, 'lora_B': None}
        
        # Find all LoRA parameters (PEFT naming: .lora_A.default.weight, .lora_B.default.weight)
        lora_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad and 'lora_' in name.lower():
                lora_params[name] = param
        
        logger.info(f"[PEFT] Found {len(lora_params)} trainable LoRA parameters")
        if lora_params:
            logger.info(f"[PEFT] Sample parameter names: {list(lora_params.keys())[:3]}")
        
        # First, collect all A and B matrices separately
        a_matrices = []
        b_matrices = []
        
        for name, param in lora_params.items():
            # PEFT names end with .lora_A.default.weight or .lora_B.default.weight
            if '.lora_A.' in name:
                # Extract base name (everything before .lora_A.)
                parts = name.split('.lora_A.')
                base_name = parts[0]
                b_name = f"{base_name}.lora_B.{parts[1]}"  # Reconstruct B name with same suffix
                
                # Get A matrix
                a_matrices.append(param.clone().detach())
                logger.debug(f"[PEFT] Found lora_A: {name} with shape {param.shape}")
                
                # Get corresponding B matrix if it exists
                if b_name in lora_params:
                    b_param = lora_params[b_name]
                    b_matrices.append(b_param.clone().detach())
                    logger.debug(f"[PEFT] Found matching lora_B: {b_name} with shape {b_param.shape}")
                else:
                    logger.warning(f"No matching B matrix found for {name}")
                    logger.warning(f"Looking for: {b_name}")
                    logger.warning(f"Available names: {list(lora_params.keys())}")
                    # Add a zero tensor with appropriate shape for B if A exists but B doesn't
                    b_matrices.append(torch.zeros_like(param.clone().detach().t()))
        
        # Stack A matrices vertically and B matrices horizontally
        if a_matrices:
            params[task]['lora_A'] = torch.cat(a_matrices, dim=0)
            logger.info(f"[PEFT] Stacked {len(a_matrices)} lora_A matrices into shape {params[task]['lora_A'].shape}")
            
            # Only stack B matrices if we have the same number of B matrices as A matrices
            if b_matrices and len(b_matrices) == len(a_matrices):
                # Ensure all B matrices have the same number of rows before concatenation
                max_rows = max(b.shape[0] for b in b_matrices)
                padded_b_matrices = []
                
                for b in b_matrices:
                    if b.shape[0] < max_rows:
                        # Pad with zeros if needed
                        padding = torch.zeros(max_rows - b.shape[0], b.shape[1], 
                                           device=b.device, dtype=b.dtype)
                        padded_b = torch.cat([b, padding], dim=0)
                        padded_b_matrices.append(padded_b)
                    else:
                        padded_b_matrices.append(b)
                
                params[task]['lora_B'] = torch.cat(padded_b_matrices, dim=1)
                logger.info(f"[PEFT] Stacked {len(b_matrices)} lora_B matrices into shape {params[task]['lora_B'].shape}")
            else:
                logger.warning(f"Mismatch in number of A and B matrices: {len(a_matrices)} A vs {len(b_matrices)} B")
        
        # Verify we have both A and B matrices
        if params[task]['lora_A'] is None and params[task]['lora_B'] is None:
            logger.warning("No LoRA parameters found in the model!")
        elif params[task]['lora_A'] is None or params[task]['lora_B'] is None:
            logger.warning(f"Missing LoRA parameters: A={params[task]['lora_A'] is not None}, B={params[task]['lora_B'] is not None}")
        else:
            logger.info(f"[PEFT] Successfully extracted LoRA parameters for task '{task}'")
        
        return params
    
    def load_trainable_params(self, params: Dict[str, Dict[str, torch.Tensor]]):
        """Load trainable parameters for federated learning
        
        Args:
            params: Dictionary mapping task names to parameter dictionaries with 'lora_A' and 'lora_B' keys
        """
        # Get all LoRA parameters (PEFT naming: .lora_A.default.weight, .lora_B.default.weight)
        lora_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad and 'lora_' in name.lower():
                lora_params[name] = param
        
        # Track which parameters we've updated
        updated_params = set()
        
        # Default task name
        task = 'stsb'
        
        if task not in params or params[task] is None:
            logger.warning(f"No parameters found for task '{task}' in the provided parameters")
            return
            
        # Get the aggregated A and B matrices
        agg_A = params[task].get('lora_A')
        agg_B = params[task].get('lora_B')
        
        if agg_A is None or agg_B is None:
            logger.warning(f"Missing A or B matrix for task '{task}': A={agg_A is not None}, B={agg_B is not None}")
            return
        
        logger.info(f"[PEFT] Loading aggregated parameters: A={agg_A.shape}, B={agg_B.shape}")
        
        # Split the aggregated matrices and update individual parameters
        a_start = 0
        b_col_start = 0
        
        # First pass: process all A matrices
        for name, param in lora_params.items():
            # PEFT names contain .lora_A. or .lora_B.
            if '.lora_A.' in name:
                a_end = a_start + param.shape[0]
                if a_end > agg_A.shape[0]:
                    logger.warning(f"Aggregated A matrix too small for parameter {name}")
                    continue
                    
                # Update the parameter
                param.data.copy_(agg_A[a_start:a_end].to(param.device))
                updated_params.add(name)
                logger.debug(f"[PEFT] Updated {name}: [{a_start}:{a_end}]")
                
                # Get corresponding B matrix
                parts = name.split('.lora_A.')
                base_name = parts[0]
                b_name = f"{base_name}.lora_B.{parts[1]}"
                
                if b_name in lora_params:
                    b_param = lora_params[b_name]
                    b_col_end = b_col_start + b_param.shape[1]
                    
                    if b_col_end > agg_B.shape[1]:
                        logger.warning(f"Aggregated B matrix too small for parameter {b_name}")
                        continue
                    
                    # Update the B parameter
                    b_param.data.copy_(agg_B[:, b_col_start:b_col_end].to(b_param.device))
                    updated_params.add(b_name)
                    logger.debug(f"[PEFT] Updated {b_name}: [:, {b_col_start}:{b_col_end}]")
                    
                    # Move to next B matrix block
                    b_col_start = b_col_end
                
                # Move to next A matrix block
                a_start = a_end
        
        # Second pass: handle any remaining B matrices that weren't processed
        for name, param in lora_params.items():
            if '.lora_B.' in name and name not in updated_params:
                logger.warning(f"Could not update parameter {name} - no corresponding A matrix found")
        
        # Log which parameters were updated
        if updated_params:
            logger.info(f"[PEFT] Updated {len(updated_params)} LoRA parameters")
        else:
            logger.warning("No LoRA parameters were updated")
    
    def get_task_dataloader(self, task: str, batch_size: int = 8, dataset_data: Dict = None):
        """Get DataLoader for a specific task
        
        Args:
            task: Task name (e.g., 'stsb')
            batch_size: Batch size for the DataLoader
            dataset_data: Dictionary containing dataset data with one of:
                - 'input_ids', 'attention_mask', 'labels' keys (tokenized data)
                - 'text1', 'text2', 'score' keys (text data)
                - 'texts', 'labels' keys (concatenated text data)
        """
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer
        
        if dataset_data is None:
            raise ValueError("Dataset data must be provided")
            
        logger.info(f"[PEFT LoRA] Getting dataloader for task: {task}")
        logger.info(f"[PEFT LoRA] Dataset data keys: {list(dataset_data.keys())}")
        
        # Log detailed dataset structure
        logger.info("="*80)
        logger.info("DATASET STRUCTURE INSPECTION:")
        logger.info(f"Dataset type: {type(dataset_data)}")
        logger.info(f"Dataset keys: {list(dataset_data.keys())}")
        
        # Log sample data for each key
        for key, value in dataset_data.items():
            logger.info(f"- Key: {key} (type: {type(value)})")
            if hasattr(value, '__len__'):
                logger.info(f"  Length: {len(value)}")
                if len(value) > 0:
                    sample = value[0] if hasattr(value, '__getitem__') else next(iter(value))
                    logger.info(f"  Sample value: {sample}")
                    logger.info(f"  Sample type: {type(sample)}")
        logger.info("="*80)
        
        # Check if data is already tokenized
        if all(key in dataset_data for key in ['input_ids', 'attention_mask', 'labels']):
            # Data is already tokenized
            import torch
            from torch.utils.data import TensorDataset
            
            logger.info("[PEFT LoRA] Using pre-tokenized data format")
            input_ids = torch.tensor(dataset_data['input_ids'])
            attention_mask = torch.tensor(dataset_data['attention_mask'])
            labels = torch.tensor(dataset_data['labels'])
            
            dataset = TensorDataset(input_ids, attention_mask, labels)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
        # Check for texts and labels format (from STSBDatasetHandler)
        elif 'texts' in dataset_data and 'labels' in dataset_data:
            logger.info("[PEFT LoRA] Detected 'texts' and 'labels' format")
            
            class TextLabelDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_length=128):
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    import torch  # Import torch here to avoid scope issues
                    # Split the text back into sentence1 and sentence2
                    text = self.texts[idx]
                    if ' [SEP] ' in text:
                        sentence1, sentence2 = text.split(' [SEP] ', 1)
                    else:
                        sentence1, sentence2 = text, text  # Fallback if separator not found
                    
                    encoding = self.tokenizer(
                        text=sentence1,
                        text_pair=sentence2,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                        'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
                    }
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
            
            # Create dataset
            dataset = TextLabelDataset(
                texts=dataset_data['texts'],
                labels=dataset_data['labels'],
                tokenizer=tokenizer
            )
            
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Handle text1, text2, score format
        elif all(key in dataset_data for key in ['text1', 'text2', 'score']):
            logger.info("[PEFT LoRA] Detected 'text1', 'text2', 'score' format")
            # Create a custom dataset for tokenization on the fly
            class STSDataset(Dataset):
                def __init__(self, text1, text2, scores, tokenizer, max_length=128):
                    self.text1 = text1
                    self.text2 = text2
                    self.scores = scores
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.scores)
                
                def __getitem__(self, idx):
                    import torch  # Import torch here to avoid scope issues
                    encoding = self.tokenizer(
                        text=self.text1[idx],
                        text_pair=self.text2[idx],
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                        'labels': torch.tensor(self.scores[idx], dtype=torch.float32)
                    }
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
            
            # Create dataset
            dataset = STSDataset(
                text1=dataset_data['text1'],
                text2=dataset_data['text2'],
                scores=dataset_data['score'],
                tokenizer=tokenizer
            )
            
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Handle additional dataset formats
        elif all(key in dataset_data for key in ['sentence1', 'sentence2', 'label']):
            # Create a custom dataset for tokenization on the fly
            class SentencePairDataset(Dataset):
                def __init__(self, sentence1, sentence2, labels, tokenizer, max_length=128):
                    self.sentence1 = sentence1
                    self.sentence2 = sentence2
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.labels)
                
                def __getitem__(self, idx):
                    encoding = self.tokenizer(
                        text=self.sentence1[idx],
                        text_pair=self.sentence2[idx],
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                        'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                    }
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
            
            # Create dataset
            dataset = SentencePairDataset(
                sentence1=dataset_data['sentence1'],
                sentence2=dataset_data['sentence2'],
                labels=dataset_data['label'],
                tokenizer=tokenizer
            )
            
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        else:
            # Try to handle other formats by checking for common keys
            if 'train' in dataset_data and 'validation' in dataset_data:
                logger.info("[PEFT LoRA] Detected 'train' and 'validation' split format")
                # Use the training split by default
                return self.get_task_dataloader(task, batch_size, dataset_data['train'])
            
            # If we get here, the format is not supported
            raise ValueError(
                "Unsupported dataset format. Expected one of:\n"
                "1. Tokenized data: {'input_ids': [...], 'attention_mask': [...], 'labels': [...]}\n"
                "2. Text data with labels: {'texts': [...], 'labels': [...]}\n"
                "3. STS-B text data: {'text1': [...], 'text2': [...], 'score': [...]}\n"
                f"Got keys: {list(dataset_data.keys())}"
            )


class PEFTAggregator:
    """Aggregates PEFT LoRA parameters from multiple clients"""
    
    def __init__(self):
        self.aggregation_history = []
    
    def aggregate_lora_updates(self, client_updates: List[Dict], client_weights: List[float] = None) -> Dict[str, Dict]:
        """
        Aggregate LoRA parameters using federated averaging
        Compatible with LoRAAggregator interface
        
        Args:
            client_updates: List of update dictionaries with 'lora_updates' key
            client_weights: Optional list of weights for each client (default: uniform)
            
        Returns:
            Aggregated parameters per task
        """
        if not client_updates:
            logger.warning("No client updates provided for aggregation")
            return {}
        
        # Use uniform weights if not provided
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        aggregated_params = {}
        
        # Get all unique tasks across clients
        all_tasks = set()
        for update in client_updates:
            all_tasks.update(update['lora_updates'].keys())
        
        logger.info(f"[PEFT Aggregator] Aggregating updates for tasks: {all_tasks}")
        
        # Aggregate parameters for each task
        for task in all_tasks:
            task_params = {}
            
            # Get parameters for this task from all clients that have it
            task_updates = []
            task_weights = []
            
            for i, update in enumerate(client_updates):
                if task in update['lora_updates']:
                    task_updates.append(update['lora_updates'][task])
                    task_weights.append(client_weights[i])
            
            if task_updates:
                # Aggregate each parameter type (lora_A, lora_B)
                for param_name in ['lora_A', 'lora_B']:
                    if param_name in task_updates[0] and task_updates[0][param_name] is not None:
                        # Weighted average of parameter matrices
                        weighted_sum = sum(
                            update[param_name] * weight
                            for update, weight in zip(task_updates, task_weights)
                        )
                        task_params[param_name] = weighted_sum
                        logger.debug(f"[PEFT Aggregator] Aggregated {param_name} for task {task}: shape {weighted_sum.shape}")
                
                aggregated_params[task] = task_params
                logger.info(f"[PEFT Aggregator] Task {task}: aggregated {len(task_params)} parameter matrices from {len(task_updates)} clients")
        
        # Record aggregation
        self.aggregation_history.append({
            'timestamp': torch.tensor([0.0]),
            'num_clients': len(client_updates),
            'tasks_aggregated': list(aggregated_params.keys()),
            'aggregation_weights': client_weights
        })
        
        return aggregated_params
    
    def get_aggregation_summary(self):
        """Get summary of aggregation history"""
        return {
            'total_rounds': len(self.aggregation_history),
            'recent_rounds': self.aggregation_history[-5:],  # Last 5 rounds
            'total_clients': sum(r['num_clients'] for r in self.aggregation_history)
        }
