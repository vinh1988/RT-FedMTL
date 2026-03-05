# Unified Data Contributions: All Models Experimental Framework

## Experimental Overview
This document provides a comprehensive analysis of experimental configurations and data contributions that apply to ALL models in the federated learning framework, covering 9 distinct experimental paradigms across 5 Small Language Model architectures (DistilBERT, Medium BERT, Mini BERT, MiniLM, Tiny BERT).

## 1. UNIVERSAL EXPERIMENTAL CONFIGURATIONS

### 1.1 Centralized Training Experiments

#### **Experiment 1: Multi-Task Learning (MTL)**
- **Configuration**: `centralized-mtl-all-tasks/`
- **Models**: All 5 SLM variants (DistilBERT, Medium BERT, Mini BERT, MiniLM, Tiny BERT)
- **Training Protocol**: 25 epochs, batch size 8, learning rate 2e-5
- **Data Contribution**: 
  - SST-2: 66,477 training samples, 872 validation, 1,821 test samples
  - QQP: 323,415 training samples, 40,431 validation, 39,081 test samples
  - STS-B: 4,249 training samples, 1,500 validation, 1,379 test samples
- **Total Dataset**: 394,141 training samples across all three NLU tasks
- **Resource Monitoring**: GPU/CPU tracking, validation logging, 10-second sampling interval

#### **Experiment 2: Single-Task Learning - SST-2**
- **Configuration**: `centralized-single-task-sst2/`
- **Models**: All 5 SLM variants with SST-2 specific fine-tuning
- **Data Contribution**: 66,477 training samples, 872 validation, 1,821 test samples
- **Training Protocol**: 25 epochs, batch size 8, learning rate 2e-5, warmup 500 steps
- **Task Type**: Binary classification (sentiment analysis)

#### **Experiment 3: Single-Task Learning - QQP**
- **Configuration**: `centralized-single-task-qqp/`
- **Models**: All 5 SLM variants with QQP specific fine-tuning
- **Data Contribution**: 323,415 training samples, 40,431 validation, 39,081 test samples
- **Training Protocol**: 25 epochs, batch size 8, learning rate 2e-5, warmup 1,000 steps
- **Task Type**: Binary classification (paraphrase detection)

#### **Experiment 4: Single-Task Learning - STS-B**
- **Configuration**: `centralized-single-task-stsb/`
- **Models**: All 5 SLM variants with STS-B specific fine-tuning
- **Data Contribution**: 4,249 training samples, 1,500 validation, 1,379 test samples
- **Training Protocol**: 25 epochs, batch size 8, learning rate 2e-5, warmup 200 steps
- **Task Type**: Regression (semantic similarity)

### 1.2 Federated Learning Experiments

#### **Experiment 5: Multi-Task Federated Learning (Standard)**
- **Configuration**: `fl-mtl-slms-{model}-stsb-qqp-sst2/`
- **Models**: All 5 SLM variants for federated multi-task learning
- **Federated Protocol**: Standard FedAvg with task-aware weighting
- **Training Configuration**: 25-50 rounds, 1-3 clients, 1 local epoch per round
- **Data Contribution**:
  - SST-2: 66,477 training samples, 872 validation samples
  - QQP: 323,415 training samples, 40,431 validation samples
  - STS-B: 4,249 training samples, 1,500 validation samples
- **Data Distribution**: IID partitioning across all three tasks
- **Communication**: WebSocket port 8771, timeout management, retry mechanisms

#### **Experiment 6: Multi-Task Federated Learning (LoRA)**
- **Configuration**: `fl-mtl-slms-{model}-stsb-qqp-sst2-lora/`
- **Models**: All 5 SLM variants with LoRA fine-tuning
- **LoRA Configuration**: Rank 8, alpha 16.0, dropout 0.1
- **Target Modules**: Attention mechanisms (query, key, value, output dense)
- **Training Configuration**: 30 rounds, 1-3 clients, learning rate 2e-5
- **Data Contribution**:
  - SST-2: 66,477 training samples, 872 validation samples
  - QQP: 323,415 training samples, 40,431 validation samples
  - STS-B: 4,249 training samples, 1,500 validation samples
- **Communication Efficiency**: Selective parameter transmission for bandwidth reduction

#### **Experiment 7: Multi-Task Federated Learning (Non-IID, 9 Clients)**
- **Configuration**: `fl-mtl-slms-{model}-non-iid-stsb-qqp-sst2-3client-each/`
- **Models**: All 5 SLM variants with enhanced client configuration
- **Federated Protocol**: Advanced FedAvg with non-IID handling
- **Client Configuration**: 9 clients (3 per task), true federated learning
- **Data Contribution**:
  - SST-2: 66,477 training samples, 872 validation samples (split across 3 clients)
  - QQP: 323,415 training samples, 40,431 validation samples (split across 3 clients)
  - STS-B: 4,249 training samples, 1,500 validation samples (split across 3 clients)
- **Data Distribution**: Non-IID with Dirichlet distribution (α=0.5)
- **Communication**: WebSocket port 8775, enhanced timeout management

#### **Experiment 8: Single-Task Federated Learning - QQP (Non-IID)**
- **Configuration**: `fl-slms-{model}-non-iid-qqp/`
- **Models**: All 5 SLM variants for single-task QQP federated learning
- **Data Partitioning**: True federated with 1/3 dataset split per client
- **Sample Distribution**: 107,805 samples per client (323,415 total ÷ 3)
- **Non-IID Configuration**: Dirichlet distribution (α=0.5), minority class oversampling
- **Client Management**: 3 clients with normalized weights
- **Validation**: Full QQP validation set (40,431 samples, not split)

#### **Experiment 9: Single-Task Federated Learning - SST-2 (Non-IID)**
- **Configuration**: `fl-slms-{model}-non-iid-sst2/`
- **Models**: All 5 SLM variants for single-task SST-2 federated learning
- **Data Partitioning**: True federated with 1/3 dataset split per client
- **Sample Distribution**: 22,159 samples per client (66,477 total ÷ 3)
- **Non-IID Configuration**: Dirichlet distribution (α=0.5), minority class oversampling
- **Client Management**: 3 clients with normalized weights
- **Validation**: Full SST2 validation set (872 samples, not split)

#### **Experiment 10: Single-Task Federated Learning - STS-B (Non-IID)**
- **Configuration**: `fl-slms-{model}-non-iid-stsb/`
- **Models**: All 5 SLM variants for single-task STS-B federated learning
- **Data Partitioning**: True federated with 1/3 dataset split per client
- **Sample Distribution**: 1,416 samples per client (4,249 total ÷ 3)
- **Non-IID Configuration**: Dirichlet distribution (α=0.5), minority class oversampling
- **Client Management**: 3 clients with normalized weights
- **Validation**: Full STS-B validation set (1,500 samples, not split)

## 2. UNIVERSAL DATA CONTRIBUTION ANALYSIS

### 2.1 Dataset Utilization Patterns

#### **Centralized Experiments**
- **Full Dataset Access**: Complete access to all three NLU benchmarks
- **Multi-Task Integration**: Simultaneous learning across sentiment analysis, paraphrase detection, and semantic similarity
- **Single-Task Specialization**: Individual fine-tuning for each specific task
- **Reproducibility**: Consistent hyperparameters across all experiments (seed 42)

#### **Federated Experiments**
- **Communication Protocols**: WebSocket-based with comprehensive timeout management
- **Client Scalability**: 1-9 clients across different configurations
- **Data Heterogeneity**: IID and non-IID partitioning strategies
- **Aggregation Strategies**: Standard FedAvg and task-aware weighting

#### **Validation Set Handling**
- **Centralized**: Full validation sets (not split)
- **Federated Single-Task**: Full validation sets (not split according to comments)
- **Federated Multi-Task**: Full validation sets (not split in multi-task)

### 2.2 Universal Experimental Design Matrix

| Experiment | Paradigm | Task Configuration | Models | Clients | Data Distribution | Key Features |
|-------------|-------------|-------------------|---------|---------|----------------|-------------|
| 1 | Centralized MTL | Multi-Task | 5 SLMs | N/A | Full dataset access |
| 2 | Centralized Single | SST-2 Only | 5 SLMs | N/A | Task-specific fine-tuning |
| 3 | Centralized Single | QQP Only | 5 SLMs | N/A | Task-specific fine-tuning |
| 4 | Centralized Single | STS-B Only | 5 SLMs | N/A | Task-specific fine-tuning |
| 5 | Federated MTL | Multi-Task | 5 SLMs | 1-3 | IID partitioning |
| 6 | Federated MTL | Multi-Task | 5 SLMs | 1-3 | LoRA fine-tuning |
| 7 | Federated MTL | Multi-Task | 5 SLMs | 9 | Non-IID, 3 per task |
| 8 | Federated Single | QQP Only | 5 SLMs | 3 | Non-IID, 1/3 split |
| 9 | Federated Single | SST-2 Only | 5 SLMs | 3 | Non-IID, 1/3 split |
| 10 | Federated Single | STS-B Only | 5 SLMs | 3 | Non-IID, 1/3 split |

### 2.3 Universal Configuration Standardization

#### **Common Hyperparameters Across All Models and Experiments**
- **Learning Rate**: 2e-5 (consistent across centralized and federated)
- **Batch Size**: 8 (centralized), 8-16 (federated for efficiency)
- **Training Epochs**: 25 (centralized), 1 local epoch per federated round
- **Weight Decay**: 0.01 with gradient clipping (max norm 1.0)
- **Random Seed**: 42 for reproducibility
- **Warmup Steps**: 200-1,000 depending on task complexity

#### **Federated-Specific Settings**
- **Communication Ports**: 8771-8775 (avoiding conflicts)
- **Timeout Management**: 60s client timeout, 30s WebSocket timeout
- **Round Timeout**: 3,400s (56.7 minutes) for client update collection
- **Retry Mechanism**: 3 retry attempts with exponential backoff
- **Resource Monitoring**: 10-second sampling interval with comprehensive logging

### 2.4 Advanced Experimental Features

#### **Parameter-Efficient Fine-Tuning**
- **LoRA Integration**: Low-rank adaptation for communication reduction
- **Targeted Module Selection**: Attention mechanism optimization
- **Rank Configuration**: Rank 8 for optimal parameter-efficiency balance
- **Dropout Regularization**: 0.1 dropout for improved generalization

#### **Data Heterogeneity Handling**
- **Non-IID Simulation**: Dirichlet distribution with α=0.5
- **Class Balancing**: Minority class oversampling for robust learning
- **Client Weighting**: Normalized client weights for fair aggregation
- **Concept Drift**: Realistic data distribution challenges

### 2.5 Resource and Communication Analysis

#### **Bandwidth Optimization**
- **Selective Transmission**: Only changed parameters transmitted
- **Delta Compression**: Efficient parameter delta encoding
- **Adaptive Frequency**: Dynamic update frequency based on model stability
- **Module Targeting**: Specific attention module fine-tuning

#### **Computational Efficiency**
- **Edge Deployment**: SLMs optimized for resource-constrained devices
- **Memory Management**: Efficient parameter usage and caching
- **Energy Optimization**: Reduced computational overhead through LoRA
- **Scalability Testing**: Performance evaluation across 1-9 client configurations

## 3. UNIVERSAL CONTRIBUTION SUMMARY

### 3.1 Experimental Coverage
- **Total Experiments**: 9 distinct configurations applied to 5 models = 45 total experiments
- **Model Variants**: 5 SLM architectures across all experimental paradigms
- **Task Coverage**: SST-2, QQP, STS-B across centralized and federated settings
- **Data Distribution**: IID and non-IID partitioning strategies
- **Client Scalability**: 1-9 clients for scalability assessment

### 3.2 Universal Dataset Totals
- **SST-2**: 66,477 training + 872 validation + 1,821 test = 69,170 total
- **QQP**: 323,415 training + 40,431 validation + 39,081 test = 402,927 total
- **STS-B**: 4,249 training + 1,500 validation + 1,379 test = 7,128 total
- **Grand Total**: 479,225 samples across all three NLU tasks

### 3.3 Model Architecture Coverage
- **DistilBERT**: 6 layers, 768 hidden dimensions, 66M parameters
- **Medium BERT**: 8 layers, 512 hidden dimensions, 41M parameters
- **Mini BERT**: 4 layers, 256 hidden dimensions, 11M parameters
- **MiniLM**: 6 layers, 384 hidden dimensions, 22M parameters
- **Tiny BERT**: 2 layers, 128 hidden dimensions, 4.4M parameters

### 3.4 Research Impact
This universal experimental design enables:
- **Comprehensive comparative analysis** across 5 model architectures
- **Rigorous evaluation** between centralized and federated approaches
- **Multi-task learning assessment** with single-task baselines
- **Parameter efficiency evaluation** through LoRA fine-tuning
- **Data heterogeneity analysis** through non-IID experiments
- **Scalability evaluation** across varying client configurations
- **Communication efficiency optimization** for real-world deployment

### 3.5 Academic Value
The universal experimental contributions provide:
- **Reproducible research** with detailed configuration documentation
- **Comprehensive evaluation** across multiple model architectures
- **Practical insights** for federated learning deployment
- **Parameter-efficient methods** for communication-constrained environments
- **Real-world relevance** through realistic data heterogeneity simulation
- **Scalable framework** applicable to various SLM architectures

---

## Universal Configuration Files Summary
All experimental configurations are documented in YAML files with:
- **Model specifications**: Architecture details and parameter counts for all 5 SLMs
- **Training protocols**: Hyperparameters and optimization settings
- **Communication settings**: Network configuration and timeout management
- **Resource monitoring**: GPU/CPU tracking and validation logging
- **Data handling**: Partitioning strategies and sample distribution

## KEY UNIVERSAL FEATURES:
1. **Model Agnostic**: Framework applies to all 5 SLM architectures
2. **Consistent Data**: Same dataset quantities across all models
3. **Standardized Protocols**: Uniform training and communication settings
4. **Comprehensive Coverage**: All experimental paradigms for each model
5. **Reproducible Research**: Exact configurations for academic validation

This UNIVERSAL data contribution analysis provides a comprehensive foundation for reproducible research across all Small Language Model architectures with exact sample quantities and standardized experimental protocols.
