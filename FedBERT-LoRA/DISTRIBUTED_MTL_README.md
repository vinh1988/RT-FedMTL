# 🚀 Distributed Multi-Task Learning (MTL) System

## Overview

This system implements **Distributed Multi-Task Learning** without federated learning. Unlike traditional federated learning where clients share and aggregate model parameters, this system focuses on:

- **Independent Training**: Each client trains on its specific dataset independently
- **Multi-Task Learning**: Each client performs MTL with transfer learning between tasks
- **Coordination**: Server coordinates training rounds but doesn't aggregate parameters
- **Transfer Learning**: Knowledge distillation enables transfer learning between tasks

## Architecture

### Key Differences from Federated Learning

| Aspect | Traditional Federated Learning | Distributed MTL |
|--------|--------------------------------|------------------|
| **Parameter Sharing** | Clients share parameters with server | No parameter sharing |
| **Model Aggregation** | Server aggregates client models | No aggregation |
| **Training Focus** | Collaborative model training | Independent MTL with transfer learning |
| **Communication** | Heavy (full model parameters) | Light (metrics only) |
| **Privacy** | Strong privacy guarantees | Privacy through independence |

### System Components

#### 🖥️ **Server (BERT-Base)**
- Coordinates training rounds
- Uses BERT-Base as teacher model
- Collects and logs client metrics
- No parameter aggregation

#### 🤖 **Clients (Dataset-Specific)**
- Each client holds one specific dataset (SST2, QQP, or STSB)
- Performs multi-task learning with transfer learning
- Trains multiple models with knowledge distillation
- Reports metrics back to server

## Features

### ✅ **Multi-Task Learning**
- Each client trains multiple models simultaneously
- Transfer learning between classification and regression tasks
- Knowledge distillation for improved performance

### ✅ **Transfer Learning**
- Knowledge distillation between different task types
- Improved generalization across tasks
- Efficient knowledge transfer without parameter sharing

### ✅ **WebSocket Communication**
- Maintains WebSocket for coordination
- Lightweight communication (metrics only)
- Real-time progress monitoring

### ✅ **Comprehensive Metrics**
- Task-specific performance metrics
- Transfer learning efficiency measurements
- Knowledge retention analysis
- Memory and timing statistics

## Quick Start

### 🚀 **Complete System Launch**

```bash
cd /home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA
source venv/bin/activate
./run_distributed_mtl.sh

# Select option 1 for complete system
```

This launches:
- **Server**: BERT-Base coordination server
- **Client SST2**: Trains on sentiment analysis dataset
- **Client QQP**: Trains on question pair dataset
- **Client STSB**: Trains on semantic similarity dataset

### 📋 **Manual Operation**

#### **Start Server Only**
```bash
python distributed_mtl_system.py --mode server --rounds 5
```

#### **Start Individual Clients**
```bash
# SST2 Client (Sentiment Analysis)
python distributed_mtl_system.py --mode client --client_id client_sst2 --dataset sst2

# QQP Client (Question Pairs)
python distributed_mtl_system.py --mode client --client_id client_qqp --dataset qqp

# STSB Client (Semantic Similarity)
python distributed_mtl_system.py --mode client --client_id client_stsb --dataset stsb
```

## Configuration

### **System Parameters**

```ini
[DISTRIBUTED_MTL_SETTINGS]
# Server configuration
server_model = bert-base-uncased
port = 8771

# Training settings
num_rounds = 5
local_epochs = 2
batch_size = 16
learning_rate = 2e-5

# Multi-task learning settings
distillation_temperature = 3.0
distillation_alpha = 0.5

# Data settings
samples_per_client = 800
data_distribution = iid

### **Dataset Information**

| Dataset | Task Type | Description |
|---------|-----------|-------------|
| **SST2** | Classification | Sentiment Analysis (Positive/Negative) |
| **QQP** | Classification | Question Pair Classification (Duplicate/Similar) |
| **STSB** | Regression | Semantic Textual Similarity (0-1 scale) |

## Multi-Task Learning Architecture

{{ ... }}

1. **Data Loading**: Each client loads its specific dataset
2. **Model Initialization**: Multiple models created for different task types
3. **Transfer Learning**:
   - Classification models for binary tasks
   - Regression models for similarity tasks
   - Knowledge distillation between models
4. **Joint Training**: All models trained simultaneously with shared representations
5. **Metrics Collection**: Performance metrics calculated for each task

### **Transfer Learning Mechanism**

- **Knowledge Distillation**: Teacher-student learning between task types
- **Temperature Scaling**: Soft targets for better knowledge transfer
- **Alpha Balancing**: Balance between task loss and distillation loss

## Metrics and Monitoring

### **Client Metrics**
- **Task-specific accuracy/score** for each trained model
- **Transfer efficiency**: How well knowledge transfers between tasks
- **Knowledge retention**: Consistency across different tasks
- **Training time and memory usage**

### **System Monitoring**
- Server logs: `distributed_mtl.log`
- Real-time progress tracking
- Resource utilization statistics

## Key Benefits

### ✅ **Advantages over Federated Learning**

1. **Reduced Communication**: No parameter sharing, only metrics exchange
2. **Better Privacy**: Complete model independence between clients
3. **Scalability**: No aggregation bottlenecks
4. **Flexibility**: Each client can use different architectures
5. **Task Specialization**: Clients focus on specific domains

### ✅ **Multi-Task Learning Benefits**

1. **Improved Generalization**: Transfer learning across related tasks
2. **Data Efficiency**: Better utilization of limited datasets
3. **Knowledge Sharing**: Implicit knowledge transfer without parameter exchange
4. **Robustness**: Better performance through task relationships

## Results and Analysis

### **📊 CSV Output Files**

The system automatically saves comprehensive metrics to CSV files for analysis:

#### **📈 Client Metrics (`client_metrics_YYYYMMDD_HHMMSS.csv`)**
Detailed per-client, per-task performance metrics:
```csv
round,client_id,dataset,task_type,accuracy,precision,recall,f1_score,loss,kd_loss,task_loss,training_time,memory_usage_mb,transfer_efficiency,knowledge_retention
1,client_sst2,sst2,binary_classification,0.58,0.58,0.58,0.56,0.46,0.24,0.68,2.51,1644.0,0.59,0.0
1,client_sst2,sst2,regression,-0.40,0.47,0.02,0.59,0.26,0.18,0.35,2.51,1644.0,0.59,0.0
```

**Columns:**
- `round`: Training round number
- `client_id`: Unique client identifier
- `dataset`: Dataset name (sst2, qqp, stsb)
- `task_type`: Type of model (binary_classification, regression)
- `accuracy`: Task-specific accuracy metric
- `precision, recall, f1_score`: Classification metrics
- `loss, kd_loss, task_loss`: Training loss components
- `training_time`: Time taken for training (seconds)
- `memory_usage_mb`: Memory consumption
- `transfer_efficiency`: Knowledge transfer quality
- `knowledge_retention`: Performance consistency across tasks

#### **📈 Round Metrics (`round_metrics_YYYYMMDD_HHMMSS.csv`)**
Aggregated results per training round:
```csv
round,num_clients,avg_accuracy,accuracy_std,avg_transfer_efficiency,avg_knowledge_retention,total_training_time,communication_time
1,3,0.091,0.0,0.593,0.0,2.512,0.0
2,3,0.353,0.061,0.456,0.010,12.174,0.0
```

**Columns:**
- `round`: Training round number
- `num_clients`: Number of participating clients
- `avg_accuracy`: Average accuracy across all clients and tasks
- `accuracy_std`: Standard deviation of accuracy (lower = more consistent)
- `avg_transfer_efficiency`: Average knowledge transfer efficiency
- `avg_knowledge_retention`: Average knowledge retention score
- `total_training_time`: Total training time for all clients
- `communication_time`: Communication overhead (currently 0.0)

### **📊 Analysis and Visualization**

#### **Key Metrics to Track:**

1. **Transfer Learning Effectiveness**:
   - `transfer_efficiency`: Higher values indicate better knowledge transfer
   - `knowledge_retention`: Measures consistency across different tasks

2. **Training Performance**:
   - `accuracy`: Task-specific performance improvement over rounds
   - `training_time`: Computational efficiency
   - `memory_usage_mb`: Resource consumption

3. **System Scalability**:
   - `accuracy_std`: Consistency across clients (lower = better fairness)
   - `num_clients`: System participation rate

#### **Example Analysis Queries:**

```bash
# Average performance per dataset
awk -F',' 'NR>1 {sum[$3] += $5; count[$3]++} END {for (d in sum) print d ": " sum[d]/count[d]}' client_metrics_*.csv

# Transfer efficiency trend over rounds
awk -F',' 'NR>1 {print $1 "," $14}' round_metrics_*.csv | sort -n

# Memory usage comparison
awk -F',' 'NR>1 {print $3 "," $13}' client_metrics_*.csv | sort | uniq -c
### **Common Issues**

1. **Port Conflicts**: Ensure port 8771 is available
2. **Memory Issues**: Reduce batch size or samples per client
3. **Connection Timeouts**: Check network connectivity

### **Debug Commands**

```bash
# Check port availability
netstat -tuln | grep 8771

# Check running processes
ps aux | grep distributed_mtl

# View logs
tail -f distributed_mtl.log
```

## Performance Characteristics

- **Communication**: Lightweight (metrics only)
- **Scalability**: Linear scaling with number of clients
- **Privacy**: Maximum privacy (no parameter sharing)
- **Training Speed**: Faster than federated learning (no aggregation)
- **Resource Usage**: Lower memory footprint

---

## 🎯 **Ready for Distributed MTL Training!**

The system is optimized for **distributed multi-task learning** with **transfer learning** capabilities. Each client specializes in a specific dataset while benefiting from multi-task learning and knowledge transfer.
