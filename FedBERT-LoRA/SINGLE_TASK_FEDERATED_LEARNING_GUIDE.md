# Single-Task Federated Learning Guide

## Overview

This guide explains how to train the federated learning system in **single-task mode**, where multiple clients collaborate on the **same task** with **partitioned data**.

### What is Single-Task Federated Learning?

**Single-Task FL** is the traditional federated learning approach where:
- Multiple clients train on the **same task** (e.g., all on SST-2)
- Each client has a **different subset** of the data
- Server aggregates updates to improve a **single global model**
- No multi-task learning involved

### Multi-Task vs Single-Task Comparison

| Aspect | Multi-Task FL (Main Branch) | Single-Task FL (This Branch) |
|--------|---------------------------|----------------------------|
| **Tasks per Client** | 1 different task per client | Same task for all clients |
| **Example** | Client1=SST-2, Client2=QQP, Client3=STS-B | All clients=SST-2 |
| **Data** | Full dataset per task | Partitioned dataset per client |
| **Model** | Multi-task heads | Single-task head |
| **Aggregation** | Cross-task knowledge sharing | Same-task model averaging |
| **Use Case** | Multi-task learning research | Traditional FL, privacy-preserving |
| **Complexity** | Higher (task coordination) | Lower (single objective) |

---

## Quick Start

### 1. Server Setup

Start the federated learning server:

```bash
python federated_main.py --mode server --config federated_config.yaml
```

The server will:
- Listen on port 8771 (WebSocket)
- Wait for 3 clients to connect
- Aggregate updates for the specified task

### 2. Client Setup (Single Task)

Start 3 clients, all working on the **same task**:

#### Example 1: All Clients on SST-2

```bash
# Terminal 2 - Client 1
python federated_main.py --mode client --client_id sst2_client_1 --tasks sst2

# Terminal 3 - Client 2
python federated_main.py --mode client --client_id sst2_client_2 --tasks sst2

# Terminal 4 - Client 3
python federated_main.py --mode client --client_id sst2_client_3 --tasks sst2
```

#### Example 2: All Clients on QQP

```bash
# Terminal 2 - Client 1
python federated_main.py --mode client --client_id qqp_client_1 --tasks qqp

# Terminal 3 - Client 2
python federated_main.py --mode client --client_id qqp_client_2 --tasks qqp

# Terminal 4 - Client 3
python federated_main.py --mode client --client_id qqp_client_3 --tasks qqp
```

#### Example 3: All Clients on STS-B

```bash
# Terminal 2 - Client 1
python federated_main.py --mode client --client_id stsb_client_1 --tasks stsb

# Terminal 3 - Client 2
python federated_main.py --mode client --client_id stsb_client_2 --tasks stsb

# Terminal 4 - Client 3
python federated_main.py --mode client --client_id stsb_client_3 --tasks stsb
```

---

## Data Partitioning Strategies

**Critical:** Each client must have a **different subset** of the data to simulate real federated learning scenarios.

### Strategy 1: IID (Independent and Identically Distributed)

**Best for:** Baseline comparisons, homogeneous clients

Each client gets a **random subset** of the data with similar distribution.

```python
# Example: 66,477 SST-2 samples → 3 clients
Client 1: samples 0-22,158      (33.3%, random selection)
Client 2: samples 22,159-44,317 (33.3%, random selection)
Client 3: samples 44,318-66,476 (33.3%, random selection)
```

**Configuration:**
```yaml
task_configs:
  sst2:
    train_samples: 66477        # Total dataset
    val_samples: 872
    partition_strategy: "iid"
    num_clients: 3
```

**Expected Result:**
- All clients have similar accuracy
- Fast convergence
- Good generalization

### Strategy 2: Non-IID (Heterogeneous Data)

**Best for:** Realistic federated learning, privacy-preserving scenarios

Each client gets data with **different characteristics** (e.g., label skew, feature distribution).

#### Option A: Label Skew

```python
# Example: SST-2 (binary sentiment)
Client 1: 80% positive, 20% negative reviews
Client 2: 50% positive, 50% negative reviews
Client 3: 20% positive, 80% negative reviews
```

#### Option B: Quantity Skew

```python
# Example: Different data amounts
Client 1: 10,000 samples (15%)
Client 2: 26,477 samples (40%)
Client 3: 30,000 samples (45%)
```

**Configuration:**
```yaml
task_configs:
  sst2:
    partition_strategy: "non_iid"
    label_skew: 0.7           # Higher = more skewed
    quantity_skew: true
    alpha: 0.5                # Dirichlet distribution parameter
```

**Expected Result:**
- Clients have different local accuracies
- Slower convergence
- Tests robustness of FL algorithm

### Strategy 3: User-Based Partitioning

**Best for:** Realistic user scenarios (e.g., mobile devices)

Data naturally partitioned by user ID (if available in dataset).

```python
# Example: Twitter sentiment by user
Client 1: Users 1-1000's tweets
Client 2: Users 1001-2000's tweets
Client 3: Users 2001-3000's tweets
```

---

## Configuration Guide

### Basic Configuration (federated_config.yaml)

```yaml
model:
  server_model: "bert-base-uncased"
  client_model: "prajjwal1/bert-tiny"

lora:
  rank: 16
  alpha: 64.0
  dropout: 0.1
  unfreeze_layers: 2

training:
  num_rounds: 30
  expected_clients: 3           # All on same task
  local_epochs: 1
  batch_size: 8
  learning_rate: 0.0002

# Single-Task Configuration
task_configs:
  sst2:                          # Primary task
    train_samples: 66477
    val_samples: 872
    partition_strategy: "iid"   # or "non_iid"
    num_clients: 3
    random_seed: 42

communication:
  port: 8771
  round_timeout: 3400
  send_timeout: 3600
```

### Advanced Configuration

#### For Non-IID Training:

```yaml
task_configs:
  sst2:
    train_samples: 66477
    val_samples: 872
    partition_strategy: "non_iid"
    
    # Non-IID parameters
    label_distribution:
      client_1: [0.8, 0.2]      # 80% class 0, 20% class 1
      client_2: [0.5, 0.5]      # Balanced
      client_3: [0.2, 0.8]      # 20% class 0, 80% class 1
    
    # Or use Dirichlet distribution
    dirichlet_alpha: 0.5        # Lower = more skewed
    
    # Quantity skew
    client_samples:
      client_1: 20000
      client_2: 25000
      client_3: 21477
```

#### For Different Tasks:

Just change the task name and samples:

```yaml
task_configs:
  qqp:                           # For QQP task
    train_samples: 323415        # Full QQP dataset
    val_samples: 40431
    partition_strategy: "iid"
    num_clients: 3

  stsb:                          # For STS-B task
    train_samples: 4249
    val_samples: 1500
    partition_strategy: "iid"
    num_clients: 3
```

---

## Implementation Details

### Data Partitioning Code

If not already implemented, add this to `src/datasets/federated_datasets.py`:

```python
class DatasetPartitioner:
    """Partition dataset for single-task federated learning"""
    
    def __init__(self, dataset, num_clients, strategy='iid', seed=42):
        self.dataset = dataset
        self.num_clients = num_clients
        self.strategy = strategy
        self.seed = seed
        
    def partition_iid(self, client_id):
        """IID partitioning: random split"""
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.dataset))
        
        # Split into num_clients chunks
        client_indices = np.array_split(indices, self.num_clients)
        
        # Get this client's chunk
        client_idx = int(client_id.split('_')[-1]) - 1
        return self.dataset.select(client_indices[client_idx])
    
    def partition_non_iid(self, client_id, alpha=0.5):
        """Non-IID partitioning: label skew using Dirichlet"""
        labels = np.array(self.dataset['label'])
        num_classes = len(np.unique(labels))
        
        # Dirichlet distribution for label allocation
        label_distribution = np.random.dirichlet([alpha] * num_classes, self.num_clients)
        
        # Allocate samples based on label distribution
        client_idx = int(client_id.split('_')[-1]) - 1
        client_samples = []
        
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            num_samples = int(len(class_indices) * label_distribution[client_idx][class_id])
            selected = np.random.choice(class_indices, num_samples, replace=False)
            client_samples.extend(selected)
        
        return self.dataset.select(client_samples)
```

### Client Initialization

Modify `federated_client.py` to use partitioner:

```python
def initialize_dataset_handlers(self) -> Dict[str, Any]:
    """Initialize dataset handlers with partitioning"""
    handlers = {}
    
    for task in self.tasks:
        if task in self.config.task_configs:
            task_config = self.config.task_configs[task]
            
            # Create partitioner
            dataset_config = DatasetConfig(
                task_name=task,
                train_samples=task_config.get('train_samples'),
                val_samples=task_config.get('val_samples'),
                random_seed=task_config.get('random_seed', 42),
                
                # Single-task FL specific
                client_id=self.client_id,
                partition_strategy=task_config.get('partition_strategy', 'iid'),
                num_clients=self.config.expected_clients
            )
            
            handlers[task] = DatasetFactory.create_handler(task, dataset_config)
    
    return handlers
```

---

## Expected Results

### SST-2 (Sentiment Analysis)

**IID Partitioning:**
```
Round 10:
  Client 1: 88.5% (similar to others)
  Client 2: 88.2%
  Client 3: 88.7%
  Global: 89.1%

Round 30:
  Global: 92-93% (comparable to multi-task 92.89%)
```

**Non-IID Partitioning:**
```
Round 10:
  Client 1: 85.2% (positive-heavy)
  Client 2: 88.5% (balanced)
  Client 3: 84.8% (negative-heavy)
  Global: 87.5%

Round 30:
  Global: 90-91% (slightly lower than IID)
```

### QQP (Question Pairs)

**IID Partitioning:**
```
With full dataset (323K samples):
Round 30:
  Global: 86-88% (better than multi-task 78.97% with 32K)
```

**Note:** QQP with full dataset will take much longer (~3 hours per round with batch_size=8)

### STS-B (Semantic Similarity)

**IID Partitioning:**
```
Round 15:
  Global: 75-77% correlation (comparable to multi-task 73.87%)
```

---

## Advantages of Single-Task FL

### ✅ Pros:

1. **Simpler Architecture**
   - No multi-task coordination
   - Single model to optimize
   - Easier debugging

2. **Better Task-Specific Performance**
   - Full focus on one task
   - No task interference
   - Potentially higher accuracy

3. **Traditional FL Benchmark**
   - Standard FL scenario
   - Easier comparison with other work
   - Well-studied problem

4. **Scalability**
   - Can use full dataset (if partitioned)
   - More clients = more data coverage
   - Natural horizontal scaling

5. **Privacy-Preserving**
   - Same as multi-task
   - Data stays on client devices
   - Only model updates shared

### ⚠️ Cons vs Multi-Task:

1. **No Cross-Task Learning**
   - Can't leverage knowledge from other tasks
   - No transfer learning benefits

2. **Requires More Clients**
   - Need 3+ clients per task
   - Total: 9 clients for 3 tasks (vs 3 in multi-task)

3. **More Computational Resources**
   - 3 separate training sessions needed
   - 3x server time
   - 3x communication overhead

---

## Comparison: Expected Performance

### Single-Task (This Branch) vs Multi-Task (Main Branch)

| Task | Multi-Task (3 clients) | Single-Task IID (3 clients) | Single-Task Non-IID |
|------|----------------------|---------------------------|-------------------|
| **SST-2** | 92.89% | **93-94%** (expected) | 90-91% |
| **QQP** (full) | 78.97% (32K) | **86-88%** (323K) | 84-86% |
| **STS-B** | 73.87% | **75-77%** | 72-74% |

**Hypothesis:** Single-task should achieve **slightly higher accuracy** on each task because:
- Full focus on one objective
- No task interference
- Can use full dataset
- More training data per task (3 clients × data)

---

## Troubleshooting

### Issue 1: All Clients Getting Same Data

**Symptom:** All clients converge to same accuracy immediately

**Solution:** Verify data partitioning is working:
```python
# Add logging in dataset handler
logger.info(f"Client {client_id} has {len(dataset)} samples")
logger.info(f"First 5 sample indices: {dataset[:5]['idx']}")
```

### Issue 2: One Client Not Participating

**Symptom:** Server shows "Waiting for: ['sst2_client_3']"

**Solution:** 
- Check client_id matches expected name
- Verify task name is correct
- Check network connection

### Issue 3: Poor Global Model Performance

**Symptom:** Global accuracy < individual client accuracy

**Solution:**
- Check aggregation weights (should be by data size)
- Verify all clients are training properly
- Increase training rounds

### Issue 4: Data Imbalance

**Symptom:** Large accuracy variation between clients

**Solution:**
- If intentional (non-IID): This is expected
- If unintentional: Fix partitioning to balance data

---

## Running Experiments

### Experiment 1: IID vs Non-IID

```bash
# Run 1: IID partitioning
# Edit config: partition_strategy: "iid"
python federated_main.py --mode server &
python federated_main.py --mode client --client_id sst2_client_1 --tasks sst2 &
python federated_main.py --mode client --client_id sst2_client_2 --tasks sst2 &
python federated_main.py --mode client --client_id sst2_client_3 --tasks sst2

# Run 2: Non-IID partitioning
# Edit config: partition_strategy: "non_iid", alpha: 0.5
# Repeat same commands
```

**Compare:** Convergence speed, final accuracy, client variance

### Experiment 2: Number of Clients

```bash
# Test with 2, 3, 5, 10 clients
# Hypothesis: More clients = better global model
```

### Experiment 3: Multi-Task vs Single-Task

```bash
# Compare:
# - Multi-task: 3 clients, 3 tasks (main branch)
# - Single-task: 3 clients per task = 9 total (this branch)
```

---

## Best Practices

1. **Always Partition Data:**
   - Use IID for baseline
   - Use non-IID for realistic FL

2. **Monitor Client Convergence:**
   - Log individual client metrics
   - Check for stragglers
   - Identify non-participating clients

3. **Adjust Aggregation:**
   - Weight by data size
   - Consider client reliability
   - Handle dropouts gracefully

4. **Save Checkpoints:**
   - Save global model each round
   - Enable rollback if needed
   - Track best validation accuracy

5. **Compare Fairly:**
   - Same total data amount
   - Same number of rounds
   - Same hyperparameters

---

## Next Steps

### For Research:

1. **Benchmark All Three Tasks:**
   - Run SST-2, QQP, STS-B separately
   - Compare with multi-task results
   - Publish comparison table

2. **Test Different Partitioning:**
   - IID vs Non-IID
   - Different alpha values
   - Different client numbers

3. **Optimize for Single-Task:**
   - Task-specific learning rates
   - Task-specific architectures
   - Task-specific aggregation strategies

### For Production:

1. **Scale to More Clients:**
   - Test with 10, 50, 100 clients
   - Implement client sampling
   - Add fault tolerance

2. **Privacy Analysis:**
   - Measure differential privacy
   - Test against inference attacks
   - Add secure aggregation

3. **Deployment:**
   - Containerize clients
   - Deploy on edge devices
   - Monitor in production

---

## Summary

**Single-Task Federated Learning** is the traditional FL approach where multiple clients collaborate on the same task with partitioned data.

**Key Commands:**
```bash
# Server
python federated_main.py --mode server --config federated_config.yaml

# Clients (all same task)
python federated_main.py --mode client --client_id client_1 --tasks sst2
python federated_main.py --mode client --client_id client_2 --tasks sst2
python federated_main.py --mode client --client_id client_3 --tasks sst2
```

**Expected Outcome:**
- Similar or better accuracy than multi-task
- Traditional FL scenario
- Easier to compare with other FL work

**Use When:**
- Benchmarking against traditional FL
- Single task is the goal
- Want best task-specific performance
- Have multiple clients with same task data

**Main Branch (Multi-Task) Use When:**
- Want cross-task learning
- Have diverse client tasks
- Research multi-task FL
- Limited number of clients

---

**Generated:** October 26, 2025  
**Branch:** FL_Client_Server_Single_Task  
**For:** Single-task federated learning experiments

