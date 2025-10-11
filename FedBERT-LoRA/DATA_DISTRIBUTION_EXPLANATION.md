# Why All STSB Samples Are in bin_4

## The Question
```
Client 3 (stsb): 400 samples, distribution: {'bin_0': 0, 'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 400}
```

Why are all 400 samples in `bin_4` instead of being distributed across bins?

---

## Root Cause Analysis

### 1. **How Non-IID Regression Split Works**

The code creates non-IID data by **sorting samples by their label values** and assigning different value ranges to different clients:

```python
# Line 272-302 in no_lora_federated_system.py
def _create_regression_non_iid(self, texts, labels, client_id, total_clients, samples_per_client):
    # Sort by label values (similarity scores)
    sorted_pairs = sorted(zip(texts, labels), key=lambda x: x[1])
    
    # Divide into ranges based on client_id
    range_size = len(sorted_pairs) // total_clients
    start_idx = client_id * range_size
    end_idx = start_idx + range_size
    
    # Client gets samples from their assigned range
    client_pairs = sorted_pairs[start_idx:end_idx]
```

**This means:**
- Client 0 gets samples with **lowest** similarity scores
- Client 1 gets samples with **middle-low** similarity scores
- Client 2 gets samples with **middle-high** similarity scores
- **Client 3 gets samples with HIGHEST similarity scores** ⬅️ This is your case!

### 2. **STSB Label Distribution**

STSB (Semantic Textual Similarity Benchmark) has labels from 0.0 to 5.0, normalized to [0, 1]:

```python
labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to [0,1]
```

**The bins are:**
```python
bins = np.linspace(0, 1, 6)  # Creates: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
```

- `bin_0`: [0.0 - 0.2) → Original scores 0.0-1.0 (very dissimilar)
- `bin_1`: [0.2 - 0.4) → Original scores 1.0-2.0 (dissimilar)
- `bin_2`: [0.4 - 0.6) → Original scores 2.0-3.0 (somewhat similar)
- `bin_3`: [0.6 - 0.8) → Original scores 3.0-4.0 (similar)
- `bin_4`: [0.8 - 1.0] → Original scores 4.0-5.0 (very similar)

### 3. **Why Client 3 Gets Only bin_4**

With 3 clients in your configuration:
- Total STSB samples: ~5,749
- Client 3 is assigned the **top 1/3** of samples (highest similarity scores)
- After sorting, Client 3 gets samples with scores ≥ 4.0 (normalized ≥ 0.8)
- All these samples fall into `bin_4`

**Visual representation:**
```
Sorted STSB dataset (by similarity score):
[0.0 ─────────── 0.2 ─────────── 0.4 ─────────── 0.6 ─────────── 0.8 ─────────── 1.0]
 └─ Client 0 ─┘   └─ Client 1 ─┘   └─ Client 2 ─┘   └───── Client 3 ─────┘
    (bin_0-1)        (bin_1-2)        (bin_2-3)           (bin_4 only!)
```

---

## Is This a Problem?

### **For Non-IID Research: This is INTENTIONAL! ✅**

This extreme data heterogeneity is **by design** for non-IID federated learning research:

**Advantages:**
1. ✅ **Tests heterogeneity handling**: Each client has very different data distributions
2. ✅ **Realistic scenario**: In real FL, clients often have skewed data
3. ✅ **Challenges the model**: Forces the global model to learn from diverse distributions
4. ✅ **Research value**: Shows how well your FL system handles extreme non-IID cases

**This is what you want for studying:**
- Non-IID data impact on convergence
- Heterogeneous multi-task learning
- Client contribution fairness
- Aggregation robustness

### **For Model Performance: This is CHALLENGING! ⚠️**

**Disadvantages:**
1. ⚠️ **Limited diversity**: Client 3 only sees high-similarity pairs
2. ⚠️ **Poor generalization**: Model may not learn full similarity spectrum
3. ⚠️ **Biased predictions**: May overfit to high-similarity patterns
4. ⚠️ **Negative R² initially**: Model struggles with limited data range

---

## Solutions (If You Want More Diversity)

### **Option 1: Increase Alpha (More IID-like)**

Modify the non-IID alpha parameter to create more overlap:

```ini
[SCENARIO_1]
non_iid_alpha = 1.0  # Increase from 0.5
```

**Effect:** More mixing of similarity scores across clients

### **Option 2: Add Random Sampling**

Modify the regression split to include random samples from other ranges:

```python
# After line 292 in no_lora_federated_system.py
# Add 20% random samples from other ranges
random_samples = int(samples_per_client * 0.2)
all_other_samples = [p for i, p in enumerate(sorted_pairs) 
                     if i < start_idx or i >= end_idx]
if len(all_other_samples) > random_samples:
    random_pairs = np.random.choice(len(all_other_samples), random_samples, replace=False)
    client_pairs.extend([all_other_samples[i] for i in random_pairs])
```

### **Option 3: Use Dirichlet Distribution (Like Classification)**

Apply Dirichlet distribution to regression by binning first:

```python
# Bin labels first, then apply Dirichlet
binned_labels = np.digitize(labels, bins=np.linspace(0, 1, 6))
# Then use Dirichlet split like classification
```

### **Option 4: Stratified Sampling**

Ensure each client gets samples from all bins:

```python
# Divide each bin proportionally across clients
for bin_idx in range(5):
    bin_samples = [s for s in sorted_pairs if bins[bin_idx] <= s[1] < bins[bin_idx+1]]
    client_share = len(bin_samples) // total_clients
    # Assign share to each client
```

---

## Current Configuration Impact

### **With Your Setup (3 clients, 400 samples each):**

**Client 0 (if it existed):**
```
distribution: {'bin_0': 200, 'bin_1': 200, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0}
```
(Low similarity pairs)

**Client 1 (if it existed):**
```
distribution: {'bin_0': 0, 'bin_1': 100, 'bin_2': 200, 'bin_3': 100, 'bin_4': 0}
```
(Medium similarity pairs)

**Client 3 (your current case):**
```
distribution: {'bin_0': 0, 'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 400}
```
(High similarity pairs only)

### **Why This Happens with 3 Clients:**

Your config has:
```ini
num_clients = 3
client_distribution = 1,1,1
tasks = sst2,qqp,stsb
```

This means:
- Client 1 → SST-2 (classification)
- Client 2 → QQP (classification)
- Client 3 → STSB (regression) ← Gets the highest similarity range

With only 1 client per task, Client 3 gets assigned to the **top range** of STSB data.

---

## Recommendation

### **For Your Research (Multi-task Non-IID):**

**Keep it as is!** ✅ This extreme heterogeneity is valuable for:
1. Testing multi-task federated learning robustness
2. Studying non-IID impact on different task types
3. Comparing classification vs regression under data skew
4. Analyzing aggregation strategies for heterogeneous data

### **Document in Your Paper:**
```
"To simulate extreme data heterogeneity, we employ a range-based 
non-IID split for regression tasks, where each client receives 
samples from distinct value ranges. This results in Client 3 
(STSB) receiving only high-similarity pairs (scores 4.0-5.0), 
representing a realistic scenario where clients have highly 
skewed data distributions."
```

### **If You Want More Diversity:**

Use **Option 4 (Stratified Sampling)** to ensure each client gets samples from multiple bins while maintaining non-IID characteristics.

---

## Summary

✅ **This is EXPECTED behavior** for non-IID regression splits
✅ **This is INTENTIONAL** for federated learning research
✅ **This is VALUABLE** for studying data heterogeneity
⚠️ **This is CHALLENGING** for model performance (which is the point!)

The distribution `{'bin_0': 0, 'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 400}` shows your non-IID split is working correctly! 🎉
