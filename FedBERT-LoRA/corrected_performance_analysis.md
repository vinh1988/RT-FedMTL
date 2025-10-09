# Corrected Performance Analysis - Complete Data Extraction

## Systematic Analysis of All Experiment Results

After carefully examining all 163 CSV files across 4 scenario directories, here are the **actual performance results**:

## **Actual Performance Data (from CSV files)**

### **Scenario 1: Heterogeneous Multi-Task (No LoRA)**
**Configuration**: 3 clients, 30 rounds
- **Final Accuracy**: 0.6651 (66.51%)
- **Final F1-Score**: 0.6651 (66.51%)
- **Final Communication Time**: 4.99s
- **Final Memory Usage**: 4691.1MB

**2-clients configuration**:
- **Final Accuracy**: 0.9970 (99.70%)
- **Final F1-Score**: 0.9970 (99.70%)
- **Final Communication Time**: 4.57s
- **Final Memory Usage**: 3562.0MB

### **Scenario 2: Heterogeneous Multi-Task (LoRA Simulated)**
**Configuration**: 3 clients, 30 rounds
- **Final Accuracy**: 0.6651 (66.51%)
- **Final F1-Score**: 0.6651 (66.51%)
- **Final Communication Time**: 5.14s
- **Final Memory Usage**: 4729.1MB

**2-clients configuration**:
- **Final Accuracy**: 0.9978 (99.78%)
- **Final F1-Score**: 0.9978 (99.78%)
- **Final Communication Time**: 4.93s
- **Final Memory Usage**: 3571.3MB

### **Scenario 3: Heterogeneous Single-Task**
**QQP Task (3 clients, 30 rounds)**:
- **Final Accuracy**: 0.6644 (66.44%)
- **Final F1-Score**: 0.6644 (66.44%)
- **Final Communication Time**: 4.74s
- **Final Memory Usage**: 4715.4MB

**SST2 Task (3 clients, 30 rounds)**:
- **Final Accuracy**: 0.6639 (66.39%)
- **Final F1-Score**: 0.6639 (66.39%)
- **Final Communication Time**: 4.67s
- **Final Memory Usage**: 4727.9MB

**STS-B Task (3 clients, 30 rounds)**:
- **Final Accuracy**: 0.6652 (66.52%)
- **Final F1-Score**: 0.6652 (66.52%)
- **Final Communication Time**: 4.67s
- **Final Memory Usage**: 4712.8MB

### **Scenario 4: Homogeneous Multi-Task**
**Status**: No data files found (empty directory)

## **LoRA Enhancement Summary (from CSV)**
- **Accuracy Improvement**: 15% (0.55 → 0.63)
- **Parameter Reduction**: 90% (4.4M → 440K parameters)
- **Communication Reduction**: 75% (15.2MB → 3.8MB)
- **Training Time Reduction**: 40% (100s → 60s)

## **Key Findings**

### **1. README Values Are NOT in CSV Files**
The README values (0.4397, 0.3793, etc.) are **completely absent** from all 163 CSV files. They appear to be:
- Simulated/expected values
- From a different experiment run not captured in CSV
- From a different configuration not present in the data

### **2. Actual Performance is Much Higher**
- **3-clients, 30 rounds**: ~66% accuracy and F1-score
- **2-clients, 30 rounds**: ~99% accuracy and F1-score
- **Communication times**: 4.5-5.2 seconds
- **Memory usage**: 3500-4700 MB

### **3. Performance Rankings (Actual Data)**

**F1-Score Rankings (3-clients, 30 rounds)**:
1. **Scenario 2 (LoRA)**: 0.6651 (Best)
2. **Scenario 1 (No LoRA)**: 0.6651 (Tied for Best)
3. **Scenario 3 STS-B**: 0.6652 (Best Single-Task)
4. **Scenario 3 QQP**: 0.6644 (Second Single-Task)
5. **Scenario 3 SST2**: 0.6639 (Third Single-Task)

**Communication Efficiency (3-clients, 30 rounds)**:
1. **Scenario 3 SST2**: 4.67s (Fastest)
2. **Scenario 3 STS-B**: 4.67s (Tied for Fastest)
3. **Scenario 3 QQP**: 4.74s (Third)
4. **Scenario 1**: 4.99s (Fourth)
5. **Scenario 2**: 5.14s (Slowest)

**Memory Efficiency (3-clients, 30 rounds)**:
1. **Scenario 3 STS-B**: 4712.8MB (Most Efficient)
2. **Scenario 3 QQP**: 4715.4MB (Second)
3. **Scenario 1**: 4691.1MB (Third)
4. **Scenario 3 SST2**: 4727.9MB (Fourth)
5. **Scenario 2**: 4729.1MB (Least Efficient)

## **Data Structure Analysis**

### **File Types**:
1. **Scalability Metrics**: Aggregated performance across rounds (accuracy, F1, communication time, memory)
2. **Participation Metrics**: Individual client performance per round
3. **Non-IID Metrics**: Data heterogeneity and fairness measures
4. **LoRA Enhancement Summary**: Simulated LoRA benefits

### **Client Configurations**:
- **2 clients**: 2-30 rounds
- **3 clients**: 2-30 rounds  
- **4-10 clients**: Various round counts

## **Conclusion**

The generated documents contain **completely incorrect performance data** that doesn't match any of the actual experiment results. The README values appear to be simulated or from a different experiment run entirely.

**Correct Performance Data**:
- **3-clients, 30 rounds**: ~66% accuracy/F1-score, 4.5-5.2s communication, 4700MB memory
- **2-clients, 30 rounds**: ~99% accuracy/F1-score, 4.5-4.9s communication, 3500MB memory

**All documents need to be updated with the correct performance data from the actual CSV files.**
