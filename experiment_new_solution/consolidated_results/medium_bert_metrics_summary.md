# MEDIUM-BERT Comprehensive Metrics Summary

> [!NOTE]
> **Resource_Usage** priority: `Avg_GPU_Mem_GB` > `Avg_GPU_Usage`.

| Paradigm | Task | Experiment | SST-2 | | QQP | | STS-B | | Time | Res |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | **Acc** | **F1** | **Acc** | **F1** | **Pear** | **Spear** | **(s)** | **(GB)** |
| Centralized | Multi-Task (MTL) | centralized-mtl-all-tasks | 0.8865 | 0.8884 | 0.8333 | 0.8777 | 0.8615 | 0.8584 | 19257.40 | 2.6671 |
| Centralized | Single-Task | centralized-single-task-qqp | - | - | 0.8358 | 0.8859 | - | - | 3205.51 | 2.6467 |
| Centralized | Single-Task | centralized-single-task-sst2 | 0.8933 | 0.8970 | - | - | - | - | 13492.62 | 2.6711 |
| Centralized | Single-Task | centralized-single-task-stsb | - | - | - | - | 0.8673 | 0.8648 | 3381.83 | 2.6467 |
| FL | Multi-Task (MTL) | fl-mtl-slms-bertmedium-stsb-qqp-sst2 | 0.9392 | 0.9450 | 0.8962 | 0.8655 | 0.8258 | 0.8052 | 45258.21 | 0.6807 |
| FL | Multi-Task (MTL) | fl-mtl-slms-bertmedium-stsb-qqp-sst2-lora | 0.8131 | 0.8362 | 0.7388 | 0.6805 | 0.6823 | 0.6512 | 12885.74 | 0.1840 |
| FL | Multi-Task (MTL) | fl-mtl-slms-medium-bert-non-iid-stsb-qqp-sst2-3client-each | 0.8933 | 0.8988 | 0.8183 | 0.7441 | 0.4219 | 0.4132 | 33374.50 | 0.6790 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-qqp | - | - | 0.7989 | 0.6852 | - | - | 23880.89 | 0.6789 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-sst2 | 0.9300 | 0.9348 | - | - | - | - | 15439.04 | 0.6786 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-stsb | - | - | - | - | 0.6521 | 0.6212 | 10261.92 | 0.6810 |
