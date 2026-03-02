# MINI-LM Comprehensive Metrics Summary

> [!NOTE]
> **Resource_Usage** priority: `Avg_GPU_Mem_GB` > `Avg_GPU_Usage`.

| Paradigm | Task | Experiment | SST-2 | | QQP | | STS-B | | Time | Res |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | **Acc** | **F1** | **Acc** | **F1** | **Pear** | **Spear** | **(s)** | **(GB)** |
| Centralized | Multi-Task (MTL) | centralized-mtl-all-tasks | 0.8796 | 0.8829 | 0.7917 | 0.8649 | 0.8580 | 0.8560 | 12437.12 | 1.5045 |
| Centralized | Single-Task | centralized-single-task-qqp | - | - | 0.7990 | 0.8656 | - | - | 2388.13 | 1.5067 |
| Centralized | Single-Task | centralized-single-task-sst2 | 0.9014 | 0.9051 | - | - | - | - | 8334.10 | 1.4950 |
| Centralized | Single-Task | centralized-single-task-stsb | - | - | - | - | 0.8620 | 0.8592 | 2696.27 | 1.4929 |
| FL | Multi-Task (MTL) | fl-mtl-slms-mini-lm-sts-qqp-sst2-lora | 0.7787 | 0.8102 | 0.7271 | 0.6060 | 0.0703 | 0.0621 | 18808.08 | 0.1090 |
| FL | Multi-Task (MTL) | fl-mtl-slms-mini-lm-stsb-qqp-sst2 | 0.9369 | 0.9428 | 0.8966 | 0.8662 | 0.8384 | 0.8197 | 27930.06 | 0.3835 |
| FL | Multi-Task (MTL) | fl-mtl-slms-mini-lm-non-iid-stsb-qqp-sst2-3client-each | 0.8888 | 0.8974 | 0.8629 | 0.8150 | 0.7003 | 0.7061 | 20765.57 | 0.3846 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-qqp | - | - | 0.8421 | 0.7644 | - | - | 14583.05 | 0.3842 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-sst2 | 0.9243 | 0.9293 | - | - | - | - | 8447.20 | 0.3836 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-stsb | - | - | - | - | 0.7877 | 0.7489 | 4920.48 | 0.3840 |
