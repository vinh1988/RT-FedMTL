# TINY_BERT Comprehensive Metrics Summary

> [!NOTE]
> **Resource_Usage** priority: `Avg_GPU_Mem_GB` > `Avg_GPU_Usage`.

| Paradigm | Task | Experiment | SST-2 | | QQP | | STS-B | | Time | Res |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | **Acc** | **F1** | **Acc** | **F1** | **Pear** | **Spear** | **(s)** | **(GB)** |
| Centralized | Multi-Task (MTL) | centralized-mtl-all-tasks | 0.8211 | 0.8263 | 0.7230 | 0.8101 | 0.6474 | 0.6426 | 9227.55 | 0.3440 |
| Centralized | Single-Task | centralized-single-task-qqp | - | - | 0.7157 | 0.8139 | - | - | 2315.39 | 0.3427 |
| Centralized | Single-Task | centralized-single-task-sst2 | 0.8314 | 0.8358 | - | - | - | - | 4857.08 | 0.3427 |
| Centralized | Single-Task | centralized-single-task-stsb | - | - | - | - | 0.7270 | 0.7230 | 2337.42 | 0.3427 |
| FL | Multi-Task (MTL) | fl-mtl-slms-berttiny-stsb-qqp-sst2 | 0.9014 | 0.9145 | 0.7950 | 0.7494 | 0.5212 | 0.4754 | 6778.00 | 0.0881 |
| FL | Multi-Task (MTL) | fl-mtl-slms-berttiny-stsb-qqp-sst2-lora | 0.7569 | 0.7902 | 0.6766 | 0.4475 | 0.3636 | 0.3505 | 3560.68 | 0.0356 |
| FL | Multi-Task (MTL) | fl-mtl-slms-tiny-bert-non-iid-stsb-qqp-sst2-3client-each | 0.7810 | 0.7782 | 0.6895 | 0.4733 | 0.0081 | 0.0063 | 6975.28 | 0.0881 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-qqp | - | - | 0.7010 | 0.3887 | - | - | 6226.63 | 0.0881 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-sst2 | 0.9060 | 0.9122 | - | - | - | - | 2514.06 | 0.0881 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-stsb | - | - | - | - | 0.2186 | 0.1755 | 1182.03 | 0.0881 |
