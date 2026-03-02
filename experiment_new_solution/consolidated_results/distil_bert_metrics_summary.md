# DISTIL-BERT Comprehensive Metrics Summary

> [!NOTE]
> **Resource_Usage** priority: `Avg_GPU_Mem_GB` > `Avg_GPU_Usage`.

| Paradigm | Task | Experiment | SST-2 | | QQP | | STS-B | | Time | Res |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | **Acc** | **F1** | **Acc** | **F1** | **Pear** | **Spear** | **(s)** | **(GB)** |
| Centralized | Multi-Task (MTL) | centralized-mtl-all-tasks | 0.9037 | 0.9077 | 0.8137 | 0.8778 | 0.8635 | 0.8614 | 24449.68 | 4.3214 |
| Centralized | Single-Task | centralized-single-task-qqp | - | - | 0.8456 | 0.8934 | - | - | 4290.59 | 4.2547 |
| Centralized | Single-Task | centralized-single-task-sst2 | 0.9002 | 0.9061 | - | - | - | - | 18514.82 | 4.2460 |
| Centralized | Single-Task | centralized-single-task-stsb | - | - | - | - | 0.8712 | 0.8691 | 4561.78 | 4.2470 |
| FL | Multi-Task (MTL) | fl-mtl-slms-disti-bert-stsb-qqp-sst2 | 0.9461 | 0.9514 | 0.8913 | 0.8610 | 0.8284 | 0.8136 | 56454.46 | 1.0896 |
| FL | Multi-Task (MTL) | fl-mtl-slms-disti-bert-stsb-qqp-sst2-lora | 0.8257 | 0.8433 | 0.7535 | 0.6754 | 0.5108 | 0.4919 | 21245.26 | 0.2852 |
| FL | Multi-Task (MTL) | fl-mtl-slms-disti-bert-non-iid-stsb-qqp-sst2-3client-each | 0.9232 | 0.9298 | 0.8579 | 0.8054 | 0.6963 | 0.6967 | 52903.97 | 1.0968 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-qqp | - | - | 0.8408 | 0.7724 | - | - | 34904.44 | 1.0956 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-sst2 | 0.9346 | 0.9392 | - | - | - | - | 23586.44 | 1.0972 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-stsb | - | - | - | - | 0.7753 | 0.7421 | 15452.52 | 1.0988 |
