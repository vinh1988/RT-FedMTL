# MINI-BERT Comprehensive Metrics Summary

> [!NOTE]
> **Resource_Usage** priority: `Avg_GPU_Mem_GB` > `Avg_GPU_Usage`.

| Paradigm | Task | Experiment | SST-2 | | QQP | | STS-B | | Time | Res |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | **Acc** | **F1** | **Acc** | **F1** | **Pear** | **Spear** | **(s)** | **(GB)** |
| Centralized | Multi-Task (MTL) | centralized-mtl-all-tasks | 0.8589 | 0.8650 | 0.7770 | 0.8580 | 0.7989 | 0.7973 | 3652.50 | 2.4651 |
| Centralized | Single-Task | centralized-single-task-qqp | - | - | 0.7990 | 0.8536 | - | - | 318.6076 | 2.4487 |
| Centralized | Single-Task | centralized-single-task-sst2 | 0.8727 | 0.8773 | - | - | - | - | 3348.26 | 2.4487 |
| Centralized | Single-Task | centralized-single-task-stsb | - | - | - | - | 0.8316 | 0.8283 | 411.4504 | 2.4487 |
| FL | Multi-Task (MTL) | fl-mtl-slms-bertmini-stsb-qqp-sst2 | 0.9266 | 0.9329 | 0.8350 | 0.7981 | 0.6940 | 0.6550 | 13341.01 | 0.1969 |
| FL | Multi-Task (MTL) | fl-mtl-slms-bertmini-stsb-qqp-sst2-lora | 0.7649 | 0.7915 | 0.6958 | 0.5602 | 0.2824 | 0.2773 | 4191.67 | 0.0630 |
| FL | Multi-Task (MTL) | fl-mtl-slms-mini-bert-non-iid-stsb-qqp-sst2-3client-each | 0.8234 | 0.8226 | 0.7748 | 0.6668 | 0.4009 | 0.3939 | 10317.55 | 0.1969 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-qqp | - | - | 0.7674 | 0.6201 | - | - | 9642.59 | 0.1969 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-sst2 | 0.9151 | 0.9196 | - | - | - | - | 4552.75 | 0.1969 |
| FL | Single-Task | fl-slms-mini-lm-non-iid-stsb | - | - | - | - | 0.6504 | 0.6150 | 2547.10 | 0.1969 |
