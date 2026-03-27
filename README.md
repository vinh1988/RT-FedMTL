# RT-FedMTL: A Novel Real-Time Federated Multi-Task Learning Framework with Small Language Models for Natural Language Understanding Tasks
![Made With python](https://img.shields.io/badge/Made%20with-Python-brightgreen)![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)![Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-green.svg)


## Abstract
Natural language understanding is a crucial component in intelligent systems such as social media monitoring, conversational agents, and information retrieval. Deploying these systems in real application scenarios requires not only accurate but also efficient, privacy-preserving, and capable of handling multiple tasks simultaneously. Centralized training of large language models based approaches raises significant challenges in terms of data privacy, communication overhead, and inference latency. This paper proposes RT-FedMTL, a novel real-time federated multi-task learning framework with small language models for efficient and privacy-preserving natural language understanding. Particularly, our framework unifies federated learning and multi-task learning to enable collaborative training across distributed clients while preserving data locality and leveraging shared representations across tasks. We utilize WebSocket protocols and small language models tuning to reduce the feedback loop and provide near-instantaneous global updates. Extensive experiments conducted on five small language model architectures with respect to three standard natural language understanding tasks to get the best trade-off between performance and resource usage. A significant result demonstrates RT-FedMTL can drastically minimize per-client GPU memory requirements (up to 74-92%) versus centralized multi-task learning baselines, while keeping performance for multiple benchmarks.

## Results
The consolidated LaTeX results for all experiments can be found here: [Consolidated Results](https://github.com/vinh1988/RT-FedMTL/tree/main/experiment_new_solution/consolidated_results/latex)

## Citation
If you find this work useful in your research, please cite:

**Quang-Vinh Pham and Quang-Hung Le**, "RT-FedMTL: A Novel Real-Time Federated Multi-Task Learning Framework with Small Language Models for Natural Language Understanding Tasks," *Journal of Communications Software and Systems*, vol. XX, no. X, March 2026.

### BibTeX
```bibtex
@article{pham2026rtfedmtl,
  title={RT-FedMTL: A Novel Real-Time Federated Multi-Task Learning Framework with Small Language Models for Natural Language Understanding Tasks},
  author={Pham, Quang-Vinh and Le, Quang-Hung},
  year={2026},
  month={March}
}
```
