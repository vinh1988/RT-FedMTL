from setuptools import setup, find_packages

setup(
    name="fedbert-lora",
    version="0.1.0",
    description="Heterogeneous Federated Learning with BERT-base Server and TinyBERT Clients using LoRA",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "flwr>=1.5.0",
        "peft>=0.4.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
