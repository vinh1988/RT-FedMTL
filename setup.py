#!/usr/bin/env python3
"""
Setup script for FedAvgLS: Federated Learning with DistilBART and MobileBART
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fedavgls",
    version="1.0.0",
    author="FedAvgLS Team",
    author_email="fedavgls@example.com",
    description="Federated Learning with DistilBART and MobileBART for Cross-Architecture Knowledge Transfer",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/FedAvgLS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "fedavgls-train=fedavgls.examples.train_fedmkt_20news:main",
            "fedavgls-eval=fedavgls.examples.evaluate_models:main",
            "fedavgls-demo=fedavgls.demo_federated_training:demo_federated_training",
        ],
    },
    include_package_data=True,
    package_data={
        "fedavgls": [
            "config/*.yaml",
            "config/*.yml",
            "*.md",
            "*.txt",
        ],
    },
    keywords=[
        "federated-learning",
        "distilbart",
        "mobilebart",
        "knowledge-transfer",
        "nlp",
        "text-classification",
        "privacy-preserving",
        "cross-architecture",
        "bart",
        "transformer"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-repo/FedAvgLS/issues",
        "Source": "https://github.com/your-repo/FedAvgLS",
        "Documentation": "https://github.com/your-repo/FedAvgLS#readme",
        "Research Paper": "https://arxiv.org/abs/your-paper-id",
    },
)
