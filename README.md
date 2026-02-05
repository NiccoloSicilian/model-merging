# Model Merging Utilities

This repository provides utilities and methods for weight-space model merging — a process that allows you to combine multiple pretrained models into a single model by operating directly on their weights. This can be useful for multi-task learning, model compression, or combining specialized models into a single generalist model.

The project is designed to be modular and extensible, allowing you to implement new merging methods and easily evaluate merged models on multiple tasks.

## 🚀 Installation

You can install the project using `uv`:
```bash
uv sync
```

This will install all dependencies and prepare the environment for running the scripts and notebooks.

## 📂 Project Structure
```
/
├── src/                     
│   ├── model_merging/       # Core package containing merging logic
│   │   ├── __init__.py
│   │   ├── main.py          # Main entry point for model merging operations
│   │   ├── merger/          # Directory for implementing merging strategies
│   │   └── utils.py         # Helper functions and utilities
├── conf/                    # Configuration files for different tasks and mergers
│   ├── multitask.yaml       # Multi-task merging configuration
│   └── merger/              # Configuration for individual merging methods
├── scripts/                 # Evaluation and utility scripts
│   └── evaluate_multi_task_merging.py
├── notebooks/               # Example notebooks to reproduce experiments
├── pyproject.toml           # Project configuration
├── README.md                # This file
└── LICENSE                  # License information
```

## 🔧 Core Features

### 1. Multi-Task Merging

The repository supports merging multiple models into a single multi-task model. You can configure which models to merge and which tasks to evaluate using the `conf/multitask.yaml` file.

To run a multi-task evaluation:
```bash
uv run scripts/evaluate_multi_task_merging.py
```

This script merges the models according to the specified strategy and evaluates performance on the defined tasks.

### 2. Extensible Merging Methods

You can define new weight-space merging strategies by:

1. Creating a new class in `src/model_merging/merger/`.
2. Adding a corresponding configuration in `conf/merger/`.
3. Updating the `merger` field in your `multitask.yaml` configuration to use your new method.

This modular design makes it easy to experiment with new algorithms for combining models.

### 3. Reproducibility

Results can also be reproduced and explored interactively using the notebooks provided in the `notebooks/` directory.

## 📜 License

This project is licensed under the MIT License. See LICENSE for more details.
