# Multimodal Recommender System

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-2.2.1+-ee4c2c.svg" alt="PyTorch Version">
</div>
<br>

A PyTorch-based framework for building multimodal recommendation systems that integrate visual, textual, and numerical features to generate personalized recommendations.

Please, check the docs folder if you need more information.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Quickstart in 5 Minutes](#quickstart-in-5-minutes)
- [Installation](#installation)
- [Detailed Workflow](#detailed-workflow)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Configuration](#2-configuration)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Data Splitting](#4-data-splitting)
  - [5. Feature Caching](#5-feature-caching)
  - [6. Training](#6-training)
  - [7. Evaluation](#7-evaluation)
  - [8. Generating Recommendations](#8-generating-recommendations)
- [Advanced Management](#advanced-management)
  - [Cache Management](#cache-management)
  - [Checkpoint Management](#checkpoint-management)
- [Project Structure](#project-structure)

## Overview

This system implements a neural recommender that combines multiple data modalities to overcome the limitations of traditional collaborative filtering methods:

-   **Visual Features**: Extracted from item images using pre-trained models (e.g., ResNet, CLIP, DINO).
-   **Textual Features**: Processed from item descriptions using language models (e.g., Sentence-BERT, MPNet).
-   **Numerical Features**: Item metadata and interaction statistics.

The architecture uses attention mechanisms to fuse multimodal representations and supports contrastive learning for improved vision-text alignment.

## Key Features

-   **Flexible Architecture**: Configurable fusion of user/item embeddings and multimodal features.
-   **Pre-trained Models**: Support for a variety of vision and language backbones from Hugging Face.
-   **Modular Data Processing**: Robust preprocessing pipeline with automatic validation, cleaning, and compression.
-   **Data Splitting Strategies**: Support for stratified, temporal, and user/item-based splits.
-   **Efficient Training**: Configurable optimizers and schedulers, early stopping, and Weights & Biases integration.
-   **Comprehensive Evaluation**: Standard metrics (Precision, Recall, NDCG, MRR) and baseline comparisons (Popularity, ItemKNN).
-   **Performance Optimized**: Feature caching system with LRU memory management to accelerate training and inference.

## Quickstart in 5 Minutes

Follow these steps to get the system running with sample data.

```bash
# 1. Clone the repository
git clone [https://github.com/your_user/your_repo.git](https://github.com/your_user/your_repo.git)
cd your_repo

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Preprocess the sample data
# (This will clean, validate, and prepare the data in data/raw/)
python scripts/preprocess_data.py --config configs/simple_config.yaml

# 4. Create the train/validation/test splits
python scripts/create_splits.py --config configs/simple_config.yaml

# 5. Train the model
# (Use --device cpu if you don't have a CUDA-compatible GPU)
python scripts/train.py --config configs/simple_config.yaml --device cuda

# 6. Evaluate the trained model
python scripts/evaluate.py --config configs/simple_config.yaml --device cuda
````

## Installation

### Prerequisites

  - Python 3.11+
  - PyTorch 2.2.1+
  - Transformers 4.47.1+
  - A CUDA-capable GPU is recommended for fast training.

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your_user/your_repo.git](https://github.com/your_user/your_repo.git)
    cd your_repo
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Detailed Workflow

### 1\. Data Preparation

Organize your data in the `data/raw/` folder with the following structure:

  - `item_info.csv`: Item metadata. Must contain `item_id` and columns for textual and numerical features.
  - `interactions.csv`: User-item interactions. Requires `user_id` and `item_id` columns.
  - `images/`: A directory containing item images, named as `{item_id}.jpg`.

### 2\. Configuration

Edit the configuration files in `configs/` to adjust parameters.

  - **`simple_config.yaml`**: Contains essential parameters to get started. Ideal for initial experiments.
  - **`advanced_config.yaml`**: Offers granular control over all aspects of the model, training, and data.

For more details, see the [Configuration Guide](https://www.google.com/search?q=docs/configuration.md).

### 3\. Data Preprocessing

This script validates, cleans, and processes the raw data.

```bash
python scripts/preprocess_data.py --config configs/simple_config.yaml
```

### 4\. Data Splitting

Create standardized datasets for training, validation, and testing.

```bash
python scripts/create_splits.py --config configs/simple_config.yaml
```

### 5\. Feature Caching

(Optional but highly recommended) Pre-compute multimodal features to dramatically speed up training.

```bash
python scripts/precompute_cache.py --config configs/simple_config.yaml
```

### 6\. Training

Train the multimodal recommender.

```bash
python scripts/train.py --config configs/simple_config.yaml --device cuda
```

Or resume training (will automatically search in models/checkpoints/<vision_model>_<text_model>/).

```bash
python scripts/train.py --config configs/simple_config.yaml --resume last_checkpoint.pth
```

You can enable Weights & Biases tracking by adding the `--use_wandb` and `--wandb_project "MyProject"` flags.

### 7\. Evaluation

Evaluate the trained model on the test set.

```bash
python scripts/evaluate.py --config configs/simple_config.yaml --recommender_type multimodal --eval_task retrieval
```

The script also allows for evaluating baselines:

```bash
# Evaluate popularity baseline
python scripts/evaluate.py --config configs/simple_config.yaml --recommender_type popularity
```

**Example Evaluation Output:**

| Metric                | Value   |
| --------------------- | ------- |
| avg\_precision\_at\_k    | 0.1234  |
| avg\_recall\_at\_k       | 0.2345  |
| avg\_f1\_at\_k           | 0.1618  |
| avg\_hit\_rate\_at\_k     | 0.6789  |
| avg\_ndcg\_at\_k         | 0.4567  |
| avg\_mrr               | 0.3890  |

### 8\. Generating Recommendations

Generate a list of recommendations for specific users.

```bash
python scripts/generate_recommendations.py --config configs/simple_config.yaml --users user_123 user_456
```

## Advanced Management

### Cache Management

The `scripts/cache.py` script allows you to inspect and clear feature caches.

```bash
# List all available feature caches
python scripts/cache.py list

# View stats for a specific cache
python scripts/cache.py stats resnet_sentence-bert

# Clear the cache for a model combination
python scripts/cache.py clear resnet_sentence-bert
```

### Checkpoint Management

The `scripts/checkpoint_manager.py` script helps organize saved checkpoints.

```bash
# List all checkpoints and their organization status
python scripts/checkpoint_manager.py list

# Automatically organize checkpoints into model-specific directories
python scripts/checkpoint_manager.py organize
```

## Project Structure

```
multimodal-recommender/
├── configs/              # YAML configuration files
├── data/                 # Raw, processed, and split data
├── docs/                 # Additional documentation
├── models/               # Saved model checkpoints
├── results/              # Evaluation results and logs
├── scripts/              # Executable scripts (train, evaluate, etc.)
├── src/                  # Source code for the framework
│   ├── config.py         # Configuration management
│   ├── data/             # Data and preprocessing modules
│   ├── evaluation/       # Metrics and evaluation task modules
│   ├── inference/        # Logic for generating recommendations
│   ├── models/           # Model architectures
│   └── training/         # Training logic
├── tests/                # Unit and integration tests
└── requirements.txt      # Python dependencies
```
