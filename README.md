# Multimodal Recommender System

<div align="center">

    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">

    <img src="https://img.shields.io/badge/PyTorch-2.2.1+-ee4c2c.svg" alt="PyTorch Version">

    <img src="https://img.shields.io/badge/Transformers-4.47.1+-orange.svg" alt="Transformers Version">

</div>

<br>


A PyTorch-based framework for building and evaluating multimodal recommendation systems that integrate visual, textual, and numerical features to generate personalized recommendations. This repository is specially designed to work with the [PixelRec dataset](https://github.com/westlake-repl/PixelRec) and provides a comprehensive suite of tools for training, evaluation, and hyperparameter optimization.

-----

## Contents

  - [Problem Statement](https://www.google.com/search?q=%23problem-statement)
  - [Motivation](https://www.google.com/search?q=%23motivation)
  - [Methodology](https://www.google.com/search?q=%23methodology)
  - [Results and Evaluation](https://www.google.com/search?q=%23results-and-evaluation)
  - [Key Features](https://www.google.com/search?q=%23key-features)
  - [Quickstart in 5 Minutes](https://www.google.com/search?q=%23quickstart-in-5-minutes)
  - [Installation](https://www.google.com/search?q=%23installation)
  - [Detailed Workflow](https://www.google.com/search?q=%23detailed-workflow)
      - [1. Data Preparation](https://www.google.com/search?q=%231-data-preparation)
      - [2. Configuration](https://www.google.com/search?q=%232-configuration)
      - [3. Data Preprocessing](https://www.google.com/search?q=%233-data-preprocessing)
      - [4. Data Splitting](https://www.google.com/search?q=%234-data-splitting)
      - [5. Feature Caching](https://www.google.com/search?q=%235-feature-caching)
      - [6. Training](https://www.google.com/search?q=%236-training)
      - [7. Hyperparameter Optimization](https://www.google.com/search?q=%237-hyperparameter-optimization)
      - [8. Evaluation](https://www.google.com/search?q=%238-evaluation)
      - [9. Generating Recommendations](https://www.google.com/search?q=%239-generating-recommendations)
  - [Advanced Management](https://www.google.com/search?q=%23advanced-management)
      - [Checkpoint Management](https://www.google.com/search?q=%23checkpoint-management)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Troubleshooting](https://www.google.com/search?q=%23troubleshooting)

*For detailed documentation, please refer to the `docs` folder.*

-----

## Problem Statement

The primary challenge addressed by this project is the limitation of traditional recommender systems, which often rely on a single source of information (e.g., user-item interaction matrices). Such systems struggle with data sparsity and the "cold-start" problem, where new users or items have insufficient interaction data to make accurate predictions. This project aims to overcome these limitations by developing a flexible and robust multimodal recommender system that can effectively leverage rich side information from various data types—specifically visual, textual, and numerical features—to improve recommendation quality and personalization.

## Motivation

In many modern e-commerce, social media, and content platforms, items are described by more than just their ID. They have images, titles, descriptions, and other metadata. This rich, multimodal information is often underutilized by conventional recommendation algorithms like collaborative filtering. The motivation for this project is to harness this auxiliary data to:

  - **Enhance Recommendation Accuracy**: By understanding the underlying content of items, the model can make more informed recommendations, even for items with sparse interaction data.
  - **Mitigate the Cold-Start Problem**: When a new item is introduced, its visual and textual features can be used to recommend it to relevant users immediately, without needing any interaction history.
  - **Improve Recommendation Diversity and Novelty**: By analyzing item content, the system can recommend a wider variety of items, moving beyond simple popularity-based suggestions and introducing users to more novel content.
  - **Provide a Flexible Research Framework**: To create a modular and configurable codebase that allows researchers and practitioners to easily experiment with different pre-trained models, fusion techniques, and training strategies.

## Methodology

This system implements a neural recommender that combines multiple data modalities to generate a score for a given user-item pair. The core architecture is designed to be modular and is composed of three main stages: Feature Extraction, Feature Fusion, and Prediction.

1.  **Feature Extraction**:

      - **ID Embeddings**: Users, items, and categorical tags are mapped to dense embedding vectors that are learned during training.
      - **Visual Features**: Pre-trained vision models from the Hugging Face library (e.g., **ResNet, CLIP, DINO, ConvNext**) are used to extract feature vectors from item images.
      - **Textual Features**: Pre-trained language models (e.g., **Sentence-BERT, MPNet, BERT, RoBERTa**) are used to encode item titles and descriptions into semantic embeddings.
      - **Numerical Features**: Metadata such as price or interaction statistics are processed through a linear projection layer.

2.  **Feature Fusion**:
    The embeddings from all active modalities are projected into a common latent space. These projected features are then fused to create a single, unified representation of the user-item interaction. The framework supports several fusion strategies, which can be configured in the `.yaml` file:

      - **Concatenation**: The simplest approach, where all feature vectors are concatenated.
      - **Gated Fusion**: A mechanism where a neural network learns to assign weights (gates) to each modality, controlling their contribution to the final representation.
      - **Attention Fusion**: A self-attention mechanism is applied to the sequence of modal features, allowing the model to dynamically weigh the importance of each modality in the context of others.

3.  **Prediction**:

      - The final fused vector is passed through a multi-layer perceptron (MLP) that outputs a single score, which is then passed through a sigmoid function to predict the probability of a positive interaction.
      - **Contrastive Learning**: The model can be optionally trained with an auxiliary contrastive loss (inspired by CLIP) to align the vision and text embedding spaces, encouraging the model to learn more meaningful cross-modal representations.

The entire process, from data loading to model training and evaluation, is managed through configurable YAML files, ensuring full reproducibility.

## Results and Evaluation

The performance of the recommender models is assessed using a comprehensive evaluation framework implemented in the `src/evaluation` module. The evaluation script (`scripts/evaluate.py`) supports multiple tasks and a wide range of metrics.

  - **Evaluation Tasks**:

      - **Retrieval**: Measures the model's ability to retrieve relevant items from a larger set of candidates (positives + sampled negatives).
      - **Ranking**: Evaluates how well the model ranks a given set of known relevant items.

  - **Accuracy Metrics**:

      - **Precision@K, Recall@K, F1@K**: Standard metrics for evaluating the relevance of the top-K recommended items.
      - **NDCG@K (Normalized Discounted Cumulative Gain)**: A measure of ranking quality that rewards placing relevant items higher in the list.
      - **MRR (Mean Reciprocal Rank)**: The average of the reciprocal ranks of the first relevant item for each user.

  - **Beyond-Accuracy Metrics**:

      - **Novelty & Coverage**: Measures the ability of the model to recommend non-obvious (long-tail) items and the percentage of the item catalog it can recommend.
      - **Diversity (Intra-List Similarity)**: Quantifies how different the items are within a single recommendation list, based on the cosine similarity of their embeddings.

  - **Baselines for Comparison**:
    The framework includes standard baseline models (`popularity`, `random`, `item_knn`) to provide a reference point for the performance of the multimodal model.

Results from an evaluation run are saved to a JSON file in the `results/` directory, containing the computed metrics and other metadata about the run.

## Key Features

  - **Flexible Architecture**: Configurable fusion of user/item embeddings and multimodal features.
  - **Pre-trained Models**: Support for various vision and language backbones from Hugging Face.
  - **Hyperparameter Optimization**: Integrated Optuna support for automated hyperparameter search.
  - **Modular Data Processing**: Robust preprocessing pipeline with automatic validation, cleaning, and compression.
  - **Data Splitting Strategies**: Support for stratified, temporal, and user/item-based splits.
  - **Efficient Training**: Advanced optimizers, learning rate schedulers, early stopping, and Weights & Biases integration.
  - **Comprehensive Evaluation**: Standard and beyond-accuracy metrics (Novelty, Coverage, Diversity) and baseline comparisons.
  - **Performance Optimized**: Feature caching system with LRU memory management.
  - **Production Ready**: Checkpoint management, model versioning, and inference utilities.

## Quickstart in 5 Minutes

Follow these steps to get the system running with sample data:

```bash
# 1. Clone the repository
git clone https://github.com/joacodef/pixelrec_multimodal.git
cd pixelrec_multimodal

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download PixelRec data
# Choose a sample from https://github.com/westlake-repl/PixelRec
# Organize data as follows in data/raw/:
# - images/                            # All images in original format
# - interactions/interactions.csv      # Columns: item_id, user_id, timestamp
# - item_info/item_info.csv           # Item metadata

# 4. Preprocess the data
python scripts/preprocess_data.py --config configs/simple_config_example.yaml

# 5. Create train/validation/test splits
python scripts/create_splits.py --config configs/simple_config_example.yaml

# 6. Train the model
python scripts/train.py --config configs/simple_config_example.yaml --device cuda

# 7. Evaluate the trained model
python scripts/evaluate.py --config configs/simple_config_example.yaml --device cuda
```

## Installation

### Prerequisites

  - Python 3.11+
  - PyTorch 2.2.1+
  - Transformers 4.47.1+
  - CUDA-capable GPU (recommended for training)
  - 16GB+ RAM
  - 50GB+ disk space (for datasets and caches)

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/joacodef/pixelrec_multimodal.git
    cd pixelrec_multimodal
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Optuna (optional, for hyperparameter optimization):**

    ```bash
    pip install optuna optuna-dashboard
    ```

## Detailed Workflow

### 1\. Data Preparation

Organize your data in the `data/raw/` folder with the following structure:

```
data/raw/
├── item_info/
│   └── item_info.csv         # Item metadata (item_id, title, description, numerical features)
├── interactions/
│   └── interactions.csv      # User-item interactions (user_id, item_id, timestamp)
└── images/
    ├── item_001.jpg         # Images named by item_id
    ├── item_002.jpg
    └── ...
```

### 2\. Configuration

The system uses YAML configuration files. Two templates are provided:

  - **`configs/simple_config.yaml`**: Essential parameters for quick start
  - **`configs/advanced_config.yaml`**: Full control over all system components

Key configuration sections:

```yaml
model:
  vision_model: resnet          # Options: clip, resnet, dino, convnext
  language_model: sentence-bert # Options: sentence-bert, mpnet, bert, roberta
  embedding_dim: 128
  use_contrastive: true
  dropout_rate: 0.3
  fusion_hidden_dims: [512, 256, 128]

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 30
  patience: 5
  optimizer_type: adam          # Options: adam, adamw, sgd
  use_lr_scheduler: true
```

For detailed configuration options, see [docs/configuration.md](https://www.google.com/search?q=docs/configuration.md).

### 3\. Data Preprocessing

Validates, cleans, and processes raw data:

```bash
python scripts/preprocess_data.py --config configs/simple_config_example.yaml

# Force reprocessing of all data
python scripts/preprocess_data.py --config configs/simple_config_example.yaml --force-reprocess

# Skip automatic feature caching
python scripts/preprocess_data.py --config configs/simple_config_example.yaml --skip-caching
```

This script:

  - Validates and cleans text data
  - Compresses and resizes images based on configuration
  - Processes numerical features
  - Filters out invalid or incomplete records
  - Generates processed files in `data/processed/`

### 4\. Data Splitting

Create reproducible train/validation/test splits:

```bash
python scripts/create_splits.py --config configs/simple_config_example.yaml

# Create a smaller sample for testing
python scripts/create_splits.py --config configs/simple_config_example.yaml --sample_n 10000
```

Split strategies (configured in YAML):

  - **Random**: Stratified random splitting
  - **Temporal**: Time-based splitting
  - **User-based**: Leave-one-user-out
  - **Item-based**: Cold-start evaluation

### 5\. Feature Caching

Pre-compute features for faster training (highly recommended):

```bash
python scripts/precompute_cache.py --config configs/simple_config_example.yaml

# Force recomputation of all features
python scripts/precompute_cache.py --config configs/simple_config_example.yaml --force_recompute

# Process only first 1000 items (for testing)
python scripts/precompute_cache.py --config configs/simple_config_example.yaml --max_items 1000
```

### 6\. Training

Train the multimodal recommender:

```bash
# Basic training
python scripts/train.py --config configs/simple_config_example.yaml --device cuda

# Resume from checkpoint
python scripts/train.py --config configs/simple_config_example.yaml --resume last_model.pth

# With Weights & Biases logging
python scripts/train.py --config configs/simple_config_example.yaml --use_wandb --wandb_project "MyProject"

# With custom run name
python scripts/train.py --config configs/simple_config_example.yaml --use_wandb --wandb_run_name "experiment_v2"
```

Training features:

  - Automatic mixed precision (AMP) support
  - Gradient accumulation for large batch sizes
  - Early stopping with patience
  - Learning rate scheduling
  - Checkpoint saving (best and last models)

### 7\. Hyperparameter Optimization

Use Optuna for automated hyperparameter search:

```bash
# Basic hyperparameter search
python scripts/hyperparameter_search.py \
    --config configs/simple_config_example.yaml \
    --n_trials 50 \
    --optimize_metric val_loss

# Advanced search with storage and pruning
python scripts/hyperparameter_search.py \
    --config configs/simple_config_example.yaml \
    --n_trials 100 \
    --study_name "full_search" \
    --storage "sqlite:///optuna_study.db" \
    --optimize_metric val_f1_score \
    --direction maximize \
    --pruning \
    --use_wandb

# Resume interrupted search
python scripts/hyperparameter_search.py \
    --config configs/simple_config_example.yaml \
    --n_trials 50 \
    --study_name "full_search" \
    --storage "sqlite:///optuna_study.db" \
    --resume

# Parallel search across multiple GPUs/machines
python scripts/hyperparameter_search.py \
    --config configs/simple_config_example.yaml \
    --n_trials 200 \
    --storage "sqlite:///optuna_study.db" \
    --parallel
```

The hyperparameter search optimizes:

  - Learning rate, batch size, weight decay
  - Model architecture (embedding dimensions, hidden layers, dropout rates)
  - Activation functions and normalization strategies
  - Loss function weights
  - Optimizer settings
  - Contrastive learning parameters

Results are saved in:

  - `optuna_results/best_params.json`: Best hyperparameters found
  - `optuna_results/best_config.yaml`: Ready-to-use configuration file
  - `optuna_results/optimization_history.html`: Visualization of the search

### 8\. Evaluation

Evaluate trained models or baselines:

```bash
# Evaluate the trained multimodal model
python scripts/evaluate.py --config configs/simple_config_example.yaml \
    --recommender_type multimodal \
    --eval_task retrieval

# Evaluate with ranking task
python scripts/evaluate.py --config configs/simple_config_example.yaml \
    --recommender_type multimodal \
    --eval_task ranking

# Evaluate popularity baseline
python scripts/evaluate.py --config configs/simple_config_example.yaml \
    --recommender_type popularity

# Evaluate ItemKNN baseline
python scripts/evaluate.py --config configs/simple_config_example.yaml \
    --recommender_type itemknn

# Save predictions for analysis
python scripts/evaluate.py --config configs/simple_config_example.yaml \
    --save_predictions results/predictions.json
```

Metrics computed:

  - Precision@K, Recall@K, F1@K
  - Hit Rate@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MRR (Mean Reciprocal Rank)

### 9\. Generating Recommendations

Generate personalized recommendations:

```bash
# For specific users
python scripts/generate_recommendations.py \
    --config configs/simple_config_example.yaml \
    --users user_123 user_456

# From a file of user IDs
python scripts/generate_recommendations.py \
    --config configs/simple_config_example.yaml \
    --user_file user_list.txt

# For random sample of users
python scripts/generate_recommendations.py \
    --config configs/simple_config_example.yaml \
    --sample_users 10 \
    --output sampled_recommendations.json

# With diversity-aware algorithm (if implemented)
python scripts/generate_recommendations.py \
    --config configs/simple_config_example.yaml \
    --users user_789 \
    --use_diversity
```

## Advanced Management

### Checkpoint Management

Organize and manage model checkpoints:

```bash
# List all checkpoints
python scripts/checkpoint_manager.py list

# Organize checkpoints into model-specific directories
python scripts/checkpoint_manager.py organize

# Clean up old checkpoints (keep only best and last)
python scripts/checkpoint_manager.py cleanup

# Archive checkpoints for a specific experiment
python scripts/checkpoint_manager.py archive --experiment_name "baseline_v1"
```

## Project Structure

```
multimodal-recommender/
├── configs/                 # YAML configuration files
│   ├── simple_config.yaml  # Quick start configuration
│   └── advanced_config.yaml # Full configuration options
├── data/                    # Data directory
│   ├── raw/                 # Original unprocessed data
│   ├── processed/           # Cleaned and processed data
│   ├── splits/              # Train/val/test splits
│   └── cache/               # Precomputed features
├── docs/                    # Documentation
│   ├── configuration.md     # Configuration guide
│   ├── commands.md          # Command reference
│   └── architecture.md      # Model architecture details
├── models/                  # Saved models and checkpoints
│   ├── checkpoints/         # Training checkpoints
│   └── encoders/            # User/item encoders
├── results/                 # Evaluation results and logs
├── optuna_results/          # Hyperparameter search results
├── scripts/                 # Executable scripts
│   ├── preprocess_data.py
│   ├── create_splits.py
│   ├── train.py
│   ├── hyperparameter_search.py
│   ├── evaluate.py
│   └── ...
├── src/                     # Source code
│   ├── config.py            # Configuration management
│   ├── data/                # Data loading and processing
│   ├── models/              # Model architectures
│   ├── training/            # Training logic
│   ├── evaluation/          # Evaluation metrics
│   └── inference/           # Recommendation generation
├── tests/                   # Unit and integration tests
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Troubleshooting

### Common Issues

1.  **CUDA Out of Memory**

      - Reduce batch size in configuration
      - Enable gradient accumulation
      - Use mixed precision training
      - Reduce model dimensions

2.  **Slow Training**

      - Use feature caching: `python scripts/precompute_cache.py`
      - Increase number of data loader workers
      - Use smaller image sizes in preprocessing
      - Enable mixed precision training

3.  **Poor Model Performance**

      - Run hyperparameter optimization
      - Increase model capacity (embedding dimensions, hidden layers)
      - Adjust loss weights (contrastive vs BCE)
      - Try different pre-trained backbones
      - Ensure sufficient training data

4.  **Data Processing Errors**

      - Check data format matches expected structure
      - Verify all item IDs in interactions have corresponding images
      - Ensure text fields are not empty
      - Validate numerical features are properly formatted

For more detailed troubleshooting, consult the documentation in the `docs/` folder or open an issue on GitHub.
