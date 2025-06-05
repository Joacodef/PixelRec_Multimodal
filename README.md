# Multimodal Recommender System

This repository provides a comprehensive framework for building and experimenting with multimodal recommender systems. It leverages visual, textual, and numerical information from datasets to provide personalized recommendations. The system is highly configurable, allowing for easy modification of model architectures, data processing pipelines, and evaluation strategies.

---

## Core Features

* **Multimodal Data Integration**: Processes items with associated images, textual descriptions, and numerical features for comprehensive recommendation modeling.

* **Flexible Model Architecture**:
    * Main model class: `MultimodalRecommender`.
    * Supports various pre-trained vision models (CLIP, DINO, ResNet, ConvNeXT) and language models (Sentence-BERT, MPNet, BERT, RoBERTa).
    * Configurable fusion mechanisms with multi-head self-attention for combining diverse feature sets.
    * Optional contrastive learning for better vision-text alignment (primarily with CLIP-based vision models).
    * Fully configurable network architectures including fusion layers, activation functions, batch normalization, and initialization methods (via `advanced_config.yaml`).

* **Advanced Data Preprocessing & Handling**:
    * **Data Preprocessing Pipeline**: `scripts/preprocess_data.py` handles text cleaning, image validation, optional image compression/resizing, and numerical feature scaling.
    * **Flexible Data Splitting**: `scripts/create_splits.py` generates train/validation/test splits using a stratified strategy with activity filtering and optional dataset sampling (using the `--sample_n` argument).
    * **Efficient Feature Caching**: `SimpleFeatureCache` system for item features (images, text, numerical) with configurable memory and optional disk persistence to accelerate training and inference.
    * **Comprehensive Data Loading**: `MultimodalDataset` class handles negative sampling, numerical feature normalization, optional text augmentation, and robust data validation.

* **Robust Training Framework**:
    * Configurable optimizers (AdamW, Adam, SGD) with full parameter control.
    * Advanced learning rate scheduling (ReduceLROnPlateau, Cosine, Step).
    * Gradient clipping, early stopping, and checkpoint management.
    * Optional Weights & Biases integration for experiment tracking.
    * Support for training resumption and model state persistence.

* **Comprehensive Evaluation Framework**:
    * **Task-Based Evaluation**: Simplified evaluation tasks focusing on `Top-K Retrieval` and `Top-K Ranking`.
    * **Standard Metrics**: Precision@k, Recall@k, F1@k, Hit Rate@k, NDCG@k, MRR.
    * **Baseline Comparisons**: Random, Popularity, ItemKNN, and UserKNN recommenders for benchmarking.

* **Inference and Recommendation Generation**:
    * Efficient batch processing for large-scale recommendation generation.
    * Pre-computed item feature caching for fast inference via `SimpleFeatureCache`.
    * Support for candidate filtering and seen-item exclusion.

* **Modular Architecture**:
    * Clear separation of concerns across configuration, data handling, models, training, evaluation, and inference.
    * Extensive YAML-based configuration system (`simple_config.yaml` for essentials, `advanced_config.yaml` for full control) with nested dataclasses.
    * Consistent interfaces for easy component swapping and experimentation.

---

## Directory Structure

```
PixelRec_Multimodal/
├── configs/                    # Configuration files
│   ├── simple_config.yaml      # Essential settings
│   └── advanced_config.yaml    # All configurable options
├── data/                       # Data directories (paths configurable)
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data and scalers
│   ├── splits/                 # Train/val/test splits
│   └── cache/                  # Feature cache (if disk persistence is enabled)
├── models/                     # Model checkpoints and encoders
│   └── checkpoints/            # Training checkpoints
├── results/                    # Evaluation results and figures
├── scripts/                    # Execution scripts
│   ├── preprocess_data.py      # Data preprocessing
│   ├── create_splits.py        # Data splitting
│   ├── extract_encoders.py     # Encoder extraction utility
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
│   └── generate_recommendations.py # Recommendation generation
├── src/                        # Source code
│   ├── config.py               # Configuration dataclasses
│   ├── data/                   # Data processing modules
│   │   ├── dataset.py          # MultimodalDataset class
│   │   ├── simple_cache.py     # Simple feature caching system
│   │   ├── splitting.py        # Data splitting strategies
│   │   ├── preprocessing.py    # Preprocessing utilities
│   │   └── processors/         # Modular data processors
│   ├── evaluation/             # Evaluation framework
│   │   ├── tasks.py            # Task-based evaluators
│   │   ├── metrics.py          # Standard metrics
│   │   ├── novelty.py          # Novelty and diversity metrics
│   │   └── advanced_metrics.py # Additional metrics (MRR, Hit Rate etc.)
│   ├── inference/              # Inference and recommendation
│   │   ├── recommender.py      # Main recommender class
│   │   └── baseline_recommenders.py # Baseline implementations
│   ├── models/                 # Model architectures
│   │   ├── multimodal.py       # Main model class (MultimodalRecommender)
│   │   ├── layers.py           # Custom layers (e.g., CrossModalAttention, potentially used internally)
│   │   └── losses.py           # Loss functions
│   └── training/               # Training framework
│       └── trainer.py          # Trainer class
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

---

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The system is operated through command-line scripts that can be configured via YAML files. All scripts support the `--config` parameter. Refer to `docs/configuration.md` for details on using `simple_config.yaml` (recommended for most cases) and `advanced_config.yaml` (for deep experimentation).

### 1. Configuration

Start by configuring `configs/simple_config.yaml` or `configs/advanced_config.yaml`. Key sections include:

* **`model`**: Architecture selection, embedding dimensions.
* **`training`**: Batch size, learning rate, optimizer settings.
* **`data`**: File paths, preprocessing options, caching settings, splitting parameters.
* **`recommendation`**: Top-k values, filtering options.

### 2. Data Preprocessing

Process raw data including text cleaning, image validation, numerical scaling, and optional compression:

```bash
python scripts/preprocess_data.py --config configs/simple_config.yaml
```

### 3. Data Splitting

Create standardized train/validation/test splits:

```bash
python scripts/create_splits.py --config configs/simple_config.yaml
```

Advanced options:
```bash
# Sample dataset before splitting
python scripts/create_splits.py --config configs/simple_config.yaml --sample_n 100000
```
This script uses a stratified splitting strategy by default.

### 4. Model Training

Train the multimodal recommender:

```bash
python scripts/train.py --config configs/simple_config.yaml --device cuda
```

Training features:
```bash
# With experiment tracking
python scripts/train.py --config configs/simple_config.yaml \
    --use_wandb --wandb_project MyProject --wandb_entity username

# Resume from checkpoint
python scripts/train.py --config configs/simple_config.yaml \
    --resume models/checkpoints/best_model.pth
```

### 5. Model Evaluation

Evaluate trained models using task-specific metrics:

```bash
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/split_tiny/test.csv \
    --train_data data/splits/split_tiny/train.csv \
    --eval_task retrieval \
    --recommender_type multimodal \
    --output results/multimodal_retrieval_metrics.json
```

Evaluation tasks available:
* **`retrieval`**: Top-K retrieval of novel items.
* **`ranking`**: Top-K ranking quality (evaluates ranking of known relevant items).

Baseline comparison:
```bash
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/split_tiny/test.csv \
    --train_data data/splits/split_tiny/train.csv \
    --eval_task retrieval \
    --recommender_type popularity \
    --output results/popularity_retrieval_metrics.json
```

### 6. Recommendation Generation

Generate recommendations for users:

```bash
python scripts/generate_recommendations.py --config configs/simple_config.yaml \
    --users user_A user_B \
    --output results/user_recommendations.json
```

Generation modes:
* Specific users: `--users user_1 user_2`
* User file: `--user_file path/to/users.txt`
* Random sample: `--sample_users N`

---

## Configuration Reference

Refer to `docs/configuration.md` and the `configs/simple_config.yaml` / `configs/advanced_config.yaml` files for detailed parameter explanations.

### Key Configuration Areas:

* **Model (`model`)**:
    * `vision_model`, `language_model`: Select pre-trained backbones.
    * `embedding_dim`: Size of latent embeddings.
    * `use_contrastive`: Enable/disable contrastive loss.
    * (Advanced) `fusion_hidden_dims`, `num_attention_heads`, `dropout_rate`, etc.

* **Data (`data`)**:
    * Paths to raw, processed, and split data.
    * `cache_config`: Configure `SimpleFeatureCache` (enabled, max_memory_items, cache_directory, use_disk).
    * `numerical_features_cols`, `numerical_normalization_method`.
    * (Advanced) `text_augmentation`, `offline_image_compression`, `offline_text_cleaning`, `splitting` parameters.

* **Training (`training`)**:
    * `batch_size`, `epochs`, `learning_rate`.
    * (Advanced) `optimizer_type`, `weight_decay`, `lr_scheduler_type`, `loss_weights`.

---

## Key Features in Detail

### Simple Feature Caching System

The `SimpleFeatureCache` (`src/data/simple_cache.py`) provides an efficient way to cache processed item features (image tensors, tokenized text, numerical features).
* **Configuration**: Enabled via `data.cache_config` in YAML.
    * `enabled`: Turn caching on/off.
    * `max_memory_items`: Max items in memory (LRU eviction).
    * `cache_directory`: Path for disk persistence.
    * `use_disk`: Enable saving/loading cache to/from disk.
* Used by `MultimodalDataset` during `__getitem__` to speed up data loading after the first epoch.
* The `Recommender` class for inference also uses a simple in-memory cache.

### Simplified Evaluation Tasks

The evaluation framework (`src/evaluation/tasks.py`) focuses on:

* **Top-K Retrieval**: Measures how well the model retrieves relevant (novel) items from a candidate set. Can use negative sampling for efficiency.
* **Top-K Ranking**: Focuses on the ranking quality of a known set of relevant items.

### Baseline Recommenders

Available in `src/inference/baseline_recommenders.py` for comparison:
* Random, Popularity, ItemKNN, UserKNN.

---

## Performance Optimization

### Training Performance
* **Feature Caching**: `SimpleFeatureCache` caches item features after first processing for faster subsequent epoch loading.
* **Efficient Data Loading**: Multi-worker data loaders with prefetching.
* **Gradient Clipping**: Prevents training instability.

### Inference Performance
* **Batch Processing**: `Recommender` class supports batch scoring.
* **Feature Caching**: Inference `Recommender` uses an internal in-memory cache for item features.

---

## Extensibility

The modular design supports easy extension:

### Adding New Models
1.  Modify `src/models/multimodal.py` or create a new model class.
2.  Update `src/config.py` if new model-specific configurations are needed.
3.  Adjust model instantiation in `scripts/train.py`, `scripts/evaluate.py`, and `scripts/generate_recommendations.py`.

### Adding New Evaluation Metrics/Tasks
1.  For new metrics, add functions to `src/evaluation/metrics.py` or `src/evaluation/advanced_metrics.py`.
2.  For new tasks, implement a new evaluator class in `src/evaluation/tasks.py` inheriting from `BaseEvaluator`, add to `EvaluationTask` enum, and update the factory function `create_evaluator`.

---

## Examples

### Quick Start
```bash
# 1. Preprocess data (text, images, numericals)
python scripts/preprocess_data.py --config configs/simple_config.yaml

# 2. Create train/val/test splits
python scripts/create_splits.py --config configs/simple_config.yaml

# 3. Train model
python scripts/train.py --config configs/simple_config.yaml

# 4. Evaluate
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/split_tiny/test.csv \
    --train_data data/splits/split_tiny/train.csv \
    --recommender_type multimodal --eval_task retrieval
```

This framework provides a streamlined and configurable solution for multimodal recommendation research and deployment.
