# Multimodal Recommender System

This repository contains a framework for building and experimenting with multimodal recommender systems. It uses visual, textual, and numerical data to generate recommendations. The system is designed with a focus on configurability for model architecture, data processing, and evaluation.

---

## Features

* **Multimodal Data Integration**: Processes items with images, text, and numerical features.
* **Flexible Model Architecture**:
    * Core model: `MultimodalRecommender` (formerly `PretrainedMultimodalRecommender`).
    * Supports various pre-trained vision (e.g., CLIP, ResNet, DINO, ConvNeXT) and language models (e.g., Sentence-BERT, MPNet, BERT, RoBERTa).
    * Configurable fusion methods, including multi-head self-attention.
    * Optional contrastive learning for vision-text alignment (primarily designed for CLIP).
* **Modular Data Processing Pipeline**:
    * `scripts/preprocess_data.py`: Uses modular processors (`ImageProcessor`, `TextProcessor`, `NumericalProcessor`, `DataFilter`) for text cleaning, image validation/compression, and numerical feature scaling.
    * `scripts/create_splits.py`: Generates train/validation/test splits, typically using a stratified strategy with activity filtering and optional dataset sampling.
    * **Feature Caching**:
        * `SimpleFeatureCache` (`src/data/simple_cache.py`): Centralized item feature caching (image, text, numerical) for training and inference. Cache is stored in model-specific subdirectories (e.g., `cache/resnet_sentence-bert/`). Supports memory LRU and optional disk persistence.
        * `scripts/precompute_cache.py`: Allows precomputing and saving all item features to disk for a given model configuration, significantly speeding up initial training runs.
        * `scripts/cache.py`: CLI tool for managing feature caches (list, clear, stats).
    * `MultimodalDataset` (`src/data/dataset.py`): Manages data loading, negative sampling, numerical normalization, and optional text augmentation. Integrates with `SimpleFeatureCache`.
* **Training Framework**:
    * Configurable optimizers (AdamW, Adam, SGD) and learning rate schedulers (ReduceLROnPlateau, Cosine, Step).
    * Includes gradient clipping, early stopping, and checkpoint management.
    * Optional Weights & Biases integration for experiment tracking.
    * Detailed progress monitoring and logging during training.
* **Evaluation Framework**:
    * Task-based evaluation for `Top-K Retrieval` and `Top-K Ranking`.
    * Standard metrics: Precision@k, Recall@k, F1@k, HitRate@k, NDCG@k, MRR.
    * Includes baseline recommenders (Random, Popularity, ItemKNN, UserKNN).
    * Support for negative sampling during retrieval evaluation for efficiency.
* **Inference**:
    * `scripts/generate_recommendations.py`: Generates recommendations for specified users.
    * Utilizes item feature caching for faster inference.
* **Utility Scripts**:
    * `scripts/extract_encoders.py`: Saves fitted user and item label encoders from the training data.
* **Modular Design**:
    * Separation of concerns for configuration, data, models, training, evaluation, and inference.
    * YAML-based configuration (`configs/simple_config.yaml` for essential settings, `configs/advanced_config.yaml` for detailed control).
* **Error Logging**: Training errors are logged to `results/error_log.json`.

---

## Directory Structure

```
PixelRec_Multimodal/
├── configs/                    # Configuration files (simple_config.yaml, advanced_config.yaml)
├── data/                       # Data directories (paths defined in config)
│   ├── raw/                    # Raw item_info, interactions, images
│   ├── processed/              # Processed item_info, interactions, images, scaler
│   ├── splits/                 # Train/validation/test splits
│   └── cache/                  # Root directory for SimpleFeatureCache (e.g., cache/resnet_sentence-bert/)
├── models/                     # Model checkpoints and saved encoders
│   └── checkpoints/
├── results/                    # Evaluation outputs, training metadata, figures, error logs
│   ├── figures/                # Training curve plots
│   └── results/                # JSON files for evaluation metrics of different recommenders
├── scripts/                    # Execution scripts (preprocess_data.py, train.py, etc.)
├── src/                        # Source code
│   ├── config.py               # Configuration dataclasses
│   ├── data/                   # Data handling (Dataset, Cache, Splitting, Processors)
│   │   └── processors/         # Modular data processors
│   ├── evaluation/             # Evaluation (Tasks, Metrics, Novelty)
│   ├── inference/              # Inference (Recommender, Baselines)
│   ├── models/                 # Model architectures (MultimodalRecommender, Layers, Losses)
│   └── training/               # Training (Trainer)
├── docs/                       # Documentation files
│   └── configuration.md
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

---

## Setup

1.  **Clone the repository.**
2.  **Create and activate a Python virtual environment.**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    You might also install the package in editable mode if you plan to modify the source:
    ```bash
    pip install -e .
    ```

---

## Usage

The system is operated via command-line scripts. All scripts accept a `--config` parameter to specify the configuration file (defaults to `configs/simple_config.yaml`).

### 1. Configuration

Modify `configs/simple_config.yaml` (recommended for most uses) or `configs/advanced_config.yaml` (for more detailed control). Refer to `docs/configuration.md` for details. Key sections:
* `model`: Architecture choices, embedding dimensions, pre-trained model names.
* `training`: Batch size, learning rate, optimizer details, epochs, patience.
* `data`: File paths, `cache_config` (for `SimpleFeatureCache`), numerical feature handling, augmentation, splitting parameters.
* `recommendation`: Top-k values for output.

### 2. Data Preprocessing

Process raw data (text cleaning, image validation/compression, numerical scaling):
```bash
python scripts/preprocess_data.py --config configs/advanced_config.yaml
```

### 3. Data Splitting

Create train/validation/test splits. Optionally, sample the dataset first:
```bash
python scripts/create_splits.py --config configs/advanced_config.yaml
# To sample 100,000 interactions before splitting:
python scripts/create_splits.py --config configs/advanced_config.yaml --sample_n 100000
```

### 4. (Optional) Precompute Features

Precompute and cache all item features to disk. This significantly speeds up the first training epoch and subsequent runs with the same model configuration. The cache is stored based on vision and language model names (e.g., `cache/resnet_sentence-bert/`).
```bash
python scripts/precompute_cache.py --config configs/advanced_config.yaml
# For a quick test with fewer items:
python scripts/precompute_cache.py --config configs/advanced_config.yaml --max_items 1000
```

### 5. Model Training

Train the multimodal recommender model:
```bash
python scripts/train.py --config configs/advanced_config.yaml --device cuda
```
Additional options:
```bash
# With Weights & Biases tracking
python scripts/train.py --config configs/advanced_config.yaml --use_wandb --wandb_project YourProject --wandb_entity YourUsername

# Resume training from a checkpoint
python scripts/train.py --config configs/advanced_config.yaml --resume models/checkpoints/best_model.pth
```
The training script saves model checkpoints, encoders, training curves, and metadata to the `results_dir` and `checkpoint_dir` specified in the config.

### 6. Model Evaluation

Evaluate the trained model or baselines.
```bash
# Evaluate multimodal model for retrieval
python scripts/evaluate.py --config configs/advanced_config.yaml \
    --test_data data/splits/split_tiny/test.csv \
    --train_data data/splits/split_tiny/train.csv \
    --eval_task retrieval \
    --recommender_type multimodal \
    --output multimodal_advanced_retrieval_metrics.json # Output saved in config.results_dir

# Evaluate popularity baseline for retrieval
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/split_tiny/test.csv \
    --train_data data/splits/split_tiny/train.csv \
    --eval_task retrieval \
    --recommender_type popularity \
    --output popularity_simple_retrieval_metrics.json
```
Available `eval_task` options: `retrieval`, `ranking`.
Available `recommender_type` options: `multimodal`, `random`, `popularity`, `item_knn`, `user_knn`.

### 7. Recommendation Generation

Generate top-K recommendations for specified users:
```bash
python scripts/generate_recommendations.py --config configs/advanced_config.yaml \
    --users user_A user_B \
    --output user_recommendations.json # Output saved in config.results_dir
```
Specify users via `--users <id1> <id2>`, `--user_file <path_to_file>`, or `--sample_users <N>`.

### 8. Cache Management

Manage `SimpleFeatureCache` directories:
```bash
# List all detected model-specific caches
python scripts/cache.py list

# Show stats for a specific cache
python scripts/cache.py stats resnet_sentence-bert

# Clear a specific cache
python scripts/cache.py clear resnet_sentence-bert

# Clear all caches (use with caution)
python scripts/cache.py clear --all
```

### 9. Extract Encoders (Utility)

If encoders were not saved during training or you need to re-extract them from the processed data:
```bash
python scripts/extract_encoders.py --config configs/advanced_config.yaml
```
This saves `user_encoder.pkl` and `item_encoder.pkl` to the `models/checkpoints/encoders/` directory (or as configured).

---

## Configuration Details

Refer to `docs/configuration.md`, `configs/simple_config.yaml`, and `configs/advanced_config.yaml` for detailed parameter descriptions. The system uses defaults defined in `src/config.py` for parameters not explicitly set in the chosen YAML file.

### Key Configuration Sections:

* **`model`**:
    * `vision_model`, `language_model`: Select pre-trained backbones (e.g., `resnet`, `sentence-bert`).
    * `embedding_dim`: Size of latent embeddings.
    * `use_contrastive`: Enable/disable contrastive learning (works best with CLIP vision model).
    * Advanced architectural parameters like `freeze_vision`, `freeze_language`, `fusion_hidden_dims`, `num_attention_heads`, etc.
* **`data`**:
    * Paths: `item_info_path`, `interactions_path`, `image_folder`, `processed_item_info_path`, `split_data_path`, etc.
    * `cache_config`: Controls the `SimpleFeatureCache`.
        * `enabled`: Boolean to turn caching on/off.
        * `max_memory_items`: Maximum items for in-memory LRU.
        * `cache_directory`: **Base directory for model-specific caches** (e.g., `cache/`). The actual cache will be in a subfolder like `cache/visionX_langY/`.
        * `use_disk`: Boolean to enable disk persistence for the cache.
    * `numerical_features_cols`: List of columns for numerical features.
    * `numerical_normalization_method`: Method for scaling numerical features.
    * Advanced settings for text augmentation, image compression/validation, text cleaning, and data splitting.
* **`training`**: Learning parameters (learning rate, epochs, batch size), optimizer type and its parameters, scheduler settings, loss component weights (`bce_weight`, `contrastive_weight`).
* **`recommendation`**: `top_k` for recommendation lists, and advanced settings for diversity/novelty.
* **Root Level**: `checkpoint_dir`, `results_dir`.

---

## Extensibility

The system's modularity allows for extensions:

* **New Models**: Modify `src/models/multimodal.py` or add new model classes. Update configurations in `src/config.py` (e.g., `MODEL_CONFIGS`) and script instantiations as needed.
* **New Data Processors**: Add new processor classes in `src/data/processors/` and integrate them into `scripts/preprocess_data.py`.
* **New Evaluation Metrics/Tasks**: Add metrics to `src/evaluation/metrics.py` or `src/evaluation/advanced_metrics.py`. New tasks can be created by extending `BaseEvaluator` in `src/evaluation/tasks.py`.

---
