# Multimodal Recommender System

This repository contains a framework for building and experimenting with multimodal recommender systems. It uses visual, textual, and numerical data to generate recommendations. The system is designed with a focus on configurability for model architecture, data processing, and evaluation.

---

## Features

* **Multimodal Data Integration**: Processes items with images, text, and numerical features.
* **Flexible Model Architecture**:
    * Core model: `MultimodalRecommender`.
    * Supports various pre-trained vision (e.g., CLIP, ResNet, DINO) and language models (e.g., Sentence-BERT, MPNet).
    * Configurable fusion methods, including multi-head self-attention.
    * Optional contrastive learning for vision-text alignment.
* **Data Processing Pipeline**:
    * `scripts/preprocess_data.py`: Handles text cleaning, image validation/compression, and numerical feature scaling.
    * `scripts/create_splits.py`: Generates train/validation/test splits, typically using a stratified strategy with activity filtering.
    * **Feature Caching**: `SimpleFeatureCache` for item features (image, text, numerical) to accelerate data loading, with optional memory and disk persistence.
    * `MultimodalDataset`: Manages data loading, negative sampling, numerical normalization, and optional text augmentation.
* **Training Framework**:
    * Configurable optimizers (AdamW, Adam, SGD) and learning rate schedulers (ReduceLROnPlateau, Cosine, Step).
    * Includes gradient clipping, early stopping, and checkpoint management.
    * Optional Weights & Biases integration.
* **Evaluation Framework**:
    * Task-based evaluation for `Top-K Retrieval` and `Top-K Ranking`.
    * Standard metrics: Precision@k, Recall@k, NDCG@k, MRR.
    * Includes baseline recommenders (Random, Popularity, ItemKNN, UserKNN).
* **Inference**:
    * Batch processing for recommendation generation.
    * Utilizes item feature caching for faster inference.
* **Modular Design**:
    * Separation of concerns for configuration, data, models, training, evaluation, and inference.
    * YAML-based configuration (`simple_config.yaml` for essential settings, `advanced_config.yaml` for detailed control).

---

## Directory Structure

```
PixelRec_Multimodal/
├── configs/                    # Configuration files (simple_config.yaml, advanced_config.yaml)
├── data/                       # Data directories (paths defined in config)
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── cache/
├── models/                     # Model checkpoints
│   └── checkpoints/
├── results/                    # Evaluation outputs
├── scripts/                    # Execution scripts (preprocess_data.py, train.py, etc.)
├── src/                        # Source code
│   ├── config.py               # Configuration dataclasses
│   ├── data/                   # Data handling (Dataset, Cache, Splitting, Processors)
│   ├── evaluation/             # Evaluation (Tasks, Metrics)
│   ├── inference/              # Inference (Recommender, Baselines)
│   ├── models/                 # Model architectures (MultimodalRecommender, Layers, Losses)
│   └── training/               # Training (Trainer)
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

---

## Usage

The system is operated via command-line scripts. All scripts accept a `--config` parameter to specify the configuration file.

### 1. Configuration

Modify `configs/simple_config.yaml` (recommended for most uses) or `configs/advanced_config.yaml` (for more detailed control). Key sections:
* `model`: Architecture choices, embedding dimensions.
* `training`: Batch size, learning rate, optimizer details.
* `data`: File paths, preprocessing settings, caching, splitting parameters.
* `recommendation`: Top-k values for output.

### 2. Data Preprocessing

Process raw data (text cleaning, image validation/compression, numerical scaling):
```bash
python scripts/preprocess_data.py --config configs/simple_config.yaml
```

### 3. Data Splitting

Create train/validation/test splits. Optionally, sample the dataset first:
```bash
python scripts/create_splits.py --config configs/simple_config.yaml
# To sample 100,000 interactions before splitting:
python scripts/create_splits.py --config configs/simple_config.yaml --sample_n 100000
```

### 4. Model Training

Train the multimodal recommender model:
```bash
python scripts/train.py --config configs/simple_config.yaml --device cuda
```
Additional options:
```bash
# With Weights & Biases tracking
python scripts/train.py --config configs/simple_config.yaml --use_wandb --wandb_project YourProject --wandb_entity YourUsername

# Resume training from a checkpoint
python scripts/train.py --config configs/simple_config.yaml --resume models/checkpoints/best_model.pth
```

### 5. Model Evaluation

Evaluate the trained model or baselines:
```bash
# Evaluate multimodal model for retrieval
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/split_tiny/test.csv \
    --train_data data/splits/split_tiny/train.csv \
    --eval_task retrieval \
    --recommender_type multimodal \
    --output results/multimodal_retrieval_metrics.json

# Evaluate popularity baseline for retrieval
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/split_tiny/test.csv \
    --train_data data/splits/split_tiny/train.csv \
    --eval_task retrieval \
    --recommender_type popularity \
    --output results/popularity_retrieval_metrics.json
```
Available `eval_task` options: `retrieval`, `ranking`.

### 6. Recommendation Generation

Generate top-K recommendations:
```bash
python scripts/generate_recommendations.py --config configs/simple_config.yaml \
    --users user_A user_B \
    --output results/user_recommendations.json
```
Specify users via `--users`, `--user_file`, or `--sample_users N`.

---

## Configuration

Refer to `docs/configuration.md`, `configs/simple_config.yaml`, and `configs/advanced_config.yaml` for detailed parameter descriptions. The system uses defaults for parameters not explicitly set in the chosen YAML file.

### Key Configuration Sections:

* **`model`**: Select pre-trained backbones, embedding dimensions, contrastive learning.
* **`data`**: Paths, cache settings (`data.cache_config`), numerical feature handling, augmentation.
* **`training`**: Learning parameters, optimizer, scheduler, loss weights.

---

## Extensibility

The system's modularity allows for extensions:

* **New Models**: Modify `src/models/multimodal.py` or add new model classes. Update configurations and script instantiations as needed.
* **New Evaluation Metrics/Tasks**: Add metrics to `src/evaluation/metrics.py` or tasks by extending `BaseEvaluator` in `src/evaluation/tasks.py`.

---
