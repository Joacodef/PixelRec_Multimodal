# Multimodal Recommender System

This repository provides a comprehensive framework for building and experimenting with multimodal recommender systems. It is designed to leverage both visual and textual information from datasets (e.g., those similar in structure to PixelRec/Pixel200k) to provide personalized recommendations. The system is highly configurable, allowing for easy modification of model architectures, data processing pipelines, and evaluation strategies.

---

## Core Features

* **Multimodal Data Integration**: Designed to process items with associated images, textual descriptions, and numerical features.
* **Flexible Model Architectures**:
    * Supports two main model classes: `PretrainedMultimodalRecommender` (default) and `EnhancedMultimodalRecommender` (which includes cross-modal attention layers). These are configurable via `model.model_class` in the configuration file.
    * Utilizes various pre-trained vision models (e.g., CLIP, DINO, ResNet, ConvNeXT) and language models (e.g., Sentence-BERT, MPNet, BERT, RoBERTa), selectable in `model.vision_model` and `model.language_model`.
    * Includes configurable fusion mechanisms, such as multi-head self-attention for combining diverse feature sets. The fusion network's architecture (hidden dimensions, activation, batch normalization) is fully configurable.
    * Option for contrastive learning (primarily with CLIP-based vision models) to better align vision and text representations, controlled by `model.use_contrastive`.
* **Advanced Data Preprocessing & Handling**:
    * `scripts/preprocess_data.py`: Performs initial data cleaning based on configurations in `data.offline_text_cleaning` (HTML removal, Unicode normalization, lowercasing) and `data.offline_image_validation` (corruption checks, dimension checks).
    * Optional offline image compression and resizing via `data.offline_image_compression` settings, with processed images saved to a new directory.
    * `scripts/create_splits.py`: Generates train/validation/test splits using strategies from `src/data/splitting.py` (defaulting to stratified split after activity filtering). Allows optional dataset sampling via `--sample_n`.
    * `MultimodalDataset` class handles data loading, negative sampling, numerical feature normalization (log1p, standardization, min-max via `data.numerical_normalization_method`), and text augmentation.
    * `SharedImageCache`: Efficiently caches processed image tensors to disk or in memory to accelerate data loading during training, enabled by `data.cache_processed_images`.
* **Robust Training Framework**:
    * The `Trainer` class manages the training loop with configurable optimizers (AdamW, Adam, SGD), learning rate schedulers (ReduceLROnPlateau, Cosine, Step), gradient clipping, and early stopping.
    * Integration with Weights & Biases for experiment tracking and visualization, enabled via the `--use_wandb` flag in `scripts/train.py`.
* **Comprehensive Evaluation**:
    * `scripts/evaluate.py` utilizes a task-based evaluation framework (`src/evaluation/tasks.py`) supporting various scenarios: 'retrieval', 'ranking', 'next_item', 'cold_user', 'cold_item', 'beyond_accuracy', and 'legacy'.
    * Calculates standard recommendation metrics (Precision@k, Recall@k, NDCG@k, MAP) and beyond-accuracy metrics like novelty, diversity, and catalog coverage.
    * Additional advanced metrics are available in `src/evaluation/advanced_metrics.py` (e.g., MRR, Gini, Serendipity).
* **Baseline Recommenders**: Includes Random, Popularity, ItemKNN, and UserKNN baselines for comparison, evaluable via `scripts/evaluate.py --recommender_type <baseline_name>`.
* **Inference and Recommendation**:
    * `scripts/generate_recommendations.py` generates top-K recommendations for specified users or a sample of users.
    * Supports filtering of seen items and generation of diverse recommendations using an MMR-like re-ranking technique (`--use_diversity` flag), which also reports novelty metrics for the generated list.
* **Modularity and Configuration**:
    * The codebase is organized into distinct modules for configuration, data handling, model architectures, training, evaluation, and inference.
    * System behavior is extensively controlled via YAML configuration files (e.g., `configs/default_config.yaml`) parsed by `src/config.py`.

---

## Directory Structure

The project is structured as follows:

* `PixelRec_Multimodal/` (or your chosen root project name)
    * `configs/` - Configuration files (e.g., `default_config.yaml`)
    * `data/` - Placeholder for raw, processed, and split data (actual paths defined in config)
        * `raw/item_info/`, `raw/interactions/`, `raw/images/`
        * `processed/`
        * `splits/`
    * `models/` - Placeholder for saved model checkpoints and encoders (actual paths defined in config)
    * `results/` - Placeholder for evaluation results, figures, generated recommendations, etc. (actual paths defined in config)
    * `scripts/` - High-level Python scripts for executing pipeline stages:
        * `preprocess_data.py` - Data preprocessing (cleaning, image validation/compression).
        * `create_splits.py` - Creating standardized train/validation/test data splits.
        * `extract_encoders.py` - Utility to (re)generate and save user/item encoders.
        * `train.py` - Model training.
        * `evaluate.py` - Model and baseline evaluation.
        * `generate_recommendations.py` - Generating recommendations for users.
    * `src/` - Source code for the recommender system:
        * `config.py` - Dataclasses for managing all configurations.
        * `data/` - Modules for `MultimodalDataset`, `SharedImageCache`, data splitting, and preprocessing utilities.
        * `evaluation/` - Modules for various evaluation metrics, tasks, and novelty/diversity calculations.
        * `inference/` - Modules for the main `Recommender` logic and baseline model implementations.
        * `models/` - Modules for model architectures (`PretrainedMultimodalRecommender`, `EnhancedMultimodalRecommender`), custom layers (`CrossModalAttention`), and loss functions.
        * `training/` - Module for the `Trainer` class and related training utilities.
    * `requirements.txt` - List of Python dependencies.
    * `setup.py` - Python package setup script.
    * `README.md` - This file.

---

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    # Example: git clone <your-repository-url>
    # cd <repository-name>
    ```

2.  **Create and activate a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The primary way to interact with the system is by running the Python scripts located in the `scripts/` directory. Ensure your data (item information, interactions, images) is accessible and the paths are correctly specified in your configuration file (e.g., `configs/default_config.yaml`).

1.  **Configure your Setup:**
    Modify `configs/default_config.yaml` (or create a new one, e.g., `my_config.yaml`) to point to your data paths, define model parameters, training settings, etc. All subsequent commands will use the `--config` argument to specify which configuration to use.

2.  **Preprocess Data:**
    This script handles the initial processing of your raw data. It cleans text, validates images, optionally compresses/resizes images, and filters items/interactions based on your configuration.
    ```bash
    python scripts/preprocess_data.py --config configs/default_config.yaml
    ```

3.  **Create Data Splits:**
    This script creates standardized train, validation, and test splits from the processed interaction data.
    ```bash
    python scripts/create_splits.py --config configs/default_config.yaml
    ```
    * Optionally, sample the dataset before splitting:
        ```bash
        python scripts/create_splits.py --config configs/default_config.yaml --sample_n 100000
        ```

4.  **Extract Encoders (Optional but Recommended for Consistency):**
    The training script typically saves encoders. This utility can be used if you need to (re)generate them based on the full processed dataset without running a full training.
    ```bash
    python scripts/extract_encoders.py --config configs/default_config.yaml
    ```

5.  **Train the Model:**
    This script trains the multimodal recommender using the specified configuration and data splits.
    ```bash
    python scripts/train.py --config configs/default_config.yaml --device cuda  # Use 'cpu' if CUDA is not available
    ```
    * Enable Weights & Biases logging: `--use_wandb --wandb_project MyRecSysProject --wandb_entity <your_entity>`
    * Resume training: `--resume <path_to_checkpoint.pth>`

6.  **Evaluate the Model:**
    Evaluate the trained model or baselines on a test set using various tasks and metrics.
    ```bash
    python scripts/evaluate.py --config configs/default_config.yaml \
        --test_data data/splits/your_split/test.csv \
        --train_data data/splits/your_split/train.csv \
        --eval_task retrieval \
        --recommender_type multimodal \
        --output results/multimodal_retrieval_metrics.json
    ```
    * To evaluate a baseline (e.g., Popularity on the 'ranking' task):
        ```bash
        python scripts/evaluate.py --config configs/default_config.yaml \
            --test_data data/splits/your_split/test.csv \
            --train_data data/splits/your_split/train.csv \
            --eval_task ranking \
            --recommender_type popularity \
            --output results/popularity_ranking_metrics.json
        ```
    * **Important**: For 'retrieval' and other tasks that depend on distinguishing novel items, always provide the `--train_data` argument pointing to the corresponding training split.

7.  **Generate Recommendations:**
    Generate top-K recommendations for users with a trained model.
    ```bash
    python scripts/generate_recommendations.py --config configs/default_config.yaml \
        --users user_id_1 user_id_2 \
        --output results/user_recommendations.json
    ```
    * Provide a file of user IDs: `--user_file <path_to_user_ids.txt>`
    * Generate for a random sample of users: `--sample_users 100`
    * Enable diverse recommendations and novelty metrics: `--use_diversity`
    * Load pre-computed item features cache for faster inference: `--embeddings_cache <path_to_cache.pkl>`

---

## Configuration

The system's behavior is extensively controlled via YAML configuration files (e.g., `configs/default_config.yaml`), parsed by `src/config.py` into structured dataclasses. Key configurable sections include:

* **`model`**:
    * `model_class`: `pretrained` or `enhanced`.
    * `vision_model`, `language_model`: Names of pre-trained backbones.
    * `embedding_dim`, `dropout_rate`, `contrastive_temperature`.
    * Architectural details for attention, fusion network (`fusion_hidden_dims`, `fusion_activation`, `use_batch_norm`), projection layers, and cross-modal attention (for 'enhanced' model).
* **`training`**:
    * Batch size, epochs, learning rate, weight decay, optimizer settings, LR scheduler parameters, early stopping patience, gradient clipping.
* **`data`**:
    * Paths for raw, processed, and split data files.
    * `image_folder`, `processed_image_destination_folder`.
    * `cache_processed_images`: Boolean to enable/disable image tensor caching.
    * `offline_image_compression`: Settings for enabling, target quality, and resize parameters for image preprocessing.
    * `offline_image_validation`: Parameters for validating images (corruption, dimensions, extensions).
    * `offline_text_cleaning`: Toggles for HTML removal, Unicode normalization, lowercasing.
    * `negative_sampling_ratio`, `numerical_features_cols`, `numerical_normalization_method`.
    * `text_augmentation`: Configuration for text augmentation strategies.
    * `splitting`: Parameters for data splitting (`train_final_ratio`, `val_final_ratio`, `test_final_ratio`, `min_interactions_per_user`, `min_interactions_per_item`, `random_state`).
* **`recommendation`**:
    * `top_k`, `diversity_weight`, `novelty_weight`, `filter_seen`, `max_candidates`.
* Global paths like `checkpoint_dir` and `results_dir`.

Refer to `src/config.py` for the full structure and all available options. You can create multiple YAML configuration files to manage different experiments and setups.
