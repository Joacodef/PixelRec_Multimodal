# Multimodal Recommender System

This repository provides a comprehensive framework for building and experimenting with multimodal recommender systems. It is designed to leverage both visual and textual information from datasets (such as PixelRec) to provide personalized recommendations. The system is highly configurable, allowing for easy modification of model architectures, data processing pipelines, and evaluation metrics.

---

## Core Features

* **Multimodal Data Integration**: Designed to process items that have associated images and textual descriptions.
* **Flexible Model Architectures**:
    * Supports two main model classes: `PretrainedMultimodalRecommender` (default) and `EnhancedMultimodalRecommender` (with cross-modal attention), configurable via `model.model_class`.
    * Utilizes various pre-trained vision models (e.g., CLIP, DINO, ResNet, ConvNeXT) and language models (e.g., Sentence-BERT, MPNet, BERT, RoBERTa), selectable in the configuration.
    * Includes configurable fusion mechanisms, such as multi-head self-attention for combining diverse feature sets. The `EnhancedMultimodalRecommender` adds cross-modal attention layers.
    * Option for contrastive learning (primarily with CLIP-based models) to better align vision and text representations.
* **Advanced Data Preprocessing**:
    * Handles text cleaning (HTML removal, Unicode normalization, lowercasing), image validation (corruption checks, dimension checks), and optional offline image compression/resizing.
    * Supports numerical feature normalization (log1p, standardization, min-max) and text augmentation (random delete, random swap).
    * The `scripts/create_splits.py` script generates train/validation/test splits using a stratified approach by default, ensuring users are represented across splits. Other splitting strategies (user-based, item-based, temporal, leave-one-out) are available programmatically within `src/data/splitting.py`.
* **Robust Training Framework**:
    * The `Trainer` class manages the training loop with support for different optimizers (AdamW, Adam, SGD), learning rate schedulers (ReduceLROnPlateau, Cosine, Step), gradient clipping, and early stopping, all configurable.
    * Integrates with Weights & Biases for experiment tracking and visualization, enabled via the `--use_wandb` flag in `scripts/train.py`.
    * Features an image caching mechanism (`SharedImageCache`) to speed up training by pre-processing and storing image tensors, configurable via `data.cache_processed_images`.
* **Comprehensive Evaluation**:
    * The `scripts/evaluate.py` script calculates standard recommendation metrics such as Precision@k, Recall@k, NDCG@k, and catalog coverage.
    * Novelty and diversity metrics (e.g., self-information, IIF, long-tail percentage, personalized novelty) can be generated for recommendation lists when using `scripts/generate_recommendations.py` with the `--use_diversity` flag.
    * The codebase includes additional advanced metrics in `src/evaluation/advanced_metrics.py` (e.g., MRR, Gini, Serendipity) available for custom analysis.
* **Baseline Recommenders**: Includes several baseline recommenders (Random, Popularity, ItemKNN, UserKNN) which can be evaluated using `scripts/evaluate.py` via the `--recommender_type` argument.
* **Inference and Recommendation**:
    * Generates top-K recommendations for specified users using `scripts/generate_recommendations.py`.
    * Allows filtering of items already seen by the user during recommendation.
    * Supports generation of diverse recommendations through a re-ranking technique (MMR-like) considering item embeddings and novelty, enabled via the `--use_diversity` flag in `scripts/generate_recommendations.py`.
* **Modularity**: The codebase is organized into distinct modules for configuration, data handling, model architectures, training procedures, evaluation, and inference.

---

## Directory Structure

The project is structured as follows:

* `PixelRec_Multimodal/`
    * `configs/` - Configuration files (e.g., `default_config.yaml`)
    * `data/` - Placeholder for raw and processed data (paths defined in config)
        * `raw/`
        * `processed/`
    * `models/` - Placeholder for saved model checkpoints and encoders (paths defined in config)
    * `results/` - Placeholder for evaluation results, figures, etc. (paths defined in config)
    * `scripts/` - High-level scripts for executing pipeline stages
        * `preprocess_data.py` - Script for data preprocessing.
        * `create_splits.py` - Script for creating standardized train/validation/test data splits.
        * `extract_encoders.py` - Script to extract and save user/item encoders from processed data.
        * `train.py` - Script for model training.
        * `evaluate.py` - Script for evaluating trained models and baselines.
        * `generate_recommendations.py` - Script for generating recommendations.
    * `src/` - Source code for the recommender system
        * `config.py` - Dataclasses for managing configurations.
        * `data/` - Modules for dataset handling (`MultimodalDataset`), preprocessing, splitting, and image caching (`SharedImageCache`).
        * `evaluation/` - Modules for various evaluation metrics (standard, novelty, advanced).
        * `inference/` - Modules for recommendation generation (`Recommender`) and baseline models.
        * `models/` - Modules for model architectures (`PretrainedMultimodalRecommender`, `EnhancedMultimodalRecommender`), custom layers (`CrossModalAttention`), and loss functions.
        * `training/` - Module for the training loop (`Trainer`) and related utilities.
    * `requirements.txt` - List of Python dependencies.
    * `setup.py` - Python package setup script.

---

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    # Example: git clone <your-repository-url>
    # cd PixelRec_Multimodal-3da503c994117442d348b7de6ee14e88731b6565
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

The main way to interact with the system is by running the Python scripts located in the `scripts/` directory. Ensure your data (item information, interactions, images) is accessible and the paths are correctly specified in your configuration file (e.g., `configs/default_config.yaml`).

1.  **Preprocess Data:**
    This script handles the initial processing of your raw data based on the settings in your configuration file. This includes cleaning text, validating and processing images (e.g., compression), and filtering items and interactions.
    ```bash
    python scripts/preprocess_data.py --config configs/default_config.yaml
    ```

2.  **Create Data Splits:**
    This script creates standardized train, validation, and test splits from the processed interaction data. It uses a stratified splitting strategy by default. The output paths for splits are defined in the configuration.
    ```bash
    python scripts/create_splits.py --config configs/default_config.yaml
    ```
    * Optionally, you can sample the dataset before splitting using `--sample_n <number_of_interactions>`.

3.  **Extract Encoders (Optional but Recommended for Consistency):**
    If you need to ensure encoders (for user/item IDs) are based on the entirety of your processed data before training or if they were not saved during a previous training run, you can (re)generate and save them. The training script also saves encoders.
    ```bash
    python scripts/extract_encoders.py --config configs/default_config.yaml
    ```

4.  **Train the Model:**
    This script trains the multimodal recommender. It manages data loading (using pre-defined splits), numerical feature scaling, model initialization, and the overall training process. Encoders for users and items are typically fitted on the full dataset and saved during this process.
    ```bash
    python scripts/train.py --config configs/default_config.yaml --device cuda  # Use 'cpu' if CUDA is not available
    ```
    * Optional: Add `--use_wandb` to log metrics to Weights & Biases.
    * Optional: Use `--resume <path_to_checkpoint.pth>` to resume training from a saved checkpoint.

5.  **Evaluate the Model:**
    After training, this script evaluates the model's performance on a test dataset. It reports metrics like P@k, R@k, NDCG@k, and catalog coverage. It can also evaluate baseline recommenders.
    ```bash
    python scripts/evaluate.py --config configs/default_config.yaml --test_data <path_to_your_test_interactions.csv> --output results/evaluation_metrics.json
    ```
    * To evaluate a baseline model (e.g., Popularity):
        ```bash
        python scripts/evaluate.py --config configs/default_config.yaml --test_data <path_to_your_test_interactions.csv> --output results/popularity_metrics.json --recommender_type popularity
        ```

6.  **Generate Recommendations:**
    Use this script to generate recommendations for specific users or a sample of users with a trained model.
    ```bash
    python scripts/generate_recommendations.py --config configs/default_config.yaml --users <user_id_1> <user_id_2> --output results/user_recommendations.json
    ```
    * You can also provide a file with user IDs using `--user_file <path_to_user_ids.txt>`.
    * To generate for a random sample, use `--sample_users <number_of_users>`.
    * To enable diversity and novelty in recommendations (and get related metrics), add the `--use_diversity` flag.

---

## Configuration

The entire system is highly configurable through the `configs/default_config.yaml` file (or copies thereof). This YAML file allows detailed customization of:
* **Model Architecture**: Choice of `model_class` ('pretrained' or 'enhanced'), vision and language models, embedding dimensions, fusion network layers (including attention parameters), activation functions, dropout rates, cross-modal attention settings (for 'enhanced' model), etc.
* **Training Parameters**: Batch size, number of epochs, learning rate, weight_decay, optimizer type (AdamW, Adam, SGD) and its parameters, learning rate scheduler settings, early stopping patience, gradient clipping, etc.
* **Data Handling**: Paths to raw, processed, and split data, image folder, image processing settings (compression, validation, caching), text cleaning and augmentation strategies, numerical feature normalization methods, data splitting configurations (min interactions for filtering), etc.
* **Recommendation Settings**: Top-K value for recommendations, weights for diversity and novelty in re-ranking, candidate generation parameters, etc.
* **Output Directories**: Paths for saving checkpoints, results, logs, and encoders.

You can create copies of this file or modify it directly to experiment with different setups.
