# Multimodal Recommender System

This repository provides a comprehensive framework for building and experimenting with multimodal recommender systems. It is designed to leverage both visual and textual information from datasets (such as PixelRec) to provide personalized recommendations. The system is highly configurable, allowing for easy modification of model architectures, data processing pipelines, and evaluation metrics.

---

## Core Features

* **Multimodal Data Integration**: Designed to process items that have associated images and textual descriptions.
* **Flexible Model Architectures**:
    * Supports various pre-trained vision models (e.g., CLIP, DINO, ResNet, ConvNeXT) and language models (e.g., Sentence-BERT, MPNet, BERT, RoBERTa).
    * Includes configurable fusion mechanisms, such as attention layers and cross-modal attention, for combining diverse feature sets.
    * Option for contrastive learning to better align vision and text representations.
* **Advanced Data Preprocessing**:
    * Handles text cleaning (HTML removal, Unicode normalization), image validation (corruption checks, dimension checks), and optional offline image compression/resizing.
    * Supports numerical feature normalization and text augmentation.
    * Offers multiple data splitting strategies tailored for recommender systems (e.g., stratified, user-based, temporal).
* **Robust Training Framework**:
    * `Trainer` class manages the training loop with support for different optimizers, learning rate schedulers, gradient clipping, and early stopping.
    * Integrates with Weights & Biases for experiment tracking and visualization.
    * Features an image caching mechanism to speed up training by pre-processing and storing image tensors.
* **Comprehensive Evaluation**:
    * Calculates standard recommendation metrics such as Precision@k, Recall@k, and NDCG@k.
    * Includes metrics for catalog coverage, novelty, and diversity.
* **Inference and Recommendation**:
    * Generates top-K recommendations for specified users.
    * Allows filtering of items already seen by the user.
    * Supports generation of diverse recommendations through re-ranking techniques.
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
        * `preprocess_data.py` - Script for data preprocessing
        * `train.py` - Script for model training
        * `evaluate.py` - Script for evaluating trained models
        * `generate_recommendations.py` - Script for generating recommendations
    * `src/` - Source code for the recommender system
        * `config.py` - Dataclasses for managing configurations
        * `data/` - Modules for dataset handling, preprocessing, splitting, and image caching
        * `evaluation/` - Modules for various evaluation metrics
        * `inference/` - Modules for recommendation generation and serving
        * `models/` - Modules for model architectures, custom layers, and loss functions
        * `training/` - Module for the training loop and related utilities
    * `requirements.txt` - List of Python dependencies
    * `setup.py` - Python package setup script

---

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    # Example: git clone <your-repository-url>
    # cd PixelRec_Multimodal-7d42358499bb9df4f2516e823eb8c7562c1b6e4a
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

2.  **Train the Model:**
    This script trains the multimodal recommender. It manages data splitting, numerical feature scaling, model initialization, and the overall training process.
    ```bash
    python scripts/train.py --config configs/default_config.yaml --device cuda  # Use 'cpu' if CUDA is not available
    ```
    * Optional: Add `--use_wandb` to log metrics to Weights & Biases.
    * Optional: Use `--resume <path_to_checkpoint.pth>` to resume training from a saved checkpoint.

3.  **Evaluate the Model:**
    After training, this script evaluates the model's performance on a test dataset.
    ```bash
    python scripts/evaluate.py --config configs/default_config.yaml --test_data <path_to_your_test_interactions.csv> --output results/evaluation_metrics.json
    ```

4.  **Generate Recommendations:**
    Use this script to generate recommendations for specific users or a sample of users with a trained model.
    ```bash
    python scripts/generate_recommendations.py --config configs/default_config.yaml --users <user_id_1> <user_id_2> --output results/user_recommendations.json
    ```
    * You can also provide a file with user IDs using `--user_file <path_to_user_ids.txt>`.
    * To generate for a random sample, use `--sample_users <number_of_users>`.
    * To enable diversity in recommendations, add the `--use_diversity` flag.

---

## Configuration

The entire system is highly configurable through the `configs/default_config.yaml` file. This YAML file allows detailed customization of:
* **Model Architecture**: Choice of vision and language models, embedding dimensions, fusion network layers, activation functions, dropout rates, etc.
* **Training Parameters**: Batch size, number of epochs, learning rate, weight decay, optimizer type, learning rate scheduler settings, early stopping patience, etc.
* **Data Handling**: Paths to raw and processed data, image folder, image processing settings (compression, validation), text augmentation strategies, numerical feature normalization methods, data splitting configurations, etc.
* **Recommendation Settings**: Top-K value for recommendations, weights for diversity and novelty in re-ranking, etc.
* **Output Directories**: Paths for saving checkpoints, results, and logs.

You can create copies of this file or modify it directly to experiment with different setups.