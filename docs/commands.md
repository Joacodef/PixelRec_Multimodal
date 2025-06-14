# Script Commands Reference

This document provides a comprehensive reference for all executable scripts located in the `scripts/` directory. These scripts are the primary interface for running the end-to-end recommendation system pipeline.

## Main Workflow Scripts

These scripts are intended to be run in sequence to process data, train a model, and generate recommendations.

### `preprocess_data.py`

* **Purpose**: Cleans and preprocesses the raw `interactions.csv` and `item_info.csv` files. It filters data based on the configured criteria and saves the processed DataFrame.
* **Usage**:
    ```bash
    python scripts/preprocess_data.py --config <path_to_config.yaml>
    ```

### `create_splits.py`

* **Purpose**: Splits the preprocessed interaction data into training, validation, and test sets according to the strategy defined in the configuration file (e.g., random, time-based).
* **Usage**:
    ```bash
    python scripts/create_splits.py --config <path_to_config.yaml>
    ```

### `extract_encoders.py`

* **Purpose**: Creates and saves encoders for all specified categorical features. This step is necessary to transform categorical variables into a numerical format for the model.
* **Usage**:
    ```bash
    python scripts/extract_encoders.py --config <path_to_config.yaml>
    ```

### `precompute_cache.py`

* **Purpose**: Pre-computes and caches computationally expensive features, such as image and text embeddings. This can significantly speed up the training process by avoiding redundant calculations in each epoch.
* **Usage**:
    ```bash
    python scripts/precompute_cache.py --config <path_to_config.yaml>
    ```

### `train.py`

* **Purpose**: Executes the model training loop using the training and validation data splits. It handles model initialization, optimization, and saving checkpoints.
* **Usage**:
    ```bash
    python scripts/train.py --config <path_to_config.yaml>
    ```

### `evaluate.py`

* **Purpose**: Evaluates the trained model on the test set. It computes and saves various recommendation metrics, such as Precision@K, Recall@K, and NDCG@K.
* **Usage**:
    ```bash
    python scripts/evaluate.py --config <path_to_config.yaml>
    ```

### `generate_recommendations.py`

* **Purpose**: Uses the trained model to generate and save top-K recommendations for each user in the test set.
* **Usage**:
    ```bash
    python scripts/generate_recommendations.py --config <path_to_config.yaml>
    ```

## Advanced & Utility Scripts

These scripts provide supplementary functionality for tasks like hyperparameter optimization and managing model artifacts.

### `hyperparameter_search.py`

* **Purpose**: Performs automated hyperparameter optimization using Optuna. It runs multiple training trials based on the search space defined in the configuration file to find the best model parameters.
* **Usage**:
    ```bash
    python scripts/hyperparameter_search.py --config <path_to_config.yaml>
    ```

### `checkpoint_manager.py`

* **Purpose**: A utility script to manage saved model checkpoints. It allows for listing all checkpoints or deleting specific ones to manage disk space.
* **Usage**:
    * To list all checkpoints:
        ```bash
        python scripts/checkpoint_manager.py --list --checkpoint_dir <path_to_output_directory>
        ```
    * To delete a specific checkpoint:
        ```bash
        python scripts/checkpoint_manager.py --delete <checkpoint_filename> --checkpoint_dir <path_to_output_directory>
        ```

## Development Scripts

### `run_tests.py`

* **Purpose**: Executes the entire suite of unit and integration tests for the project. This script is primarily used for development and continuous integration to ensure code quality and correctness.
* **Usage**:
    ```bash
    python scripts/run_tests.py
    ```