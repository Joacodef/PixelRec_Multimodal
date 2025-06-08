This guide provides a comprehensive overview of all command-line scripts available in the Multimodal Recommender System. These scripts, located in the `scripts/` directory, cover data preprocessing, model training, evaluation, and various utility functions. Each script is designed to be executed from the root directory of the repository.

To run any command, use the following syntax:

```bash
python scripts/<script_name>.py --<argument> <value>
```

Many commands accept a `--config` argument to specify a YAML configuration file (e.g., `configs/simple_config.yaml` or `configs/advanced_config.yaml`).

## 1. Data Preparation and Preprocessing

These commands are used to prepare your raw data for model training.

### `scripts/preprocess_data.py`

This script performs a full data preprocessing pipeline, including text cleaning, image validation and compression, numerical feature handling, and filtering.

* **Description**: Validates, cleans, processes raw item information and user interactions, and saves them to a processed data directory.
* **Arguments**:
    * `--config <path>`: Path to the configuration file. (Default: `configs/simple_config.yaml`)
    * `--skip-caching`: If set, skips the feature caching step during preprocessing.
    * `--force-reprocess`: If set, forces reprocessing of all images and features, overwriting existing ones.
* **Example**:
    ```bash
    python scripts/preprocess_data.py --config configs/simple_config.yaml
    ```

### `scripts/create_splits.py`

This script divides your processed data into training, validation, and test sets.

* **Description**: Creates standardized and reproducible data splits (train, validation, test) from the processed interactions data based on configured ratios and filtering criteria.
* **Arguments**:
    * `--config <path>`: Path to the configuration file. (Default: `configs/simple_config.yaml`)
    * `--sample_n <int>`: Optional. Number of random interactions to sample from the dataset before splitting. If not provided, the full dataset is used.
* **Example**:
    ```bash
    python scripts/create_splits.py --config configs/simple_config.yaml --sample_n 10000
    ```

### `scripts/precompute_cache.py`

This script pre-computes and caches multimodal features for faster data loading during training.

* **Description**: Iterates through all items in the dataset, processes their visual, textual, and numerical data to generate feature tensors, and saves these features to disk.
* **Arguments**:
    * `--config <path>`: Path to the configuration file. (Required)
    * `--force_recompute`: If set, forces recomputation of all items, overwriting existing cache files.
    * `--max_items <int>`: Optional. Limits the number of items to process, useful for debugging.
* **Example**:
    ```bash
    python scripts/precompute_cache.py --config configs/simple_config.yaml --force_recompute
    ```

## 2. Model Training and Management

These commands handle the training of your multimodal recommender model and the management of its checkpoints.

### `scripts/train.py`

This is the main script for training the multimodal recommender model.

* **Description**: Orchestrates the entire model training process, including loading data, initializing the model, running training and validation loops, and saving checkpoints.
* **Arguments**:
    * `--config <path>`: Path to the configuration file. (Default: `configs/simple_config.yaml`)
    * `--resume <path>`: Path to a checkpoint file to resume training from.
    * `--device <str>`: Device to use for training (`cuda` or `cpu`). (Default: `cuda` if available, else `cpu`)
    * `--use_wandb`: Enable Weights & Biases logging for experiment tracking.
    * `--wandb_project <str>`: Weights & Biases project name. (Default: `MultimodalRecommender`)
    * `--wandb_entity <str>`: Weights & Biases entity (username or team).
    * `--wandb_run_name <str>`: Weights & Biases run name for this training.
    * `--verbose`: Enable verbose output during training.
* **Example**:
    ```bash
    python scripts/train.py --config configs/simple_config.yaml --device cuda --use_wandb
    ```

### `scripts/extract_encoders.py`

This script extracts and saves the LabelEncoders for user and item IDs.

* **Description**: Reads the complete processed interaction and item data to create and fit `LabelEncoder` objects, mapping unique user and item IDs to integer indices. These are essential for model embeddings.
* **Arguments**:
    * `--config <path>`: Path to the configuration file used for training. (Required)
* **Example**:
    ```bash
    python scripts/extract_encoders.py --config configs/simple_config.yaml
    ```

### `scripts/checkpoint_manager.py`

This utility helps organize and manage saved model checkpoints.

* **Description**: Provides functionalities to list, organize, and summarize model checkpoints. It helps maintain a structured checkpoint directory by sorting models into subdirectories based on their specific configurations.
* **Subcommands**:
    * `list`: Lists all checkpoints and their organization status.
        * `--checkpoint-dir <path>`: Checkpoint directory to scan. (Default: `models/checkpoints`)
        * **Example**: `python scripts/checkpoint_manager.py list`
    * `organize`: Automatically organizes checkpoints by model combination.
        * `--checkpoint-dir <path>`: Checkpoint directory to organize. (Default: `models/checkpoints`)
        * `--dry-run`: Show what would be done without moving files.
        * **Example**: `python scripts/checkpoint_manager.py organize --dry-run`
    * `organize-manual`: Manually organizes checkpoints with unknown model combinations via an interactive prompt.
        * `--checkpoint-dir <path>`: Checkpoint directory. (Default: `models/checkpoints`)
        * **Example**: `python scripts/checkpoint_manager.py organize-manual`
    * `info`: Creates a JSON file with checkpoint information.
        * `--checkpoint-dir <path>`: Checkpoint directory. (Default: `models/checkpoints`)
        * **Example**: `python scripts/checkpoint_manager.py info`

## 3. Evaluation and Recommendation

These commands are used to evaluate your trained model and generate recommendations.

### `scripts/evaluate.py`

This script evaluates the performance of trained recommender models.

* **Description**: Loads a trained model, initializes a specified recommender, loads test datasets, executes a specified evaluation task (e.g., retrieval, ranking), and reports performance metrics.
* **Arguments**:
    * `--config <path>`: Path to the configuration file. (Default: `configs/simple_config.yaml`)
    * `--test_data <path>`: Path to the test data CSV file. (Required)
    * `--train_data <path>`: Path to the training data CSV file, used for user history in filtering seen items.
    * `--output <path>`: Path to save evaluation results JSON. (Default: `evaluation_results.json`)
    * `--device <str>`: Device for evaluation (`cuda` or `cpu`). (Default: `cuda` if available, else `cpu`)
    * `--recommender_type <str>`: Type of recommender to evaluate. Options: `multimodal`, `random`, `popularity`, `item_knn`, `user_knn`. (Default: `multimodal`)
    * `--eval_task <str>`: Evaluation task to perform. Options: `retrieval`, `ranking`. (Default: `retrieval`)
    * `--save_predictions <path>`: Path to save user-level predictions.
    * `--warmup_recommender_cache`: Warm-up the Recommender's feature cache before evaluation.
    * `--num_workers <int>`: Number of parallel workers for evaluation. (Default: 4)
    * `--use_sampling`: Enable negative sampling for faster retrieval evaluation. (Default: True)
    * `--no_sampling`: Disable negative sampling. Overrides `--use_sampling`.
    * `--num_negatives <int>`: Number of negative samples per positive item for retrieval tasks. (Default: 100)
    * `--sampling_strategy <str>`: Negative sampling strategy. Options: `random`, `popularity`, `popularity_inverse`. (Default: `random`)
    * `--checkpoint_name <str>`: Name of the checkpoint file to load (e.g., `best_model.pth`). (Default: `best_model.pth`)
* **Examples**:
    ```bash
    python scripts/evaluate.py --config configs/simple_config.yaml --test_data data/splits/split_tiny/test.csv --train_data data/splits/split_tiny/train.csv --recommender_type multimodal --eval_task retrieval --device cuda
    python scripts/evaluate.py --config configs/simple_config.yaml --test_data data/splits/split_tiny/test.csv --recommender_type popularity
    ```

### `scripts/generate_recommendations.py`

This script generates personalized recommendations using a trained model.

* **Description**: Loads a trained multimodal recommender model and associated data to generate top-K recommendations for a specified set of users.
* **Arguments**:
    * `--config <path>`: Path to the configuration file. (Default: `configs/simple_config.yaml`)
    * `--users <user_id1> [<user_id2> ...]`: A list of user IDs to generate recommendations for.
    * `--user_file <path>`: Path to a file containing user IDs, one per line.
    * `--sample_users <int>`: Number of random users to sample from the dataset for recommendations.
    * `--use_diversity`: Use a diversity-aware recommendation algorithm (if implemented).
    * `--output <filename>`: Name of the output JSON file. (Default: `recommendations.json`)
    * `--device <str>`: Device for inference (`cuda` or `cpu`). (Default: `cuda` if available, else `cpu`)
* **Examples**:
    ```bash
    python scripts/generate_recommendations.py --config configs/simple_config.yaml --users u1 u2 u3
    python scripts/generate_recommendations.py --config configs/simple_config.yaml --sample_users 10 --output sampled_recs.json
    ```

## 4. Utility Commands

These commands provide general utility functions for the system.

### `scripts/cache.py`

This script offers command-line tools for managing multimodal feature caches.

* **Description**: Provides a set of tools to interact with the feature caches generated during data preprocessing and training. It allows for listing available caches, viewing their statistics, and clearing them.
* **Subcommands**:
    * `list`: Lists all available feature caches.
        * **Example**: `python scripts/cache.py list`
    * `clear <vision_model>_<language_model>`: Clears the cache for a specific model combination.
        * **Example**: `python scripts/cache.py clear resnet_sentence-bert`
    * `clear --all`: Clears all feature caches.
        * **Example**: `python scripts/cache.py clear --all`
    * `stats <vision_model>_<language_model>`: Shows statistics for a specific cache.
        * **Example**: `python scripts/cache.py stats clip_mpnet`

### `scripts/run_tests.py`

This script is the entry point for running the project's test suite.

* **Description**: Utilizes Python's native `unittest` framework to discover and execute tests. It provides options to run the entire test suite, only unit tests, or a specific test module.
* **Arguments**:
    * `--unit`: Run only unit tests (located in `tests/unit/`).
    * `--test <module_path>`: Run a specific test module (e.g., `tests.unit.src.data.test_data_filter`).
* **Examples**:
    ```bash
    python scripts/run_tests.py                 # Run all tests
    python scripts/run_tests.py --unit          # Run only unit tests
    python scripts/run_tests.py --test tests.unit.src.models.test_multimodal
    ```