# Configuration File Reference

The entire behavior of the training and evaluation pipeline is controlled by a single YAML configuration file. This document provides a comprehensive reference for all available parameters.

## Top-Level Parameters

These are the main sections of the configuration file.

| Key                 | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| **`paths`** | Defines all input and output paths for data, models, and results.           |
| **`dataset`** | Specifies filenames and settings related to the dataset being used.         |
| **`modality`** | Configures which data modalities to use (text, image, etc.) and their sources. |
| **`training`** | Contains all hyperparameters and settings for the model training process.   |
| **`evaluation`** | Defines settings for the model evaluation process, including metrics.       |
| **`model`** | Configures the architecture of the multimodal recommender model itself.     |
| **`hyper_search`** | Contains settings for running a hyperparameter search with Optuna.          |

---

### `paths`

This section defines all the necessary paths for the project.

| Parameter                  | Type   | Description                                                     |
| -------------------------- | ------ | --------------------------------------------------------------- |
| **`data_path`** | string | Path to the directory containing the raw data CSV files.        |
| **`images_path`** | string | Path to the directory containing the item image files.          |
| **`data_preprocessed_path`**| string | Path where the preprocessed data will be saved.                 |
| **`split_path`** | string | Path where the data splits (train/val/test) will be saved.      |
| **`encoders_path`** | string | Path where the fitted encoders for categorical features will be saved. |
| **`cache_path`** | string | Path to the directory for caching precomputed embeddings.       |
| **`output_path`** | string | Root directory where model checkpoints and results will be saved. |

---

### `dataset`

| Parameter                    | Type    | Description                                                                     |
| ---------------------------- | ------- | ------------------------------------------------------------------------------- |
| **`interactions_filename`** | string  | Filename of the interactions CSV file (e.g., "interactions.csv").               |
| **`item_info_filename`** | string  | Filename of the item metadata CSV file (e.g., "item_info.csv").                 |
| **`min_user_interactions`** | integer | Minimum number of interactions a user must have to be included.                 |
| **`min_item_interactions`** | integer | Minimum number of interactions an item must have to be included.                |
| **`image_format`** | string  | The file extension for the images (e.g., 'jpg', 'png').                         |
| **`split_strategy`** | string  | Method for splitting data. Options: `random`, `time`. Default: `random`.         |
| **`test_size`** | float   | Proportion of the dataset to include in the test split. Default: `0.2`.         |

---

### `modality`

This section defines the input features for the model.

| Parameter           | Type          | Description                                                    |
| ------------------- | ------------- | -------------------------------------------------------------- |
| **`text_cols`** | list[string]  | List of column names from `item_info.csv` to be used as text features. |
| **`categorical_cols`**| list[string]  | List of column names to be treated as categorical features.    |
| **`numerical_cols`**| list[string]  | List of column names to be treated as numerical features.      |
| **`use_item_id`** | boolean       | Whether to use the item ID as a feature. Default: `false`.     |
| **`use_user_id`** | boolean       | Whether to use the user ID as a feature. Default: `true`.      |
| **`use_image`** | boolean       | Whether to use image features. Default: `true`.                |

---

### `training`

This section controls the training process.

| Parameter                | Type          | Description                                                                 |
| ------------------------ | ------------- | --------------------------------------------------------------------------- |
| **`epochs`** | integer       | The total number of training epochs.                                        |
| **`batch_size`** | integer       | The number of samples per batch.                                            |
| **`learning_rate`** | float         | The learning rate for the optimizer.                                        |
| **`optimizer`** | string        | The optimizer to use. Options: `adam`, `sgd`. Default: `adam`.              |
| **`num_workers`** | integer       | The number of worker processes for data loading. Default: `4`.              |
| **`gpus`** | integer       | The number of GPUs to use for training. `0` for CPU. Default: `1`.          |
| **`use_cache`** | boolean       | Whether to use precomputed and cached features to speed up training.        |
| **`image_augmentation`** | boolean       | Whether to apply random augmentations to images during training.            |
| **`text_model_name`** | string        | The name of the pre-trained sentence-transformer model for text features.    |
| **`vision_model_name`** | string        | The name of the pre-trained vision transformer (TIMM) model for images.     |

---

### `evaluation`

| Parameter              | Type        | Description                                                              |
| ---------------------- | ----------- | ------------------------------------------------------------------------ |
| **`k`** | integer     | The number of items to recommend for calculating top-K metrics.          |
| **`metrics`** | list[string]| List of metrics to compute. E.g., `['Precision@k', 'Recall@k', 'NDCG@k']`.|
| **`recommender_type`** | string      | Type of recommender for evaluation. Options: `multimodal`, `random`, `popularity`. Default: `multimodal`.|

---

### `model`

This section defines the neural network architecture.

| Parameter                   | Type          | Description                                                              |
| --------------------------- | ------------- | ------------------------------------------------------------------------ |
| **`embedding_dim`** | integer       | The dimensionality of the common embedding space for all modalities.     |
| **`fusion_type`** | string        | The mechanism to fuse multimodal features. Options: `concatenate`, `attention`, `gated`. |
| **`dropout_rate`** | float         | The dropout rate to apply for regularization.                            |
| **`id_embedding_dim`** | integer       | The embedding dimension for user and item IDs if used as features.       |
| **`use_contrastive_loss`** | boolean       | Whether to add a contrastive loss term to align vision and text.         |
| **`contrastive_loss_weight`**| float         | The weight to apply to the contrastive loss component.                   |

---

### `hyper_search`

This section configures the hyperparameter search.

| Parameter     | Type    | Description                                                               |
| ------------- | ------- | ------------------------------------------------------------------------- |
| **`n_trials`**| integer | The number of optimization trials to run.                                 |
| **`direction`**| string  | The optimization direction. Options: `maximize` or `minimize`.            |
| **`metric`** | string  | The evaluation metric to optimize (e.g., `NDCG@k`).                       |