# Comprehensive Configuration Guide

## Overview

This system utilizes YAML-based configuration files to manage all aspects of model architecture, data processing, training, and recommendation generation. This approach provides a flexible and reproducible way to run experiments.

Two primary configuration files are provided:

1.  **`configs/simple_config.yaml`**: Contains the most essential settings for a quick setup and common use cases. It's the best place to start.
2.  **`configs/advanced_config.yaml`**: Offers a comprehensive set of options for detailed experimentation, research, and fine-tuning every component of the pipeline.

All executable scripts in the `scripts/` directory accept a `--config` argument to specify which configuration file to use.

## Configuration Principles

### Inheritance and Defaults

The system is built on a "defaults-first" principle. The core configuration structure is defined by Python dataclasses in `src/config.py`. If a parameter is **not** specified in your YAML file, the system will automatically use the default value defined in these dataclasses.

This means you can create minimal YAML files containing only the parameters you wish to override, making your experimental configurations clean and easy to manage.

### Model-Specific Checkpointing and Caching

To keep experiments organized, the framework automatically saves model checkpoints and feature caches into directories named after the model combination being used.

* **Checkpoints**: Saved under `checkpoint_dir/<vision_model>_<language_model>/`. For example, `models/checkpoints/resnet_sentence-bert/`.
* **Encoders**: User/item ID encoders are shared and saved in `checkpoint_dir/encoders/`.
* **Feature Caches**: Saved under `data.cache_config.cache_directory/<vision_model>_<language_model>/`. For example, `cache/resnet_sentence-bert/`.

---

## Main Configuration Sections

### `model`

Defines the architecture of the multimodal recommender.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`vision_model`** | `string` | The pre-trained vision backbone to use. | `'resnet'` |
| | | *Options*: `'clip'`, `'resnet'`, `'dino'`, `'convnext'` | |
| **`language_model`** | `string` | The pre-trained language backbone for text features. | `'sentence-bert'`|
| | | *Options*: `'sentence-bert'`, `'mpnet'`, `'bert'`, `'roberta'` | |
| **`embedding_dim`** | `int` | The size of the latent embeddings for users, items, and projected modalities. | `64` |
| `use_contrastive` | `bool` | Enables or disables CLIP-style contrastive learning. **Note**: This is most effective when `vision_model` is `'clip'`. | `True` |
| `freeze_vision` | `bool` | If `True`, the weights of the pre-trained vision backbone are frozen and not updated during training. | `True` |
| `freeze_language` | `bool` | If `True`, the weights of the pre-trained language backbone are frozen. | `True` |
| `contrastive_temperature`| `float`| The temperature parameter for the contrastive loss function. A learnable temperature is used in the model. | `0.07` |
| `dropout_rate` | `float`| Dropout rate used in projection layers and the fusion network for regularization. | `0.3` |
| `num_attention_heads`| `int` | Number of heads in the multi-head self-attention layer that fuses the different feature embeddings. | `4` |
| `attention_dropout` | `float`| Dropout rate specifically for the attention mechanism. | `0.1` |
| `fusion_hidden_dims`| `List[int]`| A list of integers defining the sizes of the hidden layers in the final fusion MLP. | `[512, 256, 128]` |
| `fusion_activation` | `string` | Activation function for the fusion MLP. *Options*: `'relu'`, `'gelu'`, `'tanh'`, `'leaky_relu'`, `'silu'`. | `'relu'` |
| `use_batch_norm` | `bool` | If `True`, adds a batch normalization layer after each hidden layer in the fusion MLP. | `True` |
| `projection_hidden_dim`| `int`| An optional intermediate hidden layer dimension for the modality projection layers. If `null`, a direct linear projection is used. | `null` |
| `final_activation` | `string`| The activation function for the final output layer. *Options*: `'sigmoid'`, `'tanh'`, `'none'`. | `'sigmoid'` |
| `init_method` | `string`| Method for initializing the weights of user and item embedding layers. *Options*: `'xavier_uniform'`, `'xavier_normal'`, `'kaiming_uniform'`, `'kaiming_normal'`. | `'xavier_uniform'` |

### `training`

Controls the entire training process.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`batch_size`** | `int` | Number of samples per training batch. Adjust based on available GPU memory. | `64` |
| **`learning_rate`** | `float`| The initial learning rate for the optimizer. | `0.001` |
| **`epochs`** | `int` | The maximum number of training epochs to run. | `30` |
| `patience` | `int` | Number of epochs to wait for improvement in validation loss before early stopping is triggered. | `10` |
| `weight_decay` | `float`| The L2 regularization factor applied by the optimizer. | `0.01` |
| `gradient_clip`| `float`| The maximum norm for gradient clipping to prevent exploding gradients. | `1.0` |
| `num_workers` | `int` | Number of worker processes for the DataLoader. Set to `0` for debugging, `4` or `8` for performance. | `8` |
| `contrastive_weight`| `float`| The weight of the contrastive loss component in the total loss calculation. | `0.1` |
| `bce_weight` | `float`| The weight of the binary cross-entropy loss component in the total loss. | `1.0` |
| `use_lr_scheduler` | `bool` | If `True`, enables a learning rate scheduler to adjust the learning rate during training. | `True` |
| `lr_scheduler_type`| `string`| The type of learning rate scheduler to use. *Options*: `'reduce_on_plateau'`, `'cosine'`, `'step'`. | `'reduce_on_plateau'` |
| `lr_scheduler_patience`| `int` | For `reduce_on_plateau`: number of epochs with no improvement after which learning rate will be reduced. For `step`: how many epochs before stepping the LR. | `2` |
| `lr_scheduler_factor`| `float`| The factor by which the learning rate will be reduced. `new_lr = lr * factor`. | `0.5` |
| `lr_scheduler_min_lr`| `float`| The lower bound on the learning rate. | `1e-6` |
| `optimizer_type` | `string`| The optimizer algorithm to use. *Options*: `'adamw'`, `'adam'`, `'sgd'`. | `'adamw'` |
| `adam_beta1` | `float`| The `beta1` parameter for Adam and AdamW optimizers. | `0.9` |
| `adam_beta2` | `float`| The `beta2` parameter for Adam and AdamW optimizers. | `0.999`|
| `adam_eps` | `float`| The `epsilon` parameter for Adam and AdamW for numerical stability. | `1e-8` |

### `data`

Manages data paths, preprocessing rules, caching, and data loading settings.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`item_info_path`**| `string` | Path to the **raw** CSV file containing item metadata. | `data/raw/item_info/item_info_sample.csv`|
| **`interactions_path`**| `string`| Path to the **raw** CSV file containing user-item interactions. |`data/raw/interactions/interactions_sample.csv`|
| **`image_folder`**| `string` | Path to the directory containing **raw** item images. | `data/raw/images` |
| `processed_item_info_path` | `string` | Path where the **processed** item metadata CSV will be saved/loaded. | `data/processed/item_info.csv` |
| `processed_interactions_path`|`string` | Path where the **processed** interactions CSV will be saved/loaded. |`data/processed/interactions.csv`|
| `scaler_path`|`string` | Path to save or load the fitted numerical feature scaler (a `.pkl` file). |`data/processed/numerical_scaler.pkl`|
| `processed_image_destination_folder`|`string`| Directory where validated and optionally compressed images are stored by the preprocessing script. | `data/processed/images`|
| `split_data_path` | `string` | Base directory where train/validation/test splits will be created. |`data/splits/split_tiny`|
| `train_data_path`| `string` | Full path to the training data CSV file. |`data/splits/split_tiny/train.csv`|
| `val_data_path` | `string` | Full path to the validation data CSV file. |`data/splits/split_tiny/val.csv`|
| `test_data_path`| `string` | Full path to the testing data CSV file. |`data/splits/split_tiny/test.csv`|
| **`cache_config`**| `object` | A nested object for configuring the `SimpleFeatureCache`. | See below. |
| `enabled` | `bool` | Turns item feature caching on or off. Caching significantly speeds up training after the first epoch. | `True` |
| `max_memory_items`| `int` | Maximum number of items to keep in the in-memory LRU cache. | `1000` |
| `cache_directory`| `string` | **Base directory** for feature caches. Model-specific subdirectories will be created here. | `'cache'` |
| `use_disk` | `bool` | If `True`, the feature cache will be persisted to disk, allowing it to be reused across different runs. | `False` |
| **`numerical_features_cols`**|`List[string]`| A list of column names in `item_info.csv` to be treated as numerical features. | (List of 7 features) |
| `negative_sampling_ratio`| `float`| The ratio of negative samples to positive samples to generate during training. `1.0` means one negative sample for each positive interaction. | `1.0` |
| `numerical_normalization_method`|`string`| Method for scaling numerical features. *Options*: `'standardization'`, `'min_max'`, `'log1p'`, `'none'`.|`'standardization'`|
| `text_augmentation` | `object`| Nested configuration for text augmentation during training. | See below. |
| `enabled` | `bool` | If `True`, applies augmentation to text features during training. | `False` |
| `augmentation_type`| `string`| Type of augmentation. *Options*: `'random_delete'`, `'random_swap'`. |`'random_delete'`|
| `delete_prob` | `float`| Probability of deleting each word for `random_delete`. | `0.1` |
| `swap_prob` | `float`| Probability of swapping adjacent words for `random_swap`. | `0.1` |
| `offline_image_compression`| `object` | Nested configuration for image compression during preprocessing. | See below. |
| `enabled` | `bool` | Enable/disable image compression. | `True` |
| `compress_if_kb_larger_than`| `int` | Only compress images larger than this size in kilobytes. | `500` |
| `target_quality` | `int` | The quality setting for JPEG compression (1-95). | `85` |
| `resize_if_pixels_larger_than`|`List[int]`| A `[width, height]` list. Resize image if its dimensions exceed these values. | `[2048, 2048]` |
| `resize_target_longest_edge`|`int` | When resizing, the longest edge of the image will be scaled down to this size. | `1024`|
| `offline_image_validation`| `object` | Nested configuration for validating images during preprocessing. | See below. |
| `check_corrupted` | `bool` | If `True`, the system will attempt to identify and discard corrupted image files. |`True` |
| `min_width` | `int` | Minimum allowed width for an image in pixels. | `64` |
| `min_height` | `int` | Minimum allowed height for an image in pixels. | `64` |
| `allowed_extensions`|`List[string]`| A list of valid image file extensions. | `['.jpg', '.jpeg', '.png']`|
| `offline_text_cleaning`| `object` | Nested configuration for cleaning text fields during preprocessing. | See below. |
| `remove_html` | `bool` | Remove HTML tags from text fields. | `True` |
| `normalize_unicode`| `bool` | Normalize unicode characters (e.g., converting special characters to standard forms).|`True`|
| `to_lowercase`| `bool` | Convert all text to lowercase. | `True` |
| `splitting` | `object`| Nested configuration for creating train/val/test splits. | See below. |
| `random_state` |`int`| Seed for the random number generator to ensure reproducible splits. | `42` |
| `train_final_ratio`|`float`| The proportion of the data to be used for the training set. |`0.6` |
| `val_final_ratio` |`float`| The proportion of the data for the validation set. |`0.2`|
| `test_final_ratio`|`float`| The proportion of the data for the test set. | `0.2` |
| `min_interactions_per_user`|`int`| The minimum number of interactions a user must have to be included in the dataset after filtering. |`5`|
| `min_interactions_per_item`|`int`| The minimum number of interactions an item must have to be included. |`5`|
| `validate_no_leakage`|`bool`| If `True`, prints statistics about user/item overlap between splits. | `True` |

### `recommendation`

Parameters for generating recommendations during inference.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`top_k`** | `int` | The number of recommendations to generate per user. | `50` |
| `filter_seen` | `bool` | If `True`, filters out items that the user has previously interacted with from the final recommendations. | `True` |
| `diversity_weight`| `float`| Weight for a diversity-promoting algorithm during reranking. **Note**: Requires a corresponding implementation in the `Recommender` class. | `0.3` |
| `novelty_weight`| `float`| Weight for a novelty-promoting algorithm. **Note**: Requires implementation. | `0.2` |
| `max_candidates`| `int` | The maximum number of candidate items to score before final ranking. Useful for speeding up inference on very large item catalogs. | `1000`|

### Root Level Parameters

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `checkpoint_dir`| `string`| The base directory to save model checkpoints and encoders. |`'models/checkpoints'`|
| `results_dir` | `string` | The base directory to save all outputs, such as evaluation results, training metadata, and figures. | `'results'`|