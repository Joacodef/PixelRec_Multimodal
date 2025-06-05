# Configuration Usage Guide

## Overview

This system utilizes YAML-based configuration files to manage model, training, data, and recommendation parameters. Two primary configuration files are provided:

1.  **`configs/simple_config.yaml`**: Contains essential settings for quick setup and common use cases.
2.  **`configs/advanced_config.yaml`**: Offers a comprehensive set of options for detailed experimentation and fine-tuning.

All executable scripts in the `scripts/` directory accept a `--config` argument to specify which configuration file to use (defaulting to `configs/simple_config.yaml` if not provided for some scripts like `train.py`).

## Using Configuration Files

### Simple Configuration (`simple_config.yaml`)

For most common tasks and initial experiments, `simple_config.yaml` is recommended. It allows you to quickly adjust core parameters.

**Example Usage:**
```bash
python scripts/train.py --config configs/simple_config.yaml
python scripts/evaluate.py --config configs/simple_config.yaml --test_data data/splits/split_tiny/test.csv --train_data data/splits/split_tiny/train.csv --eval_task retrieval
```

**Customizing `simple_config.yaml`:**
Edit the file to change parameters such as:
* `model.vision_model`: e.g., `clip`, `resnet`, `dino`, `convnext`
* `model.language_model`: e.g., `sentence-bert`, `mpnet`, `bert`, `roberta`
* `model.embedding_dim`: e.g., `64`, `128`, `256`
* `training.batch_size`: Adjust based on available GPU memory.
* `training.learning_rate`: Modify the learning rate.
* `data.item_info_path`, `data.interactions_path`, `data.image_folder`: Specify paths to your data.
* `data.cache_config`: Configure feature caching.

### Advanced Configuration (`advanced_config.yaml`)

For in-depth research and fine-grained control over all parameters, use `advanced_config.yaml`.

**Example Usage:**
```bash
python scripts/train.py --config configs/advanced_config.yaml
```

The `advanced_config.yaml` file allows customization of:
* **Model Architecture**: Fusion layer dimensions, attention heads, activation functions, dropout rates, initialization methods.
* **Training Parameters**: Optimizer types (AdamW, Adam, SGD), learning rate scheduler details, weight decay, loss component weights.
* **Data Processing**: Text augmentation strategies, image compression settings, numerical feature normalization methods, data splitting parameters.
* **Recommendation Settings**: Diversity and novelty weighting, candidate selection limits.

## Configuration Inheritance and Defaults

The system is designed to use default values for parameters that are not explicitly specified in the loaded YAML file. This means you can create minimal configuration files containing only the parameters you wish to change from their defaults. The `Config` class (`src/config.py`) defines these defaults.

For example, if you use `simple_config.yaml`, parameters related to advanced model architecture details (like `model.fusion_hidden_dims` or `training.optimizer_type`) will be automatically set to their default values as defined in `src/config.py`. The `Config.from_yaml` method handles this merging of YAML-specified values with dataclass defaults.

## Key Configuration Sections

Below are the main sections found in the configuration files and their purpose:

### `model`
Defines the architecture of the multimodal recommender.
* `vision_model`: Specifies the pre-trained vision backbone. Options include `clip`, `resnet`, `dino`, `convnext`. The actual Hugging Face model names and expected dimensions are defined in `MODEL_CONFIGS` in `src/config.py`.
* `language_model`: Specifies the pre-trained language backbone. Options include `sentence-bert`, `mpnet`, `bert`, `roberta`. Similar to vision models, details are in `MODEL_CONFIGS`.
* `embedding_dim`: Sets the size of the latent embeddings for users, items, and projected modalities.
* `use_contrastive`: Enables or disables contrastive learning (primarily for vision-text alignment if using CLIP).
* **Advanced options** (more extensively in `advanced_config.yaml`):
    * `freeze_vision`, `freeze_language`: Boolean flags to freeze the weights of the pre-trained backbones.
    * `contrastive_temperature`: Temperature parameter for the contrastive loss.
    * `dropout_rate`: Dropout rate used in various parts of the model (e.g., projection layers, fusion network).
    * `num_attention_heads`: Number of heads in the multi-head self-attention layer for feature fusion.
    * `attention_dropout`: Dropout rate within the multi-head attention mechanism.
    * `fusion_hidden_dims`: A list of integers defining the sizes of hidden layers in the final fusion MLP.
    * `fusion_activation`: Activation function used in the fusion MLP (e.g., `relu`, `gelu`).
    * `use_batch_norm`: Boolean to enable batch normalization in the fusion MLP.
    * `projection_hidden_dim`: Optional integer for an intermediate hidden layer dimension in modality projection layers. If `null` or `None`, a direct projection is used.
    * `final_activation`: Activation function for the final output layer (e.g., `sigmoid`, `tanh`, `none`).
    * `init_method`: Method for initializing embedding layers (e.g., `xavier_uniform`, `kaiming_normal`).

### `training`
Controls the training process.
* `batch_size`: Number of samples per training batch. Adjust based on GPU memory.
* `learning_rate`: Initial learning rate for the optimizer.
* `epochs`: Maximum number of training epochs.
* `patience`: Number of epochs to wait for improvement in validation loss before early stopping.
* **Advanced options** (more extensively in `advanced_config.yaml`):
    * `weight_decay`: L2 regularization factor for the optimizer.
    * `gradient_clip`: Maximum norm for gradient clipping.
    * `num_workers`: Number of worker processes for the DataLoader.
    * `contrastive_weight`: Weight for the contrastive loss component in the total loss.
    * `bce_weight`: Weight for the binary cross-entropy loss component.
    * `use_lr_scheduler`: Boolean to enable a learning rate scheduler.
    * `lr_scheduler_type`: Type of scheduler (e.g., `reduce_on_plateau`, `cosine`, `step`).
    * `lr_scheduler_patience`, `lr_scheduler_factor`, `lr_scheduler_min_lr`: Parameters specific to the chosen scheduler.
    * `optimizer_type`: Type of optimizer (e.g., `adamw`, `adam`, `sgd`).
    * `adam_beta1`, `adam_beta2`, `adam_eps`: Parameters for Adam-based optimizers.

### `data`
Manages data paths, preprocessing, and loading.
* **Paths**:
    * `item_info_path`, `interactions_path`, `image_folder`: Paths to raw data.
    * `processed_item_info_path`, `processed_interactions_path`: Paths where preprocessed data will be saved/loaded from.
    * `scaler_path`: Path to save/load the numerical feature scaler.
    * `processed_image_destination_folder`: Directory where processed (e.g., validated, compressed) images are stored.
    * `split_data_path`: Base directory for train/val/test splits.
    * `train_data_path`, `val_data_path`, `test_data_path`: Paths to specific split files.
* `cache_config`: Nested configuration for the `SimpleFeatureCache`.
    * `enabled`: Boolean to turn item feature caching on/off.
    * `max_memory_items`: Maximum number of items to keep in the in-memory LRU cache.
    * `cache_directory`: **Base directory** for feature caches. Model-specific subdirectories (e.g., `resnet_sentence-bert/`) will be created under this path by `SimpleFeatureCache`. For example, if `cache_directory` is `cache/`, features for ResNet vision and Sentence-BERT language models will be stored in `cache/resnet_sentence-bert/`.
    * `use_disk`: Boolean to enable disk persistence for the cached features.
* `numerical_features_cols`: List of column names in `item_info_df` to be treated as numerical features.
* **Advanced options** (more extensively in `advanced_config.yaml`):
    * `negative_sampling_ratio`: Ratio of negative samples to positive samples for training.
    * `numerical_normalization_method`: Method for scaling numerical features (e.g., `standardization`, `min_max`, `log1p`, `none`).
    * `text_augmentation`: Nested configuration for text augmentation during training.
        * `enabled`, `augmentation_type`, `delete_prob`, `swap_prob`.
    * `offline_image_compression`: Nested configuration for image compression during preprocessing.
        * `enabled`, `compress_if_kb_larger_than`, `target_quality`, `resize_if_pixels_larger_than`, `resize_target_longest_edge`.
    * `offline_image_validation`: Nested configuration for image validation.
        * `check_corrupted`, `min_width`, `min_height`, `allowed_extensions`.
    * `offline_text_cleaning`: Nested configuration for text cleaning.
        * `remove_html`, `normalize_unicode`, `to_lowercase`.
    * `splitting`: Nested configuration for data splitting.
        * `random_state`, `train_final_ratio`, `val_final_ratio`, `test_final_ratio`, `min_interactions_per_user`, `min_interactions_per_item`, `validate_no_leakage`.

### `recommendation`
Parameters for generating recommendations during inference.
* `top_k`: Number of recommendations to generate per user.
* **Advanced options** (in `advanced_config.yaml`):
    * `filter_seen`: Boolean to filter out items previously interacted with by the user.
    * `diversity_weight`, `novelty_weight`: Weights for diversity and novelty in reranking (Note: actual implementation of diverse/novel recommendations needs to be present in `Recommender` class).
    * `max_candidates`: Maximum candidate items to consider before ranking.

### Root Level
* `checkpoint_dir`: Directory to save model checkpoints (e.g., `best_model.pth`, `final_model.pth`) and encoders.
* `results_dir`: Directory to save evaluation results, training metadata, figures, and other outputs.

## Example Workflows

### Quick Experiment:
1.  Copy `configs/simple_config.yaml` to a new file, e.g., `configs/my_experiment.yaml`.
2.  Edit `configs/my_experiment.yaml` to change a few key parameters (e.g., `model.vision_model`, `training.batch_size`).
3.  Run scripts using your custom config:
    ```bash
    python scripts/preprocess_data.py --config configs/my_experiment.yaml
    python scripts/create_splits.py --config configs/my_experiment.yaml
    python scripts/train.py --config configs/my_experiment.yaml
    ```

### Detailed Research:
1.  Use `configs/advanced_config.yaml` as a base or create a copy.
2.  Modify specific architectural, training, or data processing parameters for your experiment.
3.  Run scripts:
    ```bash
    python scripts/preprocess_data.py --config configs/advanced_config.yaml
    python scripts/create_splits.py --config configs/advanced_config.yaml
    python scripts/train.py --config configs/advanced_config.yaml
    ```
4.  Ensure to save the exact configuration file used alongside your results for reproducibility. The training script automatically saves the run's configuration to the `results_dir`.

