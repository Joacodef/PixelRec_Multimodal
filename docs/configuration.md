Okay, I will generate the `configuration.md` file based on the provided YAML files and the existing `configuration.md`.

```markdown
# Configuration Usage Guide

## Overview

This system utilizes YAML-based configuration files to manage model, training, data, and recommendation parameters. Two primary configuration files are provided:

1.  **`configs/simple_config.yaml`**: Contains essential settings for quick setup and common use cases.
2.  **`configs/advanced_config.yaml`**: Offers a comprehensive set of options for detailed experimentation and fine-tuning.

All executable scripts in the `scripts/` directory accept a `--config` argument to specify which configuration file to use.

## Using Configuration Files

### Simple Configuration (`simple_config.yaml`)

For most common tasks and initial experiments, `simple_config.yaml` is recommended. It allows you to quickly adjust core parameters.

**Example Usage:**
```bash
python scripts/train.py --config configs/simple_config.yaml
python scripts/evaluate.py --config configs/simple_config.yaml --test_data <path_to_test_data> --train_data <path_to_train_data> --eval_task retrieval
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

For example, if you use `simple_config.yaml`, parameters related to advanced model architecture details (like `fusion_hidden_dims` or `optimizer_type`) will be automatically set to their default values as defined in `src/config.py`.

## Key Configuration Sections

Below are the main sections found in the configuration files and their purpose:

### `model`
Defines the architecture of the multimodal recommender.
* `vision_model`, `language_model`: Specifies the pre-trained backbones.
* `embedding_dim`: Sets the size of the latent embeddings.
* `use_contrastive`: Enables or disables contrastive learning.
* **Advanced options** (in `advanced_config.yaml`): `freeze_vision`, `freeze_language`, `contrastive_temperature`, `dropout_rate`, `num_attention_heads`, `fusion_hidden_dims`, `fusion_activation`, `use_batch_norm`, `projection_hidden_dim`, `final_activation`, `init_method`.

### `training`
Controls the training process.
* `batch_size`, `learning_rate`, `epochs`, `patience`: Basic training loop parameters.
* **Advanced options** (in `advanced_config.yaml`): `weight_decay`, `gradient_clip`, `num_workers`, `contrastive_weight`, `bce_weight`, `use_lr_scheduler`, `lr_scheduler_type`, `optimizer_type`, and specific optimizer parameters (e.g., `adam_beta1`).

### `data`
Manages data paths, preprocessing, and loading.
* Paths: `item_info_path`, `interactions_path`, `image_folder`, `processed_item_info_path`, `processed_interactions_path`, `split_data_path`, etc.
* `cache_config`: Nested configuration for the `SimpleFeatureCache`.
    * `enabled`: Boolean to turn caching on/off.
    * `max_memory_items`: Maximum items to keep in memory.
    * `cache_directory`: Path for disk-based cache.
    * `use_disk`: Boolean to enable disk persistence.
* `numerical_features_cols`: List of columns to be treated as numerical features.
* **Advanced options** (in `advanced_config.yaml`): `scaler_path`, `processed_image_destination_folder`, `negative_sampling_ratio`, `numerical_normalization_method`, and nested configurations for `text_augmentation`, `offline_image_compression`, `offline_image_validation`, `offline_text_cleaning`, and `splitting`.

### `recommendation`
Parameters for generating recommendations.
* `top_k`: Number of recommendations to generate.
* **Advanced options** (in `advanced_config.yaml`): `filter_seen`, `diversity_weight`, `novelty_weight`, `max_candidates`.

### Root Level
* `checkpoint_dir`: Directory to save model checkpoints.
* `results_dir`: Directory to save evaluation results and other outputs.

## Example Workflows

### Quick Experiment:
1.  Copy `configs/simple_config.yaml` to a new file, e.g., `configs/my_experiment.yaml`.
2.  Edit `configs/my_experiment.yaml` to change a few key parameters (e.g., `model.vision_model`, `training.batch_size`).
3.  Run scripts using your custom config:
    ```bash
    python scripts/train.py --config configs/my_experiment.yaml
    ```

### Detailed Research:
1.  Use `configs/advanced_config.yaml` as a base.
2.  Modify specific architectural, training, or data processing parameters for your experiment.
3.  Run scripts:
    ```bash
    python scripts/train.py --config configs/advanced_config.yaml
    ```
    (Or your modified version of the advanced config).
4.  Ensure to save the exact configuration file used alongside your results for reproducibility. The training script automatically saves the run's configuration to the `results_dir`.

This approach provides flexibility, allowing users to start with simple settings and progressively delve into more complex configurations as needed.
```