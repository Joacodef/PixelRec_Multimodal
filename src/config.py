# src/config.py
"""
Defines the configuration structure for the multimodal recommender system.

This module utilizes Python's dataclasses to create a hierarchical and type-safe
configuration system. It consolidates all tunable parameters—from model
architecture and training settings to data paths and preprocessing rules—into
a single, manageable structure. The configuration can be easily loaded from
and saved to YAML files, promoting reproducibility and simplifying experimentation.
"""
from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import Optional, Dict, Any, List, Union, Tuple
import yaml
from pathlib import Path

# A dictionary that centralizes the configurations for various pre-trained models.
# It stores the Hugging Face model identifier and the expected output dimension for each model.
MODEL_CONFIGS = {
    'vision': {
        'clip': {'name': 'openai/clip-vit-base-patch32', 'dim': 768, 'text_dim': 512},
        'dino': {'name': 'facebook/dinov2-base', 'dim': 768},
        'resnet': {'name': 'microsoft/resnet-50', 'dim': 2048},
        'convnext': {'name': 'facebook/convnext-base-224', 'dim': 1024}
    },
    'language': {
        'sentence-bert': {'name': 'sentence-transformers/all-MiniLM-L6-v2', 'dim': 384},
        'mpnet': {'name': 'sentence-transformers/all-mpnet-base-v2', 'dim': 768},
        'bert': {'name': 'bert-base-uncased', 'dim': 768},
        'roberta': {'name': 'roberta-base', 'dim': 768}
    }
}

@dataclass
class ModelConfig:
    """Specifies the architecture and parameters of the recommender model."""
    # The pre-trained vision model to use as a feature extractor.
    vision_model: Optional[str] = 'resnet'
    # The pre-trained language model to use for text feature extraction.
    language_model: Optional[str] = 'sentence-bert'
    # The dimensionality of the latent embeddings for users, items, and projected features.
    embedding_dim: int = 64
    # The method for combining multimodal features ('concatenate', 'attention', 'gated').
    fusion_type: str = 'concatenate'
    # If True, enables an additional contrastive loss to align vision and text representations.
    use_contrastive: bool = True
    
    # If True, the weights of the pre-trained vision model are not updated during training.
    freeze_vision: bool = True
    # If True, the weights of the pre-trained language model are not updated during training.
    freeze_language: bool = True
    # The temperature parameter for scaling logits in the contrastive loss function.
    contrastive_temperature: float = 0.07
    # The dropout rate applied for regularization in the projection and fusion layers.
    dropout_rate: float = 0.3
    # The number of parallel attention heads in the self-attention fusion mechanism.
    num_attention_heads: int = 4
    # The dropout rate applied within the attention mechanism.
    attention_dropout: float = 0.1
    # A list of integers defining the dimensions of the hidden layers in the final fusion network.
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    # The activation function used in the fusion network.
    fusion_activation: str = 'relu'
    # If True, a batch normalization layer is added after each hidden layer in the fusion network.
    use_batch_norm: bool = True
    # The dimension of an optional intermediate hidden layer in the modality projection networks.
    projection_hidden_dim: Optional[int] = None
    # The activation function for the final output layer of the model.
    final_activation: str = 'sigmoid'
    # The method used to initialize the weights of the user and item embedding layers.
    init_method: str = 'xavier_uniform'

@dataclass
class TrainingConfig:
    """Specifies the parameters for the model training process."""
    # The number of samples per batch fed to the model during training.
    batch_size: int = 64
    # The initial learning rate for the optimizer.
    learning_rate: float = 0.001
    # The maximum number of full passes through the training dataset.
    epochs: int = 30
    # The number of epochs to wait for improvement in validation loss before stopping the training early.
    patience: int = 10
    
    # The L2 regularization factor applied by the optimizer to prevent overfitting.
    weight_decay: float = 0.01
    # The maximum norm for gradients, used for clipping to prevent exploding gradients.
    gradient_clip: float = 1.0
    # The number of subprocesses to use for data loading.
    num_workers: int = 8
    # The weight of the contrastive loss component in the final combined loss.
    contrastive_weight: float = 0.1
    # The weight of the binary cross-entropy loss component in the final combined loss.
    bce_weight: float = 1.0
    # If True, a learning rate scheduler will be used to adjust the learning rate during training.
    use_lr_scheduler: bool = True
    # The type of learning rate scheduler to use.
    lr_scheduler_type: str = 'reduce_on_plateau'
    # The number of epochs with no improvement after which the learning rate will be reduced.
    lr_scheduler_patience: int = 2
    # The factor by which the learning rate will be reduced (new_lr = lr * factor).
    lr_scheduler_factor: float = 0.5
    # A lower bound on the learning rate for all parameter groups.
    lr_scheduler_min_lr: float = 1e-6
    # The optimization algorithm to use for training.
    optimizer_type: str = 'adamw'
    # The beta1 hyperparameter for the Adam and AdamW optimizers.
    adam_beta1: float = 0.9
    # The beta2 hyperparameter for the Adam and AdamW optimizers.
    adam_beta2: float = 0.999
    # The epsilon hyperparameter for the Adam and AdamW optimizers for numerical stability.
    adam_eps: float = 1e-8

@dataclass
class SimpleCacheConfig:
    """Configures the caching behavior for processed item features."""
    # If True, enables the feature caching system.
    enabled: bool = True
    # The maximum number of items to hold in the in-memory LRU cache.
    max_memory_items: int = 1000
    # The base directory where feature caches will be stored.
    cache_directory: str = 'data/cache/features'
    # If True, persists the feature cache to disk for reuse across sessions.
    use_disk: bool = False

@dataclass
class TextAugmentationConfig:
    """Configures text augmentation applied during the training data loading."""
    # If True, enables the application of text augmentation.
    enabled: bool = False
    # The type of text augmentation to apply.
    augmentation_type: str = 'random_delete'
    # The probability of deleting a word during 'random_delete' augmentation.
    delete_prob: float = 0.1
    # The probability of swapping adjacent words during 'random_swap' augmentation.
    swap_prob: float = 0.1

@dataclass
class ImageAugmentationConfig:
    """Configuration for image augmentation during training"""
    enabled: bool = False
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    random_crop: bool = True
    crop_scale: List[float] = field(default_factory=lambda: [0.8, 1.0])
    horizontal_flip: bool = True
    rotation_degrees: float = 10
    gaussian_blur: bool = True
    blur_kernel_size: List[int] = field(default_factory=lambda: [5, 9])
    gaussian_noise: bool = False
    noise_std: float = 0.01

    def __post_init__(self):
        """Validate augmentation parameters after initialization."""
        if self.brightness < 0:
            raise ValueError("Brightness factor must be non-negative.")
        if self.contrast < 0:
            raise ValueError("Contrast factor must be non-negative.")
        if self.saturation < 0:
            raise ValueError("Saturation factor must be non-negative.")
        if not (0 <= self.hue <= 0.5):
            raise ValueError("Hue factor must be between 0 and 0.5.")
        if self.random_crop and (not (0 < self.crop_scale[0] <= self.crop_scale[1] <= 1.0)):
            raise ValueError("Invalid crop_scale. Must be [min, max] with 0 < min <= max <= 1.0.")

@dataclass
class ImageValidationConfig:
    """Configures the validation rules for images during offline preprocessing."""
    # If True, attempts to identify and filter out corrupted or unreadable image files.
    check_corrupted: bool = True
    # The minimum allowed width for an image in pixels.
    min_width: int = 64
    # The minimum allowed height for an image in pixels.
    min_height: int = 64
    # A list of valid image file extensions to consider during processing.
    allowed_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png'])

@dataclass
class OfflineTextCleaningConfig:
    """Configures text cleaning rules applied during offline preprocessing."""
    # If True, removes HTML tags from all text fields.
    remove_html: bool = True
    # If True, normalizes Unicode characters to a standard form.
    normalize_unicode: bool = True
    # If True, converts all text to lowercase.
    to_lowercase: bool = True

@dataclass
class DataSplittingConfig:
    """Configures the strategy for splitting data into train, validation, and test sets."""
    # The splitting strategy to use.
    strategy: str = 'user' 
    # The column to use for stratification (only applicable for 'stratified_temporal' strategy).
    stratify_by: Optional[str] = None
    # The minimum number of appearances for a tag to not be grouped.
    tag_grouping_threshold: Optional[int] = None
    # The seed for the random number generator to ensure reproducible splits.
    random_state: int = 42
    # The proportion of the data to allocate to the final training set.
    train_final_ratio: float = 0.6
    # The proportion of the data to allocate to the final validation set.
    val_final_ratio: float = 0.2
    # The proportion of the data to allocate to the final test set.
    test_final_ratio: float = 0.2
    # The minimum number of interactions a user must have to be included in the dataset.
    min_interactions_per_user: int = 5
    # The minimum number of interactions an item must have to be included in the dataset.
    min_interactions_per_item: int = 5
    # If True, prints statistics about user and item overlap between the generated splits.
    validate_no_leakage: bool = True

@dataclass
class OfflineImageCompressionConfig:
    """Configures image compression rules applied during offline preprocessing."""
    # If True, enables the image compression process.
    enabled: bool = True
    # Only images larger than this size in kilobytes will be considered for compression.
    compress_if_kb_larger_than: int = 500
    # The quality setting for JPEG compression, on a scale of 1-95.
    target_quality: int = 85
    # A list [width, height] specifying the dimension thresholds for resizing.
    resize_if_pixels_larger_than: Optional[List[int]] = field(default_factory=lambda: [2048, 2048])
    # When resizing, the longest edge of the image will be scaled down to this size in pixels.
    resize_target_longest_edge: Optional[int] = 1024


# Add this dataclass to src/config.py after the RecommendationConfig dataclass

@dataclass
class HyperparameterSearchConfig:
    """Configures Optuna hyperparameter optimization settings."""
    # Number of Optuna trials to run
    n_trials: int = 100
    
    # Name for the Optuna study (auto-generated if None)
    study_name: Optional[str] = None
    
    # Database URL for distributed optimization (e.g., 'sqlite:///study.db')
    # If None, the study is stored in memory
    storage: Optional[str] = None
    
    # Direction of optimization: 'minimize' or 'maximize'
    direction: str = 'minimize'
    
    # Metric to optimize (e.g., 'val_loss', 'ndcg@10', 'recall@5')
    metric: str = 'val_loss'
    
    # Enable trial pruning based on intermediate values
    enable_pruning: bool = True
    
    # Pruner type: 'median', 'percentile', 'hyperband'
    pruner_type: str = 'median'
    
    # Number of parallel jobs (-1 for all available cores)
    n_jobs: int = 1
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Directory to save trial results
    output_dir: str = 'optuna_trials'
    
    # Search space definition for hyperparameters
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # Training hyperparameters
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-2,
            'log': True
        },
        'batch_size': {
            'type': 'categorical',
            'choices': [16, 32, 64, 128]
        },
        'weight_decay': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-2,
            'log': True
        },
        'patience': {
            'type': 'int',
            'low': 2,
            'high': 10
        },
        'gradient_clip': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0
        },
        
        # Model hyperparameters
        'embedding_dim': {
            'type': 'categorical',
            'choices': [64, 128, 256, 512]
        },
        'fusion_type': {
            'type': 'categorical',
            'choices': ['concatenate', 'attention', 'gated']
        },
        'dropout_rate': {
            'type': 'float',
            'low': 0.1,
            'high': 0.5
        },
        'fusion_hidden_dims': {
            'type': 'categorical',
            'choices': [[256, 128], [512, 256], [128, 64], [256, 128, 64]]
        },
        
        # Loss weights
        'contrastive_weight': {
            'type': 'float',
            'low': 0.0,
            'high': 1.0
        },
        'bce_weight': {
            'type': 'float',
            'low': 0.5,
            'high': 1.0
        },
        
        # Optimizer settings
        'optimizer_type': {
            'type': 'categorical',
            'choices': ['adam', 'adamw', 'sgd']
        },
        'adam_beta1': {
            'type': 'float',
            'low': 0.8,
            'high': 0.99,
            'condition': 'optimizer_type in ["adam", "adamw"]'
        },
        'adam_beta2': {
            'type': 'float',
            'low': 0.9,
            'high': 0.999,
            'condition': 'optimizer_type in ["adam", "adamw"]'
        },
        
        # Learning rate scheduler
        'use_lr_scheduler': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'lr_scheduler_type': {
            'type': 'categorical',
            'choices': ['reduce_on_plateau', 'cosine', 'step'],
            'condition': 'use_lr_scheduler == True'
        },
        'lr_scheduler_factor': {
            'type': 'float',
            'low': 0.1,
            'high': 0.9,
            'condition': 'use_lr_scheduler == True'
        }
    })
    
    # Additional sampler configuration
    sampler_config: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'TPESampler',  # Options: 'TPESampler', 'RandomSampler', 'CmaEsSampler'
        'n_startup_trials': 10,
        'n_ei_candidates': 24,
        'multivariate': False,
        'group': False,
        'warn_independent_sampling': True
    })
    
    # Pruner configuration
    pruner_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_startup_trials': 5,
        'n_warmup_steps': 0,
        'interval_steps': 1,
        # For MedianPruner
        'percentile': 50.0,
        # For HyperbandPruner
        'min_resource': 1,
        'max_resource': 'auto',
        'reduction_factor': 3
    })
    
    # Whether to save intermediate model checkpoints for each trial
    save_trial_checkpoints: bool = False
    
    # Whether to delete unsuccessful trial data to save space
    delete_unsuccessful_trials: bool = True
    
    # Minimum improvement required to update best trial
    min_improvement_threshold: float = 1e-4
    
    # Resume from previous study if it exists
    resume_if_exists: bool = True
    
    # Visualization settings
    create_visualizations: bool = True
    visualization_formats: List[str] = field(default_factory=lambda: ['html', 'png'])
    
    def get_parameter_config(self, param_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Dictionary with parameter configuration
        """
        return self.search_space.get(param_name, {})
    
    def validate(self):
        """Validate the hyperparameter search configuration."""
        valid_directions = ['minimize', 'maximize']
        if self.direction not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}")
        
        valid_pruner_types = ['median', 'percentile', 'hyperband']
        if self.pruner_type not in valid_pruner_types:
            raise ValueError(f"pruner_type must be one of {valid_pruner_types}")
        
        # Validate search space
        for param_name, param_config in self.search_space.items():
            if 'type' not in param_config:
                raise ValueError(f"Parameter {param_name} must have a 'type' field")
            
            param_type = param_config['type']
            if param_type == 'float' or param_type == 'int':
                if 'low' not in param_config or 'high' not in param_config:
                    raise ValueError(f"Parameter {param_name} of type {param_type} must have 'low' and 'high' fields")
            elif param_type == 'categorical':
                if 'choices' not in param_config:
                    raise ValueError(f"Parameter {param_name} of type categorical must have 'choices' field")

@dataclass
class DataConfig:
    """Consolidates all data-related configurations."""
    # Path to the raw CSV file containing item metadata.
    item_info_path: str = 'data/processed/item_info.csv'
    # Path to the raw CSV file containing user-item interactions.
    interactions_path: str = 'data/processed/interactions.csv'
    # Path to the directory containing raw item images.
    image_folder: str = 'data/raw/images'
    # Path where the processed item metadata CSV will be stored.
    processed_item_info_path: str = 'data/processed/item_info.csv'
    # Path where the processed interactions CSV will be stored.
    processed_interactions_path: str = 'data/processed/interactions.csv'
    # Base directory where train, validation, and test splits will be stored.
    split_data_path: str = 'data/splits/split_1'
    # Full path to the final training data CSV file.
    train_data_path: str = 'data/splits/split_1/train.csv'
    # Full path to the final validation data CSV file.
    val_data_path: str = 'data/splits/split_1/val.csv'
    # Full path to the final test data CSV file.
    test_data_path: str = 'data/splits/split_1/test.csv'
    
    # Nested configuration for the feature caching system.
    cache_config: SimpleCacheConfig = field(default_factory=SimpleCacheConfig)
    
    # Path for saving or loading the fitted numerical feature scaler.
    scaler_path: str = 'data/processed/numerical_scaler.pkl'
    # Directory where validated and optionally compressed images are stored.
    processed_image_destination_folder: Optional[str] = 'data/processed/images'
    # The strategy for sampling negative items during training.
    negative_sampling_strategy: str = 'random'
    # The ratio of negative samples to positive samples generated for training.
    negative_sampling_ratio: float = 1.0    
    # The method used for scaling numerical features.
    numerical_normalization_method: str = 'standardization'
    # A list of column names in the item metadata to be used as numerical features.
    numerical_features_cols: List[str] = field(default_factory=lambda: [
        'view_number', 'comment_number', 'thumbup_number',
        'share_number', 'coin_number', 'favorite_number', 'barrage_number'
    ])
    # A list of column names in the item metadata to be used as categorical features.
    categorical_features_cols: List[str] = field(default_factory=lambda: ['tag'])
    
    # Nested configuration for text augmentation.
    text_augmentation: TextAugmentationConfig = field(default_factory=TextAugmentationConfig)
    # Nested configuration for image augmentation.
    image_augmentation: ImageAugmentationConfig = field(default_factory=ImageAugmentationConfig)
    # Nested configuration for offline image compression.
    offline_image_compression: OfflineImageCompressionConfig = field(default_factory=OfflineImageCompressionConfig)
    # Nested configuration for offline image validation.
    offline_image_validation: ImageValidationConfig = field(default_factory=ImageValidationConfig)
    # Nested configuration for offline text cleaning.
    offline_text_cleaning: OfflineTextCleaningConfig = field(default_factory=OfflineTextCleaningConfig)
    # Nested configuration for creating data splits.
    splitting: DataSplittingConfig = field(default_factory=DataSplittingConfig)
    
    def __post_init__(self):
        """Initializes properties for backward compatibility."""
        self.cache_processed_images = self.cache_config.enabled
        self.cache_features = self.cache_config.enabled
        self.cache_max_items = self.cache_config.max_memory_items
        self.cache_dir = self.cache_config.cache_directory
        self.cache_to_disk = self.cache_config.use_disk

@dataclass
class RecommendationConfig:
    """Configures parameters for recommendation generation during inference."""
    # The number of recommendations to generate per user.
    top_k: int = 50
    
    # The weight for a diversity-promoting algorithm during reranking.
    diversity_weight: float = 0.3
    # The weight for a novelty-promoting algorithm during reranking.
    novelty_weight: float = 0.2
    # If True, filters items the user has already seen from the final recommendations.
    filter_seen: bool = True
    # The maximum number of candidate items to score before final ranking.
    max_candidates: int = 1000

@dataclass
class Config:
    """The main configuration class that aggregates all other configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    hyperparameter_search: HyperparameterSearchConfig = field(default_factory=HyperparameterSearchConfig)
    # The base directory where model checkpoints and encoders are saved.
    checkpoint_dir: str = 'models/checkpoints'
    # The base directory where all results, such as logs and metrics, are saved.
    results_dir: str = 'results'

    @property
    def model_specific_checkpoint_dir(self) -> str:
        """
        Generates the path to the directory for storing model-specific checkpoints.
        
        Returns:
            A string representing the path, e.g., 'models/checkpoints/resnet_sentence-bert'.
        """
        model_combo = f"{self.model.vision_model}_{self.model.language_model}"
        return f"{self.checkpoint_dir}/{model_combo}"
    
    @property
    def shared_encoders_dir(self) -> str:
        """
        Returns the path to the directory for storing shared encoder files.
        
        Returns:
            A string representing the path, e.g., 'models/checkpoints/encoders'.
        """
        return f"{self.checkpoint_dir}/encoders"
    
    def get_model_checkpoint_path(self, filename: str) -> str:
        """
        Constructs the full path for a given model checkpoint filename.

        Args:
            filename (str): The name of the checkpoint file (e.g., 'best_model.pth').

        Returns:
            The full path to the model checkpoint file.
        """
        return f"{self.model_specific_checkpoint_dir}/{filename}"
    
    def get_encoder_path(self, encoder_name: str) -> str:
        """
        Constructs the full path for a given encoder filename.

        Args:
            encoder_name (str): The name of the encoder file (e.g., 'user_encoder.pkl').

        Returns:
            The full path to the encoder file.
        """
        return f"{self.shared_encoders_dir}/{encoder_name}"

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Loads the configuration from a YAML file.

        This method populates the dataclasses with values from the YAML file.
        If a parameter is missing in the file, it falls back to the default
        value defined in the corresponding dataclass.

        Args:
            path (str): The file path to the YAML configuration file.

        Returns:
            A populated Config object.
        """
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        def _create_with_defaults(dc_type: Any, cfg_dict: Optional[Dict]) -> Any:
            """
            Recursively creates dataclass instances, applying defaults for missing values.

            Args:
                dc_type: The dataclass type to instantiate.
                cfg_dict: The dictionary of values loaded from YAML for this level.

            Returns:
                An instance of the specified dataclass.
            """
            if cfg_dict is None:
                return dc_type()
            
            default_instance = dc_type()
            final_args = {}
            dataclass_fields = {f.name: f for f in fields(dc_type)}
            
            for field_name, field_info in dataclass_fields.items():
                if field_name in cfg_dict:
                    value = cfg_dict[field_name]
                    field_type_hint = field_info.type
                    actual_field_type = field_type_hint
                    
                    if getattr(field_type_hint, '__origin__', None) is Union:
                        non_none_types = [t for t in field_type_hint.__args__ if t is not type(None)]
                        if non_none_types:
                            actual_field_type = non_none_types[0]
                    
                    if is_dataclass(actual_field_type) and isinstance(value, dict):
                        final_args[field_name] = _create_with_defaults(actual_field_type, value)
                    else:
                        final_args[field_name] = value
                else:
                    final_args[field_name] = getattr(default_instance, field_name)
            
            if dc_type == DataConfig:
                old_cache_keys = ['cache_features', 'cache_processed_images', 'cache_max_items', 'cache_dir', 'cache_to_disk']
                old_cache_params = {}
                for key in old_cache_keys:
                    if key in cfg_dict:
                        old_cache_params[key] = cfg_dict[key]
                
                if old_cache_params:
                    cache_enabled = old_cache_params.get('cache_features', old_cache_params.get('cache_processed_images', True))
                    final_args['cache_config'] = SimpleCacheConfig(
                        enabled=cache_enabled,
                        max_memory_items=old_cache_params.get('cache_max_items', 1000),
                        cache_directory=old_cache_params.get('cache_dir', 'data/cache/features'),
                        use_disk=old_cache_params.get('cache_to_disk', False)
                    )
            
            return dc_type(**final_args)

        model_config = _create_with_defaults(ModelConfig, yaml_config.get('model'))
        training_config = _create_with_defaults(TrainingConfig, yaml_config.get('training'))
        data_config = _create_with_defaults(DataConfig, yaml_config.get('data'))
        rec_config = _create_with_defaults(RecommendationConfig, yaml_config.get('recommendation'))
        hyperparam_config = _create_with_defaults(HyperparameterSearchConfig, yaml_config.get('hyperparameter_search'))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            recommendation=rec_config,
            hyperparameter_search=hyperparam_config,
            checkpoint_dir=yaml_config.get('checkpoint_dir', 'models/checkpoints'),
            results_dir=yaml_config.get('results_dir', 'results')
        )

    def to_yaml(self, path: str):
        """
        Saves the current configuration object to a YAML file.

        This method recursively converts the nested dataclasses into a dictionary
        and then serializes it to a human-readable YAML file.

        Args:
            path (str): The destination file path for the YAML output.
        """
        def as_dict_recursive(data_obj: Any) -> Any:
            if is_dataclass(data_obj):
                result = {}
                for f in fields(data_obj):
                    if f.name.startswith('_'):
                        continue
                    value = getattr(data_obj, f.name)
                    result[f.name] = as_dict_recursive(value)
                return result
            elif isinstance(data_obj, list):
                return [as_dict_recursive(i) for i in data_obj]
            elif isinstance(data_obj, dict):
                return {k: as_dict_recursive(v) for k, v in data_obj.items()}
            else:
                return data_obj
        
        config_dict = as_dict_recursive(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieves detailed information about the configured models.

        Returns:
            A dictionary containing the key names, Hugging Face model names,
            and output dimensions for the selected vision and language models.
        """
        vision_model_key = self.model.vision_model
        language_model_key = self.model.language_model
        return {
            'vision': {
                'key_name': vision_model_key,
                'pretrained_model_name': MODEL_CONFIGS['vision'][vision_model_key]['name'],
                'output_dimension': MODEL_CONFIGS['vision'][vision_model_key]['dim']
            },
            'language': {
                'key_name': language_model_key,
                'pretrained_model_name': MODEL_CONFIGS['language'][language_model_key]['name'],
                'output_dimension': MODEL_CONFIGS['language'][language_model_key]['dim']
            }
        }