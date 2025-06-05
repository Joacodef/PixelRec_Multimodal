# src/config.py - Simplified without cross-modal attention complexity
"""
Configuration module with simplified model architecture
"""
from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import Optional, Dict, Any, List, Union
import yaml
from pathlib import Path

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
    """Simplified model configuration without cross-modal attention complexity"""
    # ESSENTIAL (always required)
    vision_model: str = 'resnet'
    language_model: str = 'sentence-bert'
    embedding_dim: int = 64
    use_contrastive: bool = True
    
    # ADVANCED (with sensible defaults)
    freeze_vision: bool = True
    freeze_language: bool = True
    contrastive_temperature: float = 0.07
    dropout_rate: float = 0.3
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    fusion_activation: str = 'relu'
    use_batch_norm: bool = True
    projection_hidden_dim: Optional[int] = None
    final_activation: str = 'sigmoid'
    init_method: str = 'xavier_uniform'

@dataclass
class TrainingConfig:
    """Training configuration with smart defaults"""
    # ESSENTIAL (always required)
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 30
    patience: int = 10
    
    # ADVANCED (with sensible defaults)
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    num_workers: int = 8
    contrastive_weight: float = 0.1
    bce_weight: float = 1.0
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = 'reduce_on_plateau'
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    optimizer_type: str = 'adamw'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

@dataclass
class SimpleCacheConfig:
    """Simplified cache configuration"""
    enabled: bool = True
    max_memory_items: int = 1000
    cache_directory: str = 'data/cache/features'
    use_disk: bool = False

@dataclass
class TextAugmentationConfig:
    """Text augmentation configuration with defaults"""
    enabled: bool = False
    augmentation_type: str = 'random_delete'
    delete_prob: float = 0.1
    swap_prob: float = 0.1

@dataclass
class ImageValidationConfig:
    """Image validation configuration with defaults"""
    check_corrupted: bool = True
    min_width: int = 64
    min_height: int = 64
    allowed_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png'])

@dataclass
class OfflineTextCleaningConfig:
    """Text cleaning configuration with defaults"""
    remove_html: bool = True
    normalize_unicode: bool = True
    to_lowercase: bool = True

@dataclass
class DataSplittingConfig:
    """Data splitting configuration with defaults"""
    random_state: int = 42
    train_final_ratio: float = 0.6
    val_final_ratio: float = 0.2
    test_final_ratio: float = 0.2
    min_interactions_per_user: int = 5
    min_interactions_per_item: int = 5
    validate_no_leakage: bool = True

@dataclass
class OfflineImageCompressionConfig:
    """Image compression configuration with defaults"""
    enabled: bool = True
    compress_if_kb_larger_than: int = 500
    target_quality: int = 85
    resize_if_pixels_larger_than: Optional[List[int]] = field(default_factory=lambda: [2048, 2048])
    resize_target_longest_edge: Optional[int] = 1024

@dataclass
class DataConfig:
    """Data configuration with smart defaults for essential vs advanced settings"""
    # ESSENTIAL paths (must be provided)
    item_info_path: str = 'data/processed/item_info.csv'
    interactions_path: str = 'data/processed/interactions.csv'
    image_folder: str = 'data/raw/images'
    processed_item_info_path: str = 'data/processed/item_info.csv'
    processed_interactions_path: str = 'data/processed/interactions.csv'
    split_data_path: str = 'data/splits/split_1'
    train_data_path: str = 'data/splits/split_1/train.csv'
    val_data_path: str = 'data/splits/split_1/val.csv'
    test_data_path: str = 'data/splits/split_1/test.csv'
    
    # ESSENTIAL cache config
    cache_config: SimpleCacheConfig = field(default_factory=SimpleCacheConfig)
    
    # ADVANCED settings (with defaults)
    scaler_path: str = 'data/processed/numerical_scaler.pkl'
    processed_image_destination_folder: Optional[str] = 'data/processed/images'
    negative_sampling_ratio: float = 1.0
    numerical_normalization_method: str = 'standardization'
    numerical_features_cols: List[str] = field(default_factory=lambda: [
        'view_number', 'comment_number', 'thumbup_number',
        'share_number', 'coin_number', 'favorite_number', 'barrage_number'
    ])
    
    # ADVANCED nested configs (with defaults)
    text_augmentation: TextAugmentationConfig = field(default_factory=TextAugmentationConfig)
    offline_image_compression: OfflineImageCompressionConfig = field(default_factory=OfflineImageCompressionConfig)
    offline_image_validation: ImageValidationConfig = field(default_factory=ImageValidationConfig)
    offline_text_cleaning: OfflineTextCleaningConfig = field(default_factory=OfflineTextCleaningConfig)
    splitting: DataSplittingConfig = field(default_factory=DataSplittingConfig)
    
    def __post_init__(self):
        """Set up backward compatibility properties"""
        self.cache_processed_images = self.cache_config.enabled
        self.cache_features = self.cache_config.enabled
        self.cache_max_items = self.cache_config.max_memory_items
        self.cache_dir = self.cache_config.cache_directory
        self.cache_to_disk = self.cache_config.use_disk

@dataclass
class RecommendationConfig:
    """Recommendation configuration with smart defaults"""
    # ESSENTIAL
    top_k: int = 50
    
    # ADVANCED
    diversity_weight: float = 0.3
    novelty_weight: float = 0.2
    filter_seen: bool = True
    max_candidates: int = 1000

@dataclass
class Config:
    """Main configuration class with smart defaults"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    checkpoint_dir: str = 'models/checkpoints'
    results_dir: str = 'results'

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration with smart defaults for missing parameters"""
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        def _create_with_defaults(dc_type: Any, cfg_dict: Optional[Dict]) -> Any:
            """Create dataclass instance with defaults for missing values"""
            if cfg_dict is None:
                return dc_type()
            
            # Get default instance to use for missing values
            default_instance = dc_type()
            
            # Start with defaults, then update with provided values
            final_args = {}
            
            # Get all field names from the dataclass
            dataclass_fields = {f.name: f for f in fields(dc_type)}
            
            # Process each field
            for field_name, field_info in dataclass_fields.items():
                if field_name in cfg_dict:
                    # Value provided in config
                    value = cfg_dict[field_name]
                    
                    # Check if this field is a nested dataclass
                    field_type_hint = field_info.type
                    actual_field_type = field_type_hint
                    
                    # Handle Optional types
                    if getattr(field_type_hint, '__origin__', None) is Union:
                        non_none_types = [t for t in field_type_hint.__args__ if t is not type(None)]
                        if non_none_types:
                            actual_field_type = non_none_types[0]
                    
                    # Handle nested dataclasses
                    if is_dataclass(actual_field_type) and isinstance(value, dict):
                        final_args[field_name] = _create_with_defaults(actual_field_type, value)
                    else:
                        final_args[field_name] = value
                else:
                    # Value not provided, use default
                    final_args[field_name] = getattr(default_instance, field_name)
            
            # Handle special case for DataConfig cache parameters
            if dc_type == DataConfig:
                # Convert old-style cache parameters if present
                old_cache_keys = ['cache_features', 'cache_processed_images', 'cache_max_items', 'cache_dir', 'cache_to_disk']
                old_cache_params = {}
                for key in old_cache_keys:
                    if key in cfg_dict:
                        old_cache_params[key] = cfg_dict[key]
                
                if old_cache_params:
                    # Create cache_config from old parameters
                    cache_enabled = old_cache_params.get('cache_features', old_cache_params.get('cache_processed_images', True))
                    final_args['cache_config'] = SimpleCacheConfig(
                        enabled=cache_enabled,
                        max_memory_items=old_cache_params.get('cache_max_items', 1000),
                        cache_directory=old_cache_params.get('cache_dir', 'data/cache/features'),
                        use_disk=old_cache_params.get('cache_to_disk', False)
                    )
            
            return dc_type(**final_args)

        # Create config sections with smart defaults
        model_config = _create_with_defaults(ModelConfig, yaml_config.get('model'))
        training_config = _create_with_defaults(TrainingConfig, yaml_config.get('training'))
        data_config = _create_with_defaults(DataConfig, yaml_config.get('data'))
        rec_config = _create_with_defaults(RecommendationConfig, yaml_config.get('recommendation'))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            recommendation=rec_config,
            checkpoint_dir=yaml_config.get('checkpoint_dir', 'models/checkpoints'),
            results_dir=yaml_config.get('results_dir', 'results')
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML"""
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
        """Get model information"""
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