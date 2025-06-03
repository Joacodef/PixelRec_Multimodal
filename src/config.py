# src/config.py
"""
Configuration module for multimodal recommender
"""
from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import Optional, Dict, Any, List, Union
import yaml
from pathlib import Path

# MODEL_CONFIGS definition (keep existing)
MODEL_CONFIGS = {
    'vision': {
        'clip': {
            'name': 'openai/clip-vit-base-patch32',
            'dim': 768
        },
        'dino': {
            'name': 'facebook/dinov2-base',
            'dim': 768
        },
        'resnet': {
            'name': 'microsoft/resnet-50',
            'dim': 2048
        },
        'convnext': {
            'name': 'facebook/convnext-base-224',
            'dim': 1024
        }
    },
    'language': {
        'sentence-bert': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dim': 384
        },
        'mpnet': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'dim': 768
        },
        'bert': {
            'name': 'bert-base-uncased',
            'dim': 768
        },
        'roberta': {
            'name': 'roberta-base',
            'dim': 768
        }
    }
}

@dataclass
class ModelConfig:
    """Model configuration parameters with all architectural details"""
    # Model selection
    model_class: str = 'pretrained'  # 'pretrained' or 'enhanced'
    vision_model: str = 'clip'
    language_model: str = 'sentence-bert'
    
    # Embedding dimensions
    embedding_dim: int = 128
    
    # Freezing pre-trained components
    freeze_vision: bool = True
    freeze_language: bool = True
    
    # Contrastive learning
    use_contrastive: bool = True
    contrastive_temperature: float = 0.07
    
    # Regularization
    dropout_rate: float = 0.3
    
    # Architecture details
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Fusion network architecture
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    fusion_activation: str = 'relu'  # 'relu', 'gelu', 'tanh'
    use_batch_norm: bool = True
    
    # Projection layer dimensions
    projection_hidden_dim: Optional[int] = None  # If None, no hidden layer in projections
    
    # Cross-modal attention (for enhanced model)
    use_cross_modal_attention: bool = True
    cross_modal_attention_weight: float = 0.5
    
    # Additional architectural choices
    final_activation: str = 'sigmoid'  # 'sigmoid' or 'none'
    init_method: str = 'xavier_uniform'  # 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 3
    gradient_clip: float = 1.0
    num_workers: int = 4
    contrastive_weight: float = 0.1
    bce_weight: float = 1.0
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', 'step'
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    
    # Optimizer settings
    optimizer_type: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

@dataclass
class TextAugmentationConfig:
    """Configuration for text augmentation strategies"""
    enabled: bool = False
    augmentation_type: str = 'random_delete'
    delete_prob: float = 0.1
    swap_prob: float = 0.1

@dataclass
class ImageValidationConfig:
    """Configuration for offline image validation checks"""
    check_corrupted: bool = True
    min_width: int = 64
    min_height: int = 64
    allowed_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png'])

@dataclass
class OfflineTextCleaningConfig:
    """Configuration for offline text cleaning operations"""
    remove_html: bool = True
    normalize_unicode: bool = True
    to_lowercase: bool = True

@dataclass
class DataSplittingConfig:
    """Configuration for data splitting strategies"""    
    # Global random state for reproducibility of splits
    random_state: int = 42
    
    # Final desired ratios for Train, Validation, Test splits
    # These should sum to 1.0
    train_final_ratio: float = 0.6
    val_final_ratio: float = 0.2
    test_final_ratio: float = 0.2

    # Minimum interactions required per user/item for filtering before splitting
    min_interactions_per_user: int = 5
    min_interactions_per_item: int = 5
    
    # Validation
    validate_no_leakage: bool = True  # Check for user/item leakage

@dataclass
class OfflineImageCompressionConfig: # Ensure this dataclass is defined
    """Configuration for offline image compression."""
    enabled: bool = True
    compress_if_kb_larger_than: int = 500
    target_quality: int = 85
    resize_if_pixels_larger_than: Optional[List[int]] = field(default_factory=lambda: [2048, 2048])
    resize_target_longest_edge: Optional[int] = 1024

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration parameters"""
    item_info_path: str = 'data/raw/item_info/Pixel200K.csv'
    interactions_path: str = 'data/raw/interactions/Pixel200K.csv'
    image_folder: str = 'data/raw/images'
    processed_image_destination_folder: Optional[str] = 'data/processed/images_for_training'
    processed_item_info_path: str = 'data/processed/item_info_processed.csv'
    processed_interactions_path: str = 'data/processed/interactions_processed.csv'
    train_data_path: str = "data/splits/split_1/train.csv"
    val_data_path: str = "data/splits/split_1/val.csv"
    test_data_path: str = "data/splits/split_1/test.csv"
    scaler_path: str = 'data/processed/numerical_scaler.pkl'
    sample_size: Optional[int] = None
    negative_sampling_ratio: float = 1.0
    # train_val_split: float = 0.8 # This seems to be an old field
    text_augmentation: TextAugmentationConfig = field(default_factory=TextAugmentationConfig)
    numerical_normalization_method: str = 'log1p'
    # Ensure the type hint uses the defined dataclass name
    offline_image_compression: OfflineImageCompressionConfig = field(default_factory=OfflineImageCompressionConfig)
    offline_image_validation: ImageValidationConfig = field(default_factory=ImageValidationConfig)
    offline_text_cleaning: OfflineTextCleaningConfig = field(default_factory=OfflineTextCleaningConfig)
    splitting: DataSplittingConfig = field(default_factory=DataSplittingConfig)
    numerical_features_cols: List[str] = field(default_factory=lambda: [
        'view_number', 'comment_number', 'thumbup_number',
        'share_number', 'coin_number', 'favorite_number', 'barrage_number'
    ])
    cache_processed_images: bool = False

@dataclass
class RecommendationConfig:
    """Recommendation generation configuration parameters"""
    top_k: int = 10
    diversity_weight: float = 0.3
    novelty_weight: float = 0.2
    filter_seen: bool = True
    max_candidates: int = 1000

@dataclass
class Config:
    """Main configuration class aggregating all specific configurations"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    checkpoint_dir: str = 'models/checkpoints'
    results_dir: str = 'results'

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Loads configuration from a YAML file, correctly instantiating nested dataclasses."""
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        def _create_dataclass_from_dict(dc_type: Any, cfg_dict: Optional[Dict], default_instance: Any) -> Any:
            if cfg_dict is None: # If the key doesn't exist in YAML, return the default instance
                return default_instance
            
            # Start with default values, then update with YAML values
            current_args = asdict(default_instance) # Get defaults as dict
            current_args.update(cfg_dict) # Update with values from YAML
            
            # Recursively instantiate nested dataclasses
            for field_name, field_info in dc_type.__dataclass_fields__.items():
                field_type_hint = field_info.type
                actual_field_type = None
                is_optional = getattr(field_type_hint, '__origin__', None) is Union and \
                              type(None) in getattr(field_type_hint, '__args__', ())
                
                if is_optional:
                    possible_types = [t for t in field_type_hint.__args__ if t is not type(None)]
                    if possible_types:
                        actual_field_type = possible_types[0]
                else:
                    actual_field_type = field_type_hint
                
                # Handle cases where field_type_hint might be a string (forward reference)
                if isinstance(actual_field_type, str):
                    # Attempt to resolve string to actual type (basic implementation)
                    resolved_type = globals().get(actual_field_type) # Or locals(), or a more robust mechanism
                    if resolved_type and is_dataclass(resolved_type):
                        actual_field_type = resolved_type
                    else: # Fallback or raise error if type string can't be resolved
                        actual_field_type = None


                if actual_field_type and is_dataclass(actual_field_type) and \
                   isinstance(current_args.get(field_name), dict):
                    # Get the default instance for the nested dataclass field
                    nested_default_instance = getattr(default_instance, field_name)
                    current_args[field_name] = _create_dataclass_from_dict(
                        actual_field_type, 
                        current_args[field_name], 
                        nested_default_instance
                    )
            
            return dc_type(**current_args)

        # Instantiate main config sections
        model_config = _create_dataclass_from_dict(ModelConfig, yaml_config.get('model'), ModelConfig())
        training_config = _create_dataclass_from_dict(TrainingConfig, yaml_config.get('training'), TrainingConfig())
        rec_config = _create_dataclass_from_dict(RecommendationConfig, yaml_config.get('recommendation'), RecommendationConfig())

        # Instantiate DataConfig and its nested dataclasses
        data_yaml_dict = yaml_config.get('data', {})
        default_data_instance = DataConfig() # Get a default DataConfig instance

        # For DataConfig, we directly pass its yaml_dict and its default instance
        # to _create_dataclass_from_dict. The recursive nature of 
        # _create_dataclass_from_dict will handle the nested fields like
        # text_augmentation, offline_image_validation, offline_text_cleaning, splitting,
        # and importantly, offline_image_compression.
        
        data_config = _create_dataclass_from_dict(DataConfig, data_yaml_dict, default_data_instance)

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            recommendation=rec_config,
            checkpoint_dir=yaml_config.get('checkpoint_dir', 'models/checkpoints'),
            results_dir=yaml_config.get('results_dir', 'results')
        )

    def to_yaml(self, path: str):
        """Saves the current configuration to a YAML file."""
        def as_dict_recursive(data_obj: Any) -> Any:
            if is_dataclass(data_obj):
                return {f.name: as_dict_recursive(getattr(data_obj, f.name)) for f in fields(data_obj)}
            elif isinstance(data_obj, list):
                return [as_dict_recursive(i) for i in data_obj]
            elif isinstance(data_obj, dict):
                return {k: as_dict_recursive(v) for k, v in data_obj.items()}
            else:
                return data_obj
        
        config_dict_to_save = as_dict_recursive(self)
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, 'w') as f:
            yaml.dump(config_dict_to_save, f, default_flow_style=False, sort_keys=False)

    def get_model_info(self) -> Dict[str, Any]:
        """Retrieves detailed information about the selected vision and language models."""
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