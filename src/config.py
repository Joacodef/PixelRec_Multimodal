# src/config.py
"""
Configuration module for multimodal recommender
"""
# Updated import: added is_dataclass
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, Dict, Any, List, Union # Ensure Union is imported for type hints
import yaml
from pathlib import Path

# MODEL_CONFIGS definition remains the same...
MODEL_CONFIGS = {
    'vision': {
        'clip': {
            'name': 'openai/clip-vit-base-patch32',
            'dim': 512
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
    """Model configuration parameters"""
    vision_model: str = 'clip'
    language_model: str = 'sentence-bert'
    embedding_dim: int = 128
    freeze_vision: bool = True
    freeze_language: bool = True
    use_contrastive: bool = True
    dropout_rate: float = 0.3

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
class DataConfig:
    """Data loading and preprocessing configuration parameters"""
    item_info_path: str = 'data/raw/item_info/Pixel200K.csv'
    interactions_path: str = 'data/raw/interactions/Pixel200K.csv'
    image_folder: str = 'data/raw/images'
    processed_item_info_path: str = 'data/processed/item_info_processed.csv'
    processed_interactions_path: str = 'data/processed/interactions_processed.csv'
    scaler_path: str = 'data/processed/numerical_scaler.pkl'
    sample_size: Optional[int] = None
    negative_sampling_ratio: float = 1.0
    train_val_split: float = 0.8
    text_augmentation: TextAugmentationConfig = field(default_factory=TextAugmentationConfig)
    numerical_normalization_method: str = 'log1p'
    offline_image_validation: ImageValidationConfig = field(default_factory=ImageValidationConfig)
    offline_text_cleaning: OfflineTextCleaningConfig = field(default_factory=OfflineTextCleaningConfig)
    numerical_features_cols: List[str] = field(default_factory=lambda: [
        'view_number', 'comment_number', 'thumbup_number',
        'share_number', 'coin_number', 'favorite_number', 'barrage_number'
    ])

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
    log_dir: str = 'logs/tensorboard'
    results_dir: str = 'results'

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Loads configuration from a YAML file, correctly instantiating nested dataclasses."""
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        def _create_dataclass_from_dict(dc_type: Any, cfg_dict: Optional[Dict], default_instance: Any) -> Any:
            if cfg_dict is None:
                return default_instance
            
            current_args = asdict(default_instance) # Use asdict from the import
            current_args.update(cfg_dict)
            
            for field_name, field_type_hint in dc_type.__annotations__.items():
                # Check for Optional[DataclassType] or DataclassType
                actual_field_type = None
                is_optional = getattr(field_type_hint, '__origin__', None) is Union and \
                              type(None) in getattr(field_type_hint, '__args__', ())
                
                if is_optional:
                    # Get the non-None type from Optional[T]
                    possible_types = [t for t in field_type_hint.__args__ if t is not type(None)]
                    if possible_types:
                        actual_field_type = possible_types[0]
                else:
                    actual_field_type = field_type_hint

                if actual_field_type and is_dataclass(actual_field_type) and \
                   isinstance(current_args.get(field_name), dict):
                    # Recursively create instance for nested dataclass
                    current_args[field_name] = actual_field_type(**current_args[field_name])
            
            return dc_type(**current_args)

        model_config = _create_dataclass_from_dict(ModelConfig, yaml_config.get('model'), ModelConfig())
        training_config = _create_dataclass_from_dict(TrainingConfig, yaml_config.get('training'), TrainingConfig())
        rec_config = _create_dataclass_from_dict(RecommendationConfig, yaml_config.get('recommendation'), RecommendationConfig())

        data_yaml_dict = yaml_config.get('data', {})
        default_data_instance = DataConfig()

        # Explicitly instantiate nested dataclasses within DataConfig
        text_aug_instance = _create_dataclass_from_dict(
            TextAugmentationConfig,
            data_yaml_dict.get('text_augmentation'),
            default_data_instance.text_augmentation
        )
        img_val_instance = _create_dataclass_from_dict(
            ImageValidationConfig,
            data_yaml_dict.get('offline_image_validation'),
            default_data_instance.offline_image_validation
        )
        text_clean_instance = _create_dataclass_from_dict(
            OfflineTextCleaningConfig,
            data_yaml_dict.get('offline_text_cleaning'),
            default_data_instance.offline_text_cleaning
        )

        data_config_args = asdict(default_data_instance)
        if data_yaml_dict:
            data_config_args.update(data_yaml_dict)
        
        data_config_args['text_augmentation'] = text_aug_instance
        data_config_args['offline_image_validation'] = img_val_instance
        data_config_args['offline_text_cleaning'] = text_clean_instance
        
        data_config = DataConfig(**data_config_args)

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            recommendation=rec_config,
            checkpoint_dir=yaml_config.get('checkpoint_dir', cls.checkpoint_dir),
            log_dir=yaml_config.get('log_dir', cls.log_dir),
            results_dir=yaml_config.get('results_dir', cls.results_dir)
        )

    def to_yaml(self, path: str):
        """Saves the current configuration to a YAML file."""
        def as_dict_recursive(data_obj: Any) -> Any:
            if is_dataclass(data_obj): # Use imported is_dataclass
                return {f.name: as_dict_recursive(getattr(data_obj, f.name)) for f in dataclasses.fields(data_obj)}
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