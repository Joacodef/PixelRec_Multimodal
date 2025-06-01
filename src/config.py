"""
Configuration module for multimodal recommender
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
from pathlib import Path


# Model configurations for pre-trained models
MODEL_CONFIGS = {
    'vision': {
        'clip': {
            'name': 'openai/clip-vit-base-patch32',
            'dim': 512
        },
        'dino': {
            'name': 'facebook/dino-vitb16',
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
    """Model configuration"""
    vision_model: str = 'clip'
    language_model: str = 'sentence-bert'
    embedding_dim: int = 128
    freeze_vision: bool = True
    freeze_language: bool = True
    use_contrastive: bool = True
    dropout_rate: float = 0.3


@dataclass
class TrainingConfig:
    """Training configuration"""
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
    """Configuration for text augmentation"""
    enabled: bool = False
    augmentation_type: str = 'random_delete'  # 'random_delete', 'random_swap'
    delete_prob: float = 0.1
    swap_prob: float = 0.1


@dataclass
class ImageValidationConfig:
    """Configuration for offline image validation"""
    check_corrupted: bool = True
    min_width: int = 64
    min_height: int = 64
    allowed_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png'])


@dataclass
class OfflineTextCleaningConfig:
    """Configuration for offline text cleaning"""
    remove_html: bool = True
    normalize_unicode: bool = True
    to_lowercase: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
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
    numerical_normalization_method: str = 'log1p'  # 'log1p', 'standardization', 'min_max', 'none'
    offline_image_validation: ImageValidationConfig = field(default_factory=ImageValidationConfig)
    offline_text_cleaning: OfflineTextCleaningConfig = field(default_factory=OfflineTextCleaningConfig)
    numerical_features_cols: List[str] = field(default_factory=lambda: [
        'view_number', 'comment_number', 'thumbup_number',
        'share_number', 'coin_number', 'favorite_number', 'barrage_number'
    ])


@dataclass
class RecommendationConfig:
    """Recommendation configuration"""
    top_k: int = 10
    diversity_weight: float = 0.3
    novelty_weight: float = 0.2
    filter_seen: bool = True
    max_candidates: int = 1000


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)

    checkpoint_dir: str = 'models/checkpoints'
    log_dir: str = 'logs/tensorboard'
    results_dir: str = 'results'

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create a default DataConfig instance to access default values
        default_data_config = DataConfig()
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(
                item_info_path=config_dict.get('data', {}).get('item_info_path', default_data_config.item_info_path),
                interactions_path=config_dict.get('data', {}).get('interactions_path', default_data_config.interactions_path),
                image_folder=config_dict.get('data', {}).get('image_folder', default_data_config.image_folder),
                processed_item_info_path=config_dict.get('data', {}).get('processed_item_info_path', default_data_config.processed_item_info_path),
                processed_interactions_path=config_dict.get('data', {}).get('processed_interactions_path', default_data_config.processed_interactions_path),
                scaler_path=config_dict.get('data', {}).get('scaler_path', default_data_config.scaler_path),
                sample_size=config_dict.get('data', {}).get('sample_size', default_data_config.sample_size),
                negative_sampling_ratio=config_dict.get('data', {}).get('negative_sampling_ratio', default_data_config.negative_sampling_ratio),
                train_val_split=config_dict.get('data', {}).get('train_val_split', default_data_config.train_val_split),
                text_augmentation=TextAugmentationConfig(**config_dict.get('data', {}).get('text_augmentation', {})),
                numerical_normalization_method=config_dict.get('data', {}).get('numerical_normalization_method', default_data_config.numerical_normalization_method),
                offline_image_validation=ImageValidationConfig(**config_dict.get('data', {}).get('offline_image_validation', {})),
                offline_text_cleaning=OfflineTextCleaningConfig(**config_dict.get('data', {}).get('offline_text_cleaning', {})),
                numerical_features_cols=config_dict.get('data', {}).get('numerical_features_cols', default_data_config.numerical_features_cols)
            ),
            recommendation=RecommendationConfig(**config_dict.get('recommendation', {})),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'models/checkpoints'),
            log_dir=config_dict.get('log_dir', 'logs/tensorboard'),
            results_dir=config_dict.get('results_dir', 'results')
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        # Helper to convert dataclasses to dicts recursively
        def as_dict(data):
            if hasattr(data, '__dict__'):
                return {k: as_dict(v) for k, v in data.__dict__.items()}
            elif isinstance(data, list):
                return [as_dict(i) for i in data]
            else:
                return data

        config_dict = as_dict(self)

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about selected models"""
        return {
            'vision': {
                'name': self.model.vision_model,
                'pretrained_name': MODEL_CONFIGS['vision'][self.model.vision_model]['name'],
                'dim': MODEL_CONFIGS['vision'][self.model.vision_model]['dim']
            },
            'language': {
                'name': self.model.language_model,
                'pretrained_name': MODEL_CONFIGS['language'][self.model.language_model]['name'],
                'dim': MODEL_CONFIGS['language'][self.model.language_model]['dim']
            }
        }