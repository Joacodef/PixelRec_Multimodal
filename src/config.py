"""
Configuration module for multimodal recommender
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
class DataConfig:
    """Data configuration"""
    item_info_path: str = 'data/raw/item_info/Pixel200K.csv'
    interactions_path: str = 'data/raw/interactions/Pixel200K.csv'
    image_folder: str = 'data/raw/images'
    sample_size: Optional[int] = None
    negative_sampling_ratio: float = 1.0
    train_val_split: float = 0.8


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
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    recommendation: RecommendationConfig
    
    # Paths
    checkpoint_dir: str = 'models/checkpoints'
    log_dir: str = 'logs/tensorboard'
    results_dir: str = 'results'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            recommendation=RecommendationConfig(**config_dict.get('recommendation', {})),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'models/checkpoints'),
            log_dir=config_dict.get('log_dir', 'logs/tensorboard'),
            results_dir=config_dict.get('results_dir', 'results')
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'recommendation': self.recommendation.__dict__,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'results_dir': self.results_dir
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
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