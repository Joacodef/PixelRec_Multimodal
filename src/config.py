from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vision_model: str = 'clip'
    language_model: str = 'sentence-bert'
    embedding_dim: int = 128
    freeze_vision: bool = True
    freeze_language: bool = True
    use_contrastive: bool = True
    
@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    patience: int = 3
    num_workers: int = 4
    
@dataclass
class DataConfig:
    item_info_path: str = 'data/raw/item_info/Pixel200K.csv'
    interactions_path: str = 'data/raw/interactions/Pixel200K.csv'
    image_folder: str = 'data/raw/images'
    sample_size: Optional[int] = None