"""
Dataset module for multimodal recommendation system
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoImageProcessor
from tqdm import tqdm
import random
from typing import Dict, Optional, Tuple

from ..config import MODEL_CONFIGS


class MultimodalDataset(Dataset):
    """Dataset for multimodal recommendation with images and text"""
    
    def __init__(
        self, 
        interactions_df: pd.DataFrame, 
        item_info_df: pd.DataFrame, 
        image_folder: str,
        vision_model_name: str = 'clip', 
        language_model_name: str = 'sentence-bert',
        create_negative_samples: bool = True,
        negative_sampling_ratio: float = 1.0
    ):
        """
        Initialize the multimodal dataset.
        
        Args:
            interactions_df: DataFrame with user-item interactions
            item_info_df: DataFrame with item metadata
            image_folder: Path to folder containing item images
            vision_model_name: Name of the vision model to use
            language_model_name: Name of the language model to use
            create_negative_samples: Whether to create negative samples
            negative_sampling_ratio: Ratio of negative to positive samples
        """
        self.interactions = interactions_df
        self.item_info = item_info_df.set_index('item_id')
        self.image_folder = image_folder
        self.negative_sampling_ratio = negative_sampling_ratio
        
        # Get model configurations
        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name]
        self.language_config = MODEL_CONFIGS['language'][language_model_name]
        
        # Initialize processors based on model choice
        self._init_processors(vision_model_name)
        
        # Encode user and item IDs
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.interactions['user_idx'] = self.user_encoder.fit_transform(
            self.interactions['user_id']
        )
        self.interactions['item_idx'] = self.item_encoder.fit_transform(
            self.interactions['item_id']
        )
        
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        
        # Create negative samples if requested
        if create_negative_samples:
            self.create_negative_samples()
        else:
            self.all_samples = self.interactions.copy()
            self.all_samples['label'] = 1
    
    def _init_processors(self, vision_model_name: str):
        """Initialize image and text processors"""
        if vision_model_name == 'clip':
            from transformers import CLIPProcessor
            self.image_processor = CLIPProcessor.from_pretrained(
                self.vision_config['name']
            )
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.vision_config['name']
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.language_config['name']
        )
    
    def create_negative_samples(self):
        """Create negative samples for training"""
        all_items = set(self.interactions['item_id'].unique())
        negative_samples = []
        
        # Limit users for efficiency in large datasets
        users_to_process = self.interactions['user_id'].unique()
        if len(users_to_process) > 1000:
            users_to_process = np.random.choice(
                users_to_process, 1000, replace=False
            )
        
        for user_id in tqdm(users_to_process, desc="Creating negative samples"):
            user_items = set(
                self.interactions[
                    self.interactions['user_id'] == user_id
                ]['item_id']
            )
            negative_items = list(all_items - user_items)
            
            n_positive = len(user_items)
            n_negative = min(
                int(n_positive * self.negative_sampling_ratio), 
                len(negative_items)
            )
            
            if n_negative > 0:
                sampled_negative = random.sample(negative_items, n_negative)
                
                for item_id in sampled_negative:
                    negative_samples.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'label': 0
                    })
        
        positive_df = self.interactions.copy()
        positive_df['label'] = 1
        
        negative_df = pd.DataFrame(negative_samples)
        negative_df['user_idx'] = self.user_encoder.transform(
            negative_df['user_id']
        )
        negative_df['item_idx'] = self.item_encoder.transform(
            negative_df['item_id']
        )
        
        self.all_samples = pd.concat(
            [positive_df, negative_df], 
            ignore_index=True
        )
        self.all_samples = self.all_samples.sample(frac=1).reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.all_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.all_samples.iloc[idx]
        user_idx = row['user_idx']
        item_idx = row['item_idx']
        item_id = row['item_id']
        label = row['label']
        
        # Get item info
        item_info = self._get_item_info(item_id)
        
        # Process text
        text = self._get_item_text(item_info)
        text_tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True,
            max_length=128, 
            return_tensors='pt'
        )
        
        # Process numerical features
        numerical_features = self._get_numerical_features(item_info)
        
        # Load and process image
        image = self._load_and_process_image(item_id)
        
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(item_idx, dtype=torch.long),
            'image': image,
            'text_input_ids': text_tokens['input_ids'].squeeze(),
            'text_attention_mask': text_tokens['attention_mask'].squeeze(),
            'numerical_features': numerical_features,
            'label': torch.tensor(label, dtype=torch.float32)
        }
    
    def _get_item_info(self, item_id: str) -> pd.Series:
        """Get item information with fallback for missing items"""
        try:
            return self.item_info.loc[item_id]
        except:
            # Return empty series with expected columns
            return pd.Series({
                'title': '',
                'tag': '',
                'description': '',
                'view_number': 0,
                'comment_number': 0,
                'thumbup_number': 0,
                'share_number': 0,
                'coin_number': 0,
                'favorite_number': 0,
                'barrage_number': 0
            })
    
    def _get_item_text(self, item_info: pd.Series) -> str:
        """Extract and combine text features from item info"""
        title = str(item_info['title']) if pd.notna(item_info['title']) else ""
        tag = str(item_info['tag']) if pd.notna(item_info['tag']) else ""
        description = str(item_info['description']) if pd.notna(item_info['description']) else ""
        
        return f"{title} {tag} {description}"
    
    def _get_numerical_features(self, item_info: pd.Series) -> torch.Tensor:
        """Extract and transform numerical features"""
        numerical_features = []
        
        for col in ['view_number', 'comment_number', 'thumbup_number', 
                   'share_number', 'coin_number', 'favorite_number', 'barrage_number']:
            val = item_info[col] if pd.notna(item_info[col]) else 0
            # Log transform for better scaling
            numerical_features.append(np.log1p(float(val)))
        
        return torch.tensor(numerical_features, dtype=torch.float32)
    
    def _load_and_process_image(self, item_id: str) -> torch.Tensor:
        """Load and process image with fallback for missing images"""
        image_path = os.path.join(self.image_folder, f"{item_id}.jpg")
        
        # Try different extensions if default doesn't exist
        if not os.path.exists(image_path):
            for ext in ['.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                alt_path = os.path.join(self.image_folder, f"{item_id}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Create a placeholder image
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process image
        if hasattr(self.image_processor, 'preprocess'):
            image_tensor = self.image_processor.preprocess(
                image, return_tensors='pt'
            )['pixel_values'].squeeze()
        else:
            image_tensor = self.image_processor(
                image, return_tensors='pt'
            )['pixel_values'].squeeze()
        
        return image_tensor
    
    def get_item_popularity(self) -> Dict[str, float]:
        """Get item popularity scores based on view count"""
        popularity = {}
        for item_id in self.item_info.index:
            popularity[item_id] = float(
                self.item_info.loc[item_id, 'view_number']
            ) if pd.notna(self.item_info.loc[item_id, 'view_number']) else 0.0
        return popularity
    
    def get_user_history(self, user_id: str) -> set:
        """Get items that a user has interacted with"""
        return set(
            self.interactions[
                self.interactions['user_id'] == user_id
            ]['item_id']
        )