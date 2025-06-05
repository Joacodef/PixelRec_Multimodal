# src/data/dataset.py - Simplified version
"""
Simplified dataset with single cache system
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoImageProcessor, CLIPTokenizer
from tqdm import tqdm
from typing import Dict, Optional, Any, List

from .simple_cache import SimpleFeatureCache


class MultimodalDataset(Dataset):
    """Simplified multimodal dataset with single cache"""

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        item_info_df: pd.DataFrame,
        image_folder: str,
        vision_model_name: str = 'clip',
        language_model_name: str = 'sentence-bert',
        create_negative_samples: bool = True,
        # Simplified cache options
        cache_features: bool = True,
        cache_max_items: int = 1000,
        cache_dir: Optional[str] = None,
        cache_to_disk: bool = False,
        **kwargs  # For backward compatibility
    ):
        self.interactions = interactions_df.copy()
        self.item_info = item_info_df.set_index('item_id')
        self.image_folder = image_folder
        
        # Initialize single cache
        if cache_features:
            self.feature_cache = SimpleFeatureCache(
                max_memory_items=cache_max_items,
                cache_dir=cache_dir,
                use_disk=cache_to_disk
            )
        else:
            self.feature_cache = None
        
        # Initialize encoders and processors (simplified)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self._init_processors(vision_model_name, language_model_name)
        
        # Fit encoders
        if not self.interactions.empty:
            self.interactions['user_idx'] = self.user_encoder.fit_transform(self.interactions['user_id'])
            self.interactions['item_idx'] = self.item_encoder.fit_transform(self.interactions['item_id'])
            self.n_users = len(self.user_encoder.classes_)
            self.n_items = len(self.item_encoder.classes_)
        
        # Create samples
        if create_negative_samples:
            self.all_samples = self._create_samples_with_negatives()
        else:
            self.all_samples = self.interactions.copy()
            if 'label' not in self.all_samples.columns:
                self.all_samples['label'] = 1

    def _init_processors(self, vision_model_name: str, language_model_name: str):
        """Initialize image and text processors"""
        from ..config import MODEL_CONFIGS
        
        # Vision processor
        if vision_model_name == 'clip':
            from transformers import CLIPProcessor
            self.image_processor = CLIPProcessor.from_pretrained(
                MODEL_CONFIGS['vision'][vision_model_name]['name']
            ).image_processor
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(
                MODEL_CONFIGS['vision'][vision_model_name]['name']
            )
        
        # Text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIGS['language'][language_model_name]['name']
        )
        
        # CLIP tokenizer for contrastive learning (if using CLIP vision)
        if vision_model_name == 'clip':
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                MODEL_CONFIGS['vision'][vision_model_name]['name']
            )
        else:
            self.clip_tokenizer = None

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        row = self.all_samples.iloc[idx]
        item_id = row['item_id']
        
        # Try to get features from cache
        features = None
        if self.feature_cache:
            features = self.feature_cache.get(item_id)
        
        # Process features if not cached
        if features is None:
            features = self._process_item_features(item_id)
            
            # Cache the features
            if self.feature_cache:
                self.feature_cache.set(item_id, features)
        
        # Add user/item indices and label
        batch = {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'item_idx': torch.tensor(row['item_idx'], dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.float32),
            **features
        }
        
        return batch

    def _process_item_features(self, item_id: str) -> Dict[str, torch.Tensor]:
        """Process all features for an item"""
        # Get item info
        if item_id in self.item_info.index:
            item_row = self.item_info.loc[item_id]
        else:
            # Create dummy item info
            item_row = pd.Series({
                'title': '', 'tag': '', 'description': '',
                'view_number': 0, 'comment_number': 0, 'thumbup_number': 0,
                'share_number': 0, 'coin_number': 0, 'favorite_number': 0, 'barrage_number': 0
            })
        
        # Process image
        image_tensor = self._load_and_process_image(item_id)
        
        # Process text
        text_content = f"{item_row.get('title', '')} {item_row.get('tag', '')} {item_row.get('description', '')}".strip()
        
        # Main tokenizer
        text_tokens = self.tokenizer(
            text_content,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Numerical features
        numerical_features = torch.tensor([
            float(item_row.get('view_number', 0)),
            float(item_row.get('comment_number', 0)),
            float(item_row.get('thumbup_number', 0)),
            float(item_row.get('share_number', 0)),
            float(item_row.get('coin_number', 0)),
            float(item_row.get('favorite_number', 0)),
            float(item_row.get('barrage_number', 0)),
        ], dtype=torch.float32)
        
        features = {
            'image': image_tensor,
            'text_input_ids': text_tokens['input_ids'].squeeze(0),
            'text_attention_mask': text_tokens['attention_mask'].squeeze(0),
            'numerical_features': numerical_features
        }
        
        # Add CLIP tokens if available
        if self.clip_tokenizer:
            clip_tokens = self.clip_tokenizer(
                text_content,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors='pt'
            )
            features['clip_text_input_ids'] = clip_tokens['input_ids'].squeeze(0)
            features['clip_text_attention_mask'] = clip_tokens['attention_mask'].squeeze(0)
        
        return features

    def _load_and_process_image(self, item_id: str) -> torch.Tensor:
        """Load and process image for an item"""
        # Find image file
        base_path = os.path.join(self.image_folder, str(item_id))
        image_path = None
        
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            potential_path = f"{base_path}{ext}"
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        # Load image or create placeholder
        try:
            if image_path:
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.new('RGB', (224, 224), color='grey')
            
            # Process image
            processed = self.image_processor(images=image, return_tensors='pt')
            
            if isinstance(processed, dict) and 'pixel_values' in processed:
                image_tensor = processed['pixel_values'].squeeze(0)
            else:
                image_tensor = processed.squeeze(0) if hasattr(processed, 'squeeze') else processed
            
            # Ensure correct shape (3, H, W)
            if image_tensor.dim() == 2:
                image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
            elif image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.repeat(3, 1, 1)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error processing image for item {item_id}: {e}")
            return torch.zeros(3, 224, 224, dtype=torch.float32)

    def _create_samples_with_negatives(self) -> pd.DataFrame:
        """Create positive and negative samples"""
        positive_df = self.interactions.copy()
        positive_df['label'] = 1
        
        # Simple negative sampling
        negative_samples = []
        all_items = set(self.interactions['item_id'].unique())
        
        for user_id in tqdm(self.interactions['user_id'].unique(), desc="Creating negative samples"):
            user_items = set(self.interactions[self.interactions['user_id'] == user_id]['item_id'])
            negative_items = list(all_items - user_items)
            
            # Sample same number of negatives as positives
            num_negatives = min(len(user_items), len(negative_items))
            if num_negatives > 0:
                sampled_negatives = np.random.choice(negative_items, num_negatives, replace=False)
                user_idx = self.user_encoder.transform([user_id])[0]
                
                for neg_item in sampled_negatives:
                    item_idx = self.item_encoder.transform([neg_item])[0]
                    negative_samples.append({
                        'user_id': user_id,
                        'item_id': neg_item,
                        'user_idx': user_idx,
                        'item_idx': item_idx,
                        'label': 0
                    })
        
        negative_df = pd.DataFrame(negative_samples)
        all_samples = pd.concat([positive_df, negative_df], ignore_index=True)
        return all_samples.sample(frac=1, random_state=42).reset_index(drop=True)

    def get_item_popularity(self) -> Dict[str, float]:
        """Get item popularity scores"""
        return self.interactions['item_id'].value_counts().to_dict()

    def get_user_history(self, user_id: str) -> set:
        """Get user's interaction history"""
        if user_id not in self.user_encoder.classes_:
            return set()
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        return set(user_interactions['item_id'].unique())