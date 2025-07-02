# src/data/dataset.py
"""
Defines the MultimodalDataset class for the recommender system.

This module is responsible for loading and preparing the data for model
training and evaluation. It integrates user-item interactions with multimodal
item features (visual, textual, numerical), handles the fitting of user/item
encoders, and provides on-the-fly feature processing with an optional caching
layer for improved performance. It also includes logic for generating negative
samples, a crucial step for training recommendation models.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoImageProcessor, CLIPProcessor
from tqdm import tqdm
from typing import Dict, Optional, Any, List
import traceback

from .simple_cache import SimpleFeatureCache
from ..config import MODEL_CONFIGS, TextAugmentationConfig, ImageAugmentationConfig
from ..data.preprocessing import augment_text
from .processors import ImageProcessor, TextProcessor, NumericalProcessor




class MultimodalDataset(Dataset):
    """
    A PyTorch Dataset for loading multimodal recommendation data.

    This class orchestrates the loading of user-item interactions and associated
    item metadata. It processes various modalities (image, text, numerical),
    handles label encoding for users and items, generates negative samples for
    training, and manages feature caching to accelerate data loading.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        item_info_df: pd.DataFrame,
        image_folder: str,
        vision_model_name: Optional[str] = 'clip',
        language_model_name: Optional[str] = 'sentence-bert',
        create_negative_samples: bool = True,
        numerical_feat_cols: Optional[List[str]] = None,
        categorical_feat_cols: Optional[List[str]] = None,
        cache_features: bool = True,
        cache_max_items: int = 1000,
        cache_dir: Optional[str] = None,
        cache_to_disk: bool = False,
        user_encoder: LabelEncoder = None,
        item_encoder: LabelEncoder = None,
        tag_encoder: LabelEncoder = None,
        **kwargs
    ):
        """
        Initializes the MultimodalDataset.

        Args:
            interactions_df (pd.DataFrame): DataFrame of user-item interactions.
            item_info_df (pd.DataFrame): DataFrame of item metadata.
            image_folder (str): Path to the directory containing item images.
            vision_model_name (Optional[str]): The key for the vision model config. If None, vision is disabled.
            language_model_name (Optional[str]): The key for the language model config. If None, language is disabled.
            create_negative_samples (bool): If True, generates negative samples for training.
            numerical_feat_cols (Optional[List[str]]): A list of column names for numerical features.
            categorical_feat_cols (Optional[List[str]]): A list of column names for categorical features.
            cache_features (bool): If True, enables the feature caching system.
            cache_max_items (int): The maximum number of items for the in-memory cache.
            cache_dir (Optional[str]): The base directory for the feature cache.
            cache_to_disk (bool): If True, persists the cache to disk.
            user_encoder (LabelEncoder): An optional pre-fitted user LabelEncoder.
            item_encoder (LabelEncoder): An optional pre-fitted item LabelEncoder.
            tag_encoder (LabelEncoder): An optional pre-fitted tag LabelEncoder.
            **kwargs: Additional optional arguments for configuration.
        """
        self.interactions = interactions_df.copy()
        self.item_info_df_original = item_info_df.copy()
        self.item_info = item_info_df.set_index('item_id')
        self.image_folder = image_folder

        self.vision_enabled = vision_model_name is not None
        self.language_enabled = language_model_name is not None
        self.numerical_enabled = numerical_feat_cols is not None and len(numerical_feat_cols) > 0

        valid_item_ids = set(self.item_info_df_original['item_id'].astype(str))
        original_interaction_count = len(self.interactions)
        self.interactions = self.interactions[
            self.interactions['item_id'].astype(str).isin(valid_item_ids)
        ].reset_index(drop=True)
        if len(self.interactions) < original_interaction_count:
            print(f"INFO: Dropped {original_interaction_count - len(self.interactions)} interactions "
                  "that had no corresponding item metadata.")

        self.numerical_feat_cols = numerical_feat_cols or []
        self.categorical_feat_cols = categorical_feat_cols or []

        self.negative_sampling_strategy = kwargs.get('negative_sampling_strategy', 'random')
        self.negative_sampling_ratio = float(kwargs.get('negative_sampling_ratio', 1.0))
        self.text_augmentation_config = kwargs.get('text_augmentation_config', TextAugmentationConfig(enabled=False))
        self.image_augmentation_config = kwargs.get('image_augmentation_config', ImageAugmentationConfig(enabled=False))
        self.numerical_normalization_method = kwargs.get('numerical_normalization_method', 'none')
        self.numerical_scaler = kwargs.get('numerical_scaler', None)
        self.is_train_mode = kwargs.get('is_train_mode', False)

        self.image_processor = None
        self.clip_tokenizer = None  # Initialize clip_tokenizer attribute
        self.image_processor = ImageProcessor(
        model_name=vision_model_name,
        augmentation_config=self.image_augmentation_config,
        is_train=self.is_train_mode
        )
        # If using clip for vision, we also need its text tokenizer for contrastive loss
        if vision_model_name == 'clip':
            from transformers import CLIPProcessor
            vision_hf_name = MODEL_CONFIGS['vision']['clip']['name']
            clip_processor = CLIPProcessor.from_pretrained(vision_hf_name)
            self.clip_tokenizer = clip_processor.tokenizer

        self.text_processor = None
        if self.language_enabled:
            self.text_processor = TextProcessor(
                model_name=language_model_name,
                augmentation_config=self.text_augmentation_config
            )

        self.numerical_processor = None
        if self.numerical_enabled:
            self.numerical_processor = NumericalProcessor(
                numerical_cols=self.numerical_feat_cols,
                normalization_method=self.numerical_normalization_method,
                scaler=self.numerical_scaler
            )
            if self.numerical_processor.scaler and not hasattr(self.numerical_processor.scaler, 'scale_'):
                 self.numerical_processor.fit_scaler(self.item_info_df_original)

        self.user_encoder = user_encoder if user_encoder is not None else LabelEncoder()
        self.item_encoder = item_encoder if item_encoder is not None else LabelEncoder()
        if not hasattr(self.user_encoder, 'classes_'):
            self.user_encoder.fit(self.interactions['user_id'].astype(str))
        if not hasattr(self.item_encoder, 'classes_'):
            self.all_item_ids = self.item_info_df_original['item_id'].astype(str).unique()
            self.item_encoder.fit(self.all_item_ids)

        for col in self.categorical_feat_cols:
             if col == 'tag':
                self.item_info_df_original[col] = self.item_info_df_original[col].fillna('unknown')
                self.item_info[col] = self.item_info[col].fillna('unknown')
                self.tag_encoder = tag_encoder if tag_encoder is not None else LabelEncoder()
                if not hasattr(self.tag_encoder, 'classes_'):
                    self.tag_encoder.fit(self.item_info_df_original[col])
                self.n_tags = len(self.tag_encoder.classes_)

        self.feature_cache = None
        if cache_features:
            base_dir_for_sfc = cache_dir if cache_dir else "cache"
            self.feature_cache = SimpleFeatureCache(
                vision_model=vision_model_name if self.vision_enabled else None,
                language_model=language_model_name if self.language_enabled else None,
                base_cache_dir=base_dir_for_sfc,
                max_memory_items=cache_max_items,
                use_disk=cache_to_disk
            )

        if not self.interactions.empty:
            self.interactions['user_idx'] = self.user_encoder.transform(self.interactions['user_id'].astype(str))
            self.interactions['item_idx'] = self.item_encoder.transform(self.interactions['item_id'].astype(str))
            self.n_users = len(self.user_encoder.classes_)
            self.n_items = len(self.item_encoder.classes_)
        else:
            self.n_users = 0
            self.n_items = 0

        if create_negative_samples:
            self.all_samples = self._create_samples_with_negatives()
        else:
            self.all_samples = self.interactions.copy()
            if 'label' not in self.all_samples.columns and not self.all_samples.empty:
                self.all_samples['label'] = 1

    def _init_processors(self, vision_model_name: str, language_model_name: str):
        """
        Initializes the required Hugging Face image and text processors.

        Args:
            vision_model_name (str): The key for the vision model.
            language_model_name (str): The key for the language model.
        """
        vision_hf_name = MODEL_CONFIGS['vision'].get(vision_model_name, {}).get('name', MODEL_CONFIGS['vision']['clip']['name'])
        language_hf_name = MODEL_CONFIGS['language'].get(language_model_name, {}).get('name', MODEL_CONFIGS['language']['sentence-bert']['name'])

        if vision_model_name == 'clip':
            from transformers import CLIPProcessor
            clip_processor = CLIPProcessor.from_pretrained(vision_hf_name, cache_dir="models/cache")
            self.image_processor = clip_processor.image_processor
            self.clip_tokenizer_for_contrastive = clip_processor.tokenizer
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(vision_hf_name, cache_dir="models/cache")
            self.clip_tokenizer_for_contrastive = None

        self.tokenizer = AutoTokenizer.from_pretrained(language_hf_name, cache_dir="models/cache")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single sample from the dataset by its index.

        This method first attempts to retrieve pre-computed features from the
        cache. If not found, it calls a helper method to process the features
        on-the-fly, which respects which modalities are enabled. It assembles
        and returns a dictionary of tensors ready for model input.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing all necessary data
                                     tensors for a single sample.
        """
        row = self.all_samples.iloc[idx]
        item_id = str(row['item_id'])

        features = self.feature_cache.get(item_id) if self.feature_cache else None

        if features is None:
            features = self._get_item_features(item_id)
            if self.feature_cache and features:
                self.feature_cache.set(item_id, features)
        
        batch = {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'item_idx': torch.tensor(row['item_idx'], dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.float32),
        }
        batch.update(features)
        return batch

    # src/data/dataset.py

    # src/data/dataset.py

    def _get_item_features(self, item_id: str) -> Dict[str, torch.Tensor]:
        """
        Processes all enabled features for a given item ID.
        """
        try:
            item_info = self.item_info.loc[item_id]
        except KeyError:
            return self._get_placeholder_features()

        features = {}

        # Vision features processing 
        if self.vision_enabled and self.image_processor:
            image_path = f"{self.image_folder}/{item_id}.jpg"
            features['image'] = self.image_processor.load_and_transform_image(image_path)
        elif self.image_processor:
            features['image'] = self.image_processor.get_placeholder_tensor()
        

        # Conditionally process text features 
        if self.language_enabled and self.text_processor:
            text_content = str(item_info.get('description', ''))
            text_features = self.text_processor.process_text(text_content)
            features.update(text_features)

        # Process Numerical Features 
        if self.numerical_enabled and self.numerical_processor:
            features['numerical_features'] = self.numerical_processor.get_features(item_info)
        elif self.numerical_processor:
            features['numerical_features'] = self.numerical_processor.get_placeholder_tensor()

        # Process Categorical Features
        if 'tag' in self.categorical_feat_cols:
            tag_str = item_info.get('tag', 'unknown')
            tag_idx = self.tag_encoder.transform([tag_str])[0]
            features['tag_idx'] = torch.tensor(tag_idx, dtype=torch.long)
        else:
            features['tag_idx'] = torch.tensor(0, dtype=torch.long)

        # CLIP features processing
        if self.clip_tokenizer:
            text_content = str(item_info.get('description', ''))
            clip_tokens = self.clip_tokenizer(
                text_content, padding='max_length', truncation=True, max_length=77, return_tensors='pt'
            )
            features['clip_text_input_ids'] = clip_tokens['input_ids'].squeeze(0)
            features['clip_text_attention_mask'] = clip_tokens['attention_mask'].squeeze(0)

        return features
    

    def _get_placeholder_features(self) -> Dict[str, torch.Tensor]:
        """
        Generates a full set of placeholder features for all modalities.
        
        This is used as a fallback when an item's metadata cannot be found.
        """
        features = {}
        if self.image_processor:
            features['image'] = self.image_processor.get_placeholder_tensor()
        if self.text_processor:
            features.update(self.text_processor.get_placeholder_tensors())
        if self.numerical_processor:
            features['numerical_features'] = self.numerical_processor.get_placeholder_tensor()
        
        features['tag_idx'] = torch.tensor(0, dtype=torch.long)
        
        return features



    def _create_samples_with_negatives(self) -> pd.DataFrame:
        """
        Generates negative samples for each positive interaction.

        For each user, it samples items that the user has not interacted with
        to create negative examples for training the model.

        Returns:
            pd.DataFrame: A DataFrame containing both positive and negative samples,
                          shuffled randomly.
        """
        if self.interactions.empty or not hasattr(self.item_encoder, 'classes_') or len(self.item_encoder.classes_) == 0:
            return pd.DataFrame()

        positive_df = self.interactions.copy()
        positive_df['label'] = 1
        
        all_item_indices = np.arange(len(self.item_encoder.classes_))
        # Prepare for popularity-based sampling if needed.
        item_pop_weights = None
        if self.negative_sampling_strategy in ['popularity', 'popularity_inverse']:
            # Calculate global item popularity from the interactions data.
            item_counts = self.interactions['item_id'].astype(str).value_counts()
            
            # Create a weights array aligned with the item encoder's classes.
            item_pop_weights = np.zeros(len(self.item_encoder.classes_))
            for item_id, count in item_counts.items():
                if item_id in self.item_encoder.classes_:
                    idx = self.item_encoder.transform([item_id])[0]
                    if self.negative_sampling_strategy == 'popularity':
                        item_pop_weights[idx] = count
                    else: # popularity_inverse
                        item_pop_weights[idx] = 1.0 / count

            # Normalize to get probabilities.
            total_weight = item_pop_weights.sum()
            if total_weight > 0:
                item_pop_weights /= total_weight
            else: # Fallback to uniform if all weights are zero.
                item_pop_weights = None

        neg_samples = []
        for user_idx, group in tqdm(self.interactions.groupby('user_idx'), desc="Creating negative samples"):
            pos_indices = group['item_idx'].unique()
            neg_candidates = np.setdiff1d(all_item_indices, pos_indices, assume_unique=True)
            
            num_negatives = min(len(neg_candidates), int(len(pos_indices) * self.negative_sampling_ratio))
            if num_negatives > 0:
                if self.negative_sampling_strategy != 'random' and item_pop_weights is not None:
                    # Use popularity-based sampling.
                    candidate_weights = item_pop_weights[neg_candidates]
                    
                    # Normalize weights for the current candidate set.
                    sum_candidate_weights = candidate_weights.sum()
                    if sum_candidate_weights > 0:
                        p = candidate_weights / sum_candidate_weights
                        sampled_neg_indices = np.random.choice(
                            neg_candidates, num_negatives, replace=False, p=p
                        )
                    else: # Fallback for candidates with zero total weight.
                        sampled_neg_indices = np.random.choice(
                            neg_candidates, num_negatives, replace=False
                        )
                else:
                    # Default to random sampling.
                    sampled_neg_indices = np.random.choice(
                        neg_candidates, num_negatives, replace=False
                    )
                
                user_id = group['user_id'].iloc[0]
                item_ids = self.item_encoder.inverse_transform(sampled_neg_indices)
                
                for item_idx, item_id in zip(sampled_neg_indices, item_ids):
                    neg_samples.append([user_id, item_id, user_idx, item_idx, 0])

        negative_df = pd.DataFrame(neg_samples, columns=['user_id', 'item_id', 'user_idx', 'item_idx', 'label'])
        
        return pd.concat([positive_df, negative_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    def _init_image_augmentations(self):
        """Initialize image augmentation pipeline"""
        if self.is_train_mode and hasattr(self, 'image_augmentation_config') and self.image_augmentation_config.enabled:
            from torchvision import transforms
            
            augmentation_list = []
            
            # Color jittering
            if any([self.image_augmentation_config.brightness,
                    self.image_augmentation_config.contrast,
                    self.image_augmentation_config.saturation,
                    self.image_augmentation_config.hue]):
                augmentation_list.append(
                    transforms.ColorJitter(
                        brightness=self.image_augmentation_config.brightness,
                        contrast=self.image_augmentation_config.contrast,
                        saturation=self.image_augmentation_config.saturation,
                        hue=self.image_augmentation_config.hue
                    )
                )
            
            # Random crop and resize
            if self.image_augmentation_config.random_crop:
                augmentation_list.append(
                    transforms.RandomResizedCrop(
                        224,
                        scale=tuple(self.image_augmentation_config.crop_scale),  # Convert list to tuple
                        ratio=(0.75, 1.33)
                    )
                )
            
            # Horizontal flip
            if self.image_augmentation_config.horizontal_flip:
                augmentation_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            # Rotation
            if self.image_augmentation_config.rotation_degrees > 0:
                augmentation_list.append(
                    transforms.RandomRotation(degrees=self.image_augmentation_config.rotation_degrees)
                )
            
            # Gaussian blur
            if self.image_augmentation_config.gaussian_blur:
                augmentation_list.append(
                    transforms.RandomApply([
                        transforms.GaussianBlur(
                            kernel_size=tuple(self.image_augmentation_config.blur_kernel_size),  # Convert list to tuple
                            sigma=(0.1, 2.0)
                        )
                    ], p=0.5)
                )
        
            self.image_augmentation = transforms.Compose(augmentation_list)
        else:
            self.image_augmentation = None


    def get_user_history(self, user_id: str) -> set:
        """
        Retrieves the set of items a user has interacted with.

        Args:
            user_id (str): The ID of the user.

        Returns:
            set: A set of item IDs from the user's interaction history.
        """
        if not hasattr(self.user_encoder, 'classes_') or user_id not in self.user_encoder.classes_:
            return set()
        user_idx = self.user_encoder.transform([user_id])[0]
        item_indices = self.interactions[self.interactions['user_idx'] == user_idx]['item_idx'].tolist()
        return set(self.item_encoder.inverse_transform(item_indices))