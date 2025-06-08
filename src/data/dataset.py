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
from ..config import MODEL_CONFIGS, TextAugmentationConfig
from ..data.preprocessing import augment_text


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
        vision_model_name: str = 'clip',
        language_model_name: str = 'sentence-bert',
        create_negative_samples: bool = True,
        numerical_feat_cols: Optional[List[str]] = None,
        cache_features: bool = True,
        cache_max_items: int = 1000,
        cache_dir: Optional[str] = None,
        cache_to_disk: bool = False,
        **kwargs
    ):
        """
        Initializes the MultimodalDataset.

        Args:
            interactions_df (pd.DataFrame): DataFrame of user-item interactions.
            item_info_df (pd.DataFrame): DataFrame of item metadata.
            image_folder (str): Path to the directory containing item images.
            vision_model_name (str): The key for the vision model configuration.
            language_model_name (str): The key for the language model configuration.
            create_negative_samples (bool): If True, generates negative samples for training.
            numerical_feat_cols (Optional[List[str]]): A list of column names for numerical features.
            cache_features (bool): If True, enables the feature caching system.
            cache_max_items (int): The maximum number of items for the in-memory cache.
            cache_dir (Optional[str]): The base directory for the feature cache.
            cache_to_disk (bool): If True, persists the cache to disk.
            **kwargs: Additional optional arguments for configuration.
        """
        self.interactions = interactions_df.copy()
        self.item_info_df_original = item_info_df.copy()
        self.item_info = item_info_df.set_index('item_id')
        self.image_folder = image_folder

        # Ensures that interactions only include items present in the item metadata.
        valid_item_ids = set(self.item_info_df_original['item_id'].astype(str))
        original_interaction_count = len(self.interactions)
        self.interactions = self.interactions[
            self.interactions['item_id'].astype(str).isin(valid_item_ids)
        ].reset_index(drop=True)
        
        if len(self.interactions) < original_interaction_count:
            print(f"INFO: Dropped {original_interaction_count - len(self.interactions)} interactions "
                  "that had no corresponding item metadata.")

        # Sets the numerical feature columns to use.
        if numerical_feat_cols is not None:
            self.numerical_feat_cols = numerical_feat_cols
        else:
            self.numerical_feat_cols = [
                'view_number', 'comment_number', 'thumbup_number',
                'share_number', 'coin_number', 'favorite_number', 'barrage_number'
            ]

        # Sets up other configuration parameters from kwargs.
        self.negative_sampling_ratio = float(kwargs.get('negative_sampling_ratio', 1.0))
        self.text_augmentation_config = kwargs.get('text_augmentation_config', TextAugmentationConfig(enabled=False))
        self.numerical_normalization_method = kwargs.get('numerical_normalization_method', 'none')
        self.numerical_scaler = kwargs.get('numerical_scaler', None)
        self.is_train_mode = kwargs.get('is_train_mode', False)

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Initializes the necessary Hugging Face processors.
        self._init_processors(vision_model_name, language_model_name)

        # Initializes the feature cache if enabled.
        self.feature_cache = None
        if cache_features:
            base_dir_for_sfc = cache_dir if cache_dir else "cache"
            self.feature_cache = SimpleFeatureCache(
                vision_model=vision_model_name,
                language_model=language_model_name,
                base_cache_dir=base_dir_for_sfc,
                max_memory_items=cache_max_items,
                use_disk=cache_to_disk
            )

        # Fits encoders on the entire user and item catalogs to ensure consistency.
        self.user_encoder.fit(self.interactions['user_id'].astype(str))
        self.item_encoder.fit(self.item_info_df_original['item_id'].astype(str))
        
        # Transforms interaction data to numerical indices.
        if not self.interactions.empty:
            self.interactions['user_idx'] = self.user_encoder.transform(self.interactions['user_id'].astype(str))
            self.interactions['item_idx'] = self.item_encoder.transform(self.interactions['item_id'].astype(str))
            self.n_users = len(self.user_encoder.classes_)
            self.n_items = len(self.item_encoder.classes_)
        else:
            self.n_users = 0
            self.n_items = 0

        # Creates the final set of samples, including negatives if specified.
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
        cache. If not found, it processes the features on-the-fly. It assembles
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
            features = self._process_item_features(item_id)
            if self.feature_cache and features:
                self.feature_cache.set(item_id, features)
        
        # Provides dummy features if processing fails, to prevent crashes.
        if features is None:
            dummy_len = getattr(self.tokenizer, 'model_max_length', 128)
            features = {
                'image': torch.zeros(3, 224, 224),
                'text_input_ids': torch.zeros(dummy_len, dtype=torch.long),
                'text_attention_mask': torch.zeros(dummy_len, dtype=torch.long),
                'numerical_features': torch.zeros(len(self.numerical_feat_cols)),
            }
            if self.clip_tokenizer_for_contrastive:
                features['clip_text_input_ids'] = torch.zeros(77, dtype=torch.long)
                features['clip_text_attention_mask'] = torch.zeros(77, dtype=torch.long)

        batch = {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'item_idx': torch.tensor(row['item_idx'], dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.float32),
        }
        batch.update(features)
        return batch

    def _process_item_features(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Processes all multimodal features for a single item.

        This involves loading and processing the image, combining and tokenizing
        text fields, applying text augmentation if in training mode, and scaling
        numerical features.

        Args:
            item_id (str): The ID of the item to process.

        Returns:
            Optional[Dict[str, torch.Tensor]]: A dictionary of feature tensors,
                                               or None if processing fails.
        """
        try:
            item_row = self.item_info.loc[item_id]
            image_tensor = self._load_and_process_image(item_id)
            text_content = f"{item_row.get('title', '')} {item_row.get('tag', '')} {item_row.get('description', '')}".strip()
            
            if self.is_train_mode and self.text_augmentation_config.enabled:
                aug_args = {key: value for key, value in self.text_augmentation_config.__dict__.items() if key != 'enabled'}
                text_content = augment_text(text_content, **aug_args)

            text_tokens = self.tokenizer(text_content, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            
            numerical_values = [float(item_row.get(col, 0)) for col in self.numerical_feat_cols]
            numerical_features_np = np.nan_to_num(np.array(numerical_values, dtype=np.float32)).reshape(1, -1)
            
            if self.numerical_scaler and self.numerical_normalization_method not in ['none', 'log1p']:
                n_features_expected = self.numerical_scaler.n_features_in_
                n_features_provided = numerical_features_np.shape[1]

                if n_features_provided != n_features_expected:
                    print(f"\nWARNING for item '{item_id}': Shape mismatch for numerical features. Scaler expects {n_features_expected}, but got {n_features_provided}.")
                    numerical_features_np = np.zeros((1, n_features_expected), dtype=np.float32)

                numerical_features_np = self.numerical_scaler.transform(numerical_features_np)
            
            features = {
                'image': image_tensor,
                'text_input_ids': text_tokens['input_ids'].squeeze(0),
                'text_attention_mask': text_tokens['attention_mask'].squeeze(0),
                'numerical_features': torch.tensor(numerical_features_np.flatten(), dtype=torch.float32)
            }

            if self.clip_tokenizer_for_contrastive:
                clip_tokens = self.clip_tokenizer_for_contrastive(text_content, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
                features['clip_text_input_ids'] = clip_tokens['input_ids'].squeeze(0)
                features['clip_text_attention_mask'] = clip_tokens['attention_mask'].squeeze(0)
            
            return features
        except KeyError:
            return None
        except Exception as e:
            print(f"\n--- ERROR in _process_item_features for item ID '{item_id}' ---")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {e}")
            traceback.print_exc()
            print("--- End of Error Trace ---")
            return None

    def _load_and_process_image(self, item_id: str) -> torch.Tensor:
        """
        Loads an image from disk and processes it.

        Handles cases where the image file is missing by returning a placeholder tensor.

        Args:
            item_id (str): The ID of the item whose image is to be loaded.

        Returns:
            torch.Tensor: The processed image tensor.
        """
        image_path = os.path.join(self.image_folder, f"{item_id}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224), color="grey")
        
        processed_np = self.image_processor(images=[image], return_tensors="np")['pixel_values']
        
        return torch.tensor(processed_np, dtype=torch.float32).squeeze(0)

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
        
        neg_samples = []
        for user_idx, group in tqdm(self.interactions.groupby('user_idx'), desc="Creating negative samples"):
            pos_indices = group['item_idx'].unique()
            neg_candidates = np.setdiff1d(all_item_indices, pos_indices, assume_unique=True)
            
            num_negatives = min(len(neg_candidates), int(len(pos_indices) * self.negative_sampling_ratio))
            if num_negatives > 0:
                sampled_neg_indices = np.random.choice(neg_candidates, num_negatives, replace=False)
                
                user_id = group['user_id'].iloc[0]
                item_ids = self.item_encoder.inverse_transform(sampled_neg_indices)
                
                for item_idx, item_id in zip(sampled_neg_indices, item_ids):
                    neg_samples.append([user_id, item_id, user_idx, item_idx, 0])

        negative_df = pd.DataFrame(neg_samples, columns=['user_id', 'item_id', 'user_idx', 'item_idx', 'label'])
        
        return pd.concat([positive_df, negative_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

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