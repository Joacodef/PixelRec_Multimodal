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
import pickle
from typing import Dict, Optional, Tuple, Any, Union, List

from ..config import MODEL_CONFIGS, TextAugmentationConfig # Import TextAugmentationConfig
from .preprocessing import augment_text, normalize_features # Import necessary functions


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
        negative_sampling_ratio: float = 1.0,
        text_augmentation_config: Optional[TextAugmentationConfig] = None, # Added
        numerical_feat_cols: List[str] = [], # Added
        numerical_normalization_method: str = 'none', # Added
        numerical_scaler: Optional[Any] = None, # Added for pre-fitted scaler
        is_train_mode: bool = False # Added to control augmentation
    ):
        """
        Initialize the multimodal dataset.
        # ... (otros argumentos)
        Args:
            # ...
            text_augmentation_config: Configuration for text augmentation.
            numerical_feat_cols: List of column names for numerical features.
            numerical_normalization_method: Method for normalizing numerical features.
            numerical_scaler: Pre-fitted scaler for numerical features.
            is_train_mode: Boolean indicating if the dataset is for training (enables augmentation).
        """
        self.interactions = interactions_df.copy()
        self.item_info = item_info_df.set_index('item_id')
        self.image_folder = image_folder
        self.negative_sampling_ratio = negative_sampling_ratio
        self.text_augmentation_config = text_augmentation_config
        self.numerical_feat_cols = numerical_feat_cols
        self.numerical_normalization_method = numerical_normalization_method
        self.numerical_scaler = numerical_scaler # Store the pre-fitted scaler
        self.is_train_mode = is_train_mode

        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name]
        self.language_config = MODEL_CONFIGS['language'][language_model_name]

        self._init_processors(vision_model_name)

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        self.interactions['user_idx'] = self.user_encoder.fit_transform(self.interactions['user_id'])
        self.interactions['item_idx'] = self.item_encoder.fit_transform(self.interactions['item_id'])

        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        
        # Pre-process and cache numerical features from item_info if they are to be used
        # This helps in applying normalization consistently if a scaler is fitted on training data
        if self.numerical_feat_cols and self.numerical_normalization_method != 'none':
            self._preprocess_all_item_numerical_features()


        if create_negative_samples:
            self.all_samples = self._create_samples_with_labels()
        else:
            self.all_samples = self.interactions.copy()
            if 'label' not in self.all_samples.columns: # Ensure label column exists
                 self.all_samples['label'] = 1 # Assuming these are positive interactions

    def _init_processors(self, vision_model_name: str):
        if vision_model_name == 'clip':
            from transformers import CLIPProcessor
            # Use CLIPProcessor for both image and text if vision is CLIP and language is compatible
            # For this example, sticking to separate processors as per original structure
            self.image_processor = CLIPProcessor.from_pretrained(self.vision_config['name']).image_processor
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_config['name'])

        self.tokenizer = AutoTokenizer.from_pretrained(self.language_config['name'])

    def _preprocess_all_item_numerical_features(self):
        """
        Pre-processes numerical features for all items in item_info.
        This is useful if a scaler was fitted on training data and needs to be applied.
        """
        print("Preprocessing numerical features for all items in item_info...")
        processed_numerical_data = {}
        for item_id, row in tqdm(self.item_info.iterrows(), total=len(self.item_info), desc="Processing item numericals"):
            raw_features = []
            for col in self.numerical_feat_cols:
                val = row[col] if pd.notna(row[col]) and col in row else 0
                raw_features.append(float(val))
            
            raw_features_np = np.array(raw_features).reshape(1, -1) # Reshape for scaler
            
            if self.numerical_normalization_method != 'none' and self.numerical_normalization_method != 'log1p_hardcoded':
                 # log1p_hardcoded is a placeholder if you keep the old method as an option
                normalized_vals_np, _ = normalize_features(
                    raw_features_np,
                    method=self.numerical_normalization_method,
                    scaler=self.numerical_scaler # Use the pre-fitted scaler
                )
                processed_numerical_data[item_id] = torch.tensor(normalized_vals_np.flatten(), dtype=torch.float32)
            elif self.numerical_normalization_method == 'log1p_hardcoded': # Example for keeping old way
                processed_numerical_data[item_id] = torch.tensor([np.log1p(f) for f in raw_features], dtype=torch.float32)
            else: # 'none' or unhandled
                processed_numerical_data[item_id] = torch.tensor(raw_features, dtype=torch.float32)

        self.item_info['_processed_numerical_features'] = pd.Series(processed_numerical_data)


    def _create_samples_with_labels(self):
        """Create positive and negative samples for training."""
        positive_df = self.interactions.copy()
        positive_df['label'] = 1

        all_item_indices = set(self.interactions['item_idx'].unique())
        user_item_interaction_dict = self.interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        negative_samples_list = []
        print("Creating negative samples...")
        for user_idx, positive_item_indices in tqdm(user_item_interaction_dict.items(), desc="Negative Sampling"):
            possible_negative_indices = list(all_item_indices - positive_item_indices)
            
            n_positive = len(positive_item_indices)
            n_negative_to_sample = min(
                int(n_positive * self.negative_sampling_ratio),
                len(possible_negative_indices)
            )

            if n_negative_to_sample > 0:
                sampled_negative_indices = random.sample(possible_negative_indices, n_negative_to_sample)
                for neg_item_idx in sampled_negative_indices:
                    # We need original user_id and item_id if encoders are re-fitted or not universally available
                    # For simplicity, if encoders are stable, we can use idx.
                    # Here, assuming user_idx and item_idx are primary keys for samples.
                    negative_samples_list.append({
                        'user_idx': user_idx,
                        'item_idx': neg_item_idx,
                        'label': 0
                        # 'user_id' and 'item_id' can be mapped back if needed, but model uses idx
                    })
        
        negative_df = pd.DataFrame(negative_samples_list)

        # Concatenate and shuffle
        all_samples_df = pd.concat([positive_df[['user_idx', 'item_idx', 'label']], negative_df], ignore_index=True)
        
        # Map item_idx back to item_id for fetching item info later
        # This requires item_encoder to be fitted on the same item set.
        # Create a reverse mapping from item_idx to item_id
        idx_to_id_mapping = pd.Series(self.item_encoder.classes_, index=self.item_encoder.transform(self.item_encoder.classes_))
        all_samples_df['item_id'] = all_samples_df['item_idx'].map(idx_to_id_mapping)
        
        # Remove samples where item_id could not be mapped (if any item_idx was out of encoder's scope)
        all_samples_df.dropna(subset=['item_id'], inplace=True)

        return all_samples_df.sample(frac=1).reset_index(drop=True)


    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.all_samples.iloc[idx]
        user_idx_val = row['user_idx']
        item_idx_val = row['item_idx']
        item_id_val = row['item_id'] # Get item_id from the processed samples
        label_val = row['label']

        item_info_series = self._get_item_info(item_id_val)

        # Process text
        text_content = self._get_item_text(item_info_series)
        if self.is_train_mode and self.text_augmentation_config and self.text_augmentation_config.enabled:
            text_content = augment_text(
                text_content,
                augmentation_type=self.text_augmentation_config.augmentation_type,
                delete_prob=self.text_augmentation_config.delete_prob,
                swap_prob=self.text_augmentation_config.swap_prob
            )
        
        text_tokens = self.tokenizer(
            text_content,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') else 128, # Use model_max_length
            return_tensors='pt'
        )

        # Get numerical features
        numerical_features_tensor = self._get_item_numerical_features(item_id_val, item_info_series)


        # Load and process image
        image_tensor = self._load_and_process_image(item_id_val)

        return {
            'user_idx': torch.tensor(user_idx_val, dtype=torch.long),
            'item_idx': torch.tensor(item_idx_val, dtype=torch.long),
            'image': image_tensor,
            'text_input_ids': text_tokens['input_ids'].squeeze(0), # Squeeze batch dim
            'text_attention_mask': text_tokens['attention_mask'].squeeze(0), # Squeeze batch dim
            'numerical_features': numerical_features_tensor,
            'label': torch.tensor(label_val, dtype=torch.float32)
        }

    def _get_item_info(self, item_id: str) -> pd.Series:
        try:
            return self.item_info.loc[item_id]
        except KeyError:
            # Fallback for missing items
            # Ensure all expected columns, including numerical ones, are present for fallback
            fallback_data = {
                'title': '', 'tag': '', 'description': ''
            }
            for col in self.numerical_feat_cols:
                fallback_data[col] = 0
            # If _processed_numerical_features is expected, provide a fallback tensor
            if '_processed_numerical_features' in self.item_info.columns:
                 fallback_data['_processed_numerical_features'] = torch.zeros(len(self.numerical_feat_cols), dtype=torch.float32)

            return pd.Series(fallback_data)


    def _get_item_text(self, item_info_series: pd.Series) -> str:
        title = str(item_info_series.get('title', ''))
        tag = str(item_info_series.get('tag', ''))
        description = str(item_info_series.get('description', ''))
        return f"{title} {tag} {description}".strip()


    def _get_item_numerical_features(self, item_id: str, item_info_series: pd.Series) -> torch.Tensor:
        """Gets numerical features, potentially using pre-processed ones."""
        if '_processed_numerical_features' in self.item_info.columns and pd.notna(self.item_info.loc[item_id, '_processed_numerical_features']):
            return self.item_info.loc[item_id, '_processed_numerical_features']
        else: # Fallback or if not preprocessed
            raw_features = []
            for col in self.numerical_feat_cols:
                val = item_info_series.get(col, 0) if pd.notna(item_info_series.get(col, 0)) else 0
                raw_features.append(float(val))
            
            if not raw_features: # Handle case with no numerical features defined
                return torch.empty(0, dtype=torch.float32)

            raw_features_np = np.array(raw_features).reshape(1, -1)

            if self.numerical_normalization_method != 'none' and self.numerical_normalization_method != 'log1p_hardcoded':
                normalized_vals_np, _ = normalize_features(
                    raw_features_np,
                    method=self.numerical_normalization_method,
                    scaler=self.numerical_scaler # Apply scaler (even if it's None for log1p inside normalize_features)
                )
                return torch.tensor(normalized_vals_np.flatten(), dtype=torch.float32)
            elif self.numerical_normalization_method == 'log1p_hardcoded':
                 return torch.tensor([np.log1p(f) for f in raw_features], dtype=torch.float32)
            else: # 'none'
                return torch.tensor(raw_features, dtype=torch.float32)


    def _load_and_process_image(self, item_id: str) -> torch.Tensor:
        # Try to find image with common extensions
        base_path = os.path.join(self.image_folder, item_id)
        image_path_to_load = None
        
        # Prefer .jpg, then .png, then .jpeg as an example order
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            current_path = f"{base_path}{ext}"
            if os.path.exists(current_path):
                image_path_to_load = current_path
                break
        
        try:
            if image_path_to_load is None:
                raise FileNotFoundError (f"Image for {item_id} not found with common extensions.")
            image = Image.open(image_path_to_load).convert('RGB')
        except Exception as e:
            # print(f"Warning: Could not load image for item {item_id} (path: {image_path_to_load}). Using placeholder. Error: {e}")
            # Determine size from image_processor if possible, else default
            placeholder_size = (
                self.image_processor.size['shortest_edge']
                if hasattr(self.image_processor, 'size') and isinstance(self.image_processor.size, dict)
                else (224, 224) # Common default
            )
            if isinstance(placeholder_size, int): # If shortest_edge gives int
                placeholder_size = (placeholder_size, placeholder_size)

            image = Image.new('RGB', placeholder_size, color='white')

        # Process image based on processor type
        # CLIPProcessor might handle images differently (expects a list of images for its .preprocess)
        if 'CLIPImageProcessor' in str(type(self.image_processor)):
             # CLIP's image_processor (not the combined CLIPProcessor) typically takes images and return_tensors args
            processed_output = self.image_processor(images=image, return_tensors='pt')
            image_tensor = processed_output['pixel_values'].squeeze(0) # Squeeze batch dim
        elif hasattr(self.image_processor, 'preprocess'): # For older huggingface versions or specific processors
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].squeeze(0)
        else: # Standard AutoImageProcessor
            image_tensor = self.image_processor(image, return_tensors='pt')['pixel_values'].squeeze(0)
        
        return image_tensor


    def get_item_popularity(self) -> Dict[str, float]:
        popularity = {}
        # Use 'view_number' if available, else default to 0
        view_col = self.numerical_feat_cols[0] if self.numerical_feat_cols and 'view_number' in self.numerical_feat_cols else None
        if view_col is None and self.numerical_feat_cols: # Fallback to first numerical col if view_number not primary
            view_col = self.numerical_feat_cols[0]


        for item_id in self.item_info.index:
            if view_col and view_col in self.item_info.columns:
                pop_val = self.item_info.loc[item_id, view_col]
                popularity[item_id] = float(pop_val) if pd.notna(pop_val) else 0.0
            else:
                popularity[item_id] = 0.0 # Default if no view_number or numerical_cols
        return popularity

    def get_user_history(self, user_id_to_check: str) -> set:
        # Map user_id string to internal user_idx if necessary
        try:
            user_idx_internal = self.user_encoder.transform([user_id_to_check])[0]
        except ValueError:
            return set() # User not in encoder

        # Filter interactions by the internal user_idx
        user_interactions_df = self.interactions[self.interactions['user_idx'] == user_idx_internal]
        return set(user_interactions_df['item_id'].unique())