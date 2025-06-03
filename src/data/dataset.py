# src/data/dataset.py
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
from transformers import AutoTokenizer, AutoImageProcessor, CLIPTokenizer, CLIPProcessor
from tqdm import tqdm
import random
import pickle
from typing import Dict, Optional, Tuple, Any, Union, List

from ..config import MODEL_CONFIGS, TextAugmentationConfig
from .preprocessing import augment_text, normalize_features


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
        text_augmentation_config: Optional[TextAugmentationConfig] = None,
        numerical_feat_cols: List[str] = [],
        numerical_normalization_method: str = 'none',
        numerical_scaler: Optional[Any] = None,
        is_train_mode: bool = False
    ):
        self.interactions = interactions_df.copy()
        self.item_info = item_info_df.set_index('item_id')
        self.image_folder = image_folder
        
        self._create_negative_samples_flag = create_negative_samples
        self._negative_sampling_ratio = negative_sampling_ratio 

        self.text_augmentation_config = text_augmentation_config
        self.numerical_feat_cols = numerical_feat_cols
        self.numerical_normalization_method = numerical_normalization_method
        self.numerical_scaler = numerical_scaler
        self.is_train_mode = is_train_mode

        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name]
        self.language_config = MODEL_CONFIGS['language'][language_model_name]

        self._init_processors(vision_model_name) 

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        if not self.interactions.empty and 'user_id' in self.interactions.columns:
            self.interactions['user_idx'] = self.user_encoder.fit_transform(self.interactions['user_id'])
        elif 'user_idx' not in self.interactions.columns:
            self.interactions['user_idx'] = pd.Series(dtype=int)

        if not self.interactions.empty and 'item_id' in self.interactions.columns:
            self.interactions['item_idx'] = self.item_encoder.fit_transform(self.interactions['item_id'])
        elif 'item_idx' not in self.interactions.columns:
            self.interactions['item_idx'] = pd.Series(dtype=int)
        
        self.n_users = len(self.user_encoder.classes_) if hasattr(self.user_encoder, 'classes_') and self.user_encoder.classes_ is not None else 0
        self.n_items = len(self.item_encoder.classes_) if hasattr(self.item_encoder, 'classes_') and self.item_encoder.classes_ is not None else 0
        
        self.vision_model_name = vision_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_config['name'])

        self.clip_tokenizer_for_contrastive = None
        if self.vision_model_name == 'clip':
            clip_model_hf_name = MODEL_CONFIGS['vision']['clip']['name']
            self.clip_tokenizer_for_contrastive = CLIPTokenizer.from_pretrained(clip_model_hf_name)
        
        if self.numerical_feat_cols and self.numerical_normalization_method != 'none':
            self._preprocess_all_item_numerical_features()

        self.all_samples = pd.DataFrame()

    def _init_processors(self, vision_model_name: str):
        if vision_model_name == 'clip':
            self.image_processor = CLIPProcessor.from_pretrained(self.vision_config['name']).image_processor
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_config['name'])
        # Add a debug print to confirm processor type
        # print(f"DEBUG: Initialized self.image_processor: {type(self.image_processor)}")


    def finalize_setup(self):
        if self._create_negative_samples_flag:
            if not (hasattr(self.user_encoder, 'classes_') and self.user_encoder.classes_ is not None and len(self.user_encoder.classes_) > 0 and \
                    hasattr(self.item_encoder, 'classes_') and self.item_encoder.classes_ is not None and len(self.item_encoder.classes_) > 0):
                if self.interactions.empty:
                    self.all_samples = pd.DataFrame()
                    for col in ['user_id', 'item_id', 'user_idx', 'item_idx', 'label']:
                        if col not in self.all_samples.columns: self.all_samples[col] = pd.Series(dtype=object if col in ['user_id', 'item_id'] else int)
                    return
                self.all_samples = self.interactions.copy()
                if 'label' not in self.all_samples.columns and not self.all_samples.empty: self.all_samples['label'] = 1
                return
            self.all_samples = self._create_samples_with_labels()
        else:
            self.all_samples = self.interactions.copy()
            if 'label' not in self.all_samples.columns and not self.all_samples.empty: self.all_samples['label'] = 1
            elif self.all_samples.empty and 'label' not in self.all_samples.columns: self.all_samples['label'] = pd.Series(dtype=int)

    def _create_samples_with_labels(self):
        current_interactions = self.interactions.copy()
        if not current_interactions.empty:
            if 'user_idx' not in current_interactions.columns or current_interactions['user_idx'].isnull().all():
                if 'user_id' in current_interactions.columns and hasattr(self.user_encoder, 'classes_') and len(self.user_encoder.classes_) > 0:
                    try:
                        known_users = list(self.user_encoder.classes_)
                        current_interactions = current_interactions[current_interactions['user_id'].isin(known_users)].copy()
                        if not current_interactions.empty: current_interactions.loc[:, 'user_idx'] = self.user_encoder.transform(current_interactions['user_id'])
                        else: current_interactions['user_idx'] = pd.Series(dtype=int)
                    except Exception as e: current_interactions['user_idx'] = pd.Series(dtype=int)
                elif 'user_idx' not in current_interactions.columns : current_interactions['user_idx'] = pd.Series(dtype=int)
            if 'item_idx' not in current_interactions.columns or current_interactions['item_idx'].isnull().all():
                if 'item_id' in current_interactions.columns and hasattr(self.item_encoder, 'classes_') and len(self.item_encoder.classes_) > 0:
                    try:
                        known_items = list(self.item_encoder.classes_)
                        current_interactions = current_interactions[current_interactions['item_id'].isin(known_items)].copy()
                        if not current_interactions.empty: current_interactions.loc[:, 'item_idx'] = self.item_encoder.transform(current_interactions['item_id'])
                        else: current_interactions['item_idx'] = pd.Series(dtype=int)
                    except Exception as e: current_interactions['item_idx'] = pd.Series(dtype=int)
                elif 'item_idx' not in current_interactions.columns: current_interactions['item_idx'] = pd.Series(dtype=int)
        else:
            cols = ['user_id', 'item_id', 'user_idx', 'item_idx', 'label']
            empty_df = pd.DataFrame(columns=cols)
            for col in ['user_idx', 'item_idx', 'label']: empty_df[col] = empty_df[col].astype(int)
            for col in ['user_id', 'item_id']: empty_df[col] = empty_df[col].astype(object)
            return empty_df
        if current_interactions.empty or 'user_idx' not in current_interactions.columns or 'item_idx' not in current_interactions.columns or \
           current_interactions['user_idx'].isnull().all() or current_interactions['item_idx'].isnull().all():
            cols = ['user_id', 'item_id', 'user_idx', 'item_idx', 'label']
            empty_df = pd.DataFrame(columns=cols)
            for col in ['user_idx', 'item_idx', 'label']: empty_df[col] = empty_df[col].astype(int)
            for col in ['user_id', 'item_id']: empty_df[col] = empty_df[col].astype(object)
            return empty_df
        positive_df = current_interactions.copy()
        positive_df['label'] = 1
        idx_to_user_id = pd.Series(self.user_encoder.classes_, index=self.user_encoder.transform(self.user_encoder.classes_))
        idx_to_item_id = pd.Series(self.item_encoder.classes_, index=self.item_encoder.transform(self.item_encoder.classes_))
        all_item_indices_set = set(self.item_encoder.transform(self.item_encoder.classes_))
        user_item_interaction_dict = positive_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        negative_samples_list = []
        for user_idx, interacted_item_indices in user_item_interaction_dict.items():
            possible_negative_indices = list(all_item_indices_set - interacted_item_indices)
            num_positive = len(interacted_item_indices)
            num_negative_to_sample = min(int(num_positive * self._negative_sampling_ratio), len(possible_negative_indices))
            if num_negative_to_sample > 0:
                sampled_negative_indices = random.sample(possible_negative_indices, num_negative_to_sample)
                if user_idx not in idx_to_user_id: continue
                current_user_id_str = idx_to_user_id[user_idx] 
                for neg_item_idx in sampled_negative_indices:
                    if neg_item_idx not in idx_to_item_id: continue
                    current_item_id_str = idx_to_item_id[neg_item_idx]
                    negative_samples_list.append({'user_id': current_user_id_str, 'item_id': current_item_id_str, 'user_idx': user_idx, 'item_idx': neg_item_idx, 'label': 0})
        negative_df = pd.DataFrame(negative_samples_list)
        columns_for_concat = ['user_id', 'item_id', 'user_idx', 'item_idx', 'label']
        for col in columns_for_concat:
            if col not in positive_df.columns:
                if col == 'label': positive_df['label'] = 1
                elif col in ['user_idx', 'item_idx']: positive_df[col] = pd.Series(dtype=int)
                else: positive_df[col] = pd.Series(dtype=object)
        if not negative_df.empty:
            for col in columns_for_concat:
                if col not in negative_df.columns:
                    if col in ['user_idx', 'item_idx', 'label']: negative_df[col] = pd.Series(dtype=int)
                    else: negative_df[col] = pd.Series(dtype=object)
            all_samples_df = pd.concat([positive_df[columns_for_concat], negative_df[columns_for_concat]], ignore_index=True)
        else: all_samples_df = positive_df[columns_for_concat].copy()
        if not all_samples_df.empty:
            all_samples_df.dropna(subset=['user_idx', 'item_idx'], inplace=True)
            if not all_samples_df.empty:
                all_samples_df['user_idx'] = all_samples_df['user_idx'].astype(int)
                all_samples_df['item_idx'] = all_samples_df['item_idx'].astype(int)
            else:
                all_samples_df = pd.DataFrame(columns=columns_for_concat)
                for col_name in ['user_idx', 'item_idx', 'label']: all_samples_df[col_name] = all_samples_df[col_name].astype(int)
                for col_name in ['user_id', 'item_id']: all_samples_df[col_name] = all_samples_df[col_name].astype(object)
        return all_samples_df.sample(frac=1, random_state=42).reset_index(drop=True) if not all_samples_df.empty else all_samples_df

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.all_samples.empty or idx >= len(self.all_samples):
            raise IndexError(f"Dataset is empty or index {idx} is out of bounds for length {len(self.all_samples)}.")
        row = self.all_samples.iloc[idx]
        user_idx_val, item_idx_val, item_id_val, label_val = row['user_idx'], row['item_idx'], row['item_id'], row['label']
        item_info_series = self._get_item_info(item_id_val)
        text_content = self._get_item_text(item_info_series)
        if self.is_train_mode and self.text_augmentation_config and self.text_augmentation_config.enabled:
            text_content = augment_text(text_content, augmentation_type=self.text_augmentation_config.augmentation_type, delete_prob=self.text_augmentation_config.delete_prob, swap_prob=self.text_augmentation_config.swap_prob)
        text_tokens = self.tokenizer(text_content, padding='max_length', truncation=True, max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length else 128, return_tensors='pt')
        numerical_features_tensor = self._get_item_numerical_features(item_id_val, item_info_series)
        image_tensor = self._load_and_process_image(item_id_val)
        batch = {'user_idx': torch.tensor(user_idx_val, dtype=torch.long), 'item_idx': torch.tensor(item_idx_val, dtype=torch.long), 'image': image_tensor, 'text_input_ids': text_tokens['input_ids'].squeeze(0), 'text_attention_mask': text_tokens['attention_mask'].squeeze(0), 'numerical_features': numerical_features_tensor, 'label': torch.tensor(label_val, dtype=torch.float32)}
        if self.clip_tokenizer_for_contrastive:
            clip_tokens = self.clip_tokenizer_for_contrastive(text_content, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
            batch['clip_text_input_ids'], batch['clip_text_attention_mask'] = clip_tokens['input_ids'].squeeze(0), clip_tokens['attention_mask'].squeeze(0)
        return batch

    def _preprocess_all_item_numerical_features(self):
        print("Preprocessing numerical features for all items in item_info...")
        processed_numerical_data = {}
        for item_id, row in tqdm(self.item_info.iterrows(), total=len(self.item_info), desc="Processing item numericals"):
            raw_features = [float(row.get(col, 0) if pd.notna(row.get(col)) else 0) for col in self.numerical_feat_cols]
            raw_features_np = np.array(raw_features).reshape(1, -1)
            if self.numerical_normalization_method != 'none':
                normalized_vals_np, _ = normalize_features(raw_features_np, method=self.numerical_normalization_method, scaler=self.numerical_scaler)
                processed_numerical_data[item_id] = torch.tensor(normalized_vals_np.flatten(), dtype=torch.float32)
            else: processed_numerical_data[item_id] = torch.tensor(raw_features, dtype=torch.float32)
        self.item_info['_processed_numerical_features'] = pd.Series(processed_numerical_data, index=processed_numerical_data.keys())

    def _get_item_info(self, item_id: str) -> pd.Series:
        try:
            if item_id not in self.item_info.index: raise KeyError
            return self.item_info.loc[item_id]
        except KeyError:
            fallback_data = {'title': '', 'tag': '', 'description': ''}
            for col in self.numerical_feat_cols: fallback_data[col] = 0
            fallback_data['_processed_numerical_features'] = torch.zeros(len(self.numerical_feat_cols), dtype=torch.float32) if self.numerical_feat_cols else torch.empty(0, dtype=torch.float32)
            return pd.Series(fallback_data, name=item_id)

    def _get_item_text(self, item_info_series: pd.Series) -> str:
        return f"{str(item_info_series.get('title', ''))} {str(item_info_series.get('tag', ''))} {str(item_info_series.get('description', ''))}".strip()

    def _get_item_numerical_features(self, item_id: str, item_info_series: pd.Series) -> torch.Tensor:
        if '_processed_numerical_features' in self.item_info.columns and item_id in self.item_info.index and pd.notna(self.item_info.loc[item_id, '_processed_numerical_features']):
            processed_feature = self.item_info.loc[item_id, '_processed_numerical_features']
            if isinstance(processed_feature, torch.Tensor): return processed_feature
        raw_features = [float(item_info_series.get(col, 0) if pd.notna(item_info_series.get(col,0)) else 0) for col in self.numerical_feat_cols]
        if not raw_features: return torch.empty(0, dtype=torch.float32)
        raw_features_np = np.array(raw_features).reshape(1, -1)
        if self.numerical_normalization_method != 'none':
            normalized_vals_np, _ = normalize_features(raw_features_np, method=self.numerical_normalization_method, scaler=self.numerical_scaler)
            return torch.tensor(normalized_vals_np.flatten(), dtype=torch.float32)
        return torch.tensor(raw_features, dtype=torch.float32)

    def _load_and_process_image(self, item_id: str) -> torch.Tensor:
        base_path = os.path.join(self.image_folder, str(item_id))
        image_path_to_load = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            current_path = f"{base_path}{ext}"
            if os.path.exists(current_path): image_path_to_load = current_path; break
        
        placeholder_size = (224, 224)
        try:
            if hasattr(self.image_processor, 'size'):
                processor_size = self.image_processor.size
                if isinstance(processor_size, dict) and 'shortest_edge' in processor_size:
                    size_val = processor_size['shortest_edge']; placeholder_size = (size_val, size_val)
                elif isinstance(processor_size, (tuple, list)) and len(processor_size) >= 2: placeholder_size = (processor_size[0], processor_size[1])
                elif isinstance(processor_size, int): placeholder_size = (processor_size, processor_size)
        except Exception: pass

        try:
            if image_path_to_load is None: raise FileNotFoundError (f"Image for {item_id} not found.")
            image = Image.open(image_path_to_load).convert('RGB')
        except Exception as e: image = Image.new('RGB', placeholder_size, color='grey')

        try: processed_output = self.image_processor(images=image, return_tensors='pt')
        except Exception as proc_err:
            # print(f"ERROR during image_processor call for item {item_id} with {type(self.image_processor)}. Error: {proc_err}. Using dummy tensor.") # More detailed error
            return torch.zeros(3, placeholder_size[0], placeholder_size[1])

        # ----- START DEBUGGING BLOCK -----
        # print(f"DEBUG: item {item_id}, type(self.image_processor)={type(self.image_processor)}")
        # print(f"DEBUG: item {item_id}, type(processed_output)={type(processed_output)}")
        # if hasattr(processed_output, 'keys'):
        #     print(f"DEBUG: item {item_id}, processed_output.keys()={list(processed_output.keys())}")
        # else:
        #     print(f"DEBUG: item {item_id}, processed_output has no 'keys' attribute.")
        # if hasattr(processed_output, 'data') and isinstance(processed_output.data, dict):
        #      print(f"DEBUG: item {item_id}, processed_output.data.keys()={list(processed_output.data.keys())}")
        # ----- END DEBUGGING BLOCK -----

        image_tensor = None
        if isinstance(processed_output, dict) and 'pixel_values' in processed_output:
            image_tensor = processed_output['pixel_values']
            if image_tensor.ndim == 4 and image_tensor.shape[0] == 1 : # Batch of 1
                 image_tensor = image_tensor.squeeze(0)
        elif torch.is_tensor(processed_output): 
            image_tensor = processed_output
            if image_tensor.ndim == 4 and image_tensor.shape[0] == 1: image_tensor = image_tensor.squeeze(0) 
        
        if image_tensor is None:
            # This is where your original warning comes from
            # print(f"Warning: Image processor output for {item_id} is type {type(processed_output)}, not dict or tensor with 'pixel_values'. Using dummy tensor.")
            return torch.zeros(3, placeholder_size[0], placeholder_size[1])

        if image_tensor.ndim == 2: image_tensor = image_tensor.unsqueeze(0) # Add channel for grayscale
        if image_tensor.ndim == 3 and image_tensor.shape[0] == 1: image_tensor = image_tensor.repeat(3,1,1) # Convert grayscale to RGB
        
        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            # print(f"Warning: Final image tensor for {item_id} has unexpected shape {image_tensor.shape}. Using dummy tensor.")
            return torch.zeros(3, placeholder_size[0], placeholder_size[1])
        return image_tensor

    def get_item_popularity(self) -> Dict[str, float]:
        popularity = {}
        view_col = 'view_number' if 'view_number' in self.numerical_feat_cols else (self.numerical_feat_cols[0] if self.numerical_feat_cols else None)
        for item_id_val in self.item_info.index:
            if view_col and view_col in self.item_info.columns:
                pop_val = self.item_info.loc[item_id_val, view_col]
                popularity[item_id_val] = float(pop_val) if pd.notna(pop_val) else 0.0
            else: popularity[item_id_val] = 0.0
        return popularity

    def get_user_history(self, user_id_to_check: str) -> set:
        if not hasattr(self.user_encoder, 'classes_') or self.user_encoder.classes_ is None: return set()
        try:
            if user_id_to_check not in self.user_encoder.classes_: return set()
            user_idx_internal = self.user_encoder.transform([user_id_to_check])[0]
        except ValueError: return set()
        if 'user_idx' not in self.interactions.columns or self.interactions.empty: return set()
        user_interactions_df = self.interactions[self.interactions['user_idx'] == user_idx_internal]
        return set(user_interactions_df['item_id'].unique())