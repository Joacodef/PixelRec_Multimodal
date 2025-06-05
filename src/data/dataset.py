# src/data/dataset.py - Fixed version with proper initialization order
"""
Simplified dataset with single cache system - FIXED
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

from ..config import MODEL_CONFIGS, TextAugmentationConfig
from ..data.preprocessing import augment_text


class MultimodalDataset(Dataset):
    """Simplified multimodal dataset with single cache - FIXED initialization order"""

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
        **kwargs  # For backward compatibility and other parameters
    ):
        print("Initializing MultimodalDataset...")
        
        # Store data
        print(f"  - Loading {len(interactions_df)} interactions and {len(item_info_df)} items")
        self.interactions = interactions_df.copy()
        # Store original item_info_df for potential use before setting index
        self.item_info_df_original = item_info_df.copy()
        self.item_info = item_info_df.set_index('item_id')
        self.image_folder = image_folder

        # Store parameters from kwargs or set defaults
        self.negative_sampling_ratio = float(kwargs.get('negative_sampling_ratio', 1.0))
        self.text_augmentation_config = kwargs.get('text_augmentation_config', TextAugmentationConfig(enabled=False))
        self.numerical_feat_cols = kwargs.get('numerical_feat_cols', [
            'view_number', 'comment_number', 'thumbup_number',
            'share_number', 'coin_number', 'favorite_number', 'barrage_number'
        ])
        self.numerical_normalization_method = kwargs.get('numerical_normalization_method', 'none')
        self.numerical_scaler = kwargs.get('numerical_scaler', None)
        self.is_train_mode = kwargs.get('is_train_mode', False)

        # Initialize encoders FIRST
        print("  - Initializing user and item encoders...")
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Initialize processors
        print("  - Initializing image and text processors...")
        self._init_processors(vision_model_name, language_model_name)

        # Initialize single cache with model-specific directory
        if cache_features:
            # Auto-generate cache directory based on models
            cache_name = f"{vision_model_name}_{language_model_name}"
            auto_cache_dir = f"cache/{cache_name}"
            
            # Use provided cache_dir or auto-generated one
            effective_cache_dir = cache_dir if cache_dir else auto_cache_dir
            
            print(f"    - Cache directory: {effective_cache_dir}")
            
            self.feature_cache = SimpleFeatureCache(
                max_memory_items=cache_max_items,
                cache_dir=effective_cache_dir,
                use_disk=cache_to_disk,
                vision_model=vision_model_name,
                language_model=language_model_name
            )
        else:
            self.feature_cache = None

        # Fit encoders AFTER they are initialized
        print("  - Fitting user and item encoders...")
        if not self.interactions.empty:
            self.interactions['user_idx'] = self.user_encoder.fit_transform(self.interactions['user_id'].astype(str))
            self.interactions['item_idx'] = self.item_encoder.fit_transform(self.interactions['item_id'].astype(str))
            self.n_users = len(self.user_encoder.classes_)
            self.n_items = len(self.item_encoder.classes_)
            print(f"    Fitted: {self.n_users} users, {self.n_items} items")
        else:
            # Handle empty interactions: attempt to fit encoders from item_info if possible,
            # otherwise n_users/n_items will be 0 or based on item_info only.
            self.n_users = 0
            self.n_items = 0
            if 'user_id' in self.item_info_df_original.columns and not self.item_info_df_original.empty:
                try:
                    self.user_encoder.fit(self.item_info_df_original['user_id'].astype(str).unique())
                    self.n_users = len(self.user_encoder.classes_)
                except Exception:
                    pass
            
            if 'item_id' in self.item_info_df_original.columns and not self.item_info_df_original.empty:
                try:
                    self.item_encoder.fit(self.item_info_df_original['item_id'].astype(str).unique())
                    self.n_items = len(self.item_encoder.classes_)
                except Exception:
                    pass

        # Create samples
        if create_negative_samples:
            print("  - Creating negative samples...")
            self.all_samples = self._create_samples_with_negatives()
            print(f"    Created {len(self.all_samples)} total samples")
        else:
            self.all_samples = self.interactions.copy()
            if 'label' not in self.all_samples.columns and not self.all_samples.empty:
                self.all_samples['label'] = 1
            elif self.all_samples.empty:
                self.all_samples = pd.DataFrame(columns=list(self.interactions.columns) + ['label'] if not self.interactions.empty else ['user_id', 'item_id', 'user_idx', 'item_idx', 'label'])

        print("MultimodalDataset initialization complete")

    def _init_processors(self, vision_model_name: str, language_model_name: str):
        """Initialize image and text processors with progress indicators"""
        print(f"    - Loading vision processor for {vision_model_name}...")
        
        # Vision processor
        vision_hf_name = MODEL_CONFIGS['vision'].get(vision_model_name, {}).get('name')
        if not vision_hf_name:
            print(f"      Warning: Vision model '{vision_model_name}' not found, defaulting to CLIP")
            vision_hf_name = MODEL_CONFIGS['vision']['clip']['name']
            vision_model_name = 'clip'

        if vision_model_name == 'clip':
            from transformers import CLIPProcessor
            # CLIPProcessor contains both image_processor and tokenizer
            clip_processor = CLIPProcessor.from_pretrained(
                vision_hf_name,
                cache_dir="models/cache"
            )
            self.image_processor = clip_processor.image_processor
            self.clip_tokenizer = clip_processor.tokenizer 
            self.clip_tokenizer_for_contrastive = self.clip_tokenizer
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(
                vision_hf_name,
                cache_dir="models/cache"
            )
            self.clip_tokenizer = None
            self.clip_tokenizer_for_contrastive = None

        print(f"    - Loading language tokenizer for {language_model_name}...")
        # Main Text tokenizer
        language_hf_name = MODEL_CONFIGS['language'].get(language_model_name, {}).get('name')
        if not language_hf_name:
            print(f"      Warning: Language model '{language_model_name}' not found, defaulting to Sentence-BERT")
            language_hf_name = MODEL_CONFIGS['language']['sentence-bert']['name']

        self.tokenizer = AutoTokenizer.from_pretrained(
            language_hf_name,
            cache_dir="models/cache"
        )

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        if idx >= len(self.all_samples):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.all_samples)}")
        row = self.all_samples.iloc[idx]
        item_id = str(row['item_id'])

        # Try to get features from cache
        features = None
        if self.feature_cache:
            features = self.feature_cache.get(item_id)

        # Process features if not cached
        if features is None:
            features = self._process_item_features(item_id)
            if self.feature_cache and features is not None:
                self.feature_cache.set(item_id, features)
            elif features is None:
                # Create dummy features to prevent collate errors
                print(f"Warning: Failed to process features for item {item_id}. Using dummy features.")
                dummy_text_len = getattr(self.tokenizer, 'model_max_length', 128)
                dummy_clip_text_len = 77
                features = {
                    'image': torch.zeros(3, 224, 224, dtype=torch.float32),
                    'text_input_ids': torch.zeros(dummy_text_len, dtype=torch.long),
                    'text_attention_mask': torch.zeros(dummy_text_len, dtype=torch.long),
                    'numerical_features': torch.zeros(len(self.numerical_feat_cols), dtype=torch.float32),
                }
                if self.clip_tokenizer_for_contrastive:
                    features['clip_text_input_ids'] = torch.zeros(dummy_clip_text_len, dtype=torch.long)
                    features['clip_text_attention_mask'] = torch.zeros(dummy_clip_text_len, dtype=torch.long)

        batch = {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'item_idx': torch.tensor(row['item_idx'], dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.float32),
        }
        batch.update(features)

        return batch

    def _process_item_features(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Process all features for an item"""
        try:
            # Get item info
            if item_id in self.item_info.index:
                item_row = self.item_info.loc[item_id]
            else:
                # Create dummy item info using self.numerical_feat_cols
                dummy_data = {col: 0 for col in self.numerical_feat_cols}
                dummy_data.update({'title': '', 'tag': '', 'description': ''})
                item_row = pd.Series(dummy_data)
                item_row.name = item_id

            # Process image
            image_tensor = self._load_and_process_image(item_id)

            # Process text
            text_content = f"{item_row.get('title', '')} {item_row.get('tag', '')} {item_row.get('description', '')}".strip()
            
            # Apply text augmentation if configured and in training mode
            if self.is_train_mode and self.text_augmentation_config and self.text_augmentation_config.enabled:
                text_content = augment_text(
                    text_content,
                    augmentation_type=self.text_augmentation_config.augmentation_type,
                    delete_prob=self.text_augmentation_config.delete_prob,
                    swap_prob=self.text_augmentation_config.swap_prob
                )

            # Main tokenizer
            main_tokenizer_max_len = getattr(self.tokenizer, 'model_max_length', 128)
            text_tokens = self.tokenizer(
                text_content,
                padding='max_length',
                truncation=True,
                max_length=main_tokenizer_max_len,
                return_tensors='pt'
            )

            # Numerical features
            numerical_values = []
            for col in self.numerical_feat_cols:
                value = float(item_row.get(col, 0))
                # Robust handling of invalid values
                if not np.isfinite(value):
                    if not hasattr(self, '_nan_warning_shown'):
                        print(f"Warning: Invalid value {value} in column '{col}' for item {item_id}, using 0")
                        self._nan_warning_shown = True
                    value = 0.0
                numerical_values.append(value)

            numerical_features_np = np.array(numerical_values, dtype=np.float32).reshape(1, -1)

            if self.numerical_scaler is not None and self.numerical_normalization_method not in ['none', 'log1p']:
                try:
                    numerical_features_np = self.numerical_scaler.transform(numerical_features_np)
                except Exception as e:
                    if not hasattr(self, '_scaler_warning_shown'):
                        print(f"Warning: Could not transform numerical features: {e}")
                        self._scaler_warning_shown = True

            numerical_features_tensor = torch.tensor(numerical_features_np.flatten(), dtype=torch.float32)
            
            # Ensure numerical_features_tensor has the correct fixed size
            expected_num_feat_len = len(self.numerical_feat_cols)
            if numerical_features_tensor.shape[0] != expected_num_feat_len:
                if not hasattr(self, '_size_warning_shown'):
                    print(f"Warning: Numerical features for item {item_id} has shape {numerical_features_tensor.shape}, expected {expected_num_feat_len}. Padding/truncating.")
                    self._size_warning_shown = True
                if numerical_features_tensor.shape[0] < expected_num_feat_len:
                    padding = torch.zeros(expected_num_feat_len - numerical_features_tensor.shape[0], dtype=torch.float32)
                    numerical_features_tensor = torch.cat((numerical_features_tensor, padding))
                else:
                    numerical_features_tensor = numerical_features_tensor[:expected_num_feat_len]

            features = {
                'image': image_tensor,
                'text_input_ids': text_tokens['input_ids'].squeeze(0),
                'text_attention_mask': text_tokens['attention_mask'].squeeze(0),
                'numerical_features': numerical_features_tensor
            }

            if self.clip_tokenizer_for_contrastive:
                clip_tokenizer_max_len = 77
                clip_tokens = self.clip_tokenizer_for_contrastive(
                    text_content,
                    padding='max_length',
                    truncation=True,
                    max_length=clip_tokenizer_max_len,
                    return_tensors='pt'
                )
                features['clip_text_input_ids'] = clip_tokens['input_ids'].squeeze(0)
                features['clip_text_attention_mask'] = clip_tokens['attention_mask'].squeeze(0)
            
            return features
        except Exception as e:
            print(f"Error processing features for item {item_id}: {e}")
            return None

    def _load_and_process_image(self, item_id: str) -> torch.Tensor:
        """Optimized image loading with better error handling and caching"""
        
        # Check cache first
        if self.feature_cache:
            cached_features = self.feature_cache.get(item_id)
            if cached_features and 'image' in cached_features:
                if isinstance(cached_features['image'], torch.Tensor):
                    return cached_features['image']

        # Find image path efficiently
        base_path = os.path.join(self.image_folder, str(item_id))
        image_path = None
        
        # Check common extensions in order of likelihood
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_path = f"{base_path}{ext}"
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        # Determine target size
        default_size = (224, 224)
        if hasattr(self.image_processor, 'size'):
            processor_size = self.image_processor.size
            if isinstance(processor_size, dict) and 'shortest_edge' in processor_size:
                size_val = processor_size['shortest_edge']
                default_size = (size_val, size_val)
            elif isinstance(processor_size, (tuple, list)) and len(processor_size) >= 2:
                default_size = (processor_size[0], processor_size[1])
            elif isinstance(processor_size, int):
                default_size = (processor_size, processor_size)

        try:
            # Load image with optimization
            if image_path:
                # Use PIL optimization
                with Image.open(image_path) as image:
                    image = image.convert('RGB')
                    # Resize if image is much larger than needed to save memory
                    if max(image.size) > max(default_size) * 2:
                        image.thumbnail((max(default_size) * 2, max(default_size) * 2), Image.Resampling.LANCZOS)
            else:
                # Create placeholder
                image = Image.new('RGB', default_size, color='grey')

            # Process image
            processed = self.image_processor(images=image, return_tensors='pt')
            
            # Extract tensor
            image_tensor = None
            if hasattr(processed, 'data') and isinstance(processed.data, dict) and 'pixel_values' in processed.data:
                image_tensor = processed.data['pixel_values']
            elif isinstance(processed, dict) and 'pixel_values' in processed:
                image_tensor = processed['pixel_values']

            if not isinstance(image_tensor, torch.Tensor):
                raise ValueError(f"Failed to extract pixel_values tensor for item {item_id}")

            # Normalize tensor dimensions
            if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
            elif image_tensor.ndim != 3:
                raise ValueError(f"Unexpected tensor dimensions: {image_tensor.shape}")

            # Ensure 3 channels
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.repeat(3, 1, 1)
            
            if not (image_tensor.ndim == 3 and image_tensor.shape[0] == 3):
                raise ValueError(f"Invalid final tensor shape: {image_tensor.shape}")
            
            return image_tensor

        except Exception as e:
            # Only print warning for first few failures to avoid spam
            if not hasattr(self, '_image_error_count'):
                self._image_error_count = 0
            
            if self._image_error_count < 5:
                print(f"Warning: Error processing image for item {item_id}: {str(e)[:100]}...")
                self._image_error_count += 1
            elif self._image_error_count == 5:
                print("Warning: Suppressing further image processing error messages...")
                self._image_error_count += 1
            
            return torch.zeros(3, default_size[0], default_size[1], dtype=torch.float32)

    def _create_samples_with_negatives(self) -> pd.DataFrame:
        """Creates positive interaction samples and generates corresponding negative samples for each user."""
        
        if self.interactions.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'user_idx', 'item_idx', 'label'])

        # Creates a copy of interactions to serve as the base for positive samples
        positive_df = self.interactions.copy()
        if 'label' not in positive_df.columns:
            positive_df['label'] = 1

        # List to store records for negative samples
        negative_sample_records = []

        # Check if the item encoder has been fitted
        if not hasattr(self.item_encoder, 'classes_') or self.item_encoder.classes_ is None or len(self.item_encoder.classes_) == 0:
            all_item_ids_str_list_temp = self.item_info_df_original['item_id'].astype(str).unique().tolist()
            if not all_item_ids_str_list_temp:
                return positive_df.sample(frac=1, random_state=42).reset_index(drop=True) if not positive_df.empty else pd.DataFrame(columns=['user_id', 'item_id', 'user_idx', 'item_idx', 'label'])
            print("Warning: Item encoder not properly initialized with classes. Negative sampling might be incomplete.")
            return positive_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Creates array of all item indices for efficient operations
        all_item_indices_global = np.arange(len(self.item_encoder.classes_))

        # Create mapping from user_idx to user_id
        user_idx_to_user_id_map = {}
        if hasattr(self.user_encoder, 'classes_') and self.user_encoder.classes_ is not None and len(self.user_encoder.classes_) > 0:
            user_idx_to_user_id_map = pd.Series(
                self.user_encoder.classes_,
                index=self.user_encoder.transform(self.user_encoder.classes_)
            ).to_dict()
        
        # Group interactions by user_idx
        grouped_user_interactions = self.interactions.groupby('user_idx')

        # Generate negative samples for each user
        for user_idx_val, user_interactions_df in tqdm(grouped_user_interactions, desc="Creating negative samples"):
            user_positive_items_str_set = set(user_interactions_df['item_id'].astype(str))
            
            if not user_positive_items_str_set:
                continue

            try:
                user_positive_item_indices_arr = self.item_encoder.transform(list(user_positive_items_str_set))
            except ValueError:
                continue
            
            # Create boolean mask for positive items
            is_positive_mask = np.zeros(len(all_item_indices_global), dtype=bool)
            is_positive_mask[user_positive_item_indices_arr] = True
            
            # Get available negative indices
            available_negative_indices = all_item_indices_global[~is_positive_mask]
            
            if len(available_negative_indices) == 0:
                continue

            num_positives_for_user = len(user_positive_items_str_set)
            num_negatives_to_sample = int(num_positives_for_user * self.negative_sampling_ratio)
            num_negatives_to_sample = min(num_negatives_to_sample, len(available_negative_indices))

            if num_negatives_to_sample > 0:
                sampled_item_idx_vals_arr = np.random.choice(
                    available_negative_indices,
                    num_negatives_to_sample,
                    replace=False
                )

                try:
                    sampled_negatives_str_list = self.item_encoder.inverse_transform(sampled_item_idx_vals_arr)
                except ValueError:
                    continue
                
                user_id_str = user_idx_to_user_id_map.get(user_idx_val)
                if user_id_str is None:
                    user_id_str = str(user_idx_val)
                
                for neg_item_str, item_idx_val in zip(sampled_negatives_str_list, sampled_item_idx_vals_arr):
                    negative_sample_records.append((
                        user_id_str,
                        neg_item_str,
                        user_idx_val,
                        item_idx_val,
                        0  # label for negative samples
                    ))
        
        negative_df_columns = ['user_id', 'item_id', 'user_idx', 'item_idx', 'label']
        negative_df = pd.DataFrame(negative_sample_records, columns=negative_df_columns) if negative_sample_records else pd.DataFrame(columns=positive_df.columns)
        
        all_samples = pd.concat([positive_df, negative_df], ignore_index=True)
        return all_samples.sample(frac=1, random_state=42).reset_index(drop=True)

    def get_item_popularity(self) -> Dict[str, int]:
        """Get item popularity scores (counts)"""
        if self.interactions.empty:
            return {}
        return self.interactions['item_id'].astype(str).value_counts().to_dict()

    def get_user_history(self, user_id: str) -> set[str]:
        """Get user's interaction history (set of string item IDs)"""
        if user_id not in self.user_encoder.classes_ or self.interactions.empty:
            return set()
        user_interactions_df = self.interactions[self.interactions['user_id'].astype(str) == user_id]
        return set(user_interactions_df['item_id'].astype(str).unique())

    def _get_item_text(self, item_row: pd.Series) -> str:
        """Extract text content from item row"""
        title = str(item_row.get('title', ''))
        tag = str(item_row.get('tag', ''))
        description = str(item_row.get('description', ''))
        return f"{title} {tag} {description}".strip()

    def _get_item_numerical_features(self, item_id: str, item_row: pd.Series) -> torch.Tensor:
        """Extract numerical features from item row"""
        numerical_values = []
        for col in self.numerical_feat_cols:
            value = float(item_row.get(col, 0))
            # Replace NaN/Inf with 0
            if np.isnan(value) or np.isinf(value):
                print(f"Warning: NaN/Inf found in {col} for item {item_id}, replacing with 0")
                value = 0.0
            numerical_values.append(value)
        
        return torch.tensor(numerical_values, dtype=torch.float32)