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

from ..config import MODEL_CONFIGS, TextAugmentationConfig # Added TextAugmentationConfig for type hint
from ..data.preprocessing import augment_text # For using text_augmentation_config


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
        **kwargs  # For backward compatibility and other parameters
    ):
        self.interactions = interactions_df.copy()
        # Store original item_info_df for potential use before setting index
        self.item_info_df_original = item_info_df.copy()
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

        # Store parameters from kwargs or set defaults
        self.negative_sampling_ratio = float(kwargs.get('negative_sampling_ratio', 1.0))
        self.text_augmentation_config = kwargs.get('text_augmentation_config', TextAugmentationConfig(enabled=False))
        self.numerical_feat_cols = kwargs.get('numerical_feat_cols', [
            'view_number', 'comment_number', 'thumbup_number',
            'share_number', 'coin_number', 'favorite_number', 'barrage_number'
        ])
        self.numerical_normalization_method = kwargs.get('numerical_normalization_method', 'none')
        self.numerical_scaler = kwargs.get('numerical_scaler', None)
        self.is_train_mode = kwargs.get('is_train_mode', False) # Default to False if not for training

        # Initialize encoders and processors
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self._init_processors(vision_model_name, language_model_name)

        # Fit encoders
        if not self.interactions.empty:
            self.interactions['user_idx'] = self.user_encoder.fit_transform(self.interactions['user_id'].astype(str))
            self.interactions['item_idx'] = self.item_encoder.fit_transform(self.interactions['item_id'].astype(str))
            self.n_users = len(self.user_encoder.classes_)
            self.n_items = len(self.item_encoder.classes_)
        else:
            # Handle empty interactions: attempt to fit encoders from item_info if possible,
            # otherwise n_users/n_items will be 0 or based on item_info only.
            self.n_users = 0
            self.n_items = 0
            if 'user_id' in self.item_info_df_original.columns and not self.item_info_df_original.empty :
                try:
                    self.user_encoder.fit(self.item_info_df_original['user_id'].astype(str).unique())
                    self.n_users = len(self.user_encoder.classes_)
                except Exception: # Catch if all user_ids are NaN or other issues
                    pass # n_users remains 0
            
            if 'item_id' in self.item_info_df_original.columns and not self.item_info_df_original.empty:
                try:
                    self.item_encoder.fit(self.item_info_df_original['item_id'].astype(str).unique())
                    self.n_items = len(self.item_encoder.classes_)
                except Exception:
                    pass # n_items remains 0


        # Create samples
        if create_negative_samples:
            self.all_samples = self._create_samples_with_negatives()
        else:
            self.all_samples = self.interactions.copy()
            if 'label' not in self.all_samples.columns and not self.all_samples.empty :
                self.all_samples['label'] = 1
            elif self.all_samples.empty:
                 self.all_samples = pd.DataFrame(columns=list(self.interactions.columns) + ['label'] if not self.interactions.empty else ['user_id', 'item_id', 'user_idx', 'item_idx', 'label'] )


    def _init_processors(self, vision_model_name: str, language_model_name: str):
        """Initialize image and text processors"""
        # Vision processor
        vision_hf_name = MODEL_CONFIGS['vision'].get(vision_model_name, {}).get('name')
        if not vision_hf_name: # Fallback if model_key or name is missing
            print(f"Warning: Vision model '{vision_model_name}' not in MODEL_CONFIGS or name missing. Defaulting CLIP.")
            vision_hf_name = MODEL_CONFIGS['vision']['clip']['name'] # Default to CLIP
            vision_model_name = 'clip'


        if vision_model_name == 'clip':
            from transformers import CLIPProcessor
            # CLIPProcessor contains both image_processor and tokenizer
            clip_processor = CLIPProcessor.from_pretrained(vision_hf_name)
            self.image_processor = clip_processor.image_processor
            self.clip_tokenizer = clip_processor.tokenizer 
            self.clip_tokenizer_for_contrastive = self.clip_tokenizer # Explicitly set
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(vision_hf_name)
            self.clip_tokenizer = None # No separate CLIP tokenizer if main vision model isn't CLIP
            self.clip_tokenizer_for_contrastive = None # Explicitly set

        # Main Text tokenizer
        language_hf_name = MODEL_CONFIGS['language'].get(language_model_name, {}).get('name')
        if not language_hf_name: # Fallback
            print(f"Warning: Language model '{language_model_name}' not in MODEL_CONFIGS or name missing. Defaulting Sentence-BERT.")
            language_hf_name = MODEL_CONFIGS['language']['sentence-bert']['name']

        self.tokenizer = AutoTokenizer.from_pretrained(language_hf_name)
        
        # If vision model is CLIP but language_model_name is different,
        # self.clip_tokenizer (for vision's text tower) is already set.
        # self.tokenizer is for the main language modality.
        # self.clip_tokenizer_for_contrastive should typically be the tokenizer paired with the vision model if contrastive loss is used.

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        if idx >= len(self.all_samples):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.all_samples)}")
        row = self.all_samples.iloc[idx]
        item_id = str(row['item_id']) # Ensure item_id is string

        # Try to get features from cache
        features = None
        if self.feature_cache:
            features = self.feature_cache.get(item_id)

        # Process features if not cached
        if features is None:
            features = self._process_item_features(item_id)
            if self.feature_cache and features is not None: # Cache only if features were processed
                self.feature_cache.set(item_id, features)
            elif features is None: # Failed to process features
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
        batch.update(features) # Add features to the batch dictionary

        return batch

    def _process_item_features(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Process all features for an item"""
        try:
            # Get item info
            if item_id in self.item_info.index:
                item_row = self.item_info.loc[item_id]
            else:
                # Create dummy item info using self.numerical_feat_cols
                self.item_info_df_original.columns = self.item_info_df_original.columns.astype(str) # Ensure columns are strings
                dummy_data = {col: 0 for col in self.numerical_feat_cols}
                dummy_data.update({'title': '', 'tag': '', 'description': ''})
                item_row = pd.Series(dummy_data)
                item_row.name = item_id # ensure item_row has a name, which is item_id

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
            numerical_values = [float(item_row.get(col, 0)) for col in self.numerical_feat_cols]
            numerical_features_np = np.array(numerical_values, dtype=np.float32).reshape(1, -1)

            if self.numerical_scaler is not None and self.numerical_normalization_method not in ['none', 'log1p']:
                try:
                    numerical_features_np = self.numerical_scaler.transform(numerical_features_np)
                except Exception as e:
                    print(f"Warning: Could not transform numerical features for item {item_id} with scaler: {e}. Using unscaled features.")
            elif self.numerical_normalization_method == 'log1p':
                if np.any(numerical_features_np < 0):
                     print(f"Warning: log1p applied to negative values for item {item_id}")
                numerical_features_np = np.log1p(numerical_features_np)
            numerical_features_tensor = torch.tensor(numerical_features_np.flatten(), dtype=torch.float32)
            # Ensure numerical_features_tensor has the correct fixed size
            expected_num_feat_len = len(self.numerical_feat_cols)
            if numerical_features_tensor.shape[0] != expected_num_feat_len:
                 print(f"Warning: Numerical features for item {item_id} has shape {numerical_features_tensor.shape}, expected {expected_num_feat_len}. Padding/truncating.")
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
                clip_tokenizer_max_len = 77 # Standard CLIP max length
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
        """Load and process image for an item"""
        base_path = os.path.join(self.image_folder, str(item_id)) # item_id is already str
        image_path = None

        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            potential_path = f"{base_path}{ext}"
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        default_size = (
            self.image_processor.size['shortest_edge']
            if isinstance(self.image_processor.size, dict) and 'shortest_edge' in self.image_processor.size
            else getattr(self.image_processor, 'size', 224) # Fallback
        )
        if isinstance(default_size, int): default_size = (default_size, default_size)


        try:
            if image_path:
                image = Image.open(image_path).convert('RGB')
            else: # Create placeholder
                image = Image.new('RGB', default_size, color='grey')

            processed = self.image_processor(images=image, return_tensors='pt')
            
            image_tensor = processed.get('pixel_values', processed) if isinstance(processed, dict) else processed
            image_tensor = image_tensor.squeeze(0)

            if image_tensor.dim() == 2: # Grayscale that needs expansion
                image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
            elif image_tensor.shape[0] == 1: # Single channel that needs expansion
                 image_tensor = image_tensor.repeat(3, 1, 1)
            
            return image_tensor

        except Exception as e:
            print(f"Error processing image for item {item_id} (path: {image_path}): {e}. Returning zero tensor.")
            return torch.zeros(3, default_size[0], default_size[1], dtype=torch.float32)


    def _create_samples_with_negatives(self) -> pd.DataFrame:
        """Create positive and negative samples, using self.negative_sampling_ratio"""
        if self.interactions.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'user_idx', 'item_idx', 'label'])

        positive_df = self.interactions.copy()
        if 'label' not in positive_df.columns:
            positive_df['label'] = 1
        
        negative_samples = []
        
        if not hasattr(self.item_encoder, 'classes_') or self.item_encoder.classes_ is None or len(self.item_encoder.classes_) == 0:
            all_item_ids_str_list = self.item_info_df_original['item_id'].astype(str).unique().tolist()
            if not all_item_ids_str_list : # If no items from item_info either
                 return positive_df.sample(frac=1, random_state=42).reset_index(drop=True) if not positive_df.empty else positive_df
        else:
            all_item_ids_str_list = self.item_encoder.classes_.tolist()

        all_items_set = set(all_item_ids_str_list)
        
        unique_user_ids_str = self.interactions['user_id'].astype(str).unique()

        for user_id_str in tqdm(unique_user_ids_str, desc="Creating negative samples"):
            user_positive_interactions = self.interactions[self.interactions['user_id'].astype(str) == user_id_str]
            user_positive_items_set = set(user_positive_interactions['item_id'].astype(str))
            
            available_negative_items = list(all_items_set - user_positive_items_set)
            
            if not available_negative_items:
                continue

            num_positives_for_user = len(user_positive_items_set)
            num_negatives_to_sample = int(num_positives_for_user * self.negative_sampling_ratio)
            num_negatives_to_sample = min(num_negatives_to_sample, len(available_negative_items))

            if num_negatives_to_sample > 0:
                sampled_negatives_str_list = np.random.choice(available_negative_items, num_negatives_to_sample, replace=False)
                
                try:
                    user_idx_val = self.user_encoder.transform([user_id_str])[0]
                except ValueError: # Should not happen if user_id_str comes from self.interactions
                    print(f"User {user_id_str} not in encoder during negative sampling.")
                    continue

                for neg_item_str in sampled_negatives_str_list:
                    try:
                        item_idx_val = self.item_encoder.transform([neg_item_str])[0]
                    except ValueError: # Can happen if all_items_set was augmented from item_info
                        print(f"Item {neg_item_str} not in encoder during negative sampling.")
                        continue
                        
                    negative_samples.append({
                        'user_id': user_id_str, 'item_id': neg_item_str,
                        'user_idx': user_idx_val, 'item_idx': item_idx_val,
                        'label': 0
                    })
        
        negative_df = pd.DataFrame(negative_samples) if negative_samples else pd.DataFrame(columns=positive_df.columns)
        
        all_samples = pd.concat([positive_df, negative_df], ignore_index=True)
        return all_samples.sample(frac=1, random_state=42).reset_index(drop=True)


    def get_item_popularity(self) -> Dict[str, int]: # Changed value to int for counts
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
        return torch.tensor([
            float(item_row.get('view_number', 0)),
            float(item_row.get('comment_number', 0)),
            float(item_row.get('thumbup_number', 0)),
            float(item_row.get('share_number', 0)),
            float(item_row.get('coin_number', 0)),
            float(item_row.get('favorite_number', 0)),
            float(item_row.get('barrage_number', 0)),
        ], dtype=torch.float32)
