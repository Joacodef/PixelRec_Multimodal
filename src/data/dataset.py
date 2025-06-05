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
        # Loads an image for a given item_id, processes it using the configured
        # Hugging Face image processor, and returns it as a PyTorch tensor.
        # Handles caching and falls back to a placeholder if image processing fails.

        if self.feature_cache:
            cached_features = self.feature_cache.get(item_id)
            if cached_features and 'image' in cached_features:
                if isinstance(cached_features['image'], torch.Tensor):
                    return cached_features['image']

        base_path = os.path.join(self.image_folder, str(item_id))
        base_path = base_path.replace('\\', '/')
        image_path = None

        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            potential_path = f"{base_path}{ext}"
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
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
            if image_path:
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.new('RGB', default_size, color='grey')

            processed = self.image_processor(images=image, return_tensors='pt')
            
            image_tensor = None
            # Prioritize checking processed.data as BatchFeature often stores tensors there.
            if hasattr(processed, 'data') and isinstance(processed.data, dict) and 'pixel_values' in processed.data:
                image_tensor = processed.data['pixel_values']
            # Fallback to direct access if it's a dict and .data wasn't the way.
            elif isinstance(processed, dict) and 'pixel_values' in processed:
                image_tensor = processed['pixel_values']

            # Validate that we successfully extracted a tensor.
            if not isinstance(image_tensor, torch.Tensor):
                output_type_name = type(processed).__name__
                data_attr_repr = str(processed.data) if hasattr(processed, 'data') else "N/A (no .data attr)"
                dict_repr = str(dict(processed)) if isinstance(processed, dict) else "N/A (not a dict)"
                
                raise ValueError(
                    f"Image processor output for item {item_id} (path: {image_path}) did not yield a 'pixel_values' Tensor.\n"
                    f"Output type from processor: {output_type_name}.\n"
                    f"Value obtained for image_tensor: {str(image_tensor)[:200]} (Type: {type(image_tensor).__name__}).\n"
                    f"Representation of processed.data: {data_attr_repr[:500]}...\n"
                    f"Representation of dict(processed): {dict_repr[:500]}..."
                )

            # Process the extracted tensor (squeeze, channel checks).
            if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
            elif image_tensor.ndim != 3:
                 raise ValueError(
                    f"Image tensor for item {item_id} has unexpected ndim after squeeze: {image_tensor.ndim}. "
                    f"Shape: {image_tensor.shape}"
                )

            if image_tensor.shape[0] == 1: 
                 image_tensor = image_tensor.repeat(3, 1, 1)
            
            if not (image_tensor.ndim == 3 and image_tensor.shape[0] == 3):
                raise ValueError(
                    f"Image tensor for item {item_id} does not have 3 channels after standardization. "
                    f"Shape: {image_tensor.shape}"
                )
            
            return image_tensor

        except Exception as e:
            import traceback
            detailed_error_traceback = traceback.format_exc()
            error_type = type(e).__name__
            error_args = e.args
            
            print(f"Error processing image for item {item_id} (path: {image_path}).\n"
                  f"Type: {error_type}, Args: {error_args}, Str: {str(e)}.\n"
                  f"Traceback:\n{detailed_error_traceback}\n"
                  f"Returning zero tensor.")
            
            return torch.zeros(3, default_size[0], default_size[1], dtype=torch.float32)


    def _create_samples_with_negatives(self) -> pd.DataFrame:
        # Creates positive interaction samples and generates corresponding negative samples for each user.
        # Positive interactions are labeled as 1. Negative samples, items not interacted with by the user,
        # are labeled as 0. The number of negative samples is proportional to the number of positive
        # interactions for that user, based on self.negative_sampling_ratio.
        # This method is optimized for performance by minimizing per-user overhead,
        # using integer indices for sampling, and efficient DataFrame construction.

        if self.interactions.empty:
            # Returns an empty DataFrame with standard columns if there are no interactions.
            return pd.DataFrame(columns=['user_id', 'item_id', 'user_idx', 'item_idx', 'label'])

        # Creates a copy of interactions to serve as the base for positive samples.
        positive_df = self.interactions.copy()
        if 'label' not in positive_df.columns:
            # Assigns a 'label' of 1 to all interactions in the positive set.
            positive_df['label'] = 1

        # List to store records for negative samples, each record as a tuple.
        negative_sample_records = []

        # Checks if the item encoder has been fitted and contains class information.
        # If not, attempts to use item_info_df_original as a fallback for item list,
        # or returns only positive samples if no items can be determined.
        if not hasattr(self.item_encoder, 'classes_') or self.item_encoder.classes_ is None or len(self.item_encoder.classes_) == 0:
            all_item_ids_str_list_temp = self.item_info_df_original['item_id'].astype(str).unique().tolist()
            if not all_item_ids_str_list_temp:
                 # No items available for sampling; return shuffled positive samples or an empty DataFrame.
                 return positive_df.sample(frac=1, random_state=42).reset_index(drop=True) if not positive_df.empty else pd.DataFrame(columns=['user_id', 'item_id', 'user_idx', 'item_idx', 'label'])
            # This warning indicates that negative sampling might be based on a potentially incomplete item list.
            print("Warning: Item encoder not properly initialized with classes. Negative sampling might be incomplete or use items from item_info_df_original.")
            # Fallback to using positive samples if item encoder isn't ready for robust negative sampling.
            # Depending on requirements, one might choose to raise an error or handle differently.
            return positive_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Creates a NumPy array of all unique item indices (0 to N-1) for efficient operations.
        all_item_indices_global = np.arange(len(self.item_encoder.classes_))

        # Creates a mapping from integer user_idx back to string user_id.
        # This is used to store the original string user_id in the final samples table.
        # It assumes self.user_encoder is fitted and classes_ are available.
        user_idx_to_user_id_map = {}
        if hasattr(self.user_encoder, 'classes_') and self.user_encoder.classes_ is not None and len(self.user_encoder.classes_) > 0:
            user_idx_to_user_id_map = pd.Series(
                self.user_encoder.classes_,
                index=self.user_encoder.transform(self.user_encoder.classes_)
            ).to_dict()
        
        # Groups interactions by the integer 'user_idx'. This is generally faster than grouping by strings.
        # Assumes 'user_idx' column exists and is correctly populated in self.interactions.
        grouped_user_interactions = self.interactions.groupby('user_idx')

        # Iterates over each user group to generate negative samples.
        for user_idx_val, user_interactions_df in tqdm(grouped_user_interactions, desc="Creating negative samples"):
            # user_idx_val is the integer index of the current user.
            # user_interactions_df contains all interactions for this user.

            # Collects the string representations of item IDs the current user has interacted with.
            user_positive_items_str_set = set(user_interactions_df['item_id'].astype(str))
            
            if not user_positive_items_str_set:
                # Skips if the user (unexpectedly) has no positive items in their group.
                continue

            try:
                # Transforms the user's positive item strings to their integer indices.
                # This is a batch transformation for all positive items of the current user.
                user_positive_item_indices_arr = self.item_encoder.transform(list(user_positive_items_str_set))
            except ValueError:
                # Handles cases where some positive items might not be in the encoder.
                # print(f"Warning: Could not transform some positive items for user_idx {user_idx_val}.") # Optional
                continue
            
            # Creates a boolean mask to identify positive items within the global list of all item indices.
            is_positive_mask = np.zeros(len(all_item_indices_global), dtype=bool)
            is_positive_mask[user_positive_item_indices_arr] = True # Marks positive item indices.
            
            # Derives the array of available negative indices by inverting the mask.
            # These are integer indices of items not interacted with by the user.
            available_negative_indices = all_item_indices_global[~is_positive_mask]
            
            if len(available_negative_indices) == 0:
                # Skips user if no items are available for negative sampling (e.g., user interacted with all items).
                continue

            num_positives_for_user = len(user_positive_items_str_set)
            # Calculates the number of negative samples to generate for this user.
            num_negatives_to_sample = int(num_positives_for_user * self.negative_sampling_ratio)
            # Ensures not to sample more negative items than are available.
            num_negatives_to_sample = min(num_negatives_to_sample, len(available_negative_indices))

            if num_negatives_to_sample > 0:
                # Randomly samples integer indices from the available negative indices without replacement.
                sampled_item_idx_vals_arr = np.random.choice(
                    available_negative_indices,
                    num_negatives_to_sample,
                    replace=False
                )

                try:
                    # Converts the sampled item indices back to their original string IDs for record-keeping.
                    # This is a batch operation on the (small) list of sampled negative indices.
                    sampled_negatives_str_list = self.item_encoder.inverse_transform(sampled_item_idx_vals_arr)
                except ValueError:
                    # Handles cases where some sampled indices might not be in the encoder's inverse map.
                    # print(f"Warning: Could not inverse_transform some sampled negative indices for user_idx {user_idx_val}.") # Optional
                    continue
                
                # Retrieves the original string user ID for the current user_idx.
                user_id_str = user_idx_to_user_id_map.get(user_idx_val)
                if user_id_str is None:
                    # Fallback or warning if mapping is incomplete.
                    # print(f"Warning: user_id_str not found for user_idx {user_idx_val}. Using user_idx as string.") # Optional
                    user_id_str = str(user_idx_val) 
                
                # Appends negative sample data as tuples to the records list.
                for neg_item_str, item_idx_val in zip(sampled_negatives_str_list, sampled_item_idx_vals_arr):
                    negative_sample_records.append((
                        user_id_str,
                        neg_item_str,
                        user_idx_val,
                        item_idx_val,
                        0 # label for negative samples
                    ))
        
        # Defines columns for the negative samples DataFrame.
        negative_df_columns = ['user_id', 'item_id', 'user_idx', 'item_idx', 'label']
        # Converts the list of tuples (records) into a pandas DataFrame.
        # If no negative samples were generated, an empty DataFrame with columns from positive_df is created
        # to ensure consistent structure for concatenation.
        negative_df = pd.DataFrame(negative_sample_records, columns=negative_df_columns) if negative_sample_records else pd.DataFrame(columns=positive_df.columns)
        
        # Concatenates the DataFrame of positive interactions with the DataFrame of negative samples.
        all_samples = pd.concat([positive_df, negative_df], ignore_index=True)
        
        # Shuffles all samples randomly for unbiased training and resets the DataFrame index.
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
