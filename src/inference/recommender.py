# src/inference/recommender.py
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import logging

from ..data.dataset import MultimodalDataset


class Recommender:
    """
    Multimodal recommender for inference with consistent string ID handling
    """

    def __init__(self, model: torch.nn.Module, dataset: MultimodalDataset, device: torch.device,
                 cache_max_items: int = 1000, cache_dir: Optional[str] = None, cache_to_disk: bool = False):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()

        # Simple in-memory cache
        self.feature_cache: Dict[str, Dict[str, torch.Tensor]] = {} # Item IDs (strings) as keys
        self.cache_max_items = cache_max_items

        # Item info for quick lookup - ensure string keys
        self.item_info_dict = {}
        if hasattr(self.dataset, 'item_info_df_original') and not self.dataset.item_info_df_original.empty:
            # Convert index to string and create dictionary
            item_df = self.dataset.item_info_df_original.copy()
            item_df['item_id'] = item_df['item_id'].astype(str)
            self.item_info_dict = item_df.set_index('item_id').to_dict('index')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_recommendations(self, user_id: str, top_k: int = 10,
                          filter_seen: bool = True, candidates: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Get top-K recommendations for a user

        Args:
            user_id: User ID (string) to generate recommendations for
            top_k: Number of recommendations to return
            filter_seen: Whether to filter out items the user has already interacted with
            candidates: Optional list of candidate item IDs (strings) to consider

        Returns:
            List of (item_id_str, score) tuples sorted by score descending
        """
        try:
            # Ensure user_id is string
            user_id = str(user_id)
            
            # Check if user exists in encoder
            if not hasattr(self.dataset.user_encoder, 'classes_') or self.dataset.user_encoder.classes_ is None:
                self.logger.warning(f"User encoder not properly initialized")
                return []
                
            user_classes = [str(cls) for cls in self.dataset.user_encoder.classes_]
            if user_id not in user_classes:
                self.logger.warning(f"User {user_id} not found in training data")
                return []

            user_encoded = self.dataset.user_encoder.transform([user_id])[0]
            user_tensor = torch.tensor([user_encoded], dtype=torch.long, device=self.device)

            # Get candidate items - ensure all are strings
            current_candidate_items: List[str] = []
            if candidates is None:
                if (hasattr(self.dataset.item_encoder, 'classes_') and 
                    self.dataset.item_encoder.classes_ is not None):
                    current_candidate_items = [str(item) for item in self.dataset.item_encoder.classes_]
                else:
                    return []
            else:
                # Ensure candidates are strings and exist in encoder
                item_classes = [str(cls) for cls in self.dataset.item_encoder.classes_] if hasattr(self.dataset.item_encoder, 'classes_') else []
                current_candidate_items = [str(item) for item in candidates if str(item) in item_classes]

            if not current_candidate_items:
                return []

            # Filter seen items if requested
            if filter_seen:
                seen_items: set[str] = self._get_user_interactions(user_id)
                current_candidate_items = [item for item in current_candidate_items if item not in seen_items]

            if not current_candidate_items:
                return []

            # Score all candidate items
            item_scores: List[Tuple[str, float]] = []

            # Process in batches to manage memory
            batch_size = 32
            for i in range(0, len(current_candidate_items), batch_size):
                batch_item_ids: List[str] = current_candidate_items[i:i + batch_size]
                batch_scores_float: List[float] = self._score_items_batch(user_tensor, batch_item_ids)

                for item_id_str, score_float in zip(batch_item_ids, batch_scores_float):
                    item_scores.append((item_id_str, score_float))

            # Sort by score and return top-K
            item_scores.sort(key=lambda x: x[1], reverse=True)
            return item_scores[:top_k]

        except Exception as e:
            self.logger.error(f"Error generating recommendations for user {user_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def get_item_score(self, user_id: str, item_id: str) -> float:
        """
        Get the score for a specific user-item pair.
        User ID and Item ID should be strings.
        Required by ranking evaluator.
        """
        try:
            # Ensure string types
            user_id = str(user_id)
            item_id = str(item_id)
            
            # Check if user and item exist in encoders
            if not hasattr(self.dataset.user_encoder, 'classes_') or self.dataset.user_encoder.classes_ is None:
                self.logger.warning(f"User encoder not properly initialized for get_item_score.")
                return 0.0
                
            if not hasattr(self.dataset.item_encoder, 'classes_') or self.dataset.item_encoder.classes_ is None:
                self.logger.warning(f"Item encoder not properly initialized for get_item_score.")
                return 0.0
            
            user_classes = [str(cls) for cls in self.dataset.user_encoder.classes_]
            item_classes = [str(cls) for cls in self.dataset.item_encoder.classes_]
            
            if user_id not in user_classes:
                self.logger.warning(f"User {user_id} not in encoder for get_item_score.")
                return 0.0
            if item_id not in item_classes:
                self.logger.warning(f"Item {item_id} not in encoder for get_item_score.")
                return 0.0

            user_encoded = self.dataset.user_encoder.transform([user_id])[0]
            user_tensor = torch.tensor([user_encoded], dtype=torch.long, device=self.device)

            # Get item features and score
            scores_list: List[float] = self._score_items_batch(user_tensor, [item_id])
            if scores_list:
                return scores_list[0]
            return 0.0

        except Exception as e:
            self.logger.error(f"Error getting score for user {user_id}, item {item_id}: {e}")
            return 0.0

    def _score_items_batch(self, user_tensor: torch.Tensor, item_ids_str: List[str]) -> List[float]:
        """Score a batch of items (strings) for a given user"""
        try:
            batch_size = len(item_ids_str)
            if batch_size == 0:
                return []
                
            user_batch = user_tensor.repeat(batch_size, 1).squeeze()
            if user_batch.dim() == 0: # If batch_size was 1, repeat might not add a dim
                 user_batch = user_tensor 

            # Prepare item data
            item_idx_tensors_list = [] # Stores integer indices for model
            images_list = []
            text_input_ids_list = []
            text_attention_masks_list = []
            numerical_features_list = []
            
            # CLIP specific inputs, if used by the model
            clip_text_input_ids_list = []
            clip_text_attention_masks_list = []
            
            valid_item_ids_for_batch = []

            for item_id_str in item_ids_str:
                # Ensure item_id_str is string
                item_id_str = str(item_id_str)
                
                # Get or compute item features
                item_features = self._get_item_features(item_id_str)
                if item_features is None:
                    # The warning is now more descriptive inside _get_item_features
                    self.logger.warning(f"Could not get features for item {item_id_str}, it will be skipped in batch.")
                    continue # Skip this item if features can't be loaded/processed

                # Transform item ID to index
                try:
                    item_encoded_idx = self.dataset.item_encoder.transform([item_id_str])[0]
                except ValueError:
                    self.logger.warning(f"Item {item_id_str} not found in item encoder, skipping.")
                    continue
                    
                item_idx_tensors_list.append(item_encoded_idx)
                
                images_list.append(item_features['image'])
                text_input_ids_list.append(item_features['text_input_ids'])
                text_attention_masks_list.append(item_features['text_attention_mask'])
                numerical_features_list.append(item_features['numerical_features'])

                if 'clip_text_input_ids' in item_features and 'clip_text_attention_mask' in item_features:
                    clip_text_input_ids_list.append(item_features['clip_text_input_ids'])
                    clip_text_attention_masks_list.append(item_features['clip_text_attention_mask'])
                
                valid_item_ids_for_batch.append(item_id_str)

            if not valid_item_ids_for_batch: # All items in batch were invalid
                return [0.0] * len(item_ids_str) # Return zero scores for original requested items

            # Convert to tensors
            item_batch_indices = torch.tensor(item_idx_tensors_list, dtype=torch.long, device=self.device)
            image_batch = torch.stack(images_list).to(self.device)
            text_ids_batch = torch.stack(text_input_ids_list).to(self.device)
            text_masks_batch = torch.stack(text_attention_masks_list).to(self.device)
            numerical_batch = torch.stack(numerical_features_list).to(self.device)
            
            model_input_dict = {
                'user_idx': user_batch,
                'item_idx': item_batch_indices,
                'image': image_batch,
                'text_input_ids': text_ids_batch,
                'text_attention_mask': text_masks_batch,
                'numerical_features': numerical_batch,
                'return_embeddings': False # For scoring, we don't need embeddings back
            }

            # Add CLIP specific inputs if they were collected
            if clip_text_input_ids_list and clip_text_attention_masks_list:
                model_input_dict['clip_text_input_ids'] = torch.stack(clip_text_input_ids_list).to(self.device)
                model_input_dict['clip_text_attention_mask'] = torch.stack(clip_text_attention_masks_list).to(self.device)

            # Get scores from model
            with torch.no_grad():
                scores_tensor = self.model(**model_input_dict)
                
            scores_for_valid_items = scores_tensor.squeeze().cpu().tolist()
            if not isinstance(scores_for_valid_items, list): # Handle single item case
                scores_for_valid_items = [scores_for_valid_items]

            # Map scores back to the original item_ids_str order, assigning 0 to failed items
            final_scores_map = {item_id_str: score for item_id_str, score in zip(valid_item_ids_for_batch, scores_for_valid_items)}
            results = [final_scores_map.get(original_id, 0.0) for original_id in item_ids_str]
            return results

        except Exception as e:
            self.logger.error(f"Error scoring batch for user_tensor {user_tensor.shape if hasattr(user_tensor, 'shape') else 'N/A'} and items {item_ids_str}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [0.0] * len(item_ids_str)

    def _get_item_features(self, item_id_str: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get or compute features for an item (string ID) with simple caching and improved logging."""
        item_id_str = str(item_id_str)
        
        if item_id_str in self.feature_cache:
            return self.feature_cache[item_id_str]

        try:
            if item_id_str not in self.item_info_dict:
                self.logger.warning(
                    f"Item '{item_id_str}' not found in the recommender's item metadata dictionary. "
                    f"This suggests a mismatch between the interaction data and item metadata files."
                )
                return None

            features = self.dataset._process_item_features(item_id_str)

            if features is None:
                 self.logger.warning(
                    f"Feature processing for item '{item_id_str}' returned None. "
                    f"This could be due to a missing image or other data issue for this specific item."
                 )
                 return None

            if len(self.feature_cache) < self.cache_max_items:
                self.feature_cache[item_id_str] = features

            return features

        except Exception as e:
            self.logger.error(f"An unexpected error occurred while getting features for item '{item_id_str}': {e}", exc_info=True)
            return None

    def _get_user_interactions(self, user_id_str: str) -> set[str]:
        """Get set of items (strings) the user (string ID) has interacted with"""
        try:
            # Ensure user_id_str is string
            user_id_str = str(user_id_str)
            
            # Ensure dataset's get_user_history is called with string ID
            return self.dataset.get_user_history(user_id_str) # dataset.get_user_history expects str
        except Exception as e:
            self.logger.error(f"Error getting user interactions for {user_id_str}: {e}")
            return set()

    def print_cache_stats(self):
        """Print cache statistics"""
        print(f"Feature cache: {len(self.feature_cache)} items cached")
        print(f"Cache capacity: {self.cache_max_items}")

    def clear_cache(self):
        """Clear the feature cache"""
        self.feature_cache.clear()
        print("Feature cache cleared")