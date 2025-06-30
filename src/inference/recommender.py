"""
Provides the core recommender class for generating recommendations during inference.

This module defines the `Recommender` class, which integrates a trained multimodal model
with the dataset's feature processing and encoding capabilities. It handles
retrieving item features, scoring items for a given user, and generating
top-K recommendations, with support for filtering seen items and using
an in-memory cache for efficiency.
"""
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import logging

from ..data.dataset import MultimodalDataset


class Recommender:
    """
    Manages the generation of recommendations using a trained multimodal model.

    This class provides a unified interface for inference, allowing for scoring
    individual items or generating a list of top-K recommendations for a given user.
    It incorporates mechanisms for efficient feature retrieval, including an
    in-memory cache for frequently accessed item features.
    """

    def __init__(self, model: torch.nn.Module, dataset: MultimodalDataset, device: torch.device,
                 cache_max_items: int = 1000, cache_dir: Optional[str] = None, cache_to_disk: bool = False):
        """
        Initializes the Recommender for inference.
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()

        self.feature_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.cache_max_items = cache_max_items

        self.item_info_dict = {}
        if hasattr(self.dataset, 'item_info_df_original') and not self.dataset.item_info_df_original.empty:
            item_df = self.dataset.item_info_df_original.copy()
            item_df['item_id'] = item_df['item_id'].astype(str)
            self.item_info_dict = item_df.set_index('item_id').to_dict('index')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_recommendations(self, user_id: str, top_k: int = 10,
                          filter_seen: bool = True, candidates: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generates a list of top-K item recommendations for a specified user.
        """
        try:
            user_id = str(user_id)
            
            if not hasattr(self.dataset.user_encoder, 'classes_') or self.dataset.user_encoder.classes_ is None:
                self.logger.warning(f"User encoder not properly initialized.")
                return []
                
            user_classes = [str(cls) for cls in self.dataset.user_encoder.classes_]
            if user_id not in user_classes:
                self.logger.warning(f"User '{user_id}' not found in the trained user encoder.")
                return []

            user_encoded = self.dataset.user_encoder.transform([user_id])[0]
            user_tensor = torch.tensor([user_encoded], dtype=torch.long, device=self.device)

            current_candidate_items: List[str] = []
            if candidates is None:
                if (hasattr(self.dataset.item_encoder, 'classes_') and 
                    self.dataset.item_encoder.classes_ is not None):
                    current_candidate_items = [str(item) for item in self.dataset.item_encoder.classes_]
                else:
                    self.logger.warning(f"Item encoder not properly initialized.")
                    return []
            else:
                item_classes = [str(cls) for cls in self.dataset.item_encoder.classes_] if hasattr(self.dataset.item_encoder, 'classes_') else []
                current_candidate_items = [str(item) for item in candidates if str(item) in item_classes]

            if not current_candidate_items:
                self.logger.info(f"No valid candidate items found for user '{user_id}'.")
                return []

            if filter_seen:
                seen_items: set[str] = self._get_user_interactions(user_id)
                current_candidate_items = [item for item in current_candidate_items if item not in seen_items]

            if not current_candidate_items:
                self.logger.info(f"All candidate items for user '{user_id}' have been filtered.")
                return []

            item_scores: List[Tuple[str, float]] = []
            batch_size = 256 # Increased batch size for better performance
            for i in range(0, len(current_candidate_items), batch_size):
                batch_item_ids: List[str] = current_candidate_items[i:i + batch_size]
                batch_scores_float: List[float] = self._score_items_batch(user_tensor, batch_item_ids)

                for item_id_str, score_float in zip(batch_item_ids, batch_scores_float):
                    item_scores.append((item_id_str, score_float))

            item_scores.sort(key=lambda x: x[1], reverse=True)
            return item_scores[:top_k]

        except Exception as e:
            self.logger.error(f"Error generating recommendations for user '{user_id}': {e}", exc_info=True)
            return []

    def get_item_score(self, user_id: str, item_id: str) -> float:
        """
        Retrieves the predicted relevance score for a single user-item pair.
        """
        try:
            user_id = str(user_id)
            item_id = str(item_id)
            
            if not hasattr(self.dataset.user_encoder, 'classes_') or self.dataset.user_encoder.classes_ is None:
                return 0.0
            if not hasattr(self.dataset.item_encoder, 'classes_') or self.dataset.item_encoder.classes_ is None:
                return 0.0
            
            user_classes = [str(cls) for cls in self.dataset.user_encoder.classes_]
            item_classes = [str(cls) for cls in self.dataset.item_encoder.classes_]
            
            if user_id not in user_classes or item_id not in item_classes:
                return 0.0

            user_encoded = self.dataset.user_encoder.transform([user_id])[0]
            user_tensor = torch.tensor([user_encoded], dtype=torch.long, device=self.device)

            scores_list: List[float] = self._score_items_batch(user_tensor, [item_id])
            if scores_list:
                return scores_list[0]
            return 0.0

        except Exception as e:
            self.logger.error(f"Error getting score for user '{user_id}', item '{item_id}': {e}", exc_info=True)
            return 0.0

    # --- START OF MODIFIED SECTION ---
    def _score_items_batch(self, user_tensor: torch.Tensor, item_ids_str: List[str]) -> List[float]:
        """
        Scores a batch of items for a given user using a fully batched approach.

        This method processes all items in the list simultaneously, creating a single
        batch of tensors to pass to the model. This is significantly more efficient
        than processing items in a loop.
        """
        try:
            batch_size = len(item_ids_str)
            if batch_size == 0:
                return []

            # 1. Get features for all items in the batch
            item_features_list = []
            valid_item_ids_for_batch = []
            
            # This loop is fast as it primarily hits the cache
            for item_id in item_ids_str:
                features = self._get_item_features(item_id)
                if features:
                    item_features_list.append(features)
                    valid_item_ids_for_batch.append(item_id)

            if not valid_item_ids_for_batch:
                self.logger.warning("No valid items found in the batch to score.")
                return [0.0] * len(item_ids_str)
            
            # 2. Collate the list of feature dicts into a single batch dict of tensors
            # This stacks the tensors from each item's feature dictionary
            first_item_keys = item_features_list[0].keys()
            collated_item_features = {
                key: torch.stack([d[key] for d in item_features_list])
                for key in first_item_keys
            }

            # 3. Prepare user tensor and create the final model input batch
            actual_batch_size = len(valid_item_ids_for_batch)
            user_batch = user_tensor.repeat(actual_batch_size)

            model_input_dict = {
                'user_idx': user_batch,
                **collated_item_features
            }
            model_input_dict = {k: v.to(self.device) for k, v in model_input_dict.items()}

            # Add item_idx, which is also a feature
            item_encoded_indices = self.dataset.item_encoder.transform(valid_item_ids_for_batch)
            model_input_dict['item_idx'] = torch.tensor(item_encoded_indices, dtype=torch.long, device=self.device)

            # 4. Perform a single, efficient, batched forward pass
            with torch.no_grad():
                scores_tensor = self.model(**model_input_dict)
            
            scores_for_valid_items = scores_tensor.squeeze().cpu().tolist()
            if not isinstance(scores_for_valid_items, list):
                scores_for_valid_items = [scores_for_valid_items]

            # 5. Map scores back to the original item list, assigning 0.0 to failed items
            final_scores_map = {item_id: score for item_id, score in zip(valid_item_ids_for_batch, scores_for_valid_items)}
            results = [final_scores_map.get(original_id, 0.0) for original_id in item_ids_str]
            
            return results

        except Exception as e:
            self.logger.error(f"Error during batched scoring for user (tensor shape: {user_tensor.shape}): {e}", exc_info=True)
            return [0.0] * len(item_ids_str)
    # --- END OF MODIFIED SECTION ---

    def _get_item_features(self, item_id_str: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieves or computes multimodal features for a single item ID, utilizing a cache.
        """
        item_id_str = str(item_id_str)
        
        if self.dataset.feature_cache and self.dataset.feature_cache.get(item_id_str):
             return self.dataset.feature_cache.get(item_id_str)
        
        if item_id_str in self.feature_cache:
            return self.feature_cache[item_id_str]

        try:
            if item_id_str not in self.item_info_dict:
                self.logger.warning(f"Item '{item_id_str}' not found in metadata.")
                return None

            features = self.dataset._get_item_features(item_id_str)

            if features is None:
                self.logger.warning(f"Feature processing for item '{item_id_str}' returned None.")
                return None

            if len(self.feature_cache) < self.cache_max_items:
                self.feature_cache[item_id_str] = features

            return features

        except Exception as e:
            self.logger.error(f"Error getting features for item '{item_id_str}': {e}", exc_info=True)
            return None

    def _get_user_interactions(self, user_id_str: str) -> set[str]:
        """
        Retrieves the set of items that a given user has historically interacted with.
        """
        try:
            user_id_str = str(user_id_str)
            return self.dataset.get_user_history(user_id_str)
        except Exception as e:
            self.logger.error(f"Error getting user interactions for user '{user_id_str}': {e}", exc_info=True)
            return set()

    def print_cache_stats(self):
        """
        Prints current statistics about the in-memory feature cache.
        """
        print(f"Feature cache: {len(self.feature_cache)} items cached")
        print(f"Cache capacity: {self.cache_max_items}")

    def clear_cache(self):
        """
        Clears all items from the in-memory feature cache.
        """
        self.feature_cache.clear()
        print("Feature cache cleared")