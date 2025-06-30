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

        Args:
            model: The trained PyTorch multimodal recommender model.
            dataset: An instance of `MultimodalDataset` providing access to item features,
                     user/item encoders, and interaction history.
            device: The PyTorch device (e.g., 'cpu' or 'cuda') on which the model
                    and tensors should reside for inference.
            cache_max_items: The maximum number of item features to store in the
                             in-memory cache (Least Recently Used policy implied).
            cache_dir: An optional directory path for on-disk caching of features (currently not used).
                       This parameter is kept for compatibility but the current cache is in-memory only.
            cache_to_disk: A boolean indicating whether to use disk caching (currently not used).
                           This parameter is kept for compatibility.
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        # Sets the model to evaluation mode to disable dropout and batch normalization updates.
        self.model.eval()

        # Initializes a simple in-memory dictionary for caching item features.
        self.feature_cache: Dict[str, Dict[str, torch.Tensor]] = {} # Item IDs (strings) as keys
        self.cache_max_items = cache_max_items

        # Prepares a dictionary for quick lookup of item metadata, ensuring string keys.
        self.item_info_dict = {}
        if hasattr(self.dataset, 'item_info_df_original') and not self.dataset.item_info_df_original.empty:
            # Converts the 'item_id' column to string type and sets it as the index for dictionary conversion.
            item_df = self.dataset.item_info_df_original.copy()
            item_df['item_id'] = item_df['item_id'].astype(str)
            self.item_info_dict = item_df.set_index('item_id').to_dict('index')

        # Configures logging for the class.
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_recommendations(self, user_id: str, top_k: int = 10,
                          filter_seen: bool = True, candidates: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Generates a list of top-K item recommendations for a specified user.

        This method retrieves a set of candidate items, optionally filters out
        items the user has already interacted with, scores the remaining
        candidates using the multimodal model, and returns the top-K items
        sorted by their predicted relevance scores.

        Args:
            user_id: The ID (string) of the user for whom to generate recommendations.
            top_k: The desired number of recommendations to return.
            filter_seen: A boolean flag. If True, items that the user has
                         already interacted with in their history will be
                         excluded from the recommendation list.
            candidates: An optional list of specific item IDs (strings) to
                        consider as candidates for recommendation. If None,
                        all items known to the system will be considered.

        Returns:
            A list of (item_id_string, score_float) tuples, sorted in
            descending order of score. Returns an empty list if no
            recommendations can be generated (e.g., user not found, no valid candidates).
        """
        try:
            # Ensures the user ID is a string for consistent processing.
            user_id = str(user_id)
            
            # Checks if the user encoder is properly initialized in the dataset.
            if not hasattr(self.dataset.user_encoder, 'classes_') or self.dataset.user_encoder.classes_ is None:
                self.logger.warning(f"User encoder not properly initialized. Cannot generate recommendations.")
                return []
                
            # Retrieves all unique user IDs from the encoder classes.
            user_classes = [str(cls) for cls in self.dataset.user_encoder.classes_]
            if user_id not in user_classes:
                self.logger.warning(f"User '{user_id}' not found in the trained user encoder. Cannot generate recommendations.")
                return []

            # Encodes the user ID to its numerical index for model input.
            user_encoded = self.dataset.user_encoder.transform([user_id])[0]
            # Converts the encoded user ID to a PyTorch tensor and moves it to the specified device.
            user_tensor = torch.tensor([user_encoded], dtype=torch.long, device=self.device)

            # Determines the set of candidate items to score.
            current_candidate_items: List[str] = []
            if candidates is None:
                # If no specific candidates are provided, uses all items from the item encoder.
                if (hasattr(self.dataset.item_encoder, 'classes_') and 
                    self.dataset.item_encoder.classes_ is not None):
                    current_candidate_items = [str(item) for item in self.dataset.item_encoder.classes_]
                else:
                    self.logger.warning(f"Item encoder not properly initialized. No items available as candidates.")
                    return []
            else:
                # If candidates are provided, filters them to ensure they exist in the item encoder.
                item_classes = [str(cls) for cls in self.dataset.item_encoder.classes_] if hasattr(self.dataset.item_encoder, 'classes_') else []
                current_candidate_items = [str(item) for item in candidates if str(item) in item_classes]

            if not current_candidate_items:
                self.logger.info(f"No valid candidate items found for user '{user_id}'. Returning empty recommendations.")
                return []

            # Filters out items the user has already seen if `filter_seen` is True.
            if filter_seen:
                seen_items: set[str] = self._get_user_interactions(user_id)
                current_candidate_items = [item for item in current_candidate_items if item not in seen_items]

            if not current_candidate_items:
                self.logger.info(f"All candidate items for user '{user_id}' have been seen or filtered. Returning empty recommendations.")
                return []

            # Scores all candidate items in batches to manage memory and computational load.
            item_scores: List[Tuple[str, float]] = []
            batch_size = 32 # Defines the batch size for scoring items.
            for i in range(0, len(current_candidate_items), batch_size):
                batch_item_ids: List[str] = current_candidate_items[i:i + batch_size]
                # Calls the internal method to score a batch of items.
                batch_scores_float: List[float] = self._score_items_batch(user_tensor, batch_item_ids)

                # Combines item IDs with their scores.
                for item_id_str, score_float in zip(batch_item_ids, batch_scores_float):
                    item_scores.append((item_id_str, score_float))

            # Sorts the items by their predicted scores in descending order.
            item_scores.sort(key=lambda x: x[1], reverse=True)
            # Returns only the top-K recommendations.
            return item_scores[:top_k]

        except Exception as e:
            self.logger.error(f"An unexpected error occurred while generating recommendations for user '{user_id}': {e}", exc_info=True)
            return []

    def get_item_score(self, user_id: str, item_id: str) -> float:
        """
        Retrieves the predicted relevance score for a single user-item pair.

        This method is designed for use in evaluation tasks (e.g., ranking evaluation)
        where individual item scores are needed rather than a full list of recommendations.

        Args:
            user_id: The ID (string) of the user.
            item_id: The ID (string) of the item.

        Returns:
            The predicted relevance score (float) for the user-item pair. Returns 0.0
            if the user or item is not found in the encoders or if an error occurs.
        """
        try:
            # Ensures both user and item IDs are strings.
            user_id = str(user_id)
            item_id = str(item_id)
            
            # Validates that user and item encoders are properly initialized.
            if not hasattr(self.dataset.user_encoder, 'classes_') or self.dataset.user_encoder.classes_ is None:
                self.logger.warning(f"User encoder not properly initialized. Cannot get score for user '{user_id}'.")
                return 0.0
                
            if not hasattr(self.dataset.item_encoder, 'classes_') or self.dataset.item_encoder.classes_ is None:
                self.logger.warning(f"Item encoder not properly initialized. Cannot get score for item '{item_id}'.")
                return 0.0
            
            # Retrieves all unique user and item IDs from their respective encoders.
            user_classes = [str(cls) for cls in self.dataset.user_encoder.classes_]
            item_classes = [str(cls) for cls in self.dataset.item_encoder.classes_]
            
            # Checks if the user and item exist in the encoder's vocabulary.
            if user_id not in user_classes:
                self.logger.warning(f"User '{user_id}' not found in user encoder. Cannot get score.")
                return 0.0
            if item_id not in item_classes:
                self.logger.warning(f"Item '{item_id}' not found in item encoder. Cannot get score.")
                return 0.0

            # Encodes the user ID to its numerical index.
            user_encoded = self.dataset.user_encoder.transform([user_id])[0]
            # Converts the encoded user ID to a PyTorch tensor and moves it to the device.
            user_tensor = torch.tensor([user_encoded], dtype=torch.long, device=self.device)

            # Scores the single item using the batch scoring method.
            scores_list: List[float] = self._score_items_batch(user_tensor, [item_id])
            if scores_list:
                return scores_list[0] # Returns the single score from the list.
            return 0.0

        except Exception as e:
            self.logger.error(f"An unexpected error occurred while getting score for user '{user_id}', item '{item_id}': {e}", exc_info=True)
            return 0.0

    def _score_items_batch(self, user_tensor: torch.Tensor, item_ids_str: List[str]) -> List[float]:
        """
        Scores a batch of items for a given user using the trained model.

        This internal method prepares the input tensors for the model (user, item,
        and multimodal features for each item) and performs a forward pass to
        obtain relevance scores. It handles cases where some items in the batch
        might be invalid or missing features.

        Args:
            user_tensor: A PyTorch tensor representing the encoded user ID.
                         This tensor is typically of shape `(1,)` or `(batch_size,)`.
            item_ids_str: A list of item IDs (strings) to be scored in the current batch.

        Returns:
            A list of float scores corresponding to the input `item_ids_str`,
            in the same order. Items that could not be processed will have a score of 0.0.
        """
        try:
            batch_size = len(item_ids_str)
            if batch_size == 0:
                return []
                
            # Repeats the user tensor to match the batch size for item processing.
            user_batch = user_tensor.repeat(batch_size, 1).squeeze()
            if user_batch.dim() == 0: # Handles the edge case where batch_size is 1, repeat might not expand dimensions.
                user_batch = user_tensor 

            # Initializes lists to accumulate prepared item data for the batch.
            item_idx_tensors_list = [] # Stores integer indices for model input.
            images_list = []
            text_input_ids_list = []
            text_attention_masks_list = []
            numerical_features_list = []
            tag_idx_list = [] # CHANGED: Added list for tag indices.
            
            # Initializes lists for CLIP-specific text inputs, if applicable to the model.
            clip_text_input_ids_list = []
            clip_text_attention_masks_list = []
            
            valid_item_ids_for_batch = [] # Tracks item IDs that were successfully processed for the batch.

            # Iterates through each item in the batch to prepare its features.
            for item_id_str in item_ids_str:
                item_id_str = str(item_id_str) # Ensures item ID is a string.
                
                # Retrieves or computes multimodal features for the current item.
                item_features = self._get_item_features(item_id_str)
                if item_features is None:
                    # If features cannot be obtained, logs a warning and skips the item in the batch.
                    self.logger.warning(f"Could not get features for item '{item_id_str}'. It will be skipped in this batch.")
                    continue 

                # Encodes the item ID to its numerical index.
                try:
                    item_encoded_idx = self.dataset.item_encoder.transform([item_id_str])[0]
                except ValueError:
                    self.logger.warning(f"Item '{item_id_str}' not found in item encoder. Skipping it in this batch.")
                    continue
                    
                # Appends all prepared features to their respective lists.
                item_idx_tensors_list.append(item_encoded_idx)
                images_list.append(item_features['image'])
                text_input_ids_list.append(item_features['text_input_ids'])
                text_attention_masks_list.append(item_features['text_attention_mask'])
                numerical_features_list.append(item_features['numerical_features'])
                
                # CHANGED: Added logic to append tag index.
                if 'tag_idx' in item_features:
                    tag_idx_list.append(item_features['tag_idx'])

                # Appends CLIP-specific inputs if they are present in the item features.
                if 'clip_text_input_ids' in item_features and 'clip_text_attention_mask' in item_features:
                    clip_text_input_ids_list.append(item_features['clip_text_input_ids'])
                    clip_text_attention_masks_list.append(item_features['clip_text_attention_mask'])
                
                # Adds the item ID to the list of successfully processed items for this batch.
                valid_item_ids_for_batch.append(item_id_str)

            # Returns zero scores for all original items requested if no items could be validly processed.
            if not valid_item_ids_for_batch: 
                return [0.0] * len(item_ids_str) 

            # Stacks the collected lists into single tensors and moves them to the device.
            item_batch_indices = torch.tensor(item_idx_tensors_list, dtype=torch.long, device=self.device)
            image_batch = torch.stack(images_list).to(self.device)
            text_ids_batch = torch.stack(text_input_ids_list).to(self.device)
            text_masks_batch = torch.stack(text_attention_masks_list).to(self.device)
            numerical_batch = torch.stack(numerical_features_list).to(self.device)
            
            # CHANGED: Stack the tag indices into a tensor.
            tag_idx_batch = torch.stack(tag_idx_list).to(self.device) if tag_idx_list else None
            
            # Prepares the dictionary of inputs for the model's forward pass.
            model_input_dict = {
                'user_idx': user_batch,
                'item_idx': item_batch_indices,
                'image': image_batch,
                'text_input_ids': text_ids_batch,
                'text_attention_mask': text_masks_batch,
                'numerical_features': numerical_batch,
                'return_embeddings': False # Indicates that only scores are needed from the model.
            }

            # CHANGED: Add tag_idx to the model input dictionary.
            if tag_idx_batch is not None:
                model_input_dict['tag_idx'] = tag_idx_batch

            # Adds CLIP-specific inputs to the model dictionary if they were collected.
            if clip_text_input_ids_list and clip_text_attention_masks_list:
                model_input_dict['clip_text_input_ids'] = torch.stack(clip_text_input_ids_list).to(self.device)
                model_input_dict['clip_text_attention_mask'] = torch.stack(clip_text_attention_masks_list).to(self.device)

            # Performs the forward pass through the model without gradient tracking.
            with torch.no_grad():
                scores_tensor = self.model(**model_input_dict)
                
            # Extracts scores, converts them to a Python list, and handles single-item batches.
            scores_for_valid_items = scores_tensor.squeeze().cpu().tolist()
            if not isinstance(scores_for_valid_items, list): 
                scores_for_valid_items = [scores_for_valid_items]

            # Maps the scores back to the order of the original `item_ids_str` list.
            # Items that failed processing will receive a score of 0.0.
            final_scores_map = {item_id_str: score for item_id_str, score in zip(valid_item_ids_for_batch, scores_for_valid_items)}
            results = [final_scores_map.get(original_id, 0.0) for original_id in item_ids_str]
            return results

        except Exception as e:
            self.logger.error(f"An unexpected error occurred while scoring batch for user (shape: {user_tensor.shape if hasattr(user_tensor, 'shape') else 'N/A'}) and items {item_ids_str}: {e}", exc_info=True)
            # Returns a list of zero scores if a critical error occurs during batch scoring.
            return [0.0] * len(item_ids_str)

    def _get_item_features(self, item_id_str: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieves or computes multimodal features for a single item ID, utilizing a cache.

        This method first checks the in-memory cache for the item's features.
        If not found, it delegates to the dataset's internal feature processing
        logic, then caches the result. It provides robust error handling and logging.

        Args:
            item_id_str: The ID (string) of the item for which to retrieve features.

        Returns:
            A dictionary of PyTorch tensors representing the item's multimodal features,
            or None if the item is not found in metadata or feature processing fails.
        """
        item_id_str = str(item_id_str) # Ensures the item ID is a string.
        
        # Returns features directly from the cache if available.
        if item_id_str in self.feature_cache:
            return self.feature_cache[item_id_str]

        try:
            # Checks if the item ID exists in the pre-loaded item metadata.
            if item_id_str not in self.item_info_dict:
                self.logger.warning(
                    f"Item '{item_id_str}' not found in the recommender's item metadata. "
                    f"This indicates a data integrity issue where an item ID is requested "
                    f"but no corresponding metadata is available."
                )
                return None

            # Delegates to the dataset to process and retrieve the item's features.
            features = self.dataset._get_item_features(item_id_str)

            if features is None:
                 self.logger.warning(
                    f"Feature processing for item '{item_id_str}' returned None. "
                    f"This could be due to a missing image file, corrupted data, or "
                    f"an issue during feature extraction for this specific item."
                 )
                 return None

            # Caches the retrieved features if the cache has capacity.
            if len(self.feature_cache) < self.cache_max_items:
                self.feature_cache[item_id_str] = features

            return features

        except Exception as e:
            self.logger.error(f"An unexpected error occurred while getting features for item '{item_id_str}': {e}", exc_info=True)
            return None

    def _get_user_interactions(self, user_id_str: str) -> set[str]:
        """
        Retrieves the set of items (as strings) that a given user has historically interacted with.

        This method wraps the `get_user_history` method of the `MultimodalDataset`
        to ensure robust handling of user IDs and provide logging for errors.

        Args:
            user_id_str: The ID (string) of the user.

        Returns:
            A set of item IDs (string) that the user has interacted with.
            Returns an empty set if an error occurs or the user has no history.
        """
        try:
            user_id_str = str(user_id_str) # Ensures the user ID is a string.
            # Delegates to the dataset's method to retrieve user history.
            return self.dataset.get_user_history(user_id_str)
        except Exception as e:
            self.logger.error(f"Error getting user interactions for user '{user_id_str}': {e}", exc_info=True)
            return set()

    def print_cache_stats(self):
        """
        Prints current statistics about the in-memory feature cache.

        This includes the number of items currently cached and the maximum
        capacity of the cache.
        """
        print(f"Feature cache: {len(self.feature_cache)} items cached")
        print(f"Cache capacity: {self.cache_max_items}")

    def clear_cache(self):
        """
        Clears all items from the in-memory feature cache.

        This can be useful for freeing up memory or for ensuring a fresh
        start for feature loading.
        """
        self.feature_cache.clear()
        print("Feature cache cleared")