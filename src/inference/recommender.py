# src/inference/recommender.py - Simplified without complex caching
"""
Simplified multimodal recommender for inference with streamlined caching
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
    Simplified multimodal recommender for inference
    Removed complex caching in favor of simple in-memory cache
    """
    
    def __init__(self, model: torch.nn.Module, dataset: MultimodalDataset, device: torch.device,
                 cache_max_items: int = 1000, cache_dir: Optional[str] = None, cache_to_disk: bool = False):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()
        
        # Simple in-memory cache
        self.feature_cache = {}
        self.cache_max_items = cache_max_items
        
        # Item info for quick lookup
        self.item_info_dict = self.dataset.item_info_df.set_index('item_id').to_dict('index')
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_recommendations(self, user_id: int, top_k: int = 10, 
                          filter_seen: bool = True, candidates: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Get top-K recommendations for a user
        
        Args:
            user_id: User ID to generate recommendations for
            top_k: Number of recommendations to return
            filter_seen: Whether to filter out items the user has already interacted with
            candidates: Optional list of candidate items to consider
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        try:
            # Encode user ID
            if user_id not in self.dataset.user_encoder.classes_:
                self.logger.warning(f"User {user_id} not found in training data")
                return []
            
            user_encoded = self.dataset.user_encoder.transform([user_id])[0]
            user_tensor = torch.tensor([user_encoded], dtype=torch.long, device=self.device)
            
            # Get candidate items
            if candidates is None:
                candidate_items = self.dataset.item_encoder.classes_.tolist()
            else:
                candidate_items = [item for item in candidates 
                                 if item in self.dataset.item_encoder.classes_]
            
            if not candidate_items:
                return []
            
            # Filter seen items if requested
            if filter_seen:
                seen_items = self._get_user_interactions(user_id)
                candidate_items = [item for item in candidate_items if item not in seen_items]
            
            if not candidate_items:
                return []
            
            # Score all candidate items
            item_scores = []
            
            # Process in batches to manage memory
            batch_size = 32
            for i in range(0, len(candidate_items), batch_size):
                batch_items = candidate_items[i:i + batch_size]
                batch_scores = self._score_items_batch(user_tensor, batch_items)
                
                for item_id, score in zip(batch_items, batch_scores):
                    item_scores.append((item_id, float(score)))
            
            # Sort by score and return top-K
            item_scores.sort(key=lambda x: x[1], reverse=True)
            return item_scores[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []
    
    def get_item_score(self, user_id: int, item_id: int) -> float:
        """
        Get the score for a specific user-item pair
        Required by ranking evaluator
        """
        try:
            # Encode user and item IDs
            if user_id not in self.dataset.user_encoder.classes_:
                return 0.0
            if item_id not in self.dataset.item_encoder.classes_:
                return 0.0
            
            user_encoded = self.dataset.user_encoder.transform([user_id])[0]
            user_tensor = torch.tensor([user_encoded], dtype=torch.long, device=self.device)
            
            # Get item features and score
            score = self._score_items_batch(user_tensor, [item_id])[0]
            return float(score)
            
        except Exception as e:
            self.logger.error(f"Error getting score for user {user_id}, item {item_id}: {e}")
           return 0.0
   
   def _score_items_batch(self, user_tensor: torch.Tensor, item_ids: List[int]) -> List[float]:
       """Score a batch of items for a given user"""
       try:
           batch_size = len(item_ids)
           user_batch = user_tensor.repeat(batch_size, 1).squeeze()
           
           # Prepare item data
           item_tensors = []
           images = []
           text_input_ids = []
           text_attention_masks = []
           numerical_features = []
           
           for item_id in item_ids:
               # Get or compute item features
               item_features = self._get_item_features(item_id)
               if item_features is None:
                   # Return 0 score for items we can't process
                   return [0.0] * batch_size
               
               item_encoded = self.dataset.item_encoder.transform([item_id])[0]
               item_tensors.append(item_encoded)
               images.append(item_features['image'])
               text_input_ids.append(item_features['text_input_ids'])
               text_attention_masks.append(item_features['text_attention_mask'])
               numerical_features.append(item_features['numerical_features'])
           
           # Convert to tensors
           item_batch = torch.tensor(item_tensors, dtype=torch.long, device=self.device)
           image_batch = torch.stack(images).to(self.device)
           text_ids_batch = torch.stack(text_input_ids).to(self.device)
           text_masks_batch = torch.stack(text_attention_masks).to(self.device)
           numerical_batch = torch.stack(numerical_features).to(self.device)
           
           # Get scores from model
           with torch.no_grad():
               scores = self.model(
                   user_idx=user_batch,
                   item_idx=item_batch,
                   image=image_batch,
                   text_input_ids=text_ids_batch,
                   text_attention_mask=text_masks_batch,
                   numerical_features=numerical_batch,
                   return_embeddings=False
               )
               return scores.squeeze().cpu().tolist()
               
       except Exception as e:
           self.logger.error(f"Error scoring batch: {e}")
           return [0.0] * len(item_ids)
   
   def _get_item_features(self, item_id: int) -> Optional[Dict[str, torch.Tensor]]:
       """Get or compute features for an item with simple caching"""
       # Check cache first
       if item_id in self.feature_cache:
           return self.feature_cache[item_id]
       
       try:
           # Get item info
           if item_id not in self.item_info_dict:
               return None
           
           item_info = self.item_info_dict[item_id]
           
           # Get features from dataset
           try:
               # Use dataset's feature extraction
               features = self.dataset._process_item_features(str(item_id))
               
               # Cache the features if we have room
               if len(self.feature_cache) < self.cache_max_items:
                   self.feature_cache[item_id] = features
               
               return features
               
           except Exception as e:
               self.logger.error(f"Error extracting features for item {item_id}: {e}")
               return None
               
       except Exception as e:
           self.logger.error(f"Error getting item features for {item_id}: {e}")
           return None
   
   def _get_user_interactions(self, user_id: int) -> set:
       """Get set of items the user has interacted with"""
       try:
           # Look in the dataset's interactions
           if hasattr(self.dataset, 'interactions'):
               user_interactions = self.dataset.interactions[
                   self.dataset.interactions['user_id'] == user_id
               ]['item_id'].tolist()
               return set(user_interactions)
           return set()
       except Exception as e:
           self.logger.error(f"Error getting user interactions for {user_id}: {e}")
           return set()
   
   def print_cache_stats(self):
       """Print cache statistics"""
       print(f"Feature cache: {len(self.feature_cache)} items cached")
       print(f"Cache capacity: {self.cache_max_items}")
       
   def clear_cache(self):
       """Clear the feature cache"""
       self.feature_cache.clear()
       print("Feature cache cleared")