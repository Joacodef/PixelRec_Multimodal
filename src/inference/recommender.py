# src/inference/recommender.py - Simplified version
"""
Simplified recommendation with single cache system
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from ..data.simple_cache import SimpleFeatureCache


class Recommender:
    """Simplified recommender with single cache"""
    
    def __init__(
        self,
        model: nn.Module,
        dataset,  # MultimodalDataset
        device: torch.device,
        cache_max_items: int = 1000,
        cache_dir: Optional[str] = None,
        cache_to_disk: bool = False
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        
        # Single unified cache
        self.feature_cache = SimpleFeatureCache(
            max_memory_items=cache_max_items,
            cache_dir=cache_dir,
            use_disk=cache_to_disk
        )
        
        self.model.eval()
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None,
        batch_size: int = 256
    ) -> List[Tuple[str, float]]:
        """Get top-k recommendations for a user"""
        
        # Get user index
        try:
            user_idx = self.dataset.user_encoder.transform([user_id])[0]
        except:
            return []
        
        # Get candidate items
        if candidates is None:
            candidates = list(self.dataset.item_encoder.classes_)
        
        # Filter seen items
        if filter_seen:
            user_history = self.dataset.get_user_history(user_id)
            candidates = [item for item in candidates if item not in user_history]
        
        # Score items in batches
        scores = self._score_items_batch(user_idx, candidates, batch_size)
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _score_items_batch(
        self, 
        user_idx: int, 
        item_ids: List[str], 
        batch_size: int = 256
    ) -> List[Tuple[str, float]]:
        """Score items in batches"""
        all_scores = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(item_ids), batch_size), desc="Scoring items", leave=False):
                batch_items = item_ids[i:i + batch_size]
                batch_scores = self._score_single_batch(user_idx, batch_items)
                all_scores.extend(batch_scores)
        
        return all_scores
    
    def _score_single_batch(
        self, 
        user_idx: int, 
        item_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """Score a single batch of items"""
        valid_items = []
        item_indices = []
        batch_features = []
        
        # Get features and indices for valid items
        for item_id in item_ids:
            try:
                # Get item index
                item_idx = self.dataset.item_encoder.transform([item_id])[0]
                
                # Get cached features or process them
                features = self.feature_cache.get(item_id)
                if features is None:
                    features = self.dataset._process_item_features(item_id)
                    self.feature_cache.set(item_id, features)
                
                valid_items.append(item_id)
                item_indices.append(item_idx)
                batch_features.append(features)
                
            except Exception as e:
                continue  # Skip invalid items
        
        if not valid_items:
            return []
        
        # Prepare batch tensors
        batch_size = len(valid_items)
        user_tensor = torch.full((batch_size,), user_idx, dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(item_indices, dtype=torch.long).to(self.device)
        
        # Stack features
        images = torch.stack([f['image'] for f in batch_features]).to(self.device)
        text_ids = torch.stack([f['text_input_ids'] for f in batch_features]).to(self.device)
        text_masks = torch.stack([f['text_attention_mask'] for f in batch_features]).to(self.device)
        numerical = torch.stack([f['numerical_features'] for f in batch_features]).to(self.device)
        
        # Prepare model input
        model_input = {
            'user_idx': user_tensor,
            'item_idx': item_tensor,
            'image': images,
            'text_input_ids': text_ids,
            'text_attention_mask': text_masks,
            'numerical_features': numerical
        }
        
        # Add CLIP inputs if available
        if 'clip_text_input_ids' in batch_features[0]:
            clip_ids = torch.stack([f['clip_text_input_ids'] for f in batch_features]).to(self.device)
            clip_masks = torch.stack([f['clip_text_attention_mask'] for f in batch_features]).to(self.device)
            model_input['clip_text_input_ids'] = clip_ids
            model_input['clip_text_attention_mask'] = clip_masks
        
        # Get predictions
        output = self.model(**model_input)
        if isinstance(output, tuple):
            scores = output[0].squeeze()
        else:
            scores = output.squeeze()
        
        # Handle single item case
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        
        # Return results
        results = []
        for item_id, score in zip(valid_items, scores.cpu().numpy()):
            results.append((item_id, float(score)))
        
        return results
    
    def print_cache_stats(self):
        """Print cache statistics"""
        self.feature_cache.print_stats()