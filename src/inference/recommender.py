"""
Recommendation generation and inference
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle # Ensure pickle is imported
from pathlib import Path

from ..data.dataset import MultimodalDataset
# NoveltyMetrics and DiversityCalculator are imported but DiversityCalculator is not used in this file.
# NoveltyMetrics is used.
from ..evaluation.novelty import NoveltyMetrics, DiversityCalculator 


class Recommender:
    """Class for generating recommendations"""
    
    def __init__(
        self,
        model: nn.Module,
        dataset: MultimodalDataset,
        device: torch.device,
        item_embeddings_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ):
        """
        Initialize recommender.
        
        Args:
            model: Trained model
            dataset: Dataset with encoders and data
            device: Device for inference
            item_embeddings_cache: Pre-computed item features (renamed for clarity from embeddings)
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        # This cache stores precomputed item features, not just embeddings.
        self.item_features_cache = item_embeddings_cache or {} 
        
        # Initialize metrics calculators
        item_popularity = self.dataset.get_item_popularity() #
        user_history = [
            (row['user_id'], row['item_id']) 
            for _, row in self.dataset.interactions.iterrows()
        ]
        
        self.novelty_metrics = NoveltyMetrics(item_popularity, user_history) #
        
        # Put model in eval mode
        self.model.eval()
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k recommendations for a user.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_seen: Whether to filter already seen items
            candidates: Optional list of candidate items (if None, uses all items)
            
        Returns:
            List of (item_id, score) tuples
        """
        # Get user index
        try:
            user_idx = self.dataset.user_encoder.transform([user_id])[0]
        except Exception as e: # Catch more specific exception if possible
            print(f"User {user_id} not found in training data or encoder error: {e}")
            return []
        
        # Get candidate items
        if candidates is None:
            candidates = list(self.dataset.item_encoder.classes_)
        
        # Filter seen items if requested
        if filter_seen:
            user_items_history = self.dataset.get_user_history(user_id) #
            candidates = [item for item in candidates if item not in user_items_history]
        
        # Score all candidate items
        scores = self._score_items(user_idx, candidates)
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_diverse_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        diversity_weight: float = 0.3,
        novelty_weight: float = 0.2,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, any]], Dict[str, float]]:
        """
        Get diverse recommendations with novelty considerations.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            diversity_weight: Weight for diversity (0-1)
            novelty_weight: Weight for novelty (0-1)
            filter_seen: Whether to filter seen items
            candidates: Optional candidate items
            
        Returns:
            Tuple of (recommendations, novelty_metrics)
        """
        # Get user index
        try:
            user_idx = self.dataset.user_encoder.transform([user_id])[0]
        except Exception as e: # Catch more specific exception if possible
            print(f"User {user_id} not found in training data or encoder error: {e}")
            return [], {}
        
        # Get candidate items
        if candidates is None:
            # Limiting candidates for efficiency, can be adjusted
            candidates = list(self.dataset.item_encoder.classes_)[:1000]  
        
        # Filter seen items
        if filter_seen:
            user_items_history = self.dataset.get_user_history(user_id) #
            candidates = [item for item in candidates if item not in user_items_history]
        
        if not candidates:
            print(f"No candidates left for user {user_id} after filtering.")
            return [], {}

        # Score items with embeddings
        scored_items = self._score_items_with_embeddings(user_idx, candidates)
        
        if not scored_items:
            print(f"No items could be scored for user {user_id}.")
            return [], {}

        # Initial ranking by score
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        
        # Rerank considering diversity and novelty
        final_recommendations = self._rerank_for_diversity(
            scored_items,
            top_k,
            diversity_weight,
            novelty_weight
        )
        
        # Calculate metrics
        rec_ids = [r['item_id'] for r in final_recommendations]
        # Calculate novelty metrics for the final reranked list
        metrics = self.novelty_metrics.calculate_metrics(rec_ids, user_id) 
        
        return final_recommendations, metrics
    
    def _score_items(
        self, 
        user_idx: int, 
        items: List[str]
    ) -> List[Tuple[str, float]]:
        """Score a list of items for a user"""
        scores = []
        
        with torch.no_grad():
            for item_id in tqdm(items, desc="Scoring items"):
                try:
                    score = self._score_single_item(user_idx, item_id)
                    scores.append((item_id, score))
                except Exception: # Catch more specific exception if possible
                    # Optionally log the error for item_id
                    continue # Skip item if scoring fails
        
        return scores
    
    def _score_single_item(self, user_idx: int, item_id: str) -> float:
        """Score a single item for a user"""
        # Get item features
        item_features = self._get_item_features(item_id)
        if item_features is None:
            # Return a very low score or handle as an error
            return -float('inf') 
        
        # Get item index
        try:
            item_idx_arr = self.dataset.item_encoder.transform([item_id])
            if len(item_idx_arr) == 0: return -float('inf') # Should not happen if item_id is in encoder
            item_idx = item_idx_arr[0]
        except Exception: # Catch more specific exception if possible
             # Item not in encoder, should ideally not happen if candidates are from item_encoder.classes_
            return -float('inf')
        
        # Create tensors
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
        
        # Score
        # Model forward pass arguments should match the model's definition
        score = self.model(
            user_idx=user_tensor,
            item_idx=item_tensor,
            image=item_features['image'].unsqueeze(0).to(self.device),
            text_input_ids=item_features['text_ids'].unsqueeze(0).to(self.device),
            text_attention_mask=item_features['text_mask'].unsqueeze(0).to(self.device),
            numerical_features=item_features['numerical'].unsqueeze(0).to(self.device)
        ).item() #
        
        return score
    
    def _score_items_with_embeddings(
        self, 
        user_idx: int, 
        items: List[str]
    ) -> List[Dict[str, any]]:
        """Score items and return with embeddings and other info"""
        scored_items_list = [] # Renamed to avoid conflict
        
        with torch.no_grad():
            for item_id in tqdm(items, desc="Scoring items with embeddings"):
                try:
                    # Get item features
                    item_features = self._get_item_features(item_id)
                    if item_features is None:
                        continue
                    
                    # Get item index
                    item_idx_arr = self.dataset.item_encoder.transform([item_id])
                    if len(item_idx_arr) == 0: continue
                    item_idx = item_idx_arr[0]
                    
                    # Create tensors
                    user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                    item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
                    
                    # Get score and embedding
                    # The model's forward method needs to support return_embeddings=True
                    output_tuple = self.model(
                        user_idx=user_tensor,
                        item_idx=item_tensor,
                        image=item_features['image'].unsqueeze(0).to(self.device),
                        text_input_ids=item_features['text_ids'].unsqueeze(0).to(self.device),
                        text_attention_mask=item_features['text_mask'].unsqueeze(0).to(self.device),
                        numerical_features=item_features['numerical'].unsqueeze(0).to(self.device),
                        return_embeddings=True 
                    ) #
                    
                    # Unpack the tuple carefully based on model's output structure when return_embeddings=True
                    # Assuming: score, vision_output_raw, clip_text_features_raw, item_representation_for_diversity
                    # The 'item_emb' for diversity should be a consistent representation of the item.
                    # Let's assume the fourth element is the desired item representation.
                    score = output_tuple[0] 
                    item_representation_for_diversity = output_tuple[3] # This is vision_emb from model forward pass

                    # Get item popularity from NoveltyMetrics instance
                    popularity = self.novelty_metrics.item_popularity.get(item_id, 0.0) 
                    
                    scored_items_list.append({
                        'item_id': item_id,
                        'score': score.item(),
                        'embedding': item_representation_for_diversity.cpu().numpy().squeeze(), # Ensure it's a 1D array
                        'popularity': popularity
                    })
                    
                except Exception: # Catch more specific exception if possible
                    # Optionally log the error for item_id
                    continue
        
        return scored_items_list
    
    def _get_item_features(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get preprocessed features for an item, using cache if available."""
        # Check cache first
        if item_id in self.item_features_cache:
            return self.item_features_cache[item_id]
        
        try:
            # Get item info using method from dataset
            item_info_series = self.dataset._get_item_info(item_id) 
            
            # Process features using methods from dataset
            text = self.dataset._get_item_text(item_info_series) 
            text_tokens = self.dataset.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128, # Should match training configuration
                return_tensors='pt'
            )
            
            numerical_features_tensor = self.dataset._get_numerical_features(item_info_series) #
            image_tensor = self.dataset._load_and_process_image(item_id) #
            
            features = {
                'text_ids': text_tokens['input_ids'].squeeze(),
                'text_mask': text_tokens['attention_mask'].squeeze(),
                'numerical': numerical_features_tensor,
                'image': image_tensor
            }
            
            # Cache for future use
            self.item_features_cache[item_id] = features
            
            return features
            
        except Exception: # Catch more specific exception if possible
             # Optionally log this error
            return None
    
    def _rerank_for_diversity(
        self,
        scored_items: List[Dict[str, any]], # Expects 'item_id', 'score', 'embedding', 'popularity'
        top_k: int,
        diversity_weight: float,
        novelty_weight: float
    ) -> List[Dict[str, any]]:
        """Rerank items considering diversity and novelty using Maximal Marginal Relevance (MMR) like approach."""
        final_recommendations = []
        # Make sure selected_embeddings stores embeddings of items already in final_recommendations
        selected_embeddings = [] 
        
        # Calculate total weight for relevance once
        relevance_weight = 1.0 - diversity_weight - novelty_weight
        if relevance_weight < 0: # Ensure relevance weight is not negative
            print("Warning: Sum of diversity and novelty weights exceeds 1.0. Clamping relevance_weight.")
            # Adjust other weights or clamp relevance_weight, e.g., normalize all three
            total_dynamic_weight = diversity_weight + novelty_weight
            if total_dynamic_weight > 0 : # Avoid division by zero if both are zero
                diversity_weight /= total_dynamic_weight
                novelty_weight /= total_dynamic_weight
            relevance_weight = 0.0 # Or distribute remaining budget

        # Calculate max_popularity once from the initial scored_items list if it's not empty
        if not scored_items:
            return []
            
        # Max popularity for normalization, add 1 to avoid division by zero and ensure ratio < 1
        all_popularities = [s['popularity'] for s in scored_items if 'popularity' in s]
        if not all_popularities: # Handle case where no items have popularity
             max_popularity = 1.0 # Default if no popularities available
        else:
            max_popularity = max(all_popularities) + 1.0


        # Make a copy of scored_items to modify/pop from during reranking
        candidate_pool = list(scored_items)

        while len(final_recommendations) < top_k and candidate_pool:
            best_item_for_current_step = None
            max_combined_score_for_current_step = -float('inf')
            best_item_idx_in_pool = -1

            for i, item in enumerate(candidate_pool):
                # Base relevance score
                current_item_score = relevance_weight * item['score']
                
                # Diversity score (similarity to already selected items)
                # Lower similarity to selected items means higher diversity contribution
                if selected_embeddings and diversity_weight > 0:
                    item_embedding = item.get('embedding')
                    if item_embedding is not None:
                        # Calculate similarity to already selected items (e.g., max similarity or avg similarity)
                        # Here, we calculate min distance (which is 1 - max similarity for cosine)
                        # Distances are preferred for MMR-like: score - lambda * max_sim
                        # Or: score + lambda * min_dist
                        similarities = [
                            # Cosine similarity: np.dot(item_embedding, emb) / (np.linalg.norm(item_embedding) * np.linalg.norm(emb))
                            # We want to penalize high similarity. So, subtract similarity or add distance.
                            # Let's use 1 - cosine_similarity as distance for diversity.
                            1 - (np.dot(item_embedding.flatten(), s_emb.flatten()) / 
                                 (np.linalg.norm(item_embedding.flatten()) * np.linalg.norm(s_emb.flatten()) + 1e-9)) # adding epsilon for stability
                            for s_emb in selected_embeddings
                        ]
                        # If we want to maximize distance, we take the minimum distance to any selected item
                        # and add it. Or, for MMR, we subtract max similarity.
                        # Using average distance here as an example for diversity contribution
                        diversity_contribution = np.mean(similarities) if similarities else 0
                        current_item_score += diversity_weight * diversity_contribution
                    else: # No embedding for item
                        pass # No diversity adjustment

                # Novelty score (less popular items are more novel)
                if novelty_weight > 0:
                    # Ensure item has 'popularity' key
                    item_popularity_val = item.get('popularity', 0.0)
                    # Novelty is 1 - (normalized popularity). Higher for less popular items.
                    normalized_popularity = item_popularity_val / max_popularity
                    novelty_score_val = 1.0 - normalized_popularity 
                    current_item_score += novelty_weight * novelty_score_val
                
                if current_item_score > max_combined_score_for_current_step:
                    max_combined_score_for_current_step = current_item_score
                    best_item_for_current_step = item
                    best_item_idx_in_pool = i
            
            if best_item_for_current_step is not None:
                final_recommendations.append(best_item_for_current_step)
                if 'embedding' in best_item_for_current_step and best_item_for_current_step['embedding'] is not None:
                    selected_embeddings.append(best_item_for_current_step['embedding'])
                # Remove the selected item from the candidate pool
                candidate_pool.pop(best_item_idx_in_pool) 
            else:
                # No suitable item found in this step, break to avoid infinite loop
                break
                
        return final_recommendations

    def load_item_features_cache(self, cache_path: str):
        """Loads the item features cache from a pickle file."""
        try:
            with open(cache_path, 'rb') as f:
                self.item_features_cache = pickle.load(f)
            print(f"Item features cache loaded from {cache_path}")
        except FileNotFoundError:
            print(f"Warning: Item features cache file not found at {cache_path}. Starting with an empty cache.")
            self.item_features_cache = {}
        except Exception as e:
            print(f"Error loading item features cache from {cache_path}: {e}")
            self.item_features_cache = {}

    def save_item_features_cache(self, cache_path: str):
        """Saves the current item features cache to a pickle file."""
        try:
            # Ensure parent directory exists
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.item_features_cache, f)
            print(f"Item features cache saved to {cache_path}")
        except Exception as e:
            print(f"Error saving item features cache to {cache_path}: {e}")