"""
Recommendation generation and inference
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle
from pathlib import Path

from ..data.dataset import MultimodalDataset
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
            item_embeddings_cache: Pre-computed item embeddings
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.item_embeddings_cache = item_embeddings_cache or {}
        
        # Initialize metrics calculators
        item_popularity = self.dataset.get_item_popularity()
        user_history = [
            (row['user_id'], row['item_id']) 
            for _, row in self.dataset.interactions.iterrows()
        ]
        
        self.novelty_metrics = NoveltyMetrics(item_popularity, user_history)
        
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
        except:
            print(f"User {user_id} not found in training data")
            return []
        
        # Get candidate items
        if candidates is None:
            candidates = list(self.dataset.item_encoder.classes_)
        
        # Filter seen items if requested
        if filter_seen:
            user_items = self.dataset.get_user_history(user_id)
            candidates = [item for item in candidates if item not in user_items]
        
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
        except:
            print(f"User {user_id} not found in training data")
            return [], {}
        
        # Get candidate items
        if candidates is None:
            candidates = list(self.dataset.item_encoder.classes_)[:1000]  # Limit for efficiency
        
        # Filter seen items
        if filter_seen:
            user_items = self.dataset.get_user_history(user_id)
            candidates = [item for item in candidates if item not in user_items]
        
        # Score items with embeddings
        scored_items = self._score_items_with_embeddings(user_idx, candidates)
        
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
                except:
                    continue
        
        return scores
    
    def _score_single_item(self, user_idx: int, item_id: str) -> float:
        """Score a single item for a user"""
        # Get item features
        item_features = self._get_item_features(item_id)
        if item_features is None:
            return 0.0
        
        # Get item index
        try:
            item_idx = self.dataset.item_encoder.transform([item_id])[0]
        except:
            return 0.0
        
        # Create tensors
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
        
        # Score
        score = self.model(
            user_tensor,
            item_tensor,
            item_features['image'].unsqueeze(0).to(self.device),
            item_features['text_ids'].unsqueeze(0).to(self.device),
            item_features['text_mask'].unsqueeze(0).to(self.device),
            item_features['numerical'].unsqueeze(0).to(self.device)
        ).item()
        
        return score
    
    def _score_items_with_embeddings(
        self, 
        user_idx: int, 
        items: List[str]
    ) -> List[Dict[str, any]]:
        """Score items and return with embeddings"""
        scored_items = []
        
        with torch.no_grad():
            for item_id in tqdm(items, desc="Scoring items with embeddings"):
                try:
                    # Get item features
                    item_features = self._get_item_features(item_id)
                    if item_features is None:
                        continue
                    
                    # Get item index
                    item_idx = self.dataset.item_encoder.transform([item_id])[0]
                    
                    # Create tensors
                    user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                    item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
                    
                    # Get score and embedding
                    score, _, _, item_emb = self.model(
                        user_tensor,
                        item_tensor,
                        item_features['image'].unsqueeze(0).to(self.device),
                        item_features['text_ids'].unsqueeze(0).to(self.device),
                        item_features['text_mask'].unsqueeze(0).to(self.device),
                        item_features['numerical'].unsqueeze(0).to(self.device),
                        return_embeddings=True
                    )
                    
                    popularity = self.novelty_metrics.item_popularity.get(item_id, 0)
                    
                    scored_items.append({
                        'item_id': item_id,
                        'score': score.item(),
                        'embedding': item_emb.cpu().numpy(),
                        'popularity': popularity
                    })
                    
                except:
                    continue
        
        return scored_items
    
    def _get_item_features(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get preprocessed features for an item"""
        # Check cache first
        if item_id in self.item_embeddings_cache:
            return self.item_embeddings_cache[item_id]
        
        try:
            # Get item info
            item_info = self.dataset._get_item_info(item_id)
            
            # Process features
            text = self.dataset._get_item_text(item_info)
            text_tokens = self.dataset.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            numerical_features = self.dataset._get_numerical_features(item_info)
            image = self.dataset._load_and_process_image(item_id)
            
            features = {
                'text_ids': text_tokens['input_ids'].squeeze(),
                'text_mask': text_tokens['attention_mask'].squeeze(),
                'numerical': numerical_features,
                'image': image
            }
            
            # Cache for future use
            self.item_embeddings_cache[item_id] = features
            
            return features
            
        except:
            return None
    
    def _rerank_for_diversity(
        self,
        scored_items: List[Dict[str, any]],
        top_k: int,
        diversity_weight: float,
        novelty_weight: float
    ) -> List[Dict[str, any]]:
        """Rerank items considering diversity and novelty"""
        final_recommendations = []
        selected_embeddings = []
        
        # Normalize weights
        total_weight = 1.0
        relevance_weight = total_weight - diversity_weight - novelty_weight
        
        for _ in range(min(top_k, len(scored_items))):
            best_score = -float('inf')
            best_item = None
            
            for item in scored_items:
                if item['item_id'] in [r['item_id'] for r in final_recommendations]:
                    continue
                
                # Base relevance score
                combined_score = relevance_weight * item['score']
                
                # Diversity score
                if selected_embeddings and diversity_weight > 0:
                    distances = [
                        np.linalg.norm(item['embedding'] - emb)
                        for emb in selected_embeddings
                    ]
                    diversity_score = np.mean(distances)
                    combined_score += diversity_weight * (diversity_score / 10)
                
                # Novelty score
                if novelty_weight > 0:
                    max_popularity = max(
                        s['popularity'] for s in scored_items
                    ) + 1
                    novelty_score = 1 - (item['popularity'] / max