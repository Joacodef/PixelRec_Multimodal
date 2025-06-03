# src/inference/baseline_recommenders.py
"""
Baseline recommenders for comparison with the multimodal system
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path


class BaselineRecommender:
    """Base class for baseline recommenders"""
    
    def __init__(self, dataset, device=None):
        """
        Initialize baseline recommender
        
        Args:
            dataset: MultimodalDataset instance (for compatibility)
            device: Device for computation (not used by baselines)
        """
        self.dataset = dataset
        self.interactions = dataset.interactions
        self.item_popularity = self._calculate_item_popularity()
        self.user_items = self._build_user_item_dict()
        self.all_items = list(self.dataset.item_encoder.classes_)
        
    def _calculate_item_popularity(self):
        """Calculate item popularity scores"""
        return self.interactions['item_id'].value_counts().to_dict()
    
    def _build_user_item_dict(self):
        """Build dictionary of items per user"""
        return self.interactions.groupby('user_id')['item_id'].apply(set).to_dict()
    
    def get_user_history(self, user_id: str) -> set:
        """Get user's interaction history"""
        return self.user_items.get(user_id, set())
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get recommendations - to be implemented by subclasses"""
        raise NotImplementedError


class RandomRecommender(BaselineRecommender):
    """Random baseline recommender"""
    
    def __init__(self, dataset, device=None, random_seed=42):
        super().__init__(dataset, device)
        np.random.seed(random_seed)
        
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get random recommendations"""
        
        # Get candidate items
        if candidates is None:
            candidates = self.all_items.copy()
        
        # Filter seen items
        if filter_seen:
            seen_items = self.get_user_history(user_id)
            candidates = [item for item in candidates if item not in seen_items]
        
        # Random sampling
        n_recommendations = min(top_k, len(candidates))
        if n_recommendations == 0:
            return []
        
        recommended_items = np.random.choice(candidates, n_recommendations, replace=False)
        
        # Return with random scores
        return [(item, np.random.random()) for item in recommended_items]


class PopularityRecommender(BaselineRecommender):
    """Popularity-based baseline recommender"""
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get most popular items as recommendations"""
        
        # Get candidate items
        if candidates is None:
            candidates = self.all_items
        
        # Filter seen items
        if filter_seen:
            seen_items = self.get_user_history(user_id)
            candidates = [item for item in candidates if item not in seen_items]
        
        # Sort by popularity
        item_scores = [(item, self.item_popularity.get(item, 0)) for item in candidates]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        if item_scores:
            max_score = item_scores[0][1]
            if max_score > 0:
                item_scores = [(item, score/max_score) for item, score in item_scores]
        
        return item_scores[:top_k]


class ItemKNNRecommender(BaselineRecommender):
    """Item-based collaborative filtering recommender"""
    
    def __init__(self, dataset, device=None, k_neighbors=50):
        super().__init__(dataset, device)
        self.k_neighbors = k_neighbors
        self._build_item_similarity_matrix()
        
    def _build_item_similarity_matrix(self):
        """Build item-item similarity matrix"""
        print("Building item similarity matrix...")
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(self.dataset.user_encoder.classes_)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.dataset.item_encoder.classes_)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Build sparse user-item matrix
        interactions = self.interactions[
            (self.interactions['user_id'].isin(self.user_to_idx)) &
            (self.interactions['item_id'].isin(self.item_to_idx))
        ]
        
        row_indices = [self.user_to_idx[user] for user in interactions['user_id']]
        col_indices = [self.item_to_idx[item] for item in interactions['item_id']]
        data = np.ones(len(interactions))
        
        self.user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )
        
        # Calculate item-item similarities (cosine similarity)
        print("Calculating item similarities...")
        self.item_similarities = cosine_similarity(self.user_item_matrix.T, dense_output=False)
        
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get item-based recommendations"""
        
        # Check if user exists
        if user_id not in self.user_to_idx:
            # Cold start - return popular items
            return PopularityRecommender(self.dataset).get_recommendations(
                user_id, top_k, filter_seen, candidates
            )
        
        # Get user's items
        user_items = self.get_user_history(user_id)
        if not user_items:
            return []
        
        # Calculate scores for all items
        scores = np.zeros(len(self.item_to_idx))
        
        for item in user_items:
            if item in self.item_to_idx:
                item_idx = self.item_to_idx[item]
                # Add similarities from this item
                scores += self.item_similarities[item_idx].toarray().flatten()
        
        # Normalize by number of user items
        scores = scores / len(user_items)
        
        # Get candidates
        if candidates is None:
            candidates = self.all_items
        
        # Create recommendations
        recommendations = []
        for item in candidates:
            if filter_seen and item in user_items:
                continue
            if item in self.item_to_idx:
                item_idx = self.item_to_idx[item]
                if scores[item_idx] > 0:
                    recommendations.append((item, float(scores[item_idx])))
        
        # Sort and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]


class UserKNNRecommender(BaselineRecommender):
    """User-based collaborative filtering recommender"""
    
    def __init__(self, dataset, device=None, k_neighbors=50):
        super().__init__(dataset, device)
        self.k_neighbors = k_neighbors
        self._build_user_item_matrix()
        
    def _build_user_item_matrix(self):
        """Build user-item matrix"""
        print("Building user-item matrix for UserKNN...")
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(self.dataset.user_encoder.classes_)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.dataset.item_encoder.classes_)}
        
        # Build sparse matrix
        interactions = self.interactions[
            (self.interactions['user_id'].isin(self.user_to_idx)) &
            (self.interactions['item_id'].isin(self.item_to_idx))
        ]
        
        row_indices = [self.user_to_idx[user] for user in interactions['user_id']]
        col_indices = [self.item_to_idx[item] for item in interactions['item_id']]
        data = np.ones(len(interactions))
        
        self.user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )
        
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get user-based recommendations"""
        
        # Check if user exists
        if user_id not in self.user_to_idx:
            # Cold start
            return PopularityRecommender(self.dataset).get_recommendations(
                user_id, top_k, filter_seen, candidates
            )
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix[user_idx]
        
        # Find similar users
        similarities = cosine_similarity(user_vector, self.user_item_matrix).flatten()
        similarities[user_idx] = 0  # Exclude self
        
        # Get top-k similar users
        top_users_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
        top_users_indices = top_users_indices[similarities[top_users_indices] > 0]
        
        if len(top_users_indices) == 0:
            return []
        
        # Aggregate items from similar users
        item_scores = np.zeros(len(self.item_to_idx))
        
        for similar_user_idx in top_users_indices:
            weight = similarities[similar_user_idx]
            item_scores += weight * self.user_item_matrix[similar_user_idx].toarray().flatten()
        
        # Normalize scores
        item_scores = item_scores / np.sum(similarities[top_users_indices])
        
        # Get user's seen items
        user_items = self.get_user_history(user_id)
        
        # Get candidates
        if candidates is None:
            candidates = self.all_items
        
        # Create recommendations
        recommendations = []
        for item in candidates:
            if filter_seen and item in user_items:
                continue
            if item in self.item_to_idx:
                item_idx = self.item_to_idx[item]
                if item_scores[item_idx] > 0:
                    recommendations.append((item, float(item_scores[item_idx])))
        
        # Sort and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]