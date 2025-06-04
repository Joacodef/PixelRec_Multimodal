"""
Baseline recommenders for comparison with the multimodal system
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path


class BaselineRecommender:
    """Base class for baseline recommenders"""
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize baseline recommender
        
        Args:
            dataset: MultimodalDataset instance (for compatibility)
            device: Device for computation (not used by baselines)
            history_interactions_df: Optional DataFrame to use for building user history
        """
        self.dataset: Any = dataset
        
        # Use provided history_interactions_df if available and not empty
        if history_interactions_df is not None and not history_interactions_df.empty:
            self.interactions_for_model: pd.DataFrame = history_interactions_df
        else:
            self.interactions_for_model = dataset.interactions
            if history_interactions_df is not None and history_interactions_df.empty:
                print("Warning: Provided history_interactions_df is empty. Falling back to dataset.interactions.")

        # Global item popularity
        self.item_popularity: Dict[str, int] = self._calculate_item_popularity(dataset.interactions)
        
        # User items for "seen" history
        self.user_items: Dict[str, set] = self._build_user_item_dict(self.interactions_for_model)
        
        # All items represent the entire catalog
        self.all_items: List[str] = (
            list(self.dataset.item_encoder.classes_) 
            if hasattr(self.dataset.item_encoder, 'classes_') and self.dataset.item_encoder.classes_ is not None 
            else []
        )
        
    def _calculate_item_popularity(self, interactions_df: pd.DataFrame) -> Dict[str, int]:
        """Calculate item popularity scores from a given interactions DataFrame."""
        if 'item_id' not in interactions_df.columns or interactions_df.empty:
            return {}
        return interactions_df['item_id'].value_counts().to_dict()
    
    def _build_user_item_dict(self, interactions_df: pd.DataFrame) -> Dict[str, set]:
        """Build dictionary of items per user from a given interactions DataFrame."""
        if ('user_id' not in interactions_df.columns or 
            'item_id' not in interactions_df.columns or 
            interactions_df.empty):
            return {}
        return interactions_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    def get_user_history(self, user_id: str) -> set:
        """Get user's interaction history (based on interactions_for_model)"""
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
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        random_seed: int = 42, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        np.random.seed(random_seed)
        
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get random recommendations"""
        
        current_candidates: List[str] = []
        if candidates is not None:
            current_candidates = list(candidates)
        elif self.all_items:
            current_candidates = self.all_items.copy()
        else:
             return []
        
        if filter_seen:
            seen_items: set = self.get_user_history(user_id)
            current_candidates = [item for item in current_candidates if item not in seen_items]
        
        n_recommendations: int = min(top_k, len(current_candidates))
        if n_recommendations == 0 or not current_candidates:
            return []
        
        recommended_items: np.ndarray = np.random.choice(current_candidates, n_recommendations, replace=False)
        
        return [(item, np.random.random()) for item in recommended_items]


class PopularityRecommender(BaselineRecommender):
    """Popularity-based baseline recommender - OPTIMIZED VERSION"""
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        self._precompute_popularity_ranking()
    
    def _precompute_popularity_ranking(self) -> None:
        """Pre-compute a sorted list of all items by popularity"""
        all_items_with_scores: List[Tuple[str, Union[int, float]]] = [
            (item, self.item_popularity.get(item, 0)) 
            for item in self.all_items
        ]
        
        all_items_with_scores.sort(key=lambda x: x[1], reverse=True)
        self.sorted_items: List[Tuple[str, Union[int, float]]] = all_items_with_scores
        
        if self.sorted_items:
            max_score: Union[int, float] = self.sorted_items[0][1] if self.sorted_items[0][1] > 0 else 1.0
            self.sorted_items_normalized: List[Tuple[str, float]] = [
                (item, score / max_score) 
                for item, score in self.sorted_items
            ]
        else:
            self.sorted_items_normalized = []
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get most popular items as recommendations - OPTIMIZED"""
        seen_items: set = set()
        if filter_seen:
            seen_items = self.get_user_history(user_id)
            if not isinstance(seen_items, set):
                seen_items = set(seen_items)
        
        recommendations: List[Tuple[str, float]] = []
        
        # Determine the pool of items to rank by popularity
        items_to_consider: List[Tuple[str, float]] = []
        if candidates is not None:
            candidate_set: set = set(candidates)
            items_to_consider = [
                (item, score) for item, score in self.sorted_items_normalized 
                if item in candidate_set
            ]
        else:
            items_to_consider = self.sorted_items_normalized

        for item, score in items_to_consider:
            if item in seen_items:
                continue
            recommendations.append((item, score))
            if len(recommendations) >= top_k:
                break
        return recommendations


class ItemKNNRecommender(BaselineRecommender):
    """Item-based collaborative filtering recommender"""
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        k_neighbors: int = 50, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        self.k_neighbors: int = k_neighbors
        self._build_item_similarity_matrix()
        
    def _build_item_similarity_matrix(self) -> None:
        """Build item-item similarity matrix using self.interactions_for_model"""
        print("Building item similarity matrix for ItemKNN...")
        
        self.user_to_idx: Dict[str, int] = {
            user: idx for idx, user in enumerate(self.dataset.user_encoder.classes_)
        }
        self.item_to_idx: Dict[str, int] = {
            item: idx for idx, item in enumerate(self.dataset.item_encoder.classes_)
        }
        self.idx_to_item: Dict[int, str] = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Filter interactions to those users/items present in the encoders
        interactions: pd.DataFrame = self.interactions_for_model[
            (self.interactions_for_model['user_id'].isin(self.user_to_idx)) &
            (self.interactions_for_model['item_id'].isin(self.item_to_idx))
        ].copy()

        if interactions.empty:
            print("Warning: No interactions available for ItemKNN model building after filtering.")
            self.item_similarities: csr_matrix = csr_matrix((len(self.item_to_idx), len(self.item_to_idx)))
            return

        interactions.loc[:, 'user_idx_map'] = interactions['user_id'].map(self.user_to_idx)
        interactions.loc[:, 'item_idx_map'] = interactions['item_id'].map(self.item_to_idx)

        row_indices: List[int] = interactions['user_idx_map'].tolist()
        col_indices: List[int] = interactions['item_idx_map'].tolist()
        data: np.ndarray = np.ones(len(interactions))
        
        user_item_matrix: csr_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )
        
        print("Calculating item similarities for ItemKNN...")
        if user_item_matrix.shape[1] > 0:
            self.item_similarities = cosine_similarity(user_item_matrix.T, dense_output=False)
        else:
            self.item_similarities = csr_matrix((len(self.item_to_idx), len(self.item_to_idx)))

    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get item-based recommendations"""
        if user_id not in self.user_to_idx:
            return PopularityRecommender(
                self.dataset, 
                history_interactions_df=self.interactions_for_model
            ).get_recommendations(user_id, top_k, filter_seen, candidates)
        
        user_interacted_items_history: set = self.get_user_history(user_id)
        if not user_interacted_items_history:
            return []
        
        scores: np.ndarray = np.zeros(len(self.item_to_idx))
        
        for item_id_hist in user_interacted_items_history:
            if item_id_hist in self.item_to_idx:
                item_idx_hist: int = self.item_to_idx[item_id_hist]
                if item_idx_hist < self.item_similarities.shape[0]:
                    scores += self.item_similarities[item_idx_hist].toarray().flatten()
        
        if len(user_interacted_items_history) > 0:
            scores /= len(user_interacted_items_history)
        
        recommendations: List[Tuple[str, float]] = []
        
        item_pool: List[str] = candidates if candidates is not None else self.all_items

        for item_id_cand in item_pool:
            if item_id_cand in self.item_to_idx:
                item_idx_cand: int = self.item_to_idx[item_id_cand]
                if scores[item_idx_cand] > 1e-9:
                    if filter_seen and item_id_cand in user_interacted_items_history:
                        continue
                    recommendations.append((item_id_cand, float(scores[item_idx_cand])))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]


class UserKNNRecommender(BaselineRecommender):
    """User-based collaborative filtering recommender"""
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        k_neighbors: int = 50, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        self.k_neighbors: int = k_neighbors
        self._build_user_item_matrix()
        
    def _build_user_item_matrix(self) -> None:
        """Build user-item matrix using self.interactions_for_model"""
        print("Building user-item matrix for UserKNN...")
        
        self.user_to_idx: Dict[str, int] = {
            user: idx for idx, user in enumerate(self.dataset.user_encoder.classes_)
        }
        self.item_to_idx: Dict[str, int] = {
            item: idx for idx, item in enumerate(self.dataset.item_encoder.classes_)
        }
        
        interactions: pd.DataFrame = self.interactions_for_model[
            (self.interactions_for_model['user_id'].isin(self.user_to_idx)) &
            (self.interactions_for_model['item_id'].isin(self.item_to_idx))
        ].copy()

        if interactions.empty:
            print("Warning: No interactions available for UserKNN model building after filtering.")
            self.user_item_matrix: csr_matrix = csr_matrix((len(self.user_to_idx), len(self.item_to_idx)))
            self.user_similarities: csr_matrix = csr_matrix((len(self.user_to_idx), len(self.user_to_idx)))
            return

        interactions.loc[:, 'user_idx_map'] = interactions['user_id'].map(self.user_to_idx)
        interactions.loc[:, 'item_idx_map'] = interactions['item_id'].map(self.item_to_idx)
        
        row_indices: List[int] = interactions['user_idx_map'].tolist()
        col_indices: List[int] = interactions['item_idx_map'].tolist()
        data: np.ndarray = np.ones(len(interactions))
        
        self.user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )
        
        print("Calculating user similarities for UserKNN...")
        if self.user_item_matrix.shape[0] > 0:
            self.user_similarities = cosine_similarity(self.user_item_matrix, dense_output=False)
        else:
            self.user_similarities = csr_matrix((len(self.user_to_idx), len(self.user_to_idx)))

    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get user-based recommendations"""
        if user_id not in self.user_to_idx:
            return PopularityRecommender(
                self.dataset, 
                history_interactions_df=self.interactions_for_model
            ).get_recommendations(user_id, top_k, filter_seen, candidates)
        
        target_user_idx: int = self.user_to_idx[user_id]
        
        if target_user_idx >= self.user_similarities.shape[0]:
             print(f"Warning: User index {target_user_idx} out of bounds for user_similarities matrix.")
             return []

        user_sim_vector: np.ndarray = self.user_similarities[target_user_idx].toarray().flatten()
        user_sim_vector[target_user_idx] = 0  # Exclude self
        
        # Get top-N similar users
        similar_user_indices: np.ndarray = np.argsort(user_sim_vector)[-self.k_neighbors:][::-1]
        similar_user_indices = similar_user_indices[user_sim_vector[similar_user_indices] > 1e-9]

        if len(similar_user_indices) == 0:
            return []
        
        # Aggregate item scores from similar users
        item_scores: np.ndarray = np.zeros(len(self.item_to_idx))
        sum_similarities: float = 0
        
        for sim_user_idx in similar_user_indices:
            if sim_user_idx < self.user_item_matrix.shape[0]:
                similarity_weight: float = user_sim_vector[sim_user_idx]
                item_scores += similarity_weight * self.user_item_matrix[sim_user_idx].toarray().flatten()
                sum_similarities += similarity_weight
        
        if sum_similarities > 1e-9:
            item_scores /= sum_similarities
        
        user_interacted_items_history: set = self.get_user_history(user_id)
        recommendations: List[Tuple[str, float]] = []
        
        item_pool: List[str] = candidates if candidates is not None else self.all_items

        for item_id_cand in item_pool:
            if item_id_cand in self.item_to_idx:
                item_idx_cand: int = self.item_to_idx[item_id_cand]
                if item_scores[item_idx_cand] > 1e-9:
                    if filter_seen and item_id_cand in user_interacted_items_history:
                        continue
                    recommendations.append((item_id_cand, float(item_scores[item_idx_cand])))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]