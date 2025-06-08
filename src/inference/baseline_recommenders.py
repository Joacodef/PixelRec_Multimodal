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
            self.interactions_for_model: pd.DataFrame = history_interactions_df.copy()
            # Ensure string types
            self.interactions_for_model['user_id'] = self.interactions_for_model['user_id'].astype(str)
            self.interactions_for_model['item_id'] = self.interactions_for_model['item_id'].astype(str)
        else:
            self.interactions_for_model = dataset.interactions.copy()
            if history_interactions_df is not None and history_interactions_df.empty:
                print("Warning: Provided history_interactions_df is empty. Falling back to dataset.interactions.")

        # Global item popularity (string keys)
        self.item_popularity: Dict[str, int] = self._calculate_item_popularity(dataset.interactions)
        
        # User items for "seen" history (string keys)
        self.user_items: Dict[str, set] = self._build_user_item_dict(self.interactions_for_model)
        
        # All items represent the entire catalog (string IDs)
        self.all_items: List[str] = self._get_all_item_ids()
        
    def _get_all_item_ids(self) -> List[str]:
        """Get all available item IDs as strings"""
        if (hasattr(self.dataset.item_encoder, 'classes_') and 
            self.dataset.item_encoder.classes_ is not None):
            return [str(item_id) for item_id in self.dataset.item_encoder.classes_]
        return []
        
    def _calculate_item_popularity(self, interactions_df: pd.DataFrame) -> Dict[str, int]:
        """Calculate item popularity scores from a given interactions DataFrame."""
        if 'item_id' not in interactions_df.columns or interactions_df.empty:
            return {}
        # Ensure string keys
        return interactions_df['item_id'].astype(str).value_counts().to_dict()
    
    def _build_user_item_dict(self, interactions_df: pd.DataFrame) -> Dict[str, set]:
        """Build dictionary of items per user from a given interactions DataFrame."""
        if ('user_id' not in interactions_df.columns or 
            'item_id' not in interactions_df.columns or 
            interactions_df.empty):
            return {}
        
        # Ensure string types
        interactions_df = interactions_df.copy()
        interactions_df['user_id'] = interactions_df['user_id'].astype(str)
        interactions_df['item_id'] = interactions_df['item_id'].astype(str)
        
        return interactions_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    def get_user_history(self, user_id: str) -> set:
        """Get user's interaction history (based on interactions_for_model)"""
        return self.user_items.get(str(user_id), set())
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get recommendations - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """Get score for a specific user-item pair - default implementation"""
        # Default implementation: get all recommendations and find the score for this item
        # Subclasses should override this for efficiency
        recommendations = self.get_recommendations(
            user_id=str(user_id),
            top_k=1000,  # Get many to find the item
            filter_seen=False,  # Don't filter to ensure we can score any item
            candidates=None
        )
        
        item_id_str = str(item_id)
        for rec_item_id, score in recommendations:
            if str(rec_item_id) == item_id_str:
                return score
        
        return 0.0  # Item not found in recommendations


class RandomRecommender(BaselineRecommender):
    """Random baseline recommender """
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        random_seed: int = 42, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get random recommendations - returns string item IDs"""
        
        # Ensure all candidates are strings
        current_candidates: List[str] = []
        if candidates is not None:
            current_candidates = [str(item) for item in candidates]
        elif self.all_items:
            current_candidates = [str(item) for item in self.all_items]
        else:
             return []
        
        if filter_seen:
            seen_items: set = self.get_user_history(str(user_id))
            current_candidates = [item for item in current_candidates if item not in seen_items]
        
        n_recommendations: int = min(top_k, len(current_candidates))
        if n_recommendations == 0 or not current_candidates:
            return []
        
        recommended_items: np.ndarray = np.random.choice(current_candidates, n_recommendations, replace=False)
        
        # Return string item IDs with random scores
        return [(str(item), np.random.random()) for item in recommended_items]
    
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """Get random score for a specific user-item pair"""
        # Create deterministic but user/item specific random score
        user_id_str = str(user_id)
        item_id_str = str(item_id)
        
        # Check if item exists in our catalog
        if item_id_str not in self.all_items:
            return 0.0
        
        # Create deterministic seed based on user and item
        seed_string = f"{user_id_str}_{item_id_str}_{self.random_seed}"
        item_seed = hash(seed_string) % (2**31)
        
        # Generate deterministic random score
        local_random = np.random.RandomState(item_seed)
        return float(local_random.random())


class PopularityRecommender(BaselineRecommender):
    """Popularity-based baseline recommender"""
    
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
            (str(item), self.item_popularity.get(str(item), 0)) 
            for item in self.all_items
        ]
        
        all_items_with_scores.sort(key=lambda x: x[1], reverse=True)
        self.sorted_items: List[Tuple[str, Union[int, float]]] = all_items_with_scores
        
        if self.sorted_items:
            max_score: Union[int, float] = self.sorted_items[0][1] if self.sorted_items[0][1] > 0 else 1.0
            self.sorted_items_normalized: List[Tuple[str, float]] = [
                (str(item), score / max_score) 
                for item, score in self.sorted_items
            ]
            # Create lookup dictionary for fast item score retrieval
            self.item_score_lookup: Dict[str, float] = {
                str(item): score for item, score in self.sorted_items_normalized
            }
        else:
            self.sorted_items_normalized = []
            self.item_score_lookup = {}
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get most popular items as recommendations - returns string item IDs"""
        seen_items: set = set()
        if filter_seen:
            seen_items = self.get_user_history(str(user_id))
            if not isinstance(seen_items, set):
                seen_items = set(str(item) for item in seen_items)
        
        recommendations: List[Tuple[str, float]] = []
        
        # Determine the pool of items to rank by popularity
        items_to_consider: List[Tuple[str, float]] = []
        if candidates is not None:
            candidate_set: set = set(str(item) for item in candidates)
            items_to_consider = [
                (str(item), score) for item, score in self.sorted_items_normalized 
                if str(item) in candidate_set
            ]
        else:
            items_to_consider = [(str(item), score) for item, score in self.sorted_items_normalized]

        for item, score in items_to_consider:
            if str(item) in seen_items:
                continue
            recommendations.append((str(item), score))
            if len(recommendations) >= top_k:
                break
        return recommendations
    
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """Get popularity score for a specific item"""
        item_id_str = str(item_id)
        return self.item_score_lookup.get(item_id_str, 0.0)


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
        
        # Create mappings for encoder classes (string to index)
        if (hasattr(self.dataset.user_encoder, 'classes_') and 
            self.dataset.user_encoder.classes_ is not None):
            self.user_to_idx: Dict[str, int] = {
                str(user): idx for idx, user in enumerate(self.dataset.user_encoder.classes_)
            }
        else:
            self.user_to_idx = {}
            
        if (hasattr(self.dataset.item_encoder, 'classes_') and 
            self.dataset.item_encoder.classes_ is not None):
            self.item_to_idx: Dict[str, int] = {
                str(item): idx for idx, item in enumerate(self.dataset.item_encoder.classes_)
            }
            self.idx_to_item: Dict[int, str] = {
                idx: str(item) for item, idx in self.item_to_idx.items()
            }
        else:
            self.item_to_idx = {}
            self.idx_to_item = {}
        
        # Filter interactions to those users/items present in the encoders
        interactions: pd.DataFrame = self.interactions_for_model[
            (self.interactions_for_model['user_id'].astype(str).isin(self.user_to_idx)) &
            (self.interactions_for_model['item_id'].astype(str).isin(self.item_to_idx))
        ].copy()

        if interactions.empty:
            print("Warning: No interactions available for ItemKNN model building after filtering.")
            self.item_similarities: csr_matrix = csr_matrix((len(self.item_to_idx), len(self.item_to_idx)))
            return

        interactions.loc[:, 'user_idx_map'] = interactions['user_id'].astype(str).map(self.user_to_idx)
        interactions.loc[:, 'item_idx_map'] = interactions['item_id'].astype(str).map(self.item_to_idx)

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
        """Get item-based recommendations - returns string item IDs"""
        user_id_str = str(user_id)
        
        if user_id_str not in self.user_to_idx:
            return PopularityRecommender(
                self.dataset, 
                history_interactions_df=self.interactions_for_model
            ).get_recommendations(user_id_str, top_k, filter_seen, candidates)
        
        user_interacted_items_history: set = self.get_user_history(user_id_str)
        if not user_interacted_items_history:
            return []
        
        scores: np.ndarray = np.zeros(len(self.item_to_idx))
        
        for item_id_hist in user_interacted_items_history:
            item_id_hist_str = str(item_id_hist)
            if item_id_hist_str in self.item_to_idx:
                item_idx_hist: int = self.item_to_idx[item_id_hist_str]
                if item_idx_hist < self.item_similarities.shape[0]:
                    scores += self.item_similarities[item_idx_hist].toarray().flatten()
        
        if len(user_interacted_items_history) > 0:
            scores /= len(user_interacted_items_history)
        
        recommendations: List[Tuple[str, float]] = []
        
        # Ensure candidates are strings
        item_pool: List[str] = []
        if candidates is not None:
            item_pool = [str(item) for item in candidates]
        else:
            item_pool = [str(item) for item in self.all_items]

        for item_id_cand in item_pool:
            item_id_cand_str = str(item_id_cand)
            if item_id_cand_str in self.item_to_idx:
                item_idx_cand: int = self.item_to_idx[item_id_cand_str]
                if scores[item_idx_cand] > 1e-9:
                    if filter_seen and item_id_cand_str in user_interacted_items_history:
                        continue
                    recommendations.append((item_id_cand_str, float(scores[item_idx_cand])))
        
        # Added missing return statement and sorting
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
        
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """Get item-based collaborative filtering score for a specific user-item pair"""
        user_id_str = str(user_id)
        item_id_str = str(item_id)
        
        if user_id_str not in self.user_to_idx or item_id_str not in self.item_to_idx:
            return 0.0
        
        user_interacted_items_history: set = self.get_user_history(user_id_str)
        if not user_interacted_items_history:
            return 0.0
        
        target_item_idx = self.item_to_idx[item_id_str]
        if target_item_idx >= self.item_similarities.shape[0]:
            return 0.0
        
        # Calculate score based on similarities with user's historical items
        score = 0.0
        count = 0
        
        for hist_item_id in user_interacted_items_history:
            hist_item_id_str = str(hist_item_id)
            if hist_item_id_str in self.item_to_idx:
                hist_item_idx = self.item_to_idx[hist_item_id_str]
                if hist_item_idx < self.item_similarities.shape[0]:
                    similarity = self.item_similarities[hist_item_idx, target_item_idx]
                    if hasattr(similarity, 'item'):  # Handle sparse matrix
                        similarity = similarity.item()
                    score += float(similarity)
                    count += 1
        
        return score / count if count > 0 else 0.0


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
        
        # Create mappings for encoder classes (string to index)
        if (hasattr(self.dataset.user_encoder, 'classes_') and 
            self.dataset.user_encoder.classes_ is not None):
            self.user_to_idx: Dict[str, int] = {
                str(user): idx for idx, user in enumerate(self.dataset.user_encoder.classes_)
            }
        else:
            self.user_to_idx = {}
            
        if (hasattr(self.dataset.item_encoder, 'classes_') and 
            self.dataset.item_encoder.classes_ is not None):
            self.item_to_idx: Dict[str, int] = {
                str(item): idx for idx, item in enumerate(self.dataset.item_encoder.classes_)
            }
        else:
            self.item_to_idx = {}
        
        interactions: pd.DataFrame = self.interactions_for_model[
            (self.interactions_for_model['user_id'].astype(str).isin(self.user_to_idx)) &
            (self.interactions_for_model['item_id'].astype(str).isin(self.item_to_idx))
        ].copy()

        if interactions.empty:
            print("Warning: No interactions available for UserKNN model building after filtering.")
            self.user_item_matrix: csr_matrix = csr_matrix((len(self.user_to_idx), len(self.item_to_idx)))
            self.user_similarities: csr_matrix = csr_matrix((len(self.user_to_idx), len(self.user_to_idx)))
            return

        interactions.loc[:, 'user_idx_map'] = interactions['user_id'].astype(str).map(self.user_to_idx)
        interactions.loc[:, 'item_idx_map'] = interactions['item_id'].astype(str).map(self.item_to_idx)
        
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
        """Get user-based recommendations - returns string item IDs"""
        user_id_str = str(user_id)
        
        if user_id_str not in self.user_to_idx:
            return PopularityRecommender(
                self.dataset, 
                history_interactions_df=self.interactions_for_model
            ).get_recommendations(user_id_str, top_k, filter_seen, candidates)
        
        target_user_idx: int = self.user_to_idx[user_id_str]
        
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
        
        user_interacted_items_history: set = self.get_user_history(user_id_str)
        recommendations: List[Tuple[str, float]] = []
        
        # Ensure candidates are strings
        item_pool: List[str] = []
        if candidates is not None:
            item_pool = [str(item) for item in candidates]
        else:
            item_pool = [str(item) for item in self.all_items]

        for item_id_cand in item_pool:
            item_id_cand_str = str(item_id_cand)
            if item_id_cand_str in self.item_to_idx:
                item_idx_cand: int = self.item_to_idx[item_id_cand_str]
                if item_scores[item_idx_cand] > 1e-9:
                    if filter_seen and item_id_cand_str in user_interacted_items_history:
                        continue
                    recommendations.append((item_id_cand_str, float(item_scores[item_idx_cand])))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """Get user-based collaborative filtering score for a specific user-item pair"""
        user_id_str = str(user_id)
        item_id_str = str(item_id)
        
        if user_id_str not in self.user_to_idx or item_id_str not in self.item_to_idx:
            return 0.0
        
        target_user_idx = self.user_to_idx[user_id_str]
        target_item_idx = self.item_to_idx[item_id_str]
        
        if (target_user_idx >= self.user_similarities.shape[0] or 
            target_item_idx >= self.user_item_matrix.shape[1]):
            return 0.0
        
        # Get similarity vector for target user
        user_sim_vector = self.user_similarities[target_user_idx].toarray().flatten()
        user_sim_vector[target_user_idx] = 0  # Exclude self
        
        # Get top similar users
        similar_user_indices = np.argsort(user_sim_vector)[-self.k_neighbors:][::-1]
        similar_user_indices = similar_user_indices[user_sim_vector[similar_user_indices] > 1e-9]
        
        if len(similar_user_indices) == 0:
            return 0.0
        
        # Calculate weighted score for the target item
        weighted_score = 0.0
        sum_similarities = 0.0
        
        for sim_user_idx in similar_user_indices:
            if sim_user_idx < self.user_item_matrix.shape[0]:
                similarity = user_sim_vector[sim_user_idx]
                # Check if similar user interacted with target item
                interaction = self.user_item_matrix[sim_user_idx, target_item_idx]
                if hasattr(interaction, 'item'):  # Handle sparse matrix
                    interaction = interaction.item()
                
                weighted_score += similarity * float(interaction)
                sum_similarities += similarity
        
        return weighted_score / sum_similarities if sum_similarities > 1e-9 else 0.0