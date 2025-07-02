"""
Contains implementations of various baseline recommender systems.

This module provides basic recommender algorithms such as Random, Popularity,
Item-KNN, and User-KNN. These baselines serve as reference points for
comparing the performance of more complex recommendation models.
Each recommender class provides a standardized interface for generating
recommendations and scoring individual user-item pairs.
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
    """
    Base class for all baseline recommender systems.

    This class provides common initialization routines and utility methods
    shared by various baseline recommendation algorithms, such as calculating
    item popularity and building user-item interaction dictionaries.
    Subclasses must implement the `get_recommendations` method.
    """
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initializes the base recommender.
        
        Args:
            dataset: A dataset instance, typically `MultimodalDataset`, used to access
                     encoders and overall item catalog information.
            device: The computation device (e.g., 'cpu', 'cuda'). This parameter is included
                    for API consistency but is generally not utilized by baseline models.
            history_interactions_df: An optional DataFrame containing historical interactions.
                                     If provided, this DataFrame is used to build the user's
                                     interaction history for filtering seen items. If None,
                                     the interactions from the `dataset` are used.
        """
        self.dataset: Any = dataset
        
        # Uses the provided history interactions DataFrame if available and not empty.
        if history_interactions_df is not None and not history_interactions_df.empty:
            self.interactions_for_model: pd.DataFrame = history_interactions_df.copy()
            # Ensures user_id and item_id columns are of string type for consistency.
            self.interactions_for_model['user_id'] = self.interactions_for_model['user_id'].astype(str)
            self.interactions_for_model['item_id'] = self.interactions_for_model['item_id'].astype(str)
        else:
            # Falls back to the dataset's interactions if no specific history DataFrame is provided.
            self.interactions_for_model = dataset.interactions.copy()
            if history_interactions_df is not None and history_interactions_df.empty:
                print("Warning: Provided history_interactions_df is empty. Falling back to dataset.interactions.")

        # Calculates and stores global item popularity based on initial dataset interactions.
        self.item_popularity: Dict[str, int] = self._calculate_item_popularity(dataset.interactions)
        
        # Builds a dictionary mapping each user to the set of items they have interacted with.
        self.user_items: Dict[str, set] = self._build_user_item_dict(self.interactions_for_model)
        
        # Retrieves a list of all available item IDs in the catalog.
        self.all_items: List[str] = self._get_all_item_ids()
        
    def _get_all_item_ids(self) -> List[str]:
        """
        Retrieves all unique item IDs from the dataset's item encoder.

        This method ensures that the list of all items represents the complete
        catalog known to the system, which is crucial for operations like
        sampling or generating recommendations.

        Returns:
            A list of all available item IDs (string). Returns an empty list if
            the item encoder is not available or has no classes.
        """
        if (hasattr(self.dataset.item_encoder, 'classes_') and 
            self.dataset.item_encoder.classes_ is not None):
            # Converts all encoder classes to string type.
            return [str(item_id) for item_id in self.dataset.item_encoder.classes_]
        return []
        
    def _calculate_item_popularity(self, interactions_df: pd.DataFrame) -> Dict[str, int]:
        """
        Calculates the popularity score for each item based on its interaction count.

        Args:
            interactions_df: A pandas DataFrame containing user-item interactions.
                             It must have an 'item_id' column.

        Returns:
            A dictionary where keys are item IDs (string) and values are their
            corresponding popularity counts (integer). Returns an empty dictionary
            if the input DataFrame is empty or lacks an 'item_id' column.
        """
        if 'item_id' not in interactions_df.columns or interactions_df.empty:
            return {}
        # Counts the occurrences of each item_id and converts the result to a dictionary.
        return interactions_df['item_id'].astype(str).value_counts().to_dict()
    
    def _build_user_item_dict(self, interactions_df: pd.DataFrame) -> Dict[str, set]:
        """
        Builds a dictionary mapping each user ID to a set of items they have interacted with.

        This dictionary serves as a lookup for a user's historical interactions,
        often used for filtering out already seen items from recommendations.

        Args:
            interactions_df: A pandas DataFrame containing user-item interactions.
                             It must have 'user_id' and 'item_id' columns.

        Returns:
            A dictionary where keys are user IDs (string) and values are sets of
            item IDs (string) interacted with by that user. Returns an empty dictionary
            if the input DataFrame is empty or lacks required columns.
        """
        if ('user_id' not in interactions_df.columns or 
            'item_id' not in interactions_df.columns or 
            interactions_df.empty):
            return {}
        
        # Creates a copy to avoid modifying the original DataFrame.
        interactions_df = interactions_df.copy()
        # Ensures user_id and item_id columns are of string type.
        interactions_df['user_id'] = interactions_df['user_id'].astype(str)
        interactions_df['item_id'] = interactions_df['item_id'].astype(str)
        
        # Groups interactions by user and collects all unique items for each user into a set.
        return interactions_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    def get_user_history(self, user_id: str) -> set:
        """
        Retrieves the set of items that a given user has historically interacted with.

        Args:
            user_id: The ID (string) of the user.

        Returns:
            A set of item IDs (string) representing the user's interaction history.
            Returns an empty set if the user has no history or is not found.
        """
        return self.user_items.get(str(user_id), set())
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Generates a list of top-K item recommendations for a given user.

        This is an abstract method that must be implemented by subclasses.
        It defines the standard interface for generating recommendations
        across all baseline models.

        Args:
            user_id: The ID (string) of the user for whom to generate recommendations.
            top_k: The maximum number of recommendations to return.
            filter_seen: If True, items already interacted with by the user will be
                         excluded from the recommendations.
            candidates: An optional list of item IDs (string) to consider as candidates
                        for recommendation. If None, all available items in the catalog
                        are considered.

        Returns:
            A list of (item_id, score) tuples, where `item_id` is a string and `score`
            is a float, sorted in descending order of score.
        
        Raises:
            NotImplementedError: This method must be overridden by concrete subclasses.
        """
        raise NotImplementedError
    
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """
        Retrieves the predicted relevance score for a specific user-item pair.

        This is the default implementation that might be inefficient for some
        baseline models, as it calls `get_recommendations` for a large number
        of items and then looks up the score. Subclasses should override this
        method for more efficient scoring if possible.

        Args:
            user_id: The ID (string) of the user.
            item_id: The ID (string) of the item.

        Returns:
            The predicted relevance score (float) for the user-item pair. Returns 0.0
            if the item is not found or not scored.
        """
        # Retrieves a large number of recommendations to ensure the target item is likely included.
        recommendations = self.get_recommendations(
            user_id=str(user_id),
            top_k=1000,  # Retrieves a large number of recommendations to cover most items.
            filter_seen=False,  # Disables filtering to ensure all items can be scored.
            candidates=None
        )
        
        item_id_str = str(item_id)
        # Iterates through the recommendations to find the score for the specific item.
        for rec_item_id, score in recommendations:
            if str(rec_item_id) == item_id_str:
                return score
        
        # Returns 0.0 if the item is not found in the recommendations.
        return 0.0


class RandomRecommender(BaselineRecommender):
    """
    Implements a random baseline recommender.

    This recommender generates recommendations by randomly selecting items
    from the available catalog. It serves as a minimal benchmark for
    recommender system performance.
    """
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        random_seed: int = 42, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initializes the RandomRecommender.

        Args:
            dataset: A dataset instance.
            device: The computation device.
            random_seed: An integer seed for the random number generator to ensure
                         reproducible random recommendations.
            history_interactions_df: Optional DataFrame for building user history.
        """
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        self.random_seed = random_seed
        # Sets NumPy's random seed for reproducibility.
        np.random.seed(random_seed)
        
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Generates random recommendations for a user.
        
        Items are chosen randomly from the candidate pool, optionally
        filtering out items the user has already seen. Each recommended item
        is assigned a random score.

        Args:
            user_id: The ID (string) of the user.
            top_k: The number of recommendations to return.
            filter_seen: If True, filters out previously seen items.
            candidates: An optional list of item IDs (string) to choose from.
                        If None, all available items are considered.

        Returns:
            A list of (item_id, score) tuples, representing random recommendations.
        """
        
        # Prepares the pool of items from which to select recommendations.
        current_candidates: List[str] = []
        if candidates is not None:
            # Ensures all candidate items are strings.
            current_candidates = [str(item) for item in candidates]
        elif self.all_items:
            # Uses the full catalog of items if no specific candidates are provided.
            current_candidates = [str(item) for item in self.all_items]
        else:
             return []
        
        if filter_seen:
            # Retrieves and converts user's seen items to a set for efficient lookup.
            seen_items: set = self.get_user_history(str(user_id))
            # Removes seen items from the candidate pool.
            current_candidates = [item for item in current_candidates if item not in seen_items]
        
        # Determines the actual number of recommendations to generate based on `top_k`
        # and the number of available candidates.
        n_recommendations: int = min(top_k, len(current_candidates))
        if n_recommendations == 0 or not current_candidates:
            return []
        
        # Randomly selects items from the filtered candidate list without replacement.
        recommended_items: np.ndarray = np.random.choice(current_candidates, n_recommendations, replace=False)
        
        # Assigns a random score to each recommended item.
        return [(str(item), np.random.random()) for item in recommended_items]
    
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """
        Retrieves a deterministic random score for a specific user-item pair.

        This score is generated based on a combination of the user ID, item ID,
        and the random seed, ensuring reproducibility for the same pair.

        Args:
            user_id: The ID (string) of the user.
            item_id: The ID (string) of the item.

        Returns:
            A float representing the deterministic random score for the pair.
            Returns 0.0 if the item is not in the catalog.
        """
        user_id_str = str(user_id)
        item_id_str = str(item_id)
        
        # Returns 0.0 if the item is not part of the known item catalog.
        if item_id_str not in self.all_items:
            return 0.0
        
        # Creates a seed string combining user ID, item ID, and the global random seed
        # to ensure the generated score is deterministic for this specific pair.
        seed_string = f"{user_id_str}_{item_id_str}_{self.random_seed}"
        # Converts the seed string into a 32-bit integer for use with `RandomState`.
        item_seed = hash(seed_string) % (2**31)
        
        # Initializes a local random number generator with the deterministic seed.
        local_random = np.random.RandomState(item_seed)
        # Generates a random float as the score.
        return float(local_random.random())


class PopularityRecommender(BaselineRecommender):
    """
    Implements a popularity-based baseline recommender.

    This recommender recommends items based solely on their global popularity
    (i.e., how often they appear in the interactions data), with more popular
    items being recommended first.
    """
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initializes the PopularityRecommender.

        Args:
            dataset: A dataset instance.
            device: The computation device.
            history_interactions_df: Optional DataFrame for building user history.
        """
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        # Pre-computes the sorted list of all items by popularity during initialization.
        self._precompute_popularity_ranking()
    
    def _precompute_popularity_ranking(self) -> None:
        """
        Pre-computes a sorted list of all items by their popularity score.

        Items are ranked from most popular to least popular. Popularity scores
        are normalized to be between 0 and 1, where the most popular item has a score of 1.0.
        """
        # Creates a list of all items with their raw popularity scores.
        all_items_with_scores: List[Tuple[str, Union[int, float]]] = [
            (str(item), self.item_popularity.get(str(item), 0)) 
            for item in self.all_items
        ]
        
        # Sorts the items by popularity score in descending order.
        all_items_with_scores.sort(key=lambda x: x[1], reverse=True)
        self.sorted_items: List[Tuple[str, Union[int, float]]] = all_items_with_scores
        
        if self.sorted_items:
            # Determines the maximum popularity score for normalization.
            max_score: Union[int, float] = self.sorted_items[0][1] if self.sorted_items[0][1] > 0 else 1.0
            # Normalizes popularity scores to be between 0 and 1.
            self.sorted_items_normalized: List[Tuple[str, float]] = [
                (str(item), score / max_score) 
                for item, score in self.sorted_items
            ]
            # Creates a lookup dictionary for efficient retrieval of normalized item scores.
            self.item_score_lookup: Dict[str, float] = {
                str(item): score for item, score in self.sorted_items_normalized
            }
        else:
            # Initializes empty lists and dictionaries if no items are available.
            self.sorted_items_normalized = []
            self.item_score_lookup = {}
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Generates popularity-based recommendations for a user.

        Items are recommended in descending order of their global popularity.
        Optionally, items already seen by the user can be filtered out.

        Args:
            user_id: The ID (string) of the user.
            top_k: The number of recommendations to return.
            filter_seen: If True, filters out previously seen items.
            candidates: An optional list of item IDs (string) to choose from.
                        If None, all available items are considered.

        Returns:
            A list of (item_id, score) tuples, representing popularity-based recommendations.
        """
        seen_items: set = set()
        if filter_seen:
            # Retrieves and processes the user's interaction history for filtering.
            seen_items = self.get_user_history(str(user_id))
            if not isinstance(seen_items, set):
                seen_items = set(str(item) for item in seen_items)
        
        recommendations: List[Tuple[str, float]] = []
        
        # Determines the pool of items to consider for ranking by popularity.
        items_to_consider: List[Tuple[str, float]] = []
        if candidates is not None:
            # Filters normalized sorted items to include only those present in the candidates list.
            candidate_set: set = set(str(item) for item in candidates)
            items_to_consider = [
                (str(item), score) for item, score in self.sorted_items_normalized 
                if str(item) in candidate_set
            ]
        else:
            # Uses all normalized sorted items if no specific candidates are provided.
            items_to_consider = [(str(item), score) for item, score in self.sorted_items_normalized]

        # Iterates through the popularity-ranked items, adding them to recommendations
        # if they haven't been seen by the user, until `top_k` recommendations are met.
        for item, score in items_to_consider:
            if str(item) in seen_items:
                continue
            recommendations.append((str(item), score))
            if len(recommendations) >= top_k:
                break
        return recommendations
    
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """
        Retrieves the popularity score for a specific item.

        Args:
            user_id: The ID (string) of the user. This is included for API consistency
                     but is not used in popularity-based scoring.
            item_id: The ID (string) of the item.

        Returns:
            The normalized popularity score (float) for the item. Returns 0.0 if
            the item is not found in the popularity lookup.
        """
        item_id_str = str(item_id)
        # Returns the normalized popularity score directly from the pre-computed lookup.
        return self.item_score_lookup.get(item_id_str, 0.0)


class ItemKNNRecommender(BaselineRecommender):
    """
    Implements an item-based collaborative filtering recommender (Item-KNN).

    This recommender suggests items to a user that are similar to the items
    the user has previously interacted with. Similarity between items is
    typically calculated using cosine similarity based on user-item interaction patterns.
    """
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        k_neighbors: int = 50, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initializes the ItemKNNRecommender.

        Args:
            dataset: A dataset instance.
            device: The computation device.
            k_neighbors: The number of nearest neighbors (similar items) to consider
                         when generating recommendations.
            history_interactions_df: Optional DataFrame for building user history.
        """
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        self.k_neighbors: int = k_neighbors
        # Builds the item-item similarity matrix during initialization.
        self._build_item_similarity_matrix()
        
    def _build_item_similarity_matrix(self) -> None:
        """
        Builds the item-item similarity matrix using cosine similarity on user-item interactions.

        The matrix stores the similarity scores between all pairs of items.
        This matrix is used to find items similar to those a user has interacted with.
        """
        print("Building item similarity matrix for ItemKNN...")
        
        # Creates mappings from string user/item IDs to integer indices, based on dataset encoders.
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
            # Creates an inverse mapping from integer index to string item ID.
            self.idx_to_item: Dict[int, str] = {
                idx: str(item) for item, idx in self.item_to_idx.items()
            }
        else:
            self.item_to_idx = {}
            self.idx_to_item = {}
        
        # Filters interactions to include only users and items that exist in the encoders.
        interactions: pd.DataFrame = self.interactions_for_model[
            (self.interactions_for_model['user_id'].astype(str).isin(self.user_to_idx)) &
            (self.interactions_for_model['item_id'].astype(str).isin(self.item_to_idx))
        ].copy()

        if interactions.empty:
            print("Warning: No interactions available for ItemKNN model building after filtering.")
            # Initializes an empty sparse matrix if no interactions are available.
            self.item_similarities: csr_matrix = csr_matrix((len(self.item_to_idx), len(self.item_to_idx)))
            return

        # Maps user and item IDs to their corresponding integer indices.
        interactions.loc[:, 'user_idx_map'] = interactions['user_id'].astype(str).map(self.user_to_idx)
        interactions.loc[:, 'item_idx_map'] = interactions['item_id'].astype(str).map(self.item_to_idx)

        # Prepares data for creating a sparse user-item interaction matrix.
        row_indices: List[int] = interactions['user_idx_map'].tolist()
        col_indices: List[int] = interactions['item_idx_map'].tolist()
        data: np.ndarray = np.ones(len(interactions)) # All interactions are implicitly 1 (positive).
        
        # Creates a sparse matrix where rows are users and columns are items.
        user_item_matrix: csr_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )
        
        print("Calculating item similarities for ItemKNN...")
        # Calculates cosine similarity between item vectors (transposed user-item matrix).
        # This results in an item-by-item similarity matrix.
        if user_item_matrix.shape[1] > 0:
            self.item_similarities = cosine_similarity(user_item_matrix.T, dense_output=False)
        else:
            # Initializes an empty similarity matrix if no items are present.
            self.item_similarities = csr_matrix((len(self.item_to_idx), len(self.item_to_idx)))

    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Generates item-based collaborative filtering recommendations for a user.

        Recommendations are generated by aggregating the scores of items similar
        to those the user has already interacted with.

        Args:
            user_id: The ID (string) of the user.
            top_k: The number of recommendations to return.
            filter_seen: If True, filters out previously seen items.
            candidates: An optional list of item IDs (string) to choose from.
                        If None, all available items are considered.

        Returns:
            A list of (item_id, score) tuples, representing Item-KNN recommendations.
        """
        user_id_str = str(user_id)
        
        # If the user is not found in the encoder, falls back to popularity-based recommendations.
        if user_id_str not in self.user_to_idx:
            return PopularityRecommender(
                self.dataset, 
                history_interactions_df=self.interactions_for_model
            ).get_recommendations(user_id_str, top_k, filter_seen, candidates)
        
        # Retrieves the set of items the user has already interacted with.
        user_interacted_items_history: set = self.get_user_history(user_id_str)
        if not user_interacted_items_history:
            return []
        
        # Initializes an array to store aggregated scores for all items in the catalog.
        scores: np.ndarray = np.zeros(len(self.item_to_idx))
        
        # Aggregates scores from items similar to those in the user's history.
        for item_id_hist in user_interacted_items_history:
            item_id_hist_str = str(item_id_hist)
            if item_id_hist_str in self.item_to_idx:
                item_idx_hist: int = self.item_to_idx[item_id_hist_str]
                # Adds the similarity scores of historical item to all other items.
                if item_idx_hist < self.item_similarities.shape[0]:
                    scores += self.item_similarities[item_idx_hist].toarray().flatten()
        
        # Averages the scores if any historical items were processed.
        if len(user_interacted_items_history) > 0:
            scores /= len(user_interacted_items_history)
        
        recommendations: List[Tuple[str, float]] = []
        
        # Prepares the pool of candidate items to rank.
        item_pool: List[str] = []
        if candidates is not None:
            item_pool = [str(item) for item in candidates]
        else:
            item_pool = [str(item) for item in self.all_items]

        # Iterates through the item pool, adding items to recommendations based on their scores.
        for item_id_cand in item_pool:
            item_id_cand_str = str(item_id_cand)
            if item_id_cand_str in self.item_to_idx:
                item_idx_cand: int = self.item_to_idx[item_id_cand_str]
                # Adds item to recommendations if its score is positive (i.e., it has some similarity).
                #if scores[item_idx_cand] > 1e-9:
                if filter_seen and item_id_cand_str in user_interacted_items_history:
                    continue # Skips if item is seen and filtering is enabled.
                recommendations.append((item_id_cand_str, float(scores[item_idx_cand])))
        
        # Sorts the recommendations by score in descending order and returns the top-K.
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
        
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """
        Retrieves the Item-KNN predicted relevance score for a specific user-item pair.

        The score is calculated by considering the similarity between the target
        item and all items the user has historically interacted with.

        Args:
            user_id: The ID (string) of the user.
            item_id: The ID (string) of the item for which to get the score.

        Returns:
            The predicted relevance score (float) for the user-item pair. Returns 0.0
            if the user or item is not found, or if the user has no history.
        """
        user_id_str = str(user_id)
        item_id_str = str(item_id)
        
        # Returns 0.0 if user or item is not found in the encoders.
        if user_id_str not in self.user_to_idx or item_id_str not in self.item_to_idx:
            return 0.0
        
        # Retrieves the target item's integer index.
        target_item_idx = self.item_to_idx[item_id_str]
        
        # Returns 0.0 if the target item index is out of bounds for the similarity matrix.
        if target_item_idx >= self.item_similarities.shape[0]:
            return 0.0
        
        # Retrieves the user's historical interactions.
        user_interacted_items_history: set = self.get_user_history(user_id_str)
        if not user_interacted_items_history:
            return 0.0
        
        # Calculates the weighted score based on similarities with historical items.
        score = 0.0
        count = 0
        
        for hist_item_id in user_interacted_items_history:
            hist_item_id_str = str(hist_item_id)
            if hist_item_id_str in self.item_to_idx:
                hist_item_idx = self.item_to_idx[hist_item_id_str]
                if hist_item_idx < self.item_similarities.shape[0]:
                    # Gets the similarity between the historical item and the target item.
                    similarity = self.item_similarities[hist_item_idx, target_item_idx]
                    # Converts sparse matrix element to a scalar float if necessary.
                    if hasattr(similarity, 'item'):  
                        similarity = similarity.item()
                    score += float(similarity)
                    count += 1
        
        # Returns the average similarity, or 0.0 if no relevant similarities were found.
        return score / count if count > 0 else 0.0


class UserKNNRecommender(BaselineRecommender):
    """
    Implements a user-based collaborative filtering recommender (User-KNN).

    This recommender suggests items to a user that were liked by similar users.
    Similarity between users is typically calculated using cosine similarity
    based on their shared item interaction patterns.
    """
    
    def __init__(
        self, 
        dataset: Any, 
        device: Optional[Any] = None, 
        k_neighbors: int = 50, 
        history_interactions_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initializes the UserKNNRecommender.

        Args:
            dataset: A dataset instance.
            device: The computation device.
            k_neighbors: The number of nearest neighbors (similar users) to consider
                         when generating recommendations.
            history_interactions_df: Optional DataFrame for building user history.
        """
        super().__init__(dataset, device, history_interactions_df=history_interactions_df)
        self.k_neighbors: int = k_neighbors
        # Builds the user-item matrix and user-user similarity matrix during initialization.
        self._build_user_item_matrix()
        
    def _build_user_item_matrix(self) -> None:
        """
        Builds the user-item interaction matrix and the user-user similarity matrix.

        The user-item matrix represents interactions where rows are users and
        columns are items. The user-user similarity matrix is computed using
        cosine similarity on this user-item matrix.
        """
        print("Building user-item matrix for UserKNN...")
        
        # Creates mappings from string user/item IDs to integer indices, based on dataset encoders.
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
        
        # Filters interactions to include only users and items present in the encoders.
        interactions: pd.DataFrame = self.interactions_for_model[
            (self.interactions_for_model['user_id'].astype(str).isin(self.user_to_idx)) &
            (self.interactions_for_model['item_id'].astype(str).isin(self.item_to_idx))
        ].copy()

        if interactions.empty:
            print("Warning: No interactions available for UserKNN model building after filtering.")
            # Initializes empty sparse matrices if no interactions are available.
            self.user_item_matrix: csr_matrix = csr_matrix((len(self.user_to_idx), len(self.item_to_idx)))
            self.user_similarities: csr_matrix = csr_matrix((len(self.user_to_idx), len(self.user_to_idx)))
            return

        # Maps user and item IDs to their corresponding integer indices.
        interactions.loc[:, 'user_idx_map'] = interactions['user_id'].astype(str).map(self.user_to_idx)
        interactions.loc[:, 'item_idx_map'] = interactions['item_id'].astype(str).map(self.item_to_idx)
        
        # Prepares data for creating a sparse user-item interaction matrix.
        row_indices: List[int] = interactions['user_idx_map'].tolist()
        col_indices: List[int] = interactions['item_idx_map'].tolist()
        data: np.ndarray = np.ones(len(interactions)) # All interactions are implicitly 1 (positive).
        
        # Creates a sparse matrix where rows are users and columns are items.
        self.user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )
        
        print("Calculating user similarities for UserKNN...")
        # Calculates cosine similarity between user vectors (rows of user-item matrix).
        # This results in a user-by-user similarity matrix.
        if self.user_item_matrix.shape[0] > 0:
            self.user_similarities = cosine_similarity(self.user_item_matrix, dense_output=False)
        else:
            # Initializes an empty similarity matrix if no users are present.
            self.user_similarities = csr_matrix((len(self.user_to_idx), len(self.user_to_idx)))

    def get_recommendations(
        self,
        user_id: Any, # Changed type hint to Any for robustness
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Generates user-based collaborative filtering recommendations for a user.

        Recommendations are generated by aggregating items liked by users similar
        to the target user.

        Args:
            user_id: The ID (string) of the user.
            top_k: The number of recommendations to return.
            filter_seen: If True, filters out previously seen items.
            candidates: An optional list of item IDs (string) to choose from.
                        If None, all available items are considered.

        Returns:
            A list of (item_id, score) tuples, representing User-KNN recommendations.
        """
        # Ensure user_id_str is safely defined before any debug prints or operations
        # This modification aims to prevent the "local variable not associated" error
        user_id_str: str
        try:
            user_id_str = str(user_id)
        except Exception as e:
            # If conversion fails, assign a placeholder and log the error
            print(f"ERROR: UserKNN get_recommendations received an invalid user_id '{user_id}'. Conversion to string failed: {e}")
            user_id_str = "UNKNOWN_USER_ID"
            # Since the user_id is critical, it's safer to return an empty list here
            # to prevent further errors down the line if the ID is malformed.
            return [] 
            
        # --- START OF ADDED DEBUG PRINTS ---
        #print(f"\n--- UserKNN Recommendations Debug for User: {user_id_str} ---")
        #print(f"Top K requested: {top_k}")
        #print(f"Candidates received (if any): {candidates}")
        # --- END OF ADDED DEBUG PRINTS ---

        # If the user is not found in the encoder, falls back to popularity-based recommendations.
        if user_id_str not in self.user_to_idx:
            print(f"User {user_id_str} not in encoder. Falling back to PopularityRecommender.")
            return PopularityRecommender(
                self.dataset, 
                history_interactions_df=self.interactions_for_model
            ).get_recommendations(user_id_str, top_k, filter_seen, candidates)
        
        # Retrieves the target user's integer index.
        target_user_idx: int = self.user_to_idx[user_id_str]
        
        # Returns empty recommendations if the target user index is out of bounds for the similarity matrix.
        if target_user_idx >= self.user_similarities.shape[0]:
             print(f"Warning: User index {target_user_idx} out of bounds for user_similarities matrix.")
             return []

        # Gets the similarity vector for the target user (similarities to all other users).
        user_sim_vector: np.ndarray = self.user_similarities[target_user_idx].toarray().flatten()
        user_sim_vector[target_user_idx] = 0  # Excludes the user themselves from being a neighbor.
        
        # Identifies the top-K most similar users whose similarity is positive.
        similar_user_indices: np.ndarray = np.argsort(user_sim_vector)[-self.k_neighbors:][::-1]
        similar_user_indices = similar_user_indices[user_sim_vector[similar_user_indices] > 1e-9]

        if len(similar_user_indices) == 0:
            print("No similar users found with positive similarity.")
            return []

        # print(f"<<<<<<<<< {user_id_str} kelok este es el febug para saber: ({len(similar_user_indices)}), {similar_user_indices}")
        
        # Aggregates item scores from similar users.
        item_scores: np.ndarray = np.zeros(len(self.item_to_idx))
        sum_similarities: float = 0
        
        for sim_user_idx in similar_user_indices:
            if sim_user_idx < self.user_item_matrix.shape[0]:
                similarity_weight: float = user_sim_vector[sim_user_idx]
                # Adds the interaction vector of the similar user, weighted by their similarity.
                item_scores += similarity_weight * self.user_item_matrix[sim_user_idx].toarray().flatten()
                sum_similarities += similarity_weight
        
        # Normalizes the aggregated item scores by the sum of similarities of contributing neighbors.
        if sum_similarities > 1e-9:
            item_scores /= sum_similarities
        
        # Retrieves items seen by the target user for filtering.
        user_interacted_items_history: set = self.get_user_history(user_id_str)
        recommendations: List[Tuple[str, float]] = []
        
        # Prepares the pool of candidate items to rank.
        item_pool: List[str] = []
        if candidates is not None:
            item_pool = [str(item) for item in candidates]
        else:
            item_pool = [str(item) for item in self.all_items]

        # --- START OF ADDED DEBUG PRINTS ---
        #print(f"Items in item_pool for scoring ({len(item_pool)} items): {item_pool[:5]}... (showing first 5)")
        # --- END OF ADDED DEBUG PRINTS ---

        # Iterates through the item pool, adding items to recommendations based on their scores.
        for item_id_cand in item_pool:
            item_id_cand_str = str(item_id_cand)
            if item_id_cand_str in self.item_to_idx:
                item_idx_cand: int = self.item_to_idx[item_id_cand_str]
                # --- START OF MODIFICATION ---
                # REMOVED: if item_scores[item_idx_cand] > 1e-9:
                # This ensures all items in the pool are considered for the top-K list,
                # regardless of their score (as long as they are not seen and are in the item index).
                # --- END OF MODIFICATION ---
                if filter_seen and item_id_cand_str in user_interacted_items_history:
                    continue # Skips if item is seen and filtering is enabled.
                recommendations.append((item_id_cand_str, float(item_scores[item_idx_cand])))

        # Sorts the recommendations by score in descending order and returns the top-K.
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # --- START OF ADDED DEBUG PRINTS ---
        #print(f"Raw recommendations generated before top_k slice ({len(recommendations)} items): {recommendations}")
        #print(f"Returning top {top_k} recommendations.")
        # --- END OF ADDED DEBUG PRINTS ---

        return recommendations[:top_k]
    
    def get_item_score(self, user_id: str, item_id: str) -> float:
        """
        Retrieves the User-KNN predicted relevance score for a specific user-item pair.

        The score is calculated by considering the target item's interactions
        by users similar to the target user.

        Args:
            user_id: The ID (string) of the user.
            item_id: The ID (string) of the item for which to get the score.

        Returns:
            The predicted relevance score (float) for the user-item pair. Returns 0.0
            if the user or item is not found, or no similar users contributed to the score.
        """
        user_id_str = str(user_id)
        item_id_str = str(item_id)
        
        # Returns 0.0 if user or item is not found in the encoders.
        if user_id_str not in self.user_to_idx or item_id_str not in self.item_to_idx:
            return 0.0
        
        # Retrieves the integer indices for the target user and item.
        target_user_idx = self.user_to_idx[user_id_str]
        target_item_idx = self.item_to_idx[item_id_str]
        
        # Returns 0.0 if indices are out of bounds for the matrices.
        if (target_user_idx >= self.user_similarities.shape[0] or 
            target_item_idx >= self.user_item_matrix.shape[1]):
            return 0.0
        
        # Gets the similarity vector for the target user.
        user_sim_vector = self.user_similarities[target_user_idx].toarray().flatten()
        user_sim_vector[target_user_idx] = 0  # Excludes the user themselves.
        
        # Identifies the top-K most similar users with positive similarity.
        similar_user_indices = np.argsort(user_sim_vector)[-self.k_neighbors:][::-1]
        similar_user_indices = similar_user_indices[user_sim_vector[similar_user_indices] > 1e-9]
        
        if len(similar_user_indices) == 0:
            return 0.0
        
        # Calculates the weighted score for the target item.
        weighted_score = 0.0
        sum_similarities = 0.0
        
        for sim_user_idx in similar_user_indices:
            if sim_user_idx < self.user_item_matrix.shape[0]:
                similarity = user_sim_vector[sim_user_idx]
                # Checks if the similar user interacted with the target item.
                interaction = self.user_item_matrix[sim_user_idx, target_item_idx]
                # Converts sparse matrix element to a scalar float if necessary.
                if hasattr(interaction, 'item'):  
                    interaction = interaction.item()
                
                weighted_score += similarity * float(interaction)
                sum_similarities += similarity
        
        # Returns the normalized weighted score, or 0.0 if sum of similarities is zero.
        return weighted_score / sum_similarities if sum_similarities > 1e-9 else 0.0