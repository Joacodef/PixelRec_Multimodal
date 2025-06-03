"""
Novelty and diversity metrics for recommendation evaluation
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import pandas as pd


class NoveltyMetrics:
    """Calculate novelty and diversity metrics for recommendations"""
    
    def __init__(
        self, 
        item_popularity: Dict[str, float],
        user_history: List[Tuple[str, str]]
    ):
        """
        Initialize novelty metrics calculator.
        
        Args:
            item_popularity: Dict mapping item_id to popularity score
            user_history: List of (user_id, item_id) tuples
        """
        self.item_popularity = item_popularity
        self.user_history = user_history
        
        # Calculate derived statistics
        self.total_interactions = sum(item_popularity.values())
        self.n_users = len(set(user for user, _ in user_history))
        self.item_user_counts = self._calculate_item_user_counts()
        self.popularity_ranks = self._calculate_popularity_ranks()
        
    def _calculate_item_user_counts(self) -> Counter:
        """Calculate how many users interacted with each item"""
        return Counter(item for _, item in self.user_history)
    
    def _calculate_popularity_ranks(self) -> Dict[str, int]:
        """Calculate popularity rank for each item (0 = most popular)"""
        sorted_items = sorted(
            self.item_popularity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return {item: rank for rank, (item, _) in enumerate(sorted_items)}
    
    def calculate_metrics(
        self, 
        recommendations: List[str], 
        user_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate all novelty metrics for a set of recommendations.
        
        Args:
            recommendations: List of recommended item IDs
            user_id: User ID (optional, for personalized metrics)
            
        Returns:
            Dictionary with various novelty metrics
        """
        metrics = {}
        
        # Basic checks
        if not recommendations:
            return metrics
        
        # Self-Information / Surprisal
        metrics['avg_self_information'] = self.calculate_self_information(
            recommendations
        )
        
        # Inverse Item Frequency
        metrics['avg_iif'] = self.calculate_iif(recommendations)
        
        # Coverage
        metrics['catalog_coverage'] = self.calculate_coverage(recommendations)
        
        # Popularity statistics
        pop_stats = self.calculate_popularity_stats(recommendations)
        metrics.update(pop_stats)
        
        # Long-tail percentage
        metrics['long_tail_percentage'] = self.calculate_long_tail_percentage(
            recommendations
        )
        
        # Diversity (intra-list diversity)
        if len(recommendations) > 1:
            metrics['diversity'] = self.calculate_diversity(recommendations)
        
        # Personalized novelty (if user_id provided)
        if user_id:
            metrics['personalized_novelty'] = self.calculate_personalized_novelty(
                recommendations, 
                user_id
            )
        
        return metrics
    
    def calculate_self_information(self, items: List[str]) -> float:
        """
        Calculate average self-information (surprisal) of items.
        Higher values indicate more novel/surprising items.
        """
        self_info_scores = []
        
        for item in items:
            if item in self.item_popularity and self.total_interactions > 0:
                prob = self.item_popularity[item] / self.total_interactions
                # Add epsilon to avoid log(0) and handle floating point precision issues
                epsilon = 1e-10
                prob = max(prob, epsilon)
                self_info = -np.log2(prob)
                self_info_scores.append(self_info)
        
        return np.mean(self_info_scores) if self_info_scores else 0.0
    
    def calculate_iif(self, items: List[str]) -> float:
        """
        Calculate average Inverse Item Frequency.
        Higher values indicate items that fewer users have interacted with.
        """
        iif_scores = []
        
        for item in items:
            if item in self.item_user_counts and self.n_users > 0:
                user_count = self.item_user_counts[item]
                if user_count > 0:
                    # Add small epsilon to avoid potential floating point issues
                    iif = np.log(self.n_users / (user_count + 1e-10))
                    iif_scores.append(iif)
        
        return np.mean(iif_scores) if iif_scores else 0.0
    
    def calculate_coverage(self, items: List[str]) -> float:
        """
        Calculate catalog coverage.
        What percentage of the total catalog is being recommended.
        """
        if not self.item_popularity:
            return 0.0
        
        unique_items = set(items)
        return len(unique_items) / len(self.item_popularity)
    
    def calculate_popularity_stats(self, items: List[str]) -> Dict[str, float]:
        """Calculate statistics about item popularity ranks"""
        rank_scores = [
            self.popularity_ranks.get(item, len(self.popularity_ranks)) 
            for item in items
        ]
        
        return {
            'avg_popularity_rank': np.mean(rank_scores),
            'popularity_rank_std': np.std(rank_scores),
            'min_popularity_rank': np.min(rank_scores),
            'max_popularity_rank': np.max(rank_scores)
        }
    
    def calculate_long_tail_percentage(self, items: List[str]) -> float:
        """
        Calculate percentage of recommendations from the long tail.
        Long tail = bottom 80% of items by popularity.
        """
        if not self.popularity_ranks:
            return 0.0
        
        tail_threshold = int(len(self.popularity_ranks) * 0.2)
        tail_items = {
            item for item, rank in self.popularity_ranks.items() 
            if rank >= tail_threshold
        }
        
        tail_count = sum(1 for item in items if item in tail_items)
        return tail_count / len(items) if items else 0.0
    
    def calculate_diversity(self, items: List[str]) -> float:
        """
        Calculate intra-list diversity.
        For now, returns the percentage of unique items.
        Can be extended to use item embeddings for similarity.
        """
        return len(set(items)) / len(items) if items else 0.0
    
    def calculate_personalized_novelty(
        self, 
        items: List[str], 
        user_id: str
    ) -> float:
        """
        Calculate novelty relative to user's history.
        Items the user hasn't seen before are more novel.
        """
        user_items = set(
            item for uid, item in self.user_history 
            if uid == user_id
        )
        
        novel_items = [item for item in items if item not in user_items]
        return len(novel_items) / len(items) if items else 0.0


class DiversityCalculator:
    """Calculate diversity metrics using item embeddings"""
    
    def __init__(self, item_embeddings: Dict[str, np.ndarray]):
        """
        Initialize diversity calculator.
        
        Args:
            item_embeddings: Dict mapping item_id to embedding vector
        """
        self.item_embeddings = item_embeddings
    
    def calculate_pairwise_diversity(
        self, 
        items: List[str], 
        metric: str = 'cosine'
    ) -> float:
        """
        Calculate average pairwise diversity of items.
        
        Args:
            items: List of item IDs
            metric: Distance metric ('cosine', 'euclidean')
            
        Returns:
            Average pairwise distance
        """
        if len(items) < 2:
            return 0.0
        
        distances = []
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i] in self.item_embeddings and items[j] in self.item_embeddings:
                    emb_i = self.item_embeddings[items[i]]
                    emb_j = self.item_embeddings[items[j]]
                    
                    if metric == 'cosine':
                        # Cosine distance = 1 - cosine similarity
                        # Add epsilon to denominators to avoid division by zero
                        norm_i = np.linalg.norm(emb_i)
                        norm_j = np.linalg.norm(emb_j)
                        
                        # Handle zero vectors
                        if norm_i < 1e-10 or norm_j < 1e-10:
                            distance = 1.0  # Maximum distance for zero vectors
                        else:
                            similarity = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                            # Clip to [-1, 1] to handle floating point errors
                            similarity = np.clip(similarity, -1.0, 1.0)
                            distance = 1 - similarity
                    else:  # euclidean
                        distance = np.linalg.norm(emb_i - emb_j)
                    
                    distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def calculate_coverage_diversity(
        self, 
        recommendations_per_user: Dict[str, List[str]]
    ) -> float:
        """
        Calculate coverage diversity across all users.
        How many unique items are recommended across all users.
        
        Args:
            recommendations_per_user: Dict mapping user_id to list of recommendations
            
        Returns:
            Coverage diversity score
        """
        all_recommended_items = set()
        
        for items in recommendations_per_user.values():
            all_recommended_items.update(items)
        
        total_recommendations = sum(
            len(items) for items in recommendations_per_user.values()
        )
        
        if total_recommendations == 0:
            return 0.0
        
        return len(all_recommended_items) / total_recommendations