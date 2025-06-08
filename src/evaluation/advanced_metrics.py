# src/evaluation/advanced_metrics.py

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set

class AdvancedMetrics:
    """
    A collection of advanced metrics for evaluating recommendation systems beyond
    standard precision and recall. This class provides static methods for
    calculating metrics related to ranking quality, distribution, and novelty.
    """
    
    @staticmethod
    def calculate_mrr(recommendations: List[List[str]], relevant_items: List[Set[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank (MRR) for a set of users.

        MRR is a ranking metric that evaluates how high the first relevant item
        is in a list of recommendations. It is the average of the reciprocal
        ranks for each user. The reciprocal rank is 1 / rank of the first
        relevant item. If no relevant item is found, the rank is 0.

        Args:
            recommendations: A list of recommendation lists, where each inner
                             list contains the recommended item IDs for a single user.
            relevant_items: A list of sets, where each set contains the ground
                            truth relevant item IDs for the corresponding user.

        Returns:
            The Mean Reciprocal Rank as a float. Returns 0.0 if the input lists
            are empty.
        """
        reciprocal_ranks = []
        
        for recs, relevant in zip(recommendations, relevant_items):
            for i, item in enumerate(recs):
                if item in relevant:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
                
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def calculate_hit_rate(recommendations: List[List[str]], relevant_items: List[Set[str]]) -> float:
        """
        Calculates the Hit Rate for a set of users.

        Hit Rate measures the fraction of users for whom at least one relevant
        item was recommended in their list. It provides insight into how often
        the recommender is successful at all.

        Args:
            recommendations: A list of recommendation lists for each user.
            relevant_items: A list of sets containing relevant items for each
                            corresponding user.

        Returns:
            The Hit Rate as a float. Returns 0.0 if no recommendations are provided.
        """
        if not recommendations:
            return 0.0
        hits = 0
        for recs, relevant in zip(recommendations, relevant_items):
            if any(item in relevant for item in recs):
                hits += 1
        return hits / len(recommendations)
    
    @staticmethod
    def calculate_gini_coefficient(item_recommendations: Dict[str, int]) -> float:
        """
        Calculates the Gini coefficient for the distribution of item recommendations.

        The Gini coefficient measures the inequality of a distribution. In this
        context, it quantifies how concentrated the recommendations are among
        a small number of items. A value of 0 represents perfect equality
        (all items recommended equally), and a value closer to 1 represents
        high inequality (a few items dominate all recommendations).

        Args:
            item_recommendations: A dictionary where keys are item IDs and
                                  values are the total number of times each
                                  item was recommended.

        Returns:
            The Gini coefficient as a float. Returns 0.0 for empty inputs.
        """
        if not item_recommendations:
            return 0.0
            
        counts = np.array(list(item_recommendations.values()))
        counts = np.sort(counts)
        n = len(counts)
        
        if n == 0:
            return 0.0
        
        sum_counts = np.sum(counts)
        if sum_counts == 0:
            return 0.0
            
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * counts)) / (n * sum_counts) - (n + 1) / n
    
    @staticmethod
    def calculate_serendipity(
        recommendations: List[List[str]], 
        expected_items: List[Set[str]],
        relevant_items: List[Set[str]]
    ) -> float:
        """
        Calculates serendipity, measuring how surprising and useful recommendations are.

        Serendipity is defined as the fraction of recommended items that are
        both relevant to the user and unexpected. An item is considered
        unexpected if it is not in a pre-defined list of "expected" items
        (e.g., items from a baseline recommender or items similar to a user's
        recent history).

        Args:
            recommendations: A list of recommendation lists for each user.
            expected_items: A list of sets, where each set contains items
                            considered expected for the corresponding user.
            relevant_items: A list of sets containing the ground truth relevant
                            items for each user.

        Returns:
            The average serendipity score across all users as a float.
        """
        serendipity_scores = []
        
        for recs, expected, relevant in zip(recommendations, expected_items, relevant_items):
            unexpected_relevant = sum(1 for item in recs 
                                    if item in relevant and item not in expected)
            serendipity_scores.append(unexpected_relevant / len(recs) if recs else 0)
            
        return np.mean(serendipity_scores) if serendipity_scores else 0.0
    
    @staticmethod
    def calculate_temporal_diversity(
        recommendations: List[List[str]], 
        item_timestamps: Dict[str, float]
    ) -> float:
        """
        Calculates the temporal diversity of recommendations.

        This metric measures the standard deviation of the timestamps of the
        recommended items for each user, averaged across all users. A higher
        value indicates that the recommendations span a wider range of time,
        which can suggest more diverse content.

        Args:
            recommendations: A list of recommendation lists for each user.
            item_timestamps: A dictionary mapping item IDs to a numerical
                             timestamp (e.g., creation date).

        Returns:
            The average temporal diversity score as a float.
        """
        diversity_scores = []
        
        for recs in recommendations:
            if len(recs) < 2:
                diversity_scores.append(0.0)
                continue
                
            timestamps = [item_timestamps.get(item, 0) for item in recs]
            diversity_scores.append(np.std(timestamps))
            
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    @staticmethod
    def calculate_user_satisfaction_proxy(
        recommendations: List[List[str]],
        item_features: Dict[str, Dict[str, float]],
        user_preferences: Dict[int, Dict[str, float]]
    ) -> float:
        """
        Calculates a proxy for user satisfaction based on feature alignment.

        This metric computes the average cosine similarity between the feature
        vectors of a user's recommended items and the user's preference vector.
        It serves as a proxy for how well the recommendations align with a
        user's modeled interests.

        Args:
            recommendations: A list of recommendation lists for each user.
            item_features: A dictionary mapping item IDs to their feature vectors
                           (represented as a dictionary of feature names to values).
            user_preferences: A dictionary mapping user IDs to their preference
                              vectors (represented similarly to item features).

        Returns:
            The average satisfaction proxy score as a float.
        """
        satisfaction_scores = []
        
        for user_id, recs in enumerate(recommendations):
            if user_id not in user_preferences:
                continue
                
            user_pref = user_preferences[user_id]
            alignment_scores = []
            
            for item in recs:
                if item not in item_features:
                    continue
                    
                # Calculate cosine similarity between user preferences and item features
                item_feat = item_features[item]
                common_features = set(user_pref.keys()) & set(item_feat.keys())
                
                if not common_features:
                    continue
                    
                user_vec = np.array([user_pref[f] for f in common_features])
                item_vec = np.array([item_feat[f] for f in common_features])
                
                similarity = np.dot(user_vec, item_vec) / (
                    np.linalg.norm(user_vec) * np.linalg.norm(item_vec) + 1e-8
                )
                alignment_scores.append(similarity)
                
            if alignment_scores:
                satisfaction_scores.append(np.mean(alignment_scores))
                
        return np.mean(satisfaction_scores) if satisfaction_scores else 0.0


# src/evaluation/fairness_metrics.py

class FairnessMetrics:
    """
    A collection of metrics for evaluating the fairness of recommendations
    across different user groups and item providers.
    """
    
    @staticmethod
    def calculate_demographic_parity(
        recommendations: Dict[str, List[str]],
        user_demographics: Dict[str, str],
        demographic_attribute: str = 'gender'
    ) -> Dict[str, float]:
        """
        Calculates demographic parity across different user groups.

        Demographic parity measures whether the recommendation rate is similar
        across different demographic groups. This implementation calculates the
        rate as the number of unique items recommended to a group divided by
        the total number of recommendations given to that group.

        Args:
            recommendations: A dictionary mapping user IDs to their list of
                             recommended items.
            user_demographics: A dictionary mapping user IDs to their demographic
                               information (e.g., {'u1': {'gender': 'A'}}).
            demographic_attribute: The key for the demographic attribute to
                                   group users by (e.g., 'gender', 'age_group').

        Returns:
            A dictionary where keys are the demographic groups and values are
            their respective recommendation rates.
        """
        group_recommendations = defaultdict(list)
        
        for user_id, recs in recommendations.items():
            group = user_demographics.get(user_id, {}).get(demographic_attribute, 'unknown')
            group_recommendations[group].extend(recs)
            
        group_rates = {}
        for group, recs in group_recommendations.items():
            unique_items = len(set(recs))
            total_recs = len(recs)
            group_rates[group] = unique_items / total_recs if total_recs > 0 else 0
            
        return group_rates
    
    @staticmethod
    def calculate_provider_fairness(
        recommendations: List[List[str]],
        item_providers: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculates fairness metrics for item providers.

        This method evaluates two aspects of provider fairness:
        1. Exposure Rate: The proportion of total recommendations that belong
           to each provider.
        2. Gini Coefficient: The inequality of the exposure distribution
           across all providers.

        Args:
            recommendations: A list of recommendation lists for each user.
            item_providers: A dictionary mapping item IDs to their provider ID.

        Returns:
            A dictionary containing the exposure rates for each provider and
            the overall Gini coefficient of the exposure distribution.
        """
        provider_counts = defaultdict(int)
        total_recommendations = 0
        
        for recs in recommendations:
            for item in recs:
                provider = item_providers.get(item, 'unknown')
                provider_counts[provider] += 1
                total_recommendations += 1
        
        if total_recommendations == 0:
            return {'provider_exposure': {}, 'provider_gini': 0.0}

        provider_rates = {
            provider: count / total_recommendations 
            for provider, count in provider_counts.items()
        }
        
        counts = list(provider_counts.values())
        gini = AdvancedMetrics.calculate_gini_coefficient(
            {str(i): c for i, c in enumerate(counts)}
        )
        
        return {
            'provider_exposure': provider_rates,
            'provider_gini': gini
        }
