# src/evaluation/advanced_metrics.py

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set

class AdvancedMetrics:
    """Advanced metrics for recommendation evaluation"""
    
    @staticmethod
    def calculate_mrr(recommendations: List[List[str]], relevant_items: List[Set[str]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for recs, relevant in zip(recommendations, relevant_items):
            for i, item in enumerate(recs):
                if item in relevant:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
                
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def calculate_hit_rate(recommendations: List[List[str]], relevant_items: List[Set[str]]) -> float:
        """Calculate hit rate (percentage of users with at least one hit)"""
        hits = 0
        for recs, relevant in zip(recommendations, relevant_items):
            if any(item in relevant for item in recs):
                hits += 1
        return hits / len(recommendations)
    
    @@staticmethod
    def calculate_gini_coefficient(item_recommendations: Dict[str, int]) -> float:
        """Calculate Gini coefficient for recommendation distribution"""
        if not item_recommendations:
            return 0.0
            
        counts = np.array(list(item_recommendations.values()))
        counts = np.sort(counts)
        n = len(counts)
        
        # Handle edge cases
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
        """Calculate serendipity (unexpected but relevant recommendations)"""
        serendipity_scores = []
        
        for recs, expected, relevant in zip(recommendations, expected_items, relevant_items):
            unexpected_relevant = sum(1 for item in recs 
                                    if item in relevant and item not in expected)
            serendipity_scores.append(unexpected_relevant / len(recs) if recs else 0)
            
        return np.mean(serendipity_scores)
    
    @staticmethod
    def calculate_temporal_diversity(
        recommendations: List[List[str]], 
        item_timestamps: Dict[str, float]
    ) -> float:
        """Calculate temporal diversity of recommendations"""
        diversity_scores = []
        
        for recs in recommendations:
            if len(recs) < 2:
                diversity_scores.append(0.0)
                continue
                
            timestamps = [item_timestamps.get(item, 0) for item in recs]
            diversity_scores.append(np.std(timestamps))
            
        return np.mean(diversity_scores)
    
    @staticmethod
    def calculate_user_satisfaction_proxy(
        recommendations: List[List[str]],
        item_features: Dict[str, Dict[str, float]],
        user_preferences: Dict[int, Dict[str, float]]
    ) -> float:
        """Calculate a proxy for user satisfaction based on feature alignment"""
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
    """Metrics for evaluating recommendation fairness"""
    
    @staticmethod
    def calculate_demographic_parity(
        recommendations: Dict[str, List[str]],
        user_demographics: Dict[str, str],
        demographic_attribute: str = 'gender'
    ) -> Dict[str, float]:
        """Calculate demographic parity for different user groups"""
        group_recommendations = defaultdict(list)
        
        for user_id, recs in recommendations.items():
            group = user_demographics.get(user_id, {}).get(demographic_attribute, 'unknown')
            group_recommendations[group].extend(recs)
            
        # Calculate recommendation rate for each group
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
        """Calculate fairness for content providers"""
        provider_counts = defaultdict(int)
        total_recommendations = 0
        
        for recs in recommendations:
            for item in recs:
                provider = item_providers.get(item, 'unknown')
                provider_counts[provider] += 1
                total_recommendations += 1
                
        # Calculate exposure rate for each provider
        provider_rates = {
            provider: count / total_recommendations 
            for provider, count in provider_counts.items()
        }
        
        # Calculate Gini coefficient for provider exposure
        counts = list(provider_counts.values())
        gini = AdvancedMetrics.calculate_gini_coefficient(
            {str(i): c for i, c in enumerate(counts)}
        )
        
        return {
            'provider_exposure': provider_rates,
            'provider_gini': gini
        }