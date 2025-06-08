"""
Standard recommendation metrics.

This module provides implementations for common evaluation metrics used in recommender systems,
such as Precision@k, Recall@k, Normalized Discounted Cumulative Gain (NDCG@k), and Mean Average Precision (MAP).
These functions are designed to assess the quality of a list of recommended items against a set of relevant items.
"""
import numpy as np
from typing import List, Set

def calculate_precision_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Calculates Precision at K (P@k).

    Precision@k measures the proportion of recommended items in the top-k list
    that are relevant to the user. It indicates how many of the items the user
    was shown are actually useful.

    Args:
        recommended: A list of item IDs recommended to the user, ordered by relevance.
        relevant: A set of item IDs that are truly relevant to the user.
        k: The number of top recommendations to consider.

    Returns:
        The Precision@k score, a float between 0.0 and 1.0.
    """
    if not recommended or k == 0:
        return 0.0
    
    # Considers only the top 'k' recommended items.
    recommended_k = recommended[:k]
    # Counts the number of relevant items found within the top 'k' recommendations.
    hits = sum(1 for item in recommended_k if item in relevant)
    # Calculates precision by dividing the number of hits by 'k'.
    return hits / k

def calculate_recall_at_k(recommended: List, relevant: Set, k: int) -> float:
    """
    Calculates Recall at K (R@k).

    Recall@k measures the proportion of relevant items that are successfully
    retrieved within the top-k recommended list. It indicates how many of the
    user's total relevant items were actually found by the system.

    Args:
        recommended: A list of item IDs recommended to the user, ordered by relevance.
        relevant: A set of item IDs that are truly relevant to the user.
        k: The number of top recommendations to consider.

    Returns:
        The Recall@k score, a float between 0.0 and 1.0.
    """
    if not relevant or k == 0:
        return 0.0
    
    # Considers only the top 'k' recommended items.
    recommended_k = recommended[:k]
    # Counts the number of relevant items found within the top 'k' recommendations.
    hits = sum(1 for item in recommended_k if item in relevant)
    # Calculates recall by dividing the number of hits by the total number of relevant items.
    return hits / len(relevant)

def calculate_ndcg(recommended: List, relevant: Set, k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at K (NDCG@k).

    NDCG@k is a measure of ranking quality. It accounts for the position of
    relevant items in the ranked list, giving higher scores to relevant items
    that appear earlier in the list. It normalizes the score by the ideal DCG,
    where all relevant items are perfectly ranked at the top.

    Args:
        recommended: A list of item IDs recommended to the user, ordered by relevance.
        relevant: A set of item IDs that are truly relevant to the user.
        k: The number of top recommendations to consider.

    Returns:
        The NDCG@k score, a float between 0.0 and 1.0.
    """
    def dcg(scores):
        """Calculates Discounted Cumulative Gain (DCG)."""
        # Sums relevance scores discounted by their position.
        # log2(i + 2) is used as the denominator to handle 0-based indexing correctly
        # (first item's discount factor is log2(1+1)=log2(2)=1, no discount).
        return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
    
    # Assigns a relevance score of 1 to relevant items and 0 to irrelevant items.
    relevance_scores = [1 if item in relevant else 0 for item in recommended[:k]]
    
    # Returns 0.0 if there are no relevant items in the recommended list.
    if sum(relevance_scores) == 0:
        return 0.0
    
    # Creates an ideal relevance score list by sorting the actual relevance scores
    # in descending order to calculate the Ideal DCG (IDCG).
    ideal_scores = sorted(relevance_scores, reverse=True)
    
    # Calculates NDCG by dividing the actual DCG by the Ideal DCG.
    # Handles division by zero if IDCG is 0.
    return dcg(relevance_scores) / dcg(ideal_scores)

def calculate_map(recommended: List, relevant: Set) -> float:
    """
    Calculates Mean Average Precision (MAP).

    MAP is a single-figure measure of quality across recall levels. For each
    relevant item, it calculates the precision at the rank where that item is
    retrieved, and then averages these precision values. The mean of these
    average precisions is then taken across all queries (users).

    Args:
        recommended: A list of item IDs recommended to the user, ordered by relevance.
        relevant: A set of item IDs that are truly relevant to the user.

    Returns:
        The MAP score, a float between 0.0 and 1.0.
    """
    if not relevant:
        return 0.0
    
    precisions = []
    hits = 0
    
    # Iterates through the recommended list to calculate precision at each relevant item's rank.
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            # Calculates precision at the current rank and adds it to the list.
            precisions.append(hits / (i + 1))
    
    # Calculates the average of the precisions. If no relevant items were hit, returns 0.0.
    # Divides by the total number of relevant items, not just those hit.
    return sum(precisions) / len(relevant) if precisions else 0.0