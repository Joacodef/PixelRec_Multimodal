"""
Standard recommendation metrics
"""
import numpy as np
from typing import List, Set

def calculate_precision_at_k(recommended: List, relevant: Set, k: int) -> float:
    """Calculate Precision@k"""
    if not recommended or k == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k

def calculate_recall_at_k(recommended: List, relevant: Set, k: int) -> float:
    """Calculate Recall@k"""
    if not relevant or k == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)

def calculate_ndcg(recommended: List, relevant: Set, k: int) -> float:
    """Calculate NDCG@k"""
    def dcg(scores):
        return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
    
    relevance_scores = [1 if item in relevant else 0 for item in recommended[:k]]
    
    if sum(relevance_scores) == 0:
        return 0.0
    
    ideal_scores = sorted(relevance_scores, reverse=True)
    
    return dcg(relevance_scores) / dcg(ideal_scores)

def calculate_map(recommended: List, relevant: Set) -> float:
    """Calculate Mean Average Precision"""
    if not relevant:
        return 0.0
    
    precisions = []
    hits = 0
    
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            precisions.append(hits / (i + 1))
    
    return sum(precisions) / len(relevant) if precisions else 0.0