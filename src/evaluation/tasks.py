# src/evaluation/tasks.py 
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import random


class EvaluationTask(Enum):
    """Simplified evaluation tasks - only the most commonly used"""
    TOP_K_RETRIEVAL = "top_k_retrieval"
    TOP_K_RANKING = "top_k_ranking"


class BaseEvaluator(ABC):
    """Base class for all evaluators"""
    
    def __init__(self, recommender, test_data: pd.DataFrame, config, task_name: str, **kwargs):
        self.recommender = recommender
        self.test_data = test_data
        self.config = config
        self.task_name = task_name
        self.top_k = getattr(config.recommendation, 'top_k', 50)
        self.filter_seen = kwargs.get('filter_seen', True)
        
        # Ensure consistent string type for test data
        self.test_data = self.test_data.copy()
        self.test_data['user_id'] = self.test_data['user_id'].astype(str)
        self.test_data['item_id'] = self.test_data['item_id'].astype(str)
        
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Perform evaluation and return metrics"""
        pass
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print(f"\n=== {self.task_name} Results ===")
        for metric, value in results.items():
            if metric not in ['evaluation_metadata', 'predictions']:
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")


class TopKRetrievalEvaluator(BaseEvaluator):
    """
    Top-K retrieval evaluation with optional negative sampling for efficiency
    Measures how well the model retrieves relevant items from a candidate set
    """
    
    def __init__(self, recommender, test_data: pd.DataFrame, config, 
                 use_sampling: bool = True, num_negatives: int = 100, 
                 sampling_strategy: str = 'random', **kwargs):
        super().__init__(recommender, test_data, config, "Top-K Retrieval", **kwargs)
        self.use_sampling = use_sampling
        self.num_negatives = num_negatives
        self.sampling_strategy = sampling_strategy
        
    def _get_all_item_ids(self) -> List[str]:
        """Get all available item IDs as strings"""
        # Try to get from dataset first (most reliable)
        if (hasattr(self.recommender, 'dataset') and 
            hasattr(self.recommender.dataset, 'item_encoder') and
            hasattr(self.recommender.dataset.item_encoder, 'classes_') and
            self.recommender.dataset.item_encoder.classes_ is not None):
            return [str(item_id) for item_id in self.recommender.dataset.item_encoder.classes_]
        
        # Fallback to test data items
        return list(self.test_data['item_id'].unique())
        
    def _sample_negatives(self, user_id: str, positive_items: List[str]) -> List[str]:
        """Sample negative items for evaluation - Fixed negative sampling logic"""
        all_items = self._get_all_item_ids()
        positive_items_set = set(str(item) for item in positive_items)
        
        # Remove positive items
        negative_candidates = [item for item in all_items if item not in positive_items_set]
        
        if len(negative_candidates) < self.num_negatives:
            return negative_candidates
        
        if not negative_candidates:
            return []
        
        # Create deterministic but user-specific seed for consistent evaluation
        # Use hash of user_id to ensure different users get different but reproducible samples
        user_seed = hash(str(user_id)) % (2**31)  # Ensure positive 32-bit integer
        np.random.seed(user_seed)
        random.seed(user_seed)

        if self.sampling_strategy == 'random':
            # Use random.sample for better random distribution
            num_to_sample = min(self.num_negatives, len(negative_candidates))
            return random.sample(negative_candidates, num_to_sample)
            
        elif self.sampling_strategy == 'popularity':
            # Sample based on item popularity (more popular items more likely)
            # Ensure consistent string types for item_counts lookup
            item_counts = self.test_data['item_id'].astype(str).value_counts()
            
            # Calculate weights for each negative candidate
            weights = []
            valid_candidates = []
            
            for item in negative_candidates:
                item_str = str(item)
                count = item_counts.get(item_str, 1)  # Default count of 1 for unseen items
                if count > 0:  # Only include items with positive counts
                    weights.append(float(count))
                    valid_candidates.append(item_str)
            
            if not valid_candidates:
                return []
            
            # Normalize weights to probabilities
            weights = np.array(weights, dtype=np.float64)
            if weights.sum() == 0:
                # All weights are zero, fall back to uniform sampling
                weights = np.ones(len(weights))
            weights = weights / weights.sum()
            
            # Sample with replacement=False
            num_to_sample = min(self.num_negatives, len(valid_candidates))
            try:
                sampled_indices = np.random.choice(
                    len(valid_candidates), 
                    size=num_to_sample, 
                    replace=False, 
                    p=weights
                )
                return [valid_candidates[i] for i in sampled_indices]
            except ValueError as e:
                # If sampling fails, fall back to random sampling
                print(f"Warning: Popularity sampling failed for user {user_id}: {e}. Using random sampling.")
                return random.sample(valid_candidates, num_to_sample)
                
        else:  # popularity_inverse
            # Sample less popular items more often (inverse popularity)
            item_counts = self.test_data['item_id'].astype(str).value_counts()
            
            # Calculate inverse weights for each negative candidate
            weights = []
            valid_candidates = []
            
            for item in negative_candidates:
                item_str = str(item)
                count = item_counts.get(item_str, 1)  # Default count of 1 for unseen items
                if count > 0:  # Only include items with positive counts
                    # Inverse weight: less popular items get higher weight
                    inv_weight = 1.0 / float(count)
                    weights.append(inv_weight)
                    valid_candidates.append(item_str)
            
            if not valid_candidates:
                return []
            
            # Normalize weights to probabilities
            weights = np.array(weights, dtype=np.float64)
            if weights.sum() == 0:
                # All weights are zero, fall back to uniform sampling
                weights = np.ones(len(weights))
            weights = weights / weights.sum()
            
            # Sample with replacement=False
            num_to_sample = min(self.num_negatives, len(valid_candidates))
            try:
                sampled_indices = np.random.choice(
                    len(valid_candidates), 
                    size=num_to_sample, 
                    replace=False, 
                    p=weights
                )
                return [valid_candidates[i] for i in sampled_indices]
            except ValueError as e:
                # If sampling fails, fall back to random sampling
                print(f"Warning: Inverse popularity sampling failed for user {user_id}: {e}. Using random sampling.")
                return random.sample(valid_candidates, num_to_sample)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate top-K retrieval performance"""
        print(f"Evaluating Top-K Retrieval (K={self.top_k})")
        if self.use_sampling:
            print(f"Using negative sampling: {self.num_negatives} negatives per user, strategy: {self.sampling_strategy}")
        
        metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'f1_at_k': [],
            'hit_rate_at_k': [],
            'ndcg_at_k': [],
            'mrr': []
        }
        all_predictions = {} # Collect predictions here
        
        user_groups = self.test_data.groupby('user_id')
        
        # Set global seed for reproducible evaluation order
        np.random.seed(42)
        random.seed(42)
        
        for user_id, user_interactions in tqdm(user_groups, desc="Evaluating users"):
            # Ensure user_id is string
            user_id = str(user_id)
            positive_items = [str(item) for item in user_interactions['item_id'].tolist()]
            
            if self.use_sampling:
                # Create evaluation set with positives + sampled negatives
                negative_items = self._sample_negatives(user_id, positive_items)
                candidate_items = positive_items + negative_items
                
                # Create deterministic shuffle based on user_id for reproducibility
                user_shuffle_seed = hash(str(user_id) + "shuffle") % (2**31)
                local_random = random.Random(user_shuffle_seed)
                local_random.shuffle(candidate_items)  # Shuffle to avoid bias
            else:
                # Use all items as candidates (much slower but complete)
                candidate_items = None
            
            try:
                # Get recommendations - ensure we pass string user_id and get string item_ids back
                recommendations = self.recommender.get_recommendations(
                    user_id=user_id,
                    top_k=self.top_k,
                    filter_seen=self.filter_seen,
                    candidates=candidate_items
                )
                
                # Ensure recommendations contain string item IDs
                if recommendations:
                    # Convert any non-string item IDs to strings
                    recommendations = [(str(item_id), score) for item_id, score in recommendations]
                
                # Store predictions
                all_predictions[user_id] = recommendations
                
                if not recommendations:
                    # No recommendations - assign worst scores
                    metrics['precision_at_k'].append(0.0)
                    metrics['recall_at_k'].append(0.0)
                    metrics['f1_at_k'].append(0.0)
                    metrics['hit_rate_at_k'].append(0.0)
                    metrics['ndcg_at_k'].append(0.0)
                    metrics['mrr'].append(0.0)
                    continue
                
                recommended_items = [str(item_id) for item_id, _ in recommendations]
                
                # Calculate metrics
                relevant_items = set(positive_items)
                recommended_set = set(recommended_items)
                
                # Precision@K
                precision = len(relevant_items & recommended_set) / len(recommended_set) if recommended_set else 0.0
                
                # Recall@K
                recall = len(relevant_items & recommended_set) / len(relevant_items) if relevant_items else 0.0
                
                # F1@K
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Hit Rate@K (binary: did we hit at least one relevant item?)
                hit_rate = 1.0 if len(relevant_items & recommended_set) > 0 else 0.0
                
                # NDCG@K
                ndcg = self._calculate_ndcg(recommended_items, relevant_items, self.top_k)
                
                # MRR (Mean Reciprocal Rank)
                mrr = 0.0
                for i, item in enumerate(recommended_items, 1):
                    if item in relevant_items:
                        mrr = 1.0 / i
                        break
                
                metrics['precision_at_k'].append(precision)
                metrics['recall_at_k'].append(recall)
                metrics['f1_at_k'].append(f1)
                metrics['hit_rate_at_k'].append(hit_rate)
                metrics['ndcg_at_k'].append(ndcg)
                metrics['mrr'].append(mrr)
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                # Assign worst scores for failed cases
                for metric_list in metrics.values():
                    metric_list.append(0.0)
        
        # Aggregate results
        results = {}
        for metric_name, values in metrics.items():
            if values:
                results[f"avg_{metric_name}"] = np.mean(values)
                results[f"std_{metric_name}"] = np.std(values)
            else:
                results[f"avg_{metric_name}"] = 0.0
                results[f"std_{metric_name}"] = 0.0
        
        results['num_users_evaluated'] = len(user_groups)
        results['evaluation_method'] = 'negative_sampling' if self.use_sampling else 'full_evaluation'
        results['predictions'] = all_predictions # Add predictions to results
        
        return results
    
    def _calculate_ndcg(self, recommended_items: List[str], relevant_items: set, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        if not relevant_items:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k], 1):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG calculation (perfect ranking)
        num_relevant = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        return dcg / idcg if idcg > 0 else 0.0


class TopKRankingEvaluator(BaseEvaluator):
    """
    Top-K ranking evaluation - focuses on ranking quality of known relevant items
    Assumes test items are all relevant and measures how well they are ranked
    """
    
    def __init__(self, recommender, test_data: pd.DataFrame, config, **kwargs):
        super().__init__(recommender, test_data, config, "Top-K Ranking", **kwargs)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate ranking performance"""
        print(f"Evaluating Top-K Ranking (K={self.top_k})")
        
        metrics = {
            'avg_rank': [],
            'median_rank': [],
            'mrr': [],
            'hit_rate_at_k': [],
            'ndcg_at_k': []
        }
        all_predictions = {} # Collect predictions here
        
        user_groups = self.test_data.groupby('user_id')
        
        for user_id, user_interactions in tqdm(user_groups, desc="Evaluating ranking"):
            # Ensure user_id is string
            user_id = str(user_id)
            test_items = [str(item) for item in user_interactions['item_id'].tolist()]
            
            try:
                # Get scores for test items specifically
                item_scores = []
                for item_id in test_items:
                    try:
                        # Get score for this specific item - ensure both IDs are strings
                        score = self.recommender.get_item_score(str(user_id), str(item_id))
                        item_scores.append((str(item_id), score))
                    except Exception as e:
                        print(f"Error getting score for user {user_id}, item {item_id}: {e}")
                        item_scores.append((str(item_id), 0.0))
                
                if not item_scores:
                    # No scores available
                    for metric_list in metrics.values():
                        metric_list.append(0.0)
                    continue
                
                # Store predictions before sorting for ranking metrics
                all_predictions[user_id] = item_scores
                
                # Sort by score (descending)
                item_scores.sort(key=lambda x: x[1], reverse=True)
                ranked_items = [str(item_id) for item_id, _ in item_scores]
                
                # Calculate ranking metrics
                ranks = []
                for i, (item_id, _) in enumerate(item_scores, 1):
                    ranks.append(i)
                
                # Average and median rank
                avg_rank = np.mean(ranks)
                median_rank = np.median(ranks)
                
                # MRR (since all test items are relevant, MRR is 1/rank of best item)
                mrr = 1.0 / ranks[0] if ranks else 0.0
                
                # Hit rate@K (fraction of test items in top-K)
                hits_in_top_k = sum(1 for rank in ranks if rank <= self.top_k)
                hit_rate = hits_in_top_k / len(test_items) if test_items else 0.0
                
                # NDCG@K (treating all test items as equally relevant)
                relevant_items = set(test_items)
                ndcg = self._calculate_ndcg(ranked_items, relevant_items, self.top_k)
                
                metrics['avg_rank'].append(avg_rank)
                metrics['median_rank'].append(median_rank)
                metrics['mrr'].append(mrr)
                metrics['hit_rate_at_k'].append(hit_rate)
                metrics['ndcg_at_k'].append(ndcg)
                
            except Exception as e:
                print(f"Error evaluating ranking for user {user_id}: {e}")
                # Assign worst scores for failed cases
                metrics['avg_rank'].append(float('inf'))
                metrics['median_rank'].append(float('inf'))
                metrics['mrr'].append(0.0)
                metrics['hit_rate_at_k'].append(0.0)
                metrics['ndcg_at_k'].append(0.0)
        
        # Aggregate results
        results = {}
        for metric_name, values in metrics.items():
            if values:
                if metric_name in ['avg_rank', 'median_rank']:
                    # Filter out infinite values for rank metrics
                    finite_values = [v for v in values if np.isfinite(v)]
                    if finite_values:
                        results[f"avg_{metric_name}"] = np.mean(finite_values)
                        results[f"std_{metric_name}"] = np.std(finite_values)
                    else:
                        results[f"avg_{metric_name}"] = float('inf')
                        results[f"std_{metric_name}"] = 0.0
                else:
                    results[f"avg_{metric_name}"] = np.mean(values)
                    results[f"std_{metric_name}"] = np.std(values)
            else:
                results[f"avg_{metric_name}"] = 0.0
                results[f"std_{metric_name}"] = 0.0
        
        results['num_users_evaluated'] = len(user_groups)
        results['predictions'] = all_predictions # Add predictions to results
        
        return results
    
    def _calculate_ndcg(self, ranked_items: List[str], relevant_items: set, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        if not relevant_items:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, item in enumerate(ranked_items[:k], 1):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG calculation (perfect ranking)
        num_relevant = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        return dcg / idcg if idcg > 0 else 0.0


def create_evaluator(task: EvaluationTask, recommender, test_data: pd.DataFrame, 
                    config, **kwargs) -> BaseEvaluator:
    """Factory function to create appropriate evaluator"""
    
    if task == EvaluationTask.TOP_K_RETRIEVAL:
        return TopKRetrievalEvaluator(
            recommender=recommender,
            test_data=test_data,
            config=config,
            **kwargs
        )
    elif task == EvaluationTask.TOP_K_RANKING:
        return TopKRankingEvaluator(
            recommender=recommender,
            test_data=test_data,
            config=config,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown evaluation task: {task}")


# Backward compatibility - map old task names to new ones
TASK_MAPPING = {
    'retrieval': EvaluationTask.TOP_K_RETRIEVAL,
    'ranking': EvaluationTask.TOP_K_RANKING,
    # Removed tasks will raise an error
    'next_item': None,
    'cold_user': None,
    'cold_item': None,
    'beyond_accuracy': None,
    'session_based': None
}


def get_task_from_string(task_name: str) -> EvaluationTask:
    """Convert string task name to EvaluationTask enum"""
    if task_name in TASK_MAPPING:
        task = TASK_MAPPING[task_name]
        if task is None:
            raise ValueError(f"Task '{task_name}' has been removed in the simplified evaluation framework. "
                           f"Available tasks: {list(EvaluationTask.__members__.keys())}")
        return task
    
    # Try direct enum lookup
    try:
        return EvaluationTask(task_name)
    except ValueError:
        available_tasks = list(EvaluationTask.__members__.keys())
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available_tasks}")