# src/evaluation/tasks.py
"""
Task-based evaluation framework for recommender systems.

This module provides different evaluation tasks (retrieval, ranking, prediction, etc.)
with appropriate filtering and metrics for each scenario.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from .metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_ndcg,
    calculate_map
)
from .novelty import NoveltyMetrics


class EvaluationTask(Enum):
    """Different evaluation tasks with their specific requirements"""
    NEXT_ITEM_PREDICTION = "next_item"  # Predict next item in sequence
    RATING_PREDICTION = "rating"  # Predict exact ratings
    TOP_K_RETRIEVAL = "topk_retrieval"  # Retrieve new relevant items
    TOP_K_RANKING = "topk_ranking"  # Rank all items (including seen)
    COLD_START_USER = "cold_user"  # Evaluate on new users
    COLD_START_ITEM = "cold_item"  # Evaluate on new items
    SESSION_BASED = "session"  # Within-session recommendation
    BEYOND_ACCURACY = "beyond_accuracy"  # Diversity, novelty, fairness


class BaseEvaluator(ABC):
    """Base class for all evaluation tasks"""
    
    def __init__(
        self, 
        recommender: Any,
        test_data: pd.DataFrame,
        config: Any,
        train_data: Optional[pd.DataFrame] = None,
        val_data: Optional[pd.DataFrame] = None
    ):
        self.recommender = recommender
        self.test_data = test_data
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.top_k = int(config.recommendation.top_k)
        
        # Get training interactions from recommender if not provided
        if self.train_data is None and hasattr(recommender, 'interactions'):
            self.train_data = recommender.interactions
        
        # Build user history mapping
        self._build_user_history()
    
    def _build_user_history(self):
        """Build mapping of user -> historical items"""
        self.user_train_items = {}
        if self.train_data is not None:
            self.user_train_items = (
                self.train_data.groupby('user_id')['item_id']
                .apply(set)
                .to_dict()
            )
    
    @property
    @abstractmethod
    def filter_seen(self) -> bool:
        """Whether to filter seen items for this task"""
        pass
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Name of the evaluation task"""
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Run the evaluation and return metrics"""
        pass
    
    def get_recommendations(
        self, 
        user_id: str, 
        candidates: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Get recommendations with task-appropriate filtering"""
        if top_k is None:
            top_k = self.top_k
            
        return self.recommender.get_recommendations(
            user_id=user_id,
            top_k=top_k,
            filter_seen=self.filter_seen,
            candidates=candidates
        )
    
    def print_summary(self, results: Dict[str, float]):
        """Print evaluation summary"""
        print(f"\n{'='*50}")
        print(f"Task: {self.task_name}")
        print(f"Filter seen items: {self.filter_seen}")
        print(f"{'='*50}")
        
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric:.<40} {value:.4f}")
            else:
                print(f"{metric:.<40} {value}")


class NextItemEvaluator(BaseEvaluator):
    """
    Evaluator for next-item prediction task.
    Given a sequence of interactions, predict the next item.
    """
    
    @property
    def filter_seen(self) -> bool:
        return True  # We want to predict new items, not repeat history
    
    @property
    def task_name(self) -> str:
        return "Next-Item Prediction"
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate by holding out the last item for each user.
        Requires temporal information or assumes order in data.
        """
        hits_at_k = defaultdict(int)
        mrr_sum = 0.0
        total_users = 0
        users_skipped = 0
        
        print(f"Evaluating {self.task_name}...")
        
        # Process each user
        for user_id, user_data in tqdm(
            self.test_data.groupby('user_id'), 
            desc="Users"
        ):
            # Need at least 2 interactions
            if len(user_data) < 2:
                users_skipped += 1
                continue
            
            # Sort by timestamp if available
            if 'timestamp' in user_data.columns:
                user_data = user_data.sort_values('timestamp')
            
            # Last item is the target
            target_item = user_data.iloc[-1]['item_id']
            
            # Get recommendations
            recommendations = self.get_recommendations(user_id)
            if not recommendations:
                total_users += 1
                continue
            
            rec_items = [item for item, _ in recommendations]
            
            # Calculate metrics
            if target_item in rec_items:
                rank = rec_items.index(target_item) + 1
                mrr_sum += 1.0 / rank
                
                # Update hits at different k values
                for k in [1, 5, 10, 20, 50]:
                    if k <= len(rec_items) and rank <= k:
                        hits_at_k[k] += 1
            
            total_users += 1
        
        # Calculate final metrics
        if total_users == 0:
            return {'error': 'No users could be evaluated'}
        
        results = {
            f'hit_rate@{k}': hits / total_users 
            for k, hits in hits_at_k.items()
        }
        results.update({
            'mrr': mrr_sum / total_users,
            'n_users_evaluated': total_users,
            'n_users_skipped': users_skipped,
            'n_users_total': len(self.test_data['user_id'].unique())
        })
        
        return results


class TopKRetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for top-k retrieval task.
    Find new relevant items for users (most common evaluation).
    """
    
    @property
    def filter_seen(self) -> bool:
        return True  # Filter seen items - we want to retrieve new items
    
    @property
    def task_name(self) -> str:
        return "Top-K Retrieval (Novel Items)"
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate retrieval of novel relevant items.
        Only considers test items not seen in training.
        """
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        map_scores = []  # Store MAP scores for each user
        coverage = set()
        
        users_evaluated = 0
        users_no_novel_items = 0
        total_novel_items = 0
        total_test_items = 0
        
        unique_users = self.test_data['user_id'].unique()
        print(f"Evaluating {self.task_name} for {len(unique_users)} users...")
        
        for user_id in tqdm(unique_users, desc="Users"):
            # Get user's test items
            user_test_items = set(
                self.test_data[self.test_data['user_id'] == user_id]['item_id']
            )
            total_test_items += len(user_test_items)
            
            # Get user's training history
            user_train_items = self.user_train_items.get(user_id, set())
            
            # Only evaluate on novel items (not seen in training)
            novel_test_items = user_test_items - user_train_items
            total_novel_items += len(novel_test_items)
            
            if not novel_test_items:
                users_no_novel_items += 1
                continue
            
            # Get recommendations (with filtering)
            recommendations = self.get_recommendations(user_id)
            if not recommendations:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                ndcg_scores.append(0.0)
                map_scores.append(0.0)
                users_evaluated += 1
                continue
            
            rec_items = [item for item, _ in recommendations]
            coverage.update(rec_items)
            
            # Calculate metrics on novel items only
            precision = calculate_precision_at_k(
                rec_items, novel_test_items, self.top_k
            )
            recall = calculate_recall_at_k(
                rec_items, novel_test_items, self.top_k
            )
            ndcg = calculate_ndcg(
                rec_items, novel_test_items, self.top_k
            )
            map_score = calculate_map(
                rec_items[:self.top_k], novel_test_items
            )
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            map_scores.append(map_score)
            users_evaluated += 1
        
        # Calculate catalog coverage
        total_items_in_test = len(self.test_data['item_id'].unique())
        
        return {
            f'precision@{self.top_k}': np.mean(precision_scores) if precision_scores else 0.0,
            f'recall@{self.top_k}': np.mean(recall_scores) if recall_scores else 0.0,
            f'ndcg@{self.top_k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'map': np.mean(map_scores) if map_scores else 0.0,
            'catalog_coverage': len(coverage) / total_items_in_test,
            'n_users_evaluated': users_evaluated,
            'n_users_no_novel_items': users_no_novel_items,
            'n_users_total': len(unique_users),
            'n_unique_items_recommended': len(coverage),
            'avg_novel_items_per_user': total_novel_items / len(unique_users),
            'novel_items_ratio': total_novel_items / total_test_items if total_test_items > 0 else 0
        }


class TopKRankingEvaluator(BaseEvaluator):
    """
    Evaluator for top-k ranking task.
    Rank all items including previously seen ones.
    Tests if model learned good item representations.
    """
    
    @property
    def filter_seen(self) -> bool:
        return False  # Don't filter - test ranking of all items
    
    @property
    def task_name(self) -> str:
        return "Top-K Ranking (All Items)"
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate ranking of all items including previously seen.
        Useful for testing if model assigns high scores to relevant items.
        """
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        coverage = set()
        
        unique_users = self.test_data['user_id'].unique()
        print(f"Evaluating {self.task_name} for {len(unique_users)} users...")
        
        for user_id in tqdm(unique_users, desc="Users"):
            # Get ALL test items for user (including seen)
            user_test_items = set(
                self.test_data[self.test_data['user_id'] == user_id]['item_id']
            )
            
            if not user_test_items:
                continue
            
            # Get recommendations (without filtering)
            recommendations = self.get_recommendations(user_id)
            if not recommendations:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                ndcg_scores.append(0.0)
                continue
            
            rec_items = [item for item, _ in recommendations]
            coverage.update(rec_items)
            
            # Calculate metrics on all test items
            precision = calculate_precision_at_k(
                rec_items, user_test_items, self.top_k
            )
            recall = calculate_recall_at_k(
                rec_items, user_test_items, self.top_k
            )
            ndcg = calculate_ndcg(
                rec_items, user_test_items, self.top_k
            )
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
        
        total_items_in_test = len(self.test_data['item_id'].unique())
        
        return {
            f'precision@{self.top_k}': np.mean(precision_scores) if precision_scores else 0.0,
            f'recall@{self.top_k}': np.mean(recall_scores) if recall_scores else 0.0,
            f'ndcg@{self.top_k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'catalog_coverage': len(coverage) / total_items_in_test,
            'n_users_evaluated': len(precision_scores),
            'n_unique_items_recommended': len(coverage)
        }


class ColdStartEvaluator(BaseEvaluator):
    """
    Evaluator for cold-start scenarios.
    Tests performance on new users or new items.
    """
    
    def __init__(self, *args, cold_type: str = 'user', **kwargs):
        super().__init__(*args, **kwargs)
        self.cold_type = cold_type
        if cold_type not in ['user', 'item']:
            raise ValueError("cold_type must be 'user' or 'item'")
    
    @property
    def filter_seen(self) -> bool:
        return True  # Cold users have no history anyway
    
    @property
    def task_name(self) -> str:
        return f"Cold-Start ({self.cold_type.capitalize()})"
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on cold users or items only"""
        if self.cold_type == 'user':
            # Only evaluate on users NOT in training
            train_users = set(self.train_data['user_id'].unique())
            cold_mask = ~self.test_data['user_id'].isin(train_users)
            cold_data = self.test_data[cold_mask].copy()
            
            print(f"Found {len(cold_data['user_id'].unique())} cold users "
                  f"out of {len(self.test_data['user_id'].unique())} total test users")
            
        else:  # cold_type == 'item'
            # Only evaluate on items NOT in training
            train_items = set(self.train_data['item_id'].unique())
            cold_mask = ~self.test_data['item_id'].isin(train_items)
            cold_data = self.test_data[cold_mask].copy()
            
            print(f"Found {len(cold_data['item_id'].unique())} cold items "
                  f"out of {len(self.test_data['item_id'].unique())} total test items")
        
        if len(cold_data) == 0:
            return {
                'error': f'No cold {self.cold_type}s found in test data',
                'n_cold_entities': 0
            }
        
        # Run standard retrieval evaluation on cold subset
        retrieval_evaluator = TopKRetrievalEvaluator(
            self.recommender, cold_data, self.config, self.train_data
        )
        results = retrieval_evaluator.evaluate()
        
        # Add cold-start specific metrics
        if self.cold_type == 'user':
            results['n_cold_users'] = len(cold_data['user_id'].unique())
            results['cold_user_ratio'] = (
                results['n_cold_users'] / len(self.test_data['user_id'].unique())
            )
        else:
            results['n_cold_items'] = len(cold_data['item_id'].unique())
            results['cold_item_ratio'] = (
                results['n_cold_items'] / len(self.test_data['item_id'].unique())
            )
        
        return results


class BeyondAccuracyEvaluator(BaseEvaluator):
    """
    Evaluator for beyond-accuracy metrics.
    Includes diversity, novelty, serendipity, and fairness.
    """
    
    @property
    def filter_seen(self) -> bool:
        return True  # We want novel recommendations
    
    @property
    def task_name(self) -> str:
        return "Beyond-Accuracy Metrics"
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate diversity, novelty, and other beyond-accuracy metrics.
        """
        # First get standard accuracy metrics
        retrieval_evaluator = TopKRetrievalEvaluator(
            self.recommender, self.test_data, self.config, self.train_data
        )
        accuracy_results = retrieval_evaluator.evaluate()
        
        # Now calculate beyond-accuracy metrics
        all_recommendations = []
        user_recommendations = {}
        diversity_scores = []
        
        # Initialize novelty calculator if available
        novelty_calculator = None
        if hasattr(self.recommender.dataset, 'get_item_popularity'):
            item_popularity = self.recommender.dataset.get_item_popularity()
            user_history = [
                (row['user_id'], row['item_id']) 
                for _, row in self.train_data.iterrows()
            ]
            novelty_calculator = NoveltyMetrics(item_popularity, user_history)
        
        unique_users = self.test_data['user_id'].unique()
        print(f"\nCalculating beyond-accuracy metrics...")
        
        for user_id in tqdm(unique_users[:1000], desc="Users"):  # Sample for efficiency
            # Get diverse recommendations if available
            if hasattr(self.recommender, 'get_diverse_recommendations'):
                recs, metrics = self.recommender.get_diverse_recommendations(
                    user_id,
                    top_k=self.top_k,
                    diversity_weight=0.3,
                    novelty_weight=0.2,
                    filter_seen=True
                )
                rec_items = [r['item_id'] for r in recs]
            else:
                recs = self.get_recommendations(user_id)
                rec_items = [item for item, _ in recs]
            
            user_recommendations[user_id] = rec_items
            all_recommendations.extend(rec_items)
            
            # Calculate intra-list diversity if possible
            if hasattr(self.recommender, 'dataset') and len(rec_items) > 1:
                # Simple diversity: ratio of unique items (can be enhanced)
                diversity = len(set(rec_items)) / len(rec_items)
                diversity_scores.append(diversity)
        
        # Aggregate metrics
        unique_recommended = set(all_recommendations)
        total_items = len(self.train_data['item_id'].unique())
        
        results = {
            **accuracy_results,  # Include accuracy metrics
            'aggregate_diversity': len(unique_recommended),
            'catalog_coverage_ratio': len(unique_recommended) / total_items,
            'gini_coefficient': self._calculate_gini(all_recommendations),
            'avg_intra_list_diversity': np.mean(diversity_scores) if diversity_scores else 0.0,
        }
        
        # Add novelty metrics if available
        if novelty_calculator:
            novelty_scores = []
            for rec_list in user_recommendations.values():
                if rec_list:
                    nov_metrics = novelty_calculator.calculate_metrics(rec_list)
                    novelty_scores.append(nov_metrics.get('avg_self_information', 0))
            
            results['avg_novelty'] = np.mean(novelty_scores) if novelty_scores else 0.0
        
        return results
    
    def _calculate_gini(self, recommendations: List[str]) -> float:
        """Calculate Gini coefficient for recommendation distribution"""
        from collections import Counter
        
        if not recommendations:
            return 0.0
        
        # Count item frequencies
        item_counts = Counter(recommendations)
        counts = np.array(list(item_counts.values()))
        
        # Sort counts
        counts = np.sort(counts)
        n = len(counts)
        
        # Calculate Gini
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n


class SessionBasedEvaluator(BaseEvaluator):
    """
    Evaluator for session-based recommendations.
    Evaluates within-session performance.
    """
    
    @property
    def filter_seen(self) -> bool:
        return False  # Within session, might recommend seen items
    
    @property
    def task_name(self) -> str:
        return "Session-Based Recommendation"
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate by treating each user's interactions as a session.
        Uses sliding window approach.
        """
        # Implementation depends on session definition
        # This is a placeholder
        return {
            'session_precision@{}'.format(self.top_k): 0.0,
            'session_recall@{}'.format(self.top_k): 0.0,
            'note': 'Session-based evaluation requires session boundaries'
        }


# Factory function
def create_evaluator(
    task: EvaluationTask,
    recommender: Any,
    test_data: pd.DataFrame,
    config: Any,
    train_data: Optional[pd.DataFrame] = None,
    **kwargs
) -> BaseEvaluator:
    """
    Factory function to create the appropriate evaluator for a given task.
    
    Args:
        task: The evaluation task type
        recommender: The recommender instance
        test_data: Test dataset
        config: Configuration object
        train_data: Training dataset (optional)
        **kwargs: Additional arguments for specific evaluators
    
    Returns:
        An evaluator instance
    """
    evaluator_classes = {
        EvaluationTask.NEXT_ITEM_PREDICTION: NextItemEvaluator,
        EvaluationTask.TOP_K_RETRIEVAL: TopKRetrievalEvaluator,
        EvaluationTask.TOP_K_RANKING: TopKRankingEvaluator,
        EvaluationTask.COLD_START_USER: lambda *args, **kw: ColdStartEvaluator(*args, cold_type='user', **kw),
        EvaluationTask.COLD_START_ITEM: lambda *args, **kw: ColdStartEvaluator(*args, cold_type='item', **kw),
        EvaluationTask.SESSION_BASED: SessionBasedEvaluator,
        EvaluationTask.BEYOND_ACCURACY: BeyondAccuracyEvaluator,
    }
    
    evaluator_class = evaluator_classes.get(task)
    if not evaluator_class:
        raise ValueError(f"No evaluator implemented for task: {task}")
    
    return evaluator_class(
        recommender=recommender,
        test_data=test_data,
        config=config,
        train_data=train_data,
        **kwargs
    )