# tests/unit/src/evaluation/test_tasks.py
"""
Unit tests for the evaluation task framework, including evaluators and helpers.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.evaluation.tasks import (
    EvaluationTask,
    TopKRetrievalEvaluator,
    TopKRankingEvaluator,
    create_evaluator,
    get_task_from_string
)

class TestTaskHelpers(unittest.TestCase):
    """Tests the helper and factory functions in tasks.py."""

    def test_get_task_from_string(self):
        """Tests the conversion of string names to EvaluationTask enums."""
        self.assertEqual(get_task_from_string('retrieval'), EvaluationTask.TOP_K_RETRIEVAL)
        self.assertEqual(get_task_from_string('ranking'), EvaluationTask.TOP_K_RANKING)
        with self.assertRaises(ValueError):
            get_task_from_string('invalid_task_name')
        with self.assertRaises(ValueError):
            # Tests that a removed task raises an error.
            get_task_from_string('cold_start') 

    def test_create_evaluator(self):
        """Tests the factory function for creating evaluator instances."""
        mock_recommender = MagicMock()
        mock_config = MagicMock()
        test_data = pd.DataFrame({'user_id': ['u1'], 'item_id': ['i1']})

        # Verifies that the correct evaluator class is instantiated for each task type.
        retrieval_evaluator = create_evaluator(EvaluationTask.TOP_K_RETRIEVAL, mock_recommender, test_data, mock_config)
        self.assertIsInstance(retrieval_evaluator, TopKRetrievalEvaluator)

        ranking_evaluator = create_evaluator(EvaluationTask.TOP_K_RANKING, mock_recommender, test_data, mock_config)
        self.assertIsInstance(ranking_evaluator, TopKRankingEvaluator)


class TestTopKRetrievalEvaluator(unittest.TestCase):
    """Test cases for the TopKRetrievalEvaluator."""

    def setUp(self):
        """Set up mock objects and sample data for testing retrieval."""
        # Creates a mock recommender that returns predictable recommendations.
        self.mock_recommender = MagicMock()
        self.mock_recommender.get_recommendations.return_value = [
            ('i2', 0.9), ('i5', 0.8), ('i1', 0.7)
        ]
        # Mocks the dataset attribute needed for negative sampling.
        self.mock_recommender.dataset.item_encoder.classes_ = [f'i{j}' for j in range(1, 101)]

        # Creates mock configuration and sample test data.
        self.mock_config = MagicMock()
        self.mock_config.recommendation.top_k = 3
        self.test_data = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2'],
            'item_id': ['i1', 'i3', 'i4', 'i5']
        })

    def test_negative_sampling(self):
        """Tests the logic for sampling negative items."""
        evaluator = TopKRetrievalEvaluator(
            self.mock_recommender, self.test_data, self.mock_config, num_negatives=50
        )
        positive_items = ['i1', 'i3']
        negatives = evaluator._sample_negatives('u1', positive_items)
        
        # Verifies that the correct number of negative samples are generated.
        self.assertEqual(len(negatives), 50)
        # Verifies that none of the user's positive items are included in the negative samples.
        self.assertTrue(set(negatives).isdisjoint(set(positive_items)))

    def test_evaluate_end_to_end(self):
        """Tests the full evaluation pipeline for retrieval metrics."""
        # Configures the mock recommender to return specific results for each user.
        def get_recs_side_effect(user_id, **kwargs):
            if user_id == 'u1':
                # For user u1, recommend one relevant item ('i1') and two irrelevant items.
                return [('i2', 0.9), ('i50', 0.8), ('i1', 0.7)]
            if user_id == 'u2':
                # For user u2, recommend no relevant items.
                return [('i60', 0.9), ('i70', 0.8), ('i80', 0.7)]
            return []
        self.mock_recommender.get_recommendations.side_effect = get_recs_side_effect

        evaluator = TopKRetrievalEvaluator(
            self.mock_recommender, self.test_data, self.mock_config, use_sampling=False, num_workers=0
        )
        results = evaluator.evaluate()

        # Expected metrics for u1: Precision=1/3, Recall=1/2, MRR=1/3, NDCG=1/log2(4) / 1/log2(2)
        # Expected metrics for u2: All metrics are 0.
        # Averages are calculated over the two users.
        self.assertAlmostEqual(results['avg_precision_at_k'], (1/3 + 0)/2)
        self.assertAlmostEqual(results['avg_recall_at_k'], (1/2 + 0)/2)
        self.assertAlmostEqual(results['avg_mrr'], (1/3 + 0)/2)
        self.assertEqual(results['num_users_evaluated'], 2)


class TestTopKRankingEvaluator(unittest.TestCase):
    """Test cases for the TopKRankingEvaluator."""

    def setUp(self):
        """Set up mock objects and sample data for testing ranking."""
        # Creates a mock recommender where the score is derived from the item ID for predictable ranking.
        self.mock_recommender = MagicMock()
        def get_score_side_effect(user_id, item_id):
            return float(item_id.replace('i', '')) / 10.0
        self.mock_recommender.get_item_score.side_effect = get_score_side_effect

        # Creates mock configuration and sample test data.
        self.mock_config = MagicMock()
        self.mock_config.recommendation.top_k = 5
        self.test_data = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1'],
            'item_id': ['i5', 'i2', 'i8'] # True ranking: i8, i5, i2
        })

    def test_evaluate_ranking(self):
        """Tests the full evaluation pipeline for ranking metrics."""
        evaluator = TopKRankingEvaluator(self.mock_recommender, self.test_data, self.mock_config)
        results = evaluator.evaluate()

        # The mock recommender scores items as i8 > i5 > i2.
        # The predicted ranks for the test items will be 1, 2, and 3.
        # Expected avg_rank = (1+2+3)/3 = 2.0
        self.assertAlmostEqual(results['avg_avg_rank'], 2.0)
        # Expected median_rank = 2.0
        self.assertAlmostEqual(results['avg_median_rank'], 2.0)
        # Expected MRR = 1/1 = 1.0 (since the top-ranked item is always in the test set).
        self.assertAlmostEqual(results['avg_mrr'], 1.0)
        # All 3 items are in the top 5, so hit rate is 3/3 = 1.0
        self.assertAlmostEqual(results['avg_hit_rate_at_k'], 1.0)
        # The ranking is perfect, so NDCG should be 1.0
        self.assertAlmostEqual(results['avg_ndcg_at_k'], 1.0)
        self.assertEqual(results['num_users_evaluated'], 1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)