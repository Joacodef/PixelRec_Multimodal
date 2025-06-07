# tests/unit/src/evaluation/test_metrics.py
"""
Unit tests for the standard recommendation metrics functions.
"""
import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_ndcg,
    calculate_map
)

class TestStandardMetrics(unittest.TestCase):
    """Test cases for standard recommendation metrics."""

    def test_calculate_precision_at_k(self):
        """Tests the Precision@k calculation."""
        # Setup: A list of recommended items and a set of relevant items.
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = {'item2', 'item4', 'item6'}

        # Case 1: k=3. One hit ('item2') out of 3 recommendations. Expected precision: 1/3.
        self.assertAlmostEqual(calculate_precision_at_k(recommended, relevant, k=3), 1 / 3)

        # Case 2: k=5. Two hits ('item2', 'item4') out of 5 recommendations. Expected precision: 2/5.
        self.assertAlmostEqual(calculate_precision_at_k(recommended, relevant, k=5), 2 / 5)

        # Case 3: No hits. Expected precision: 0.
        self.assertAlmostEqual(calculate_precision_at_k(recommended, {'item9'}, k=5), 0.0)

        # Edge Case 4: k is larger than the number of recommendations. Should still divide by k.
        self.assertAlmostEqual(calculate_precision_at_k(recommended, relevant, k=10), 2 / 10)
        
        # Edge Case 5: k=0. Expected precision: 0.
        self.assertEqual(calculate_precision_at_k(recommended, relevant, k=0), 0.0)
        
        # Edge Case 6: Empty recommendations. Expected precision: 0.
        self.assertEqual(calculate_precision_at_k([], relevant, k=5), 0.0)

    def test_calculate_recall_at_k(self):
        """Tests the Recall@k calculation."""
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = {'item2', 'item4', 'item6', 'item7'} # Total of 4 relevant items.

        # Case 1: k=3. One hit ('item2') out of 4 total relevant items. Expected recall: 1/4.
        self.assertAlmostEqual(calculate_recall_at_k(recommended, relevant, k=3), 1 / 4)

        # Case 2: k=5. Two hits ('item2', 'item4') out of 4 total relevant items. Expected recall: 2/4.
        self.assertAlmostEqual(calculate_recall_at_k(recommended, relevant, k=5), 2 / 4)

        # Case 3: No hits. Expected recall: 0.
        self.assertAlmostEqual(calculate_recall_at_k(recommended, {'item9'}, k=5), 0.0)

        # Edge Case 4: Empty set of relevant items. Should return 0 to avoid division by zero.
        self.assertEqual(calculate_recall_at_k(recommended, set(), k=5), 0.0)
        
        # Edge Case 5: Empty recommendations. Expected recall: 0.
        self.assertEqual(calculate_recall_at_k([], relevant, k=5), 0.0)

    def test_calculate_ndcg(self):
        """Tests the Normalized Discounted Cumulative Gain (NDCG) calculation."""
        relevant = {'item1', 'item2', 'item3'}

        # Case 1: Perfect ranking. Expected NDCG: 1.0.
        perfect_rec = ['item1', 'item2', 'item3']
        self.assertAlmostEqual(calculate_ndcg(perfect_rec, relevant, k=3), 1.0)
        
        # Case 2: Imperfect ranking.
        imperfect_rec = ['item4', 'item1', 'item2']
        # DCG = (1/log2(2+1)) + (1/log2(3+1)) = 1/log2(3) + 1/log2(4)
        dcg = (1 / np.log2(3)) + (1 / np.log2(4))
        # IDCG = (1/log2(1+1)) + (1/log2(2+1)) = 1/log2(2) + 1/log2(3)
        idcg = (1 / np.log2(2)) + (1 / np.log2(3))
        self.assertAlmostEqual(calculate_ndcg(imperfect_rec, relevant, k=3), dcg / idcg)

        # Case 3: No relevant items in recommendations. Expected NDCG: 0.0.
        no_hit_rec = ['item4', 'item5', 'item6']
        self.assertEqual(calculate_ndcg(no_hit_rec, relevant, k=3), 0.0)

        # Edge Case 4: Empty recommendations. Expected NDCG: 0.0.
        self.assertEqual(calculate_ndcg([], relevant, k=3), 0.0)

    def test_calculate_map(self):
        """Tests the Mean Average Precision (MAP) calculation."""
        relevant = {'item1', 'item3', 'item5'}

        # Case 1: Recommendations with multiple hits.
        rec = ['item1', 'item2', 'item3', 'item4', 'item5']
        # Hit 1 ('item1') at rank 1: Precision = 1/1
        # Hit 2 ('item3') at rank 3: Precision = 2/3
        # Hit 3 ('item5') at rank 5: Precision = 3/5
        # AP = (1/1 + 2/3 + 3/5) / 3
        expected_ap = (1.0 + (2/3) + (3/5)) / 3.0
        self.assertAlmostEqual(calculate_map(rec, relevant), expected_ap)
        
        # Case 2: Recommendations with a single hit.
        rec_one_hit = ['item2', 'item4', 'item1']
        # Hit 1 ('item1') at rank 3: Precision = 1/3
        # AP = (1/3) / 3
        expected_ap_one_hit = (1/3) / 3.0
        self.assertAlmostEqual(calculate_map(rec_one_hit, relevant), expected_ap_one_hit)

        # Case 3: No hits. Expected MAP: 0.0.
        self.assertEqual(calculate_map(['item2', 'item4'], relevant), 0.0)
        
        # Edge Case 4: Empty relevant set. Expected MAP: 0.0.
        self.assertEqual(calculate_map(rec, set()), 0.0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)