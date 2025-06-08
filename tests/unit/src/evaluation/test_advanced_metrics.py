# tests/unit/src/evaluation/test_advanced_metrics.py
"""
Unit tests for the AdvancedMetrics and FairnessMetrics classes.
"""
import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.evaluation.advanced_metrics import AdvancedMetrics, FairnessMetrics

class TestAdvancedMetrics(unittest.TestCase):
    """Test cases for the AdvancedMetrics calculation methods."""

    def test_calculate_mrr(self):
        """Tests the Mean Reciprocal Rank calculation."""
        # Setup: Three users with recommendations and their relevant items.
        recommendations = [
            ['item1', 'item2', 'item3'],  # User 1: First hit at rank 2 (1/2)
            ['item4', 'item5', 'item6'],  # User 2: First hit at rank 1 (1/1)
            ['item7', 'item8', 'item9']   # User 3: No hits (0)
        ]
        relevant_items = [
            {'item2', 'item10'},
            {'item4'},
            {'item11'}
        ]
        # Expected MRR = (0.5 + 1.0 + 0.0) / 3 = 0.5
        mrr = AdvancedMetrics.calculate_mrr(recommendations, relevant_items)
        self.assertAlmostEqual(mrr, 0.5)

        # Tests edge case with no relevant items.
        self.assertAlmostEqual(AdvancedMetrics.calculate_mrr([['a']], [{'b'}]), 0.0)
        # Tests edge case with empty inputs, which should now correctly return 0.0.
        self.assertAlmostEqual(AdvancedMetrics.calculate_mrr([], []), 0.0)

    def test_calculate_hit_rate(self):
        """Tests the Hit Rate calculation."""
        recommendations = [
            ['item1', 'item2', 'item3'],  # Hit
            ['item4', 'item5', 'item6'],  # Hit
            ['item7', 'item8', 'item9']   # Miss
        ]
        relevant_items = [
            {'item2'},
            {'item5'},
            {'item10'}
        ]
        # Expected Hit Rate = 2 hits / 3 users = 0.666...
        hit_rate = AdvancedMetrics.calculate_hit_rate(recommendations, relevant_items)
        self.assertAlmostEqual(hit_rate, 2 / 3)
        
        # Tests with all users having a hit.
        all_hit_rate = AdvancedMetrics.calculate_hit_rate([['a']], [{'a'}])
        self.assertAlmostEqual(all_hit_rate, 1.0)
        # Tests with no users having a hit.
        no_hit_rate = AdvancedMetrics.calculate_hit_rate([['a']], [{'b'}])
        self.assertAlmostEqual(no_hit_rate, 0.0)

    def test_calculate_gini_coefficient(self):
        """Tests the Gini coefficient calculation for distribution inequality."""
        # Case 1: Perfect equality (all items recommended once).
        equal_dist = {'item1': 1, 'item2': 1, 'item3': 1, 'item4': 1}
        gini_equal = AdvancedMetrics.calculate_gini_coefficient(equal_dist)
        self.assertAlmostEqual(gini_equal, 0.0)

        # Case 2: Perfect inequality (only one item recommended).
        unequal_dist = {'item1': 4, 'item2': 0, 'item3': 0, 'item4': 0}
        gini_unequal = AdvancedMetrics.calculate_gini_coefficient(unequal_dist)
        self.assertAlmostEqual(gini_unequal, 1.0 - (1.0 / 4.0))

        # Case 3: Mixed distribution.
        mixed_dist = {'item1': 10, 'item2': 5, 'item3': 1, 'item4': 1}
        gini_mixed = AdvancedMetrics.calculate_gini_coefficient(mixed_dist)
        self.assertTrue(0 < gini_mixed < 1)
        
        # Tests edge case with empty input.
        self.assertEqual(AdvancedMetrics.calculate_gini_coefficient({}), 0.0)

    def test_calculate_serendipity(self):
        """Tests the serendipity calculation (unexpected but relevant) for single and multiple users."""
        # --- SUB-TEST 1: Single user case ---
        recommendations_single = [['a', 'b', 'c']]
        # 'a' is expected. 'b' and 'c' are unexpected.
        expected_items_single = [{'a'}]
        # 'b' and 'c' are relevant.
        relevant_items_single = [{'b', 'c'}]
        # Serendipitous items are 'b' and 'c' (relevant and not expected). Score = 2/3
        serendipity = AdvancedMetrics.calculate_serendipity(recommendations_single, expected_items_single, relevant_items_single)
        self.assertAlmostEqual(serendipity, 2 / 3, msg="Single user serendipity calculation failed")

        # --- SUB-TEST 2: Multi-user case with varied results ---
        recommendations_multi = [
            ['i1', 'i2', 'i3'],  # User 1: 2 serendipitous items / 3 recs = 0.666
            ['i4', 'i5', 'i6'],  # User 2: 1 serendipitous item / 3 recs = 0.333
            ['i7', 'i8'],        # User 3: 0 serendipitous items / 2 recs = 0.0
            []                   # User 4: Empty recommendations, score = 0.0
        ]
        expected_items_multi = [
            {'i10'},             # User 1: All recommendations are unexpected
            {'i4', 'i11'},       # User 2: 'i4' is expected
            {'i7', 'i12'},       # User 3: 'i7' is expected
            {'i13'}              # User 4: (no recommendations)
        ]
        relevant_items_multi = [
            {'i1', 'i2'},        # User 1: 'i1', 'i2' are relevant and unexpected
            {'i4', 'i5'},        # User 2: 'i5' is relevant and unexpected, 'i4' is relevant and expected
            {'i7', 'i14'},       # User 3: 'i7' is relevant and expected, 'i8' is not relevant
            {'i15'}              # User 4: (no recommendations)
        ]

        # Expected average serendipity: ( (2/3) + (1/3) + (0/2) + 0 ) / 4 = (1.0) / 4 = 0.25
        avg_serendipity = AdvancedMetrics.calculate_serendipity(
            recommendations_multi, expected_items_multi, relevant_items_multi
        )
        self.assertAlmostEqual(avg_serendipity, 0.25, msg="Multi-user average serendipity calculation failed")
        
        # --- SUB-TEST 3: Case with no serendipitous items ---
        no_serendipity = AdvancedMetrics.calculate_serendipity([['a']], [{'a'}], [{'a'}])
        self.assertAlmostEqual(no_serendipity, 0.0, msg="Zero serendipity case failed")


class TestFairnessMetrics(unittest.TestCase):
    """Test cases for the FairnessMetrics calculation methods."""

    def test_calculate_demographic_parity(self):
        """Tests demographic parity calculation across user groups."""
        # Setup: Recommendations for 4 users belonging to two groups.
        recommendations = {
            'u1': ['i1', 'i2'], 'u2': ['i3', 'i4'],  # Group A
            'u3': ['i5', 'i6'], 'u4': ['i7', 'i8']   # Group B
        }
        user_demographics = {
            'u1': {'gender': 'A'}, 'u2': {'gender': 'A'},
            'u3': {'gender': 'B'}, 'u4': {'gender': 'B'}
        }
        
        # Verifies that both groups have the same recommendation rate (perfect parity).
        rates = FairnessMetrics.calculate_demographic_parity(recommendations, user_demographics, 'gender')
        self.assertAlmostEqual(rates['A'], 1.0)
        self.assertAlmostEqual(rates['B'], 1.0)

        # Setup: Unequal recommendation distribution.
        recommendations_unequal = {
            'u1': ['i1', 'i2'], 'u2': ['i1', 'i2'],  # Group A: 2 unique items
            'u3': ['i5', 'i6'], 'u4': ['i7', 'i8']   # Group B: 4 unique items
        }
        rates_unequal = FairnessMetrics.calculate_demographic_parity(recommendations_unequal, user_demographics, 'gender')
        self.assertAlmostEqual(rates_unequal['A'], 2 / 4)
        self.assertAlmostEqual(rates_unequal['B'], 4 / 4)

    def test_calculate_provider_fairness(self):
        """Tests fairness metrics for item providers."""
        # Setup: 2 users' recommendations, items from 3 providers.
        recommendations = [
            ['item_a1', 'item_b1', 'item_a2'],  # User 1
            ['item_c1', 'item_b2', 'item_a3']   # User 2
        ]
        item_providers = {
            'item_a1': 'provider_A', 'item_a2': 'provider_A', 'item_a3': 'provider_A',
            'item_b1': 'provider_B', 'item_b2': 'provider_B',
            'item_c1': 'provider_C'
        }
        
        # Expected exposure: A=3/6, B=2/6, C=1/6
        fairness_results = FairnessMetrics.calculate_provider_fairness(recommendations, item_providers)
        exposure = fairness_results['provider_exposure']
        
        # Verifies the exposure rates for each provider.
        self.assertAlmostEqual(exposure['provider_A'], 3 / 6)
        self.assertAlmostEqual(exposure['provider_B'], 2 / 6)
        self.assertAlmostEqual(exposure['provider_C'], 1 / 6)
        # Verifies that provider Gini is calculated and is greater than 0, indicating some inequality.
        self.assertTrue(fairness_results['provider_gini'] > 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)