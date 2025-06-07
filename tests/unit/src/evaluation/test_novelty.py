# tests/unit/src/evaluation/test_novelty.py
"""
Unit tests for the NoveltyMetrics and DiversityCalculator classes.
"""
import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.evaluation.novelty import NoveltyMetrics, DiversityCalculator

class TestNoveltyMetrics(unittest.TestCase):
    """Test cases for the NoveltyMetrics class."""

    def setUp(self):
        """Set up sample data and instantiate NoveltyMetrics for all tests."""
        # Creates sample item popularity data.
        self.item_popularity = {
            'i_pop_1': 100,  # Very popular
            'i_pop_2': 50,
            'i_tail_1': 5,   # Long-tail item
            'i_tail_2': 2,   # Very long-tail item
            'i_user1_hist': 20 # Item in user1's history
        }
        # Creates sample user interaction history.
        self.user_history = [
            ('u1', 'i_user1_hist'), ('u1', 'i_pop_1'),
            ('u2', 'i_pop_1'), ('u2', 'i_pop_2'),
            ('u3', 'i_tail_1'), ('u3', 'i_pop_1')
        ]
        # Instantiates the class under test with the sample data.
        self.novelty_calc = NoveltyMetrics(self.item_popularity, self.user_history)

    def test_initialization(self):
        """Tests that the calculator initializes its internal stats correctly."""
        self.assertEqual(self.novelty_calc.total_interactions, 177)
        self.assertEqual(self.novelty_calc.n_users, 3)
        self.assertEqual(self.novelty_calc.item_user_counts['i_pop_1'], 3)
        # Tests popularity ranking (most popular item has rank 0).
        self.assertEqual(self.novelty_calc.popularity_ranks['i_pop_1'], 0)
        self.assertEqual(self.novelty_calc.popularity_ranks['i_tail_2'], 4)

    def test_calculate_self_information(self):
        """Tests the average self-information (surprisal) metric."""
        # A popular item should have low surprisal, a rare item should have high surprisal.
        recs = ['i_pop_1', 'i_tail_2']
        p_pop_1 = 100 / 177
        p_tail_2 = 2 / 177
        expected_si = (-np.log2(p_pop_1) + -np.log2(p_tail_2)) / 2
        self.assertAlmostEqual(self.novelty_calc.calculate_self_information(recs), expected_si)

    def test_calculate_iif(self):
        """Tests the average Inverse Item Frequency metric."""
        # An item seen by many users ('i_pop_1') should have low IIF.
        recs = ['i_pop_1', 'i_tail_1'] # i_pop_1 seen by 3 users, i_tail_1 by 1 user
        iif_pop_1 = np.log(3 / 3) # 0
        iif_tail_1 = np.log(3 / 1) # ~1.098
        expected_iif = (iif_pop_1 + iif_tail_1) / 2
        self.assertAlmostEqual(self.novelty_calc.calculate_iif(recs), expected_iif)

    def test_calculate_coverage(self):
        """Tests catalog coverage."""
        # The recommended items cover 3 out of 5 total catalog items.
        recs = ['i_pop_1', 'i_pop_2', 'i_tail_1', 'i_pop_1'] # 3 unique items
        expected_coverage = 3 / 5
        self.assertAlmostEqual(self.novelty_calc.calculate_coverage(recs), expected_coverage)

    def test_calculate_long_tail_percentage(self):
        """Tests the percentage of recommendations from the long tail."""
        # Total items = 5. Tail threshold is at 20% of items, so rank >= 1.
        # Ranks: i_pop_1(0), i_pop_2(1), i_user1_hist(2), i_tail_1(3), i_tail_2(4).
        # Tail items are those with rank >= 1.
        recs = ['i_pop_1', 'i_pop_2', 'i_tail_1', 'i_tail_2']
        # 3 out of 4 recommendations are in the long tail.
        self.assertAlmostEqual(self.novelty_calc.calculate_long_tail_percentage(recs), 3 / 4)

    def test_calculate_diversity(self):
        """Tests intra-list diversity (percentage of unique items)."""
        recs_diverse = ['a', 'b', 'c']
        recs_duplicate = ['a', 'b', 'a']
        self.assertAlmostEqual(self.novelty_calc.calculate_diversity(recs_diverse), 1.0)
        self.assertAlmostEqual(self.novelty_calc.calculate_diversity(recs_duplicate), 2 / 3)

    def test_calculate_personalized_novelty(self):
        """Tests novelty relative to a specific user's history."""
        # User 'u1' has seen 'i_user1_hist' and 'i_pop_1'.
        recs = ['i_pop_1', 'i_pop_2', 'i_tail_1']
        # 'i_pop_2' and 'i_tail_1' are novel to user 'u1'. 2 out of 3 are novel.
        novelty = self.novelty_calc.calculate_personalized_novelty(recs, 'u1')
        self.assertAlmostEqual(novelty, 2 / 3)

class TestDiversityCalculator(unittest.TestCase):
    """Test cases for the DiversityCalculator class."""

    def setUp(self):
        """Set up sample item embeddings and instantiate the calculator."""
        # Creates sample item embeddings as numpy arrays.
        self.item_embeddings = {
            'item_a': np.array([1.0, 0.0]),   # Unit vector
            'item_b': np.array([0.0, 1.0]),   # Orthogonal to A
            'item_c': np.array([1.0, 0.0]),   # Identical to A
            'item_d': np.array([-1.0, 0.0]),  # Opposite to A
        }
        # Instantiates the class under test.
        self.diversity_calc = DiversityCalculator(self.item_embeddings)

    def test_pairwise_diversity_cosine(self):
        """Tests pairwise diversity using cosine distance."""
        # Case 1: Identical items. Cosine similarity is 1, so distance should be 0.
        self.assertAlmostEqual(self.diversity_calc.calculate_pairwise_diversity(['item_a', 'item_c']), 0.0)

        # Case 2: Orthogonal items. Cosine similarity is 0, so distance is 1.
        self.assertAlmostEqual(self.diversity_calc.calculate_pairwise_diversity(['item_a', 'item_b']), 1.0)

        # Case 3: Opposite items. Cosine similarity is -1, so distance is 2.
        self.assertAlmostEqual(self.diversity_calc.calculate_pairwise_diversity(['item_a', 'item_d']), 2.0)

    def test_pairwise_diversity_euclidean(self):
        """Tests pairwise diversity using Euclidean distance."""
        # Euclidean distance between (1,0) and (0,1) is sqrt( (1-0)^2 + (0-1)^2 ) = sqrt(2).
        dist = self.diversity_calc.calculate_pairwise_diversity(['item_a', 'item_b'], metric='euclidean')
        self.assertAlmostEqual(dist, np.sqrt(2))

    def test_coverage_diversity(self):
        """Tests coverage diversity across multiple users' recommendations."""
        # Setup: Two users, with some overlap in recommendations.
        recommendations_per_user = {
            'u1': ['item_a', 'item_b', 'item_c'],
            'u2': ['item_b', 'item_d', 'item_e'] # 'item_e' is not in embeddings, should be ignored by diversity calc but not coverage.
        }
        # Unique items: {'a', 'b', 'c', 'd', 'e'} -> 5 unique items.
        # Total recommendations: 3 + 3 = 6.
        # Expected coverage = 5 / 6.
        coverage = self.diversity_calc.calculate_coverage_diversity(recommendations_per_user)
        self.assertAlmostEqual(coverage, 5 / 6)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)