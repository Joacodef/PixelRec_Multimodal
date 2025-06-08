# tests/unit/src/inference/test_baseline_recommenders.py
"""
Unit tests for the baseline recommender systems.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.inference.baseline_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ItemKNNRecommender,
    UserKNNRecommender
)
from sklearn.preprocessing import LabelEncoder

class TestBaselineRecommenders(unittest.TestCase):
    """Test cases for baseline recommendation models."""

    def setUp(self):
        """Set up a mock dataset and sample interactions for testing."""
        # Creates a sample DataFrame of user-item interactions.
        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u3', 'u4', 'u4'],
            'item_id': ['i1', 'i2', 'i3', 'i2', 'i3', 'i4', 'i1', 'i3']
        })
        # Item Popularity from this data:
        # i1: u1, u4 (popularity: 2)
        # i2: u1, u2 (popularity: 2)
        # i3: u1, u2, u4 (popularity: 3) -> most popular
        # i4: u3 (popularity: 1) -> least popular

        # Creates a mock dataset object that mimics the structure of the real MultimodalDataset.
        self.mock_dataset = MagicMock()
        self.mock_dataset.interactions = self.interactions_df
        
        # Fits LabelEncoders to the sample data to simulate the dataset's fitted encoders.
        user_encoder = LabelEncoder().fit(self.interactions_df['user_id'])
        item_encoder = LabelEncoder().fit(self.interactions_df['item_id'])
        
        self.mock_dataset.user_encoder = user_encoder
        self.mock_dataset.item_encoder = item_encoder

    def test_random_recommender(self):
        """Tests the RandomRecommender."""
        # Initializes the recommender with the mock dataset.
        recommender = RandomRecommender(self.mock_dataset)
        
        # Verifies that the recommender returns the correct number of items when not filtering.
        recs = recommender.get_recommendations('u1', top_k=2, filter_seen=False)
        self.assertEqual(len(recs), 2)
        
        # Verifies that the recommended items are from the catalog.
        all_items = set(self.mock_dataset.item_encoder.classes_)
        self.assertTrue(set(r[0] for r in recs).issubset(all_items))
        
        # Tests the filter_seen functionality.
        recs_filtered = recommender.get_recommendations('u1', top_k=4, filter_seen=True)
        u1_history = set(self.interactions_df[self.interactions_df['user_id'] == 'u1']['item_id'])
        
        # There's only one item u1 hasn't seen ('i4'). So filtered recs should have length 1.
        self.assertEqual(len(recs_filtered), 1)
        self.assertEqual(recs_filtered[0][0], 'i4')
        self.assertTrue(set(r[0] for r in recs_filtered).isdisjoint(u1_history))

    def test_popularity_recommender(self):
        """Tests the PopularityRecommender."""
        recommender = PopularityRecommender(self.mock_dataset)
        
        # Verifies that recommendations are sorted by popularity.
        recs = recommender.get_recommendations('u1', top_k=3, filter_seen=False)
        # Expected order: i3 (pop 3), then i1 and i2 (pop 2) in any order.
        rec_ids = [r[0] for r in recs]
        self.assertEqual(rec_ids[0], 'i3')
        self.assertIn('i1', rec_ids)
        self.assertIn('i2', rec_ids)
        
        # Verifies that filtering seen items works correctly. u1 has seen i1, i2, i3.
        recs_filtered = recommender.get_recommendations('u1', top_k=3, filter_seen=True)
        # The only unseen item is i4.
        self.assertEqual(len(recs_filtered), 1)
        self.assertEqual(recs_filtered[0][0], 'i4')
        
        # Tests item scoring.
        score_i3 = recommender.get_item_score('u1', 'i3')
        score_i4 = recommender.get_item_score('u1', 'i4')
        self.assertEqual(score_i3, 1.0, "Most popular item should have score 1.0")
        self.assertLess(score_i4, score_i3)

    def test_item_knn_recommender(self):
        """Tests the ItemKNNRecommender."""
        # In our data, u1 and u2 both liked i2 and i3. So i2 and i3 are similar.
        # u1 also liked i1. So, for u2, i1 should be recommended.
        recommender = ItemKNNRecommender(self.mock_dataset)
        
        # u2 liked i2, i3. i1 is similar to i2 and i3 (u1 liked all three). So i1 should be recommended.
        recs_for_u2 = recommender.get_recommendations('u2', top_k=1, filter_seen=True)
        self.assertIn(recs_for_u2[0][0], ['i1'])

        # u3 liked i4. No other user liked i4, so this user will likely get no recommendations
        # from this model.
        score = recommender.get_item_score('u2', 'i1')
        self.assertGreater(score, 0, "u2 should have a positive score for i1 due to similarity.")
        score_unrelated = recommender.get_item_score('u3', 'i1')
        self.assertEqual(score_unrelated, 0, "u3 has no items similar to i1.")
        
    def test_user_knn_recommender(self):
        """Tests the UserKNNRecommender."""
        # u1 and u2 are similar because they both liked i2 and i3.
        # u1 also liked i1. So, for u2, i1 should be recommended.
        recommender = UserKNNRecommender(self.mock_dataset)
        
        # u2's history is {i2, i3}. A similar user, u1, liked i1.
        recs_for_u2 = recommender.get_recommendations('u2', top_k=1, filter_seen=True)
        self.assertEqual(recs_for_u2[0][0], 'i1')
        
        # u3 is not similar to anyone. Expect empty recommendations.
        recs_for_u3 = recommender.get_recommendations('u3', top_k=1, filter_seen=True)
        self.assertEqual(len(recs_for_u3), 0)

        # Test scoring
        score_u2_i1 = recommender.get_item_score('u2', 'i1')
        self.assertGreater(score_u2_i1, 0, "u2 should have a positive score for i1 due to u1's history.")
        
        score_u3_i1 = recommender.get_item_score('u3', 'i1')
        self.assertEqual(score_u3_i1, 0, "u3 is not similar to any user who liked i1.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)