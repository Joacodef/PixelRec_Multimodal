# tests/unit/src/inference/test_recommender.py
"""
Unit tests for the multimodal recommender inference class.
"""
import unittest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.inference.recommender import Recommender
from sklearn.preprocessing import LabelEncoder

class TestRecommender(unittest.TestCase):
    """Test cases for the inference Recommender class."""

    def setUp(self):
        """Set up mock objects and sample data for testing."""
        # Creates a mock model. The model object itself is made callable to simulate
        # the forward pass (`model(...)`) and return predictable scores.
        self.mock_model = MagicMock(spec=torch.nn.Module)
        def model_callable_side_effect(**kwargs):
            item_indices = kwargs.get('item_idx')
            # The mock logic: returns a higher score for a lower item index (i.e., i1 > i2 > i3)
            # This returns a real torch.Tensor, which correctly supports .squeeze().cpu().tolist()
            scores = 1.0 - item_indices.float() / 10.0
            return scores
        
        # This makes the mock object itself callable, which is what happens in recommender.py
        self.mock_model.side_effect = model_callable_side_effect
        self.mock_model.eval = MagicMock() # Mock the .eval() call

        # Creates dummy interaction and item data.
        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'item_id': ['i1', 'i2', 'i3']
        })
        self.item_info_df = pd.DataFrame({
            'item_id': ['i1', 'i2', 'i3', 'i4', 'i5'],
            'title': ['t1', 't2', 't3', 't4', 't5']
        })
        
        # Creates a mock dataset object.
        self.mock_dataset = MagicMock()
        
        # Fits encoders and attaches them to the mock dataset.
        self.user_encoder = LabelEncoder().fit(self.interactions_df['user_id'])
        self.item_encoder = LabelEncoder().fit(self.item_info_df['item_id'])
        self.mock_dataset.user_encoder = self.user_encoder
        self.mock_dataset.item_encoder = self.item_encoder
        
        # Mocks the dataset's item info dictionary.
        self.mock_dataset.item_info_df_original = self.item_info_df
        
        # Mocks the method for retrieving a user's interaction history.
        user_history = self.interactions_df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.mock_dataset.get_user_history.side_effect = lambda user_id: user_history.get(user_id, set())

        # Mocks the feature processing method to return consistent dummy tensors.
        dummy_features = {
            'image': torch.zeros(3, 224, 224),
            'text_input_ids': torch.zeros(128, dtype=torch.long),
            'text_attention_mask': torch.zeros(128, dtype=torch.long),
            'numerical_features': torch.zeros(5)
        }
        self.mock_dataset._process_item_features.return_value = dummy_features

        # Initializes the Recommender instance with all the mock objects.
        self.recommender = Recommender(
            model=self.mock_model,
            dataset=self.mock_dataset,
            device=torch.device('cpu')
        )

    def test_get_recommendations_basic(self):
        """Tests basic recommendation generation and ranking."""
        recs = self.recommender.get_recommendations('u1', top_k=3, filter_seen=False)
        
        self.assertEqual(len(recs), 3)
        rec_ids = [r[0] for r in recs]
        self.assertEqual(rec_ids, ['i1', 'i2', 'i3'])

    def test_get_recommendations_filter_seen(self):
        """Tests that items already seen by the user are correctly filtered."""
        # u1 has seen i1 and i2.
        recs = self.recommender.get_recommendations('u1', top_k=5, filter_seen=True)
        rec_ids = {r[0] for r in recs}
        
        self.assertNotIn('i1', rec_ids)
        self.assertNotIn('i2', rec_ids)
        self.assertIn('i3', rec_ids)
        self.assertIn('i4', rec_ids)

    def test_get_recommendations_with_candidates(self):
        """Tests recommendation generation from a specific candidate set."""
        candidate_items = ['i1', 'i4', 'i5']
        # Set filter_seen=False to ensure all candidates are considered for ranking.
        recs = self.recommender.get_recommendations(
            'u1',
            top_k=5,
            candidates=candidate_items,
            filter_seen=False
        )
        rec_ids = [r[0] for r in recs]
        
        # Verifies that only items from the candidate set are recommended.
        self.assertEqual(len(rec_ids), 3)
        self.assertTrue(set(rec_ids).issubset(set(candidate_items)))
        # Verifies the ranking is correct based on the mock model (lower index = higher score).
        self.assertEqual(rec_ids, ['i1', 'i4', 'i5'])
        
    def test_get_recommendations_unknown_user(self):
        """Tests that an empty list is returned for a user not in the dataset."""
        recs = self.recommender.get_recommendations('unknown_user', top_k=5)
        self.assertEqual(recs, [])

    def test_get_item_score(self):
        """Tests the scoring of a single user-item pair."""
        score1 = self.recommender.get_item_score('u1', 'i3')
        score2 = self.recommender.get_item_score('u1', 'i4')
        
        self.assertIsInstance(score1, float)
        self.assertGreater(score1, score2)

    def test_feature_caching(self):
        """Tests that item features are cached to avoid redundant processing."""
        self.recommender.clear_cache()
        self.mock_dataset._process_item_features.reset_mock()
        
        features1 = self.recommender._get_item_features('i1')
        self.mock_dataset._process_item_features.assert_called_once_with('i1')
        
        features2 = self.recommender._get_item_features('i1')
        self.mock_dataset._process_item_features.assert_called_once()
        
        self.assertIsNotNone(features1)
        self.assertTrue(torch.equal(features1['image'], features2['image']))
        self.assertEqual(len(self.recommender.feature_cache), 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)