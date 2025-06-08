import unittest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from src.data.processors.feature_cache_processor import FeatureCacheProcessor

class TestFeatureCacheProcessor(unittest.TestCase):
    """Unit tests for the FeatureCacheProcessor."""

    def setUp(self):
        """Set up mocks and test data."""
        self.mock_cache_config = MagicMock()
        self.mock_cache_config.cache_directory = "dummy/cache"
        self.mock_cache_config.max_memory_items = 100
        self.mock_cache_config.strategy = "LRU"

        # Mock the dataset instance that will be passed
        self.mock_dataset = MagicMock()
        
        # A dummy item_info dataframe
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3']
        })

    @patch('src.data.processors.feature_cache_processor.ProcessedFeatureCache')
    def test_precompute_features_calls_cache_set(self, mock_processed_cache):
        """Tests that precomputation calls the cache's `set` method for each item."""
        # Arrange
        mock_cache_instance = mock_processed_cache.return_value
        # Ensure get() returns None so all items are processed
        mock_cache_instance.get.return_value = None

        # This mock will be used inside _extract_item_features
        self.mock_dataset._get_item_text.return_value = "dummy text"
        # Mock other dataset methods as needed if they were complex
        
        processor = FeatureCacheProcessor(self.mock_cache_config)
        
        # Patch the internal helper to return a predictable feature dict
        with patch.object(processor, '_extract_item_features', return_value={'feature': 'data'}) as mock_extract:
            # Act
            processor.precompute_features(self.item_info_df, self.mock_dataset)
        
            # Assert
            # Check that set was called for each item in the dataframe
            self.assertEqual(mock_cache_instance.set.call_count, len(self.item_info_df))
            mock_cache_instance.set.assert_any_call('item1', {'feature': 'data'})
            mock_cache_instance.set.assert_any_call('item2', {'feature': 'data'})
            mock_cache_instance.set.assert_any_call('item3', {'feature': 'data'})

    @patch('src.data.processors.feature_cache_processor.ProcessedFeatureCache')
    def test_precompute_features_skips_existing_items(self, mock_processed_cache):
        """Tests that items already in the cache are skipped."""
        # Arrange
        mock_cache_instance = mock_processed_cache.return_value
        # Make `get` return features for 'item2', but not others
        mock_cache_instance.get.side_effect = lambda item_id: {'feature': 'cached_data'} if item_id == 'item2' else None
        
        processor = FeatureCacheProcessor(self.mock_cache_config)
        
        with patch.object(processor, '_extract_item_features', return_value={'feature': 'new_data'}):
            # Act
            processor.precompute_features(self.item_info_df, self.mock_dataset)
        
            # Assert
            # `set` should only be called for item1 and item3
            self.assertEqual(mock_cache_instance.set.call_count, 2)
            mock_cache_instance.set.assert_any_call('item1', {'feature': 'new_data'})
            mock_cache_instance.set.assert_any_call('item3', {'feature': 'new_data'})
            
    @patch('src.data.processors.feature_cache_processor.ProcessedFeatureCache')
    def test_force_recompute_overwrites_existing(self, mock_processed_cache):
        """Tests that force_recompute=True processes all items regardless of cache state."""
        # Arrange
        mock_cache_instance = mock_processed_cache.return_value
        # `get` returns features for 'item2'
        mock_cache_instance.get.side_effect = lambda item_id: {'feature': 'cached_data'} if item_id == 'item2' else None
        
        processor = FeatureCacheProcessor(self.mock_cache_config)

        with patch.object(processor, '_extract_item_features', return_value={'feature': 'recomputed_data'}):
            # Act
            processor.precompute_features(self.item_info_df, self.mock_dataset, force_recompute=True)
        
            # Assert
            # `set` should be called for all items, including the one that was "in the cache"
            self.assertEqual(mock_cache_instance.set.call_count, 3)
            mock_cache_instance.set.assert_any_call('item2', {'feature': 'recomputed_data'})

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)