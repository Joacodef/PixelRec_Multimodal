# tests/unit/src/data/test_dataset.py
"""
Unit tests for the MultimodalDataset class.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import torch
from PIL import Image
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.data.dataset import MultimodalDataset
from src.config import TextAugmentationConfig

# A stable mock for the image processor. Returns a fixed-size numpy array.
mock_image_processor = MagicMock()
mock_image_processor.return_value = {
    'pixel_values': np.full((1, 3, 224, 224), 0.5, dtype=np.float32)
}

# A stable mock for the main text tokenizer (e.g., SentenceBERT). Returns tensors of length 128.
mock_main_tokenizer = MagicMock()
mock_main_tokenizer.return_value = {
    'input_ids': torch.ones((1, 128), dtype=torch.long),
    'attention_mask': torch.ones((1, 128), dtype=torch.long)
}

# A stable mock for the CLIP processor's own tokenizer. Returns tensors of length 77.
mock_clip_sub_tokenizer = MagicMock()
mock_clip_sub_tokenizer.return_value = {
    'input_ids': torch.ones((1, 77), dtype=torch.long),
    'attention_mask': torch.ones((1, 77), dtype=torch.long)
}

# A mock for the entire CLIPProcessor object.
mock_clip_processor = MagicMock()
mock_clip_processor.image_processor = mock_image_processor
mock_clip_processor.tokenizer = mock_clip_sub_tokenizer


@patch('transformers.CLIPProcessor.from_pretrained', return_value=mock_clip_processor)
@patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_main_tokenizer)
@patch('transformers.AutoImageProcessor.from_pretrained', return_value=mock_image_processor)
class TestMultimodalDataset(unittest.TestCase):
    """Test cases for the MultimodalDataset class."""

    def setUp(self):
        """Set up a temporary directory with dummy data for testing."""
        # Creates a temporary directory for test artifacts.
        self.test_dir = Path("test_temp_dataset")
        self.test_dir.mkdir(exist_ok=True)

        self.image_dir = self.test_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        self.cache_dir = self.test_dir / "cache"

        # Creates dummy images for testing.
        Image.new('RGB', (100, 100), color='red').save(self.image_dir / "item1.jpg")
        Image.new('RGB', (128, 128), color='green').save(self.image_dir / "item2.jpg")
        Image.new('RGB', (100, 100), color='blue').save(self.image_dir / "item3.jpg")

        # Creates dummy dataframes for item metadata and user interactions.
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item_nonexistent'], # Add nonexistent item for testing
            'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4'],
            'tag': ['A', 'B', 'A', 'C'],
            'description': ['Desc 1', 'Desc 2', 'Desc 3', 'Desc 4'],
            'view_number': [100, 200, 50, 0],
            'comment_number': [10, 20, 5, 0]
        })

        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u3'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item2']
        })
        
        # Creates and fits a dummy scaler for numerical features.
        self.numerical_cols = ['view_number', 'comment_number']
        self.scaler = StandardScaler().fit(self.item_info_df[self.numerical_cols])

    def tearDown(self):
        """Clean up the temporary directory."""
        # Removes the temporary directory and all its contents after each test.
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization_no_negative_sampling(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Test dataset initialization without negative sampling."""
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            create_negative_samples=False
        )
        self.assertEqual(len(dataset), len(self.interactions_df))
        self.assertEqual(dataset.n_users, 3)
        self.assertEqual(dataset.n_items, 3)
        self.assertTrue(all(dataset.all_samples['label'] == 1))

    def test_getitem_structure_and_types(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Test the structure, keys, and tensor types of a single item from the dataset."""
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            numerical_feat_cols=self.numerical_cols,
            vision_model_name='clip'
        )
        sample = dataset[0]
        expected_keys = [
            'user_idx', 'item_idx', 'label', 'image', 
            'text_input_ids', 'text_attention_mask', 
            'numerical_features', 'clip_text_input_ids', 'clip_text_attention_mask'
        ]
        self.assertCountEqual(sample.keys(), expected_keys)
        self.assertEqual(sample['clip_text_input_ids'].shape, (77,))

    def test_feature_caching(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Test that item features are cached correctly after first access."""
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            cache_features=True,
            cache_dir=str(self.cache_dir)
        )
        
        item_id_to_test = dataset.all_samples.iloc[0]['item_id']
        self.assertIsNone(dataset.feature_cache.get(item_id_to_test))

        _ = dataset[0]

        cached_data = dataset.feature_cache.get(item_id_to_test)
        self.assertIsNotNone(cached_data)
        self.assertIn('image', cached_data)

        with patch.object(dataset, '_process_item_features', wraps=dataset._process_item_features) as mocked_method:
            _ = dataset[0]
            mocked_method.assert_not_called()

    def test_missing_image_handling(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Test that the dataset handles missing image files gracefully."""
        # Fix: Use pd.concat instead of the removed .append method.
        new_interaction = pd.DataFrame([{'user_id': 'u4', 'item_id': 'item_nonexistent'}])
        missing_image_interactions = pd.concat([self.interactions_df, new_interaction], ignore_index=True)
        
        dataset = MultimodalDataset(
            interactions_df=missing_image_interactions,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            create_negative_samples=False
        )
        
        missing_idx = dataset.all_samples[dataset.all_samples['item_id'] == 'item_nonexistent'].index[0]
        
        # This will now pass because the TypeError is fixed, allowing _process_item_features to run.
        # It will call _load_and_process_image, which creates a grey PIL image for the missing file.
        # This PIL image is then passed to our mock image processor.
        sample = dataset[missing_idx]
        
        # The mock processor always returns a tensor of 0.5, so the mean will be 0.5.
        self.assertEqual(sample['image'].shape, (3, 224, 224))
        self.assertAlmostEqual(sample['image'].mean().item(), 0.5, delta=0.01)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)