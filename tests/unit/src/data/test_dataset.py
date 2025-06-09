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
        Image.new('RGB', (100, 100), color='yellow').save(self.image_dir / "item4.jpg")

        # Creates dummy dataframes for item metadata and user interactions.
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item_nonexistent', 'item4'],
            'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4', 'Title 5'],
            'tag': ['A', 'B', 'A', 'C', np.nan], # Added NaN to test cleaning
            'description': ['Desc 1', 'Desc 2', 'Desc 3', 'Desc 4', 'Desc 5'],
            'view_number': [100, 200, 50, 0, 150],
            'comment_number': [10, 20, 5, 0, 15]
        })

        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u4'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item2', 'item4'] # Added interaction for item4
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
        self.assertEqual(dataset.n_users, 4)
        self.assertEqual(dataset.n_items, 5)
        self.assertTrue(all(dataset.all_samples['label'] == 1))

    def test_getitem_structure_and_types(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Test the structure, keys, and tensor types of a single item from the dataset."""
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            numerical_feat_cols=self.numerical_cols,
            vision_model_name='clip',
            categorical_feat_cols=['tag'] # Activate tag processing
        )
        sample = dataset[0]

        # Added 'tag_idx' to the list of expected keys.
        expected_keys = [
            'user_idx', 'item_idx', 'label', 'image', 
            'text_input_ids', 'text_attention_mask', 
            'numerical_features', 'clip_text_input_ids', 'clip_text_attention_mask',
            'tag_idx'
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

    # This is the new test method dedicated to the tag feature.
    def test_categorical_tag_encoding(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Tests if the dataset correctly handles the categorical 'tag' feature."""
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            create_negative_samples=False,
            categorical_feat_cols=['tag']  # Activate the logic
        )

        # 1. Check that the encoder and tag count are correct
        self.assertTrue(hasattr(dataset, 'tag_encoder'))
        # Expecting 'A', 'B', 'C', and 'unknown'
        self.assertEqual(dataset.n_tags, 4)
        self.assertIn('unknown', dataset.tag_encoder.classes_)

        # 2. Check __getitem__ for an item with a standard tag ('item1', tag 'A')
        sample_item1 = dataset[0] # Corresponds to interaction ('u1', 'item1')
        self.assertIn('tag_idx', sample_item1)
        expected_idx_A = dataset.tag_encoder.transform(['A'])[0]
        self.assertEqual(sample_item1['tag_idx'].item(), expected_idx_A)

        # 3. Check __getitem__ for an item with a NaN tag ('item4', tag np.nan)
        # This is the last interaction in the dataframe.
        nan_sample_index = len(self.interactions_df) - 1
        sample_item4 = dataset[nan_sample_index]
        self.assertIn('tag_idx', sample_item4)
        expected_idx_unknown = dataset.tag_encoder.transform(['unknown'])[0]
        self.assertEqual(sample_item4['tag_idx'].item(), expected_idx_unknown)
