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

        self.assertEqual(dataset.n_items, 4)
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
        # Use pd.concat instead of the removed .append method.
        new_interaction = pd.DataFrame([{'user_id': 'u4', 'item_id': 'item_nonexistent'}])
        missing_image_interactions = pd.concat([self.interactions_df, new_interaction], ignore_index=True)
        
        dataset = MultimodalDataset(
            interactions_df=missing_image_interactions,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            create_negative_samples=False
        )
        
        missing_idx = dataset.all_samples[dataset.all_samples['item_id'] == 'item_nonexistent'].index[0]
        
        sample = dataset[missing_idx]
        
        self.assertEqual(sample['image'].shape, (3, 224, 224))
        self.assertAlmostEqual(sample['image'].mean().item(), 0.5, delta=0.01)

    def test_negative_sampling_logic(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Tests that negative sampling generates the correct number of distinct negative samples."""
        interactions = pd.DataFrame({'user_id': ['u1', 'u1'], 'item_id': ['item1', 'item2']})
        # Define the full item catalog for the encoder
        items = pd.DataFrame({'item_id': [f'item{i}' for i in range(1, 6)]})

        dataset = MultimodalDataset(
            interactions_df=interactions,
            item_info_df=items,
            image_folder=str(self.image_dir),
            create_negative_samples=True,
            negative_sampling_ratio=1.0 # One negative sample per positive one
        )
        
        self.assertEqual(len(dataset.all_samples), 4)
        
        positive_samples = dataset.all_samples[dataset.all_samples['label'] == 1]
        negative_samples = dataset.all_samples[dataset.all_samples['label'] == 0]
        
        self.assertEqual(len(positive_samples), 2)
        self.assertEqual(len(negative_samples), 2)
        
        user_positive_items = set(interactions['item_id'])
        for item in negative_samples['item_id']:
            self.assertNotIn(item, user_positive_items)

    def test_numerical_feature_scaling(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Tests that the numerical scaler is applied correctly to the features."""
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            numerical_feat_cols=self.numerical_cols,
            numerical_normalization_method='standardization',
            numerical_scaler=self.scaler,
            is_train_mode=True
        )
        
        sample = dataset[0]
        numerical_features = sample['numerical_features']
        
        original_values = self.item_info_df[self.item_info_df['item_id'] == 'item1'][self.numerical_cols].values
        self.assertFalse(torch.allclose(torch.tensor(original_values, dtype=torch.float32), numerical_features))

    def test_text_augmentation(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """Tests that text augmentation is applied during training mode."""
        aug_config = TextAugmentationConfig(enabled=True, augmentation_type='random_delete', delete_prob=1.0)
        
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            is_train_mode=True,
            text_augmentation_config=aug_config
        )

        with patch.object(dataset, 'tokenizer', wraps=dataset.tokenizer) as mocked_tokenizer:
            mocked_tokenizer.return_value = {
                'input_ids': torch.ones((1, 128), dtype=torch.long),
                'attention_mask': torch.ones((1, 128), dtype=torch.long)
            }
            _ = dataset[0]
            
            processed_text = mocked_tokenizer.call_args[0][0]
            self.assertEqual(processed_text, "")

    def test_data_integrity_and_missing_values(self, mock_img_proc, mock_auto_tok, mock_clip_proc):
        """
        This test checks for multiple common data issues at once:
        1. An interaction exists for an item_id ('item_missing_info') that is NOT in the item_info dataframe.
        2. An item ('item_missing_text') is missing its 'title' (will be NaN).
        3. An item ('item_nan_numeric') has a NaN value in a numerical column.
        """
        # 1. Create more complex data with known issues
        faulty_item_info = pd.DataFrame({
            'item_id': ['item1', 'item_missing_text', 'item_nan_numeric'],
            'title': ['Good Title', np.nan, 'NaN Numeric Title'], # Item with missing title
            'tag': ['A', 'B', 'C'],
            'description': ['Desc 1', 'Desc 2', 'Desc 3'],
            'view_number': [100, 200, np.nan], # Item with NaN in numerical feature
            'comment_number': [10, 20, 5]
        })

        faulty_interactions = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'item_id': [
                'item1',
                'item_missing_text',
                'item_nan_numeric',
                'item_missing_info' # This item does not exist in faulty_item_info
            ]
        })

        # 2. Initialize the dataset
        dataset = MultimodalDataset(
            interactions_df=faulty_interactions,
            item_info_df=faulty_item_info,
            image_folder=str(self.image_dir),
            numerical_feat_cols=['view_number', 'comment_number'],
            create_negative_samples=False
        )

        # 3. Perform assertions
        # The dataset should drop the interaction for 'item_missing_info', resulting in 3 valid samples.
        self.assertEqual(len(dataset), 3)

        # Test the item with missing text ('item_missing_text')
        # The dataset should not crash and should process the available text.
        idx_missing_text = dataset.all_samples[dataset.all_samples['item_id'] == 'item_missing_text'].index[0]
        sample_missing_text = dataset[idx_missing_text]
        self.assertIn('text_input_ids', sample_missing_text) # Should still produce text tensors

        # Test the item with a NaN numerical feature ('item_nan_numeric')
        # The dataset's _process_item_features should handle np.nan_to_num.
        idx_nan_numeric = dataset.all_samples[dataset.all_samples['item_id'] == 'item_nan_numeric'].index[0]
        sample_nan_numeric = dataset[idx_nan_numeric]
        # Check that the numerical features tensor does not contain NaN
        self.assertFalse(torch.isnan(sample_nan_numeric['numerical_features']).any())
        
        self.assertNotIn('item_missing_info', dataset.all_samples['item_id'].values)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)