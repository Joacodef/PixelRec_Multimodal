# tests/unit/src/data/test_image_augmentation.py
"""
Unit tests for image augmentation functionality in the multimodal dataset.
"""
import unittest
import torch
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from torchvision import transforms
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.data.dataset import MultimodalDataset
from src.config import ImageAugmentationConfig, TextAugmentationConfig


class TestImageAugmentation(unittest.TestCase):
    """Test cases for image augmentation in the dataset."""
    
    def setUp(self):
        """Set up test environment with temporary directories and sample data."""
        self.test_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.test_dir) / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        # Create sample images
        self.item_ids = ['item_1', 'item_2', 'item_3']
        self.image_size = (224, 224)
        
        for item_id in self.item_ids:
            # Create a distinctive image for each item
            img = Image.new('RGB', self.image_size)
            pixels = img.load()
            # Create a gradient pattern to make augmentation effects visible
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    pixels[i, j] = (
                        int(255 * i / self.image_size[0]),  # Red gradient
                        int(255 * j / self.image_size[1]),  # Green gradient
                        128  # Fixed blue
                    )
            img.save(self.image_dir / f"{item_id}.jpg")
        
        # Create item info dataframe
        self.item_info_df = pd.DataFrame({
            'item_id': self.item_ids,
            'title': [f'Title {i}' for i in range(len(self.item_ids))],
            'tag': [f'Tag {i}' for i in range(len(self.item_ids))],
            'description': [f'Description {i}' for i in range(len(self.item_ids))],
            'view_number': [100, 200, 300],
            'comment_number': [10, 20, 30],
        })
        
        # Create interactions dataframe
        self.interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_1', 'user_2'],
            'item_id': self.item_ids,
        })
    
    def tearDown(self):
        """Clean up temporary directory."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_augmentation_disabled_by_default(self):
        """Test that augmentation is disabled by default."""
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            is_train_mode=True
        )
        
        # When no augmentation config is passed, the processor should not apply it.
        # Note: We now check the effect, not the internal attribute.
        item_features1 = dataset._get_item_features('item_1')
        item_features2 = dataset._get_item_features('item_1')
        
        # Images should be identical when augmentation is disabled
        torch.testing.assert_close(item_features1['image'], item_features2['image'])
    
    def test_augmentation_enabled_training_mode(self):
        """Test that augmentation works when enabled in training mode."""
        aug_config = ImageAugmentationConfig(
            enabled=True,
            brightness=0.5,
            contrast=0.5,
            horizontal_flip=True,
            random_crop=True,
            crop_scale=[0.8, 1.0]
        )
        
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Load the same image multiple times
        images = [dataset._get_item_features('item_1')['image'] for _ in range(5)]
        
        # Check that at least some images are different due to augmentation
        differences = [torch.abs(images[0] - img).sum().item() for img in images[1:]]
        
        self.assertTrue(any(diff > 0.01 for diff in differences),
                       "Augmentation should produce different images")
    
    def test_augmentation_disabled_validation_mode(self):
        """Test that augmentation is disabled in validation mode even when config is enabled."""
        aug_config = ImageAugmentationConfig(enabled=True, brightness=0.5)
        
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            is_train_mode=False,  # Validation mode
            image_augmentation_config=aug_config
        )
        
        # In validation mode, augmentations should not be applied.
        item_features1 = dataset._get_item_features('item_1')
        item_features2 = dataset._get_item_features('item_1')
        
        # Images should be identical
        torch.testing.assert_close(item_features1['image'], item_features2['image'])

    
    def test_augmentation_with_missing_image(self):
        """Test that augmentation handles missing images gracefully."""
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            is_train_mode=True
        )
        
        # An item not in the item_info DataFrame should return placeholder features.
        features = dataset._get_item_features('non_existent_item')
        self.assertIsNotNone(features)
        self.assertEqual(features['image'].sum(), 0) # Placeholder is zeros
        
        # An item in the DataFrame but with no image file should also get a placeholder.
        self.item_info_df = pd.concat([
            self.item_info_df,
            pd.DataFrame([{'item_id': 'item_no_image', 'title': 'No Image', 'description': ''}])
        ], ignore_index=True)
        dataset.item_info = self.item_info_df.set_index('item_id')
        
        features = dataset._get_item_features('item_no_image')
        self.assertIsNotNone(features)
        self.assertEqual(features['image'].shape, torch.Size([3, 224, 224]))
        self.assertEqual(features['image'].sum(), 0)

    
    def test_augmentation_reproducibility_with_seed(self):
        """Test that augmentation can be made reproducible with random seed."""
        aug_config = ImageAugmentationConfig(
            enabled=True,
            brightness=0.5,
            horizontal_flip=True
        )
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Set same random seed and get images
        torch.manual_seed(42)
        img1 = dataset._get_item_features('item_1')['image']
        
        torch.manual_seed(42)
        img2 = dataset._get_item_features('item_1')['image']
        
        # Should be identical with same seed
        torch.testing.assert_close(img1, img2)
        
        # Different seeds should produce different results
        torch.manual_seed(43)
        img3 = dataset._get_item_features('item_1')['image']
        
        diff = torch.abs(img1 - img3).sum().item()
        self.assertGreater(diff, 0.01, "Different seeds should produce different augmentations")
    

    def test_integration_with_dataloader(self):
        """Test that augmented dataset works properly with DataLoader."""
        aug_config = ImageAugmentationConfig(
            enabled=True,
            brightness=0.3,
            horizontal_flip=True
        )
        
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            create_negative_samples=True,
            cache_features=False,
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Create a DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )
        
        # Get a few batches and ensure they work
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            # Check batch structure
            self.assertIn('image', batch)
            self.assertIn('user_idx', batch)
            self.assertIn('item_idx', batch)
            self.assertIn('label', batch)
            
            # Check shapes
            self.assertEqual(batch['image'].dim(), 4)  # [batch, channels, height, width]
            self.assertEqual(batch['image'].shape[1:], torch.Size([3, 224, 224]))
            
            if batch_count >= 2:  # Just test a couple batches
                break
        
        self.assertGreater(batch_count, 0, "Should successfully iterate through batches")


if __name__ == '__main__':
    unittest.main()