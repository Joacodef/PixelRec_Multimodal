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
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=True
        )
        
        # Check that image augmentation is None when not configured
        self.assertIsNone(dataset.image_augmentation)
        
        # Load an image and check it's not augmented
        item_features1 = dataset._process_item_features('item_1')
        item_features2 = dataset._process_item_features('item_1')
        
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
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Check that augmentation pipeline is created
        self.assertIsNotNone(dataset.image_augmentation)
        
        # Load the same image multiple times
        images = []
        for _ in range(5):
            item_features = dataset._process_item_features('item_1')
            images.append(item_features['image'])
        
        # Check that at least some images are different due to augmentation
        differences = []
        for i in range(1, len(images)):
            diff = torch.abs(images[0] - images[i]).sum().item()
            differences.append(diff)
        
        # At least one image should be different
        self.assertTrue(any(diff > 0.01 for diff in differences),
                       "Augmentation should produce different images")
    
    def test_augmentation_disabled_validation_mode(self):
        """Test that augmentation is disabled in validation mode even when config is enabled."""
        aug_config = ImageAugmentationConfig(
            enabled=True,
            brightness=0.5,
            contrast=0.5,
            horizontal_flip=True
        )
        
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=False,  # Validation mode
            image_augmentation_config=aug_config
        )
        
        # Augmentation should be None in validation mode
        self.assertIsNone(dataset.image_augmentation)
        
        # Images should be identical
        item_features1 = dataset._process_item_features('item_1')
        item_features2 = dataset._process_item_features('item_1')
        torch.testing.assert_close(item_features1['image'], item_features2['image'])
    
    def test_color_augmentations(self):
        """Test color-based augmentations (brightness, contrast, saturation, hue)."""
        aug_config = ImageAugmentationConfig(
            enabled=True,
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.2,
            # Disable geometric augmentations for this test
            random_crop=False,
            horizontal_flip=False,
            rotation_degrees=0,
            gaussian_blur=False
        )
        
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Get multiple versions of the same image
        original_image = dataset._load_and_process_image('item_1')
        augmented_images = []
        for _ in range(10):
            aug_image = dataset._load_and_process_image('item_1')
            augmented_images.append(aug_image)
        
        # Check that color values vary
        color_variations = []
        for aug_img in augmented_images:
            # Calculate mean color difference from original
            diff = torch.abs(aug_img - original_image).mean().item()
            color_variations.append(diff)
        
        # Should have some variation in colors
        self.assertTrue(max(color_variations) > 0.01,
                       "Color augmentation should produce variations")
    
    def test_geometric_augmentations(self):
        """Test geometric augmentations (crop, flip, rotation)."""
        aug_config = ImageAugmentationConfig(
            enabled=True,
            # Disable color augmentations
            brightness=0,
            contrast=0,
            saturation=0,
            hue=0,
            # Enable geometric augmentations
            random_crop=True,
            crop_scale=[0.7, 0.9],
            horizontal_flip=True,
            rotation_degrees=30,
            gaussian_blur=False
        )
        
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Collect multiple augmented versions
        images = []
        for _ in range(10):
            img = dataset._load_and_process_image('item_1')
            images.append(img)
        
        # Check for variations indicating geometric changes
        variations = []
        for i in range(1, len(images)):
            # Focus on edge regions where geometric changes are most apparent
            edge_diff = torch.abs(images[0][:, :10, :] - images[i][:, :10, :]).sum()
            edge_diff += torch.abs(images[0][:, -10:, :] - images[i][:, -10:, :]).sum()
            edge_diff += torch.abs(images[0][:, :, :10] - images[i][:, :, :10]).sum()
            edge_diff += torch.abs(images[0][:, :, -10:] - images[i][:, :, -10:]).sum()
            variations.append(edge_diff.item())
        
        # Should have significant variations due to geometric transforms
        self.assertTrue(max(variations) > 10.0,
                       "Geometric augmentation should produce significant variations")
    
    def test_blur_augmentation(self):
        """Test Gaussian blur augmentation."""
        aug_config = ImageAugmentationConfig(
            enabled=True,
            # Disable other augmentations
            brightness=0,
            contrast=0,
            saturation=0,
            hue=0,
            random_crop=False,
            horizontal_flip=False,
            rotation_degrees=0,
            # Enable blur
            gaussian_blur=True,
            blur_kernel_size=[5, 9]
        )
        
        dataset = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Get multiple versions
        images = []
        for _ in range(20):
            img = dataset._load_and_process_image('item_1')
            images.append(img)
        
        # Calculate sharpness metric (high-frequency content)
        sharpness_values = []
        for img in images:
            # Use Laplacian to measure sharpness
            img_gray = img.mean(dim=0)  # Convert to grayscale
            laplacian = torch.nn.functional.conv2d(
                img_gray.unsqueeze(0).unsqueeze(0),
                torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]]).float(),
                padding=1
            )
            sharpness = laplacian.abs().mean().item()
            sharpness_values.append(sharpness)
        
        # Should have variation in sharpness due to random blur application
        sharpness_std = np.std(sharpness_values)
        self.assertGreater(sharpness_std, 0.001,
                          "Blur augmentation should create sharpness variations")
    
    def test_augmentation_with_missing_image(self):
        """Test that augmentation handles missing images gracefully."""
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
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Try to process a non-existent item
        # Should create a grey placeholder image and process it without errors
        features = dataset._process_item_features('non_existent_item')
        self.assertIsNone(features)  # Will be None because item not in item_info
        
        # Try with an item that exists in item_info but has no image
        self.item_info_df = pd.concat([
            self.item_info_df,
            pd.DataFrame({
                'item_id': ['item_no_image'],
                'title': ['No Image Item'],
                'tag': ['missing'],
                'description': ['This item has no image'],
                'view_number': [0],
                'comment_number': [0]
            })
        ], ignore_index=True)
        
        dataset.item_info = self.item_info_df.set_index('item_id')
        
        # This should work with a grey placeholder
        features = dataset._process_item_features('item_no_image')
        self.assertIsNotNone(features)
        self.assertEqual(features['image'].shape, torch.Size([3, 224, 224]))
    
    def test_augmentation_reproducibility_with_seed(self):
        """Test that augmentation can be made reproducible with random seed."""
        aug_config = ImageAugmentationConfig(
            enabled=True,
            brightness=0.5,
            contrast=0.5,
            horizontal_flip=True,
            random_crop=True
        )
        
        # Create two datasets with same config
        dataset1 = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        dataset2 = MultimodalDataset(
            interactions_df=self.interactions_df,
            item_info_df=self.item_info_df,
            image_folder=str(self.image_dir),
            vision_model_name='resnet',
            language_model_name='sentence-bert',
            create_negative_samples=False,
            cache_features=False,
            is_train_mode=True,
            image_augmentation_config=aug_config
        )
        
        # Set same random seed and get images
        torch.manual_seed(42)
        img1 = dataset1._load_and_process_image('item_1')
        
        torch.manual_seed(42)
        img2 = dataset2._load_and_process_image('item_1')
        
        # Should be identical with same seed
        torch.testing.assert_close(img1, img2)
        
        # Different seeds should produce different results
        torch.manual_seed(42)
        img3 = dataset1._load_and_process_image('item_1')
        
        torch.manual_seed(43)
        img4 = dataset1._load_and_process_image('item_1')
        
        # Should be different with different seeds
        diff = torch.abs(img3 - img4).sum().item()
        self.assertGreater(diff, 0.01, "Different seeds should produce different augmentations")
    
    def test_augmentation_config_validation(self):
        """Test that augmentation config validates parameters properly."""
        # Test with invalid brightness value
        with self.assertRaises(ValueError):
            aug_config = ImageAugmentationConfig(
                enabled=True,
                brightness=-0.5  # Should be non-negative
            )
            # Add validation in ImageAugmentationConfig __post_init__ if needed
        
        # Test with invalid crop scale
        with self.assertRaises(ValueError):
            aug_config = ImageAugmentationConfig(
                enabled=True,
                crop_scale=[1.2, 0.8]  # min > max, should be invalid
            )
    
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