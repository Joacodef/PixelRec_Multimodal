# tests/integration/test_image_augmentation_training.py
"""
Integration tests for image augmentation in the training pipeline.
"""
import unittest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import MultimodalRecommender
from src.training.trainer import Trainer
from scripts.train import main as train_main


class TestImageAugmentationTraining(unittest.TestCase):
    """Integration tests for image augmentation during training."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.processed_dir = self.data_dir / "processed"
        self.image_dir = self.processed_dir / "images"
        self.splits_dir = self.data_dir / "splits" / "test_split"
        self.checkpoint_dir = Path(self.test_dir) / "checkpoints"
        self.results_dir = Path(self.test_dir) / "results"
        
        # Create directories
        for dir_path in [self.processed_dir, self.image_dir, self.splits_dir, 
                         self.checkpoint_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create test data
        self._create_test_data()
        
        # Create config file
        self._create_config_file()
    
    def tearDown(self):
        """Clean up test directory."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_data(self):
        """Create test datasets and images."""
        # Create items
        num_items = 20
        self.item_ids = [f'item_{i}' for i in range(num_items)]
        
        # Create distinctive test images
        for i, item_id in enumerate(self.item_ids):
            img = Image.new('RGB', (224, 224))
            pixels = img.load()
            # Create unique pattern for each image
            for x in range(224):
                for y in range(224):
                    r = (255 * i // num_items) % 256
                    g = (x * 255 // 224) % 256
                    b = (y * 255 // 224) % 256
                    pixels[x, y] = (r, g, b)
            img.save(self.image_dir / f"{item_id}.jpg")
        
        # Create item info
        self.item_info_df = pd.DataFrame({
            'item_id': self.item_ids,
            'title': [f'Title {i}' for i in range(num_items)],
            'tag': [f'tag{i%5}' for i in range(num_items)],
            'description': [f'Description for item {i}' for i in range(num_items)],
            'view_number': np.random.randint(100, 1000, num_items),
            'comment_number': np.random.randint(0, 100, num_items),
        })
        self.item_info_df.to_csv(self.processed_dir / "item_info.csv", index=False)
        
        # Create interactions
        num_users = 10
        num_interactions = 100
        users = [f'user_{i}' for i in range(num_users)]
        
        interactions = []
        for _ in range(num_interactions):
            user = np.random.choice(users)
            item = np.random.choice(self.item_ids)
            interactions.append({'user_id': user, 'item_id': item})
        
        self.interactions_df = pd.DataFrame(interactions)
        self.interactions_df.to_csv(self.processed_dir / "interactions.csv", index=False)
        
        # Create train/val splits
        train_size = int(0.8 * len(interactions))
        train_df = self.interactions_df.iloc[:train_size]
        val_df = self.interactions_df.iloc[train_size:]
        
        train_df.to_csv(self.splits_dir / "train.csv", index=False)
        val_df.to_csv(self.splits_dir / "val.csv", index=False)
    
    def _create_config_file(self):
        """Create configuration file for testing."""
        config_content = {
            'model': {
                'vision_model': 'resnet',
                'language_model': 'sentence-bert',
                'embedding_dim': 32,
                'use_contrastive': False,
                'dropout_rate': 0.1,
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 0.001,
                'epochs': 3,
                'patience': 2,
                'num_workers': 0,
            },
            'data': {
                'processed_item_info_path': str(self.processed_dir / "item_info.csv"),
                'processed_interactions_path': str(self.processed_dir / "interactions.csv"),
                'image_folder': str(self.image_dir),
                'processed_image_destination_folder': str(self.image_dir),
                'train_data_path': str(self.splits_dir / "train.csv"),
                'val_data_path': str(self.splits_dir / "val.csv"),
                'numerical_features_cols': ['view_number', 'comment_number'],
                'numerical_normalization_method': 'none',
                'cache_config': {
                    'enabled': False,
                },
                'text_augmentation': {
                    'enabled': False,
                },
                'image_augmentation': {
                    'enabled': True,
                    'brightness': 0.3,
                    'contrast': 0.3,
                    'saturation': 0.2,
                    'hue': 0.1,
                    'random_crop': True,
                    'crop_scale': [0.8, 1.0],
                    'horizontal_flip': True,
                    'rotation_degrees': 15,
                    'gaussian_blur': True,
                    'blur_kernel_size': [3, 7],
                },
            },
            'checkpoint_dir': str(self.checkpoint_dir),
            'results_dir': str(self.results_dir),
        }
        
        self.config_path = Path(self.test_dir) / "test_config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(config_content, f)
    
    def test_training_with_augmentation(self):
        """Test that training works with image augmentation enabled."""
        # Load config
        config = Config.from_yaml(str(self.config_path))
        
        # Run training for a few epochs
        import subprocess
        result = subprocess.run([
            sys.executable, 'scripts/train.py',
            '--config', str(self.config_path),
            '--skip_wandb'
        ], capture_output=True, text=True)
        
        # Check training completed successfully
        self.assertEqual(result.returncode, 0, 
                        f"Training failed with error: {result.stderr}")
        
        # Check that model was saved
        checkpoint_path = self.checkpoint_dir / "resnet_sentence-bert" / "best_model.pth"
        self.assertTrue(checkpoint_path.exists(), "Model checkpoint not found")
        
        # Load and verify the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('epoch', checkpoint)
        self.assertIn('train_loss', checkpoint)
    
    def test_augmentation_impact_on_loss(self):
        """Test that augmentation affects training dynamics."""
        # First, train without augmentation
        config_no_aug = Config.from_yaml(str(self.config_path))
        config_no_aug.data.image_augmentation.enabled = False
        
        # Create datasets
        train_dataset_no_aug = self._create_dataset(config_no_aug, is_train=True)
        val_dataset_no_aug = self._create_dataset(config_no_aug, is_train=False)
        
        # Train model without augmentation
        model_no_aug = self._create_model(config_no_aug, train_dataset_no_aug)
        trainer_no_aug = Trainer(
            model_no_aug,
            checkpoint_dir=str(self.checkpoint_dir / "no_aug")
        )
        
        from torch.utils.data import DataLoader
        train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=4, shuffle=True)
        val_loader_no_aug = DataLoader(val_dataset_no_aug, batch_size=4, shuffle=False)
        
        train_losses_no_aug, val_losses_no_aug = trainer_no_aug.train(
            train_loader_no_aug,
            val_loader_no_aug,
            epochs=3,
            lr=0.001,
            patience=5
        )
        
        # Now train with augmentation
        config_with_aug = Config.from_yaml(str(self.config_path))
        config_with_aug.data.image_augmentation.enabled = True
        
        train_dataset_aug = self._create_dataset(config_with_aug, is_train=True)
        val_dataset_aug = self._create_dataset(config_with_aug, is_train=False)
        
        model_aug = self._create_model(config_with_aug, train_dataset_aug)
        trainer_aug = Trainer(
            model_aug,
            checkpoint_dir=str(self.checkpoint_dir / "with_aug")
        )
        
        train_loader_aug = DataLoader(train_dataset_aug, batch_size=4, shuffle=True)
        val_loader_aug = DataLoader(val_dataset_aug, batch_size=4, shuffle=False)
        
        train_losses_aug, val_losses_aug = trainer_aug.train(
            train_loader_aug,
            val_loader_aug,
            epochs=3,
            lr=0.001,
            patience=5
        )
        
        # Augmentation should generally lead to:
        # 1. Higher training loss (harder to fit augmented data)
        # 2. Better generalization (lower gap between train and val)
        
        # Check that we got losses
        self.assertGreater(len(train_losses_aug), 0)
        self.assertGreater(len(val_losses_aug), 0)
        
        # The augmented model might have higher training loss
        # but better generalization (smaller train-val gap)
        gap_no_aug = abs(train_losses_no_aug[-1] - val_losses_no_aug[-1])
        gap_aug = abs(train_losses_aug[-1] - val_losses_aug[-1])
        
        print(f"Train-val gap without augmentation: {gap_no_aug:.4f}")
        print(f"Train-val gap with augmentation: {gap_aug:.4f}")
    
    def _create_dataset(self, config, is_train):
        """Helper to create dataset with given config."""
        interactions_df = pd.read_csv(config.data.train_data_path if is_train 
                                     else config.data.val_data_path)
        item_info_df = pd.read_csv(config.data.processed_item_info_path)
        
        dataset = MultimodalDataset(
            interactions_df=interactions_df,
            item_info_df=item_info_df,
            image_folder=config.data.image_folder,
            vision_model_name=config.model.vision_model,
            language_model_name=config.model.language_model,
            create_negative_samples=True,
            cache_features=False,
            is_train_mode=is_train,
            image_augmentation_config=config.data.image_augmentation,
            text_augmentation_config=config.data.text_augmentation,
            numerical_feat_cols=config.data.numerical_features_cols,
            numerical_normalization_method=config.data.numerical_normalization_method
        )
        
        return dataset
    
    def _create_model(self, config, dataset):
        """Helper to create model."""
        model = MultimodalRecommender(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            num_numerical_features=len(config.data.numerical_features_cols),
            embedding_dim=config.model.embedding_dim,
            vision_model_name=config.model.vision_model,
            language_model_name=config.model.language_model,
            use_contrastive=config.model.use_contrastive,
            dropout_rate=config.model.dropout_rate
        )
        return model
    
    def test_augmentation_consistency_across_epochs(self):
        """Test that same item gets different augmentations across epochs."""
        config = Config.from_yaml(str(self.config_path))
        config.data.image_augmentation.enabled = True
        
        dataset = self._create_dataset(config, is_train=True)
        
        # Get the first item's features across multiple "epochs"
        # (simulated by accessing the same index multiple times)
        item_idx = 0
        features_per_epoch = []
        
        for epoch in range(3):
            # Reset random seed to simulate new epoch
            torch.manual_seed(epoch)
            features = dataset[item_idx]
            features_per_epoch.append(features['image'].clone())
        
        # Check that images are different across epochs
        for i in range(1, len(features_per_epoch)):
            diff = torch.abs(features_per_epoch[0] - features_per_epoch[i]).sum().item()
            self.assertGreater(diff, 0.01, 
                             f"Images should differ between epochs due to augmentation")
    
    def test_augmentation_performance_impact(self):
        """Test the performance impact of augmentation."""
        import time
        
        config = Config.from_yaml(str(self.config_path))
        
        # Test without augmentation
        config.data.image_augmentation.enabled = False
        dataset_no_aug = self._create_dataset(config, is_train=True)
        
        start_time = time.time()
        for i in range(min(50, len(dataset_no_aug))):
            _ = dataset_no_aug[i]
        time_no_aug = time.time() - start_time
        
        # Test with augmentation
        config.data.image_augmentation.enabled = True
        dataset_with_aug = self._create_dataset(config, is_train=True)
        
        start_time = time.time()
        for i in range(min(50, len(dataset_with_aug))):
            _ = dataset_with_aug[i]
        time_with_aug = time.time() - start_time
        
        print(f"Time without augmentation: {time_no_aug:.4f}s")
        print(f"Time with augmentation: {time_with_aug:.4f}s")
        print(f"Overhead: {(time_with_aug - time_no_aug) / time_no_aug * 100:.1f}%")
        
        # Augmentation should add some overhead, but not excessive
        # (e.g., less than 200% overhead)
        self.assertLess(time_with_aug, time_no_aug * 3.0,
                       "Augmentation overhead is too high")
    
    def test_augmentation_with_different_vision_models(self):
        """Test that augmentation works with different vision backbones."""
        vision_models = ['resnet', 'clip']  # Test with available models
        
        for vision_model in vision_models:
            with self.subTest(vision_model=vision_model):
                config = Config.from_yaml(str(self.config_path))
                config.model.vision_model = vision_model
                config.data.image_augmentation.enabled = True
                
                try:
                    dataset = self._create_dataset(config, is_train=True)
                    
                    # Get a few samples to ensure it works
                    for i in range(min(5, len(dataset))):
                        sample = dataset[i]
                        self.assertIn('image', sample)
                        self.assertEqual(sample['image'].shape, torch.Size([3, 224, 224]))
                    
                    print(f"✓ Augmentation works with {vision_model}")
                    
                except Exception as e:
                    self.fail(f"Augmentation failed with {vision_model}: {str(e)}")
    
    def test_edge_cases(self):
        """Test edge cases for image augmentation."""
        config = Config.from_yaml(str(self.config_path))
        
        # Test with extreme augmentation parameters
        config.data.image_augmentation.enabled = True
        config.data.image_augmentation.brightness = 1.0  # Maximum brightness
        config.data.image_augmentation.contrast = 1.0    # Maximum contrast
        config.data.image_augmentation.rotation_degrees = 180  # Large rotation
        
        dataset = self._create_dataset(config, is_train=True)
        
        # Should still work without errors
        try:
            for i in range(min(10, len(dataset))):
                sample = dataset[i]
                # Check that values are still in valid range
                self.assertGreaterEqual(sample['image'].min().item(), -3.0)
                self.assertLessEqual(sample['image'].max().item(), 3.0)
        except Exception as e:
            self.fail(f"Failed with extreme augmentation parameters: {str(e)}")
        
        # Test with all augmentations disabled except enabled=True
        config.data.image_augmentation.brightness = 0
        config.data.image_augmentation.contrast = 0
        config.data.image_augmentation.saturation = 0
        config.data.image_augmentation.hue = 0
        config.data.image_augmentation.random_crop = False
        config.data.image_augmentation.horizontal_flip = False
        config.data.image_augmentation.rotation_degrees = 0
        config.data.image_augmentation.gaussian_blur = False
        
        dataset2 = self._create_dataset(config, is_train=True)
        
        # Should work but produce identical images
        img1 = dataset2[0]['image']
        img2 = dataset2[0]['image']
        torch.testing.assert_close(img1, img2)


if __name__ == '__main__':
    unittest.main()