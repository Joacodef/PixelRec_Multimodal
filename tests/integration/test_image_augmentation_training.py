# tests/integration/test_image_augmentation_training.py
"""
A robust, end-to-end integration test for the training pipeline.
This test first runs the actual preprocessing and splitting scripts
to create a valid test environment, then runs the training script.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml
import sys
import torch
import traceback
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.preprocess_data import main as preprocess_main
from scripts.create_splits import create_splits as create_splits_main
from scripts.train import main as train_main
from src.config import Config

class TestImageAugmentationTraining(unittest.TestCase):
    """
    This test verifies the entire pipeline:
    1. Preprocessing raw data.
    2. Creating data splits.
    3. Training the model with augmentation enabled.
    """

    def setUp(self):
        """Set up the entire test environment by running preprocessing scripts."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Define paths for raw, processed, and split data
        self.raw_dir = self.test_dir / "data" / "raw"
        self.raw_image_dir = self.raw_dir / "images"
        self.processed_dir = self.test_dir / "data" / "processed"
        self.splits_dir = self.test_dir / "data" / "splits" / "test_split"
        self.checkpoint_dir = self.test_dir / "checkpoints"
        
        # Create initial raw directories
        self.raw_image_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "item_info").mkdir(exist_ok=True)
        (self.raw_dir / "interactions").mkdir(exist_ok=True)
        
        # 1. Create dummy RAW data
        self._create_raw_data()
        
        # 2. Create a config file pointing to this temporary structure
        self._create_config_file()
        
        # 3. Run the actual preprocessing and splitting scripts
        print("\n--- Running preprocess_data.py for test setup ---")
        preprocess_main(cli_args=['--config', str(self.config_path)])
        print("\n--- Running create_splits.py for test setup ---")
        create_splits_main(config_path=str(self.config_path))
        print("\n--- Test setup complete ---")


    def tearDown(self):
        """Clean up the temporary directory."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def _create_raw_data(self):
        """Creates dummy raw data files, including images."""
        # Create item info
        num_items = 20
        item_ids = [f'item_{i}' for i in range(num_items)]
        
        pd.DataFrame({
            'item_id': item_ids,
            'title': [f'Title {i}' for i in range(num_items)],
            'description': [f'Desc {i}' for i in range(num_items)],
            'view_number': np.random.randint(100, 1000, num_items),
            'comment_number': np.random.randint(0, 100, num_items),
        }).to_csv(self.raw_dir / "item_info" / "item_info_sample.csv", index=False)
        
        # FIX: Add a loop to create a dummy image file for each item.
        # This is what the preprocess_data.py script expects to find.
        for item_id in item_ids:
            try:
                # Creates a simple 64x64 black image, which will pass validation.
                img = Image.new('RGB', (64, 64), color = 'black')
                img.save(self.raw_image_dir / f"{item_id}.jpg")
            except Exception as e:
                self.fail(f"Failed to create dummy image for {item_id}: {e}")

        # Create interactions
        interactions = [{'user_id': f'user_{i}', 'item_id': f'item_{j}'} for i in range(10) for j in range(10)]
        pd.DataFrame(interactions).to_csv(self.raw_dir / "interactions" / "interactions_sample.csv", index=False)

    def _create_config_file(self):
        """Creates a complete YAML configuration for the test pipeline."""
        self.config_path = self.test_dir / "test_config.yaml"
        config_content = {
            'model': {'vision_model': 'resnet', 'language_model': 'sentence-bert', 'embedding_dim': 16},
            'training': {'batch_size': 4, 'epochs': 1, 'patience': 1, 'num_workers': 0},
            'data': {
                'item_info_path': str(self.raw_dir / "item_info" / "item_info_sample.csv"),
                'interactions_path': str(self.raw_dir / "interactions" / "interactions_sample.csv"),
                'image_folder': str(self.raw_image_dir),
                'processed_item_info_path': str(self.processed_dir / "item_info.csv"),
                'processed_interactions_path': str(self.processed_dir / "interactions.csv"),
                'processed_image_destination_folder': str(self.processed_dir / "images"),
                'scaler_path': str(self.processed_dir / "numerical_scaler.pkl"),
                'split_data_path': str(self.splits_dir),
                'train_data_path': str(self.splits_dir / "train.csv"),
                'val_data_path': str(self.splits_dir / "val.csv"),
                'test_data_path': str(self.splits_dir / "test.csv"),
                'numerical_features_cols': ['view_number', 'comment_number'],
                'splitting': { 'min_interactions_per_user': 1, 'min_interactions_per_item': 1 },
                'image_augmentation': {'enabled': True, 'brightness': 0.2, 'contrast': 0.2},
                'cache_config': {'enabled': False},
            },
            'checkpoint_dir': str(self.checkpoint_dir),
            'results_dir': str(self.test_dir / "results"),
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_content, f)

    def test_end_to_end_training_with_augmentation(self):
        """
        Verifies that the main training script runs to completion using the
        data generated by the preprocessing and splitting scripts.
        """
        try:
            # Run the training script using the fully prepared data
            train_main(cli_args=['--config', str(self.config_path), '--device', 'cpu'])
        except Exception as e:
            self.fail(f"Training script failed with an exception: {e}\n{traceback.format_exc()}")

        # Verify that a model checkpoint was created as expected
        checkpoint_path = self.checkpoint_dir / "resnet_sentence-bert" / "best_model.pth"
        self.assertTrue(checkpoint_path.exists(), "Model checkpoint was not created.")
        
        # Verify the checkpoint is valid
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint)

if __name__ == '__main__':
    unittest.main()