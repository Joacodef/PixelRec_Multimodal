import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import json
from PIL import Image

# Add parent directory to path to import src and scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.train import main as train_main

class TestTrainScript(unittest.TestCase):
    """Integration test for the train.py script."""

    def setUp(self):
        """Set up a temporary, self-contained environment for a training run."""
        self.test_dir = Path("test_temp_train")
        self.test_dir.mkdir(exist_ok=True)

        # Define and create all necessary directories
        self.processed_data_dir = self.test_dir / "data" / "processed"
        self.splits_dir = self.test_dir / "data" / "splits" / "tiny_split"
        self.raw_image_dir = self.test_dir / "data" / "raw" / "images"
        self.checkpoints_dir = self.test_dir / "models" / "checkpoints"
        self.results_dir = self.test_dir / "results"
        self.configs_dir = self.test_dir / "configs"

        for d in [self.processed_data_dir, self.splits_dir, self.raw_image_dir,
                    self.checkpoints_dir, self.results_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- Create Dummy Data, Images, and Config ---

        # 1. Dummy "processed" item info and interactions (for encoder fitting)
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4'],
            'title': ['A', 'B', 'C', 'D'], 'tag': ['X', 'Y', 'X', 'Z'], 'description': ['d1', 'd2', 'd3', 'd4'],
            'view_number': [100, 200, 50, 150], 'comment_number': [10, 20, 5, 15]
        })
        self.processed_item_info_path = self.processed_data_dir / "item_info.csv"
        self.item_info_df.to_csv(self.processed_item_info_path, index=False)

        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u3'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item4']
        })
        self.processed_interactions_path = self.processed_data_dir / "interactions.csv"
        self.interactions_df.to_csv(self.processed_interactions_path, index=False)

        # 2. Dummy data splits (very small for a quick training epoch)
        train_split_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['item1', 'item3']})
        val_split_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['item2', 'item1']})
        self.train_path = self.splits_dir / "train.csv"
        self.val_path = self.splits_dir / "val.csv"
        train_split_df.to_csv(self.train_path, index=False)
        val_split_df.to_csv(self.val_path, index=False)
        
        # 3. Dummy images (required by the dataset loader)
        for item_id in self.item_info_df['item_id']:
            Image.new('RGB', (100, 100), color='red').save(self.raw_image_dir / f"{item_id}.jpg")

        # 4. Dummy Config File
        self.config_path = self.configs_dir / "test_config_train.yaml"
        # This config points to all the temporary files and sets up a minimal, fast run
        config_content = f"""
model:
  vision_model: resnet
  language_model: sentence-bert
  embedding_dim: 8
  use_contrastive: false # Disable for simplicity and speed in test

training:
  batch_size: 2
  epochs: 1 # Only run for one epoch
  learning_rate: 0.01
  patience: 1
  num_workers: 0 # IMPORTANT for avoiding multiprocessing issues in tests

data:
  processed_item_info_path: {self.processed_item_info_path.as_posix()}
  processed_interactions_path: {self.processed_interactions_path.as_posix()}
  image_folder: {self.raw_image_dir.as_posix()}
  scaler_path: {self.processed_data_dir / "numerical_scaler.pkl"}
  train_data_path: {self.train_path.as_posix()}
  val_data_path: {self.val_path.as_posix()}
  numerical_features_cols:
    - view_number
    - comment_number
    - non_existent_feature # Add a non-existent feature to test validation
  cache_config:
    enabled: false

checkpoint_dir: {self.checkpoints_dir.as_posix()}
results_dir: {self.results_dir.as_posix()}
"""
        self.config_path.write_text(config_content)

    def tearDown(self):
        """Clean up all temporary directories and files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_training_pipeline_single_epoch(self):
        """
        Test the entire training pipeline for a single epoch to ensure all components
        are wired correctly and the script produces the expected artifacts.
        """
        # Mock sys.argv to simulate running the script from the command line
        original_argv = sys.argv
        sys.argv = [
            'scripts/train.py',
            '--config', str(self.config_path),
            '--device', 'cpu' # Force CPU for CI/testing environments
        ]

        # Run the main training function
        train_main()

        # --- Verification ---
        # 1. Verify that all expected output files and directories were created.

        # Check for model-specific checkpoint directory
        model_specific_checkpoint_dir = self.checkpoints_dir / "resnet_sentence-bert"
        self.assertTrue(model_specific_checkpoint_dir.exists())

        # Check for saved model checkpoints
        self.assertTrue((model_specific_checkpoint_dir / "best_model.pth").exists())
        self.assertTrue((model_specific_checkpoint_dir / "last_model.pth").exists())

        # Check for saved encoders
        encoders_dir = self.checkpoints_dir / "encoders"
        self.assertTrue(encoders_dir.exists())
        self.assertTrue((encoders_dir / "user_encoder.pkl").exists())
        self.assertTrue((encoders_dir / "item_encoder.pkl").exists())

        # Check for results files (excluding the plot)
        self.assertTrue((self.results_dir / "training_metadata.json").exists())
        self.assertTrue((self.results_dir / "training_run_config.yaml").exists())

        # 2. Verify the content of the metadata file
        with open(self.results_dir / "training_metadata.json", "r") as f:
            metadata = json.load(f)

        self.assertTrue(metadata['training_completed'])
        self.assertEqual(metadata['epochs_completed'], 1)
        self.assertIn('final_train_loss', metadata)
        self.assertIsNotNone(metadata['final_train_loss'])
        
        # Verify that the numerical feature validation worked correctly
        validated_features = metadata['numerical_features_validation']['validated_features']
        missing_features = metadata['numerical_features_validation']['missing_features']
        self.assertIn('view_number', validated_features)
        self.assertIn('comment_number', validated_features)
        self.assertNotIn('non_existent_feature', validated_features)
        self.assertIn('non_existent_feature', missing_features)
        self.assertEqual(metadata['numerical_features_validation']['num_features_used'], 2)

        # Restore original sys.argv
        sys.argv = original_argv

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)