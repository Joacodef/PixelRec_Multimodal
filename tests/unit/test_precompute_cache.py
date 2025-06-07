# tests/unit/test_precompute_cache.py
"""
Unit tests for the precompute_cache.py script.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import torch
import pickle
import time  # <-- FIX: Import the time module
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import the script and src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.precompute_cache import precompute_features_cache

class TestPrecomputeCache(unittest.TestCase):
    """Test cases for the feature precomputation and caching script."""

    def setUp(self):
        """Set up a temporary environment for testing."""
        self.test_dir = Path("test_temp_cache")
        self.test_dir.mkdir(exist_ok=True)

        # Define paths
        self.processed_data_dir = self.test_dir / "data" / "processed"
        self.image_dir = self.processed_data_dir / "images"
        self.cache_dir = self.test_dir / "cache"
        self.configs_dir = self.test_dir / "configs"

        # Create directories
        for d in [self.processed_data_dir, self.image_dir, self.cache_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.vision_model = "resnet"
        self.language_model = "sentence-bert"

        # 1. Dummy Images
        Image.new('RGB', (100, 100), color='red').save(self.image_dir / "item_x.jpg")
        Image.new('RGB', (128, 128), color='green').save(self.image_dir / "item_y.jpg")
        Image.new('RGB', (100, 100), color='blue').save(self.image_dir / "item_z.jpg")

        # 2. Dummy "processed" item_info.csv
        self.numerical_cols = ['view_number', 'comment_number']
        self.item_info_df = pd.DataFrame({
            'item_id': ['item_x', 'item_y', 'item_z'],
            'title': ['Title X', 'Title Y', 'Title Z'],
            'tag': ['TagX', 'TagY', 'TagZ'],
            'description': ['Desc X', 'Desc Y', 'Desc Z'],
            'view_number': [150, 250, 350],
            'comment_number': [15, 25, 35],
        })
        self.processed_item_info_path = self.processed_data_dir / "item_info.csv"
        self.item_info_df.to_csv(self.processed_item_info_path, index=False)

        # 3. Dummy "processed" interactions.csv
        self.interactions_df = pd.DataFrame({'user_id': ['user_1'], 'item_id': ['item_x']})
        self.processed_interactions_path = self.processed_data_dir / "interactions.csv"
        self.interactions_df.to_csv(self.processed_interactions_path, index=False)

        # 4. Create a dummy scaler file
        self.scaler_path = self.processed_data_dir / "numerical_scaler.pkl"
        scaler = StandardScaler()
        scaler.fit(self.item_info_df[self.numerical_cols])
        with open(self.scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'columns': self.numerical_cols}, f)
        
        # 5. Dummy Config File
        self.config_path = self.configs_dir / "test_config_cache.yaml"
        config_content = f"""
model:
  vision_model: {self.vision_model}
  language_model: {self.language_model}

data:
  processed_item_info_path: {self.processed_item_info_path.as_posix()}
  processed_interactions_path: {self.processed_interactions_path.as_posix()}
  processed_image_destination_folder: {self.image_dir.as_posix()}
  image_folder: {self.image_dir.as_posix()}
  scaler_path: {self.scaler_path.as_posix()}
  numerical_features_cols: {self.numerical_cols}
  numerical_normalization_method: standardization

  cache_config:
    cache_directory: {self.cache_dir.as_posix()}
"""
        self.config_path.write_text(config_content)

    def tearDown(self):
        """Clean up the temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_cache_creation_and_content(self):
        """Test that cache files are created correctly and contain valid features."""
        precompute_features_cache(config_path=str(self.config_path))

        model_cache_dir = self.cache_dir / f"{self.vision_model}_{self.language_model}"
        self.assertTrue(model_cache_dir.exists(), "Model-specific cache directory was not created.")

        cached_files = list(model_cache_dir.glob("*.pt"))
        cached_filenames = [f.name for f in cached_files]
        self.assertEqual(len(cached_files), 3, "Expected 3 cache files to be created.")
        self.assertIn("item_x.pt", cached_filenames)

        cached_features = torch.load(model_cache_dir / "item_x.pt")
        self.assertIsInstance(cached_features, dict)
        
        # Check for expected feature keys in the cached dictionary
        expected_keys = ['image', 'text_input_ids', 'text_attention_mask', 'numerical_features']
        if 'clip_text_input_ids' in cached_features:
            expected_keys.extend(['clip_text_input_ids', 'clip_text_attention_mask'])
        self.assertListEqual(sorted(list(cached_features.keys())), sorted(expected_keys))
        
        self.assertEqual(cached_features['image'].shape, torch.Size([3, 224, 224]))

    def test_max_items_argument(self):
        """Test that the 'max_items' argument correctly limits the number of processed items."""
        precompute_features_cache(config_path=str(self.config_path), max_items=2)
        model_cache_dir = self.cache_dir / f"{self.vision_model}_{self.language_model}"
        cached_files = list(model_cache_dir.glob("*.pt"))
        self.assertEqual(len(cached_files), 2, "Should only process 2 items when max_items=2.")

    def test_force_recompute_argument(self):
        """Test that 'force_recompute' overwrites existing cache files."""
        model_cache_dir = self.cache_dir / f"{self.vision_model}_{self.language_model}"
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        dummy_file_path = model_cache_dir / "item_x.pt"
        dummy_file_path.touch()
        initial_mtime = dummy_file_path.stat().st_mtime

        time.sleep(0.01) # Ensure modification time will be different

        precompute_features_cache(config_path=str(self.config_path), force_recompute=True)

        self.assertTrue(dummy_file_path.exists())
        final_mtime = dummy_file_path.stat().st_mtime
        self.assertGreater(final_mtime, initial_mtime, "File should have been overwritten.")

if __name__ == '__main__':
    unittest.main()