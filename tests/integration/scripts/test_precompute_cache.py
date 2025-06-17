# tests/integration/scripts/test_precompute_cache.py

import unittest
import shutil
import time
from pathlib import Path
import yaml
import pandas as pd
import torch
import sys
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.precompute_cache import precompute_features_cache

from src.config import Config

class TestPrecomputeCache(unittest.TestCase):
    """Integration tests for the precompute_cache.py script."""

    def setUp(self):
        """Set up a temporary directory structure and a mock config file."""
        self.test_dir = Path("test_temp_cache")
        self.test_dir.mkdir(exist_ok=True)

        # Create nested directories
        self.data_dir = self.test_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.image_dir = self.processed_data_dir / "images"
        self.cache_dir = self.test_dir / "cache"
        self.config_path = self.test_dir / "config.yaml"

        for d in [self.raw_data_dir, self.processed_data_dir, self.image_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create mock data files
        self.item_info = pd.DataFrame({
            'item_id': ['item_a', 'item_b', 'item_c'],
            'description': ['desc a', 'desc b', 'desc c'],
            'views': [100, 200, 300]
        })
        self.item_info.to_csv(self.processed_data_dir / "item_info.csv", index=False)
        
        # Create dummy image files
        for item_id in self.item_info['item_id']:
            (self.image_dir / f"{item_id}.jpg").touch()

        # Create mock config from which the script will run
        self.vision_model = "resnet"
        self.language_model = "sentence-bert"  # This matches the stdout log
        
        # Define the scaler_path and create an empty dummy file for it.
        self.scaler_path = self.processed_data_dir / "scaler.pkl"
        with open(self.scaler_path, 'wb') as f:
            # The script expects a dictionary with a 'scaler' key.
            pickle.dump({'scaler': None, 'columns': []}, f)

        config_dict = {
            'data': {
                'processed_item_info_path': str(self.processed_data_dir / "item_info.csv"),
                'processed_interactions_path': str(self.processed_data_dir / "item_info.csv"),
                'processed_image_destination_folder': str(self.image_dir),
                'scaler_path': str(self.scaler_path),
                'numerical_features_cols': ['views'],
                'cache_config': {'cache_directory': str(self.cache_dir)}
            },
            'model': {
                'vision_model': self.vision_model,
                'language_model': self.language_model,
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f)

    def tearDown(self):
        """Clean up the temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _get_expected_cache_dir(self) -> Path:
        """Helper to build the expected cache directory path dynamically."""
        return self.cache_dir / f"vision_{self.vision_model}_lang_{self.language_model or 'none'}"

    def test_cache_creation_and_content(self):
        """Test that cache files are created correctly and contain valid features."""
        precompute_features_cache(config_path=str(self.config_path))
        
        model_cache_dir = self._get_expected_cache_dir()
        self.assertTrue(model_cache_dir.exists(), "Model-specific cache directory was not created.")
        
        cached_files = list(model_cache_dir.glob("*.pt"))
        self.assertEqual(len(cached_files), 3)
        
        # Verify content of one file
        features = torch.load(cached_files[0])
        self.assertIn('image', features)
        self.assertIn('text_input_ids', features)

    def test_max_items_argument(self):
        """Test that the 'max_items' argument correctly limits the number of processed items."""
        precompute_features_cache(config_path=str(self.config_path), max_items=2)
        
        model_cache_dir = self._get_expected_cache_dir()
        cached_files = list(model_cache_dir.glob("*.pt"))
        self.assertEqual(len(cached_files), 2, "Should only process 2 items when max_items=2.")

    def test_force_recompute_argument(self):
        """Test that 'force_recompute' overwrites existing cache files."""
        model_cache_dir = self._get_expected_cache_dir()
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy file for just one item that will be processed
        dummy_file_path = model_cache_dir / "item_a.pt"
        dummy_file_path.touch()
        initial_mtime = dummy_file_path.stat().st_mtime

        time.sleep(0.01)

        precompute_features_cache(config_path=str(self.config_path), force_recompute=True)

        self.assertTrue(dummy_file_path.exists())
        final_mtime = dummy_file_path.stat().st_mtime
        self.assertGreater(final_mtime, initial_mtime, "File should have been overwritten.")