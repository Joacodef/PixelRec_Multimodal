# tests/unit/test_preprocess_data.py
"""
Unit tests for the preprocess_data.py script.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import pickle
import os
from PIL import Image

# Add parent directory to path to import src modules and the script itself
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.preprocess_data import main as preprocess_main
from src.config import Config

class TestPreprocessData(unittest.TestCase):
    """Test cases for the data preprocessing pipeline."""

    def setUp(self):
        """Set up a temporary environment with dummy data and configs."""
        self.test_dir = Path("test_temp_preprocess")
        self.test_dir.mkdir(exist_ok=True)

        # Define paths
        self.raw_dir = self.test_dir / "data" / "raw"
        self.processed_dir = self.test_dir / "data" / "processed"
        self.image_dir = self.raw_dir / "images"
        self.processed_image_dir = self.processed_dir / "images"
        self.configs_dir = self.test_dir / "configs"

        # Create directories
        for d in [self.raw_dir, self.processed_dir, self.image_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- Create Dummy Data ---

        # 1. Dummy Images
        Image.new('RGB', (100, 100), color='red').save(self.image_dir / "item1.jpg")
        Image.new('RGB', (128, 128), color='green').save(self.image_dir / "item2.jpg")
        Image.new('RGB', (100, 100), color='blue').save(self.image_dir / "item3.jpg")
        Image.new('RGB', (100, 100), color='yellow').save(self.image_dir / "item5.jpg")
        (self.image_dir / "item_corrupted.jpg").write_text("this is not an image")
        Image.new('RGB', (50, 50), color='purple').save(self.image_dir / "item_small.jpg")

        # 2. Dummy Item Info with corrected description
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4', 'item5', 'item_corrupted', 'item_small'],
            'title': ['A Good Title', 'Another Title', 'Title 3', 'Title 4', 'Title 5', 'Corrupted', 'Small'],
            'tag': ['TagA', 'TagB', 'TagA', 'TagC', 'TagB', 'TagD', 'TagE'],
            'description': ['<p>Description</p>', 'Desc 2', 'Desc 3', 'Desc 4', 'Desc 5', 'Desc 6', 'Desc 7'],
            'view_number': [100, 200, 50, 300, 10, np.nan, 500],
            'comment_number': [10, 20, 5, 30, 1, 5, 50],
        })
        self.raw_item_info_path = self.raw_dir / "item_info.csv"
        self.item_info_df.to_csv(self.raw_item_info_path, index=False)

        # 3. Dummy Interactions
        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3'],
            'item_id': ['item1', 'item2', 'item1', 'item2', 'item1', 'item2', 'item3'],
        })
        self.raw_interactions_path = self.raw_dir / "interactions.csv"
        self.interactions_df.to_csv(self.raw_interactions_path, index=False)

        # 4. Dummy Config File
        self.config_path = self.configs_dir / "test_config_preprocess.yaml"
        config_content = f"""
model:
  vision_model: resnet
  language_model: sentence-bert

data:
  item_info_path: {self.raw_item_info_path.as_posix()}
  interactions_path: {self.raw_interactions_path.as_posix()}
  image_folder: {self.image_dir.as_posix()}

  processed_item_info_path: {self.processed_dir / 'item_info.csv'}
  processed_interactions_path: {self.processed_dir / 'interactions.csv'}
  processed_image_destination_folder: {self.processed_image_dir.as_posix()}
  scaler_path: {self.processed_dir / 'numerical_scaler.pkl'}

  numerical_features_cols:
    - view_number
    - comment_number
  numerical_normalization_method: standardization

  offline_image_validation:
    check_corrupted: true
    min_width: 64
    min_height: 64
    allowed_extensions: ['.jpg', '.jpeg', '.png']
  
  offline_image_compression:
    enabled: true
    compress_if_kb_larger_than: 1
    target_quality: 90

  offline_text_cleaning:
    remove_html: true
    to_lowercase: true
    normalize_unicode: true

  splitting: # Used for filtering thresholds
    min_interactions_per_user: 2
    min_interactions_per_item: 2
"""
        self.config_path.write_text(config_content)

    def tearDown(self):
        """Clean up the temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_preprocessing_pipeline(self):
        """
        Test the full preprocessing pipeline by running the main script
        and verifying all outputs and transformations.
        """
        # --- Run the script ---
        original_argv = sys.argv
        sys.argv = ['scripts/preprocess_data.py', '--config', str(self.config_path)]
        preprocess_main()
        sys.argv = original_argv

        # --- Verify Outputs ---
        processed_item_info = self.processed_dir / 'item_info.csv'
        processed_interactions = self.processed_dir / 'interactions.csv'
        scaler_file = self.processed_dir / 'numerical_scaler.pkl'

        self.assertTrue(processed_item_info.exists(), "Processed item info file not created.")
        self.assertTrue(processed_interactions.exists(), "Processed interactions file not created.")
        self.assertTrue(scaler_file.exists(), "Numerical scaler file not created.")
        self.assertTrue(self.processed_image_dir.exists(), "Processed image directory not created.")

        # --- Validate Content ---

        # 2. Validate processed images
        processed_images = os.listdir(self.processed_image_dir)
        self.assertIn("item1.jpg", processed_images)
        self.assertIn("item2.jpg", processed_images)
        self.assertIn("item3.jpg", processed_images)
        self.assertIn("item5.jpg", processed_images)
        self.assertNotIn("item_corrupted.jpg", processed_images)
        self.assertNotIn("item_small.jpg", processed_images)
        self.assertEqual(len(processed_images), 4, "Should contain all initially valid images.")

        # 3. Load processed data for validation
        items_df = pd.read_csv(processed_item_info)
        interactions_df = pd.read_csv(processed_interactions)
        with open(scaler_file, 'rb') as f:
            scaler_data = pickle.load(f)

        # 4. Validate Data Filtering
        self.assertEqual(len(items_df), 2, "Item info CSV should be filtered to 2 final items.")
        self.assertListEqual(sorted(items_df['item_id'].tolist()), ['item1', 'item2'])
        self.assertEqual(len(interactions_df['user_id'].unique()), 2, "Interactions CSV should be filtered to 2 users.")
        self.assertListEqual(sorted(interactions_df['user_id'].unique()), ['u1', 'u2'])

        # 5. Validate Text Cleaning
        item1_title = items_df[items_df['item_id'] == 'item1']['title'].iloc[0]
        self.assertEqual(item1_title, 'a good title', "Text should be lowercased.")
        item1_desc = items_df[items_df['item_id'] == 'item1']['description'].iloc[0]
        # This assertion should now pass
        self.assertEqual(item1_desc, 'description', "HTML tags should be removed.")

        # 6. Validate Numerical Processing
        self.assertFalse(items_df['view_number'].isnull().any(), "NaNs in numerical columns should be filled.")
        from sklearn.preprocessing import StandardScaler
        self.assertIsInstance(scaler_data['scaler'], StandardScaler, "Scaler should be a StandardScaler instance.")
        self.assertEqual(scaler_data['columns'], ['view_number', 'comment_number'])

if __name__ == '__main__':
    unittest.main()