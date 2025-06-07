# tests/unit/test_extract_encoders.py
"""
Unit tests for the extract_encoders.py script.
"""
import unittest
import pandas as pd
from pathlib import Path
import sys
import shutil
import pickle
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path to import the script and src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.extract_encoders import extract_encoders

class TestExtractEncoders(unittest.TestCase):
    """Test cases for the encoder extraction functionality."""

    def setUp(self):
        """Set up a temporary directory with dummy data and config."""
        self.test_dir = Path("test_temp_encoders")
        self.test_dir.mkdir(exist_ok=True)

        # Define paths
        self.processed_data_dir = self.test_dir / "data" / "processed"
        self.checkpoints_dir = self.test_dir / "models" / "checkpoints"
        self.configs_dir = self.test_dir / "configs"

        # Create directories
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # --- Create Dummy Data ---

        # 1. Dummy "processed" item_info.csv
        self.item_info_df = pd.DataFrame({
            'item_id': ['item_a', 'item_b', 'item_c'],
        })
        self.processed_item_info_path = self.processed_data_dir / "item_info.csv"
        self.item_info_df.to_csv(self.processed_item_info_path, index=False)

        # 2. Dummy "processed" interactions.csv
        self.interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_2', 'user_1', 'user_3'],
            'item_id': ['item_a', 'item_b', 'item_b', 'item_c'],
        })
        self.processed_interactions_path = self.processed_data_dir / "interactions.csv"
        self.interactions_df.to_csv(self.processed_interactions_path, index=False)

        # 3. Dummy Config File pointing to the temporary paths
        self.config_path = self.configs_dir / "test_config_encoders.yaml"
        config_content = f"""
data:
  processed_item_info_path: {self.processed_item_info_path.as_posix()}
  processed_interactions_path: {self.processed_interactions_path.as_posix()}
  # These paths are required by the dataset but not used for encoder extraction
  image_folder: 'dummy/path' 

checkpoint_dir: {self.checkpoints_dir.as_posix()}
"""
        self.config_path.write_text(config_content)

    def tearDown(self):
        """Clean up the temporary directory after the test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_encoder_extraction_and_saving(self):
        """
        Test that the script correctly creates and saves user and item encoders.
        """
        # --- Run the encoder extraction script ---
        extract_encoders(config_path=str(self.config_path))

        # --- Verify the Outputs ---
        
        # 1. Check for the existence of the encoder files
        encoders_output_dir = self.checkpoints_dir / 'encoders'
        user_encoder_path = encoders_output_dir / 'user_encoder.pkl'
        item_encoder_path = encoders_output_dir / 'item_encoder.pkl'

        self.assertTrue(encoders_output_dir.exists(), "Encoders output directory was not created.")
        self.assertTrue(user_encoder_path.exists(), "User encoder file was not saved.")
        self.assertTrue(item_encoder_path.exists(), "Item encoder file was not saved.")

        # 2. Load the saved encoders and validate their contents
        with open(user_encoder_path, 'rb') as f:
            user_encoder = pickle.load(f)

        with open(item_encoder_path, 'rb') as f:
            item_encoder = pickle.load(f)

        # Check that the loaded objects are correct
        self.assertIsInstance(user_encoder, LabelEncoder, "Saved user encoder is not a LabelEncoder instance.")
        self.assertIsInstance(item_encoder, LabelEncoder, "Saved item encoder is not a LabelEncoder instance.")

        # Check that the encoders have learned the correct classes from the dummy data
        expected_users = sorted(self.interactions_df['user_id'].unique())
        expected_items = sorted(self.interactions_df['item_id'].unique())

        self.assertListEqual(list(user_encoder.classes_), expected_users, "User encoder classes do not match expected users.")
        self.assertListEqual(list(item_encoder.classes_), expected_items, "Item encoder classes do not match expected items.")
        
        # Verify encoding works as expected
        self.assertEqual(user_encoder.transform(['user_1'])[0], 0)
        self.assertEqual(item_encoder.transform(['item_c'])[0], 2)


if __name__ == '__main__':
    unittest.main()