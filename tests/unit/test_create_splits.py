# tests/unit/test_create_splits.py
"""
Unit tests for the create_splits.py script
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import json
import os

# Add parent directory to path to import src modules and the script itself
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.create_splits import create_splits
from src.config import Config # To verify config loading
from src.data.splitting import DataSplitter # To verify splitter logic


class TestCreateSplits(unittest.TestCase):
    """Test cases for create_splits.py functionality"""
    
    def setUp(self):
        """Set up a temporary directory and dummy data/config files before each test."""
        self.test_dir = Path("test_temp_data_splits")
        self.test_dir.mkdir(exist_ok=True)

        # Define paths for dummy raw and processed data
        self.dummy_raw_data_dir = self.test_dir / "data" / "raw"
        self.dummy_processed_data_dir = self.test_dir / "data" / "processed"
        self.dummy_splits_dir = self.test_dir / "data" / "splits" / "test_split"
        self.dummy_configs_dir = self.test_dir / "configs"

        self.dummy_raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_splits_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_configs_dir.mkdir(parents=True, exist_ok=True)

        self.item_info_path = self.dummy_processed_data_dir / "item_info.csv"
        self.interactions_path = self.dummy_processed_data_dir / "interactions.csv"
        self.config_path = self.dummy_configs_dir / "test_config_splits.yaml"
        self.scaler_path = self.dummy_processed_data_dir / "numerical_scaler.pkl" # For numerical_processor

        # Create dummy item_info.csv (should be "processed" already for create_splits.py)
        # Increased number of items to 500
        self.item_info_df = pd.DataFrame({
            'item_id': [f'item{i}' for i in range(1, 501)],
            'title': [f'Title {i}' for i in range(1, 501)],
            'description': [f'Desc {i}' for i in range(1, 501)],
            'view_number': np.random.randint(100, 1000, 500),
            'comment_number': np.random.randint(10, 100, 500),
        })
        self.item_info_df.to_csv(self.item_info_path, index=False)

        # Create dummy interactions.csv (should be "processed" already for create_splits.py)
        # Increased number of users to 100, and interactions per user
        interactions_data = []
        for user_id in range(1, 101): # 100 users
            num_interactions = np.random.randint(20, 50) # 20 to 50 interactions per user
            for _ in range(num_interactions):
                item_id = np.random.randint(1, 501) # Items up to 500
                interactions_data.append({
                    'user_id': f'user{user_id}',
                    'item_id': f'item{item_id}',
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
                })
        self.interactions_df = pd.DataFrame(interactions_data)
        # Ensure user_id and item_id are strings
        self.interactions_df['user_id'] = self.interactions_df['user_id'].astype(str)
        self.interactions_df['item_id'] = self.interactions_df['item_id'].astype(str)
        self.interactions_df.to_csv(self.interactions_path, index=False)

        # Create a dummy config file
        config_content = f"""
        data:
          item_info_path: {self.item_info_path.as_posix()}
          interactions_path: {self.interactions_path.as_posix()}
          processed_item_info_path: {self.item_info_path.as_posix()}
          processed_interactions_path: {self.interactions_path.as_posix()}
          image_folder: {self.dummy_raw_data_dir.as_posix()}/images # Not directly used by create_splits, but in config
          processed_image_destination_folder: {self.dummy_processed_data_dir.as_posix()}/images
          scaler_path: {self.scaler_path.as_posix()}
          split_data_path: {self.dummy_splits_dir.as_posix()}
          train_data_path: {self.dummy_splits_dir.as_posix()}/train.csv
          val_data_path: {self.dummy_splits_dir.as_posix()}/val.csv
          test_data_path: {self.dummy_splits_dir.as_posix()}/test.csv
          numerical_features_cols:
            - view_number
            - comment_number
          splitting:
            random_state: 42
            train_final_ratio: 0.7
            val_final_ratio: 0.15
            test_final_ratio: 0.15
            min_interactions_per_user: 3 # Lower to ensure more users remain
            min_interactions_per_item: 3 # Lower to ensure more items remain
            validate_no_leakage: True
        """
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def tearDown(self):
        """Clean up the temporary directory after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_create_splits_basic_functionality(self):
        """
        Test that create_splits.py correctly creates train/val/test CSVs
        and a metadata JSON file with appropriate contents and ratios.
        """
        # Run the create_splits function from the script
        create_splits(self.config_path.as_posix())

        # Verify that output files exist
        train_file = self.dummy_splits_dir / "train.csv"
        val_file = self.dummy_splits_dir / "val.csv"
        test_file = self.dummy_splits_dir / "test.csv"
        metadata_file = self.dummy_splits_dir / "split_metadata.json"

        self.assertTrue(train_file.exists(), f"Train file {train_file} should exist.")
        self.assertTrue(val_file.exists(), f"Validation file {val_file} should exist.")
        self.assertTrue(test_file.exists(), f"Test file {test_file} should exist.")
        self.assertTrue(metadata_file.exists(), f"Metadata file {metadata_file} should exist.")

        # Load the created dataframes
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

        # Check if dataframes are not empty
        self.assertFalse(train_df.empty, "Train DataFrame should not be empty.")
        self.assertFalse(val_df.empty, "Validation DataFrame should not be empty.")
        self.assertFalse(test_df.empty, "Test DataFrame should not be empty.")

        # Check interaction counts (approximately based on ratios)
        total_interactions_after_filtering = len(train_df) + len(val_df) + len(test_df)
        initial_interactions_count = len(self.interactions_df)

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        self.assertEqual(metadata['train_size'], len(train_df))
        self.assertEqual(metadata['val_size'], len(val_df))
        self.assertEqual(metadata['test_size'], len(test_df))
        self.assertEqual(metadata['total_interactions_after_activity_filtering'], total_interactions_after_filtering)

        # Check ratios (allow for small floating point discrepancies due to stratified split)
        # Stratified split may not perfectly match ratios for small datasets, but should be close.
        configured_train_ratio = 0.7
        configured_val_ratio = 0.15
        configured_test_ratio = 0.15

        actual_train_ratio = len(train_df) / total_interactions_after_filtering
        actual_val_ratio = len(val_df) / total_interactions_after_filtering
        actual_test_ratio = len(test_df) / total_interactions_after_filtering
        
        self.assertAlmostEqual(actual_train_ratio, configured_train_ratio, delta=0.05)
        self.assertAlmostEqual(actual_val_ratio, configured_val_ratio, delta=0.05)
        self.assertAlmostEqual(actual_test_ratio, configured_test_ratio, delta=0.05)

        # Check user/item overlap for stratified split (users/items should appear in multiple splits)
        train_users = set(train_df['user_id'].unique())
        val_users = set(val_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())

        train_items = set(train_df['item_id'].unique())
        val_items = set(val_df['item_id'].unique())
        test_items = set(test_df['item_id'].unique())

        # For stratified split, we expect significant user overlap (most users should be in train, val, test)
        # The goal is to distribute interactions of users across splits, not to create cold-start users.
        self.assertTrue(len(train_users.intersection(val_users)) > 0, "Users should overlap between train and val")
        self.assertTrue(len(train_users.intersection(test_users)) > 0, "Users should overlap between train and test")
        self.assertTrue(len(val_users.intersection(test_users)) > 0, "Users should overlap between val and test")

        self.assertTrue(len(train_items.intersection(val_items)) > 0, "Items should overlap between train and val")
        self.assertTrue(len(train_items.intersection(test_items)) > 0, "Items should overlap between train and test")
        self.assertTrue(len(val_items.intersection(test_items)) > 0, "Items should overlap between val and test")

        # Verify min_interactions_per_user/item are respected by the final datasets (indirectly)
        # This is primarily handled by filter_by_activity within create_splits.py
        # Check that users/items in splits have at least min_interactions if they appeared in filtered data
        final_all_interactions_df = pd.concat([train_df, val_df, test_df])
        user_counts_final = final_all_interactions_df['user_id'].value_counts()
        item_counts_final = final_all_interactions_df['item_id'].value_counts()

        # All users/items present in the final splits should meet the minimum threshold.
        # This check is more about the filtering *before* splitting.
        self.assertTrue(all(count >= 3 for count in user_counts_final), "All users in final splits should have >= 3 interactions")
        self.assertTrue(all(count >= 3 for count in item_counts_final), "All items in final splits should have >= 3 interactions")

    def test_create_splits_with_sampling(self):
        """
        Test that create_splits.py correctly samples the dataset when `sample_n` is provided.
        """
        sample_size = 500
        # Run the create_splits function with sampling
        create_splits(self.config_path.as_posix(), sample_n=sample_size)

        metadata_file = self.dummy_splits_dir / "split_metadata.json"
        self.assertTrue(metadata_file.exists(), f"Metadata file {metadata_file} should exist.")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify that sampling was requested and applied
        self.assertEqual(metadata['requested_sample_n'], sample_size)
        
        # The 'total_interactions_after_activity_filtering' should be <= sample_size
        # (It could be less if sampling results in users/items below the min_interactions threshold)
        self.assertTrue(metadata['total_interactions_after_activity_filtering'] <= sample_size)
        
        train_file = self.dummy_splits_dir / "train.csv"
        val_file = self.dummy_splits_dir / "val.csv"
        test_file = self.dummy_splits_dir / "test.csv"
        
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

        total_interactions_in_splits = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_interactions_in_splits, metadata['total_interactions_after_activity_filtering'])

    def test_create_splits_empty_interactions_after_filtering(self):
        """
        Test that create_splits.py handles cases where no interactions remain after filtering.
        """
        # Create dummy interactions.csv that will result in no interactions after filtering
        # E.g., all users/items have < min_interactions_per_user/item
        tiny_interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'item_id': ['i1', 'i2', 'i3', 'i4']
        })
        tiny_interactions_df.to_csv(self.interactions_path, index=False)

        # Temporarily modify config to have high filtering thresholds
        config = Config.from_yaml(self.config_path.as_posix())
        config.data.splitting.min_interactions_per_user = 5
        config.data.splitting.min_interactions_per_item = 5
        config.to_yaml(self.config_path.as_posix()) # Save modified config

        # Expect the script to exit with an error
        with self.assertRaises(SystemExit) as cm:
            create_splits(self.config_path.as_posix())
        self.assertEqual(cm.exception.code, 1) # SystemExit code 1 indicates an error


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)