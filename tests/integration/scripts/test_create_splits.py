import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import sys

# Add project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.create_splits import main as create_splits_main
from src.config import Config

class TestCreateSplitsEnhanced(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and dummy data/config files."""
        self.test_dir = Path("test_temp_data_splits_enhanced")
        self.test_dir.mkdir(exist_ok=True)
        
        # Define paths for raw data, processed data, and splits
        self.raw_data_dir = self.test_dir / "data" / "raw"
        self.processed_data_dir = self.test_dir / "data" / "processed"
        self.splits_dir = self.test_dir / "data" / "splits" / "test_split"
        self.configs_dir = self.test_dir / "configs"

        for d in [self.raw_data_dir, self.processed_data_dir, self.splits_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.interactions_path = self.processed_data_dir / "interactions.csv"
        self.item_info_path = self.processed_data_dir / "item_info.csv" # For stratification
        self.config_path = self.configs_dir / "test_config_splits.yaml"

        # Create more detailed dummy interactions data
        interactions_data = []
        for user_id in range(1, 51):
            num_interactions = np.random.randint(10, 20)
            for i in range(num_interactions):
                # Ensure timestamps are spread out to test temporal split
                interactions_data.append({
                    'user_id': f'user_{user_id}',
                    'item_id': f'item_{np.random.randint(1, 101)}',
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
                })
        self.interactions_df = pd.DataFrame(interactions_data)
        self.interactions_df.to_csv(self.interactions_path, index=False)

        # Create dummy item info for stratification
        item_ids = self.interactions_df['item_id'].unique()
        item_info_data = {
            'item_id': item_ids,
            'tag': [f'tag_{np.random.randint(0, 3)}' for _ in item_ids]
        }
        pd.DataFrame(item_info_data).to_csv(self.item_info_path, index=False)

        # Create a config file pointing to the temporary data
        config_content = f"""
        data:
          processed_interactions_path: {self.interactions_path.as_posix()}
          processed_item_info_path: {self.item_info_path.as_posix()}
          split_data_path: {self.splits_dir.as_posix()}
          splitting:
            strategy: 'stratified_temporal'
            stratify_by: 'tag' # Column to use for stratification
            random_state: 42
            train_final_ratio: 0.7
            val_final_ratio: 0.15
            test_final_ratio: 0.15
            min_interactions_per_user: 5
            min_interactions_per_item: 2
            validate_no_leakage: true # This is a config option in the project
        """
        self.config_path.write_text(config_content)
    
    def tearDown(self):
        """Clean up the temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_stratified_temporal_split_properties(self):
        """
        Test the properties of the splits created by the 'stratified_temporal' strategy.
        """
        # Run the main script function
        create_splits_main(config_path=str(self.config_path))

        # Define file paths using the correct attribute: self.splits_dir
        train_file = self.splits_dir / "train.csv"
        val_file = self.splits_dir / "val.csv"
        test_file = self.splits_dir / "test.csv"

        # 1. Assert that files were created
        self.assertTrue(train_file.exists(), "train.csv was not created.")
        self.assertTrue(val_file.exists(), "val.csv was not created.")
        self.assertTrue(test_file.exists(), "test.csv was not created.")

        # 2. Load the dataframes
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        # 3. Verify temporal correctness
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        
        self.assertTrue(train_df['timestamp'].max() <= val_df['timestamp'].min())
        self.assertTrue(train_df['timestamp'].max() <= test_df['timestamp'].min())

        # 4. Verify no data leakage (all users in val/test must exist in train)
        train_users = set(train_df['user_id'].unique())
        val_users = set(val_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())
        
        self.assertTrue(val_users.issubset(train_users), "All validation users should be in the training set.")
        self.assertTrue(test_users.issubset(train_users), "All test users should be in the training set.")

        # 5. Verify split ratios are approximately correct
        # This is a bit brittle, a better test would calculate this from source. For now, it works.
        total_interactions = len(self.interactions_df)
        self.assertAlmostEqual(len(train_df) / total_interactions, 0.7, delta=0.1)
        self.assertAlmostEqual(len(val_df) / total_interactions, 0.15, delta=0.1)
        self.assertAlmostEqual(len(test_df) / total_interactions, 0.15, delta=0.1)

        # 6. Verify stratification
        # The 'tag' column is no longer expected in the output splits.
        # It must be merged from the item_info file to verify the split's properties.
        item_info_df = pd.read_csv(self.item_info_path)
        val_df_merged = pd.merge(val_df, item_info_df[['item_id', 'tag']], on='item_id', how='left')
        test_df_merged = pd.merge(test_df, item_info_df[['item_id', 'tag']], on='item_id', how='left')

        # Check if the tag distribution in validation and test sets is similar
        val_tag_dist = val_df_merged['tag'].value_counts(normalize=True)
        test_tag_dist = test_df_merged['tag'].value_counts(normalize=True)

        # Reindex to ensure we are comparing the same tags, fill missing with 0
        all_tags = val_tag_dist.index.union(test_tag_dist.index)
        val_dist_norm = val_tag_dist.reindex(all_tags, fill_value=0)
        test_dist_norm = test_tag_dist.reindex(all_tags, fill_value=0)
        dist_diff = (val_dist_norm - test_dist_norm).abs().sum()

        # We divide by 2 because the sum of absolute differences is double the total variation distance
        total_variation_distance = dist_diff / 2

        self.assertLess(total_variation_distance, 0.2, "Total variation distance between tag distributions should be small.")

if __name__ == '__main__':
    unittest.main()