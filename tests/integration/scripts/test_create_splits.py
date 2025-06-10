# tests/integration/scripts/test_create_splits.py
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import yaml
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from scripts.create_splits import main as create_splits_main
from src.config import Config

class TestCreateSplits(unittest.TestCase):
    """Test cases for create_splits.py functionality."""
    
    def setUp(self):
        """Set up a temporary directory and dummy data/config files for each test."""
        self.test_dir = Path("test_temp_data_splits")
        self.test_dir.mkdir(exist_ok=True)

        self.dummy_processed_data_dir = self.test_dir / "data" / "processed"
        self.dummy_splits_dir = self.test_dir / "data" / "splits" / "test_split"
        self.dummy_configs_dir = self.test_dir / "configs"
        self.dummy_processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_splits_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_configs_dir.mkdir(parents=True, exist_ok=True)

        self.interactions_path = self.dummy_processed_data_dir / "interactions.csv"
        self.config_path = self.dummy_configs_dir / "test_config_splits.yaml"

        interactions_data = []
        for user_id in range(1, 51):
            for _ in range(np.random.randint(10, 20)):
                interactions_data.append({
                    'user_id': f'user{user_id}',
                    'item_id': f'item{np.random.randint(1, 101)}',
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
                    'tag': f'tag_{np.random.randint(0, 4)}'
                })
        pd.DataFrame(interactions_data).to_csv(self.interactions_path, index=False)

        config_content = f"""
        data:
          processed_interactions_path: {self.interactions_path.as_posix()}
          split_data_path: {self.dummy_splits_dir.as_posix()}
          splitting:
            strategy: 'stratified_temporal'
            stratify_by: 'tag'
            random_state: 42
            train_final_ratio: 0.7
            val_final_ratio: 0.15
            test_final_ratio: 0.15
            min_interactions_per_user: 5
            min_interactions_per_item: 2
        """
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def tearDown(self):
        """Clean up the temporary directory after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_create_splits_stratified_temporal(self):
        """Test the 'stratified_temporal' strategy creates train/val/test CSVs."""
        create_splits_main(config_path=str(self.config_path))

        train_file = self.dummy_splits_dir / "train.csv"
        val_file = self.dummy_splits_dir / "val.csv"
        test_file = self.dummy_splits_dir / "test.csv"

        self.assertTrue(train_file.exists())
        self.assertTrue(val_file.exists())
        self.assertTrue(test_file.exists())

        train_df = pd.read_csv(train_file)
        self.assertFalse(train_df.empty)

if __name__ == '__main__':
    unittest.main()