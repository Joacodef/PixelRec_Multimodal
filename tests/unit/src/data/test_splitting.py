# tests/unit/src/data/test_splitting.py
"""
Unit tests for the DataSplitter class.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.data.splitting import DataSplitter

class TestDataSplitter(unittest.TestCase):
    """Test cases for the DataSplitter functionality."""

    def setUp(self):
        """Set up a sample interactions DataFrame for each test."""
        self.splitter = DataSplitter(random_state=42)
        data = {
            'user_id': ['u1']*5 + ['u2']*3 + ['u3']*4 + ['u4']*1 + ['u5']*5,
            'item_id': [f'i{i}' for i in range(18)],
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=18, freq='D')),
            'tag': ['A', 'B', 'A', 'B', 'A'] + ['B', 'A', 'A'] + ['C', 'C', 'C', 'C'] + ['A'] + ['B', 'C', 'A', 'B', 'C']
        }
        self.interactions_df = pd.DataFrame(data)

    def test_user_based_split(self):
        """Tests splitting by user, ensuring no user overlap."""
        train_df, val_df = self.splitter.user_based_split(
            self.interactions_df, train_ratio=0.5, min_interactions_per_user=4
        )
        train_users = set(train_df['user_id'].unique())
        val_users = set(val_df['user_id'].unique())
        
        self.assertEqual(len(train_users.intersection(val_users)), 0)
        self.assertEqual(len(train_users), 1)
        self.assertEqual(len(val_users), 2)

    def test_stratified_temporal_split(self):
        """Tests the chronological and stratified splitting logic."""
        train_df, val_df, test_df = self.splitter.stratified_temporal_split(
            self.interactions_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, stratify_by='tag'
        )
        self.assertFalse(train_df.empty)
        self.assertFalse(val_df.empty)
        self.assertFalse(test_df.empty)
        self.assertTrue(train_df['timestamp'].max() <= val_df['timestamp'].min())
        
        # Define val_users before using it in the assertion
        val_users = set(val_df['user_id'].unique())
        self.assertTrue(val_users.issubset(set(train_df['user_id'].unique())))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)