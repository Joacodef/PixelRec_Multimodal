# tests/unit/src/data/test_splitting.py
"""
Unit tests for the DataSplitter class, which handles various data splitting strategies.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.data.splitting import DataSplitter

class TestDataSplitter(unittest.TestCase):
    """Test cases for the DataSplitter functionality."""

    def setUp(self):
        """Set up a sample interactions DataFrame and a DataSplitter instance before each test."""
        # Initializes a DataSplitter with a fixed random state for reproducible test results.
        self.splitter = DataSplitter(random_state=42)
        
        # Creates a diverse sample DataFrame to test various splitting scenarios.
        # It includes users with different interaction counts and timestamps for temporal tests.
        data = {
            'user_id': [
                'u1', 'u1', 'u1', 'u1', 'u1',  # User with 5 interactions
                'u2', 'u2', 'u2',              # User with 3 interactions
                'u3', 'u3', 'u3', 'u3',        # User with 4 interactions
                'u4',                          # User with 1 interaction
                'u5', 'u5', 'u5', 'u5', 'u5',  # Another user with 5
            ],
            'item_id': [
                'i1', 'i2', 'i3', 'i4', 'i5',
                'i1', 'i3', 'i6',
                'i2', 'i4', 'i5', 'i7',
                'i1',
                'i8', 'i9', 'i10', 'i11', 'i12'
            ],
            'timestamp': pd.to_datetime([
                '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                '2023-01-01', '2023-01-02', '2023-01-03',
                '2023-02-01', '2023-02-02', '2023-02-03', '2023-02-04',
                '2023-03-01',
                '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14',
            ])
        }
        self.interactions_df = pd.DataFrame(data)

    def test_user_based_split(self):
        """Tests splitting by user, ensuring no user overlap between train and validation sets."""
        # Splits the data, allocating 60% of users to the training set.
        train_df, val_df = self.splitter.user_based_split(
            self.interactions_df, 
            train_ratio=0.6, 
            min_interactions_per_user=3
        )
        
        # Verifies that users with fewer than 3 interactions ('u4') are excluded.
        self.assertNotIn('u4', pd.concat([train_df, val_df])['user_id'].unique())

        train_users = set(train_df['user_id'].unique())
        val_users = set(val_df['user_id'].unique())
        
        # Verifies that the train and validation sets have distinct users.
        self.assertEqual(len(train_users.intersection(val_users)), 0)
        self.assertTrue(len(train_users) > 0)
        self.assertTrue(len(val_users) > 0)
        # 4 users have >= 3 interactions. 0.6 * 4 = 2.4 -> 2 users for train, 2 for val.
        self.assertEqual(len(train_users), 2)
        self.assertEqual(len(val_users), 2)

    def test_temporal_split(self):
        """Tests splitting by time, ensuring training data is older than validation data."""
        # Splits the data based on time, with 80% of interactions in the training set.
        train_df, val_df = self.splitter.temporal_split(self.interactions_df, train_ratio=0.8)

        # Verifies that the split ratio is approximately correct.
        self.assertAlmostEqual(len(train_df) / len(self.interactions_df), 0.8, delta=0.1)
        
        # Verifies that all training interactions occurred before or at the same time as the earliest validation interaction.
        self.assertTrue(train_df['timestamp'].max() <= val_df['timestamp'].min())

    def test_leave_one_out_split_random(self):
        """Tests the leave-one-out strategy, holding out one random item per user for validation."""
        # Filters for users with at least two interactions to ensure a meaningful split.
        filtered_df = self.interactions_df.groupby('user_id').filter(lambda x: len(x) > 1)
        train_df, val_df = self.splitter.leave_one_out_split(filtered_df, strategy='random')

        # Verifies that the validation set contains exactly one interaction for each user.
        self.assertTrue(all(val_df['user_id'].value_counts() == 1))
        # Verifies that the total number of interactions is conserved.
        self.assertEqual(len(train_df) + len(val_df), len(filtered_df))

    def test_leave_one_out_split_latest(self):
        """Tests the leave-one-out strategy, holding out the latest item per user."""
        filtered_df = self.interactions_df.groupby('user_id').filter(lambda x: len(x) > 1)
        train_df, val_df = self.splitter.leave_one_out_split(filtered_df, strategy='latest')
        
        # Finds the latest interaction for each user in the original data.
        latest_interactions = filtered_df.loc[filtered_df.groupby('user_id')['timestamp'].idxmax()]
        
        # Verifies that the validation set consists of exactly these latest interactions.
        pd.testing.assert_frame_equal(
            val_df.sort_values(by='user_id').reset_index(drop=True),
            latest_interactions.sort_values(by='user_id').reset_index(drop=True)
        )

    def test_stratified_split(self):
        """Tests stratified splitting, ensuring each user's interactions are split by the given ratio."""
        # Filters for users with enough interactions for a meaningful stratified split.
        min_interactions = 3
        filtered_df = self.interactions_df.groupby('user_id').filter(lambda x: len(x) >= min_interactions)
        
        train_df, val_df = self.splitter.stratified_split(
            filtered_df, 
            train_ratio=0.75, 
            min_interactions_per_user=min_interactions
        )
        
        # Verifies that all users from the filtered set are present in both train and validation sets.
        self.assertEqual(
            set(filtered_df['user_id'].unique()),
            set(train_df['user_id'].unique())
        )
        self.assertEqual(
            set(filtered_df['user_id'].unique()),
            set(val_df['user_id'].unique())
        )
        
        # Checks the interaction split for a specific user ('u1').
        u1_interactions_train = train_df[train_df['user_id'] == 'u1']
        u1_interactions_val = val_df[val_df['user_id'] == 'u1']
        
        # User 'u1' has 5 interactions. 75% of 5 is 3.75, so 3 should be in train, 2 in val.
        self.assertEqual(len(u1_interactions_train), 3)
        self.assertEqual(len(u1_interactions_val), 2)
        
    def test_get_split_statistics(self):
        """Tests the calculation of statistics for a given train/validation split."""
        # Creates dummy train and validation frames to test the statistics function.
        train_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['i1', 'i2']})
        val_df = pd.DataFrame({'user_id': ['u2', 'u3'], 'item_id': ['i2', 'i3']})
        
        stats = self.splitter.get_split_statistics(train_df, val_df)
        
        # Verifies all calculated statistics are correct.
        self.assertEqual(stats['train_interactions'], 2)
        self.assertEqual(stats['val_interactions'], 2)
        self.assertEqual(stats['train_users'], 2)
        self.assertEqual(stats['val_users'], 2)
        self.assertEqual(stats['user_overlap'], 1)  # User 'u2' overlaps
        self.assertEqual(stats['item_overlap'], 1)  # Item 'i2' overlaps
        self.assertAlmostEqual(stats['user_overlap_ratio'], 0.5)
        self.assertAlmostEqual(stats['item_overlap_ratio'], 0.5)

    def test_mixed_split(self):
        """Tests the mixed split for creating warm and cold start evaluation sets."""
        users = [f'u{i}' for i in range(1, 21)]  # 20 users
        items = [f'i{i}' for i in range(1, 51)]  # 50 items
        data = []
        for user in users:
            # Give users varying numbers of interactions
            # Make first 5 users "cold" (few interactions)
            num_interactions = np.random.randint(1, 4) if int(user[1:]) <= 5 else np.random.randint(5, 15)
            user_items = np.random.choice(items, num_interactions, replace=False)
            for item in user_items:
                data.append({'user_id': user, 'item_id': item})
        
        complex_df = pd.DataFrame(data)

        # Perform the mixed split
        splits = self.splitter.mixed_split(
            complex_df, 
            cold_user_ratio=0.25, # Cold users are the bottom 25% by activity
            cold_item_ratio=0.25,
            train_ratio=0.8
        )

        # Verify that all expected keys are in the output dictionary
        self.assertIn('train', splits)
        self.assertIn('val_warm', splits)
        self.assertIn('val_cold_user', splits)
        self.assertIn('val_cold_item', splits)
        self.assertIn('val_cold_both', splits)
        
        # Verify that the core train and validation sets are not empty
        self.assertFalse(splits['train'].empty)
        self.assertFalse(splits['val_warm'].empty)
        
        # Verify no overlap between the main train and warm validation interactions
        train_interactions = set(map(tuple, splits['train'][['user_id', 'item_id']].values))
        val_warm_interactions = set(map(tuple, splits['val_warm'][['user_id', 'item_id']].values))
        self.assertTrue(train_interactions.isdisjoint(val_warm_interactions), "Train and warm validation sets should not have overlapping interactions.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)