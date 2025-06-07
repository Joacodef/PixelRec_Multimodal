# tests/unit/test_data_filter.py
"""
Unit tests for the DataFilter class
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.processors.data_filter import DataFilter


class TestDataFilter(unittest.TestCase):
    """Test cases for DataFilter functionality"""
    
    def setUp(self):
        """Set up test data before each test"""
        self.data_filter = DataFilter()
        
        # Create sample item info DataFrame
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4', 'item5'],
            'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4', 'Title 5'],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        # Create sample interactions DataFrame
        self.interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2', 'user3', 'user3', 'user4'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item2', 'item4', 'item1']
        })
        
    def test_filter_interactions_by_valid_items(self):
        """Test filtering interactions by valid items"""
        # Define valid items (exclude item5)
        valid_item_ids = {'item1', 'item2', 'item3'}
        
        # Filter interactions
        filtered_df = self.data_filter.filter_interactions_by_valid_items(
            self.interactions_df, valid_item_ids
        )
        
        # Check results
        self.assertEqual(len(filtered_df), 6)  # item4 interaction should be removed
        self.assertTrue(all(item in valid_item_ids for item in filtered_df['item_id']))
        self.assertNotIn('item4', filtered_df['item_id'].values)
        
    def test_filter_interactions_empty_valid_items(self):
        """Test filtering with empty valid items set"""
        valid_item_ids = set()
        
        filtered_df = self.data_filter.filter_interactions_by_valid_items(
            self.interactions_df, valid_item_ids
        )
        
        self.assertEqual(len(filtered_df), 0)
        
    def test_filter_by_activity_min_user_interactions(self):
        """Test filtering by minimum user interactions"""
        # user1: 2 interactions, user2: 2 interactions, user3: 2 interactions, user4: 1 interaction
        filtered_df = self.data_filter.filter_by_activity(
            self.interactions_df,
            min_user_interactions=2,
            min_item_interactions=0
        )
        
        # user4 should be filtered out
        self.assertEqual(len(filtered_df), 6)
        self.assertNotIn('user4', filtered_df['user_id'].values)
        self.assertEqual(filtered_df['user_id'].nunique(), 3)
        
    def test_filter_by_activity_min_item_interactions(self):
        """Test filtering by minimum item interactions"""
        # item1: 3, item2: 2, item3: 1, item4: 1
        filtered_df = self.data_filter.filter_by_activity(
            self.interactions_df,
            min_user_interactions=0,
            min_item_interactions=2
        )
        
        # item3 and item4 should be filtered out
        self.assertEqual(len(filtered_df), 5)
        self.assertNotIn('item3', filtered_df['item_id'].values)
        self.assertNotIn('item4', filtered_df['item_id'].values)
        
    def test_filter_by_activity_combined(self):
        """Test filtering by both user and item activity"""
        filtered_df = self.data_filter.filter_by_activity(
            self.interactions_df,
            min_user_interactions=2,
            min_item_interactions=2
        )
        
        # After filtering items: only item1 and item2 remain (item3, item4 removed)
        # This leaves: user1→[item1,item2], user2→[item1], user3→[item2], user4→[item1]
        # After filtering users: only users with 2+ interactions remain
        # Only user1 still has 2 interactions after item filtering!
        expected_interactions = [
            ('user1', 'item1'), ('user1', 'item2')
        ]
        
        self.assertEqual(len(filtered_df), 2)
        actual_interactions = list(zip(filtered_df['user_id'], filtered_df['item_id']))
        self.assertEqual(set(actual_interactions), set(expected_interactions))
        
    def test_align_item_info_with_interactions(self):
        """Test aligning item info with interactions"""
        # Only items 1-4 appear in interactions, not item5
        aligned_df = self.data_filter.align_item_info_with_interactions(
            self.item_info_df,
            self.interactions_df
        )
        
        self.assertEqual(len(aligned_df), 4)
        self.assertNotIn('item5', aligned_df['item_id'].values)
        self.assertEqual(set(aligned_df['item_id']), {'item1', 'item2', 'item3', 'item4'})
        
    def test_get_filtering_stats(self):
        """Test filtering statistics calculation"""
        # Create filtered versions
        filtered_interactions = self.interactions_df.iloc[:5]  # Remove 2 interactions
        filtered_items = self.item_info_df.iloc[:3]  # Remove 2 items
        
        stats = self.data_filter.get_filtering_stats(
            self.interactions_df,
            filtered_interactions,
            self.item_info_df,
            filtered_items
        )
        
        # Check structure
        self.assertIn('interactions', stats)
        self.assertIn('users', stats)
        self.assertIn('items', stats)
        
        # Check values
        self.assertEqual(stats['interactions']['original'], 7)
        self.assertEqual(stats['interactions']['filtered'], 5)
        self.assertAlmostEqual(stats['interactions']['retention_rate'], 5/7)
        
        self.assertEqual(stats['items']['original'], 5)
        self.assertEqual(stats['items']['filtered'], 3)
        self.assertAlmostEqual(stats['items']['retention_rate'], 3/5)
        
    def test_string_type_consistency(self):
        """Test that string types are handled consistently"""
        # Create data with mixed types
        mixed_interactions = pd.DataFrame({
            'user_id': [1, 2, 3],  # integers
            'item_id': ['item1', 'item2', 'item3']  # strings
        })
        
        valid_items = {'item1', 'item2'}  # strings
        
        # Filter should handle type conversion
        filtered_df = self.data_filter.filter_interactions_by_valid_items(
            mixed_interactions, valid_items
        )
        
        self.assertEqual(len(filtered_df), 2)
        self.assertIn('item1', filtered_df['item_id'].values)
        self.assertIn('item2', filtered_df['item_id'].values)
        
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame(columns=['user_id', 'item_id'])
        
        # Test filter by valid items
        filtered = self.data_filter.filter_interactions_by_valid_items(
            empty_df, {'item1'}
        )
        self.assertEqual(len(filtered), 0)
        
        # Test filter by activity
        filtered = self.data_filter.filter_by_activity(empty_df, 1, 1)
        self.assertEqual(len(filtered), 0)
        
        # Test align item info
        aligned = self.data_filter.align_item_info_with_interactions(
            self.item_info_df, empty_df
        )
        self.assertEqual(len(aligned), 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)