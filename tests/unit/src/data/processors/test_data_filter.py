# tests/unit/test_data_filter.py
"""
Unit tests for the DataFilter processor.

This test suite verifies the functionality of the DataFilter class, ensuring
that it correctly filters interaction and item DataFrames based on various
criteria such as item validity, user activity, and item activity. It also
tests edge cases and utility functions within the class.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pytest

# Adds the project's root directory to the system path.
# This allows the test suite to import modules from the 'src' directory,
# such as the DataFilter class that is being tested.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.processors.data_filter import DataFilter


class TestDataFilter(unittest.TestCase):
    """A collection of test cases for the DataFilter class."""
    
    def setUp(self):
        """
        Initializes test fixtures before each test method is run.
        
        This method creates sample DataFrames for item metadata and user-item
        interactions, which serve as the common inputs for the tests. It also
        instantiates the DataFilter class.
        """
        self.data_filter = DataFilter()
        
        # A sample DataFrame representing item metadata.
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4', 'item5'],
            'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4', 'Title 5'],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        # A sample DataFrame representing user-item interactions.
        self.interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2', 'user3', 'user3', 'user4'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item2', 'item4', 'item1']
        })
        
    def test_filter_interactions_by_valid_items(self):
        """
        Tests the filtering of an interactions DataFrame to include only a
        specified set of valid items.
        """
        # Defines a set of item IDs that should be retained.
        valid_item_ids = {'item1', 'item2', 'item3'}
        
        # Executes the filtering operation.
        filtered_df = self.data_filter.filter_interactions_by_valid_items(
            self.interactions_df, valid_item_ids
        )
        
        # Asserts that interactions with non-valid items (e.g., 'item4') are removed.
        self.assertEqual(len(filtered_df), 6)
        self.assertTrue(all(item in valid_item_ids for item in filtered_df['item_id']))
        self.assertNotIn('item4', filtered_df['item_id'].values)
        
    def test_filter_interactions_empty_valid_items(self):
        """
        Tests the behavior of the filtering logic when the set of valid
        items is empty, expecting an empty DataFrame as output.
        """
        valid_item_ids = set()
        
        filtered_df = self.data_filter.filter_interactions_by_valid_items(
            self.interactions_df, valid_item_ids
        )
        
        self.assertEqual(len(filtered_df), 0)
        
    def test_filter_by_activity_min_user_interactions(self):
        """
        Tests filtering the interactions DataFrame to remove users with fewer
        than a specified number of interactions.
        """
        # Executes filtering, keeping users with at least 2 interactions.
        filtered_df = self.data_filter.filter_by_activity(
            self.interactions_df,
            min_user_interactions=2,
            min_item_interactions=0
        )
        
        # Asserts that 'user4', who has only one interaction, is removed.
        self.assertEqual(len(filtered_df), 6)
        self.assertNotIn('user4', filtered_df['user_id'].values)
        self.assertEqual(filtered_df['user_id'].nunique(), 3)
        
    def test_filter_by_activity_min_item_interactions(self):
        """
        Tests filtering the interactions DataFrame to remove items with fewer
        than a specified number of interactions.
        """
        # Executes filtering, keeping items with at least 2 interactions.
        filtered_df = self.data_filter.filter_by_activity(
            self.interactions_df,
            min_user_interactions=0,
            min_item_interactions=2
        )
        
        # Asserts that 'item3' and 'item4', which have one interaction each, are removed.
        self.assertEqual(len(filtered_df), 5)
        self.assertNotIn('item3', filtered_df['item_id'].values)
        self.assertNotIn('item4', filtered_df['item_id'].values)
        
    def test_filter_by_activity_combined(self):
        """
        Tests the sequential filtering by both minimum item and user
        activity levels.
        """
        filtered_df = self.data_filter.filter_by_activity(
            self.interactions_df,
            min_user_interactions=2,
            min_item_interactions=2
        )
        
        # Defines the expected interactions after the combined filtering logic.
        # Step 1 (Item Filter): Removes items with < 2 interactions ('item3', 'item4').
        # Step 2 (User Filter): On the remaining data, removes users with < 2 interactions.
        # Only user1 remains with 2 interactions after Step 1.
        expected_interactions = [
            ('user1', 'item1'), ('user1', 'item2')
        ]
        
        self.assertEqual(len(filtered_df), 2)
        actual_interactions = list(zip(filtered_df['user_id'], filtered_df['item_id']))
        self.assertEqual(set(actual_interactions), set(expected_interactions))
        
    def test_align_item_info_with_interactions(self):
        """
        Tests that the item metadata DataFrame is correctly filtered to
        contain only items present in the interactions DataFrame.
        """
        # Executes the alignment operation.
        aligned_df = self.data_filter.align_item_info_with_interactions(
            self.item_info_df,
            self.interactions_df
        )
        
        # Asserts that 'item5', which has no interactions, is removed from the item metadata.
        self.assertEqual(len(aligned_df), 4)
        self.assertNotIn('item5', aligned_df['item_id'].values)
        self.assertEqual(set(aligned_df['item_id']), {'item1', 'item2', 'item3', 'item4'})
        
    def test_get_filtering_stats(self):
        """
        Tests the calculation of filtering statistics, such as original and
        filtered counts and retention rates.
        """
        # Creates mock filtered DataFrames for the statistics calculation.
        filtered_interactions = self.interactions_df.iloc[:5]
        filtered_items = self.item_info_df.iloc[:3]
        
        # Calculates the statistics.
        stats = self.data_filter.get_filtering_stats(
            self.interactions_df,
            filtered_interactions,
            self.item_info_df,
            filtered_items
        )
        
        # Asserts that the statistics dictionary has the correct structure and values.
        self.assertIn('interactions', stats)
        self.assertIn('users', stats)
        self.assertIn('items', stats)
        
        self.assertEqual(stats['interactions']['original'], 7)
        self.assertEqual(stats['interactions']['filtered'], 5)
        self.assertAlmostEqual(stats['interactions']['retention_rate'], 5/7)
        
        self.assertEqual(stats['items']['original'], 5)
        self.assertEqual(stats['items']['filtered'], 3)
        self.assertAlmostEqual(stats['items']['retention_rate'], 3/5)
        
    def test_string_type_consistency(self):
        """
        Tests that the filtering functions are robust to mixed data types in
        ID columns by ensuring consistent string conversion.
        """
        # Creates a DataFrame with integer user IDs.
        mixed_interactions = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': ['item1', 'item2', 'item3']
        })
        
        valid_items = {'item1', 'item2'}
        
        # The filtering function should handle the type conversion internally.
        filtered_df = self.data_filter.filter_interactions_by_valid_items(
            mixed_interactions, valid_items
        )
        
        self.assertEqual(len(filtered_df), 2)
        self.assertIn('item1', filtered_df['item_id'].values)
        self.assertIn('item2', filtered_df['item_id'].values)
        
    def test_empty_dataframe_handling(self):
        """
        Tests that all filtering methods handle empty DataFrames gracefully
        without raising errors.
        """
        empty_df = pd.DataFrame(columns=['user_id', 'item_id'])
        
        # Tests filtering by valid items with an empty input DataFrame.
        filtered = self.data_filter.filter_interactions_by_valid_items(
            empty_df, {'item1'}
        )
        self.assertEqual(len(filtered), 0)
        
        # Tests filtering by activity with an empty input DataFrame.
        filtered = self.data_filter.filter_by_activity(empty_df, 1, 1)
        self.assertEqual(len(filtered), 0)
        
        # Tests aligning item info with an empty interactions DataFrame.
        aligned = self.data_filter.align_item_info_with_interactions(
            self.item_info_df, empty_df
        )
        self.assertEqual(len(aligned), 0)


# This block allows the script to be executed directly, running all tests
# in the 'tests/' directory using the pytest framework.
if __name__ == "__main__":
    # Specifies the target directory for pytest to discover and run tests.
    args = ['tests/']
    
    # Executes pytest with the specified arguments and exits with the resulting status code.
    sys.exit(pytest.main(args))