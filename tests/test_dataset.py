import unittest
import pandas as pd
import numpy as np
from src.data.dataset import MultimodalDataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'item_id': ['i1', 'i2', 'i1']
        })
        
        self.item_info_df = pd.DataFrame({
            'item_id': ['i1', 'i2'],
            'title': ['Item 1', 'Item 2'],
            'tag': ['tag1', 'tag2'],
            'description': ['desc1', 'desc2'],
            'view_number': [100, 200]
        })
    
    def test_dataset_creation(self):
        dataset = MultimodalDataset(
            self.interactions_df,
            self.item_info_df,
            'dummy_path',
            create_negative_samples=False
        )
        
        self.assertEqual(dataset.n_users, 2)
        self.assertEqual(dataset.n_items, 2)