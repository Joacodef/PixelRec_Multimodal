import unittest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import shutil
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from src.data.processors.numerical_processor import NumericalProcessor

class TestNumericalProcessor(unittest.TestCase):
    """Unit tests for the NumericalProcessor class."""

    def setUp(self):
        """Set up a sample DataFrame and a temporary directory."""
        self.processor = NumericalProcessor()
        self.data = {
            'view_count': [100, 200, 300, 400, 500],
            'comment_count': [10, 20, np.nan, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A']
        }
        self.df = pd.DataFrame(self.data)
        self.numerical_cols = ['view_count', 'comment_count']
        
        # Temporary directory for saving scaler
        self.test_dir = Path("test_temp_numerical_processor")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_fit_scaler(self):
        """Tests that a scaler can be fitted correctly."""
        scaler = self.processor.fit_scaler(self.df, self.numerical_cols, 'standardization')
        self.assertIsNotNone(self.processor.scaler)
        self.assertIsInstance(self.processor.scaler, StandardScaler)
        self.assertEqual(self.processor.fitted_columns, self.numerical_cols)
        self.assertEqual(scaler, self.processor.scaler)

    def test_transform_features_with_standardization(self):
        """Tests feature transformation using a standard scaler."""
        self.processor.fit_scaler(self.df, self.numerical_cols, 'standardization')
        _, transformed = self.processor.transform_features(self.df, self.numerical_cols, 'standardization')
        
        self.assertEqual(transformed.shape, (5, 2))
        # The mean of a standardized feature should be close to 0
        self.assertTrue(np.allclose(transformed.mean(axis=0), [0., 0.]))
        # The std dev should be close to 1
        self.assertTrue(np.allclose(transformed.std(axis=0), [1., 1.]))

    def test_transform_features_with_min_max(self):
        """Tests feature transformation using a min-max scaler."""
        self.processor.fit_scaler(self.df, self.numerical_cols, 'min_max')
        _, transformed = self.processor.transform_features(self.df, self.numerical_cols, 'min_max')

        # Min-max scaled values should be between 0 and 1
        self.assertTrue(np.all(transformed >= 0))
        self.assertTrue(np.all(transformed <= 1))

    def test_log1p_transform(self):
        """Tests the log1p transformation."""
        _, transformed = self.processor.transform_features(self.df, self.numerical_cols, 'log1p')
        expected = np.log1p(self.df[self.numerical_cols].fillna(0).values)
        self.assertTrue(np.allclose(transformed, expected))
    
    def test_save_and_load_scaler(self):
        """Tests that a scaler can be saved to and loaded from a file."""
        scaler_path = self.test_dir / "scaler.pkl"
        
        # Fit, transform, and save
        self.processor.fit_scaler(self.df, self.numerical_cols, 'standardization')
        _, original_transformed = self.processor.transform_features(self.df, self.numerical_cols, 'standardization')
        save_success = self.processor.save_scaler(scaler_path)
        self.assertTrue(save_success)
        self.assertTrue(scaler_path.exists())
        
        # Create a new processor, load, and transform
        new_processor = NumericalProcessor()
        load_success = new_processor.load_scaler(scaler_path)
        self.assertTrue(load_success)
        self.assertIsNotNone(new_processor.scaler)
        self.assertEqual(new_processor.fitted_columns, self.numerical_cols)
        
        _, new_transformed = new_processor.transform_features(self.df, self.numerical_cols, 'standardization')
        
        # The results should be identical
        self.assertTrue(np.allclose(original_transformed, new_transformed))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)