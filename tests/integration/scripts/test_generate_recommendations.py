import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import pickle
import json
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add parent directory to path to import src and scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.generate_recommendations import main as generate_main
from src.config import Config
from src.models.multimodal import MultimodalRecommender

class TestGenerateRecommendationsScript(unittest.TestCase):
    """Test cases for the generate_recommendations.py script."""

    def setUp(self):
        """Set up a temporary environment for testing."""
        self.test_dir = Path("test_temp_generate_recs")
        self.test_dir.mkdir(exist_ok=True)

        # Define and create necessary directories
        self.processed_data_dir = self.test_dir / "data" / "processed"
        self.image_dir = self.test_dir / "data" / "images"
        self.checkpoints_dir = self.test_dir / "models" / "checkpoints"
        self.model_specific_checkpoint_dir = self.checkpoints_dir / "resnet_sentence-bert"
        self.encoders_dir = self.checkpoints_dir / "encoders"
        self.results_dir = self.test_dir / "results"
        self.configs_dir = self.test_dir / "configs"

        for d in [self.processed_data_dir, self.image_dir,
                    self.model_specific_checkpoint_dir, self.encoders_dir,
                    self.results_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- Create Dummy Data and Files ---

        # 1. Dummy "processed" item info
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4', 'item5', 'item6'],
            'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4', 'Title 5', 'Title 6'],
            'description': ['Desc 1', 'Desc 2', 'Desc 3', 'Desc 4', 'Desc 5', 'Desc 6'],
            'tag': ['A', 'B', 'A', 'C', 'D', 'A'],
            'view_number': [100, 200, 50, 150, 110, 220],
            'comment_number': [10, 20, 5, 15, 12, 22]
        })
        self.processed_item_info_path = self.processed_data_dir / "item_info.csv"
        self.item_info_df.to_csv(self.processed_item_info_path, index=False)

        # 2. Dummy "processed" interactions 
        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u3'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item5', 'item6']
        })
        self.processed_interactions_path = self.processed_data_dir / "interactions.csv"
        self.interactions_df.to_csv(self.processed_interactions_path, index=False)

        # 3. Dummy Encoders
        self.user_encoder = LabelEncoder().fit(self.interactions_df['user_id'])
        self.item_encoder = LabelEncoder().fit(self.interactions_df['item_id'])
        with open(self.encoders_dir / "user_encoder.pkl", 'wb') as f:
            pickle.dump(self.user_encoder, f)
        with open(self.encoders_dir / "item_encoder.pkl", 'wb') as f:
            pickle.dump(self.item_encoder, f)

        # 4. Dummy Numerical Scaler
        self.numerical_cols = ['view_number', 'comment_number']
        scaler = StandardScaler().fit(self.item_info_df[self.numerical_cols])
        self.scaler_path = self.processed_data_dir / "numerical_scaler.pkl"
        with open(self.scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'columns': self.numerical_cols}, f)

        # 5. Dummy Model and Checkpoint
        self.model = MultimodalRecommender(
            n_users=len(self.user_encoder.classes_),
            n_items=len(self.item_encoder.classes_),
            num_numerical_features=len(self.numerical_cols),
            embedding_dim=8,
            vision_model_name='resnet',
            language_model_name='sentence-bert'
        )
        self.checkpoint_path = self.model_specific_checkpoint_dir / "best_model.pth"
        torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)

        # 6. Dummy Config File
        self.config_path = self.configs_dir / "test_config_generate.yaml"
        config_content = f"""
model:
  vision_model: resnet
  language_model: sentence-bert
  embedding_dim: 8

data:
  processed_item_info_path: {self.processed_item_info_path.as_posix()}
  processed_interactions_path: {self.processed_interactions_path.as_posix()}
  image_folder: {self.image_dir.as_posix()}
  scaler_path: {self.scaler_path.as_posix()}
  numerical_features_cols: {self.numerical_cols}
  cache_config:
    enabled: false

recommendation:
  top_k: 3

checkpoint_dir: {self.checkpoints_dir.as_posix()}
results_dir: {self.results_dir.as_posix()}
"""
        self.config_path.write_text(config_content)
        
        # 7. Dummy user file for testing
        self.user_file_path = self.test_dir / "users.txt"
        with open(self.user_file_path, "w") as f:
            f.write("u1\n")
            f.write("u2\n")

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_recommendation_generation_specific_user(self):
        """Test generating recommendations for a single user specified via CLI."""
        original_argv = sys.argv
        output_filename = "recs_specific_user.json"
        
        # Mock sys.argv to simulate command-line execution
        sys.argv = [
            'scripts/generate_recommendations.py',
            '--config', str(self.config_path),
            '--users', 'u1',
            '--output', output_filename
        ]

        # Run the main function from the script
        generate_main()

        # --- Verification ---
        output_path = self.results_dir / output_filename
        self.assertTrue(output_path.exists(), "Output JSON file was not created.")

        with open(output_path, 'r') as f:
            results = json.load(f)

        self.assertIn('u1', results)
        self.assertIsInstance(results['u1'], dict)
        self.assertIn('recommendations', results['u1'])
        
        recs = results['u1']['recommendations']
        self.assertIsInstance(recs, list)
        self.assertEqual(len(recs), 3, "Should generate top_k=3 recommendations as per config.")
        
        # Check the structure of a single recommendation
        if recs:
            self.assertIn('item_id', recs[0])
            self.assertIn('score', recs[0])
            self.assertIsInstance(recs[0]['item_id'], str)
            self.assertIsInstance(recs[0]['score'], float)

        # Restore original argv
        sys.argv = original_argv

    def test_recommendation_generation_from_file(self):
        """Test generating recommendations for users specified in a file."""
        original_argv = sys.argv
        output_filename = "recs_from_file.json"

        sys.argv = [
            'scripts/generate_recommendations.py',
            '--config', str(self.config_path),
            '--user_file', str(self.user_file_path),
            '--output', output_filename
        ]

        generate_main()
        
        # --- Verification ---
        output_path = self.results_dir / output_filename
        self.assertTrue(output_path.exists())

        with open(output_path, 'r') as f:
            results = json.load(f)

        # Check for both users from the file
        self.assertIn('u1', results)
        self.assertIn('u2', results)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results['u2']['recommendations']), 3)

        sys.argv = original_argv
        
    def test_recommendation_generation_sample_users(self):
        """Test generating recommendations for a random sample of users."""
        original_argv = sys.argv
        output_filename = "recs_from_sample.json"

        sys.argv = [
            'scripts/generate_recommendations.py',
            '--config', str(self.config_path),
            '--sample_users', '2',
            '--output', output_filename
        ]

        generate_main()
        
        # --- Verification ---
        output_path = self.results_dir / output_filename
        self.assertTrue(output_path.exists())

        with open(output_path, 'r') as f:
            results = json.load(f)

        # Check that it generated recommendations for the specified number of users
        self.assertEqual(len(results), 2)
        
        # Check that the users are from the original user list
        self.assertTrue(all(user in self.user_encoder.classes_ for user in results.keys()))

        sys.argv = original_argv

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)