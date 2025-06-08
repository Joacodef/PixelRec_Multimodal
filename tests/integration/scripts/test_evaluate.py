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

from scripts.evaluate import main as evaluate_main
from src.config import Config
from src.models.multimodal import MultimodalRecommender

class TestEvaluateScript(unittest.TestCase):
    """Test cases for the evaluate.py script."""

    def setUp(self):
        """Set up a temporary environment with dummy data, configs, and models."""
        self.test_dir = Path("test_temp_evaluate")
        self.test_dir.mkdir(exist_ok=True)

        # Define and create directories
        self.processed_data_dir = self.test_dir / "data" / "processed"
        self.splits_dir = self.test_dir / "data" / "splits" / "test_split"
        self.image_dir = self.test_dir / "data" / "images"
        self.checkpoints_dir = self.test_dir / "models" / "checkpoints"
        self.model_specific_checkpoint_dir = self.checkpoints_dir / "resnet_sentence-bert"
        self.encoders_dir = self.checkpoints_dir / "encoders"
        self.results_dir = self.test_dir / "results"
        self.configs_dir = self.test_dir / "configs"

        for d in [self.processed_data_dir, self.splits_dir, self.image_dir,
                    self.model_specific_checkpoint_dir, self.encoders_dir,
                    self.results_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- Create Dummy Data and Files ---

        # 1. Dummy "processed" item info
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4'],
            'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4'],
            'description': ['Desc 1', 'Desc 2', 'Desc 3', 'Desc 4'],
            'tag': ['A', 'B', 'A', 'C'],
            'view_number': [100, 200, 50, 150],
            'comment_number': [10, 20, 5, 15]
        })
        self.processed_item_info_path = self.processed_data_dir / "item_info.csv"
        self.item_info_df.to_csv(self.processed_item_info_path, index=False)

        # 2. Dummy "processed" interactions (for fitting encoders)
        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u3'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item4']
        })
        self.processed_interactions_path = self.processed_data_dir / "interactions.csv"
        self.interactions_df.to_csv(self.processed_interactions_path, index=False)

        # 3. Dummy "split" data for evaluation
        self.train_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['item1', 'item1']})
        self.test_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['item2', 'item3']})
        self.train_path = self.splits_dir / "train.csv"
        self.test_path = self.splits_dir / "test.csv"
        self.train_df.to_csv(self.train_path, index=False)
        self.test_df.to_csv(self.test_path, index=False)

        # 4. Dummy Encoders
        self.user_encoder = LabelEncoder().fit(self.interactions_df['user_id'])
        self.item_encoder = LabelEncoder().fit(self.interactions_df['item_id'])
        with open(self.encoders_dir / "user_encoder.pkl", 'wb') as f:
            pickle.dump(self.user_encoder, f)
        with open(self.encoders_dir / "item_encoder.pkl", 'wb') as f:
            pickle.dump(self.item_encoder, f)

        # 5. Dummy Numerical Scaler
        self.numerical_cols = ['view_number', 'comment_number']
        scaler = StandardScaler().fit(self.item_info_df[self.numerical_cols])
        self.scaler_path = self.processed_data_dir / "numerical_scaler.pkl"
        with open(self.scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'columns': self.numerical_cols}, f)

        # 6. Dummy Model and Checkpoint
        # Ensure the model used to create the checkpoint has the SAME architecture
        # as the one defined in the test YAML config file.
        self.model = MultimodalRecommender(
            n_users=len(self.user_encoder.classes_),
            n_items=len(self.item_encoder.classes_),
            num_numerical_features=len(self.numerical_cols),
            embedding_dim=8,
            vision_model_name='resnet',             # This must match the test config
            language_model_name='sentence-bert'   # This must match the test config
        )
        self.checkpoint_path = self.model_specific_checkpoint_dir / "best_model.pth"
        torch.save({'model_state_dict': self.model.state_dict()}, self.checkpoint_path)

        # 7. Dummy Config File
        self.config_path = self.configs_dir / "test_config_evaluate.yaml"
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
  top_k: 2

checkpoint_dir: {self.checkpoints_dir.as_posix()}
results_dir: {self.results_dir.as_posix()}
"""
        self.config_path.write_text(config_content)

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_evaluation_pipeline_multimodal_retrieval(self):
        """Test the full evaluation pipeline for a multimodal recommender on a retrieval task."""
        # Setup command-line arguments for the script
        original_argv = sys.argv
        sys.argv = [
            'scripts/evaluate.py',
            '--config', str(self.config_path),
            '--test_data', str(self.test_path),
            '--train_data', str(self.train_path),
            '--recommender_type', 'multimodal',
            '--eval_task', 'retrieval',
            '--num_workers', '0' # Use 0 workers for simplicity in testing
        ]

        # Run the main function of the script
        evaluate_main()

        # Verify that the results file was created
        output_file = self.results_dir / 'evaluation_results.json'
        self.assertTrue(output_file.exists())

        # Load and check the results
        with open(output_file, 'r') as f:
            results = json.load(f)

        self.assertIn('avg_precision_at_k', results)
        self.assertIn('avg_recall_at_k', results)
        self.assertIn('avg_ndcg_at_k', results)
        self.assertIn('avg_mrr', results)
        self.assertGreaterEqual(results['avg_precision_at_k'], 0.0)
        self.assertLessEqual(results['avg_precision_at_k'], 1.0)
        self.assertEqual(results['num_users_evaluated'], self.test_df['user_id'].nunique())

        # Restore original argv
        sys.argv = original_argv

    def test_evaluation_pipeline_baseline_popularity(self):
        """Test the evaluation pipeline for a baseline popularity recommender."""
        # Setup command-line arguments
        original_argv = sys.argv
        sys.argv = [
            'scripts/evaluate.py',
            '--config', str(self.config_path),
            '--test_data', str(self.test_path),
            '--train_data', str(self.train_path), # Popularity model uses train data
            '--recommender_type', 'popularity',
            '--eval_task', 'retrieval',
            '--num_workers', '0'
        ]

        # Run the script's main function
        evaluate_main()

        # Verify output
        output_file = self.results_dir / 'evaluation_results.json'
        self.assertTrue(output_file.exists())
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        self.assertIn('avg_precision_at_k', results)
        self.assertEqual(results['evaluation_metadata']['recommender_type'], 'popularity')

        # Restore original argv
        sys.argv = original_argv

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)