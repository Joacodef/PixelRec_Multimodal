import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import pickle
import json
import torch
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.evaluate import main as evaluate_main
from src.config import Config
from src.models.multimodal import MultimodalRecommender
# Import the actual evaluator to test it directly
from src.evaluation.tasks import TopKRetrievalEvaluator

# A self-contained mock recommender for predictable metric testing
class MockRecommender:
    def __init__(self, history_df):
        self.user_history = history_df.groupby('user_id')['item_id'].apply(set).to_dict()

    def get_recommendations(self, user_id, top_k, filter_seen=True, candidates=None):
        all_recs = [('item4', 0.9), ('item2', 0.8), ('item1', 0.7), ('item3', 0.6)]
        if filter_seen:
            seen_items = self.user_history.get(user_id, set())
            unseen_recs = [(item, score) for item, score in all_recs if item not in seen_items]
            return unseen_recs[:top_k]
        return all_recs[:top_k]

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

        # Create dummy data and files
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4'],
            'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4'],'description': ['Desc 1', 'Desc 2', 'Desc 3', 'Desc 4'],
            'tag': ['A', 'B', 'A', 'C'], 'view_number': [100, 200, 50, 150], 'comment_number': [10, 20, 5, 15]
        })
        self.processed_item_info_path = self.processed_data_dir / "item_info.csv"
        self.item_info_df.to_csv(self.processed_item_info_path, index=False)

        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u3'], 'item_id': ['item1', 'item2', 'item1', 'item3', 'item4']
        })
        self.processed_interactions_path = self.processed_data_dir / "interactions.csv"
        self.interactions_df.to_csv(self.processed_interactions_path, index=False)
        
        self.train_df = pd.DataFrame({'user_id': ['u1'], 'item_id': ['item1']})
        self.test_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['item2', 'item3']})
        self.train_path = self.splits_dir / "train.csv"
        self.test_path = self.splits_dir / "test.csv"
        self.train_df.to_csv(self.train_path, index=False)
        self.test_df.to_csv(self.test_path, index=False)

        self.user_encoder = LabelEncoder().fit(self.interactions_df['user_id'])
        self.item_encoder = LabelEncoder().fit(self.item_info_df['item_id'])
        with open(self.encoders_dir / "user_encoder.pkl", 'wb') as f: pickle.dump(self.user_encoder, f)
        with open(self.encoders_dir / "item_encoder.pkl", 'wb') as f: pickle.dump(self.item_encoder, f)

        self.numerical_cols = ['view_number', 'comment_number']
        scaler = StandardScaler().fit(self.item_info_df[self.numerical_cols])
        self.scaler_path = self.processed_data_dir / "numerical_scaler.pkl"
        with open(self.scaler_path, 'wb') as f: pickle.dump({'scaler': scaler, 'columns': self.numerical_cols}, f)

        self.config_path = self.configs_dir / "test_config_evaluate.yaml"
        config_content = f"""
model:
  vision_model: resnet
  language_model: sentence-bert
  embedding_dim: 8
  use_contrastive: true
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
        
        # ** FIX **: Instantiate the dummy model using all relevant parameters from the config
        # to ensure the architecture matches the one loaded by the script.
        config = Config.from_yaml(str(self.config_path))
        model_for_checkpoint = MultimodalRecommender(
            n_users=len(self.user_encoder.classes_),
            n_items=len(self.item_encoder.classes_),
            num_numerical_features=len(self.numerical_cols),
            embedding_dim=config.model.embedding_dim,
            vision_model_name=config.model.vision_model,
            language_model_name=config.model.language_model,
            use_contrastive=config.model.use_contrastive
        )
        self.checkpoint_path = self.model_specific_checkpoint_dir / "best_model.pth"
        torch.save({'model_state_dict': model_for_checkpoint.state_dict()}, self.checkpoint_path)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_evaluation_logic_with_mock_recommender(self):
        """
        Tests the core metric calculation logic directly by-passing the script's main
        function and using a mock recommender with predictable outputs.
        """
        # --- Setup Mocks and Evaluator ---
        mock_recommender = MockRecommender(history_df=self.train_df)
        config = Config.from_yaml(str(self.config_path))
        
        # Instantiate the evaluator directly to test its logic
        evaluator = TopKRetrievalEvaluator(
            recommender=mock_recommender,
            test_data=self.test_df,
            config=config,
            use_sampling=False, # Use full evaluation for this small test case
            num_workers=0
        )

        # --- Run Evaluation ---
        results = evaluator.evaluate()

        # --- Verification ---
        # User u1: history={'item1'}, relevant={'item2'}. Mock recs are ['item4', 'item2', 'item1', 'item3']. 
        # After filtering 'item1', gets ['item4', 'item2'] for K=2.
        # Precision@2 = 1/2 = 0.5. Recall@2 = 1/1 = 1.0. MRR = 1/2 = 0.5.
        #
        # User u2: history={}, relevant={'item3'}. Gets ['item4', 'item2'] for K=2.
        # Precision@2 = 0/2 = 0.0. Recall@2 = 0/1 = 0.0. MRR = 0.
        #
        # Averages: Precision=0.25, Recall=0.5, MRR=0.25
        self.assertAlmostEqual(results['avg_precision_at_k'], 0.25, places=4)
        self.assertAlmostEqual(results['avg_recall_at_k'], 0.5, places=4)
        self.assertAlmostEqual(results['avg_mrr'], 0.25, places=4)
        self.assertEqual(results['num_users_evaluated'], self.test_df['user_id'].nunique())

    def test_evaluation_pipeline_multimodal_retrieval(self):
        """
        Smoke test for the full evaluation pipeline. This ensures the script can
        load a real (but dummy) checkpoint and run without crashing.
        """
        original_argv = sys.argv
        sys.argv = [
            'scripts/evaluate.py',
            '--config', str(self.config_path),
            '--test_data', str(self.test_path),
            '--train_data', str(self.train_path),
            '--recommender_type', 'multimodal',
            '--eval_task', 'retrieval',
            '--num_workers', '0'
        ]
        
        # This should now pass because setUp creates a compatible checkpoint
        evaluate_main()
        
        output_file = self.results_dir / 'evaluation_results.json'
        self.assertTrue(output_file.exists())
        with open(output_file, 'r') as f:
            results = json.load(f)
        self.assertIn('avg_precision_at_k', results)
        self.assertGreaterEqual(results['avg_precision_at_k'], 0.0)
        self.assertEqual(results['num_users_evaluated'], self.test_df['user_id'].nunique())
        
        sys.argv = original_argv

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)