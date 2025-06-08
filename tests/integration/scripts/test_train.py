import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import json
import torch
from PIL import Image

# Add parent directory to path to import src and scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.train import main as train_main
from src.config import Config # Used to load config for model instantiation
from src.models.multimodal import MultimodalRecommender # To inspect model weights

class TestTrainScript(unittest.TestCase):
    """Integration test for the train.py script."""

    def setUp(self):
        """Set up a temporary, self-contained environment for a training run."""
        self.test_dir = Path("test_temp_train")
        self.test_dir.mkdir(exist_ok=True)

        # Define and create all necessary directories
        self.processed_data_dir = self.test_dir / "data" / "processed"
        self.splits_dir = self.test_dir / "data" / "splits" / "tiny_split"
        self.raw_image_dir = self.test_dir / "data" / "raw" / "images"
        self.checkpoints_dir = self.test_dir / "models" / "checkpoints"
        self.results_dir = self.test_dir / "results"
        self.configs_dir = self.test_dir / "configs"

        for d in [self.processed_data_dir, self.splits_dir, self.raw_image_dir,
                    self.checkpoints_dir, self.results_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- Create Dummy Data, Images, and Config ---

        # 1. Dummy "processed" item info and interactions (for encoder fitting)
        self.item_info_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3', 'item4'],
            'title': ['A', 'B', 'C', 'D'], 'tag': ['X', 'Y', 'X', 'Z'], 'description': ['d1', 'd2', 'd3', 'd4'],
            'view_number': [100, 200, 50, 150], 'comment_number': [10, 20, 5, 15]
        })
        self.processed_item_info_path = self.processed_data_dir / "item_info.csv"
        self.item_info_df.to_csv(self.processed_item_info_path, index=False)

        self.interactions_df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u3'],
            'item_id': ['item1', 'item2', 'item1', 'item3', 'item4']
        })
        self.processed_interactions_path = self.processed_data_dir / "interactions.csv"
        self.interactions_df.to_csv(self.processed_interactions_path, index=False)

        # 2. Dummy data splits (very small for a quick training epoch)
        train_split_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['item1', 'item3']})
        val_split_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['item2', 'item1']})
        self.train_path = self.splits_dir / "train.csv"
        self.val_path = self.splits_dir / "val.csv"
        train_split_df.to_csv(self.train_path, index=False)
        val_split_df.to_csv(self.val_path, index=False)
        
        # 3. Dummy images (required by the dataset loader)
        for item_id in self.item_info_df['item_id']:
            Image.new('RGB', (100, 100), color='red').save(self.raw_image_dir / f"{item_id}.jpg")

        # 4. Dummy Config File
        self.config_path = self.configs_dir / "test_config_train.yaml"
        # This config points to all the temporary files and sets up a minimal, fast run
        config_content = f"""
model:
  vision_model: resnet
  language_model: sentence-bert
  embedding_dim: 8
  use_contrastive: false
  freeze_vision: false # Ensure vision model is trainable
  freeze_language: false # Ensure language model is trainable

training:
  batch_size: 2
  epochs: 2 # Run for two epochs to verify learning
  learning_rate: 0.01
  patience: 2
  num_workers: 0 # IMPORTANT for avoiding multiprocessing issues in tests

data:
  processed_item_info_path: {self.processed_item_info_path.as_posix()}
  processed_interactions_path: {self.processed_interactions_path.as_posix()}
  image_folder: {self.raw_image_dir.as_posix()}
  scaler_path: {self.processed_data_dir / "numerical_scaler.pkl"}
  train_data_path: {self.train_path.as_posix()}
  val_data_path: {self.val_path.as_posix()}
  numerical_features_cols:
    - view_number
    - comment_number
    - non_existent_feature # Add a non-existent feature to test validation
  cache_config:
    enabled: false

checkpoint_dir: {self.checkpoints_dir.as_posix()}
results_dir: {self.results_dir.as_posix()}
"""
        self.config_path.write_text(config_content)

    def tearDown(self):
        """Clean up all temporary directories and files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_training_pipeline_and_model_learning(self):
        """
        Tests the entire training pipeline for two epochs, ensuring not only that
        it runs without errors, but also that the model's weights are updated,
        confirming that learning is taking place.
        """
        # Load config to get model parameters
        config = Config.from_yaml(str(self.config_path))
        
        # Correctly determine n_users and n_items from the full "processed"
        # data, mimicking how the training script fits the encoders.
        full_interactions_df = pd.read_csv(self.processed_interactions_path)
        full_item_info_df = pd.read_csv(self.processed_item_info_path)
        n_users = full_interactions_df['user_id'].nunique()
        n_items = full_item_info_df['item_id'].nunique()
        
        # We must use the validated numerical features count for model instantiation
        num_numerical_features = 2 # 'view_number' and 'comment_number'

        initial_model = MultimodalRecommender(
            n_users=n_users, n_items=n_items,
            num_numerical_features=num_numerical_features,
            embedding_dim=config.model.embedding_dim,
            vision_model_name=config.model.vision_model,
            language_model_name=config.model.language_model,
            freeze_vision=config.model.freeze_vision,
            freeze_language=config.model.freeze_language
        )
        
        # Store a deep copy of the initial weights of a trainable layer
        initial_user_embedding_weights = initial_model.user_embedding.weight.clone().detach()

        # Mock sys.argv to simulate running the script from the command line
        original_argv = sys.argv
        sys.argv = [
            'scripts/train.py',
            '--config', str(self.config_path),
            '--device', 'cpu' # Force CPU for CI/testing environments
        ]

        # Run the main training function
        train_main()

        # --- Verification ---
        # 1. Verify that all expected output files and directories were created.
        model_specific_checkpoint_dir = self.checkpoints_dir / "resnet_sentence-bert"
        self.assertTrue(model_specific_checkpoint_dir.exists())
        self.assertTrue((model_specific_checkpoint_dir / "best_model.pth").exists())
        self.assertTrue((model_specific_checkpoint_dir / "last_model.pth").exists())
        self.assertTrue((self.results_dir / "training_metadata.json").exists())

        # 2. Verify model learning by checking that weights have changed.
        # Load the final state of the model from the saved checkpoint
        final_checkpoint = torch.load(model_specific_checkpoint_dir / "last_model.pth", map_location='cpu')
        
        # Create a new model instance and load the trained weights into it
        # This now uses the correct n_users and n_items, preventing the size mismatch
        final_model = MultimodalRecommender(
            n_users=n_users, n_items=n_items,
            num_numerical_features=num_numerical_features,
            embedding_dim=config.model.embedding_dim,
            vision_model_name=config.model.vision_model,
            language_model_name=config.model.language_model
        )
        final_model.load_state_dict(final_checkpoint['model_state_dict'])
        final_user_embedding_weights = final_model.user_embedding.weight.clone().detach()

        # Assert that the weights are NOT equal to their initial state
        self.assertFalse(
            torch.equal(initial_user_embedding_weights, final_user_embedding_weights),
            "Model weights did not change after training, indicating a problem in the training loop."
        )

        # 3. Verify the content of the metadata file
        with open(self.results_dir / "training_metadata.json", "r") as f:
            metadata = json.load(f)

        self.assertTrue(metadata['training_completed'])
        self.assertEqual(metadata['epochs_completed'], 2, "Training should run for 2 epochs as configured.")
        self.assertIn('final_train_loss', metadata)
        self.assertIsNotNone(metadata['final_train_loss'])
        self.assertTrue(np.isfinite(metadata['final_train_loss']), "Final training loss should be a finite number.")

        # Restore original sys.argv
        sys.argv = original_argv

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)