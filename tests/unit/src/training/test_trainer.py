# tests/unit/src/training/test_trainer.py
"""
Unit tests for the training.trainer module.
"""
import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import shutil
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.training.trainer import Trainer

class SimpleMockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.activation = nn.Sigmoid()
        self.use_contrastive = False

    def forward(self, **kwargs):
        numerical_features = kwargs.get('numerical_features')
        logits = self.fc(numerical_features)
        return self.activation(logits)

class TestTrainer(unittest.TestCase):
    """Test cases for the Trainer class."""

    def setUp(self):
        """Set up a temporary environment for each test."""
        self.test_dir = Path("test_temp_trainer")
        self.test_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cpu')
        self.mock_model = SimpleMockModel().to(self.device)
        
        # Create a mock config with the expected nested structure
        self.mock_config = MagicMock()
        self.mock_config.model.vision_model = "test_vision"
        self.mock_config.model.language_model = "test_language"
        self.mock_config.model.use_contrastive = False
        # Set default early stopping behavior for most tests
        self.mock_config.training.early_stopping_metric = 'val_loss'
        self.mock_config.training.early_stopping_direction = 'minimize'


        def create_mock_batch(label_val):
            return {
                'label': torch.tensor([label_val], dtype=torch.float32),
                'user_idx': torch.tensor([0], dtype=torch.long),
                'item_idx': torch.tensor([0], dtype=torch.long),
                'image': torch.randn(1, 3, 224, 224),
                'text_input_ids': torch.zeros(1, 128, dtype=torch.long),
                'text_attention_mask': torch.zeros(1, 128, dtype=torch.long),
                'numerical_features': torch.randn(1, 10),
                'tag_idx': torch.randint(0, 10, (1,), dtype=torch.long),
                'clip_text_input_ids': torch.zeros(1, 77, dtype=torch.long),
                'clip_text_attention_mask': torch.zeros(1, 77, dtype=torch.long),
            }

        self.train_loader = [create_mock_batch(1.0)]
        self.val_loader = [create_mock_batch(0.0)]

        self.trainer = Trainer(
            model=self.mock_model,
            device=self.device,
            config=self.mock_config,
            checkpoint_dir=str(self.test_dir)
        )
        
        self.optimizer = torch.optim.SGD(self.mock_model.parameters(), lr=0.1)
        self.trainer.optimizer = self.optimizer

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization_and_checkpoint_dir_creation(self):
        """Tests that the Trainer initializes and creates the correct checkpoint directories."""
        expected_model_dir = self.test_dir / "test_vision_test_language"
        expected_encoders_dir = self.test_dir / "encoders"
        
        self.assertTrue(expected_model_dir.exists())
        self.assertTrue(expected_encoders_dir.exists())
        self.assertEqual(self.trainer.get_model_checkpoint_dir(), expected_model_dir)

    @patch('src.training.trainer.wandb', MagicMock())
    def test_train_and_validate_epoch(self):
        """Tests the execution of a single training and validation epoch."""
        self.optimizer.step = MagicMock()

        self.mock_model.train()
        train_metrics = self.trainer._train_epoch(self.train_loader, self.optimizer, gradient_clip=1.0)
        
        self.optimizer.step.assert_called_once()
        self.assertIn('total_loss', train_metrics)
        self.assertIn('accuracy', train_metrics)
        self.assertIn('f1_score', train_metrics)
        self.assertIn('precision', train_metrics)
        self.assertIn('recall', train_metrics)

        self.mock_model.eval()
        val_metrics = self.trainer._validate_epoch(self.val_loader)
        
        self.assertIn('total_loss', val_metrics)
        self.assertIn('accuracy', val_metrics)
        self.assertIn('f1_score', val_metrics)
        
    def test_save_and_load_checkpoint(self):
        """Tests that saving and loading a checkpoint works correctly."""
        initial_state = {k: v.clone() for k, v in self.mock_model.state_dict().items()}
        
        self.trainer._train_epoch(self.train_loader, self.optimizer, gradient_clip=1.0)
        
        checkpoint_filename = "test_checkpoint.pth"
        self.trainer.save_checkpoint(checkpoint_filename)
        
        expected_path = self.test_dir / "test_vision_test_language" / checkpoint_filename
        self.assertTrue(expected_path.exists())
        
        new_model = SimpleMockModel().to(self.device)
        # Instantiate the new trainer with the correct 'config' argument
        new_trainer = Trainer(
            model=new_model, 
            device=self.device, 
            config=self.mock_config, 
            checkpoint_dir=str(self.test_dir)
        )
        new_trainer.load_checkpoint(checkpoint_filename)
        
        loaded_state = new_model.state_dict()
        for key in initial_state:
            self.assertFalse(torch.equal(initial_state[key], loaded_state[key]))
            self.assertTrue(torch.equal(self.mock_model.state_dict()[key], loaded_state[key]))


    @patch('src.training.trainer.Trainer.save_checkpoint')
    def test_early_stopping_logic(self, mock_save_checkpoint):
        """Tests the flexible logic for early stopping."""
        patience = 2
        
        # --- Test Case 1: Minimizing val_loss ---
        self.trainer.config.training.early_stopping_direction = 'minimize'
        self.trainer.best_early_stopping_score = float('inf')

        # First score is always an improvement
        stop = self.trainer._check_early_stopping(score=0.5, patience=patience)
        self.assertFalse(stop)
        self.assertEqual(self.trainer.best_early_stopping_score, 0.5)
        self.assertEqual(self.trainer.patience_counter, 0)
        mock_save_checkpoint.assert_called_once_with('best_model.pth', is_best=True)

        # Worsening score increases counter
        stop = self.trainer._check_early_stopping(score=0.6, patience=patience)
        self.assertFalse(stop)
        self.assertEqual(self.trainer.patience_counter, 1)
        
        # --- Test Case 2: Maximizing val_f1_score ---
        self.trainer.config.training.early_stopping_direction = 'maximize'
        self.trainer.best_early_stopping_score = float('-inf')
        self.trainer.patience_counter = 0 # Reset counter for new test case

        # First score is always an improvement
        stop = self.trainer._check_early_stopping(score=0.8, patience=patience)
        self.assertFalse(stop)
        self.assertEqual(self.trainer.best_early_stopping_score, 0.8)
        self.assertEqual(self.trainer.patience_counter, 0)

        # Worsening score (lower F1) increases counter
        stop = self.trainer._check_early_stopping(score=0.7, patience=patience)
        self.assertFalse(stop)
        self.assertEqual(self.trainer.patience_counter, 1)
        
        # Another worsening score triggers early stopping
        stop = self.trainer._check_early_stopping(score=0.6, patience=patience)
        self.assertTrue(stop)
        self.assertEqual(self.trainer.patience_counter, 2)



    def test_nan_loss_handling_in_training(self):
        """Tests that the trainer can handle NaN loss during training without crashing."""
        nan_features = torch.tensor([[float('nan')] * 10])
        nan_batch = self.train_loader[0].copy()
        nan_batch['numerical_features'] = nan_features
        
        self.optimizer.step = MagicMock()

        original_forward = self.mock_model.forward
        def forward_with_nan_check(**kwargs):
            if torch.isnan(kwargs['numerical_features']).any():
                return torch.tensor([[float('nan')]])
            return original_forward(**kwargs)
        self.mock_model.forward = forward_with_nan_check
        
        metrics = self.trainer._train_epoch([nan_batch], self.optimizer, 1.0)

        self.optimizer.step.assert_not_called()
        self.assertTrue(np.isnan(metrics['total_loss']))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)