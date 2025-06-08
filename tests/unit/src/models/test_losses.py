# tests/unit/src/models/test_losses.py
"""
Unit tests for the loss functions used in the multimodal recommender.
"""
import unittest
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.models.losses import (
    ContrastiveLoss,
    BPRLoss,
    MultimodalRecommenderLoss
)

class TestContrastiveLoss(unittest.TestCase):
    """Test cases for the ContrastiveLoss class."""

    def setUp(self):
        """Set up the loss function and test data."""
        self.loss_fn = ContrastiveLoss(temperature=0.1)
        self.batch_size = 4
        self.embedding_dim = 16
        torch.manual_seed(42)

    def test_forward_pass_shape_and_type(self):
        """Tests that the forward pass returns a scalar tensor."""
        image_features = torch.randn(self.batch_size, self.embedding_dim)
        text_features = torch.randn(self.batch_size, self.embedding_dim)
        loss = self.loss_fn(image_features, text_features)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_perfect_alignment(self):
        """
        Tests the loss when image and text features are perfectly aligned.
        The loss should be minimal (close to zero).
        """
        features = F.normalize(torch.randn(self.batch_size, self.embedding_dim))
        loss = self.loss_fn(features, features)
        
        self.assertLess(loss.item(), 0.01)

    def test_perfect_misalignment(self):
        """
        Tests the loss when features are perfectly misaligned.
        The loss should be high and dominated by the temperature.
        """
        features = F.normalize(torch.eye(self.batch_size, self.embedding_dim))
        misaligned_text_features = torch.flip(features, dims=[0])
        
        loss = self.loss_fn(features, misaligned_text_features)
        
        # For a misaligned pair, the on-diagonal similarity is 0, and for the most
        # similar incorrect pair, the similarity is 1. The logits become 0 and 1/temperature.
        # The cross-entropy loss will be approximately -log(exp(0) / exp(1/temp)) = 1/temp.
        # Since this happens in both i2t and t2i directions, the final loss is ~1/temperature.
        expected_loss = 1 / self.loss_fn.temperature
        self.assertAlmostEqual(loss.item(), expected_loss, places=1)


class TestBPRLoss(unittest.TestCase):
    """Test cases for the BPRLoss class."""

    def setUp(self):
        """Set up the loss function."""
        self.loss_fn = BPRLoss()

    def test_well_ranked_pair(self):
        """
        Tests the loss when positive scores are much higher than negative scores.
        The loss should be close to zero.
        """
        positive_scores = torch.tensor([10.0, 9.0])
        negative_scores = torch.tensor([-10.0, -9.0])
        loss = self.loss_fn(positive_scores, negative_scores)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_poorly_ranked_pair(self):
        """
        Tests the loss when negative scores are much higher than positive scores.
        The loss should be a large positive value.
        """
        positive_scores = torch.tensor([-10.0, -9.0])
        negative_scores = torch.tensor([10.0, 9.0])
        loss = self.loss_fn(positive_scores, negative_scores)
        self.assertGreater(loss.item(), 15.0)

    def test_equally_ranked_pair(self):
        """
        Tests the loss when positive and negative scores are equal.
        The loss should be exactly log(2).
        """
        scores = torch.tensor([5.0, 0.0, -5.0])
        loss = self.loss_fn(scores, scores)
        expected_loss = np.log(2)
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)


class TestMultimodalRecommenderLoss(unittest.TestCase):
    """Test cases for the combined MultimodalRecommenderLoss."""

    def setUp(self):
        """Set up test data and loss function variants."""
        self.batch_size = 8
        self.embedding_dim = 32
        
        self.predictions = torch.rand(self.batch_size)
        self.labels = torch.randint(0, 2, (self.batch_size,)).float()
        self.vision_features = F.normalize(torch.randn(self.batch_size, self.embedding_dim))
        self.text_features = F.normalize(torch.randn(self.batch_size, self.embedding_dim))
        
        self.combined_loss_fn = MultimodalRecommenderLoss(
            use_contrastive=True,
            bce_weight=0.8,
            contrastive_weight=0.2
        )
        
        self.bce_only_loss_fn = MultimodalRecommenderLoss(use_contrastive=False)

    def test_bce_only_forward(self):
        """Tests that the loss works correctly with only the BCE component enabled."""
        result = self.bce_only_loss_fn(self.predictions, self.labels)
        
        self.assertIn('total', result)
        self.assertIn('bce', result)
        self.assertIn('contrastive', result)
        
        self.assertEqual(result['contrastive'].item(), 0.0)
        self.assertAlmostEqual(result['total'].item(), result['bce'].item(), places=6)

    def test_combined_loss_forward(self):
        """Tests the weighted combination of BCE and contrastive losses."""
        result = self.combined_loss_fn(
            self.predictions,
            self.labels,
            self.vision_features,
            self.text_features
        )
        
        bce_loss = F.binary_cross_entropy(self.predictions, self.labels)
        contrastive_loss = ContrastiveLoss()(self.vision_features, self.text_features)
        
        expected_total_loss = (0.8 * bce_loss) + (0.2 * contrastive_loss)
        
        self.assertAlmostEqual(result['bce'].item(), bce_loss.item(), places=6)
        self.assertAlmostEqual(result['contrastive'].item(), contrastive_loss.item(), places=6)
        self.assertAlmostEqual(result['total'].item(), expected_total_loss.item(), places=6)

    def test_numerical_stability_with_extreme_predictions(self):
        """
        Tests that the loss function is numerically stable when predictions are exactly 0.0 or 1.0.
        """
        extreme_predictions = torch.tensor([0.0, 1.0, 0.5, 0.2])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        
        result = self.bce_only_loss_fn(extreme_predictions, labels)
        
        self.assertTrue(torch.isfinite(result['total']))
        self.assertFalse(torch.isnan(result['total']))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)