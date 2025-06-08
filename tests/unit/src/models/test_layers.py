# tests/unit/src/models/test_layers.py
"""
Unit tests for custom model layers.
"""
import unittest
import torch
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.models.layers import CrossModalAttention

class TestCrossModalAttention(unittest.TestCase):
    """Test cases for the CrossModalAttention layer."""

    def setUp(self):
        """Set up the attention layer and test data dimensions before each test."""
        self.dim = 32  # Feature dimension
        self.batch_size = 4
        self.vision_seq_len = 5
        self.text_seq_len = 10
        
        # Ensures that test results are reproducible
        torch.manual_seed(42)
        
        # Instantiates the layer to be tested
        self.attention_layer = CrossModalAttention(dim=self.dim)

    def test_initialization(self):
        """Tests that the layer and its sub-modules are initialized correctly."""
        self.assertIsInstance(self.attention_layer, torch.nn.Module)
        self.assertEqual(self.attention_layer.query_projection.in_features, self.dim)
        self.assertEqual(self.attention_layer.key_projection.out_features, self.dim)

    def test_forward_pass_2d_inputs(self):
        """
        Tests the forward pass with 2D tensors (batch_size, dim).
        This simulates using pooled features from vision and text models.
        """
        # Creates dummy 2D tensors for vision (query) and text (key/value)
        vision_features = torch.randn(self.batch_size, self.dim)
        text_features = torch.randn(self.batch_size, self.dim)
        
        # Performs the forward pass
        output = self.attention_layer(vision_features, text_features)
        
        # Verifies that the output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.dim))

    def test_forward_pass_3d_inputs(self):
        """
        Tests the forward pass with 3D tensors (batch_size, seq_len, dim).
        This simulates using token-level features.
        """
        # Creates dummy 3D tensors
        vision_features = torch.randn(self.batch_size, self.vision_seq_len, self.dim)
        text_features = torch.randn(self.batch_size, self.text_seq_len, self.dim)
        
        output = self.attention_layer(vision_features, text_features)
        
        # The output shape should match the query's shape (vision_features)
        self.assertEqual(output.shape, (self.batch_size, self.vision_seq_len, self.dim))

    def test_forward_pass_mixed_inputs(self):
        """
        Tests the forward pass with a 2D query (vision) and 3D key/value (text).
        """
        vision_features = torch.randn(self.batch_size, self.dim)
        text_features = torch.randn(self.batch_size, self.text_seq_len, self.dim)
        
        output = self.attention_layer(vision_features, text_features)
        
        # Since the query (vision) was 2D, the output should also be 2D
        self.assertEqual(output.shape, (self.batch_size, self.dim))

    def test_attention_behavior_with_identical_keys(self):
        """
        Tests a specific scenario to verify attention logic.
        If all text tokens (keys/values) are identical, the output should be that token's
        representation, regardless of the vision query, because the attention
        weights will sum to 1 over an identical value.
        """
        vision_features = torch.randn(self.batch_size, self.dim)
        
        # Creates a single text feature and repeats it across the sequence length
        single_text_feature = torch.randn(self.batch_size, 1, self.dim)
        identical_text_features = single_text_feature.repeat(1, self.text_seq_len, 1)

        output = self.attention_layer(vision_features, identical_text_features)

        # Projects the single text feature through the value projection layer to get the expected output
        expected_output = self.attention_layer.value_projection(single_text_feature.squeeze(1))

        # Verifies that the actual output is very close to the expected output
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_batch_independence(self):
        """
        Tests that each item in a batch is processed independently.
        The output for a batch should be the same as processing each item
        individually and then stacking the results.
        """
        vision_features = torch.randn(self.batch_size, self.dim)
        text_features = torch.randn(self.batch_size, self.text_seq_len, self.dim)
        
        # Processes the full batch at once
        batch_output = self.attention_layer(vision_features, text_features)
        
        # Processes each item in the batch individually
        individual_outputs = []
        for i in range(self.batch_size):
            vision_single = vision_features[i].unsqueeze(0)
            text_single = text_features[i].unsqueeze(0)
            output_single = self.attention_layer(vision_single, text_single)
            individual_outputs.append(output_single)
        
        # Stacks the individual results to form a batch tensor
        stacked_output = torch.cat(individual_outputs, dim=0)
        
        # Verifies that the batched output is identical to the stacked individual outputs
        self.assertTrue(torch.allclose(batch_output, stacked_output, atol=1e-6))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)