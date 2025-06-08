# src/models/layers.py
"""
Defines custom neural network layers and modules used in the recommender model.

This module contains specialized layers, such as attention mechanisms, that are
not standard in PyTorch but are essential for the model's architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    Implements a cross-modal attention mechanism.

    This layer computes attention scores where one modality (e.g., vision)
    acts as the query, and another modality (e.g., text) provides the keys
    and values. The output is a representation of the key/value modality that
    is contextualized by the query modality. This is useful for fusing
    information from different sources.
    """
    def __init__(self, dim: int):
        """
        Initializes the CrossModalAttention layer and its projection layers.

        Args:
            dim (int): The feature dimension of the input and output tensors.
                       It is assumed that the query, key, and value will all be
                       projected to this same dimension for the attention
                       calculation.
        """
        super().__init__()
        # Linear projections for query, key, and value.
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the cross-modal attention mechanism.

        In this implementation, the vision features generate the query tensor,
        while the text features generate the key and value tensors. The resulting
        output is a representation of the text features, weighted by their
        relevance to the vision features.

        Args:
            vision_features (torch.Tensor): A tensor of vision features that will
                act as the query. Its shape can be (batch_size, feature_dim) for
                pooled features or (batch_size, num_tokens, feature_dim) for
                token-level features.
            text_features (torch.Tensor): A tensor of text features that will
                provide the keys and values. Its shape should be compatible with
                vision_features.

        Returns:
            torch.Tensor: A tensor representing the text features contextualized
                          by the vision features. Its shape matches the query's
                          shape (vision_features).
        """
        # Project the vision features to get the query, and text features for key/value.
        q_proj = self.query_projection(vision_features)
        k_proj = self.key_projection(text_features)
        v_proj = self.value_projection(text_features)

        # Store the original number of dimensions of the vision tensor.
        # This is used to ensure the output shape matches the input query shape.
        original_vision_ndim = vision_features.ndim

        # Unsqueeze 2D tensors to 3D to handle both pooled and token-level features uniformly.
        # This adds a sequence dimension of size 1 for consistent processing.
        if q_proj.ndim == 2:
            q_proj = q_proj.unsqueeze(1)
        if k_proj.ndim == 2:
            k_proj = k_proj.unsqueeze(1)
        if v_proj.ndim == 2:
            v_proj = v_proj.unsqueeze(1)

        # Calculate the raw attention scores (energy) via dot product of query and key.
        energy = torch.matmul(q_proj, k_proj.transpose(-2, -1))

        # Scale the energy by the square root of the feature dimension for stabilization.
        scaling_factor = math.sqrt(self.dim)
        attention_scores = energy / scaling_factor

        # Apply softmax to the last dimension to get normalized attention weights.
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the weighted sum of value vectors using the attention weights.
        output = torch.matmul(attention_weights, v_proj)

        # If the original query (vision_features) was a 2D tensor, squeeze the
        # sequence dimension from the output to match the original shape.
        if original_vision_ndim == 2 and output.shape[1] == 1:
            output = output.squeeze(1)

        return output