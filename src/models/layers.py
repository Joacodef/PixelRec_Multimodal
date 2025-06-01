import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For sqrt

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module.
    This implementation has vision_features act as the query, 
    and text_features provide the keys and values.
    The output represents text features attended by (contextualized by) vision features.
    """
    def __init__(self, dim: int):
        """
        Initializes the CrossModalAttention layer.

        Args:
            dim: The feature dimension of the input vision and text features.
                 It's assumed that query, key, and value will all be projected to this dimension.
        """
        super().__init__()
        # Linear projections for query, key, and value
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for cross-modal attention.

        Vision features generate the query.
        Text features generate the keys and values.

        Args:
            vision_features: Tensor of vision features. 
                             Shape can be (batch_size, vision_feature_dim) or 
                             (batch_size, num_vision_tokens, vision_feature_dim).
                             vision_feature_dim must match 'dim' from __init__.
            text_features: Tensor of text features.
                           Shape can be (batch_size, text_feature_dim) or
                           (batch_size, num_text_tokens, text_feature_dim).
                           text_feature_dim must match 'dim' from __init__.

        Returns:
            A tensor representing text features contextualized by vision features.
            If vision_features was (B, dim), output is (B, dim).
            If vision_features was (B, L_v, dim), output is (B, L_v, dim_v_proj).
            The content of the output is derived from the projected text values.
        """

        # Project query (from vision), key (from text), value (from text)
        # q_proj: (batch_size, num_vision_tokens_or_1, dim)
        # k_proj: (batch_size, num_text_tokens_or_1, dim)
        # v_proj: (batch_size, num_text_tokens_or_1, dim)
        q_proj = self.query_projection(vision_features)
        k_proj = self.key_projection(text_features)
        v_proj = self.value_projection(text_features)

        # Store original ndim of vision_features to conditionally squeeze output later
        original_vision_ndim = vision_features.ndim

        # Unsqueeze batch-only tensors to have a sequence length of 1
        # This makes them (batch_size, 1, dim) for consistent processing.
        if q_proj.ndim == 2:
            q_proj = q_proj.unsqueeze(1)
        if k_proj.ndim == 2:
            k_proj = k_proj.unsqueeze(1)
        if v_proj.ndim == 2:
            v_proj = v_proj.unsqueeze(1)
        
        # Scaled Dot-Product Attention
        # energy shape: (batch_size, num_vision_tokens_or_1, num_text_tokens_or_1)
        energy = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        
        # scaling_factor for stabilization
        scaling_factor = math.sqrt(self.dim) # or q_proj.size(-1)
        
        attention_scores = energy / scaling_factor
        
        # Softmax is applied over the key sequence dimension (last dim of attention_scores)
        # attention_weights shape: (batch_size, num_vision_tokens_or_1, num_text_tokens_or_1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weighted sum of value vectors (v_proj)
        # output shape: (batch_size, num_vision_tokens_or_1, dim)
        output = torch.matmul(attention_weights, v_proj)
        
        # If the original vision_features tensor was 2D (batch_size, dim),
        # and the output is (batch_size, 1, dim), squeeze the sequence dimension.
        if original_vision_ndim == 2 and output.shape[1] == 1:
            output = output.squeeze(1)
            
        return output