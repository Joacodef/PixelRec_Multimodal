# src/models/multimodal.py
"""
Multimodal recommender model architecture using pre-trained vision and language models.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForImageClassification, # For models like ResNet, ConvNeXT
    CLIPVisionModel,
    CLIPTextModel,
    Dinov2Model # For DINOv2 models
)
from typing import Optional, Tuple, Union # Adjusted Union import for type hint

from ..config import MODEL_CONFIGS # Relative import for configuration

class PretrainedMultimodalRecommender(nn.Module):
    """
    A multimodal recommender system that leverages pre-trained models for vision
    and language feature extraction, combined with user and item embeddings.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        vision_model_name: str = 'clip', # Key for vision model in MODEL_CONFIGS
        language_model_name: str = 'sentence-bert', # Key for language model in MODEL_CONFIGS
        freeze_vision: bool = True,
        freeze_language: bool = True,
        use_contrastive: bool = True, # Enables contrastive loss if vision_model_name is 'clip'
        dropout_rate: float = 0.3
    ):
        """
        Initializes the PretrainedMultimodalRecommender.

        Args:
            n_users: Number of unique users for user embeddings.
            n_items: Number of unique items for item embeddings.
            embedding_dim: Dimensionality of the user, item, and projected modal embeddings.
            vision_model_name: Identifier for the vision model configuration.
            language_model_name: Identifier for the language model configuration.
            freeze_vision: If True, freezes the weights of the pre-trained vision model.
            freeze_language: If True, freezes the weights of the pre-trained language model.
            use_contrastive: If True and using 'clip' vision model, enables contrastive learning components.
            dropout_rate: Dropout rate for regularization in projection and fusion layers.
        """
        super(PretrainedMultimodalRecommender, self).__init__()

        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name]
        self.language_config = MODEL_CONFIGS['language'][language_model_name]
        self.use_contrastive = use_contrastive and vision_model_name == 'clip'
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        # Store the number of numerical features, assuming 7 based on original design.
        # This should ideally be configurable if it can vary.
        self.num_numerical_features = 7

        self._init_embeddings(n_users, n_items)
        self._init_vision_model(vision_model_name, freeze_vision)
        self._init_language_model(language_model_name, freeze_language)
        self._init_projection_layers()
        self._init_fusion_network()

    def _init_embeddings(self, n_users: int, n_items: int):
        """Initializes user and item embedding layers."""
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _init_vision_model(self, model_key: str, freeze: bool):
        """Initializes the vision model based on the provided key."""
        hf_model_name = self.vision_config['name']
        if model_key == 'clip':
            self.vision_model = CLIPVisionModel.from_pretrained(hf_model_name)
            if self.use_contrastive:
                self.clip_text_model = CLIPTextModel.from_pretrained(hf_model_name)
        elif model_key == 'dino': # This key now loads DINOv2
            self.vision_model = Dinov2Model.from_pretrained(hf_model_name)
        else: # Handles ResNet, ConvNeXT, or other AutoModelForImageClassification compatible models
            self.vision_model = AutoModelForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=self.vision_config['dim'], # Using native dim as num_labels for feature extraction
                ignore_mismatched_sizes=True
            )
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def _init_language_model(self, model_key: str, freeze: bool):
        """Initializes the language model based on the provided key."""
        hf_model_name = self.language_config['name']
        self.language_model = AutoModel.from_pretrained(hf_model_name)
        if freeze:
            for param in self.language_model.parameters():
                param.requires_grad = False

    def _init_projection_layers(self):
        """Initializes linear projection layers for modality features."""
        self.vision_projection = nn.Sequential(
            nn.Linear(self.vision_config['dim'], self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.language_projection = nn.Sequential(
            nn.Linear(self.language_config['dim'], self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.numerical_projection = nn.Sequential(
            nn.Linear(self.num_numerical_features, self.embedding_dim), # Uses self.num_numerical_features
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        if self.use_contrastive: # Specific to CLIP usage
            self.contrastive_projection = nn.Linear(
                self.vision_config['dim'], self.embedding_dim
            )
            self.temperature = nn.Parameter(torch.tensor(0.07))

    def _init_fusion_network(self):
        """Initializes the attention-based fusion network and final prediction layers."""
        # MultiheadAttention expects input (L, N, E) where L is sequence length, N is batch size, E is embedding dim.
        # Here, 5 modal/user/item features are stacked, so L=5.
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, num_heads=4, dropout=self.dropout_rate
        )
        # The output of attention (after permutation and view) will be (N, 5 * E)
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 5, 512), nn.ReLU(), nn.Dropout(self.dropout_rate), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(self.dropout_rate), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(
        self, user_idx: torch.Tensor, item_idx: torch.Tensor, image: torch.Tensor,
        text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        clip_text_input_ids: Optional[torch.Tensor] = None, # New
        clip_text_attention_mask: Optional[torch.Tensor] = None, # New
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Performs the forward pass of the model.

        Args:
            user_idx: Tensor of user indices.
            item_idx: Tensor of item indices.
            image: Tensor of image pixel values.
            text_input_ids: Tensor of text input IDs.
            text_attention_mask: Tensor of text attention masks.
            numerical_features: Tensor of numerical features.
            return_embeddings: If True, returns intermediate embeddings along with predictions.

        Returns:
            The prediction scores (Tensor). If return_embeddings is True, also returns
            raw vision features, raw CLIP text features (if applicable), and projected vision embeddings.
        """
        batch_size = user_idx.size(0)

        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        raw_vision_feat = self._get_vision_features(image)
        projected_vision_emb = self.vision_projection(raw_vision_feat)

        raw_language_feat = self._get_language_features(text_input_ids, text_attention_mask)
        projected_language_emb = self.language_projection(raw_language_feat)

        projected_numerical_emb = self.numerical_projection(numerical_features)

        raw_clip_text_feat = None
        # self.use_contrastive is True if vision_model is 'clip' AND config model.use_contrastive is true
        if self.use_contrastive and hasattr(self, 'clip_text_model'):
            if clip_text_input_ids is not None and clip_text_attention_mask is not None:
                raw_clip_text_feat = self._get_clip_text_features(clip_text_input_ids, clip_text_attention_mask)
            else:
                raise ValueError("CLIP text input IDs and attention mask must be provided when using contrastive learning.")


        combined_features = self._apply_attention_fusion(
            user_emb, item_emb, projected_vision_emb,
            projected_language_emb, projected_numerical_emb, batch_size
        )
        output = self.fusion(combined_features)

        if return_embeddings:
            return output, raw_vision_feat, raw_clip_text_feat, projected_vision_emb
        return output
    
    def _get_clip_text_features(self, clip_input_ids: torch.Tensor, clip_attention_mask: torch.Tensor) -> Optional[torch.Tensor]: # Signature updated
        """Extracts text features using the CLIP text model, if available."""
        if hasattr(self, 'clip_text_model') and self.clip_text_model is not None:
            outputs = self.clip_text_model(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
            return outputs.pooler_output 
        return None

    def _get_vision_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extracts vision features from the image tensor using the vision model."""
        # Standardize input argument name for Hugging Face models
        model_input = {'pixel_values': image}
        
        if hasattr(self.vision_model, 'get_image_features'): # Specific to CLIP's vision model
            vision_output = self.vision_model.get_image_features(**model_input)
        else:
            outputs = self.vision_model(**model_input)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                vision_output = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'): # Fallback for models without pooler_output
                vision_output = outputs.last_hidden_state.mean(dim=1)
            else:
                # If output is a tensor (e.g. from a model without explicit pooler or last_hidden_state attribute in output object)
                # This case needs careful handling based on specific model structure
                # For AutoModelForImageClassification, output.logits might be returned if not careful
                # Assuming a feature extractor should give a feature tensor.
                # This part might need adjustment if a model doesn't fit the above patterns.
                # For now, we assume one of the above attributes will exist for feature extraction.
                raise ValueError("Vision model output structure not recognized for feature extraction.")
        return vision_output

    def _get_language_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extracts language features from text inputs using the language model."""
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            lang_feat = outputs.pooler_output
        else: # Default to mean of last hidden state
            lang_feat = outputs.last_hidden_state.mean(dim=1)
        return lang_feat

    def _get_clip_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Extracts text features using the CLIP text model, if available."""
        if hasattr(self, 'clip_text_model') and self.clip_text_model is not None:
            outputs = self.clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.pooler_output # CLIP text model typically uses pooler_output
        return None

    def _apply_attention_fusion(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor, vision_emb: torch.Tensor,
        language_emb: torch.Tensor, numerical_emb: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Combines features using self-attention and prepares for final fusion layers."""
        # Stack features: (num_features, batch_size, embedding_dim)
        features_stacked = torch.stack([user_emb, item_emb, vision_emb, language_emb, numerical_emb], dim=0)
        # Apply attention
        attended_features, _ = self.attention(features_stacked, features_stacked, features_stacked)
        # Reshape for concatenation: (batch_size, num_features * embedding_dim)
        combined = attended_features.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return combined

    def get_item_embedding(
        self, item_idx: torch.Tensor, image: torch.Tensor, text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor, numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a comprehensive item embedding by concatenating its ID-based embedding
        with projected modal features. For inference or analysis purposes.
        """
        with torch.no_grad(): # Ensure no gradients are computed during this process
            base_item_emb = self.item_embedding(item_idx)
            raw_vision_feat = self._get_vision_features(image)
            projected_vision_emb = self.vision_projection(raw_vision_feat)
            raw_language_feat = self._get_language_features(text_input_ids, text_attention_mask)
            projected_language_emb = self.language_projection(raw_language_feat)
            projected_numerical_emb = self.numerical_projection(numerical_features)

            item_full_embedding = torch.cat(
                [base_item_emb, projected_vision_emb, projected_language_emb, projected_numerical_emb],
                dim=-1 # Concatenate along the last dimension (feature dimension)
            )
        return item_full_embedding


# The EnhancedMultimodalRecommender class definition follows.
# Its comments and functionality are kept concise as per the production-ready requirement.
# If CrossModalAttention is not fully integrated or used in a production path,
# its inclusion should be minimal or clearly justified for a specific, functional purpose.

class EnhancedMultimodalRecommender(PretrainedMultimodalRecommender):
    """
    An enhanced version of the multimodal recommender, potentially incorporating
    additional mechanisms like cross-modal attention.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the EnhancedMultimodalRecommender.
        Inherits from PretrainedMultimodalRecommender and may add or override components.
        """
        super().__init__(*args, **kwargs)

        # Initialize cross-modal attention layers if they are part of this enhanced model's design.
        # The CrossModalAttention layer is assumed to be defined in '.layers'.
        from .layers import CrossModalAttention # Ensure this import path is correct.
        self.vision_text_attention = CrossModalAttention(self.embedding_dim)
        self.text_vision_attention = CrossModalAttention(self.embedding_dim)
        
        # Note: The original PretrainedMultimodalRecommender.forward pass does not use
        # the _apply_cross_modal_fusion method. If this Enhanced class intends to use it,
        # its forward() method would need to be overridden to incorporate calls to
        # _apply_cross_modal_fusion and adjust subsequent fusion logic.

    def _apply_cross_modal_fusion(
        self, vision_emb: torch.Tensor, language_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies cross-modal attention between vision and language embeddings.

        Args:
            vision_emb: Projected vision embeddings.
            language_emb: Projected language embeddings.

        Returns:
            A tuple containing enhanced vision and language embeddings.
        """
        # Vision features attend to text features
        vision_contextualized_by_text = self.vision_text_attention(vision_emb, language_emb)
        # Text features attend to vision features
        text_contextualized_by_vision = self.text_vision_attention(language_emb, vision_emb)

        # Example of enhancing original embeddings with cross-modal components
        # The combination strategy (e.g., addition, weighting) can be further refined.
        vision_enhanced = vision_emb + 0.5 * vision_contextualized_by_text
        text_enhanced = language_emb + 0.5 * text_contextualized_by_vision

        return vision_enhanced, text_enhanced