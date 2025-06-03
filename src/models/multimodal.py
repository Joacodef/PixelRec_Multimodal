# src/models/multimodal.py
"""
Multimodal recommender model architecture using pre-trained vision and language models.
Now with fully configurable architecture from config file.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    CLIPVisionModel,
    CLIPTextModel,
    Dinov2Model
)
from typing import Optional, Tuple, Union, List

from ..config import MODEL_CONFIGS


class PretrainedMultimodalRecommender(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        vision_model_name: str = 'clip',
        language_model_name: str = 'sentence-bert',
        freeze_vision: bool = True,
        freeze_language: bool = True,
        use_contrastive: bool = True,
        dropout_rate: float = 0.3,
        # Additional architectural parameters
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        fusion_hidden_dims: List[int] = None,
        fusion_activation: str = 'relu',
        use_batch_norm: bool = True,
        projection_hidden_dim: Optional[int] = None,
        final_activation: str = 'sigmoid',
        init_method: str = 'xavier_uniform',
        contrastive_temperature: float = 0.07
    ):
        super(PretrainedMultimodalRecommender, self).__init__()

        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name]
        self.language_config = MODEL_CONFIGS['language'][language_model_name]
        self.use_contrastive = use_contrastive and vision_model_name == 'clip'
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.num_numerical_features = 7
        
        # Architectural parameters
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.fusion_hidden_dims = fusion_hidden_dims or [512, 256, 128]
        self.fusion_activation = fusion_activation
        self.use_batch_norm = use_batch_norm
        self.projection_hidden_dim = projection_hidden_dim
        self.final_activation = final_activation
        self.init_method = init_method
        self.contrastive_temperature = contrastive_temperature

        self._init_embeddings(n_users, n_items)
        self._init_vision_model(vision_model_name, freeze_vision)
        self._init_language_model(language_model_name, freeze_language)
        self._init_projection_layers()
        self._init_fusion_network()

    def _get_activation(self, activation_name: str):
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'silu': nn.SiLU()
        }
        return activations.get(activation_name.lower(), nn.ReLU())

    def _init_embeddings(self, n_users: int, n_items: int):
        """Initializes user and item embedding layers with configurable initialization."""
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(n_items, self.embedding_dim)
        
        # Apply configured initialization
        if self.init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)
        elif self.init_method == 'xavier_normal':
            nn.init.xavier_normal_(self.user_embedding.weight)
            nn.init.xavier_normal_(self.item_embedding.weight)
        elif self.init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.user_embedding.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.item_embedding.weight, nonlinearity='relu')
        elif self.init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.user_embedding.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.item_embedding.weight, nonlinearity='relu')
        else:
            # Default to xavier_uniform
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)

    def _init_vision_model(self, model_key: str, freeze: bool):
        """Initializes the vision model based on the provided key."""
        hf_model_name = self.vision_config['name']
        if model_key == 'clip':
            self.vision_model = CLIPVisionModel.from_pretrained(hf_model_name)
            if self.use_contrastive:
                self.clip_text_model = CLIPTextModel.from_pretrained(hf_model_name)
        elif model_key == 'dino': 
            self.vision_model = Dinov2Model.from_pretrained(hf_model_name)
        elif model_key == 'resnet' or model_key == 'convnext': # Handle resnet and convnext specifically
            # Load as a base model to get features before a classification head
            self.vision_model = AutoModel.from_pretrained(hf_model_name)
        else: 
            # Fallback for any other vision model not explicitly handled,
            # or if you intend to use AutoModelForImageClassification for some.
            # This was your original 'else' logic.
            print(f"Warning: Vision model key '{model_key}' not explicitly handled for base model loading. Defaulting to AutoModelForImageClassification.")
            self.vision_model = AutoModelForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=self.vision_config['dim'], 
                ignore_mismatched_sizes=True # This might be relevant if using num_labels
            )
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if hasattr(self, 'clip_text_model') and self.clip_text_model and freeze:
            for param in self.clip_text_model.parameters():
                param.requires_grad = False

    def _init_language_model(self, model_key: str, freeze: bool):
        """Initializes the language model based on the provided key."""
        hf_model_name = self.language_config['name']
        self.language_model = AutoModel.from_pretrained(hf_model_name)
        if freeze:
            for param in self.language_model.parameters():
                param.requires_grad = False

    def _init_projection_layers(self):
        """Initializes linear projection layers for modality features with optional hidden layers."""
        activation = self._get_activation(self.fusion_activation)
        
        # Vision projection
        if self.projection_hidden_dim:
            self.vision_projection = nn.Sequential(
                nn.Linear(self.vision_config['dim'], self.projection_hidden_dim),
                activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.projection_hidden_dim, self.embedding_dim),
                activation,
                nn.Dropout(self.dropout_rate)
            )
        else:
            self.vision_projection = nn.Sequential(
                nn.Linear(self.vision_config['dim'], self.embedding_dim),
                activation,
                nn.Dropout(self.dropout_rate)
            )
        
        # Language projection
        if self.projection_hidden_dim:
            self.language_projection = nn.Sequential(
                nn.Linear(self.language_config['dim'], self.projection_hidden_dim),
                activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.projection_hidden_dim, self.embedding_dim),
                activation,
                nn.Dropout(self.dropout_rate)
            )
        else:
            self.language_projection = nn.Sequential(
                nn.Linear(self.language_config['dim'], self.embedding_dim),
                activation,
                nn.Dropout(self.dropout_rate)
            )
        
        # Numerical projection
        if self.projection_hidden_dim:
            self.numerical_projection = nn.Sequential(
                nn.Linear(self.num_numerical_features, self.projection_hidden_dim),
                activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.projection_hidden_dim, self.embedding_dim),
                activation,
                nn.Dropout(self.dropout_rate)
            )
        else:
            self.numerical_projection = nn.Sequential(
                nn.Linear(self.num_numerical_features, self.embedding_dim),
                activation,
                nn.Dropout(self.dropout_rate)
            )

        if self.use_contrastive:
            self.vision_contrastive_projection = nn.Linear(
                self.vision_config['dim'],
                self.embedding_dim
            )
            # For CLIP text model output dimension
            clip_text_model_raw_output_dim = 512 
            self.text_contrastive_projection = nn.Linear(
                clip_text_model_raw_output_dim, 
                self.embedding_dim
            )
            self.temperature = nn.Parameter(torch.tensor(self.contrastive_temperature))

    def _init_fusion_network(self):
        """Initializes the attention-based fusion network and final prediction layers with configurable architecture."""
        # Multi-head attention with configurable parameters
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=False  # We'll handle the dimension ordering
        )
        
        # Build fusion network layers dynamically
        activation = self._get_activation(self.fusion_activation)
        fusion_layers = []
        
        # Input dimension is 5 * embedding_dim (user, item, vision, language, numerical)
        input_dim = self.embedding_dim * 5
        
        for i, hidden_dim in enumerate(self.fusion_hidden_dims):
            fusion_layers.append(nn.Linear(input_dim, hidden_dim))
            fusion_layers.append(activation)
            fusion_layers.append(nn.Dropout(self.dropout_rate))
            
            if self.use_batch_norm:
                fusion_layers.append(nn.BatchNorm1d(hidden_dim))
            
            input_dim = hidden_dim
        
        # Final prediction layer
        fusion_layers.append(nn.Linear(input_dim, 1))
        
        # Add final activation if specified
        if self.final_activation == 'sigmoid':
            fusion_layers.append(nn.Sigmoid())
        elif self.final_activation == 'tanh':
            fusion_layers.append(nn.Tanh())
        # If 'none', no activation is added
        
        self.fusion = nn.Sequential(*fusion_layers)
    
    def _get_vision_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extracts vision features from the image tensor using the vision model."""
        model_input = {'pixel_values': image}
        vision_output = None

        # Check if the model is a CLIP vision model and has 'get_image_features'
        # The vision_config might be more reliable than model_key if self.vision_model is already instantiated
        is_clip_model_type = 'clip' in self.vision_config.get('name', '').lower() # Check against HF name

        if is_clip_model_type and hasattr(self.vision_model, 'get_image_features'):
            vision_output = self.vision_model.get_image_features(**model_input)
        else:
            # For other models like ResNet, DINOv2, ConvNeXT loaded as AutoModel or their specific model class
            outputs = self.vision_model(**model_input)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                vision_output = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                lhs = outputs.last_hidden_state
                if lhs.ndim == 4:  # Typical for CNN feature maps (batch_size, num_channels, height, width)
                    # Global average pooling across spatial dimensions
                    vision_output = torch.nn.functional.adaptive_avg_pool2d(lhs, (1, 1)).squeeze(-1).squeeze(-1)
                elif lhs.ndim == 3:  # Typical for ViT-like sequence outputs (batch_size, seq_len, hidden_dim)
                    vision_output = lhs.mean(dim=1)  # Pool over sequence length
                else:
                    raise ValueError(f"Unsupported ndim for last_hidden_state: {lhs.ndim}")
            else:
                # If neither pooler_output nor last_hidden_state is found
                # For some AutoModelForImageClassification outputs, the features might be directly in 'logits'
                # or hidden states if output_hidden_states=True was passed during model init (not the case here)
                # This path indicates a fundamental mismatch in output structure.
                raise ValueError(
                    "Vision model output structure not recognized. No 'pooler_output' or 'last_hidden_state'."
                )
        
        if vision_output is None: # Should ideally be caught by exceptions above
            raise ValueError("Failed to extract vision_output.")

        # Ensure the output is 2D: (batch_size, feature_dim)
        # This is a critical check. If it's not 2D here, the projection layer will fail.
        if vision_output.ndim != 2:
            # Attempt to view it as (batch_size, -1) if it's a product that makes sense,
            # but this is risky and indicates an issue in feature extraction logic.
            # For example, if it's (batch_size, features, 1, 1), squeeze it.
            if vision_output.ndim == 4 and vision_output.shape[2] == 1 and vision_output.shape[3] == 1:
                vision_output = vision_output.squeeze(-1).squeeze(-1)
            else:
                # If it's already something like (131072, 1), this means the problem is deeper
                # or happened inside the Hugging Face model's forward pass for this specific model type.
                raise ValueError(
                    f"Vision output from _get_vision_features is not 2D (batch_size, feature_dim). "
                    f"Got shape: {vision_output.shape}. Expected feature dim: {self.vision_config['dim']}"
                )
        
        # Final check on the feature dimension
        if vision_output.shape[1] != self.vision_config['dim']:
            raise ValueError(
                f"Vision output feature dimension ({vision_output.shape[1]}) "
                f"does not match expected config dimension ({self.vision_config['dim']})."
            )
            
        return vision_output

    def _get_language_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extracts language features from text inputs using the main language model."""
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            lang_feat = outputs.pooler_output
        else: 
            lang_feat = outputs.last_hidden_state.mean(dim=1)
        return lang_feat

    def _get_clip_text_features(self, clip_input_ids: torch.Tensor, clip_attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Extracts text features using the CLIP text model, if available."""
        if hasattr(self, 'clip_text_model') and self.clip_text_model is not None:
            outputs = self.clip_text_model(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
            return outputs.pooler_output 
        return None

    def _apply_attention_fusion(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor, vision_emb: torch.Tensor,
        language_emb: torch.Tensor, numerical_emb: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Combines features using self-attention and prepares for final fusion layers."""
        features_stacked = torch.stack([user_emb, item_emb, vision_emb, language_emb, numerical_emb], dim=0)
        attended_features, _ = self.attention(features_stacked, features_stacked, features_stacked)
        combined = attended_features.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return combined

    def forward(
        self, user_idx: torch.Tensor, item_idx: torch.Tensor, image: torch.Tensor,
        text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        clip_text_input_ids: Optional[torch.Tensor] = None, 
        clip_text_attention_mask: Optional[torch.Tensor] = None, 
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        batch_size = user_idx.size(0)

        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        raw_vision_output = self._get_vision_features(image)

        # Features for Main Recommendation Task
        projected_vision_emb_main_task = self.vision_projection(raw_vision_output)
        raw_language_feat_main_task = self._get_language_features(text_input_ids, text_attention_mask)
        projected_language_emb_main_task = self.language_projection(raw_language_feat_main_task)
        projected_numerical_emb_main_task = self.numerical_projection(numerical_features)

        # Features for Contrastive Loss (only if return_embeddings is True)
        vision_features_for_contrastive_loss = None
        text_features_for_contrastive_loss = None

        if return_embeddings: 
            if self.use_contrastive and hasattr(self, 'clip_text_model'):
                if hasattr(self, 'vision_contrastive_projection'):
                    vision_features_for_contrastive_loss = self.vision_contrastive_projection(raw_vision_output)
                
                if clip_text_input_ids is not None and clip_text_attention_mask is not None:
                    raw_clip_text_output = self._get_clip_text_features(clip_text_input_ids, clip_text_attention_mask)
                    
                    if hasattr(self, 'text_contrastive_projection') and raw_clip_text_output is not None:
                        text_features_for_contrastive_loss = self.text_contrastive_projection(raw_clip_text_output)
                else:
                    raise ValueError(
                        "CLIP text input IDs and attention mask must be provided "
                        "when 'use_contrastive' is True and 'return_embeddings' is True."
                    )
        
        # Fusion and Prediction for Main Task
        combined_features = self._apply_attention_fusion(
            user_emb, item_emb, projected_vision_emb_main_task,
            projected_language_emb_main_task, projected_numerical_emb_main_task, batch_size
        )
        output = self.fusion(combined_features)

        if return_embeddings:
            return output, vision_features_for_contrastive_loss, text_features_for_contrastive_loss, projected_vision_emb_main_task
        
        return output

    def get_item_embedding(
        self, item_idx: torch.Tensor, image: torch.Tensor, text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor, numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a comprehensive item embedding by concatenating its ID-based embedding
        with projected modal features. For inference or analysis purposes.
        """
        with torch.no_grad(): 
            base_item_emb = self.item_embedding(item_idx)
            raw_vision_feat = self._get_vision_features(image)
            projected_vision_emb = self.vision_projection(raw_vision_feat)
            raw_language_feat = self._get_language_features(text_input_ids, text_attention_mask)
            projected_language_emb = self.language_projection(raw_language_feat)
            projected_numerical_emb = self.numerical_projection(numerical_features)

            item_full_embedding = torch.cat(
                [base_item_emb, projected_vision_emb, projected_language_emb, projected_numerical_emb],
                dim=-1 
            )
        return item_full_embedding


class EnhancedMultimodalRecommender(PretrainedMultimodalRecommender):
    """
    An enhanced version of the multimodal recommender with cross-modal attention.
    Inherits all configurable parameters from PretrainedMultimodalRecommender.
    """
    def __init__(
        self, 
        *args, 
        use_cross_modal_attention: bool = True,
        cross_modal_attention_weight: float = 0.5,
        **kwargs
    ):
        """
        Initializes the EnhancedMultimodalRecommender.
        
        Args:
            use_cross_modal_attention: Whether to use cross-modal attention
            cross_modal_attention_weight: Weight for cross-modal attention contribution
            *args, **kwargs: Arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        
        self.use_cross_modal_attention = use_cross_modal_attention
        self.cross_modal_attention_weight = cross_modal_attention_weight
        
        if self.use_cross_modal_attention:
            from .layers import CrossModalAttention 
            self.vision_text_attention = CrossModalAttention(self.embedding_dim)
            self.text_vision_attention = CrossModalAttention(self.embedding_dim)
    
    def _apply_attention_fusion(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor, vision_emb: torch.Tensor,
        language_emb: torch.Tensor, numerical_emb: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Override to add cross-modal attention if enabled."""
        
        # Apply cross-modal attention if enabled
        if self.use_cross_modal_attention and hasattr(self, 'vision_text_attention'):
            vision_enhanced, text_enhanced = self._apply_cross_modal_fusion(vision_emb, language_emb)
            vision_emb = vision_enhanced
            language_emb = text_enhanced
        
        # Continue with standard attention fusion
        return super()._apply_attention_fusion(
            user_emb, item_emb, vision_emb, language_emb, numerical_emb, batch_size
        )
        
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
        vision_contextualized_by_text = self.vision_text_attention(vision_emb, language_emb)
        text_contextualized_by_vision = self.text_vision_attention(language_emb, vision_emb)

        # Weighted combination with original embeddings
        vision_enhanced = vision_emb + self.cross_modal_attention_weight * vision_contextualized_by_text
        text_enhanced = language_emb + self.cross_modal_attention_weight * text_contextualized_by_vision

        return vision_enhanced, text_enhanced