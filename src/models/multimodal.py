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
from typing import Optional, Tuple, Union

from ..config import MODEL_CONFIGS # Relative import for configuration

class PretrainedMultimodalRecommender(nn.Module):
    # __init__, _init_embeddings, _init_vision_model, 
    # _init_language_model, _init_projection_layers, _init_fusion_network,
    # _get_vision_features, _get_language_features, _get_clip_text_features,
    # _apply_attention_fusion, get_item_embedding methods remain as previously defined.
    # Ensure _init_projection_layers correctly defines:
    # self.vision_projection, self.language_projection, self.numerical_projection
    # self.vision_contrastive_projection, self.text_contrastive_projection (if use_contrastive)

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
        dropout_rate: float = 0.3
    ):
        super(PretrainedMultimodalRecommender, self).__init__()

        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name]
        self.language_config = MODEL_CONFIGS['language'][language_model_name]
        self.use_contrastive = use_contrastive and vision_model_name == 'clip'
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
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
        elif model_key == 'dino': 
            self.vision_model = Dinov2Model.from_pretrained(hf_model_name)
        else: 
            self.vision_model = AutoModelForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=self.vision_config['dim'], 
                ignore_mismatched_sizes=True
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
            nn.Linear(self.num_numerical_features, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        if self.use_contrastive:
            self.vision_contrastive_projection = nn.Linear(
                self.vision_config['dim'],
                self.embedding_dim
            )
            # Assumed raw output dimension of CLIPTextModel's pooler_output.
            # Based on previous error analysis, this appeared to be 512.
            # Standard 'openai/clip-vit-base-patch32' text encoder output is 768.
            # This value should be verified for the specific model checkpoint used.
            clip_text_model_raw_output_dim = 512 
            self.text_contrastive_projection = nn.Linear(
                clip_text_model_raw_output_dim, 
                self.embedding_dim
            )
            self.temperature = nn.Parameter(torch.tensor(0.07))


    def _init_fusion_network(self):
        """Initializes the attention-based fusion network and final prediction layers."""
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, num_heads=4, dropout=self.dropout_rate
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 5, 512), nn.ReLU(), nn.Dropout(self.dropout_rate), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(self.dropout_rate), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def _get_vision_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extracts vision features from the image tensor using the vision model."""
        model_input = {'pixel_values': image}
        
        if hasattr(self.vision_model, 'get_image_features'): 
            vision_output = self.vision_model.get_image_features(**model_input)
        else:
            outputs = self.vision_model(**model_input)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                vision_output = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'): 
                vision_output = outputs.last_hidden_state.mean(dim=1)
            else:
                raise ValueError("Vision model output structure not recognized for feature extraction.")
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
        return_embeddings: bool = False # Default is False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        batch_size = user_idx.size(0)

        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        raw_vision_output = self._get_vision_features(image) 
        
        # --- Features for Main Recommendation Task ---
        projected_vision_emb_main_task = self.vision_projection(raw_vision_output)
        raw_language_feat_main_task = self._get_language_features(text_input_ids, text_attention_mask)
        projected_language_emb_main_task = self.language_projection(raw_language_feat_main_task)
        projected_numerical_emb_main_task = self.numerical_projection(numerical_features)

        # --- Features for Contrastive Loss (only if return_embeddings is True) ---
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
                    # This error is raised if embeddings are requested for contrastive loss,
                    # contrastive learning is enabled, but necessary CLIP text inputs are missing.
                    raise ValueError(
                        "CLIP text input IDs and attention mask must be provided "
                        "when 'use_contrastive' is True and 'return_embeddings' is True."
                    )
        
        # --- Fusion and Prediction for Main Task ---
        combined_features = self._apply_attention_fusion(
            user_emb, item_emb, projected_vision_emb_main_task,
            projected_language_emb_main_task, projected_numerical_emb_main_task, batch_size
        )
        output = self.fusion(combined_features)

        if return_embeddings:
            return output, vision_features_for_contrastive_loss, text_features_for_contrastive_loss, projected_vision_emb_main_task
        
        return output # Default return: only the prediction output

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
    An enhanced version of the multimodal recommender, potentially incorporating
    additional mechanisms like cross-modal attention.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the EnhancedMultimodalRecommender.
        Inherits from PretrainedMultimodalRecommender and may add or override components.
        """
        super().__init__(*args, **kwargs)

        from .layers import CrossModalAttention 
        self.vision_text_attention = CrossModalAttention(self.embedding_dim)
        self.text_vision_attention = CrossModalAttention(self.embedding_dim)
        
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

        vision_enhanced = vision_emb + 0.5 * vision_contextualized_by_text
        text_enhanced = language_emb + 0.5 * text_contextualized_by_vision

        return vision_enhanced, text_enhanced
