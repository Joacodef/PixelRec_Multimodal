# src/models/multimodal.py
"""
Defines the main architecture for the multimodal recommendation system.

This module contains the MultimodalRecommender class, which integrates various
data modalities—such as user/item IDs, visual features from images, textual
content, and numerical metadata—to generate personalized recommendations. The
architecture is designed to be flexible, allowing for different combinations of
pre-trained models and fusion strategies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    CLIPVisionModel,
    CLIPTextModel,
    Dinov2Model
)
from typing import Optional, Tuple, Union, List

from ..config import MODEL_CONFIGS

from .layers import AttentionFusionLayer, GatedFusionLayer

from src.config import MODEL_CONFIGS
import torch.nn as nn


class MultimodalRecommender(nn.Module):
    """
    A multimodal recommender model that fuses embeddings from multiple sources.

    This model creates a comprehensive user-item interaction representation by
    combining traditional user and item embeddings with deep features extracted
    from various modalities. It uses pre-trained models for vision and language
    processing, projects their outputs into a common embedding space, and then
    fuses all features using a self-attention mechanism followed by a final
    prediction network.
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_tags: int,
        num_numerical_features: int,
        embedding_dim: int = 128,
        vision_model_name: Optional[str] = 'clip',
        language_model_name: Optional[str] = 'sentence-bert',
        freeze_vision: bool = True,
        freeze_language: bool = True,
        use_contrastive: bool = True,
        dropout_rate: float = 0.3,
        # Architectural parameters
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        fusion_hidden_dims: List[int] = None,
        fusion_activation: str = 'relu',
        use_batch_norm: bool = True,
        projection_hidden_dim: Optional[int] = None,
        final_activation: str = 'sigmoid',
        init_method: str = 'xavier_uniform',
        contrastive_temperature: float = 0.07,
        fusion_type: str = 'concatenate' 
    ):
        """
        Initializes the MultimodalRecommender model and its components.

        Args:
            n_users (int): The total number of unique users in the dataset.
            n_items (int): The total number of unique items in the dataset.
            n_tags (int): The total number of unique tags in the dataset.
            num_numerical_features (int): The number of numerical features
                                          associated with each item.
            embedding_dim (int): The dimensionality of the latent space for all
                                 embeddings.
            vision_model_name (str): The key for the pre-trained vision model
                                     to be used.
            language_model_name (str): The key for the pre-trained language
                                       model to be used.
            freeze_vision (bool): If True, the weights of the vision model are
                                  frozen and not updated during training.
            freeze_language (bool): If True, the weights of the language model
                                    are frozen.
            use_contrastive (bool): If True, enables an auxiliary contrastive
                                    loss to align vision and text spaces.
            dropout_rate (float): The dropout rate for regularization.
            num_attention_heads (int): The number of heads in the attention layer.
            attention_dropout (float): The dropout rate within the attention layer.
            fusion_hidden_dims (List[int]): A list of hidden layer dimensions for
                                            the final fusion network.
            fusion_activation (str): The activation function for the fusion network.
            use_batch_norm (bool): If True, applies batch normalization in the
                                   fusion network.
            projection_hidden_dim (Optional[int]): The dimension for an optional
                                                   hidden layer in modality projections.
            final_activation (str): The final activation function for the output.
            init_method (str): The weight initialization method for embeddings.
            contrastive_temperature (float): The temperature for the contrastive loss.
        """
        super(MultimodalRecommender, self).__init__()

        self.fusion_type = fusion_type

        self.n_users = n_users
        self.n_items = n_items
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_tags = n_tags

        
        self.use_contrastive = use_contrastive and vision_model_name == 'clip'
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.num_numerical_features = num_numerical_features

        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name

        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name] if vision_model_name else None
        self.language_config = MODEL_CONFIGS['language'][language_model_name] if language_model_name else None

        self._validate_model_configs()
        
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.fusion_hidden_dims = fusion_hidden_dims or [512, 256, 128]
        self.fusion_activation = fusion_activation
        self.use_batch_norm = use_batch_norm
        self.projection_hidden_dim = projection_hidden_dim
        self.final_activation = final_activation
        self.init_method = init_method
        self.contrastive_temperature = contrastive_temperature

        # Passes n_tags to the embedding initialization method.
        self._init_embeddings()
        self.vision_model = None
        self.clip_text_model = None
        if self.vision_model_name:
            self._init_vision_model(self.vision_model_name, freeze_vision)
        
        self.language_model = None
        if self.language_model_name:
            self._init_language_model(self.language_model_name, freeze_language)

        self._init_projection_layers()
        self._init_fusion_network()

    def _get_activation(self, activation_name: str) -> nn.Module:
        """
        Retrieves a PyTorch activation function module based on its name.

        Args:
            activation_name (str): The name of the activation function (e.g., 'relu').

        Returns:
            nn.Module: The corresponding PyTorch activation function module.
        """
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'silu': nn.SiLU()
        }
        return activations.get(activation_name.lower(), nn.ReLU())

    def _init_embeddings(self):
        """
        Initializes the user, item, and tag embedding layers.

        This method creates the embedding layers and applies a specified weight
        initialization method for better training stability.
        """
        # Initializes the embedding layers for users, items, and the new tag feature.
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.tag_embedding = nn.Embedding(self.n_tags, self.embedding_dim)
        
        # Retrieves the configured weight initialization method.
        init_method = self.init_method.lower()

        # Apply the configured weight initialization method to all embedding layers.
        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)
            nn.init.xavier_uniform_(self.tag_embedding.weight)
        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(self.user_embedding.weight)
            nn.init.xavier_normal_(self.item_embedding.weight)
            nn.init.xavier_normal_(self.tag_embedding.weight)
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.user_embedding.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.item_embedding.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.tag_embedding.weight, nonlinearity='relu')
        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.user_embedding.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.item_embedding.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.tag_embedding.weight, nonlinearity='relu')
        else:
            # Defaults to xavier_uniform if the specified method is not recognized.
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)
            nn.init.xavier_uniform_(self.tag_embedding.weight)

    def _init_vision_model(self, model_key: str, freeze: bool):
        """
        Initializes the pre-trained vision model from Hugging Face.

        Args:
            model_key (str): The key identifying the vision model in MODEL_CONFIGS.
            freeze (bool): If True, freezes the model's weights.
        """
        hf_model_name = self.vision_config['name']
        if model_key == 'clip':
            self.vision_model = CLIPVisionModel.from_pretrained(hf_model_name)
            if self.use_contrastive:
                self.clip_text_model = CLIPTextModel.from_pretrained(hf_model_name)
        elif model_key == 'dino': 
            self.vision_model = Dinov2Model.from_pretrained(hf_model_name)
        elif model_key == 'resnet' or model_key == 'convnext':
            self.vision_model = AutoModel.from_pretrained(hf_model_name)
        else: 
            self.vision_model = AutoModelForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=self.vision_config['dim'], 
                ignore_mismatched_sizes=True
            )
        # Freeze weights if configured to do so.
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if hasattr(self, 'clip_text_model') and self.clip_text_model and freeze:
            for param in self.clip_text_model.parameters():
                param.requires_grad = False

    def _init_language_model(self, model_key: str, freeze: bool):
        """
        Initializes the pre-trained language model from Hugging Face.

        Args:
            model_key (str): The key identifying the language model in MODEL_CONFIGS.
            freeze (bool): If True, freezes the model's weights.
        """
        hf_model_name = self.language_config['name']
        self.language_model = AutoModel.from_pretrained(hf_model_name)
        if freeze:
            for param in self.language_model.parameters():
                param.requires_grad = False

    def _init_projection_layers(self):
        """
        Initializes projection layers for all modalities.

        These layers project the raw feature outputs from the pre-trained models
        into the common embedding space dimension, allowing them to be fused.
        """
        activation = self._get_activation(self.fusion_activation)
        
        # Defines the vision projection network.
        if self.vision_model:
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
        
        # Defines the language projection network.
        if self.language_model:
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
        
        # Defines the numerical features projection network.
        if self.num_numerical_features > 0:
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
        else:
            self.numerical_projection = None

        # Defines separate projection layers for contrastive loss.
        if self.use_contrastive:
            self.vision_contrastive_projection = nn.Linear(
                self.vision_config['dim'],
                self.embedding_dim
            )
            clip_text_output_dim = self._get_clip_text_output_dim()
            self.text_contrastive_projection = nn.Linear(
                clip_text_output_dim, 
                self.embedding_dim
            )
            self.temperature = nn.Parameter(torch.tensor(self.contrastive_temperature))

    def _init_fusion_network(self):
        """
        Initializes the fusion mechanism and final prediction network based on the
        configured fusion_type.
        """
        self.fusion_layer = None
        num_modalities = 3  # Start with user, item, and tag.
        if self.vision_model is not None:
            num_modalities += 1
        if self.language_model is not None:
            num_modalities += 1
        if self.num_numerical_features > 0:
            num_modalities += 1

        if self.fusion_type == 'concatenate':
            # For concatenation, the input to the next layer is the sum of all feature dimensions.
            fusion_input_dim = num_modalities * self.embedding_dim
        elif self.fusion_type == 'attention':
            self.fusion_layer = AttentionFusionLayer(
                embedding_dim=self.embedding_dim,
                num_attention_heads=self.num_attention_heads,
                dropout_rate=self.attention_dropout
            )
            # The attention layer outputs a single vector of the same embedding dimension.
            fusion_input_dim = self.embedding_dim
        elif self.fusion_type == 'gated':
            self.fusion_layer = GatedFusionLayer(
                embedding_dim=self.embedding_dim,
                num_modalities=num_modalities,
                dropout_rate=self.dropout_rate
            )
            # The gated layer outputs a single vector of the same embedding dimension.
            fusion_input_dim = self.embedding_dim
        else:
            raise ValueError(f"Unknown fusion type: '{self.fusion_type}'")

        # Dynamically build the final prediction MLP.
        activation = self._get_activation(self.fusion_activation)
        prediction_layers = []
        input_dim = fusion_input_dim
        
        for hidden_dim in self.fusion_hidden_dims:
            prediction_layers.append(nn.Linear(input_dim, hidden_dim))
            prediction_layers.append(activation)
            if self.use_batch_norm:
                prediction_layers.append(nn.BatchNorm1d(hidden_dim))
            prediction_layers.append(nn.Dropout(self.dropout_rate))
            input_dim = hidden_dim
            
        prediction_layers.append(nn.Linear(input_dim, 1))
        
        if self.final_activation == 'sigmoid':
            prediction_layers.append(nn.Sigmoid())
        elif self.final_activation == 'tanh':
            prediction_layers.append(nn.Tanh())
            
        self.prediction_network = nn.Sequential(*prediction_layers)
    
    def _get_vision_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extracts vision features from an image tensor, handling different model outputs.

        This method supports various Hugging Face vision models. It correctly
        identifies the output structure (e.g., `pooler_output` vs.
        `last_hidden_state`) and applies the necessary pooling and reshaping
        to produce a consistent 2D feature tensor (batch_size, feature_dim).

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The extracted and processed vision feature tensor.

        Raises:
            ValueError: If the model's output structure is not recognized or
                        if the final feature dimensions are incorrect.
            RuntimeError: If a memory or dimension mismatch error occurs during
                          feature extraction.
        """
        model_input = {'pixel_values': image}
        vision_output = None
        is_clip_model_type = 'clip' in self.vision_config.get('name', '').lower()

        try:
            if is_clip_model_type and hasattr(self.vision_model, 'get_image_features'):
                vision_output = self.vision_model.get_image_features(**model_input)
            else:
                outputs = self.vision_model(**model_input)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    vision_output = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    lhs = outputs.last_hidden_state
                    if lhs.ndim == 4:
                        vision_output = torch.nn.functional.adaptive_avg_pool2d(lhs, (1, 1)).squeeze(-1).squeeze(-1)
                    elif lhs.ndim == 3:
                        vision_output = lhs.mean(dim=1)
                    else:
                        raise ValueError(f"Unsupported last_hidden_state dimensions: {lhs.ndim}D with shape {lhs.shape}")
                else:
                    available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_') and hasattr(outputs, attr)]
                    raise ValueError(f"Vision model output structure not recognized. Available attributes: {available_attrs}")
        except Exception as e:
            raise RuntimeError(f"Error during vision feature extraction: {e}") from e

        if vision_output is None:
            raise ValueError("Failed to extract vision features; model returned None.")

        # This block handles cases where the output might still not be 2D.
        if vision_output.ndim != 2:
            if vision_output.ndim == 4 and vision_output.shape[2] == 1 and vision_output.shape[3] == 1:
                vision_output = vision_output.squeeze(-1).squeeze(-1)
            elif vision_output.ndim == 3 and vision_output.shape[0] == 1:
                vision_output = vision_output.squeeze(0)
            elif vision_output.ndim == 1:
                vision_output = vision_output.unsqueeze(0)
            else:
                 raise ValueError(f"Vision output has unexpected dimensions: {vision_output.ndim}D. Expected 2D.")
        
        expected_dim = self.vision_config['dim']
        actual_dim = vision_output.shape[1]
        
        if actual_dim != expected_dim:
            raise ValueError(f"Vision feature dimension mismatch. Expected: {expected_dim}, Got: {actual_dim}.")
        
        return vision_output
    
    def _get_language_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts language features from text input tensors.

        Args:
            input_ids (torch.Tensor): The tokenized input IDs.
            attention_mask (torch.Tensor): The attention mask for the input.

        Returns:
            torch.Tensor: The extracted language feature tensor.
        """
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state.mean(dim=1)

    def _get_clip_text_features(self, clip_input_ids: torch.Tensor, clip_attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extracts text features specifically using the CLIP text model.

        Args:
            clip_input_ids (torch.Tensor): The tokenized input IDs for CLIP.
            clip_attention_mask (torch.Tensor): The attention mask for the CLIP input.

        Returns:
            Optional[torch.Tensor]: The extracted feature tensor, or None if the
                                    CLIP text model is not used.
        """
        if hasattr(self, 'clip_text_model') and self.clip_text_model is not None:
            outputs = self.clip_text_model(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
            return outputs.pooler_output 
        return None

    def _apply_attention_fusion(
            self,
            user_emb: torch.Tensor,
            item_emb: torch.Tensor,
            tag_emb: torch.Tensor,
            vision_emb: torch.Tensor,
            language_emb: torch.Tensor,
            numerical_emb: torch.Tensor,
            batch_size: int
    ) -> torch.Tensor:
        """
        Applies self-attention to fuse different feature embeddings.

        Args:
            user_emb, item_emb, tag_emb, vision_emb, language_emb, numerical_emb: The
                embedding tensors for each modality.
            batch_size (int): The number of samples in the batch.

        Returns:
            torch.Tensor: The concatenated, attention-fused feature tensor.
        """
        # Stacks the embeddings for all modalities, including the new tag embedding.
        features_stacked = torch.stack([
            user_emb, item_emb, tag_emb, vision_emb, language_emb, numerical_emb
        ], dim=0)
        
        # Applies the self-attention mechanism to the stacked features.
        attended_features, _ = self.attention(features_stacked, features_stacked, features_stacked)
        
        # Reshapes the output for the final fusion network.
        return attended_features.permute(1, 0, 2).contiguous().view(batch_size, -1)

    def forward(
            self,
            user_idx: torch.Tensor,
            item_idx: torch.Tensor,
            tag_idx: torch.Tensor,
            image: torch.Tensor,
            text_input_ids: torch.Tensor,
            text_attention_mask: torch.Tensor,
            numerical_features: torch.Tensor,
            clip_text_input_ids: Optional[torch.Tensor] = None,
            clip_text_attention_mask: Optional[torch.Tensor] = None,
            return_embeddings: bool = False,
            debug_this_batch: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Defines the main forward pass of the model.
        """
        batch_size = user_idx.size(0)
        
        # Create a dynamic list to hold features from active modalities.
        features_to_fuse = []

        # Always include user, item, and tag embeddings.
        features_to_fuse.append(self.user_embedding(user_idx))
        features_to_fuse.append(self.item_embedding(item_idx))
        features_to_fuse.append(self.tag_embedding(tag_idx))
        
        raw_vision_output = None
        # Conditionally process vision features.
        if self.vision_model is not None:
            raw_vision_output = self._get_vision_features(image)
            features_to_fuse.append(self.vision_projection(raw_vision_output))
        
        # Conditionally process language features.
        if self.language_model is not None:
            raw_language_feat = self._get_language_features(text_input_ids, text_attention_mask)
            features_to_fuse.append(self.language_projection(raw_language_feat))
        
        # Conditionally process numerical features.
        if self.numerical_projection is not None and self.num_numerical_features > 0:
            features_to_fuse.append(self.numerical_projection(numerical_features))

        # Generate features specifically for the contrastive loss objective.
        vision_features_for_contrastive_loss = None
        text_features_for_contrastive_loss = None
        if self.use_contrastive and raw_vision_output is not None:
            vision_features_for_contrastive_loss = self.vision_contrastive_projection(raw_vision_output)
            raw_clip_text_output = self._get_clip_text_features(clip_text_input_ids, clip_text_attention_mask)
            if raw_clip_text_output is not None:
                text_features_for_contrastive_loss = self.text_contrastive_projection(raw_clip_text_output)

        # Apply the selected fusion method to the dynamically built list of features.
        if self.fusion_type == 'concatenate':
            fused_features = torch.cat(features_to_fuse, dim=1)
        else:
            fused_features = self.fusion_layer(features_to_fuse)
        
        output = self.prediction_network(fused_features)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)

        if return_embeddings:
            # For the fourth return value, we return the main task's vision embedding.
            projected_vision_emb_main_task = self.vision_projection(raw_vision_output) if raw_vision_output is not None else None
            
            if vision_features_for_contrastive_loss is not None:
                vision_features_for_contrastive_loss = F.normalize(vision_features_for_contrastive_loss, p=2, dim=-1)
            if text_features_for_contrastive_loss is not None:
                text_features_for_contrastive_loss = F.normalize(text_features_for_contrastive_loss, p=2, dim=-1)
            
            return output, vision_features_for_contrastive_loss, text_features_for_contrastive_loss, projected_vision_emb_main_task
        
        return output

    def get_item_embedding(
        self, item_idx: torch.Tensor, image: torch.Tensor, text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor, numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a comprehensive item embedding for analysis or inference.

        Args:
            item_idx, image, text_input_ids, text_attention_mask, numerical_features:
                The tensors representing a single item's data.

        Returns:
            torch.Tensor: A single tensor representing the combined features of the item.
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

    def _validate_model_configs(self):
        """
        Validates that the selected model configurations are available and consistent.
        """
        if self.vision_model_name and self.vision_model_name not in MODEL_CONFIGS['vision']:
            raise ValueError(f"Vision model '{self.vision_model_name}' not found in MODEL_CONFIGS.")
        if self.language_model_name and self.language_model_name not in MODEL_CONFIGS['language']:
            raise ValueError(f"Language model '{self.language_model_name}' not found in MODEL_CONFIGS.")
        
        if self.vision_config:
            vision_dim = self.vision_config.get('dim')
            if not isinstance(vision_dim, int) or vision_dim <= 0:
                raise ValueError(f"Invalid vision model dimension: {vision_dim}")

    def _get_clip_text_output_dim(self) -> int:
        """
        Determines the output dimension of the CLIP text model.

        Returns:
            int: The output dimension of the CLIP text model.
        """
        if not hasattr(self, 'clip_text_model') or self.clip_text_model is None:
            return MODEL_CONFIGS['vision']['clip'].get('text_dim', 512)
        try:
            if hasattr(self.clip_text_model, 'text_projection'):
                return self.clip_text_model.text_projection.in_features
            if hasattr(self.clip_text_model, 'config') and hasattr(self.clip_text_model.config, 'text_config'):
                return self.clip_text_model.config.text_config.hidden_size
        except Exception as e:
            print(f"Warning: Could not determine CLIP text output dimension: {e}.")
        return 512

        
# Backward compatibility alias
PretrainedMultimodalRecommender = MultimodalRecommender