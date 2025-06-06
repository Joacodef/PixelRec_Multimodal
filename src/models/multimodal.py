# src/models/multimodal.py

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


class MultimodalRecommender(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        num_numerical_features: int, # Added num_numerical_features
        embedding_dim: int = 128,
        vision_model_name: str = 'clip',
        language_model_name: str = 'sentence-bert',
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
        contrastive_temperature: float = 0.07
    ):
        super(MultimodalRecommender, self).__init__()
        
        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name]
        self.language_config = MODEL_CONFIGS['language'][language_model_name]
        self.use_contrastive = use_contrastive and vision_model_name == 'clip'
        self.language_model_name = language_model_name
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.num_numerical_features = num_numerical_features # Use the passed argument

        self.vision_model_name = vision_model_name

        # Validate configurations
        self._validate_model_configs()
        
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
        elif model_key == 'resnet' or model_key == 'convnext':
            # Load as a base model to get features before a classification head
            self.vision_model = AutoModel.from_pretrained(hf_model_name)
        else: 
            print(f"Warning: Vision model key '{model_key}' not explicitly handled for base model loading. Defaulting to AutoModelForImageClassification.")
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
                nn.Linear(self.num_numerical_features, self.projection_hidden_dim), # Use self.num_numerical_features
                activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.projection_hidden_dim, self.embedding_dim),
                activation,
                nn.Dropout(self.dropout_rate)
            )
        else:
            self.numerical_projection = nn.Sequential(
                nn.Linear(self.num_numerical_features, self.embedding_dim), # Use self.num_numerical_features
                activation,
                nn.Dropout(self.dropout_rate)
            )

        if self.use_contrastive:
            self.vision_contrastive_projection = nn.Linear(
                self.vision_config['dim'],
                self.embedding_dim
            )
            
            # Dynamically determine CLIP text model output dimension
            clip_text_output_dim = self._get_clip_text_output_dim()
            self.text_contrastive_projection = nn.Linear(
                clip_text_output_dim, 
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
        # Validate input tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image)}")
        
        if image.dim() not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D tensor with shape {image.shape}")
        
        model_input = {'pixel_values': image}
        vision_output = None

        # Check if the model is a CLIP vision model and has 'get_image_features'
        is_clip_model_type = 'clip' in self.vision_config.get('name', '').lower()

        try:
            if is_clip_model_type and hasattr(self.vision_model, 'get_image_features'):
                vision_output = self.vision_model.get_image_features(**model_input)
            else:
                # For other models like ResNet, DINOv2, ConvNeXT
                outputs = self.vision_model(**model_input)
                
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    vision_output = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    lhs = outputs.last_hidden_state
                    if lhs.ndim == 4:  # CNN feature maps (batch_size, num_channels, height, width)
                        vision_output = torch.nn.functional.adaptive_avg_pool2d(lhs, (1, 1)).squeeze(-1).squeeze(-1)
                    elif lhs.ndim == 3:  # ViT-like sequence outputs (batch_size, seq_len, hidden_dim)
                        vision_output = lhs.mean(dim=1)  # Pool over sequence length
                    else:
                        raise ValueError(f"Unsupported last_hidden_state dimensions: {lhs.ndim}D with shape {lhs.shape}")
                else:
                    # Check for other possible output attributes
                    available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_') and hasattr(outputs, attr)]
                    raise ValueError(
                        f"Vision model output structure not recognized for model '{getattr(self, 'vision_model_name', 'unknown')}'. "
                        f"No 'pooler_output' or 'last_hidden_state' found. "
                        f"Available attributes: {available_attrs}"
                    )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"GPU out of memory during vision feature extraction. "
                    f"Input shape: {image.shape}, Model: {getattr(self, 'vision_model_name', 'unknown')}. "
                    f"Original error: {e}"
                ) from e
            elif "size mismatch" in str(e).lower() or "dimension" in str(e).lower():
                raise RuntimeError(
                    f"Tensor dimension mismatch in vision model '{getattr(self, 'vision_model_name', 'unknown')}'. "
                    f"Input shape: {image.shape}, Expected input format may be different. "
                    f"Original error: {e}"
                ) from e
            else:
                raise RuntimeError(
                    f"Runtime error during vision feature extraction with model '{getattr(self, 'vision_model_name', 'unknown')}'. "
                    f"Input shape: {image.shape}. Original error: {e}"
                ) from e
        
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during vision feature extraction with model '{getattr(self, 'vision_model_name', 'unknown')}'. "
                f"Input shape: {image.shape}. Error type: {type(e).__name__}. "
                f"Original error: {e}"
            ) from e
        
        # Validate that we got an output
        if vision_output is None:
            raise ValueError(
                f"Failed to extract vision features from model '{getattr(self, 'vision_model_name', 'unknown')}'. "
                f"Model returned None output. Input shape: {image.shape}"
            )

        # Validate and fix output dimensions
        if vision_output.ndim != 2:
            # Attempt to fix common dimension issues
            if vision_output.ndim == 4 and vision_output.shape[2] == 1 and vision_output.shape[3] == 1:
                vision_output = vision_output.squeeze(-1).squeeze(-1)
            elif vision_output.ndim == 3 and vision_output.shape[0] == 1:
                # Batch dimension of 1 that can be squeezed
                vision_output = vision_output.squeeze(0)
            elif vision_output.ndim == 1:
                # Add batch dimension if missing
                vision_output = vision_output.unsqueeze(0)
            else:
                raise ValueError(
                    f"Vision output has unexpected dimensions: {vision_output.ndim}D with shape {vision_output.shape}. "
                    f"Expected 2D (batch_size, feature_dim). Model: {getattr(self, 'vision_model_name', 'unknown')}, "
                    f"Input shape: {image.shape}"
                )
        
        # Final validation of feature dimension
        expected_dim = self.vision_config['dim']
        actual_dim = vision_output.shape[1] if vision_output.ndim >= 2 else vision_output.numel()
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"Vision feature dimension mismatch for model '{getattr(self, 'vision_model_name', 'unknown')}'. "
                f"Expected: {expected_dim}, Got: {actual_dim}. "
                f"Output shape: {vision_output.shape}, Input shape: {image.shape}. "
                f"This may indicate a configuration error in MODEL_CONFIGS or model version mismatch."
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
        return_embeddings: bool = False,
        debug_this_batch: bool = False # Add this argument for targeted debugging
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:
        
        def check_tensor(tensor: torch.Tensor, name: str):
            # Helper function to check and print if a tensor contains NaN or Inf.
            if debug_this_batch: # Only print for the problematic batch
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"DEBUG: NaN/Inf detected in '{name}'")
                    print(f"  Shape: {tensor.shape}")
                    print(f"  Contains NaN: {torch.isnan(tensor).any().item()}")
                    print(f"  Contains Inf: {torch.isinf(tensor).any().item()}")
                    finite_vals = tensor[torch.isfinite(tensor)]
                    if finite_vals.numel() > 0:
                        print(f"  Finite min: {finite_vals.min().item()}, max: {finite_vals.max().item()}, mean: {finite_vals.mean().item()}")
                    else:
                        print(f"  No finite values in '{name}'.")
                else:
                    print(f"DEBUG: '{name}' is clean. Shape: {tensor.shape}, Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.mean().item()}")


        batch_size = user_idx.size(0)

        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        raw_vision_output = self._get_vision_features(image)
        
        projected_vision_emb_main_task = self.vision_projection(raw_vision_output)

        raw_language_feat_main_task = self._get_language_features(text_input_ids, text_attention_mask)
        projected_language_emb_main_task = self.language_projection(raw_language_feat_main_task)

        projected_numerical_emb_main_task = self.numerical_projection(numerical_features)

        # Features for Contrastive Loss (only if return_embeddings is True)
        vision_features_for_contrastive_loss = None
        text_features_for_contrastive_loss = None

        if return_embeddings and self.use_contrastive: 
            if hasattr(self, 'vision_contrastive_projection'):
                vision_features_for_contrastive_loss = self.vision_contrastive_projection(raw_vision_output)
            
            if hasattr(self, 'clip_text_model') and clip_text_input_ids is not None and clip_text_attention_mask is not None:
                raw_clip_text_output = self._get_clip_text_features(clip_text_input_ids, clip_text_attention_mask)
                if raw_clip_text_output is not None and hasattr(self, 'text_contrastive_projection'): 
                    text_features_for_contrastive_loss = self.text_contrastive_projection(raw_clip_text_output)
        
        features_stacked = torch.stack([
            user_emb, item_emb, 
            projected_vision_emb_main_task,
            projected_language_emb_main_task, 
            projected_numerical_emb_main_task
        ], dim=0) 
        
        attended_features, _ = self.attention(features_stacked, features_stacked, features_stacked)

        combined_features = attended_features.permute(1, 0, 2).contiguous().view(batch_size, -1)

        output = self.fusion(combined_features)
        
        # Sanitize the output to prevent NaN/Inf values from crashing downstream processes.
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
        

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

    def get_user_item_score(self, user_idx: torch.Tensor, item_idx: torch.Tensor, 
                           image: torch.Tensor, text_input_ids: torch.Tensor,
                           text_attention_mask: torch.Tensor, numerical_features: torch.Tensor) -> torch.Tensor:
        """
        Get the prediction score for a specific user-item pair.
        Useful for ranking evaluation.
        """
        with torch.no_grad():
            # Forward pass to get the score
            output = self.forward(
                user_idx=user_idx,
                item_idx=item_idx,
                image=image,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                numerical_features=numerical_features,
                return_embeddings=False
            )
            return output.squeeze()  # Remove batch dimension if single item

    def _validate_model_configs(self):
        """Validate that model configurations are available and consistent."""
        # Check if vision model config exists
        if self.vision_model_name not in MODEL_CONFIGS['vision']:
            available_models = list(MODEL_CONFIGS['vision'].keys())
            raise ValueError(
                f"Vision model '{self.vision_model_name}' not found in MODEL_CONFIGS. "
                f"Available options: {available_models}"
            )
        
        # Check if language model config exists
        if self.language_model_name not in MODEL_CONFIGS['language']:
            available_models = list(MODEL_CONFIGS['language'].keys())
            raise ValueError(
                f"Language model '{self.language_model_name}' not found in MODEL_CONFIGS. "
                f"Available options: {available_models}"
            )
        
        # Validate dimensions are positive integers
        vision_dim = self.vision_config.get('dim')
        language_dim = self.language_config.get('dim')
        
        if not isinstance(vision_dim, int) or vision_dim <= 0:
            raise ValueError(f"Invalid vision model dimension: {vision_dim}")
        
        if not isinstance(language_dim, int) or language_dim <= 0:
            raise ValueError(f"Invalid language model dimension: {language_dim}")
        
        # Validate contrastive learning setup
        if self.use_contrastive and self.vision_model_name != 'clip':
            print(f"Warning: Contrastive learning enabled but vision model is '{self.vision_model_name}', not 'clip'. "
                f"This may not work as expected.")

    def _get_clip_text_output_dim(self) -> int:
        """
        Dynamically determine the output dimension of the CLIP text model.
        
        Returns:
            int: The output dimension of the CLIP text model
        """
        if not hasattr(self, 'clip_text_model') or self.clip_text_model is None:
            # Fallback: try to get from config or use reasonable default
            clip_text_dim = MODEL_CONFIGS['vision']['clip'].get('text_dim')
            if clip_text_dim is not None:
                return clip_text_dim
            
            # Last resort: use the most common CLIP text dimension
            print("Warning: Could not determine CLIP text output dimension. Using default 512.")
            return 512
        
        # Try to get the actual dimension from the model
        try:
            # Most CLIP text models have a text_projection layer
            if hasattr(self.clip_text_model, 'text_projection'):
                return self.clip_text_model.text_projection.in_features
            
            # Alternative: check the final layer of the text model
            if hasattr(self.clip_text_model, 'text_model') and hasattr(self.clip_text_model.text_model, 'final_layer_norm'):
                return self.clip_text_model.text_model.final_layer_norm.normalized_shape[0]
            
            # Another alternative: use the pooler output if available
            if hasattr(self.clip_text_model, 'config') and hasattr(self.clip_text_model.config, 'text_config'):
                text_config = self.clip_text_model.config.text_config
                if hasattr(text_config, 'hidden_size'):
                    return text_config.hidden_size
            
            # If all else fails, use a test forward pass (less efficient but reliable)
            return self._probe_clip_text_output_dim()
            
        except Exception as e:
            print(f"Warning: Error determining CLIP text output dimension: {e}. Using default 512.")
            return 512

    def _probe_clip_text_output_dim(self) -> int:
        """
        Determine CLIP text output dimension by running a test forward pass.
        
        Returns:
            int: The output dimension
        """
        try:
            # Create a dummy input
            dummy_input_ids = torch.zeros(1, 77, dtype=torch.long)  # Standard CLIP sequence length
            dummy_attention_mask = torch.ones(1, 77, dtype=torch.long)
            
            # Run forward pass
            with torch.no_grad():
                outputs = self.clip_text_model(
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask
                )
                
            # Get the pooler output dimension
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output.shape[-1]
            
            # Fallback to last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state.shape[-1]
                
        except Exception as e:
            print(f"Warning: Error probing CLIP text output dimension: {e}")
        
        return 512  # Safe default


# Backward compatibility alias
PretrainedMultimodalRecommender = MultimodalRecommender