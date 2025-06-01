"""
Multimodal recommender model architecture
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoModelForImageClassification,
    CLIPVisionModel, 
    CLIPTextModel,
    DinoModel
)
from typing import Optional, Tuple

from ..config import MODEL_CONFIGS


class PretrainedMultimodalRecommender(nn.Module):
    """Multimodal recommender using pre-trained models"""
    
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
        """
        Initialize the multimodal recommender.
        
        Args:
            n_users: Number of unique users
            n_items: Number of unique items
            embedding_dim: Dimension of embeddings
            vision_model_name: Name of vision model ('clip', 'dino', 'resnet', 'convnext')
            language_model_name: Name of language model
            freeze_vision: Whether to freeze vision model weights
            freeze_language: Whether to freeze language model weights
            use_contrastive: Whether to use contrastive learning (only for CLIP)
            dropout_rate: Dropout rate for regularization
        """
        super(PretrainedMultimodalRecommender, self).__init__()
        
        # Store configurations
        self.vision_config = MODEL_CONFIGS['vision'][vision_model_name]
        self.language_config = MODEL_CONFIGS['language'][language_model_name]
        self.use_contrastive = use_contrastive and vision_model_name == 'clip'
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        # Initialize embeddings
        self._init_embeddings(n_users, n_items)
        
        # Initialize vision model
        self._init_vision_model(vision_model_name, freeze_vision)
        
        # Initialize language model
        self._init_language_model(language_model_name, freeze_language)
        
        # Initialize projection layers
        self._init_projection_layers()
        
        # Initialize fusion network
        self._init_fusion_network()
    
    def _init_embeddings(self, n_users: int, n_items: int):
        """Initialize user and item embeddings"""
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(n_items, self.embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def _init_vision_model(self, model_name: str, freeze: bool):
        """Initialize vision model based on selection"""
        if model_name == 'clip':
            self.vision_model = CLIPVisionModel.from_pretrained(
                self.vision_config['name']
            )
            if self.use_contrastive:
                self.clip_text_model = CLIPTextModel.from_pretrained(
                    self.vision_config['name']
                )
        elif model_name == 'dino':
            self.vision_model = DinoModel.from_pretrained(
                self.vision_config['name']
            )
        else:
            self.vision_model = AutoModelForImageClassification.from_pretrained(
                self.vision_config['name'], 
                num_labels=self.embedding_dim,
                ignore_mismatched_sizes=True
            )
        
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
    
    def _init_language_model(self, model_name: str, freeze: bool):
        """Initialize language model"""
        self.language_model = AutoModel.from_pretrained(
            self.language_config['name']
        )
        
        if freeze:
            for param in self.language_model.parameters():
                param.requires_grad = False
    
    def _init_projection_layers(self):
        """Initialize projection layers for each modality"""
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
            nn.Linear(7, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        if self.use_contrastive:
            self.contrastive_projection = nn.Linear(
                self.vision_config['dim'], 
                self.embedding_dim
            )
            self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def _init_fusion_network(self):
        """Initialize the fusion network"""
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, 
            num_heads=4,
            dropout=self.dropout_rate
        )
        
        # Final fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 5, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor, 
        image: torch.Tensor, 
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor, 
        numerical_features: torch.Tensor, 
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            user_idx: User indices
            item_idx: Item indices
            image: Image tensors
            text_input_ids: Text input IDs
            text_attention_mask: Text attention mask
            numerical_features: Numerical features
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Prediction scores and optionally embeddings
        """
        batch_size = user_idx.size(0)
        
        # Get embeddings
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        # Get vision features
        vision_output = self._get_vision_features(image)
        vision_emb = self.vision_projection(vision_output)
        
        # Get language features
        language_output = self._get_language_features(
            text_input_ids, 
            text_attention_mask
        )
        language_emb = self.language_projection(language_output)
        
        # Get numerical features
        numerical_emb = self.numerical_projection(numerical_features)
        
        # Get CLIP text features for contrastive learning
        clip_text_features = None
        if self.use_contrastive and hasattr(self, 'clip_text_model'):
            clip_text_features = self._get_clip_text_features(
                text_input_ids, 
                text_attention_mask
            )
        
        # Apply attention-based fusion
        combined = self._apply_attention_fusion(
            user_emb, 
            item_emb, 
            vision_emb, 
            language_emb, 
            numerical_emb, 
            batch_size
        )
        
        # Final prediction
        output = self.fusion(combined)
        
        if return_embeddings:
            return output, vision_output, clip_text_features, vision_emb
        
        return output
    
    def _get_vision_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract vision features from image"""
        if hasattr(self.vision_model, 'get_image_features'):
            vision_output = self.vision_model.get_image_features(image)
        else:
            vision_output = self.vision_model(image).pooler_output
        return vision_output
    
    def _get_language_features(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract language features from text"""
        language_output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooler output if available, otherwise mean pooling
        if hasattr(language_output, 'pooler_output') and language_output.pooler_output is not None:
            language_feat = language_output.pooler_output
        else:
            language_feat = language_output.last_hidden_state.mean(dim=1)
        
        return language_feat
    
    def _get_clip_text_features(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get CLIP text features for contrastive learning"""
        clip_text_output = self.clip_text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return clip_text_output.pooler_output
    
    def _apply_attention_fusion(
        self, 
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        vision_emb: torch.Tensor,
        language_emb: torch.Tensor,
        numerical_emb: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Apply attention-based fusion of features"""
        # Stack features for attention
        features = torch.stack(
            [user_emb, item_emb, vision_emb, language_emb, numerical_emb], 
            dim=1
        )
        
        # Apply self-attention
        attended_features, _ = self.attention(features, features, features)
        
        # Concatenate all features
        combined = attended_features.view(batch_size, -1)
        
        return combined
    
    def get_item_embedding(
        self, 
        item_idx: torch.Tensor,
        image: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """Get the complete embedding for an item"""
        with torch.no_grad():
            # Get all embeddings
            item_emb = self.item_embedding(item_idx)
            vision_output = self._get_vision_features(image)
            vision_emb = self.vision_projection(vision_output)
            language_output = self._get_language_features(
                text_input_ids, 
                text_attention_mask
            )
            language_emb = self.language_projection(language_output)
            numerical_emb = self.numerical_projection(numerical_features)
            
            # Concatenate all item-related embeddings
            item_full_embedding = torch.cat(
                [item_emb, vision_emb, language_emb, numerical_emb], 
                dim=-1
            )
            
        return item_full_embedding