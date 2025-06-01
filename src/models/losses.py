"""
Loss functions for multimodal recommender system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """CLIP-style contrastive loss for image-text alignment"""
    
    def __init__(self, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between image and text features.
        
        Args:
            image_features: Image feature vectors
            text_features: Text feature vectors
            temperature: Temperature parameter (optional, uses default if not provided)
            
        Returns:
            Contrastive loss value
        """
        if temperature is None:
            temperature = self.temperature
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / temperature
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        # Compute cross entropy loss in both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss"""
    
    def __init__(self):
        super(BPRLoss, self).__init__()
    
    def forward(
        self, 
        positive_scores: torch.Tensor, 
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BPR loss.
        
        Args:
            positive_scores: Scores for positive items
            negative_scores: Scores for negative items
            
        Returns:
            BPR loss value
        """
        return -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))


class MultimodalRecommenderLoss(nn.Module):
    """Combined loss for multimodal recommender"""
    
    def __init__(
        self, 
        use_contrastive: bool = True,
        contrastive_weight: float = 0.1,
        bce_weight: float = 1.0
    ):
        super(MultimodalRecommenderLoss, self).__init__()
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.bce_weight = bce_weight
        
        self.bce_loss = nn.BCELoss()
        self.contrastive_loss = ContrastiveLoss()
    
    def forward(
        self, 
        predictions: torch.Tensor,
        labels: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            vision_features: Vision features for contrastive loss
            text_features: Text features for contrastive loss
            temperature: Temperature for contrastive loss
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Binary cross entropy loss
        bce_loss = self.bce_loss(predictions, labels)
        
        # Contrastive loss (if applicable)
        contrastive_loss = torch.tensor(0.0).to(predictions.device)
        if self.use_contrastive and vision_features is not None and text_features is not None:
            contrastive_loss = self.contrastive_loss(
                vision_features, 
                text_features, 
                temperature
            )
        
        # Total loss
        total_loss = (
            self.bce_weight * bce_loss + 
            self.contrastive_weight * contrastive_loss
        )
        
        return {
            'total': total_loss,
            'bce': bce_loss,
            'contrastive': contrastive_loss
        }