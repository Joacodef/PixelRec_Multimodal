# src/models/losses.py
"""
Defines custom loss functions used for training the multimodal recommender system.

This module contains implementations of specialized loss functions tailored for
recommendation tasks, including contrastive loss for aligning different
modalities and a combined loss function that integrates multiple training
objectives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """
    Implements a contrastive loss, similar to the one used in CLIP.

    This loss function encourages the model to learn a shared embedding space
    where the representations of corresponding image-text pairs are pulled
    closer together, while representations of non-corresponding pairs are
    pushed apart.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initializes the ContrastiveLoss module.

        Args:
            temperature (float): A temperature parameter that scales the logits
                                 before the softmax operation, controlling the
                                 sharpness of the distribution.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the contrastive loss between image and text feature batches.

        Args:
            image_features (torch.Tensor): A tensor of image feature vectors,
                                           typically with shape (batch_size, embedding_dim).
            text_features (torch.Tensor): A tensor of text feature vectors,
                                          typically with shape (batch_size, embedding_dim).
            temperature (Optional[torch.Tensor]): An optional learnable temperature
                                                  parameter to override the default.

        Returns:
            torch.Tensor: A scalar tensor representing the computed contrastive loss.
        """
        # Use the provided temperature parameter if available, otherwise use the default.
        if temperature is None:
            temperature = self.temperature

        # Normalize the feature vectors to unit length (L2 norm).
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute the cosine similarity matrix between all image and text features.
        # The result is scaled by the temperature parameter.
        logits = torch.matmul(image_features, text_features.t()) / temperature

        # The ground truth labels are the diagonal elements of the similarity matrix,
        # as corresponding image-text pairs are at the same index in the batch.
        labels = torch.arange(logits.size(0)).to(logits.device)

        # Compute the cross-entropy loss for image-to-text and text-to-image directions.
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        # The final loss is the average of the two directional losses.
        return (loss_i2t + loss_t2i) / 2


class BPRLoss(nn.Module):
    """
    Implements the Bayesian Personalized Ranking (BPR) loss function.

    BPR loss is a pairwise ranking loss that encourages the model to score a
    positive (interacted-with) item higher than a randomly sampled negative
    (not interacted-with) item for a given user.
    """

    def __init__(self):
        """Initializes the BPRLoss module."""
        super(BPRLoss, self).__init__()

    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the BPR loss.

        Args:
            positive_scores (torch.Tensor): The model's scores for positive items.
            negative_scores (torch.Tensor): The model's scores for negative items.

        Returns:
            torch.Tensor: A scalar tensor representing the computed BPR loss.
        """
        # The loss is calculated as the negative log-likelihood of the sigmoid
        # of the difference between positive and negative item scores.
        return -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))


class MultimodalRecommenderLoss(nn.Module):
    """
    A combined loss function for the multimodal recommender system.

    This class integrates a primary prediction loss (Binary Cross-Entropy)
    with an optional auxiliary contrastive loss. The two components are
    weighted to form the final loss value used for backpropagation.
    """

    def __init__(
        self,
        use_contrastive: bool = True,
        contrastive_weight: float = 0.1,
        bce_weight: float = 1.0
    ):
        """
        Initializes the combined loss module.

        Args:
            use_contrastive (bool): If True, the contrastive loss component is enabled.
            contrastive_weight (float): The weight factor for the contrastive loss.
            bce_weight (float): The weight factor for the BCE loss.
        """
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
        Computes the combined loss.

        Args:
            predictions (torch.Tensor): The model's output predictions, expected
                                        to be probabilities (after a sigmoid).
            labels (torch.Tensor): The ground truth binary labels (0 or 1).
            vision_features (Optional[torch.Tensor]): Vision feature vectors, required
                                                      if contrastive loss is used.
            text_features (Optional[torch.Tensor]): Text feature vectors, required
                                                    if contrastive loss is used.
            temperature (Optional[torch.Tensor]): A learnable temperature parameter
                                                  for the contrastive loss.

        Returns:
            dict: A dictionary containing the total loss and its individual
                  components ('total', 'bce', 'contrastive').
        """
        # Handles cases where the model outputs NaN or Inf to prevent crashing.
        if not torch.isfinite(predictions).all():
            nan_loss = torch.tensor(float('nan'), device=predictions.device)
            return {
                'total': nan_loss,
                'bce': nan_loss,
                'contrastive': torch.tensor(0.0, device=predictions.device)
            }

        # Clamp predictions to a small epsilon range to prevent log(0) and ensure numerical stability.
        epsilon = 1e-7
        clamped_predictions = torch.clamp(predictions, min=epsilon, max=1.0 - epsilon)

        # Compute the Binary Cross-Entropy loss for the primary recommendation task.
        bce_loss = self.bce_loss(clamped_predictions, labels)

        # Compute the contrastive loss if enabled and features are provided.
        contrastive_loss_value = torch.tensor(0.0, device=predictions.device)
        if self.use_contrastive and vision_features is not None and text_features is not None:
            contrastive_loss_value = self.contrastive_loss(
                vision_features,
                text_features,
                temperature
            )

        # Combine the losses using their respective weights.
        total_loss = (
            self.bce_weight * bce_loss +
            self.contrastive_weight * contrastive_loss_value
        )

        return {
            'total': total_loss,
            'bce': bce_loss,
            'contrastive': contrastive_loss_value
        }