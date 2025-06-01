"""
Training logic for multimodal recommender
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

from ..models.losses import MultimodalRecommenderLoss


class Trainer:
    """Trainer class for multimodal recommender"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: str = 'models/checkpoints',
        log_dir: str = 'logs/tensorboard',
        use_contrastive: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            use_contrastive: Whether to use contrastive learning
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Initialize loss function
        self.criterion = MultimodalRecommenderLoss(
            use_contrastive=use_contrastive
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 3,
        gradient_clip: float = 1.0
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            gradient_clip: Gradient clipping value
            
        Returns:
            Training and validation losses
        """
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=2, 
            factor=0.5
        )
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(
                train_loader, 
                optimizer, 
                gradient_clip
            )
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader)
            
            # Record losses
            train_losses.append(train_metrics['total_loss'])
            val_losses.append(val_metrics['total_loss'])
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['total_loss'], patience):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Print epoch summary
            self._print_epoch_summary(epoch, epochs, train_metrics, val_metrics)
        
        self.writer.close()
        return train_losses, val_losses
    
    def _train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer,
        gradient_clip: float
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        bce_loss = 0
        contrastive_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {self.epoch+1} - Training"
        )
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass with embeddings
            if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                output, vision_features, text_features, _ = self.model(
                    batch['user_idx'],
                    batch['item_idx'],
                    batch['image'],
                    batch['text_input_ids'],
                    batch['text_attention_mask'],
                    batch['numerical_features'],
                    return_embeddings=True
                )
            else:
                output = self.model(
                    batch['user_idx'],
                    batch['item_idx'],
                    batch['image'],
                    batch['text_input_ids'],
                    batch['text_attention_mask'],
                    batch['numerical_features']
                )
                vision_features = None
                text_features = None
            
            output = output.squeeze()
            
            # Calculate loss
            loss_dict = self.criterion(
                output,
                batch['label'],
                vision_features,
                text_features,
                self.model.temperature if hasattr(self.model, 'temperature') else None
            )
            
            # Backward pass
            loss_dict['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                gradient_clip
            )
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total'].item()
            bce_loss += loss_dict['bce'].item()
            contrastive_loss += loss_dict['contrastive'].item()
            
            predictions = (output > 0.5).float()
            correct += (predictions == batch['label']).sum().item()
            total += batch['label'].size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        return {
            'total_loss': total_loss / len(train_loader),
            'bce_loss': bce_loss / len(train_loader),
            'contrastive_loss': contrastive_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        bce_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {self.epoch+1} - Validation"):
                # Move batch to device
                batch = self._batch_to_device(batch)
                
                # Forward pass
                output = self.model(
                    batch['user_idx'],
                    batch['item_idx'],
                    batch['image'],
                    batch['text_input_ids'],
                    batch['text_attention_mask'],
                    batch['numerical_features']
                ).squeeze()
                
                # Calculate loss (only BCE for validation)
                loss = nn.BCELoss()(output, batch['label'])
                
                total_loss += loss.item()
                bce_loss += loss.item()
                
                predictions = (output > 0.5).float()
                correct += (predictions == batch['label']).sum().item()
                total += batch['label'].size(0)
        
        return {
            'total_loss': total_loss / len(val_loader),
            'bce_loss': bce_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    
    def _log_metrics(
        self, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float], 
        epoch: int
    ):
        """Log metrics to tensorboard"""
        # Training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        # Validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Learning rate
        lr = self.get_learning_rate()
        self.writer.add_scalar('train/learning_rate', lr, epoch)
    
    def _check_early_stopping(self, val_loss: float, patience: int) -> bool:
        """Check if early stopping criteria is met"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # Save best model
            self.save_checkpoint('best_model.pth', is_best=True)
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience
    
    def _print_epoch_summary(
        self, 
        epoch: int, 
        total_epochs: int,
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ):
        """Print epoch summary"""
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(BCE: {train_metrics['bce_loss']:.4f}, "
              f"Contrastive: {train_metrics['contrastive_loss']:.4f})")
        print(f"Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        print("-" * 50)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            print(f"Saved best model checkpoint to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0