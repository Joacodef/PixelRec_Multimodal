# src/training/trainer.py
"""
Training logic for multimodal recommender with configurable optimizer and scheduler
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import wandb

from ..models.losses import MultimodalRecommenderLoss


class Trainer:
    """Trainer class for multimodal recommender"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: str = 'models/checkpoints',
        use_contrastive: bool = True
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            use_contrastive: Whether to use contrastive learning
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loss function
        self.criterion = MultimodalRecommenderLoss(
            use_contrastive=use_contrastive
        )

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.optimizer = None
        self.scheduler = None

    def _create_optimizer(
        self,
        lr: float,
        weight_decay: float,
        optimizer_type: str = 'adamw',
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8
    ) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps
            )
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps
            )
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9  # Default momentum for SGD
            )
        else:
            print(f"Unknown optimizer type: {optimizer_type}. Using AdamW.")
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = 'reduce_on_plateau',
        patience: int = 2,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        total_epochs: int = 10
    ):
        """Create learning rate scheduler based on configuration"""
        if scheduler_type.lower() == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=patience,
                factor=factor,
                min_lr=min_lr,
                verbose=True
            )
        elif scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=min_lr
            )
        elif scheduler_type.lower() == 'step':
            # Step every 'patience' epochs
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=patience,
                gamma=factor
            )
        else:
            print(f"Unknown scheduler type: {scheduler_type}. Using ReduceLROnPlateau.")
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=patience,
                factor=factor,
                min_lr=min_lr
            )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 3,
        gradient_clip: float = 1.0,
        # Optimizer parameters
        optimizer_type: str = 'adamw',
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        # Scheduler parameters
        use_lr_scheduler: bool = True,
        lr_scheduler_type: str = 'reduce_on_plateau',
        lr_scheduler_patience: int = 2,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_min_lr: float = 1e-6
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model with configurable optimizer and scheduler.

        Returns:
            Training and validation losses
        """
        # Initialize optimizer
        optimizer = self._create_optimizer(
            lr=lr,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps
        )
        self.optimizer = optimizer
        
        # Initialize scheduler if requested
        if use_lr_scheduler:
            self.scheduler = self._create_scheduler(
                optimizer=optimizer,
                scheduler_type=lr_scheduler_type,
                patience=lr_scheduler_patience,
                factor=lr_scheduler_factor,
                min_lr=lr_scheduler_min_lr,
                total_epochs=epochs
            )

        train_losses = []
        val_losses = []

        for epoch_num in range(epochs):
            self.epoch = epoch_num

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

            # Log metrics to wandb
            self._log_metrics(train_metrics, val_metrics, self.epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()

            # Early stopping check
            if self._check_early_stopping(val_metrics['total_loss'], patience):
                print(f"Early stopping at epoch {self.epoch+1}")
                break

            # Print epoch summary
            self._print_epoch_summary(self.epoch, epochs, train_metrics, val_metrics)

        return train_losses, val_losses

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        gradient_clip: float
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss_val = 0
        bce_loss_val = 0
        contrastive_loss_val = 0
        correct_preds = 0
        total_samples = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch+1} - Training"
        )

        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch to device
            batch = self._batch_to_device(batch)

            # Prepare model inputs
            model_call_args = {
                'user_idx': batch['user_idx'],
                'item_idx': batch['item_idx'],
                'image': batch['image'],
                'text_input_ids': batch['text_input_ids'],
                'text_attention_mask': batch['text_attention_mask'],
                'numerical_features': batch['numerical_features'],
            }
            
            # Add CLIP specific inputs if they exist in the batch
            if 'clip_text_input_ids' in batch:
                model_call_args['clip_text_input_ids'] = batch['clip_text_input_ids']
            if 'clip_text_attention_mask' in batch:
                model_call_args['clip_text_attention_mask'] = batch['clip_text_attention_mask']

            # Forward pass with embeddings
            if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                model_call_args['return_embeddings'] = True
                output, vision_features, text_features, _ = self.model(**model_call_args)
            else:
                output = self.model(**model_call_args)
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
            total_loss_val += loss_dict['total'].item()
            bce_loss_val += loss_dict['bce'].item()
            contrastive_loss_val += loss_dict.get('contrastive', torch.tensor(0.0)).item()

            predictions = (output > 0.5).float()
            correct_preds += (predictions == batch['label']).sum().item()
            total_samples += batch['label'].size(0)

            # Update progress bar
            current_accuracy = correct_preds / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'acc': f"{current_accuracy:.4f}"
            })

        return {
            'total_loss': total_loss_val / len(train_loader) if len(train_loader) > 0 else 0,
            'bce_loss': bce_loss_val / len(train_loader) if len(train_loader) > 0 else 0,
            'contrastive_loss': contrastive_loss_val / len(train_loader) if len(train_loader) > 0 else 0,
            'accuracy': correct_preds / total_samples if total_samples > 0 else 0
        }

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()

        total_loss_val = 0
        bce_loss_val = 0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {self.epoch+1} - Validation"):
                # Move batch to device
                batch = self._batch_to_device(batch)

                # Prepare model inputs for validation
                model_call_args_val = {
                    'user_idx': batch['user_idx'],
                    'item_idx': batch['item_idx'],
                    'image': batch['image'],
                    'text_input_ids': batch['text_input_ids'],
                    'text_attention_mask': batch['text_attention_mask'],
                    'numerical_features': batch['numerical_features'],
                }

                # Forward pass
                output = self.model(**model_call_args_val).squeeze()

                # Calculate loss (BCE for validation)
                loss = nn.BCELoss()(output, batch['label'])

                total_loss_val += loss.item()
                bce_loss_val += loss.item()

                predictions = (output > 0.5).float()
                correct_preds += (predictions == batch['label']).sum().item()
                total_samples += batch['label'].size(0)

        return {
            'total_loss': total_loss_val / len(val_loader) if len(val_loader) > 0 else 0,
            'bce_loss': bce_loss_val / len(val_loader) if len(val_loader) > 0 else 0,
            'accuracy': correct_preds / total_samples if total_samples > 0 else 0
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
        current_epoch: int
    ):
        """Log metrics to wandb only"""
        try:
            # Check if a wandb run has been initialized
            if wandb.run is not None:
                wandb_log_data = {}
                
                # Add train metrics
                for key, value in train_metrics.items():
                    wandb_log_data[f'train/{key}'] = value
                
                # Add validation metrics
                for key, value in val_metrics.items():
                    wandb_log_data[f'val/{key}'] = value
                
                # Add learning rate
                lr = self.get_learning_rate()
                wandb_log_data['train/learning_rate'] = lr
                wandb_log_data['epoch'] = current_epoch
                
                # Log to wandb
                wandb.log(wandb_log_data, step=current_epoch)
                
        except Exception as e:
            # Print a warning if logging to wandb fails but continue training
            print(f"Warning: Failed to log to wandb: {e}")

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
        current_epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Print epoch summary"""
        train_contrastive_loss = train_metrics.get('contrastive_loss', 0.0)

        print(f"\nEpoch {current_epoch+1}/{total_epochs}")
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(BCE: {train_metrics['bce_loss']:.4f}, "
              f"Contrastive: {train_contrastive_loss:.4f})")
        print(f"Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Print current learning rate
        current_lr = self.get_learning_rate()
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # Include optimizer state if it exists
        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Include scheduler state if it exists
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            print(f"Saved best model checkpoint to {path}")
            # Save best model to wandb
            try:
                if wandb.run is not None:
                    wandb.save(str(path))
            except Exception as e:
                print(f"Warning: Failed to save checkpoint to wandb: {e}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"Warning: Checkpoint file not found at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Load optimizer state if it exists
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded optimizer state from checkpoint.")
        elif 'optimizer_state_dict' in checkpoint and not self.optimizer:
            print(f"Warning: Optimizer state found in checkpoint, but trainer's optimizer is not initialized.")
        
        # Load scheduler state if it exists
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded scheduler state from checkpoint.")

        print(f"Loaded checkpoint from {path} (epoch {self.epoch+1})")

    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        if self.optimizer is None:
            return 0.0

        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0