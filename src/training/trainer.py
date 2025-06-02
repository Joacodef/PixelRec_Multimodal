# src/training/trainer.py
"""
Training logic for multimodal recommender
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Remains for existing TensorBoard logging
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import wandb # Import wandb

from ..models.losses import MultimodalRecommenderLoss


class Trainer:
    """Trainer class for multimodal recommender"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: str = 'models/checkpoints',
        log_dir: str = 'logs/tensorboard', # For TensorBoard
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
        self.optimizer = None

    # ... (train, _train_epoch, _validate_epoch, _batch_to_device methods remain unchanged) ...
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
        self.optimizer = optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=2,
            factor=0.5
        )

        train_losses = []
        val_losses = []

        for epoch_num in range(epochs): # Renamed epoch to epoch_num to avoid conflict with self.epoch
            self.epoch = epoch_num # self.epoch is the current epoch number

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
            self._log_metrics(train_metrics, val_metrics, self.epoch) # Use self.epoch here

            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])

            # Early stopping check
            if self._check_early_stopping(val_metrics['total_loss'], patience):
                print(f"Early stopping at epoch {self.epoch+1}")
                break

            # Print epoch summary
            self._print_epoch_summary(self.epoch, epochs, train_metrics, val_metrics)

        self.writer.close() # Close TensorBoard writer
        return train_losses, val_losses

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        gradient_clip: float
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss_val = 0 # Renamed to avoid conflict with imported 'total_loss' if any
        bce_loss_val = 0 # Renamed
        contrastive_loss_val = 0 # Renamed
        correct_preds = 0 # Renamed
        total_samples = 0 # Renamed

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
            # Assumes model.temperature attribute exists if contrastive learning is part of the model
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
            # Ensure 'contrastive' key exists before accessing
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

        total_loss_val = 0 # Renamed
        bce_loss_val = 0 # Renamed
        correct_preds = 0 # Renamed
        total_samples = 0 # Renamed

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {self.epoch+1} - Validation"):
                # Move batch to device
                batch = self._batch_to_device(batch)

                # Prepare model inputs for validation
                # Note: clip_text_input_ids and clip_text_attention_mask are usually not needed for validation's forward pass
                # unless validation also specifically evaluates the contrastive aspect or embeddings.
                # Assuming standard validation focuses on the primary recommendation task output.
                model_call_args_val = {
                    'user_idx': batch['user_idx'],
                    'item_idx': batch['item_idx'],
                    'image': batch['image'],
                    'text_input_ids': batch['text_input_ids'],
                    'text_attention_mask': batch['text_attention_mask'],
                    'numerical_features': batch['numerical_features'],
                }
                # If your model's forward pass for validation *requires* these (even if None), add them:
                # if 'clip_text_input_ids' in batch: # Or more generally, ensure all expected args are present
                #     model_call_args_val['clip_text_input_ids'] = batch.get('clip_text_input_ids')
                # if 'clip_text_attention_mask' in batch:
                #    model_call_args_val['clip_text_attention_mask'] = batch.get('clip_text_attention_mask')


                # Forward pass
                output = self.model(**model_call_args_val).squeeze()


                # Calculate loss (only BCE for validation if contrastive parts are not returned/used)
                # Assuming MultimodalRecommenderLoss can handle None for vision/text features
                # or direct BCELoss for validation. For simplicity, using direct BCELoss.
                loss = nn.BCELoss()(output, batch['label'])

                total_loss_val += loss.item()
                bce_loss_val += loss.item() # Assuming total_loss for val is just BCE

                predictions = (output > 0.5).float()
                correct_preds += (predictions == batch['label']).sum().item()
                total_samples += batch['label'].size(0)

        return {
            'total_loss': total_loss_val / len(val_loader) if len(val_loader) > 0 else 0,
            'bce_loss': bce_loss_val / len(val_loader) if len(val_loader) > 0 else 0, # This will be same as total_loss here
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
        current_epoch: int # Renamed 'epoch' to avoid conflict with self.epoch if used differently
    ):
        """Log metrics to tensorboard and wandb"""
        # Log to TensorBoard
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, current_epoch) # Use current_epoch

        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, current_epoch) # Use current_epoch

        lr = self.get_learning_rate()
        self.writer.add_scalar('train/learning_rate', lr, current_epoch) # Use current_epoch

        # Log to Weights & Biases
        try:
            # Check if a wandb run has been initialized
            if wandb.run is not None:
                wandb_log_data = {}
                for key, value in train_metrics.items():
                    wandb_log_data[f'train/{key}'] = value
                for key, value in val_metrics.items():
                    wandb_log_data[f'val/{key}'] = value
                wandb_log_data['train/learning_rate'] = lr
                # wandb.log automatically uses the current epoch or step if 'epoch' is not in dict for step
                # or you can pass step=current_epoch explicitly.
                # For simplicity, wandb.log will infer step or use its own logic if 'epoch' isn't a metric.
                # If you want 'epoch' to be the x-axis for wandb charts, include it:
                wandb_log_data['epoch'] = current_epoch
                wandb.log(wandb_log_data, step=current_epoch) # Explicitly set step
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
        current_epoch: int, # Renamed
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Print epoch summary"""
        # Safely get contrastive loss, defaulting to 0 if not present
        train_contrastive_loss = train_metrics.get('contrastive_loss', 0.0)

        print(f"\nEpoch {current_epoch+1}/{total_epochs}") # Use current_epoch
        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(BCE: {train_metrics['bce_loss']:.4f}, "
              f"Contrastive: {train_contrastive_loss:.4f})")
        print(f"Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        print("-" * 50)

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch, # self.epoch is fine here as it's the state of the trainer
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        # Include optimizer state if it exists
        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()


        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            print(f"Saved best model checkpoint to {path}")
            # Potentially log to wandb that the best model was saved
            try:
                if wandb.run is not None:
                    wandb.save(str(path)) # Save the best model file to wandb
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
        self.epoch = checkpoint.get('epoch', 0) # Use .get for backward compatibility
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Use .get

        # Load optimizer state if it exists in checkpoint and self.optimizer is initialized
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded optimizer state from checkpoint.")
        elif 'optimizer_state_dict' in checkpoint and not self.optimizer:
            print(f"Warning: Optimizer state found in checkpoint, but trainer's optimizer is not initialized. Optimizer state not loaded.")


        print(f"Loaded checkpoint from {path} (epoch {self.epoch+1})")

    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        if self.optimizer is None:
            return 0.0

        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0 # Should not be reached if optimizer is not None and has param_groups