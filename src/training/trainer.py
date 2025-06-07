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
        use_contrastive: bool = True,
        model_config: Optional[object] = None  # NEW: Add model_config parameter
    ):
        self.model = model
        self.device = device
        self.base_checkpoint_dir = Path(checkpoint_dir)
        self.model_config = model_config
        
        # Create model-specific checkpoint directory for .pth files
        if model_config and hasattr(model_config, 'vision_model') and hasattr(model_config, 'language_model'):
            model_combo = f"{model_config.vision_model}_{model_config.language_model}"
            self.model_checkpoint_dir = self.base_checkpoint_dir / model_combo
        else:
            # Fallback to base directory if no model config provided
            self.model_checkpoint_dir = self.base_checkpoint_dir
            if model_config is None:
                print("Warning: No model config provided to Trainer. Using base checkpoint directory.")
        
        # Shared encoders directory (remains in base checkpoint_dir)
        self.encoders_dir = self.base_checkpoint_dir / 'encoders'
        
        # Create directories
        self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.encoders_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Trainer initialized:")
        print(f"  → Model checkpoints (.pth): {self.model_checkpoint_dir}")
        print(f"  → Shared encoders: {self.encoders_dir}")
        
        self.criterion = MultimodalRecommenderLoss(use_contrastive=use_contrastive)
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
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2), eps=adam_eps)
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2), eps=adam_eps)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            print(f"Unknown optimizer type: {optimizer_type}. Using AdamW.")
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = 'reduce_on_plateau',
        patience: int = 2,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        total_epochs: int = 10
    ):
        if scheduler_type.lower() == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr, verbose=True)
        elif scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
        elif scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=factor)
        else:
            print(f"Unknown scheduler type: {scheduler_type}. Using ReduceLROnPlateau.")
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader], 
        epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 3, 
        gradient_clip: float = 1.0,
        optimizer_type: str = 'adamw',
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        use_lr_scheduler: bool = True,
        lr_scheduler_type: str = 'reduce_on_plateau',
        lr_scheduler_patience: int = 2, 
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_min_lr: float = 1e-6
    ) -> Tuple[List[float], List[float]]:
        
        optimizer = self._create_optimizer(
            lr=lr, weight_decay=weight_decay, optimizer_type=optimizer_type,
            adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps
        )
        self.optimizer = optimizer
        
        if use_lr_scheduler:
            self.scheduler = self._create_scheduler(
                optimizer=optimizer, scheduler_type=lr_scheduler_type,
                patience=lr_scheduler_patience, factor=lr_scheduler_factor,
                min_lr=lr_scheduler_min_lr, total_epochs=epochs
            )

        train_losses = []
        val_losses = [] 

        for epoch_num in range(self.epoch, epochs): 
            self.epoch = epoch_num

            train_metrics = self._train_epoch(train_loader, optimizer, gradient_clip)
            train_losses.append(train_metrics['total_loss'])

            validation_performed_this_epoch = False
            if val_loader is not None and len(val_loader) > 0:
                val_metrics = self._validate_epoch(val_loader)
                if 'total_loss' in val_metrics: # Ensure key exists
                    val_losses.append(val_metrics['total_loss'])
                    validation_performed_this_epoch = True
                else: # Should not happen if _validate_epoch returns correctly
                    val_losses.append(np.nan)
                    validation_performed_this_epoch = False # Treat as not performed if loss is missing
            else:
                print(f"Epoch {self.epoch+1}: Validation skipped (no validation data).")
                val_metrics = {'total_loss': np.nan, 'bce_loss': np.nan, 'accuracy': 0.0, 'contrastive_loss': np.nan} 
                val_losses.append(np.nan)

            self._log_metrics(train_metrics, val_metrics, self.epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if validation_performed_this_epoch and not np.isnan(val_metrics['total_loss']):
                        self.scheduler.step(val_metrics['total_loss'])
                else: 
                    self.scheduler.step()
            
            if validation_performed_this_epoch and not np.isnan(val_metrics['total_loss']):
                if self._check_early_stopping(val_metrics['total_loss'], patience):
                    print(f"Early stopping at epoch {self.epoch+1}")
                    break 
            
            self._print_epoch_summary(self.epoch, epochs, train_metrics, val_metrics)

        return train_losses, val_losses

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        gradient_clip: float
    ) -> Dict[str, float]:
        self.model.train() # Sets the model to training mode.
        
        total_loss_val, bce_loss_val, contrastive_loss_val = 0, 0, 0
        correct_preds, total_samples = 0, 0
        # Wraps train_loader with tqdm for a progress bar.
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch+1} - Training", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad() # Clears old gradients.
            batch = self._batch_to_device(batch) # Moves batch data to the configured device.
            
            # Prepares arguments for the model's forward pass from the batch data.
            model_call_args = {
                'user_idx': batch['user_idx'], 'item_idx': batch['item_idx'], 'image': batch['image'],
                'text_input_ids': batch['text_input_ids'], 'text_attention_mask': batch['text_attention_mask'],
                'numerical_features': batch['numerical_features'],
            }
           
            # model_call_args['debug_this_batch'] = True # or model_call_args_val
            
            # Adds CLIP-specific text inputs if they are present in the batch.
            if 'clip_text_input_ids' in batch: model_call_args['clip_text_input_ids'] = batch['clip_text_input_ids']
            if 'clip_text_attention_mask' in batch: model_call_args['clip_text_attention_mask'] = batch['clip_text_attention_mask']

            # --- Existing model call logic ---
            vision_features_for_loss = None
            text_features_for_loss = None
            output_before_squeeze = None # Initialize

            if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                model_call_args['return_embeddings'] = True
                output_tuple = self.model(**model_call_args)
                output_before_squeeze, vision_features_for_loss, text_features_for_loss, _ = output_tuple
            else:
                model_call_args['return_embeddings'] = False
                output_before_squeeze = self.model(**model_call_args)
            # --- End of existing model call logic ---           
            
            output = output_before_squeeze.squeeze() # The squeeze operation           
            
            # Calculates loss using the criterion.
            loss_dict = self.criterion(
                output, batch['label'], 
                vision_features_for_loss,
                text_features_for_loss,
                self.model.temperature if hasattr(self.model, 'temperature') else None
            )
            
            # Backpropagation if total loss is finite, otherwise skip to prevent further errors.
            if torch.isfinite(loss_dict['total']):
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip) # Clips gradients.
                optimizer.step() # Updates model parameters.
            else:
                print(f"WARNING: Skipping backward pass for batch_idx {batch_idx} due to non-finite loss (NaN or Inf).")


            # Accumulates loss values (only if finite).
            if torch.isfinite(loss_dict['total']): total_loss_val += loss_dict['total'].item()
            if torch.isfinite(loss_dict['bce']): bce_loss_val += loss_dict['bce'].item()
            # Uses .get() for contrastive loss.
            if torch.isfinite(loss_dict.get('contrastive', torch.tensor(0.0))):
                contrastive_loss_val += loss_dict.get('contrastive', torch.tensor(0.0)).item()
            
            # Ensures output and label tensors are at least 1D for accuracy calculation.
            if output.ndim == 0: output = output.unsqueeze(0)
            if batch['label'].ndim == 0: batch['label'] = batch['label'].unsqueeze(0)
            
            # Calculates prediction accuracy for finite outputs.
            if output.size() == batch['label'].size() and torch.isfinite(output).all():
                predictions = (output > 0.5).float()
                correct_preds += (predictions == batch['label']).sum().item()
            total_samples += batch['label'].size(0)

            # Updates the progress bar with current loss and accuracy.
            current_loss_display = loss_dict['total'].item() if torch.isfinite(loss_dict['total']) else float('nan')
            current_accuracy = correct_preds / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix({'loss': f"{current_loss_display:.4f}", 'acc': f"{current_accuracy:.4f}"})

        # Calculates average metrics over the training epoch.
        len_train_loader = len(train_loader) if train_loader else 0
        avg_total_loss = total_loss_val / len_train_loader if len_train_loader > 0 else float('nan') # Use NaN if no valid batches
        avg_bce_loss = bce_loss_val / len_train_loader if len_train_loader > 0 else float('nan')
        avg_contrastive_loss = contrastive_loss_val / len_train_loader if len_train_loader > 0 else float('nan')
        avg_accuracy = correct_preds / total_samples if total_samples > 0 else 0.0
        
        return {
            'total_loss': avg_total_loss,
            'bce_loss': avg_bce_loss,
            'contrastive_loss': avg_contrastive_loss,
            'accuracy': avg_accuracy
        }

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval() # Sets the model to evaluation mode.
        total_loss_val, bce_loss_val, contrastive_loss_val_val = 0, 0, 0
        correct_preds, total_samples = 0, 0

        # Wrap val_loader with tqdm for a progress bar.
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {self.epoch+1} - Validation", leave=False)

        with torch.no_grad(): # Disables gradient calculations during validation.
            for batch_idx, batch in enumerate(progress_bar_val):
                batch = self._batch_to_device(batch) # Moves batch data to the configured device.
                
                # Prepares arguments for the model's forward pass from the batch data.
                model_call_args_val = {
                    'user_idx': batch['user_idx'], 'item_idx': batch['item_idx'], 'image': batch['image'],
                    'text_input_ids': batch['text_input_ids'], 'text_attention_mask': batch['text_attention_mask'],
                    'numerical_features': batch['numerical_features'],
                }
                # When preparing model_call_args_val for validation
               
                # model_call_args_val['debug_this_batch'] = True

                # Adds CLIP-specific text inputs if they are present in the batch.
                if 'clip_text_input_ids' in batch: model_call_args_val['clip_text_input_ids'] = batch['clip_text_input_ids']
                if 'clip_text_attention_mask' in batch: model_call_args_val['clip_text_attention_mask'] = batch['clip_text_attention_mask']
                
                vision_features_for_loss_val = None
                text_features_for_loss_val = None

                # Calls the model's forward pass.
                if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                    model_call_args_val['return_embeddings'] = True
                    output_tuple_val = self.model(**model_call_args_val)
                    output_val, vision_features_for_loss_val, text_features_for_loss_val, _ = output_tuple_val
                else:
                    model_call_args_val['return_embeddings'] = False
                    output_val = self.model(**model_call_args_val)

                output_val = output_val.squeeze() # Removes dimensions of size 1 from the output.

                # +++ START DIAGNOSTIC BLOCK +++
                # Checks for NaN or Inf values in the model's output before passing to loss.
                has_nan = torch.isnan(output_val).any()
                has_inf = torch.isinf(output_val).any()

                if has_nan or has_inf:
                    print(f"  Problematic batch (idx {batch_idx}) details:")
                    if 'user_idx' in batch:
                        print(f"    User indices (first 5): {batch['user_idx'][:5].tolist()}")
                    if 'item_idx' in batch:
                        print(f"    Item indices (first 5): {batch['item_idx'][:5].tolist()}")
                    print(f"\nWARNING: NaN or Inf detected in VALIDATION output_val at batch_idx {batch_idx}!")
                    print(f"  output_val contains NaN: {has_nan.item()}")
                    print(f"  output_val contains Inf: {has_inf.item()}")
                    # It's helpful to see the range of finite values if some parts are problematic
                    finite_values = output_val[torch.isfinite(output_val)]
                    if finite_values.numel() > 0:
                        print(f"  Finite values in output_val - min: {finite_values.min().item()}, max: {finite_values.max().item()}, mean: {finite_values.mean().item()}")
                    else:
                        print("  output_val contains no finite values (all NaN/Inf).")
                    # Consider printing a few problematic values from output_val if needed for more detail
                    # For example: print(output_val[torch.isnan(output_val) | torch.isinf(output_val)])
                # +++ END DIAGNOSTIC BLOCK +++

                # Calculates loss using the criterion (e.g., MultimodalRecommenderLoss).
                loss_dict_val = self.criterion(
                    output_val, batch['label'], 
                    vision_features_for_loss_val,
                    text_features_for_loss_val,
                    self.model.temperature if hasattr(self.model, 'temperature') else None
                )
                
                # Accumulates loss values.
                total_loss_val += loss_dict_val['total'].item()
                bce_loss_val += loss_dict_val['bce'].item()
                # Uses .get() for contrastive loss as it might be absent if not use_contrastive.
                contrastive_loss_val_val += loss_dict_val.get('contrastive', torch.tensor(0.0)).item()

                # Ensures output and label tensors are at least 1D for consistent processing.
                if output_val.ndim == 0: output_val = output_val.unsqueeze(0)
                if batch['label'].ndim == 0: batch['label'] = batch['label'].unsqueeze(0)

                # Calculates prediction accuracy.
                if output_val.size() == batch['label'].size():
                    predictions = (output_val > 0.5).float() # Converts probabilities to binary predictions.
                    correct_preds += (predictions == batch['label']).sum().item()
                total_samples += batch['label'].size(0)
        
        # Calculates average metrics over the validation epoch.
        len_val_loader = len(val_loader) if val_loader else 0 # Handles case of empty val_loader.
        avg_total_loss = total_loss_val / len_val_loader if len_val_loader > 0 else 0
        avg_bce_loss = bce_loss_val / len_val_loader if len_val_loader > 0 else 0
        avg_contrastive_loss = contrastive_loss_val_val / len_val_loader if len_val_loader > 0 else 0
        avg_accuracy = correct_preds / total_samples if total_samples > 0 else 0
        
        return {
            'total_loss': avg_total_loss,
            'bce_loss': avg_bce_loss,
            'contrastive_loss': avg_contrastive_loss,
            'accuracy': avg_accuracy
        }


    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], current_epoch: int):
        try:
            if wandb.run is not None:
                wandb_log_data = {}
                for key, value in train_metrics.items(): wandb_log_data[f'train/{key}'] = value
                for key, value in val_metrics.items():
                    if not np.isnan(value): wandb_log_data[f'val/{key}'] = value 
                lr = self.get_learning_rate()
                wandb_log_data['train/learning_rate'] = lr
                wandb_log_data['epoch'] = current_epoch
                wandb.log(wandb_log_data, step=current_epoch)
        except Exception as e: print(f"Warning: Failed to log to wandb: {e}")

    def _check_early_stopping(self, val_loss: float, patience: int) -> bool:
        if np.isnan(val_loss): return False 

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint('best_model.pth', is_best=True)
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience

    def _print_epoch_summary(self, current_epoch: int, total_epochs: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        train_contrastive_loss = train_metrics.get('contrastive_loss', 0.0)
        val_contrastive_loss = val_metrics.get('contrastive_loss', np.nan)

        print(f"\nEpoch {current_epoch+1}/{total_epochs}")
        print(f"Train Loss: {train_metrics['total_loss']:.4f} (BCE: {train_metrics['bce_loss']:.4f}, Contrastive: {train_contrastive_loss:.4f})")
        print(f"Train Acc: {train_metrics['accuracy']:.4f}")
        
        val_loss_str = f"{val_metrics['total_loss']:.4f}" if not np.isnan(val_metrics['total_loss']) else "N/A"
        val_bce_str = f"{val_metrics.get('bce_loss', np.nan):.4f}" if not np.isnan(val_metrics.get('bce_loss', np.nan)) else "N/A" # Added .get for safety
        val_contrastive_str = f"{val_contrastive_loss:.4f}" if not np.isnan(val_contrastive_loss) else "N/A"
        val_acc_str = f"{val_metrics['accuracy']:.4f}" if not np.isnan(val_metrics['accuracy']) else "N/A"

        print(f"Val Loss: {val_loss_str} (BCE: {val_bce_str}, Contrastive: {val_contrastive_str})")
        print(f"Val Acc: {val_acc_str}")
        
        current_lr = self.get_learning_rate()
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)

    def save_checkpoint(self, filename: str, is_best: bool = False):
        checkpoint = {'epoch': self.epoch, 'model_state_dict': self.model.state_dict(), 'best_val_loss': self.best_val_loss}
        if self.optimizer: checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        if is_best:
            print(f"Saved best model checkpoint to {path}")
            try:
                if wandb.run is not None: wandb.save(str(path))
            except Exception as e: print(f"Warning: Failed to save checkpoint to wandb: {e}")

    def load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        if not path.exists(): print(f"Warning: Checkpoint file not found at {path}"); return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint.get('epoch', 0) # Get epoch for resuming
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            try: # Add try-except for optimizer loading
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded optimizer state from checkpoint.")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Optimizer will be reinitialized.")
        elif 'optimizer_state_dict' in checkpoint and not self.optimizer:
            print(f"Warning: Optimizer state found in checkpoint, but trainer's optimizer is not initialized.")
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            try: # Add try-except for scheduler loading
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Loaded scheduler state from checkpoint.")
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}. Scheduler may not resume correctly.")
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})") # Resuming from self.epoch, so next epoch will be self.epoch + 1

    def get_learning_rate(self) -> float:
        if self.optimizer is None: return 0.0
        for param_group in self.optimizer.param_groups: return param_group['lr']
        return 0.0

    def get_model_checkpoint_dir(self) -> Path:
        """Get the model-specific checkpoint directory"""
        return self.model_checkpoint_dir

    def get_encoders_dir(self) -> Path:
        """Get the shared encoders directory"""
        return self.encoders_dir