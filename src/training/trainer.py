# src/training/trainer.py
"""
Contains the training and validation logic for the multimodal recommender model.

This module defines the Trainer class, which encapsulates the entire training
process, including the training loop, validation, optimization, learning rate
scheduling, and checkpoint management.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
import wandb

from ..models.losses import MultimodalRecommenderLoss


class Trainer:
    """
    Manages the training process for a multimodal recommender model.

    This class handles the training and validation loops, optimizer and scheduler
    creation, checkpoint saving and loading, metric logging, and early stopping.
    It is designed to be configurable to support various training setups.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Optional[object] = None, 
        checkpoint_dir: str = 'models/checkpoints',
        use_contrastive: bool = True,
        trial_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the Trainer instance.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            device (torch.device): The device (CPU or GPU) to run the training on.
            checkpoint_dir (str): The base directory to save model checkpoints and encoders.
            use_contrastive (bool): Flag to determine if contrastive loss should be used.
            config (Optional[object]): Configuration object containing model and training parameters.
            trial_info (Optional[Dict[str, Any]]): Information about the current Optuna trial,
                                                including trial number and parameters.
        """
        self.model = model
        self.device = device
        self.config = config
        self.base_checkpoint_dir = Path(checkpoint_dir)
        
        # Creates a model-specific directory for checkpoints to keep experiments organized.
        # Access the model config via self.config.model
        if config and hasattr(config, 'model'):
            model_combo = f"{config.model.vision_model}_{config.model.language_model}"
            self.model_checkpoint_dir = self.base_checkpoint_dir / model_combo
        else:
            self.model_checkpoint_dir = self.base_checkpoint_dir            
            print("Warning: No model config provided to Trainer. Using base checkpoint directory.")
        
        # Defines a separate directory for shared encoder files.
        self.encoders_dir = self.base_checkpoint_dir / 'encoders'
        
        self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.encoders_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Trainer initialized:")
        print(f"  → Model checkpoints (.pth): {self.model_checkpoint_dir}")
        print(f"  → Shared encoders: {self.encoders_dir}")
        
        # Initializes the loss function.
        self.criterion = MultimodalRecommenderLoss(use_contrastive=use_contrastive)
        
        # Initializes training state variables.
        # Initializes training state variables.
        self.epoch = 0
        # Generic early stopping state initialization
        self.patience_counter = 0
        # The config object is not available here, so we will get it later in the train method.
        self.best_early_stopping_score = None 
        self.optimizer = None
        self.scheduler = None
        self.trial_info = trial_info
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'best_metrics': {}
        }


    def _create_optimizer(
        self,
        lr: float,
        weight_decay: float,
        optimizer_type: str = 'adamw',
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8
    ) -> optim.Optimizer:
        """
        Creates an optimizer based on the specified configuration.

        Args:
            lr (float): The learning rate.
            weight_decay (float): The weight decay (L2 penalty) factor.
            optimizer_type (str): The type of optimizer to create ('adamw', 'adam', 'sgd').
            adam_beta1 (float): The beta1 parameter for Adam-based optimizers.
            adam_beta2 (float): The beta2 parameter for Adam-based optimizers.
            adam_eps (float): The epsilon parameter for Adam-based optimizers for numerical stability.

        Returns:
            optim.Optimizer: An initialized PyTorch optimizer.
        """
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
        """
        Creates a learning rate scheduler.

        Args:
            optimizer (optim.Optimizer): The optimizer to which the scheduler will be attached.
            scheduler_type (str): The type of scheduler ('reduce_on_plateau', 'cosine', 'step').
            patience (int): The patience for the scheduler.
            factor (float): The factor by which the learning rate will be reduced.
            min_lr (float): The minimum learning rate.
            total_epochs (int): The total number of epochs, used by some schedulers like 'cosine'.

        Returns:
            A PyTorch learning rate scheduler.
        """
        if scheduler_type.lower() == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr)
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
        """
        Executes the main training loop for a specified number of epochs.

        Args:
            train_loader (DataLoader): The DataLoader for the training set.
            val_loader (Optional[DataLoader]): The DataLoader for the validation set.
            epochs (int): The total number of epochs to train for.
            lr (float): The learning rate.
            weight_decay (float): The L2 regularization factor.
            patience (int): The number of epochs to wait for validation loss improvement before early stopping.
            gradient_clip (float): The maximum norm for gradient clipping.
            optimizer_type (str): The type of optimizer to use.
            adam_beta1 (float): The beta1 parameter for Adam optimizers.
            adam_beta2 (float): The beta2 parameter for Adam optimizers.
            adam_eps (float): The epsilon parameter for Adam optimizers.
            use_lr_scheduler (bool): Whether to use a learning rate scheduler.
            lr_scheduler_type (str): The type of learning rate scheduler.
            lr_scheduler_patience (int): The patience for the learning rate scheduler.
            lr_scheduler_factor (float): The factor for reducing the learning rate.
            lr_scheduler_min_lr (float): The minimum learning rate.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing the list of training
                                             losses and validation losses for each epoch.
        """
        
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

            # Executes one full pass over the training data.
            train_metrics = self._train_epoch(train_loader, optimizer, gradient_clip)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['train_losses'].append(train_metrics['total_loss'])
            
            train_losses.append(train_metrics['total_loss'])

            validation_performed_this_epoch = False
            # Executes one full pass over the validation data, if available.
            if val_loader is not None and len(val_loader) > 0:
                val_metrics = self._validate_epoch(val_loader)

                if 'total_loss' in val_metrics:
                    val_losses.append(val_metrics['total_loss'])
                    validation_performed_this_epoch = True
                else:
                    val_losses.append(np.nan)
                    validation_performed_this_epoch = False

                if validation_performed_this_epoch:
                    self.training_history['val_metrics'].append(val_metrics)
                    self.training_history['val_losses'].append(val_metrics['total_loss'])
                    
                    # Update best metrics
                    for key, value in val_metrics.items():
                        metric_name = f'val_{key}'
                        if metric_name not in self.training_history['best_metrics']:
                            self.training_history['best_metrics'][metric_name] = value
                        else:
                            # For losses, track minimum; for other metrics, track maximum
                            if 'loss' in key:
                                self.training_history['best_metrics'][metric_name] = min(
                                    self.training_history['best_metrics'][metric_name], value
                                )
                            else:
                                self.training_history['best_metrics'][metric_name] = max(
                                    self.training_history['best_metrics'][metric_name], value
                                )
            else:
                print(f"Epoch {self.epoch+1}: Validation skipped (no validation data).")
                val_metrics = {'total_loss': np.nan, 'bce_loss': np.nan, 'accuracy': 0.0, 'f1_score': 0.0, 'contrastive_loss': np.nan} 
                val_losses.append(np.nan)

            # Logs metrics to Weights & Biases if enabled.
            self._log_metrics(train_metrics, val_metrics, self.epoch)

            # Steps the learning rate scheduler.
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if validation_performed_this_epoch and not np.isnan(val_metrics['total_loss']):
                        self.scheduler.step(val_metrics['total_loss'])
                else: 
                    self.scheduler.step()
            
            # Set the initial best score on the first epoch that has validation.
            if self.best_early_stopping_score is None and validation_performed_this_epoch:
                if self.config.training.early_stopping_direction == 'minimize':
                    self.best_early_stopping_score = float('inf')
                else:
                    self.best_early_stopping_score = float('-inf')

            # Checks for early stopping criteria.
            if validation_performed_this_epoch:
                # Select the metric to monitor based on the configuration.
                monitor_metric_name = self.config.training.early_stopping_metric
                
                # The metric from config can be 'val_f1_score', but the dictionary key is 'f1_score'.
                # We remove the 'val_' prefix to get the correct key for lookup.
                lookup_key = monitor_metric_name.replace('val_', '')
                
                # The validation dictionary uses 'total_loss' as the key for the main loss.
                if lookup_key == 'loss':
                    lookup_key = 'total_loss'

                score = val_metrics.get(lookup_key)

                # Fallback to val_loss if the specified metric is still not found.
                if score is None:
                    print(f"Warning: Early stopping metric '{monitor_metric_name}' (lookup key: '{lookup_key}') not found. Defaulting to val_loss.")
                    score = val_metrics.get('total_loss')
                    self.config.training.early_stopping_direction = 'minimize'
                
                if score is not None and not np.isnan(score):
                    if self._check_early_stopping(score, patience):
                        print(f"Early stopping at epoch {self.epoch+1} based on {monitor_metric_name}")
                        self.save_checkpoint('last_model.pth')
                        break


            # Saves a checkpoint at the end of each epoch.
            self.save_checkpoint('last_model.pth')
            
            # Prints a summary of the epoch's performance.
            self._print_epoch_summary(self.epoch, epochs, train_metrics, val_metrics)

        return train_losses, val_losses

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        gradient_clip: float
    ) -> Dict[str, float]:
        """
        Performs a single training epoch.

        This method iterates over the training DataLoader, performs forward and
        backward passes, updates model weights, and computes training metrics.

        Args:
            train_loader (DataLoader): The DataLoader for the training data.
            optimizer (optim.Optimizer): The optimizer for updating model weights.
            gradient_clip (float): The value for gradient norm clipping.

        Returns:
            Dict[str, float]: A dictionary containing the average metrics for the epoch.
        """
        self.model.train()
        
        total_loss_val, bce_loss_val, contrastive_loss_val = 0.0, 0.0, 0.0
        correct_preds, total_samples, valid_batches = 0, 0, 0
        tp, fp, fn = 0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch+1} - Training", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch = self._batch_to_device(batch)
            
            model_call_args = {
                'user_idx': batch['user_idx'],
                'item_idx': batch['item_idx'],
                'tag_idx': batch['tag_idx'],
                'image': batch['image'],
                'text_input_ids': batch['text_input_ids'],
                'text_attention_mask': batch['text_attention_mask'],
                'numerical_features': batch['numerical_features'],
            }
           
            if 'clip_text_input_ids' in batch: model_call_args['clip_text_input_ids'] = batch['clip_text_input_ids']
            if 'clip_text_attention_mask' in batch: model_call_args['clip_text_attention_mask'] = batch['clip_text_attention_mask']

            vision_features_for_loss = None
            text_features_for_loss = None

            # Determines how to call the model based on whether contrastive loss is active.
            if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                model_call_args['return_embeddings'] = True
                output_tuple = self.model(**model_call_args)
                output_before_squeeze, vision_features_for_loss, text_features_for_loss, _ = output_tuple
            else:
                model_call_args['return_embeddings'] = False
                output_before_squeeze = self.model(**model_call_args)
            
            output = output_before_squeeze.squeeze(-1)
            
            # Computes the loss.
            loss_dict = self.criterion(
                output, batch['label'], 
                vision_features_for_loss,
                text_features_for_loss,
                self.model.temperature if hasattr(self.model, 'temperature') else None
            )
            
            # Performs the backward pass and optimizer step if the loss is finite.
            if torch.isfinite(loss_dict['total']):
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()
                
                # Accumulates metrics for the epoch.
                total_loss_val += loss_dict['total'].item()
                bce_loss_val += loss_dict['bce'].item()
                contrastive_loss_val += loss_dict.get('contrastive', torch.tensor(0.0)).item()
                valid_batches += 1

                if output.size() == batch['label'].size():
                    predictions = (output > 0.5).float()
                    correct_preds += (predictions == batch['label']).sum().item()
                    tp += ((predictions == 1) & (batch['label'] == 1)).sum().item()
                    fp += ((predictions == 1) & (batch['label'] == 0)).sum().item()
                    fn += ((predictions == 0) & (batch['label'] == 1)).sum().item()
            else:
                print(f"WARNING: Skipping backward pass for batch_idx {batch_idx} due to non-finite loss (NaN or Inf).")

            total_samples += batch['label'].size(0)

            # Updates the progress bar with the current loss and accuracy.
            current_loss_display = loss_dict['total'].item() if torch.isfinite(loss_dict['total']) else float('nan')
            current_accuracy = correct_preds / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix({'loss': f"{current_loss_display:.4f}", 'acc': f"{current_accuracy:.4f}"})
        
        # Calculates the average metrics for the entire epoch.
        avg_total_loss = total_loss_val / valid_batches if valid_batches > 0 else float('nan')
        avg_bce_loss = bce_loss_val / valid_batches if valid_batches > 0 else float('nan')
        avg_contrastive_loss = contrastive_loss_val / valid_batches if valid_batches > 0 else float('nan')
        avg_accuracy = correct_preds / total_samples if total_samples > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'total_loss': avg_total_loss,
            'bce_loss': avg_bce_loss,
            'contrastive_loss': avg_contrastive_loss,
            'accuracy': avg_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Performs a single validation epoch.

        This method iterates over the validation DataLoader, performs a forward
        pass, and computes validation metrics. No weight updates are performed.

        Args:
            val_loader (DataLoader): The DataLoader for the validation data.

        Returns:
            Dict[str, float]: A dictionary containing the average validation metrics.
        """
        self.model.eval()
        
        total_loss_val, bce_loss_val, contrastive_loss_val_val = 0.0, 0.0, 0.0
        correct_preds, total_samples, valid_batches = 0, 0, 0
        tp, fp, fn = 0, 0, 0

        progress_bar_val = tqdm(val_loader, desc=f"Epoch {self.epoch+1} - Validation", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar_val):
                batch = self._batch_to_device(batch)
                
                model_call_args = {
                    'user_idx': batch['user_idx'],
                    'item_idx': batch['item_idx'],
                    'image': batch['image'],
                    'text_input_ids': batch['text_input_ids'],
                    'text_attention_mask': batch['text_attention_mask'],
                    'numerical_features': batch['numerical_features']
                }
                if 'tag_idx' in batch:
                    model_call_args['tag_idx'] = batch['tag_idx']
                    
                if 'clip_text_input_ids' in batch: model_call_args['clip_text_input_ids'] = batch['clip_text_input_ids']
                if 'clip_text_attention_mask' in batch: model_call_args['clip_text_attention_mask'] = batch['clip_text_attention_mask']
                
                vision_features_for_loss_val, text_features_for_loss_val = None, None

                if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                    model_call_args['return_embeddings'] = True
                    output_tuple_val = self.model(**model_call_args)
                    output_val, vision_features_for_loss_val, text_features_for_loss_val, _ = output_tuple_val
                else:
                    model_call_args['return_embeddings'] = False
                    output_val = self.model(**model_call_args)

                output_val = output_val.squeeze(-1)

                loss_dict_val = self.criterion(
                    output_val, batch['label'], 
                    vision_features_for_loss_val,
                    text_features_for_loss_val,
                    self.model.temperature if hasattr(self.model, 'temperature') else None
                )
                
                if torch.isfinite(loss_dict_val['total']):
                    total_loss_val += loss_dict_val['total'].item()
                    bce_loss_val += loss_dict_val['bce'].item()
                    contrastive_loss_val_val += loss_dict_val.get('contrastive', torch.tensor(0.0)).item()
                    valid_batches += 1

                    if output_val.size() == batch['label'].size():
                        predictions = (output_val > 0.5).float()
                        correct_preds += (predictions == batch['label']).sum().item()
                        tp += ((predictions == 1) & (batch['label'] == 1)).sum().item()
                        fp += ((predictions == 1) & (batch['label'] == 0)).sum().item()
                        fn += ((predictions == 0) & (batch['label'] == 1)).sum().item()

                total_samples += batch['label'].size(0)
        
        avg_total_loss = total_loss_val / valid_batches if valid_batches > 0 else float('nan')
        avg_bce_loss = bce_loss_val / valid_batches if valid_batches > 0 else float('nan')
        avg_contrastive_loss = contrastive_loss_val_val / valid_batches if valid_batches > 0 else float('nan')
        avg_accuracy = correct_preds / total_samples if total_samples > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'total_loss': avg_total_loss,
            'bce_loss': avg_bce_loss,
            'contrastive_loss': avg_contrastive_loss,
            'accuracy': avg_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Moves all tensors in a batch to the configured device.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary of tensors.

        Returns:
            Dict[str, torch.Tensor]: The batch with all tensors moved to the correct device.
        """
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], current_epoch: int):
        """
        Logs training and validation metrics to Weights & Biases.

        Args:
            train_metrics (Dict[str, float]): A dictionary of training metrics.
            val_metrics (Dict[str, float]): A dictionary of validation metrics.
            current_epoch (int): The current epoch number.
        """
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

    def _check_early_stopping(self, score: float, patience: int) -> bool:
        """
        Checks if the early stopping criteria have been met based on the score.

        Args:
            score (float): The score for the current epoch (e.g., val_loss or val_f1_score).
            patience (int): The number of epochs to wait for improvement.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if np.isnan(score):
            print("Warning: Early stopping score is NaN. Skipping check for this epoch.")
            return False

        is_improvement = False
        # Determine if the current score is an improvement based on the configured direction.
        if self.config.training.early_stopping_direction == 'minimize':
            if score < self.best_early_stopping_score:
                is_improvement = True
        else: # 'maximize'
            if score > self.best_early_stopping_score:
                is_improvement = True

        if is_improvement:
            # If score has improved, update the best score and reset the patience counter.
            self.best_early_stopping_score = score
            self.patience_counter = 0
            self.save_checkpoint('best_model.pth', is_best=True)
            return False
        else:
            # If no improvement, increment the counter and check if patience is exceeded.
            self.patience_counter += 1
            return self.patience_counter >= patience

    def _print_epoch_summary(self, current_epoch: int, total_epochs: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """
        Prints a formatted summary of the completed epoch's results.

        Args:
            current_epoch (int): The current epoch number.
            total_epochs (int): The total number of epochs for the training run.
            train_metrics (Dict[str, float]): Metrics from the training epoch.
            val_metrics (Dict[str, float]): Metrics from the validation epoch.
        """
        train_contrastive_loss = train_metrics.get('contrastive_loss', 0.0)
        val_contrastive_loss = val_metrics.get('contrastive_loss', np.nan)

        print(f"\nEpoch {current_epoch+1}/{total_epochs}")
        print(f"Train Loss: {train_metrics['total_loss']:.4f} (BCE: {train_metrics['bce_loss']:.4f}, Contrastive: {train_contrastive_loss:.4f})")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['f1_score']:.4f}")
        
        val_loss_str = f"{val_metrics['total_loss']:.4f}" if not np.isnan(val_metrics['total_loss']) else "N/A"
        val_bce_str = f"{val_metrics.get('bce_loss', np.nan):.4f}" if not np.isnan(val_metrics.get('bce_loss', np.nan)) else "N/A"
        val_contrastive_str = f"{val_contrastive_loss:.4f}" if not np.isnan(val_contrastive_loss) else "N/A"
        val_acc_str = f"{val_metrics['accuracy']:.4f}" if not np.isnan(val_metrics['accuracy']) else "N/A"
        val_f1_str = f"{val_metrics.get('f1_score', np.nan):.4f}" if not np.isnan(val_metrics.get('f1_score', np.nan)) else "N/A"

        print(f"Val Loss: {val_loss_str} (BCE: {val_bce_str}, Contrastive: {val_contrastive_str})")
        print(f"Val Acc: {val_acc_str} | Val F1: {val_f1_str}")
        
        current_lr = self.get_learning_rate()
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)

    def save_checkpoint(self, filename: str, is_best: bool = False, additional_info: Optional[Dict[str, Any]] = None):
        """
        Saves the model and optimizer state to a checkpoint file.

        Args:
            filename (str): The name of the checkpoint file.
            is_best (bool): If True, indicates that this is the best model so far based on validation loss.
            additional_info (Optional[Dict[str, Any]]): Additional information to save in the checkpoint.
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            # Save the new generic best score and its context
            'best_early_stopping_score': self.best_early_stopping_score,
            'early_stopping_metric': self.config.training.early_stopping_metric,
            'early_stopping_direction': self.config.training.early_stopping_direction,
            'training_history': self.training_history,  
            'best_metrics': self.get_all_best_metrics()  
        }
        
        # Add trial information if available
        if self.trial_info:
            checkpoint['trial_info'] = self.trial_info
        
        # Add any additional info passed to the method
        if additional_info:
            checkpoint['additional_info'] = additional_info
        
        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.model_checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            print(f"Saved best model checkpoint to {path}")
            # Also save as 'best_model.pth' for easy access
            best_path = self.model_checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            try:
                if wandb.run is not None:
                    wandb.save(str(path))
                    wandb.save(str(best_path))
            except Exception as e:
                print(f"Warning: Failed to save checkpoint to wandb: {e}")


    def load_checkpoint(self, filename: str):
        """
        Loads the model and optimizer state from a checkpoint file.

        Args:
            filename (str): The name of the checkpoint file to load.
        """
        path = self.model_checkpoint_dir / filename
        if not path.exists():
            print(f"Warning: Checkpoint file not found at {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        # Load the generic best score. Fallback to old best_val_loss for backward compatibility.
        self.best_early_stopping_score = checkpoint.get('best_early_stopping_score', checkpoint.get('best_val_loss', None))

        # Restore training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        # Restore trial information if available
        if 'trial_info' in checkpoint:
            self.trial_info = checkpoint['trial_info']
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded optimizer state from checkpoint.")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Optimizer will be reinitialized.")
        elif 'optimizer_state_dict' in checkpoint and not self.optimizer:
            print(f"Warning: Optimizer state found in checkpoint, but trainer's optimizer is not initialized.")
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Loaded scheduler state from checkpoint.")
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}. Scheduler may not resume correctly.")
        
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")



    def get_learning_rate(self) -> float:
        """
        Retrieves the current learning rate from the optimizer.

        Returns:
            float: The current learning rate.
        """
        if self.optimizer is None: return 0.0
        for param_group in self.optimizer.param_groups: return param_group['lr']
        return 0.0

    def get_model_checkpoint_dir(self) -> Path:
        """
        Returns the path to the model-specific checkpoint directory.

        Returns:
            Path: The Path object for the directory.
        """
        return self.model_checkpoint_dir

    def get_encoders_dir(self) -> Path:
        """
        Returns the path to the shared encoders directory.

        Returns:
            Path: The Path object for the directory.
        """
        return self.encoders_dir

    def get_best_metric(self, metric_name: str = 'val_loss') -> float:
        """
        Returns the best value achieved for a specified metric during training.
        
        Args:
            metric_name: Name of the metric to retrieve. Common options:
                        'val_loss', 'val_accuracy', 'val_f1_score',
                        'train_loss', 'train_accuracy', 'train_f1_score'
        
        Returns:
            The best value achieved for the metric. Returns inf for losses
            (where lower is better) and -inf for metrics like accuracy
            (where higher is better) if the metric was never recorded.
        """
        # Check if we have the metric in our best_metrics tracking
        if metric_name in self.training_history['best_metrics']:
            return self.training_history['best_metrics'][metric_name]
        
        # Fallback for backward compatibility
        if metric_name == 'val_loss':
            return self.best_val_loss
        
        # Try to compute from history
        if metric_name.startswith('val_'):
            metric_key = metric_name.replace('val_', '')
            if self.training_history['val_metrics']:
                values = [m.get(metric_key, float('inf')) for m in self.training_history['val_metrics']]
                if values:
                    # For loss metrics, lower is better
                    if 'loss' in metric_name:
                        return min(values)
                    # For other metrics, higher is better
                    else:
                        return max(values)
        
        elif metric_name.startswith('train_'):
            metric_key = metric_name.replace('train_', '')
            if self.training_history['train_metrics']:
                values = [m.get(metric_key, float('inf')) for m in self.training_history['train_metrics']]
                if values:
                    if 'loss' in metric_name:
                        return min(values)
                    else:
                        return max(values)
        
        # Return appropriate default based on metric type
        if 'loss' in metric_name:
            return float('inf')
        else:
            return float('-inf')
    
    def get_all_best_metrics(self) -> Dict[str, float]:
        """
        Returns a dictionary of all best metrics achieved during training.
        
        Returns:
            Dictionary mapping metric names to their best values.
        """
        metrics = {}
        
        # Get validation metrics
        for metric_name in ['total_loss', 'bce_loss', 'contrastive_loss', 'accuracy', 'f1_score', 'precision', 'recall']:
            val_metric = f'val_{metric_name}'
            best_val = self.get_best_metric(val_metric)
            if best_val != float('inf') and best_val != float('-inf'):
                metrics[val_metric] = best_val
        
        # Get training metrics
        for metric_name in ['total_loss', 'bce_loss', 'contrastive_loss', 'accuracy', 'f1_score']:
            train_metric = f'train_{metric_name}'
            best_train = self.get_best_metric(train_metric)
            if best_train != float('inf') and best_train != float('-inf'):
                metrics[train_metric] = best_train
        
        return metrics
    
    def get_trial_number(self) -> Optional[int]:
        """
        Returns the Optuna trial number if this trainer is being used in a trial.
        
        Returns:
            Trial number or None if not part of an Optuna trial.
        """
        if self.trial_info and 'trial_number' in self.trial_info:
            return self.trial_info['trial_number']
        return None

    def update_trial_info(self, info: Dict[str, Any]):
        """
        Updates the trial information during training.
        
        Args:
            info: Dictionary with trial information to update.
        """
        if self.trial_info is None:
            self.trial_info = {}
        self.trial_info.update(info)