# scripts/train.py - Simplified without cross-modal attention
#!/usr/bin/env python
"""
Training script for the simplified multimodal recommender system
"""
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.data.processors import NumericalProcessor
from typing import List
import dataclasses
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config, TextAugmentationConfig
from src.data.dataset import MultimodalDataset
from src.models.multimodal import MultimodalRecommender
from src.training.trainer import Trainer

# Use simplified cache instead of old cache system
try:
    from src.data.simple_cache import SimpleFeatureCache
except ImportError:
    # Create a minimal fallback cache if the file doesn't exist
    class SimpleFeatureCache:
        def __init__(self, *args, **kwargs):
            self.cache = {}
            print("Using fallback SimpleFeatureCache")
        
        def get(self, item_id):
            return self.cache.get(item_id)
        
        def set(self, item_id, features):
            self.cache[item_id] = features
        
        def print_stats(self):
            print(f"Simple cache: {len(self.cache)} items")

def fit_numerical_scaler(df, numerical_cols, method, scaler_path):
    processor = NumericalProcessor()
    processor.fit_scaler(df, numerical_cols, method)
    processor.save_scaler(scaler_path)
    return processor.scaler

def load_numerical_scaler(scaler_path):
    processor = NumericalProcessor()
    processor.load_scaler(scaler_path)
    return processor.scaler

def main():
    """Main function for training the multimodal recommender system."""
    parser = argparse.ArgumentParser(description='Train multimodal recommender')
    parser.add_argument('--config', type=str, default='configs/simple_config.yaml', help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='MultimodalRecommender', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name for this training')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    data_config = config.data
    model_config = config.model
    training_config = config.training
    print(f"Loaded configuration from {args.config}")

    # Initialize Weights & Biases if requested
    if args.use_wandb:
        try:
            config_dict_for_wandb = dataclasses.asdict(config)
            wandb.init(
                project=args.wandb_project, 
                entity=args.wandb_entity, 
                name=args.wandb_run_name, 
                config=config_dict_for_wandb, 
                reinit=True
            )
            print("Weights & Biases logging enabled.")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("train/learning_rate", step_metric="epoch")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}. Proceeding without wandb logging.")
            args.use_wandb = False

    device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        # Load the full processed data to fit encoders and scalers consistently
        print("\nLoading full processed data to fit encoders and scalers...")
        item_info_df_full = pd.read_csv(data_config.processed_item_info_path)
        interactions_df_full = pd.read_csv(data_config.processed_interactions_path)

        # Initialize simplified cache system
        simple_cache_instance = None
        cache_config = data_config.cache_config
        
        if cache_config.enabled:
            print(f"Initializing SimpleFeatureCache:")
            print(f"  Strategy: Memory-based caching")
            print(f"  Max memory items: {cache_config.max_memory_items}")
            print(f"  Cache directory: {cache_config.cache_directory}")
            print(f"  Use disk: {cache_config.use_disk}")
            
            simple_cache_instance = SimpleFeatureCache(
                max_memory_items=cache_config.max_memory_items,
                cache_dir=cache_config.cache_directory,
                use_disk=cache_config.use_disk
            )
            simple_cache_instance.print_stats()
        else:
            print("Feature caching is disabled")

        # Handle numerical scaler fitting or loading
        numerical_scaler = None
        scaler_path_obj = Path(data_config.scaler_path)
        if data_config.numerical_normalization_method in ['standardization', 'min_max']:
            if scaler_path_obj.exists():
                numerical_scaler = load_numerical_scaler(scaler_path_obj)
            else:
                print(f"Scaler not found at {scaler_path_obj}. Fitting a new one...")
                numerical_scaler = fit_numerical_scaler(
                    item_info_df_full, 
                    data_config.numerical_features_cols, 
                    data_config.numerical_normalization_method, 
                    scaler_path_obj
                )

        # Determine the image folder to be used by the Dataset
        effective_image_folder = data_config.image_folder
        if hasattr(data_config, 'offline_image_compression') and \
           data_config.offline_image_compression.enabled and \
           hasattr(data_config, 'processed_image_destination_folder') and \
           data_config.processed_image_destination_folder:
            effective_image_folder = data_config.processed_image_destination_folder
            print(f"Using processed (compressed/resized) images from: {effective_image_folder}")
        else:
            print(f"Using original images from: {effective_image_folder}")

        print("\nCreating dataset for encoder fitting...")
        # Create a full dataset instance to fit the global user and item encoders
        full_dataset_for_encoders = MultimodalDataset(
            interactions_df=interactions_df_full,
            item_info_df=item_info_df_full,
            image_folder=effective_image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=False,
            negative_sampling_ratio=0,
            # Simplified cache parameters
            cache_features=cache_config.enabled,
            cache_max_items=cache_config.max_memory_items,
            cache_dir=cache_config.cache_directory,
            cache_to_disk=cache_config.use_disk,
            # Other parameters
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False
        )
        
        print(f"Fitted encoders on full dataset:")
        print(f"  Number of users: {full_dataset_for_encoders.n_users}")
        print(f"  Number of items: {full_dataset_for_encoders.n_items}")
        
        # Load pre-split data from files
        print("\nLoading pre-split training and validation data...")
        train_interactions_df = pd.read_csv(data_config.train_data_path)
        val_interactions_df = pd.read_csv(data_config.val_data_path)

        # Filter item_info_df_full to only include items present in the loaded interactions
        all_item_ids_in_splits = pd.concat([
            train_interactions_df['item_id'], 
            val_interactions_df['item_id']
        ]).unique()
        item_info_df_for_datasets = item_info_df_full[
            item_info_df_full['item_id'].isin(all_item_ids_in_splits)
        ].reset_index(drop=True)

        print(f"\nTraining interactions: {len(train_interactions_df)}")
        print(f"Validation interactions: {len(val_interactions_df)}")

        print("\nCreating dataset instances using pre-fitted encoders...")
        # Create training dataset
        train_dataset = MultimodalDataset(
            interactions_df=train_interactions_df,
            item_info_df=item_info_df_for_datasets,
            image_folder=effective_image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=data_config.text_augmentation,
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=True,
            # Simplified cache parameters
            cache_features=cache_config.enabled,
            cache_max_items=cache_config.max_memory_items,
            cache_dir=cache_config.cache_directory,
            cache_to_disk=cache_config.use_disk
        )
        
        # Assign globally fitted encoders and counts to the training dataset
        train_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        train_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        train_dataset.n_users = full_dataset_for_encoders.n_users
        train_dataset.n_items = full_dataset_for_encoders.n_items

        # Create validation dataset
        val_dataset = MultimodalDataset(
            interactions_df=val_interactions_df,
            item_info_df=item_info_df_for_datasets,
            image_folder=effective_image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=TextAugmentationConfig(enabled=False),  # No augmentation for validation
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False,
            # Simplified cache parameters
            cache_features=cache_config.enabled,
            cache_max_items=cache_config.max_memory_items,
            cache_dir=cache_config.cache_directory,
            cache_to_disk=cache_config.use_disk
        )
        
        # Assign globally fitted encoders and counts to the validation dataset
        val_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        val_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        val_dataset.n_users = full_dataset_for_encoders.n_users
        val_dataset.n_items = full_dataset_for_encoders.n_items
        
        print(f"\nDataset sizes after final setup:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

        # Create data loaders for training and validation
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=True, 
            num_workers=training_config.num_workers, 
            pin_memory=True,
            persistent_workers=True if training_config.num_workers > 0 else False,  
            prefetch_factor=2  
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False,
            num_workers=training_config.num_workers, 
            pin_memory=True,
            persistent_workers=True if training_config.num_workers > 0 else False,
            prefetch_factor=2
        ) if len(val_dataset) > 0 else None

        # Save encoders to disk before training begins
        encoders_dir = Path(config.checkpoint_dir) / 'encoders'
        encoders_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nSaving encoders before training...")
        with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.user_encoder, f)
        with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.item_encoder, f)
        print(f"Saved encoders to {encoders_dir}")

        print("\nInitializing model...")
        
        # Simplified model initialization - no more model class selection
        print("Using MultimodalRecommender")
        
        # Prepare model parameters based on configuration
        model_params = {
            'n_users': full_dataset_for_encoders.n_users,
            'n_items': full_dataset_for_encoders.n_items,
            'embedding_dim': model_config.embedding_dim,
            'vision_model_name': model_config.vision_model,
            'language_model_name': model_config.language_model,
            'freeze_vision': model_config.freeze_vision,
            'freeze_language': model_config.freeze_language,
            'use_contrastive': model_config.use_contrastive,
            'dropout_rate': model_config.dropout_rate,
            'num_attention_heads': model_config.num_attention_heads,
            'attention_dropout': model_config.attention_dropout,
            'fusion_hidden_dims': model_config.fusion_hidden_dims,
            'fusion_activation': model_config.fusion_activation,
            'use_batch_norm': model_config.use_batch_norm,
            'projection_hidden_dim': model_config.projection_hidden_dim,
            'final_activation': model_config.final_activation,
            'init_method': model_config.init_method,
            'contrastive_temperature': model_config.contrastive_temperature,
        }
        
        model = MultimodalRecommender(**model_params).to(device)

        # Print model parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Initialize trainer with model and device
        trainer = Trainer(
            model=model, 
            device=device, 
            checkpoint_dir=config.checkpoint_dir, 
            use_contrastive=model_config.use_contrastive
        )
        trainer.criterion.contrastive_weight = training_config.contrastive_weight
        trainer.criterion.bce_weight = training_config.bce_weight

        # Resume training from checkpoint if specified
        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        print("\nStarting training...")
        # Prepare training parameters for the trainer
        training_params_for_trainer = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'epochs': training_config.epochs,
            'lr': training_config.learning_rate, 
            'weight_decay': training_config.weight_decay,
            'patience': training_config.patience, 
            'gradient_clip': training_config.gradient_clip,
            'optimizer_type': training_config.optimizer_type, 
            'adam_beta1': training_config.adam_beta1,
            'adam_beta2': training_config.adam_beta2, 
            'adam_eps': training_config.adam_eps,
            'use_lr_scheduler': training_config.use_lr_scheduler, 
            'lr_scheduler_type': training_config.lr_scheduler_type,
            'lr_scheduler_patience': training_config.lr_scheduler_patience,
            'lr_scheduler_factor': training_config.lr_scheduler_factor, 
            'lr_scheduler_min_lr': training_config.lr_scheduler_min_lr
        }
        # Start training
        train_losses, val_losses = trainer.train(**training_params_for_trainer)

        # Save the final model checkpoint and encoders
        trainer.save_checkpoint('final_model.pth')
        print(f"\nSaved final model to {config.checkpoint_dir}/final_model.pth")
        
        # Save encoders again
        encoders_dir = Path(config.checkpoint_dir) / 'encoders'
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.user_encoder, f)
        with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.item_encoder, f)
        print(f"Saved encoders to {encoders_dir}")

        # Plot and save training curves if losses were recorded
        if train_losses and val_losses:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            results_fig_dir = Path(config.results_dir) / 'figures'
            results_fig_dir.mkdir(parents=True, exist_ok=True)
            plot_path = results_fig_dir / 'training_curves.png'
            plt.savefig(plot_path)
            print(f"\nSaved training curves to {plot_path}")
            if args.use_wandb and wandb.run is not None:
                try:
                    wandb.log({"training_validation_loss_curves": wandb.Image(str(plot_path))})
                except Exception as e:
                    print(f"Warning: Failed to log training curves to wandb: {e}")
        else:
            print("\nSkipping plotting training curves as training might have been skipped or no validation occurred.")

        # Save the effective configuration to results directory
        config_save_path = Path(config.results_dir) / 'training_run_config.yaml'
        config.to_yaml(str(config_save_path))
        print(f"Saved effective configuration to {config_save_path}")
        if args.use_wandb and wandb.run is not None:
            try:
                wandb.save(str(config_save_path))
            except Exception as e:
                print(f"Warning: Failed to save config to wandb: {e}")
        print("\nTraining completed!")
        
    finally:
        # Ensure Weights & Biases run is finished
        if args.use_wandb and wandb.run is not None:
            print("Finishing wandb run...")
            wandb.finish()

if __name__ == '__main__':
    main()