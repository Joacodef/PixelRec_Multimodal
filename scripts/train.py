#!/usr/bin/env python
"""
Training script for the multimodal recommender system
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List
import dataclasses
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config, TextAugmentationConfig
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.training.trainer import Trainer
from src.data.splitting import DataSplitter


def fit_numerical_scaler(df: pd.DataFrame, numerical_cols: List[str], method: str, scaler_path: Path):
    """Fits a scaler on the numerical columns of the dataframe and saves it."""
    if not numerical_cols or method in ['none', 'log1p']:
        print(f"Scaler fitting skipped for method: {method} or no numerical columns.")
        return None

    data_to_scale = df[numerical_cols].fillna(0).values
    
    if method == 'standardization':
        scaler = StandardScaler()
    elif method == 'min_max':
        scaler = MinMaxScaler()
    else:
        print(f"Unknown scaling method for fitting: {method}. No scaler fitted.")
        return None
        
    print(f"Fitting {method} scaler on {len(data_to_scale)} samples...")
    scaler.fit(data_to_scale)
    
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    return scaler


def load_numerical_scaler(scaler_path: Path):
    """Loads a pre-fitted scaler."""
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded from {scaler_path}")
        return scaler
    print(f"Warning: Scaler not found at {scaler_path}. Proceeding without pre-loaded scaler.")
    return None


def main():
    parser = argparse.ArgumentParser(description='Train multimodal recommender')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='MultimodalRecommender',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='Weights & Biases entity (username or team)'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Weights & Biases run name for this training'
    )
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)
    data_config = config.data
    model_config = config.model
    training_config = config.training
    print(f"Loaded configuration from {args.config}")

    # Initialize wandb if enabled
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
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}. Proceeding without wandb logging.")
            args.use_wandb = False

    device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        # ===== LOAD PROCESSED DATA =====
        print("\nLoading processed data...")
        item_info_df = pd.read_csv(data_config.processed_item_info_path)
        interactions_df = pd.read_csv(data_config.processed_interactions_path)

        # Apply sampling if configured
        if data_config.sample_size:
            print(f"Sampling {data_config.sample_size} interactions from processed data...")
            interactions_df = interactions_df.sample(
                n=min(data_config.sample_size, len(interactions_df)),
                random_state=42
            ).reset_index(drop=True)
            sampled_item_ids = set(interactions_df['item_id'].unique())
            item_info_df = item_info_df[item_info_df['item_id'].isin(sampled_item_ids)].reset_index(drop=True)

        # ===== NUMERICAL SCALER SETUP =====
        numerical_scaler = None
        scaler_path_obj = Path(data_config.scaler_path)
        
        if data_config.numerical_normalization_method in ['standardization', 'min_max']:
            if scaler_path_obj.exists():
                numerical_scaler = load_numerical_scaler(scaler_path_obj)
            else:
                print(f"Scaler not found at {scaler_path_obj}. Fitting a new one...")
                numerical_scaler = fit_numerical_scaler(
                    item_info_df,
                    data_config.numerical_features_cols,
                    data_config.numerical_normalization_method,
                    scaler_path_obj
                )

        # ===== DATA SPLITTING =====
        print(f"\nSplitting data using strategy: {data_config.splitting.strategy}")
        
        total_interactions = len(interactions_df)
        unique_users = interactions_df['user_id'].nunique()
        print(f"Dataset info: {total_interactions} interactions, {unique_users} unique users")
        
        splitter = DataSplitter(random_state=data_config.splitting.random_state)
        
        # Choose splitting strategy based on dataset size and configuration
        if total_interactions < 5000 or unique_users < 100:
            print("Small dataset detected. Using leave_one_out strategy.")
            train_interactions_df, val_interactions_df = splitter.leave_one_out_split(
                interactions_df,
                strategy='random'
            )
        else:
            # Use the configured splitting strategy
            if data_config.splitting.strategy == 'stratified':
                train_interactions_df, val_interactions_df = splitter.stratified_split(
                    interactions_df,
                    train_ratio=data_config.splitting.train_ratio,
                    min_interactions_per_user=data_config.splitting.min_interactions_per_user
                )
            elif data_config.splitting.strategy == 'user':
                train_interactions_df, val_interactions_df = splitter.user_based_split(
                    interactions_df,
                    train_ratio=data_config.splitting.train_ratio,
                    min_interactions_per_user=data_config.splitting.min_interactions_per_user
                )
            elif data_config.splitting.strategy == 'item':
                train_interactions_df, val_interactions_df = splitter.item_based_split(
                    interactions_df,
                    train_ratio=data_config.splitting.train_ratio,
                    min_interactions_per_item=data_config.splitting.min_interactions_per_item
                )
            elif data_config.splitting.strategy == 'temporal':
                if not data_config.splitting.timestamp_col:
                    raise ValueError("timestamp_col must be specified for temporal splitting")
                train_interactions_df, val_interactions_df = splitter.temporal_split(
                    interactions_df,
                    timestamp_col=data_config.splitting.timestamp_col,
                    train_ratio=data_config.splitting.train_ratio
                )
            elif data_config.splitting.strategy == 'leave_one_out':
                train_interactions_df, val_interactions_df = splitter.leave_one_out_split(
                    interactions_df,
                    strategy=data_config.splitting.leave_one_out_strategy
                )
            elif data_config.splitting.strategy == 'simple_random':
                train_interactions_df, val_interactions_df = splitter.simple_random_split(
                    interactions_df,
                    train_ratio=data_config.splitting.train_ratio
                )
            else:
                raise ValueError(f"Unknown splitting strategy: {data_config.splitting.strategy}")

        # Validate split quality
        split_stats = splitter.get_split_statistics(train_interactions_df, val_interactions_df)
        print(f"\nSplit statistics:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")

        print(f"\nTraining interactions: {len(train_interactions_df)}")
        print(f"Validation interactions: {len(val_interactions_df)}")

        # ===== CREATE DATASETS =====
        print("\nCreating dataset instances...")
        
        # First, create a full dataset just for fitting encoders
        full_dataset_for_encoders = MultimodalDataset(
            interactions_df=interactions_df,  # Use all interactions for encoder fitting
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=False,  # No need for negative samples just for encoders
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False
        )
        
        print(f"Fitted encoders on full dataset:")
        print(f"  Number of users: {full_dataset_for_encoders.n_users}")
        print(f"  Number of items: {full_dataset_for_encoders.n_items}")
        
        # Create training dataset with negative sampling
        train_dataset = MultimodalDataset(
            interactions_df=train_interactions_df,
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=data_config.text_augmentation,
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=True
        )
        
        # Use encoders from full dataset to ensure consistency
        train_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        train_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        train_dataset.n_users = full_dataset_for_encoders.n_users
        train_dataset.n_items = full_dataset_for_encoders.n_items
        
        # Create validation dataset (no augmentation, but with negative sampling)
        val_dataset = MultimodalDataset(
            interactions_df=val_interactions_df,
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=TextAugmentationConfig(enabled=False),
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False
        )
        
        # Use same encoders for validation
        val_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        val_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        val_dataset.n_users = full_dataset_for_encoders.n_users
        val_dataset.n_items = full_dataset_for_encoders.n_items
        
        print(f"\nDataset sizes after negative sampling:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

        # ===== CREATE DATA LOADERS =====
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            pin_memory=True
        )

        # ===== INITIALIZE MODEL =====
        print("\nInitializing model...")
        model = PretrainedMultimodalRecommender(
            n_users=full_dataset_for_encoders.n_users,
            n_items=full_dataset_for_encoders.n_items,
            embedding_dim=model_config.embedding_dim,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            freeze_vision=model_config.freeze_vision,
            freeze_language=model_config.freeze_language,
            use_contrastive=model_config.use_contrastive,
            dropout_rate=model_config.dropout_rate
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # ===== INITIALIZE TRAINER =====
        trainer = Trainer(
            model=model,
            device=device,
            checkpoint_dir=config.checkpoint_dir,
            use_contrastive=model_config.use_contrastive
        )
        
        # Set loss weights from config
        trainer.criterion.contrastive_weight = training_config.contrastive_weight
        trainer.criterion.bce_weight = training_config.bce_weight

        # Resume from checkpoint if specified
        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # ===== TRAIN MODEL =====
        print("\nStarting training...")
        train_losses, val_losses = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config.epochs,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            patience=training_config.patience,
            gradient_clip=training_config.gradient_clip
        )

        # ===== SAVE FINAL MODEL AND ARTIFACTS =====
        # Save final model
        trainer.save_checkpoint('final_model.pth')
        print(f"\nSaved final model to {config.checkpoint_dir}/final_model.pth")

        # Save encoders
        encoders_dir = Path(config.checkpoint_dir) / 'encoders'
        encoders_dir.mkdir(parents=True, exist_ok=True)
        
        with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.user_encoder, f)
        with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.item_encoder, f)
        print(f"Saved encoders to {encoders_dir}")

        # Plot training curves
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
        
        # Log plot to wandb if enabled
        if args.use_wandb and wandb.run is not None:
            try:
                wandb.log({"training_validation_loss_curves": wandb.Image(str(plot_path))})
            except Exception as e:
                print(f"Warning: Failed to log training curves to wandb: {e}")

        # Save configuration used for this run
        config_save_path = Path(config.results_dir) / 'training_run_config.yaml'
        config.to_yaml(str(config_save_path))
        print(f"Saved effective configuration to {config_save_path}")
        
        # Log config file to wandb if enabled
        if args.use_wandb and wandb.run is not None:
            try:
                wandb.save(str(config_save_path))
            except Exception as e:
                print(f"Warning: Failed to save config to wandb: {e}")

        print("\nTraining completed!")

    finally:
        # Ensure wandb run is finished if it was initialized
        if args.use_wandb and wandb.run is not None:
            print("Finishing wandb run...")
            wandb.finish()


if __name__ == '__main__':
    main()