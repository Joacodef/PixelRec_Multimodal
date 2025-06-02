#!/usr/bin/env python
"""
Training script for the multimodal recommender system
"""
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List
import dataclasses # For converting dataclass to dict for wandb config
import wandb # Import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config, TextAugmentationConfig # Ensure TextAugmentationConfig is imported if used directly
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.training.trainer import Trainer
from src.data.splitting import DataSplitter, create_robust_splits


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
    # Add wandb arguments
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='MultimodalRecommender', # Default project name
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None, # User should typically set this via command line or environment variable
        help='Weights & Biases entity (username or team)'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None, # wandb will auto-generate if None
        help='Weights & Biases run name for this training'
    )
    args = parser.parse_args()

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
            if args.wandb_run_name:
                print(f"Wandb run name: {args.wandb_run_name}")
            if args.wandb_project:
                print(f"Wandb project: {args.wandb_project}")
            if args.wandb_entity:
                print(f"Wandb entity: {args.wandb_entity}")
            else:
                print(f"Wandb entity: Using default or environment variable WANDB_ENTITY.")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}. Proceeding without wandb logging.")
            args.use_wandb = False

    device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        print("\nLoading processed data...")
        item_info_df = pd.read_csv(data_config.processed_item_info_path)
        interactions_df = pd.read_csv(data_config.processed_interactions_path)

        if data_config.sample_size:
            print(f"Sampling {data_config.sample_size} interactions from processed data...")
            interactions_df = interactions_df.sample(
                n=min(data_config.sample_size, len(interactions_df)),
                random_state=42
            ).reset_index(drop=True)
            sampled_item_ids = set(interactions_df['item_id'].unique())
            item_info_df = item_info_df[item_info_df['item_id'].isin(sampled_item_ids)].reset_index(drop=True)

        # ===== NUMERICAL SCALER SETUP =====
        numerical_scaler = None  # Initialize to None
        scaler_path_obj = Path(data_config.scaler_path)
        
        if data_config.numerical_normalization_method in ['standardization', 'min_max']:
            if scaler_path_obj.exists():
                numerical_scaler = load_numerical_scaler(scaler_path_obj)
            else:
                print(f"Scaler not found at {scaler_path_obj}. Fitting a new one based on current item_info_df...")
                if not all(col in item_info_df.columns for col in data_config.numerical_features_cols):
                    print(f"Error: Not all numerical_features_cols ({data_config.numerical_features_cols}) found in item_info_df columns ({item_info_df.columns}). Cannot fit scaler.")
                    sys.exit(1)
                numerical_scaler = fit_numerical_scaler(
                    item_info_df,
                    data_config.numerical_features_cols,
                    data_config.numerical_normalization_method,
                    scaler_path_obj
                )
                if numerical_scaler is None and data_config.numerical_normalization_method not in ['none', 'log1p']:
                    print(f"Warning: Scaler fitting failed for method {data_config.numerical_normalization_method}.")

        # ===== DATA SPLITTING =====
        print(f"\nSplitting data using strategy: {data_config.splitting.strategy}")
        
        # For small datasets, adjust the splitting strategy
        total_interactions = len(interactions_df)
        unique_users = interactions_df['user_id'].nunique()
        
        print(f"Dataset info: {total_interactions} interactions, {unique_users} unique users")
        
        # Adjust strategy for very small datasets
        if total_interactions < 5000 or unique_users < 100:
            print(f"Small dataset detected. Adjusting splitting strategy...")
            if data_config.splitting.strategy == 'stratified':
                # Use leave_one_out for small datasets
                print("Switching to leave_one_out strategy for small dataset")
                splitter = DataSplitter(random_state=data_config.splitting.random_state)
                train_interactions_df, val_interactions_df = splitter.leave_one_out_split(
                    interactions_df,
                    strategy='random'
                )
            else:
                # Use the configured strategy
                splitter = DataSplitter(random_state=data_config.splitting.random_state)
                
                if data_config.splitting.strategy == 'user':
                    train_interactions_df, val_interactions_df = splitter.user_based_split(
                        interactions_df,
                        train_ratio=data_config.splitting.train_ratio,
                        min_interactions_per_user=max(2, data_config.splitting.min_interactions_per_user)
                    )
                elif data_config.splitting.strategy == 'item':
                    train_interactions_df, val_interactions_df = splitter.item_based_split(
                        interactions_df,
                        train_ratio=data_config.splitting.train_ratio,
                        min_interactions_per_item=max(2, data_config.splitting.min_interactions_per_item)
                    )
                elif data_config.splitting.strategy == 'leave_one_out':
                    train_interactions_df, val_interactions_df = splitter.leave_one_out_split(
                        interactions_df,
                        strategy=data_config.splitting.leave_one_out_strategy
                    )
                else:
                    # Fallback to simple random split
                    print("Using simple random split as fallback")
                    train_interactions_df, val_interactions_df = splitter.simple_random_split(
                        interactions_df,
                        train_ratio=data_config.splitting.train_ratio
                    )
        else:
            splitter = DataSplitter(random_state=data_config.splitting.random_state)
            
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
            else:
                raise ValueError(f"Unknown splitting strategy: {data_config.splitting.strategy}")

        # Validate split quality
        split_stats = splitter.get_split_statistics(train_interactions_df, val_interactions_df)
        print(f"\nSplit statistics:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")

        # Check for problematic splits
        if split_stats['val_interactions'] < 10:
            print(f"WARNING: Very few validation interactions ({split_stats['val_interactions']}). Consider using a larger dataset or different splitting strategy.")
        
        if split_stats['user_overlap_ratio'] > 0.5:
            print(f"WARNING: High user overlap ({split_stats['user_overlap_ratio']:.2%}) between train and validation.")

        print(f"Training interactions: {len(train_interactions_df)}")
        print(f"Validation interactions: {len(val_interactions_df)}")

        # ===== CREATE DATASETS WITH NEGATIVE SAMPLING =====
        print("\nCreating dataset instances with negative sampling...")
        
        # Create full dataset for encoder fitting (using ALL interactions)
        full_dataset_for_encoders = MultimodalDataset(
            interactions_df=interactions_df,  # Full interactions for encoder fitting
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=False,  # Just for encoder fitting
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,  # This should now be defined
            is_train_mode=False
        )
        
        print(f"Fitted encoders on full dataset:")
        print(f"  Number of users: {full_dataset_for_encoders.n_users}")
        print(f"  Number of items: {full_dataset_for_encoders.n_items}")
        
        # Create training dataset with negative sampling
        train_dataset_instance = MultimodalDataset(
            interactions_df=train_interactions_df,
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,  # Enable negative sampling for training
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=data_config.text_augmentation,
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,  # This should now be defined
            is_train_mode=True
        )
        
        # Use encoders from full dataset
        train_dataset_instance.user_encoder = full_dataset_for_encoders.user_encoder
        train_dataset_instance.item_encoder = full_dataset_for_encoders.item_encoder
        train_dataset_instance.n_users = full_dataset_for_encoders.n_users
        train_dataset_instance.n_items = full_dataset_for_encoders.n_items
        
        # Create validation dataset with negative sampling
        val_dataset_instance = MultimodalDataset(
            interactions_df=val_interactions_df,
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,  # Enable negative sampling for validation
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=TextAugmentationConfig(enabled=False),  # No augmentation for validation
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,  # This should now be defined
            is_train_mode=False
        )
        
        # Use encoders from full dataset
        val_dataset_instance.user_encoder = full_dataset_for_encoders.user_encoder
        val_dataset_instance.item_encoder = full_dataset_for_encoders.item_encoder
        val_dataset_instance.n_users = full_dataset_for_encoders.n_users
        val_dataset_instance.n_items = full_dataset_for_encoders.n_items
        
        print(f"Training samples: {len(train_dataset_instance)}")
        print(f"Validation samples: {len(val_dataset_instance)}")

        numerical_scaler = None
        scaler_path_obj = Path(data_config.scaler_path)
        if data_config.numerical_normalization_method in ['standardization', 'min_max']:
            if scaler_path_obj.exists():
                numerical_scaler = load_numerical_scaler(scaler_path_obj)
            else:
                print(f"Scaler not found at {scaler_path_obj}. Fitting a new one based on current item_info_df...")
                if not all(col in item_info_df.columns for col in data_config.numerical_features_cols):
                    print(f"Error: Not all numerical_features_cols ({data_config.numerical_features_cols}) found in item_info_df columns ({item_info_df.columns}). Cannot fit scaler.")
                    sys.exit(1)
                numerical_scaler = fit_numerical_scaler(
                    item_info_df,
                    data_config.numerical_features_cols,
                    data_config.numerical_normalization_method,
                    scaler_path_obj
                )
                if numerical_scaler is None and data_config.numerical_normalization_method not in ['none', 'log1p']:
                     print(f"Warning: Scaler fitting failed for method {data_config.numerical_normalization_method}.")


        print("\nCreating full dataset instance for splitting...")
        full_dataset_for_splitting = MultimodalDataset(
            interactions_df=interactions_df,
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=None,
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False
        )

        print(f"Full dataset size for splitting: {len(full_dataset_for_splitting)}")
        print(f"Number of users: {full_dataset_for_splitting.n_users}")
        print(f"Number of items: {full_dataset_for_splitting.n_items}")

        train_size = int(data_config.train_val_split * len(full_dataset_for_splitting.all_samples))
        # val_size = len(full_dataset_for_splitting.all_samples) - train_size # Not explicitly used later
        
        shuffled_samples = full_dataset_for_splitting.all_samples.sample(frac=1, random_state=42).reset_index(drop=True)
        train_samples_df = shuffled_samples.iloc[:train_size]
        val_samples_df = shuffled_samples.iloc[train_size:]

        print(f"Train samples: {len(train_samples_df)}")
        print(f"Validation samples: {len(val_samples_df)}")

        train_dataset_instance = MultimodalDataset(
            interactions_df=train_samples_df,
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=False,
            text_augmentation_config=data_config.text_augmentation, # Use loaded TextAugmentationConfig
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=True
        )
        train_dataset_instance.user_encoder = full_dataset_for_splitting.user_encoder
        train_dataset_instance.item_encoder = full_dataset_for_splitting.item_encoder
        train_dataset_instance.n_users = full_dataset_for_splitting.n_users
        train_dataset_instance.n_items = full_dataset_for_splitting.n_items
        train_dataset_instance.all_samples = train_samples_df

        val_dataset_instance = MultimodalDataset(
            interactions_df=val_samples_df,
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=False,
            text_augmentation_config=TextAugmentationConfig(enabled=False), # Explicitly disable for val
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False
        )
        val_dataset_instance.user_encoder = full_dataset_for_splitting.user_encoder
        val_dataset_instance.item_encoder = full_dataset_for_splitting.item_encoder
        val_dataset_instance.n_users = full_dataset_for_splitting.n_users
        val_dataset_instance.n_items = full_dataset_for_splitting.n_items
        val_dataset_instance.all_samples = val_samples_df

        train_loader = DataLoader(
            train_dataset_instance, batch_size=training_config.batch_size, shuffle=True,
            num_workers=training_config.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset_instance, batch_size=training_config.batch_size, shuffle=False,
            num_workers=training_config.num_workers, pin_memory=True
        )

        print("\nInitializing model...")
        model_init = PretrainedMultimodalRecommender(
            n_users=full_dataset_for_splitting.n_users,
            n_items=full_dataset_for_splitting.n_items,
            embedding_dim=model_config.embedding_dim,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            freeze_vision=model_config.freeze_vision,
            freeze_language=model_config.freeze_language,
            use_contrastive=model_config.use_contrastive,
            dropout_rate=model_config.dropout_rate
        ).to(device)

        total_params = sum(p.numel() for p in model_init.parameters())
        trainable_params = sum(p.numel() for p in model_init.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        trainer = Trainer(
            model=model_init, 
            device=device,
            checkpoint_dir=config.checkpoint_dir,
            use_contrastive=model_config.use_contrastive        
        )
        # Ensure loss weights from config are applied to the criterion in Trainer
        trainer.criterion.contrastive_weight = training_config.contrastive_weight
        trainer.criterion.bce_weight = training_config.bce_weight


        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        print("\nStarting training...")
        train_losses, val_losses = trainer.train(
            train_loader=train_loader, val_loader=val_loader,
            epochs=training_config.epochs, lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay, patience=training_config.patience,
            gradient_clip=training_config.gradient_clip
        )

        final_checkpoint_path = Path(config.checkpoint_dir) / 'final_model.pth'
        trainer.save_checkpoint('final_model.pth') # Saves with self.epoch and self.optimizer state
        print(f"\nSaved final model to {final_checkpoint_path}")

        encoders_dir = Path(config.checkpoint_dir) / 'encoders'
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_splitting.user_encoder, f)
        with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_splitting.item_encoder, f)
        print(f"Saved encoders to {encoders_dir}")

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


        config_save_path = Path(config.results_dir) / 'training_run_config.yaml'
        config.to_yaml(str(config_save_path))
        print(f"Saved effective configuration to {config_save_path}")
        # Log config file to wandb if enabled
        if args.use_wandb and wandb.run is not None:
            try:
                wandb.save(str(config_save_path)) # Save the config file used for this run
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