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
from src.config import TextAugmentationConfig

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.training.trainer import Trainer


def fit_numerical_scaler(df: pd.DataFrame, numerical_cols: List[str], method: str, scaler_path: Path):
    """Fits a scaler on the numerical columns of the dataframe and saves it."""
    if not numerical_cols or method in ['none', 'log1p']: # log1p is applied per-element
        print(f"Scaler fitting skipped for method: {method} or no numerical columns.")
        return None

    data_to_scale = df[numerical_cols].fillna(0).values # Fill NaNs with 0 before scaling
    
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
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    data_config = config.data
    model_config = config.model
    training_config = config.training
    print(f"Loaded configuration from {args.config}")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("\nLoading processed data...")
    # Use processed data paths from config
    item_info_df = pd.read_csv(data_config.processed_item_info_path)
    interactions_df = pd.read_csv(data_config.processed_interactions_path)

    if data_config.sample_size:
        print(f"Sampling {data_config.sample_size} interactions from processed data...")
        interactions_df = interactions_df.sample(
            n=min(data_config.sample_size, len(interactions_df)),
            random_state=42
        ).reset_index(drop=True)
        # Filter item_info_df to only include items present in the sampled interactions
        sampled_item_ids = set(interactions_df['item_id'].unique())
        item_info_df = item_info_df[item_info_df['item_id'].isin(sampled_item_ids)].reset_index(drop=True)


    # Fit or load numerical scaler
    numerical_scaler = None
    scaler_path_obj = Path(data_config.scaler_path)
    if data_config.numerical_normalization_method in ['standardization', 'min_max']:
        if scaler_path_obj.exists():
            numerical_scaler = load_numerical_scaler(scaler_path_obj)
        else:
            # Fit scaler only on the training portion of item_info features
            # This requires knowing which items will be in the training set.
            # For simplicity, if scaler is not found, we fit on all available (processed, sampled) item_info.
            # A more rigorous approach would split interactions first, then fit on train item features.
            print(f"Scaler not found at {scaler_path_obj}. Fitting a new one based on current item_info_df...")
            # Ensure item_info_df has the numerical columns
            if not all(col in item_info_df.columns for col in data_config.numerical_features_cols):
                print(f"Error: Not all numerical_features_cols ({data_config.numerical_features_cols}) found in item_info_df columns ({item_info_df.columns}). Cannot fit scaler.")
                sys.exit(1)

            numerical_scaler = fit_numerical_scaler(
                item_info_df, # Use item_info_df that corresponds to the interactions (potentially sampled)
                data_config.numerical_features_cols,
                data_config.numerical_normalization_method,
                scaler_path_obj
            )
            if numerical_scaler is None and data_config.numerical_normalization_method not in ['none', 'log1p']:
                 print(f"Warning: Scaler fitting failed for method {data_config.numerical_normalization_method}. Numerical features might not be scaled as expected.")


    print("\nCreating full dataset instance for splitting...")
    # This dataset instance is primarily for setting up encoders and getting n_users/n_items
    # and for splitting. Negative sampling will be done by this instance.
    full_dataset_for_splitting = MultimodalDataset(
        interactions_df=interactions_df,
        item_info_df=item_info_df,
        image_folder=data_config.image_folder,
        vision_model_name=model_config.vision_model,
        language_model_name=model_config.language_model,
        create_negative_samples=True, # Let this instance handle negative sampling for the whole dataset
        negative_sampling_ratio=data_config.negative_sampling_ratio,
        text_augmentation_config=None, # Augmentation will be set per split-dataset
        numerical_feat_cols=data_config.numerical_features_cols,
        numerical_normalization_method=data_config.numerical_normalization_method,
        numerical_scaler=numerical_scaler, # Pass the fitted/loaded scaler
        is_train_mode=False # Not for training itself, but for structure
    )

    print(f"Full dataset size for splitting: {len(full_dataset_for_splitting)}")
    print(f"Number of users: {full_dataset_for_splitting.n_users}")
    print(f"Number of items: {full_dataset_for_splitting.n_items}")

    # Split dataset based on the already processed `all_samples` from full_dataset_for_splitting
    train_size = int(data_config.train_val_split * len(full_dataset_for_splitting.all_samples))
    val_size = len(full_dataset_for_splitting.all_samples) - train_size
    
    # Ensure indices are reset if we manually slice `all_samples`
    shuffled_samples = full_dataset_for_splitting.all_samples.sample(frac=1, random_state=42).reset_index(drop=True)
    train_samples_df = shuffled_samples.iloc[:train_size]
    val_samples_df = shuffled_samples.iloc[train_size:]

    print(f"Train samples: {len(train_samples_df)}")
    print(f"Validation samples: {len(val_samples_df)}")

    # Create train_dataset with augmentation enabled
    train_dataset_instance = MultimodalDataset(
        interactions_df=train_samples_df, # Pass pre-sampled and labeled train data
        item_info_df=item_info_df, # Full item info for lookups
        image_folder=data_config.image_folder,
        vision_model_name=model_config.vision_model,
        language_model_name=model_config.language_model,
        create_negative_samples=False, # Samples are already created and labeled
        text_augmentation_config=data_config.text_augmentation,
        numerical_feat_cols=data_config.numerical_features_cols,
        numerical_normalization_method=data_config.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        is_train_mode=True
    )
    # Share encoders from the initially created dataset
    train_dataset_instance.user_encoder = full_dataset_for_splitting.user_encoder
    train_dataset_instance.item_encoder = full_dataset_for_splitting.item_encoder
    train_dataset_instance.n_users = full_dataset_for_splitting.n_users
    train_dataset_instance.n_items = full_dataset_for_splitting.n_items
    train_dataset_instance.all_samples = train_samples_df # Crucial: set the data subset


    # Create val_dataset with augmentation disabled
    val_dataset_instance = MultimodalDataset(
        interactions_df=val_samples_df, # Pass pre-sampled and labeled val data
        item_info_df=item_info_df,
        image_folder=data_config.image_folder,
        vision_model_name=model_config.vision_model,
        language_model_name=model_config.language_model,
        create_negative_samples=False,
        text_augmentation_config=TextAugmentationConfig(enabled=False), # No augmentation for validation
        numerical_feat_cols=data_config.numerical_features_cols,
        numerical_normalization_method=data_config.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        is_train_mode=False
    )
    val_dataset_instance.user_encoder = full_dataset_for_splitting.user_encoder
    val_dataset_instance.item_encoder = full_dataset_for_splitting.item_encoder
    val_dataset_instance.n_users = full_dataset_for_splitting.n_users
    val_dataset_instance.n_items = full_dataset_for_splitting.n_items
    val_dataset_instance.all_samples = val_samples_df # Crucial: set the data subset


    train_loader = DataLoader(
        train_dataset_instance,
        batch_size=training_config.batch_size,
        shuffle=True, # Shuffle again, even if samples_df was shuffled, good practice
        num_workers=training_config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset_instance,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True
    )

    print("\nInitializing model...")
    # Use n_users and n_items from the dataset instance that fitted the encoders
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
        log_dir=config.log_dir,
        use_contrastive=model_config.use_contrastive
    )
    trainer.criterion.contrastive_weight = training_config.contrastive_weight # Ensure loss weights are set
    trainer.criterion.bce_weight = training_config.bce_weight


    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

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

    final_checkpoint_path = Path(config.checkpoint_dir) / 'final_model.pth'
    trainer.save_checkpoint('final_model.pth')
    print(f"\nSaved final model to {final_checkpoint_path}")

    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    encoders_dir.mkdir(parents=True, exist_ok=True)

    with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
        pickle.dump(full_dataset_for_splitting.user_encoder, f) # Save encoder from the full dataset
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
    plt.savefig(results_fig_dir / 'training_curves.png')
    print(f"\nSaved training curves to {results_fig_dir / 'training_curves.png'}")

    config_save_path = Path(config.results_dir) / 'training_run_config.yaml'
    config.to_yaml(str(config_save_path))
    print(f"Saved effective configuration to {config_save_path}")

    print("\nTraining completed!")


if __name__ == '__main__':
    main()