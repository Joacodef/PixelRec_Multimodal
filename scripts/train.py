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
from src.models.multimodal import PretrainedMultimodalRecommender, EnhancedMultimodalRecommender
from src.training.trainer import Trainer
from src.data.splitting import DataSplitter
from src.data.image_cache import SharedImageCache


def fit_numerical_scaler(df: pd.DataFrame, numerical_cols: List[str], method: str, scaler_path: Path):
    if not numerical_cols or method in ['none', 'log1p']:
        print(f"Scaler fitting skipped for method: {method} or no numerical columns.")
        return None
    data_to_scale = df[numerical_cols].fillna(0).values
    if method == 'standardization': scaler = StandardScaler()
    elif method == 'min_max': scaler = MinMaxScaler()
    else:
        print(f"Unknown scaling method for fitting: {method}. No scaler fitted.")
        return None
    print(f"Fitting {method} scaler on {len(data_to_scale)} samples...")
    scaler.fit(data_to_scale)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    return scaler

def load_numerical_scaler(scaler_path: Path):
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        print(f"Scaler loaded from {scaler_path}")
        return scaler
    print(f"Warning: Scaler not found at {scaler_path}. Proceeding without pre-loaded scaler.")
    return None

def main():
    parser = argparse.ArgumentParser(description='Train multimodal recommender')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to configuration file')
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

    # Pass the cache_processed_images flag from config to the Dataset
    cache_images_flag = getattr(data_config, 'cache_processed_images', False)

    if args.use_wandb:
        try:
            config_dict_for_wandb = dataclasses.asdict(config)
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=config_dict_for_wandb, reinit=True)
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
        # LOAD DATA FIRST - BEFORE CREATING IMAGE CACHE
        print("\nLoading processed data...")
        item_info_df = pd.read_csv(data_config.processed_item_info_path)
        interactions_df = pd.read_csv(data_config.processed_interactions_path)

        if data_config.sample_size:
            print(f"Sampling {data_config.sample_size} interactions from processed data...")
            interactions_df = interactions_df.sample(n=min(data_config.sample_size, len(interactions_df)), random_state=data_config.splitting.random_state).reset_index(drop=True)
            sampled_item_ids = set(interactions_df['item_id'].unique())
            item_info_df = item_info_df[item_info_df['item_id'].isin(sampled_item_ids)].reset_index(drop=True)

        # NOW CREATE IMAGE CACHE AFTER DATA IS LOADED
        shared_image_cache = None
        if cache_images_flag:
            cache_file_path = Path(config.checkpoint_dir) / 'image_cache.pkl'
            shared_image_cache = SharedImageCache(cache_path=cache_file_path)
            
            # Try to load existing cache
            shared_image_cache.load_from_disk()
            
            # If cache is empty or force precompute, precompute all images
            if not shared_image_cache.cache:
                print("\nPrecomputing all images for cache...")
                # Get all unique item IDs from the dataset
                all_item_ids = item_info_df['item_id'].unique().tolist()
                
                # Create a temporary dataset just to get the image processor
                temp_dataset = MultimodalDataset(
                    interactions_df=pd.DataFrame({'user_id': [], 'item_id': []}),
                    item_info_df=item_info_df,
                    image_folder=effective_image_folder,
                    vision_model_name=model_config.vision_model,
                    language_model_name=model_config.language_model,
                    create_negative_samples=False,
                    cache_processed_images=False  # Don't use cache for this temp dataset
                )
                
                # Precompute all images
                shared_image_cache.precompute_all_images(
                    item_ids=all_item_ids,
                    image_folder=effective_image_folder,
                    image_processor=temp_dataset.image_processor,
                    force_recompute=False
                )
                
                # Save the cache
                shared_image_cache.save_to_disk()
                print(f"Image cache saved with {len(shared_image_cache.cache)} images")

        # CONTINUE WITH NUMERICAL SCALER AND REST OF THE TRAINING
        numerical_scaler = None
        scaler_path_obj = Path(data_config.scaler_path)
        if data_config.numerical_normalization_method in ['standardization', 'min_max']:
            if scaler_path_obj.exists():
                numerical_scaler = load_numerical_scaler(scaler_path_obj)
            else:
                print(f"Scaler not found at {scaler_path_obj}. Fitting a new one...")
                numerical_scaler = fit_numerical_scaler(item_info_df, data_config.numerical_features_cols, data_config.numerical_normalization_method, scaler_path_obj)

        print(f"\nSplitting data using strategy: {data_config.splitting.strategy}")
        total_interactions = len(interactions_df)
        unique_users = interactions_df['user_id'].nunique()
        print(f"Dataset info: {total_interactions} interactions, {unique_users} unique users")
        splitter = DataSplitter(random_state=data_config.splitting.random_state)

        if total_interactions < 5000 or unique_users < 100:
            print("Small dataset detected. Using leave_one_out strategy.")
            loo_strategy = data_config.splitting.leave_one_out_strategy if hasattr(data_config.splitting, 'leave_one_out_strategy') else 'random'
            train_interactions_df, val_interactions_df = splitter.leave_one_out_split(interactions_df, strategy=loo_strategy)
        else:
            strategy_map = {
                'stratified': splitter.stratified_split,
                'user': splitter.user_based_split,
                'item': splitter.item_based_split,
                'temporal': splitter.temporal_split,
                'leave_one_out': splitter.leave_one_out_split,
                'simple_random': splitter.simple_random_split
            }
            split_func = strategy_map.get(data_config.splitting.strategy)
            if not split_func:
                raise ValueError(f"Unknown splitting strategy: {data_config.splitting.strategy}")

            split_params = {'interactions_df': interactions_df}
            if data_config.splitting.strategy in ['stratified', 'user', 'simple_random']:
                split_params['train_ratio'] = data_config.splitting.train_ratio
            if data_config.splitting.strategy in ['stratified', 'user']:
                split_params['min_interactions_per_user'] = data_config.splitting.min_interactions_per_user
            if data_config.splitting.strategy == 'item':
                split_params['train_ratio'] = data_config.splitting.train_ratio
                split_params['min_interactions_per_item'] = data_config.splitting.min_interactions_per_item
            if data_config.splitting.strategy == 'temporal':
                if not data_config.splitting.timestamp_col:
                    raise ValueError("timestamp_col required for temporal split.")
                split_params['timestamp_col'] = data_config.splitting.timestamp_col
                split_params['train_ratio'] = data_config.splitting.train_ratio
            if data_config.splitting.strategy == 'leave_one_out':
                split_params['strategy'] = data_config.splitting.leave_one_out_strategy
            
            train_interactions_df, val_interactions_df = split_func(**split_params)

        split_stats = splitter.get_split_statistics(train_interactions_df, val_interactions_df)
        print(f"\nSplit statistics:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")
        print(f"\nTraining interactions: {len(train_interactions_df)}")
        print(f"Validation interactions: {len(val_interactions_df)}")

        print("\nCreating dataset instances...")
        # Create datasets WITH shared_image_cache
        full_dataset_for_encoders = MultimodalDataset(
            interactions_df=interactions_df,
            item_info_df=item_info_df,
            image_folder=effective_image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=False,
            negative_sampling_ratio=0,
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False,
            cache_processed_images=cache_images_flag,
            shared_image_cache=shared_image_cache  # Pass shared cache
        )
        
        print(f"Fitted encoders on full dataset:")
        print(f"  Number of users: {full_dataset_for_encoders.n_users}")
        print(f"  Number of items: {full_dataset_for_encoders.n_items}")

        # Create actual training dataset with shared cache
        train_dataset = MultimodalDataset(
            interactions_df=train_interactions_df,
            item_info_df=item_info_df,
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
            cache_processed_images=cache_images_flag,
            shared_image_cache=shared_image_cache  # Pass shared cache
        )
        
        # Assign globally fitted encoders and counts
        train_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        train_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        train_dataset.n_users = full_dataset_for_encoders.n_users
        train_dataset.n_items = full_dataset_for_encoders.n_items
        train_dataset.finalize_setup()

        # Create validation dataset with shared cache
        val_dataset = MultimodalDataset(
            interactions_df=val_interactions_df,
            item_info_df=item_info_df,
            image_folder=effective_image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=TextAugmentationConfig(enabled=False),
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False,
            cache_processed_images=cache_images_flag,
            shared_image_cache=shared_image_cache  # Pass shared cache
        )
        
        # Assign globally fitted encoders and counts
        val_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        val_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        val_dataset.n_users = full_dataset_for_encoders.n_users
        val_dataset.n_items = full_dataset_for_encoders.n_items
        val_dataset.finalize_setup()
        
        print(f"\nDataset sizes after final setup:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

        # Continue with the rest of your training code...
        # (DataLoader creation, model initialization, training, etc.)

    finally:
        if args.use_wandb and wandb.run is not None:
            print("Finishing wandb run...")
            wandb.finish()

if __name__ == '__main__':
    main()