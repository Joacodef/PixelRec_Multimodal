# scripts/train.py - Fixed version with dynamic numerical feature handling
#!/usr/bin/env python
"""
Training script for the simplified multimodal recommender system with dynamic numerical features
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
from typing import List
import dataclasses
import wandb
import os
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config, TextAugmentationConfig
from src.data.dataset import MultimodalDataset
from src.models.multimodal import MultimodalRecommender
from src.training.trainer import Trainer
from src.data.processors import NumericalProcessor

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


def print_progress_header(step_num: int, title: str, total_steps: int = 13):
    """Print a standardized progress header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}/{total_steps}: {title.upper()}")
    print(f"{'='*60}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")


def print_progress_footer(start_time: float, additional_info: str = ""):
    """Print a standardized progress footer with timing"""
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.2f}s")
    if additional_info:
        print(f"  {additional_info}")
    print("-" * 60)


def print_system_info():
    """Print system information"""
    print("SYSTEM INFORMATION")
    print("-" * 30)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPU count: {os.cpu_count()}")


def validate_numerical_features(item_info_df: pd.DataFrame, config_numerical_cols: List[str]) -> List[str]:
    """
    Validate and filter numerical feature columns to ensure they exist in the data
    
    Args:
        item_info_df: DataFrame containing item information
        config_numerical_cols: List of numerical columns from config
        
    Returns:
        List of valid numerical columns that exist in the DataFrame
    """
    available_cols = list(item_info_df.columns)
    valid_cols = []
    missing_cols = []
    
    for col in config_numerical_cols:
        if col in available_cols:
            valid_cols.append(col)
        else:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"⚠️  Warning: The following numerical columns from config are missing in data:")
        for col in missing_cols:
            print(f"    - {col}")
        print(f"📊 Available columns in data: {available_cols}")
        print(f"✅ Using valid columns: {valid_cols}")
    
    if not valid_cols:
        print(f"❌ Error: No valid numerical columns found!")
        print(f"   Configured columns: {config_numerical_cols}")
        print(f"   Available columns: {available_cols}")
        raise ValueError("No valid numerical feature columns found in the data")
    
    return valid_cols


def fit_numerical_scaler(df, numerical_cols, method, scaler_path):
    """Fit numerical scaler with progress"""
    print(f"  → Fitting {method} scaler on {len(df)} samples...")
    processor = NumericalProcessor()
    processor.fit_scaler(df, numerical_cols, method)
    processor.save_scaler(scaler_path)
    print(f"  → Scaler saved to {scaler_path}")
    return processor.scaler


def load_numerical_scaler(scaler_path):
    """Load numerical scaler with progress"""
    print(f"  → Loading scaler from {scaler_path}")
    processor = NumericalProcessor()
    processor.load_scaler(scaler_path)
    return processor.scaler


def validate_data_integrity(interactions_df, item_info_df):
    """Validate data integrity and print statistics"""
    print("DATA INTEGRITY CHECK")
    print("-" * 30)
    
    # Check for missing values
    print(f"Interactions shape: {interactions_df.shape}")
    print(f"Item info shape: {item_info_df.shape}")
    
    # Check unique counts
    n_unique_users = interactions_df['user_id'].nunique()
    n_unique_items_interactions = interactions_df['item_id'].nunique()
    n_unique_items_info = item_info_df['item_id'].nunique()
    
    print(f"Unique users: {n_unique_users:,}")
    print(f"Unique items in interactions: {n_unique_items_interactions:,}")
    print(f"Unique items in item_info: {n_unique_items_info:,}")
    
    # Check overlap
    items_with_info = set(item_info_df['item_id'].astype(str))
    items_in_interactions = set(interactions_df['item_id'].astype(str))
    overlap = len(items_with_info & items_in_interactions)
    
    print(f"Item overlap: {overlap:,} ({100*overlap/n_unique_items_interactions:.1f}%)")
    
    if overlap < n_unique_items_interactions * 0.9:
        print("⚠️  Warning: Less than 90% of interaction items have item info")
    
    return {
        'n_users': n_unique_users,
        'n_items_interactions': n_unique_items_interactions,
        'n_items_info': n_unique_items_info,
        'overlap': overlap
    }


def create_data_loaders_with_progress(train_dataset, val_dataset, training_config):
    """Create data loaders with progress monitoring"""
    print("DATALOADER CONFIGURATION")
    print("-" * 30)
    
    # Optimize num_workers based on system
    optimal_workers = min(training_config.num_workers, os.cpu_count(), 8)
    if optimal_workers != training_config.num_workers:
        print(f"  → Adjusting workers from {training_config.num_workers} to {optimal_workers} (system optimal)")
    
    print(f"  → Batch size: {training_config.batch_size}")
    print(f"  → Workers: {optimal_workers}")
    print(f"  → Pin memory: True")
    print(f"  → Persistent workers: {optimal_workers > 0}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=True, 
        num_workers=optimal_workers, 
        pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False,  
        prefetch_factor=2  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=False,
        num_workers=optimal_workers, 
        pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False,
        prefetch_factor=2
    ) if len(val_dataset) > 0 else None
    
    print(f"  → Training batches: {len(train_loader)}")
    if val_loader:
        print(f"  → Validation batches: {len(val_loader)}")
    else:
        print("  → No validation data")
    
    return train_loader, val_loader


def main():
    """Main function for training the multimodal recommender system with dynamic numerical features."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train multimodal recommender with dynamic numerical features')
    parser.add_argument('--config', type=str, default='configs/simple_config.yaml', help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='MultimodalRecommender', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name for this training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Print header
    print("🚀 MULTIMODAL RECOMMENDER TRAINING (DYNAMIC NUMERICAL FEATURES)")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file: {args.config}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Print system info
    print_system_info()

    try:
        # STEP 1: Load configuration
        print_progress_header(1, "Loading Configuration")
        step_start = time.time()
        
        config = Config.from_yaml(args.config)
        data_config = config.data
        model_config = config.model
        training_config = config.training
        
        print(f"Configuration loaded successfully:")
        print(f"  → Vision model: {model_config.vision_model}")
        print(f"  → Language model: {model_config.language_model}")
        print(f"  → Embedding dim: {model_config.embedding_dim}")
        print(f"  → Batch size: {training_config.batch_size}")
        print(f"  → Learning rate: {training_config.learning_rate}")
        print(f"  → Epochs: {training_config.epochs}")
        print(f"  → Configured numerical features: {data_config.numerical_features_cols}")
        
        print_progress_footer(step_start)

        # STEP 2: Initialize Weights & Biases
        print_progress_header(2, "Initializing Weights & Biases")
        step_start = time.time()
        
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
                print("✓ Weights & Biases logging enabled")
                wandb.define_metric("epoch")
                wandb.define_metric("train/*", step_metric="epoch")
                wandb.define_metric("val/*", step_metric="epoch")
                wandb.define_metric("train/learning_rate", step_metric="epoch")
            except Exception as e:
                print(f"⚠️  Failed to initialize wandb: {e}")
                print("   Proceeding without wandb logging")
                args.use_wandb = False
        else:
            print("Weights & Biases logging disabled")
        
        print_progress_footer(step_start)

        # STEP 3: Setup device
        print_progress_header(3, "Setting Up Device")
        step_start = time.time()
        
        device = torch.device(args.device)
        print(f"Using device: {device}")
        
        if device.type == 'cuda':
            print(f"  → GPU: {torch.cuda.get_device_name(0)}")
            print(f"  → Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Clear cache
            torch.cuda.empty_cache()
            print("  → GPU cache cleared")
        
        print_progress_footer(step_start)

        # STEP 4: Load processed data and validate numerical features
        print_progress_header(4, "Loading and Validating Data")
        step_start = time.time()
        
        print(f"Loading data files:")
        print(f"  → Item info: {data_config.processed_item_info_path}")
        print(f"  → Interactions: {data_config.processed_interactions_path}")
        
        item_info_df_full = pd.read_csv(data_config.processed_item_info_path)
        interactions_df_full = pd.read_csv(data_config.processed_interactions_path)
        
        # CRITICAL FIX: Validate and update numerical features based on actual data
        print(f"\n📊 VALIDATING NUMERICAL FEATURES:")
        print(f"   Original config numerical features: {data_config.numerical_features_cols}")
        
        # Validate numerical features against actual data columns
        valid_numerical_features = validate_numerical_features(
            item_info_df_full, 
            data_config.numerical_features_cols
        )
        
        # Update the config with valid numerical features
        data_config.numerical_features_cols = valid_numerical_features
        print(f"   ✅ Final numerical features to use: {valid_numerical_features}")
        print(f"   📏 Number of numerical features: {len(valid_numerical_features)}")
        
        data_stats = validate_data_integrity(interactions_df_full, item_info_df_full)
        
        print_progress_footer(step_start, f"Loaded {len(item_info_df_full):,} items, {len(interactions_df_full):,} interactions")

        # STEP 5: Initialize feature cache
        print_progress_header(5, "Initializing Feature Cache")
        step_start = time.time()
        
        simple_cache_instance = None
        cache_config = data_config.cache_config
        
        if cache_config.enabled:
            # Auto-generate model-specific cache directory
            cache_name = f"{model_config.vision_model}_{model_config.language_model}"
            auto_cache_dir = f"cache/{cache_name}"
            
            # Use config cache_directory or auto-generated one
            effective_cache_dir = cache_config.cache_directory
            if cache_config.cache_directory == 'cache':
                # Default config path, use model-specific instead
                effective_cache_dir = auto_cache_dir
            
            print("Cache configuration:")
            print(f"  → Strategy: Memory-based caching")
            print(f"  → Max memory items: {cache_config.max_memory_items:,}")
            print(f"  → Cache directory: {effective_cache_dir}")
            print(f"  → Use disk: {cache_config.use_disk}")
            print(f"  → Model combination: {model_config.vision_model} + {model_config.language_model}")
            
            # Create cache directory
            cache_dir = Path(effective_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if cache exists and show stats
            existing_files = list(cache_dir.glob("*.pt"))
            if existing_files:
                total_size = sum(f.stat().st_size for f in existing_files) / (1024*1024)
                print(f"  → Found existing cache: {len(existing_files):,} files, {total_size:.1f} MB")
            else:
                print(f"  → Cache directory is empty - features will be computed during training")
            
            base_cache_directory_from_config = cache_config.cache_directory

            simple_cache_instance = SimpleFeatureCache(
                vision_model=model_config.vision_model,         
                language_model=model_config.language_model,     
                base_cache_dir=base_cache_directory_from_config, 
                max_memory_items=cache_config.max_memory_items,
                use_disk=cache_config.use_disk
            )

            simple_cache_instance.print_stats()
        else:
            print("Feature caching is disabled")
        
        print_progress_footer(step_start)

        # STEP 6: Handle numerical scaler with dynamic features
        print_progress_header(6, "Processing Numerical Scaler")
        step_start = time.time()
        
        numerical_scaler = None
        scaler_path_obj = Path(data_config.scaler_path)
        
        print(f"Numerical features (final): {valid_numerical_features}")
        print(f"Number of numerical features: {len(valid_numerical_features)}")
        print(f"Normalization method: {data_config.numerical_normalization_method}")
        
        if data_config.numerical_normalization_method in ['standardization', 'min_max']:
            if scaler_path_obj.exists():
                print("Loading existing scaler...")
                try:
                    numerical_scaler = load_numerical_scaler(scaler_path_obj)
                    print(f"  → Scaler loaded successfully")
                except Exception as e:
                    print(f"  → Error loading scaler: {e}")
                    print(f"  → Will fit new scaler...")
                    scaler_path_obj.parent.mkdir(parents=True, exist_ok=True)
                    numerical_scaler = fit_numerical_scaler(
                        item_info_df_full, 
                        valid_numerical_features,  # Use validated features
                        data_config.numerical_normalization_method, 
                        scaler_path_obj
                    )
            else:
                print("Fitting new scaler...")
                scaler_path_obj.parent.mkdir(parents=True, exist_ok=True)
                numerical_scaler = fit_numerical_scaler(
                    item_info_df_full, 
                    valid_numerical_features,  # Use validated features
                    data_config.numerical_normalization_method, 
                    scaler_path_obj
                )
        else:
            print("No scaling required (method: none or log1p)")
        
        print_progress_footer(step_start)

        # STEP 7: Determine image folder
        print_progress_header(7, "Configuring Image Processing")
        step_start = time.time()
        
        effective_image_folder = data_config.image_folder
        if (hasattr(data_config, 'offline_image_compression') and 
            data_config.offline_image_compression.enabled and 
            hasattr(data_config, 'processed_image_destination_folder') and 
            data_config.processed_image_destination_folder):
            effective_image_folder = data_config.processed_image_destination_folder
            print(f"Using processed images: {effective_image_folder}")
        else:
            print(f"Using original images: {effective_image_folder}")
        
        # Check if image folder exists
        if os.path.exists(effective_image_folder):
            image_count = len([f for f in os.listdir(effective_image_folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  → Found {image_count:,} image files")
        else:
            print(f"⚠️  Warning: Image folder does not exist: {effective_image_folder}")
        
        print_progress_footer(step_start)

        # STEP 8: Create dataset for encoder fitting with validated numerical features
        print_progress_header(8, "Creating Dataset for Encoder Fitting")
        step_start = time.time()
        
        print("Initializing full dataset for encoder fitting...")
        print("This step may take time as it initializes the vision and language models")
        print(f"Using {len(valid_numerical_features)} validated numerical features")
        
        full_dataset_for_encoders = MultimodalDataset(
            interactions_df=interactions_df_full,
            item_info_df=item_info_df_full,
            image_folder=effective_image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=False,
            negative_sampling_ratio=0,
            # Use validated numerical features
            cache_features=cache_config.enabled,
            cache_max_items=cache_config.max_memory_items,
            cache_dir=effective_cache_dir,  # Use the effective cache directory
            cache_to_disk=cache_config.use_disk,
            # Other parameters with validated numerical features
            numerical_feat_cols=valid_numerical_features,  # Use validated features
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False
        )
        
        print("Encoder fitting results:")
        print(f"  → Users: {full_dataset_for_encoders.n_users:,}")
        print(f"  → Items: {full_dataset_for_encoders.n_items:,}")
        print(f"  → Numerical features used: {len(valid_numerical_features)}")
        
        print_progress_footer(step_start)

        # STEP 9: Load pre-split data
        print_progress_header(9, "Loading Pre-Split Training Data")
        step_start = time.time()
        
        print(f"Loading split data:")
        print(f"  → Train: {data_config.train_data_path}")
        print(f"  → Val: {data_config.val_data_path}")
        
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

        print("Split data statistics:")
        print(f"  → Training interactions: {len(train_interactions_df):,}")
        print(f"  → Validation interactions: {len(val_interactions_df):,}")
        print(f"  → Items in splits: {len(item_info_df_for_datasets):,}")
        print(f"  → Train users: {train_interactions_df['user_id'].nunique():,}")
        print(f"  → Val users: {val_interactions_df['user_id'].nunique():,}")
        
        print_progress_footer(step_start)

        # STEP 10: Create training datasets with validated numerical features
        print_progress_header(10, "Creating Training Datasets")
        step_start = time.time()
        
        print("Creating training dataset with negative sampling...")
        print(f"Using {len(valid_numerical_features)} validated numerical features")
        
        train_dataset = MultimodalDataset(
            interactions_df=train_interactions_df,
            item_info_df=item_info_df_for_datasets,
            image_folder=effective_image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=data_config.text_augmentation,
            numerical_feat_cols=valid_numerical_features,  # Use validated features
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=True,
            # Simplified cache parameters with model-specific directory
            cache_features=cache_config.enabled,
            cache_max_items=cache_config.max_memory_items,
            cache_dir=effective_cache_dir,
            cache_to_disk=cache_config.use_disk
        )
        
        # Assign globally fitted encoders
        train_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        train_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        train_dataset.n_users = full_dataset_for_encoders.n_users
        train_dataset.n_items = full_dataset_for_encoders.n_items

        print("Creating validation dataset...")
        val_dataset = MultimodalDataset(
            interactions_df=val_interactions_df,
            item_info_df=item_info_df_for_datasets,
            image_folder=effective_image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True,
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=TextAugmentationConfig(enabled=False),  # No augmentation for validation
            numerical_feat_cols=valid_numerical_features,  # Use validated features
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False,
            # Simplified cache parameters with model-specific directory
            cache_features=cache_config.enabled,
            cache_max_items=cache_config.max_memory_items,
            cache_dir=effective_cache_dir,
            cache_to_disk=cache_config.use_disk
        )
        
        # Assign globally fitted encoders
        val_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        val_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        val_dataset.n_users = full_dataset_for_encoders.n_users
        val_dataset.n_items = full_dataset_for_encoders.n_items
        
        print("Dataset creation results:")
        print(f"  → Training samples: {len(train_dataset):,}")
        print(f"  → Validation samples: {len(val_dataset):,}")
        print(f"  → Negative sampling ratio: {data_config.negative_sampling_ratio}")
        print(f"  → Numerical features: {len(valid_numerical_features)}")
        
        print_progress_footer(step_start)

        # STEP 11: Create data loaders
        print_progress_header(11, "Creating Data Loaders")
        step_start = time.time()
        
        train_loader, val_loader = create_data_loaders_with_progress(
            train_dataset, val_dataset, training_config
        )
        
        print_progress_footer(step_start)

        # Save encoders before training
        print("Saving encoders...")
        encoders_dir = Path(config.checkpoint_dir) / 'encoders'
        encoders_dir.mkdir(parents=True, exist_ok=True)
        
        with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.user_encoder, f)
        with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.item_encoder, f)
        print(f"  → Encoders saved to {encoders_dir}")

        # STEP 12: Initialize model with correct number of numerical features
        print_progress_header(12, "Initializing Model", total_steps=13)
        step_start = time.time()
        
        num_numerical_features = len(valid_numerical_features)
        
        print("Model configuration:")
        print(f"  → Architecture: MultimodalRecommender")
        print(f"  → Users: {full_dataset_for_encoders.n_users:,}")
        print(f"  → Items: {full_dataset_for_encoders.n_items:,}")
        print(f"  → Numerical features: {num_numerical_features} (validated)")
        print(f"  → Feature names: {valid_numerical_features}")
        print(f"  → Embedding dim: {model_config.embedding_dim}")
        print(f"  → Vision model: {model_config.vision_model}")
        print(f"  → Language model: {model_config.language_model}")
        print(f"  → Use contrastive: {model_config.use_contrastive}")
        
        # Display checkpoint organization
        model_combo = f"{model_config.vision_model}_{model_config.language_model}"
        model_checkpoint_dir = Path(config.checkpoint_dir) / model_combo
        shared_encoders_dir = Path(config.checkpoint_dir) / 'encoders'
        
        print(f"\n📁 Checkpoint Organization:")
        print(f"  → Model checkpoints (.pth): {model_checkpoint_dir}")
        print(f"  → Shared encoders: {shared_encoders_dir}")
        
        print("\nInitializing model (this may take several minutes for model downloads)...")
        
        model_params = {
            'n_users': full_dataset_for_encoders.n_users,
            'n_items': full_dataset_for_encoders.n_items,
            'num_numerical_features': num_numerical_features,  # Use validated count
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

        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("Model statistics:")
        print(f"  → Total parameters: {total_params:,}")
        print(f"  → Trainable parameters: {trainable_params:,}")
        print(f"  → Frozen parameters: {total_params - trainable_params:,}")
        print(f"  → Model size: ~{total_params * 4 / 1e6:.1f} MB")
        
        print_progress_footer(step_start, f"Model ready with {trainable_params:,} trainable parameters")

        # STEP 13: Initialize trainer and start training
        print_progress_header(13, "Starting Training", total_steps=13)
        step_start = time.time()
        
        # Initialize trainer with model config for checkpoint organization
        trainer = Trainer(
            model=model, 
            device=device, 
            checkpoint_dir=config.checkpoint_dir, 
            use_contrastive=model_config.use_contrastive,
            model_config=model_config  # Pass model config for checkpoint paths
        )
        trainer.criterion.contrastive_weight = training_config.contrastive_weight
        trainer.criterion.bce_weight = training_config.bce_weight

        print("Training configuration:")
        print(f"  → Optimizer: {training_config.optimizer_type}")
        print(f"  → Learning rate: {training_config.learning_rate}")
        print(f"  → Weight decay: {training_config.weight_decay}")
        print(f"  → Batch size: {training_config.batch_size}")
        print(f"  → Epochs: {training_config.epochs}")
        print(f"  → Patience: {training_config.patience}")
        print(f"  → Gradient clip: {training_config.gradient_clip}")
        print(f"  → Contrastive weight: {training_config.contrastive_weight}")
        print(f"  → BCE weight: {training_config.bce_weight}")

        # Resume from checkpoint if specified
        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Save encoders before training (in shared directory)
        print("Saving encoders to shared directory...")
        encoders_dir = trainer.get_encoders_dir()
        
        with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.user_encoder, f)
        with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
            pickle.dump(full_dataset_for_encoders.item_encoder, f)
        print(f"  → Encoders saved to {encoders_dir}")

        print(f"\n🚀 Starting training...")
        print("=" * 60)

        # Save updated configuration with validated numerical features
        print("Saving updated configuration with validated numerical features...")
        updated_config_path = Path(config.results_dir) / 'training_run_config_validated.yaml'
        config.to_yaml(str(updated_config_path))
        print(f"✓ Updated configuration saved to {updated_config_path}")

        # Prepare training parameters
        training_params = {
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
        train_losses, val_losses = trainer.train(**training_params)

        # Plot and save training curves
        if train_losses and val_losses:
            plt.figure(figsize=(12, 8))
            
            # Main plot
            plt.subplot(2, 2, 1)
            plt.plot(train_losses, label='Train Loss', linewidth=2)
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Smoothed plot (if enough epochs)
            if len(train_losses) > 5:
                plt.subplot(2, 2, 2)
                # Simple moving average
                window = min(5, len(train_losses) // 3)
                train_smooth = pd.Series(train_losses).rolling(window=window).mean()
                val_smooth = pd.Series(val_losses).rolling(window=window).mean()
                
                plt.plot(train_smooth, label=f'Train Loss (MA-{window})', linewidth=2)
                plt.plot(val_smooth, label=f'Val Loss (MA-{window})', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Smoothed Training Curves')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Loss distribution
            plt.subplot(2, 2, 3)
            plt.hist(train_losses, bins=20, alpha=0.7, label='Train Loss', density=True)
            plt.hist(val_losses, bins=20, alpha=0.7, label='Val Loss', density=True)
            plt.xlabel('Loss Value')
            plt.ylabel('Density')
            plt.title('Loss Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Training summary stats
            plt.subplot(2, 2, 4)
            plt.axis('off')
            
            # Calculate statistics
            min_train_loss = min(train_losses)
            min_val_loss = min(val_losses)
            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]
            
            stats_text = f"""Training Summary

Epochs Completed: {len(train_losses)}
Training Time: {training_time/3600:.2f} hours

Final Losses:
  Train: {final_train_loss:.6f}
  Validation: {final_val_loss:.6f}

Best Losses:
  Train: {min_train_loss:.6f}
  Validation: {min_val_loss:.6f}

Model Config:
  Vision: {model_config.vision_model}
  Language: {model_config.language_model}
  Embedding Dim: {model_config.embedding_dim}
  Batch Size: {training_config.batch_size}
  Learning Rate: {training_config.learning_rate}

Data Stats:
  Users: {full_dataset_for_encoders.n_users:,}
  Items: {full_dataset_for_encoders.n_items:,}
  Train Samples: {len(train_dataset):,}
  Val Samples: {len(val_dataset):,}
  Numerical Features: {num_numerical_features}"""
            
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=8)
            
            plt.tight_layout()
            results_fig_dir = Path(config.results_dir) / 'figures'
            results_fig_dir.mkdir(parents=True, exist_ok=True)
            plot_path = results_fig_dir / 'training_curves.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Training curves saved to {plot_path}")
            
            if args.use_wandb and wandb.run is not None:
                try:
                    wandb.log({"training_validation_loss_curves": wandb.Image(str(plot_path))})
                    print("✓ Training curves logged to wandb")
                except Exception as e:
                    print(f"⚠️  Failed to log training curves to wandb: {e}")
        else:
            print("⚠️  No training curves to plot (training may have been skipped)")

        # Save training metadata with validated numerical features
        training_metadata = {
            'training_completed': True,
            'completion_time': datetime.now().isoformat(),
            'training_duration_hours': training_time / 3600,
            'epochs_completed': len(train_losses) if train_losses else 0,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_train_loss': min(train_losses) if train_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
            'model_config': dataclasses.asdict(model_config),
            'training_config': dataclasses.asdict(training_config),
            'data_stats': data_stats,
            'model_params': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'frozen_parameters': total_params - trainable_params
            },
            'device_info': {
                'device': str(device),
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            'numerical_features_validation': {
                'original_config_features': config.data.numerical_features_cols,
                'validated_features': valid_numerical_features,
                'num_features_used': num_numerical_features,
                'missing_features': [col for col in config.data.numerical_features_cols if col not in valid_numerical_features]
            }
        }
        
        metadata_path = Path(config.results_dir) / 'training_metadata.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2, default=str)
        print(f"✓ Training metadata saved to {metadata_path}")

        # Save the effective configuration with validated numerical features
        config_save_path = Path(config.results_dir) / 'training_run_config.yaml'
        config.to_yaml(str(config_save_path))
        print(f"✓ Configuration saved to {config_save_path}")
        
        if args.use_wandb and wandb.run is not None:
            try:
                wandb.save(str(config_save_path))
                wandb.save(str(metadata_path))
                wandb.save(str(updated_config_path))
                if 'plot_path' in locals():
                    wandb.save(str(plot_path))
                print("✓ Files saved to wandb")
            except Exception as e:
                print(f"⚠️  Failed to save files to wandb: {e}")

        # Print final summary
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 Training Summary:")
        print(f"   → Duration: {training_time/3600:.2f} hours")
        print(f"   → Epochs: {len(train_losses) if train_losses else 0}")
        print(f"   → Final train loss: {train_losses[-1]:.6f}" if train_losses else "   → No training loss recorded")
        print(f"   → Final val loss: {val_losses[-1]:.6f}" if val_losses else "   → No validation loss recorded")
        print(f"   → Best val loss: {min(val_losses):.6f}" if val_losses else "   → No validation loss recorded")
        print(f"📏 Numerical Features:")
        print(f"   → Features used: {num_numerical_features}")
        print(f"   → Feature names: {valid_numerical_features}")
        print(f"📁 Outputs saved to: {config.results_dir}")
        print(f"🤖 Model checkpoint: {model_checkpoint_dir}/best_model.pth")
        print(f"🔄 Encoders: {encoders_dir}")
        
        # Performance recommendations
        print(f"\n💡 Performance Notes:")
        if training_time > 0:
            samples_per_second = (len(train_dataset) * len(train_losses)) / training_time
            print(f"   → Training speed: {samples_per_second:.1f} samples/second")
            
            if samples_per_second < 100:
                print("   → Consider reducing image resolution or batch size for faster training")
            elif samples_per_second > 1000:
                print("   → Training speed is excellent!")
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"   → Peak GPU memory: {gpu_memory_used:.1f} GB")
            
            if gpu_memory_used > 10:
                print("   → High GPU memory usage - consider reducing batch size")
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saving current state...")
        
        # Save emergency checkpoint if trainer exists
        if 'trainer' in locals():
            try:
                trainer.save_checkpoint('interrupted_model.pth')
                print(f"✓ Emergency checkpoint saved to {config.checkpoint_dir}/interrupted_model.pth")
            except Exception as e:
                print(f"❌ Failed to save emergency checkpoint: {e}")
        
        # Save partial results if available
        if 'train_losses' in locals() and train_losses:
            try:
                partial_results = {
                    'interrupted': True,
                    'interruption_time': datetime.now().isoformat(),
                    'epochs_completed': len(train_losses),
                    'train_losses': train_losses,
                    'val_losses': val_losses if 'val_losses' in locals() else [],
                    'validated_numerical_features': valid_numerical_features if 'valid_numerical_features' in locals() else []
                }
                
                interrupted_path = Path(config.results_dir) / 'interrupted_training.json'
                interrupted_path.parent.mkdir(parents=True, exist_ok=True)
                
                import json
                with open(interrupted_path, 'w') as f:
                    json.dump(partial_results, f, indent=2)
                print(f"✓ Partial results saved to {interrupted_path}")
            except Exception as e:
                print(f"❌ Failed to save partial results: {e}")
        else:
            print("No training progress to save (training may not have started)")
        
        raise
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        try:
            error_info = {
                'error_occurred': True,
                'error_time': datetime.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'validated_numerical_features': valid_numerical_features if 'valid_numerical_features' in locals() else 'Not computed yet'
            }
            
            if 'config' in locals():
                error_path = Path(config.results_dir) / 'error_log.json'
                error_path.parent.mkdir(parents=True, exist_ok=True)
                
                import json
                with open(error_path, 'w') as f:
                    json.dump(error_info, f, indent=2)
                print(f"✓ Error log saved to {error_path}")
        except Exception as save_error:
            print(f"❌ Failed to save error log: {save_error}")
        
        raise
        
    finally:
        # Cleanup and finish wandb
        if args.use_wandb and wandb.run is not None:
            print("Finishing wandb run...")
            try:
                wandb.finish()
                print("✓ Wandb run finished")
            except Exception as e:
                print(f"⚠️  Warning during wandb cleanup: {e}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU cache cleared")
        
        print(f"\nSession ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()