#!/usr/bin/env python3
"""
Training script for the multimodal recommender system.

This script provides both a command-line interface for training models and
a programmatic interface for hyperparameter optimization. The main() function
preserves backward compatibility while run_training() enables integration
with tools like Optuna.
"""

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, MODEL_CONFIGS
from src.data.simple_cache import SimpleFeatureCache
from src.data.dataset import MultimodalDataset
from src.models.multimodal import MultimodalRecommender
from src.training.trainer import Trainer

# Suppress specific warnings from transformers library
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def print_progress_header(step: int, title: str, total_steps: int = 13):
    """
    Prints a formatted progress header for tracking the training pipeline steps.

    Args:
        step: The current step number.
        title: A descriptive title for the current step.
        total_steps: The total number of steps in the pipeline.
    """
    print(f"\n{'='*60}")
    print(f"STEP {step}/{total_steps}: {title}")
    print(f"{'='*60}")


def print_progress_footer(start_time: float):
    """
    Prints a footer with the elapsed time for a step.

    Args:
        start_time: The timestamp when the step started.
    """
    elapsed = time.time() - start_time
    print(f"✓ Completed in {elapsed:.2f}s")


def print_system_info():
    """Displays system information relevant to the training environment."""
    print("\nSYSTEM INFO")
    print("-" * 30)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU cores: {os.cpu_count()}")
    print("-" * 30)


def create_dataloaders(
    train_dataset: 'Dataset',
    val_dataset: 'Dataset',
    training_config: Any
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Creates optimized DataLoader instances for training and validation.

    This function configures DataLoaders with optimal settings based on system
    capabilities and the provided training configuration.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        training_config (TrainingConfig): The configuration object for training parameters.

    Returns:
        Tuple[DataLoader, Optional[DataLoader]]: A tuple containing the training
        DataLoader and the validation DataLoader (or None if no validation data).
    """
    print("DATALOADER CONFIGURATION")
    print("-" * 30)
    
    # Determine the optimal number of workers based on system capabilities.
    optimal_workers = min(training_config.num_workers, os.cpu_count(), 8)
    if optimal_workers != training_config.num_workers:
        print(f"Adjusting workers from {training_config.num_workers} to {optimal_workers} (system optimal)")
    
    print(f"Batch size: {training_config.batch_size}")
    print(f"Workers: {optimal_workers}")
    print(f"Pin memory: True")
    
    loader_args = {
        'batch_size': training_config.batch_size,
        'num_workers': optimal_workers,
        'pin_memory': True,
    }

    if optimal_workers > 0:
        loader_args['persistent_workers'] = True
        loader_args['prefetch_factor'] = 2
        print(f"Persistent workers: True")
        print(f"Prefetch factor: 2")
    else:
        print(f"Persistent workers: False")
        print(f"Prefetch factor: None")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_args
        )
    
    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    else:
        print("No validation data")
    
    return train_loader, val_loader


def run_training(config: Config, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute the training pipeline and return results.
    
    This function encapsulates the entire training logic, making it reusable
    for both standalone training and hyperparameter optimization.
    
    Args:
        config: Configuration object with all settings
        args: Command line arguments namespace
        
    Returns:
        Dictionary containing training results:
        - 'best_val_loss': Best validation loss achieved
        - 'final_val_loss': Final validation loss
        - 'best_train_loss': Best training loss achieved  
        - 'final_train_loss': Final training loss
        - 'epochs_completed': Number of epochs completed
        - 'training_time': Total training time in seconds
        - 'model_path': Path to best model checkpoint
        - 'train_losses': List of training losses per epoch
        - 'val_losses': List of validation losses per epoch
        - 'metadata': Full training metadata dictionary
    """
    # Extract sub-configurations
    data_config = config.data
    model_config = config.model
    training_config = config.training
    
    # Store original numerical features before validation
    original_numerical_features_from_config = data_config.numerical_features_cols.copy()
    
    # STEP 3: Initialize Weights & Biases (if enabled)
    print_progress_header(3, "Initializing Weights & Biases", total_steps=13)
    step_start = time.time()
    
    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name
            if not run_name:
                models_used = f"{model_config.vision_model}_{model_config.language_model}"
                dataset_name = Path(data_config.train_data_path).parent.name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                run_name = f"{models_used}_{dataset_name}_{timestamp}"
            
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    'model_config': dataclasses.asdict(model_config),
                    'training_config': dataclasses.asdict(training_config),
                    'data_config': dataclasses.asdict(data_config)
                }
            )
            print(f"W&B run initialized: {wandb.run.name}")
        except ImportError:
            print("Warning: wandb not installed. Proceeding without W&B logging.")
            args.use_wandb = False
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            args.use_wandb = False
    else:
        print("W&B logging disabled")
    
    print_progress_footer(step_start)
    
    # STEP 4: Set up device
    print_progress_header(4, "Setting up Device", total_steps=13)
    step_start = time.time()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print_progress_footer(step_start)
    
    # STEP 5: Load data
    print_progress_header(5, "Loading Data", total_steps=13)
    step_start = time.time()
    
    print(f"Loading training data from: {data_config.train_data_path}")
    train_data = pd.read_csv(data_config.train_data_path)
    print(f"Training interactions: {len(train_data):,}")
    
    print(f"Loading validation data from: {data_config.val_data_path}")
    val_data = pd.read_csv(data_config.val_data_path)
    print(f"Validation interactions: {len(val_data):,}")
    
    print(f"Loading item information from: {data_config.processed_item_info_path}")
    item_info = pd.read_csv(data_config.processed_item_info_path)
    print(f"Total items: {len(item_info):,}")
    
    print_progress_footer(step_start)
    
    # STEP 6: Validate numerical features
    print_progress_header(6, "Validating Numerical Features", total_steps=13)
    step_start = time.time()
    
    print(f"Configured numerical features: {data_config.numerical_features_cols}")
    valid_numerical_features = [col for col in data_config.numerical_features_cols if col in item_info.columns]
    missing_features = [col for col in data_config.numerical_features_cols if col not in item_info.columns]
    
    if missing_features:
        print(f"⚠️  Warning: The following features are missing from item_info: {missing_features}")
        print(f"   Continuing with available features: {valid_numerical_features}")
    else:
        print(f"✓ All configured numerical features are present")
    
    data_config.numerical_features_cols = valid_numerical_features
    num_numerical_features = len(valid_numerical_features) if valid_numerical_features else 0
    print(f"Number of numerical features to use: {num_numerical_features}")
    
    print_progress_footer(step_start)
    
    # STEP 7: Initialize cache
    print_progress_header(7, "Initializing Feature Cache", total_steps=13)
    step_start = time.time()
    
    cache_key = f"{model_config.vision_model}_{model_config.language_model}"
    
    if data_config.cache_config.enabled:
        if data_config.cache_config.cache_directory:
            cache_base_dir = Path(data_config.cache_config.cache_directory)
        else:
            cache_base_dir = Path(config.checkpoint_dir).parent / 'cache'
        
        cache_dir = cache_base_dir / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Cache enabled: {cache_dir}")
        print(f"Max memory items: {data_config.cache_config.max_memory_items}")
        print(f"Persist to disk: {data_config.cache_config.use_disk}")
    else:
        cache_dir = None
        print("Feature caching disabled")
    
    cache = SimpleFeatureCache(
        enabled=data_config.cache_config.enabled,
        cache_dir=str(cache_dir) if cache_dir else None,
        max_memory_items=data_config.cache_config.max_memory_items,
        persist_to_disk=data_config.cache_config.use_disk
    )
    
    if cache.enabled and cache_dir:
        cache.print_stats()
    
    print_progress_footer(step_start)
    
    # STEP 8: Fit or load scaler
    print_progress_header(8, "Preparing Numerical Scaler", total_steps=13)
    step_start = time.time()
    
    all_interactions = pd.read_csv(data_config.processed_interactions_path)
    
    full_dataset_for_encoders = MultimodalDataset(
        interactions_df=all_interactions,
        item_info_df=item_info,
        image_folder=data_config.processed_image_destination_folder or data_config.image_folder,
        model_config=model_config,
        device=device,
        numerical_scaler=None,
        numerical_feature_columns=valid_numerical_features,
        cache=None,
        is_training=False,
        augment_text=False,
        augment_image=False
    )
    
    scaler_path = Path(data_config.scaler_path)
    if scaler_path.exists():
        print(f"Loading existing scaler from: {scaler_path}")
        full_dataset_for_encoders.load_scaler(str(scaler_path))
    else:
        print(f"Fitting new scaler using method: {data_config.numerical_normalization_method}")
        full_dataset_for_encoders.fit_numerical_scaler(method=data_config.numerical_normalization_method)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        full_dataset_for_encoders.save_scaler(str(scaler_path))
        print(f"Scaler saved to: {scaler_path}")
    
    print_progress_footer(step_start)
    
    # STEP 9: Create datasets
    print_progress_header(9, "Creating Datasets", total_steps=13)
    step_start = time.time()
    
    print("Creating training dataset...")
    train_dataset = MultimodalDataset(
        interactions_df=train_data,
        item_info_df=item_info,
        image_folder=data_config.processed_image_destination_folder or data_config.image_folder,
        model_config=model_config,
        device=device,
        encoder_mappings={
            'user_encoder': full_dataset_for_encoders.user_encoder,
            'item_encoder': full_dataset_for_encoders.item_encoder
        },
        numerical_scaler=full_dataset_for_encoders.numerical_scaler,
        numerical_feature_columns=valid_numerical_features,
        text_column='item_name',
        categorical_feature_columns=data_config.categorical_features_cols,
        negative_sampling_ratio=data_config.negative_sampling_ratio,
        cache=cache,
        is_training=True,
        augment_text=data_config.text_augmentation.enabled,
        augment_image=data_config.image_augmentation.enabled,
        augmentation_config={
            'text': data_config.text_augmentation,
            'image': data_config.image_augmentation
        }
    )
    
    print("Creating validation dataset...")
    val_dataset = MultimodalDataset(
        interactions_df=val_data,
        item_info_df=item_info,
        image_folder=data_config.processed_image_destination_folder or data_config.image_folder,
        model_config=model_config,
        device=device,
        encoder_mappings={
            'user_encoder': full_dataset_for_encoders.user_encoder,
            'item_encoder': full_dataset_for_encoders.item_encoder
        },
        numerical_scaler=full_dataset_for_encoders.numerical_scaler,
        numerical_feature_columns=valid_numerical_features,
        text_column='item_name',
        categorical_feature_columns=data_config.categorical_features_cols,
        negative_sampling_ratio=data_config.negative_sampling_ratio,
        cache=cache,
        is_training=False,
        augment_text=False,
        augment_image=False
    )
    
    data_stats = {
        'train_interactions': len(train_data),
        'val_interactions': len(val_data),
        'total_users': full_dataset_for_encoders.n_users,
        'total_items': full_dataset_for_encoders.n_items,
        'total_tags': full_dataset_for_encoders.n_tags,
        'numerical_features': num_numerical_features
    }
    
    print(f"\nDataset statistics:")
    for key, value in data_stats.items():
        print(f"  {key}: {value:,}")
    
    print_progress_footer(step_start)
    
    # STEP 10: Create dataloaders
    print_progress_header(10, "Creating DataLoaders", total_steps=13)
    step_start = time.time()
    
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, training_config)
    
    print_progress_footer(step_start)
    
    # STEP 11: Initialize model
    print_progress_header(11, "Initializing Model", total_steps=13)
    step_start = time.time()
    
    print(f"Creating MultimodalRecommender with:")
    print(f"  Vision model: {model_config.vision_model}")
    print(f"  Language model: {model_config.language_model}")
    print(f"  Embedding dim: {model_config.embedding_dim}")
    print(f"  Users: {full_dataset_for_encoders.n_users:,}")
    print(f"  Items: {full_dataset_for_encoders.n_items:,}")
    print(f"  Tags: {full_dataset_for_encoders.n_tags:,}")
    print(f"  Numerical features: {num_numerical_features}")
    
    model = MultimodalRecommender(
        config=model_config,
        n_users=full_dataset_for_encoders.n_users,
        n_items=full_dataset_for_encoders.n_items,
        n_tags=full_dataset_for_encoders.n_tags,
        num_numerical_features=num_numerical_features
    ).to(device)
    
    print_progress_footer(step_start)
    
    # STEP 12: Initialize trainer
    print_progress_header(12, "Initializing Trainer", total_steps=13)
    step_start = time.time()
    
    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
        use_contrastive=model_config.use_contrastive,
        model_config=model_config
    )
    trainer.criterion.contrastive_weight = training_config.contrastive_weight
    trainer.criterion.bce_weight = training_config.bce_weight
    
    print("Training configuration:")
    print(f"Optimizer: {training_config.optimizer_type}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Weight decay: {training_config.weight_decay}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Epochs: {training_config.epochs}")
    print(f"Patience: {training_config.patience}")
    print(f"Gradient clip: {training_config.gradient_clip}")
    print(f"Contrastive weight: {training_config.contrastive_weight}")
    print(f"BCE weight: {training_config.bce_weight}")
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Save encoders
    print("Saving encoders to shared directory...")
    encoders_dir = trainer.get_encoders_dir()
    
    with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
        pickle.dump(full_dataset_for_encoders.user_encoder, f)
    with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
        pickle.dump(full_dataset_for_encoders.item_encoder, f)
    print(f"Encoders saved to {encoders_dir}")
    
    print(f"\nStarting training...")
    print("=" * 60)
    
    # Save configuration
    print("Saving configuration...")
    updated_config_path = Path(config.results_dir) / 'training_run_config_validated.yaml'
    config.to_yaml(str(updated_config_path))
    print(f"Updated configuration saved to {updated_config_path}")
    
    print_progress_footer(step_start)
    
    # STEP 13: Train
    print_progress_header(13, "Starting Training", total_steps=13)
    step_start = time.time()
    
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
    
    training_start_time = time.time()
    train_losses, val_losses = trainer.train(**training_params)
    training_time = time.time() - training_start_time
    
    # Prepare results
    results = {
        'best_val_loss': min(val_losses) if val_losses else float('inf'),
        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
        'best_train_loss': min(train_losses) if train_losses else float('inf'),
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'epochs_completed': len(train_losses) if train_losses else 0,
        'training_time': training_time,
        'model_path': str(trainer.get_model_checkpoint_path('best_model.pth')),
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # Save metadata
    training_metadata = {
        'training_completed': True,
        'completion_time': datetime.now().isoformat(),
        'training_duration_hours': training_time / 3600,
        'epochs_completed': results['epochs_completed'],
        'final_train_loss': results['final_train_loss'],
        'final_val_loss': results['final_val_loss'],
        'best_train_loss': results['best_train_loss'],
        'best_val_loss': results['best_val_loss'],
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
            'original_config_features': original_numerical_features_from_config,
            'validated_features': valid_numerical_features,
            'num_features_used': num_numerical_features,
            'missing_features': missing_features
        }
    }
    
    metadata_path = Path(config.results_dir) / 'training_metadata.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2, default=str)
    print(f"Training metadata saved to {metadata_path}")
    
    # Save final configuration
    config_save_path = Path(config.results_dir) / 'training_run_config.yaml'
    config.to_yaml(str(config_save_path))
    print(f"Configuration saved to {config_save_path}")
    
    if args.use_wandb and 'wandb' in sys.modules:
        try:
            wandb = sys.modules['wandb']
            wandb.save(str(config_save_path))
            wandb.save(str(metadata_path))
            wandb.save(str(updated_config_path))
            print("Files saved to wandb")
        except Exception as e:
            print(f"Failed to save files to wandb: {e}")
    
    results['metadata'] = training_metadata
    
    print_progress_footer(step_start)
    
    return results


def main(cli_args: Optional[List[str]] = None):
    """
    Main function to execute the full training pipeline for the multimodal recommender.

    The pipeline consists of the following steps:
    1.  Parse command-line arguments.
    2.  Load model, data, and training configurations from a YAML file.
    3.  Initialize Weights & Biases for experiment tracking (optional).
    4.  Set up the computation device (CPU or GPU).
    5.  Load the processed item and interaction data.
    6.  Validate that the numerical features specified in the config exist in the data.
    7.  Initialize the feature cache system for faster data loading.
    8.  Fit or load the numerical feature scaler.
    9.  Create dataset instances for fitting encoders and for training/validation splits.
    10. Instantiate the multimodal recommender model.
    11. Initialize the Trainer, which manages the training loop, optimization, and checkpointing.
    12. Run the training process.
    13. Save final model, metadata, and configuration files.
    """
    
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Train multimodal recommender with dynamic numerical features')
    parser.add_argument('--config', type=str, default='configs/simple_config.yaml', help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='MultimodalRecommender', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name for this training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args(cli_args)

    # Print a header for the training run.
    print("MULTIMODAL RECOMMENDER TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file: {args.config}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Display system information.
    print_system_info()

    try:
        # STEP 1: Load configuration from the specified YAML file.
        print_progress_header(1, "Loading Configuration")
        step_start = time.time()
        
        config = Config.from_yaml(args.config)
        data_config = config.data
        model_config = config.model
        training_config = config.training
        
        print(f"Configuration loaded successfully:")
        print(f"Vision model: {model_config.vision_model}")
        print(f"Language model: {model_config.language_model}")
        print(f"Embedding dim: {model_config.embedding_dim}")
        print(f"Batch size: {training_config.batch_size}")
        print(f"Learning rate: {training_config.learning_rate}")
        print(f"Epochs: {training_config.epochs}")
        print(f"Configured numerical features: {data_config.numerical_features_cols}")
        
        print_progress_footer(step_start)

        # STEP 2: Validate configuration
        print_progress_header(2, "Validating Configuration")
        step_start = time.time()
        
        # Ensure required directories exist
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate model configurations
        if model_config.vision_model not in MODEL_CONFIGS['vision']:
            raise ValueError(f"Unknown vision model: {model_config.vision_model}")
        if model_config.language_model not in MODEL_CONFIGS['language']:
            raise ValueError(f"Unknown language model: {model_config.language_model}")
        
        print("✓ Configuration validated")
        print_progress_footer(step_start)
        
        # Run the training pipeline
        results = run_training(config, args)
        
        # Print final summary
        print("\n" + "=" * 80)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total time: {results['training_time']/3600:.2f} hours")
        print(f"Epochs completed: {results['epochs_completed']}")
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
        print(f"Final validation loss: {results['final_val_loss']:.4f}")
        print(f"Model checkpoint: {results['model_path']}")
        print("=" * 80)
        
        # Clean up wandb if used
        if args.use_wandb and 'wandb' in sys.modules:
            wandb = sys.modules['wandb']
            if hasattr(wandb, 'run') and wandb.run is not None:
                try:
                    wandb.finish()
                except Exception as e:
                    print(f"Warning: Failed to close wandb run: {e}")
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user")
        print("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Training failed with exception:")
        print(f"{type(e).__name__}: {str(e)}")
        print(f"{'='*60}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()