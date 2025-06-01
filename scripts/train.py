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
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.training.trainer import Trainer


def main():
    # Parse arguments
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
    
    # Load configuration
    config = Config.from_yaml(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    item_info_df = pd.read_csv(config.data.item_info_path)
    interactions_df = pd.read_csv(config.data.interactions_path)
    
    # Sample data if specified
    if config.data.sample_size:
        print(f"Sampling {config.data.sample_size} interactions...")
        interactions_df = interactions_df.sample(
            n=min(config.data.sample_size, len(interactions_df)), 
            random_state=42
        )
    
    # Create dataset
    print("\nCreating dataset...")
    print(f"Vision model: {config.model.vision_model}")
    print(f"Language model: {config.model.language_model}")
    
    dataset = MultimodalDataset(
        interactions_df,
        item_info_df,
        config.data.image_folder,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        negative_sampling_ratio=config.data.negative_sampling_ratio
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of users: {dataset.n_users}")
    print(f"Number of items: {dataset.n_items}")
    
    # Split dataset
    train_size = int(config.data.train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = PretrainedMultimodalRecommender(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=config.model.embedding_dim,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        freeze_vision=config.model.freeze_vision,
        freeze_language=config.model.freeze_language,
        use_contrastive=config.model.use_contrastive,
        dropout_rate=config.model.dropout_rate
    ).to(device)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        use_contrastive=config.model.use_contrastive
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        patience=config.training.patience,
        gradient_clip=config.training.gradient_clip
    )
    
    # Save final model
    final_checkpoint_path = Path(config.checkpoint_dir) / 'final_model.pth'
    trainer.save_checkpoint('final_model.pth')
    print(f"\nSaved final model to {final_checkpoint_path}")
    
    # Save encoders
    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    encoders_dir.mkdir(parents=True, exist_ok=True)
    
    with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.user_encoder, f)
    
    with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.item_encoder, f)
    
    print(f"Saved encoders to {encoders_dir}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    results_dir = Path(config.results_dir) / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'training_curves.png')
    print(f"\nSaved training curves to {results_dir / 'training_curves.png'}")
    
    # Save configuration
    config.to_yaml(Path(config.results_dir) / 'train_config.yaml')
    print(f"Saved configuration to {Path(config.results_dir) / 'train_config.yaml'}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()