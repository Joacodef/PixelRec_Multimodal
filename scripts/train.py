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

    if args.use_wandb:
        try:
            config_dict_for_wandb = dataclasses.asdict(config)
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=config_dict_for_wandb, reinit=True)
            print("Weights & Biases logging enabled.")
            wandb.define_metric("epoch") # Define epoch as a metric
            wandb.define_metric("train/*", step_metric="epoch") # Set epoch as x-axis for train metrics
            wandb.define_metric("val/*", step_metric="epoch")   # Set epoch as x-axis for val metrics
            wandb.define_metric("train/learning_rate", step_metric="epoch")
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
            interactions_df = interactions_df.sample(n=min(data_config.sample_size, len(interactions_df)), random_state=data_config.splitting.random_state).reset_index(drop=True)
            sampled_item_ids = set(interactions_df['item_id'].unique())
            item_info_df = item_info_df[item_info_df['item_id'].isin(sampled_item_ids)].reset_index(drop=True)

        numerical_scaler = None
        scaler_path_obj = Path(data_config.scaler_path)
        if data_config.numerical_normalization_method in ['standardization', 'min_max']:
            if scaler_path_obj.exists():
                numerical_scaler = load_numerical_scaler(scaler_path_obj)
            else:
                print(f"Scaler not found at {scaler_path_obj}. Fitting a new one...")
                # Ensure item_info_df used for fitting scaler is representative (e.g., from non-sampled or training split)
                # For simplicity, using the potentially sampled item_info_df here.
                # Consider fitting scaler only on training portion of item_info_df in a more complex setup.
                numerical_scaler = fit_numerical_scaler(item_info_df, data_config.numerical_features_cols, data_config.numerical_normalization_method, scaler_path_obj)

        print(f"\nSplitting data using strategy: {data_config.splitting.strategy}")
        total_interactions = len(interactions_df)
        unique_users = interactions_df['user_id'].nunique()
        print(f"Dataset info: {total_interactions} interactions, {unique_users} unique users")
        splitter = DataSplitter(random_state=data_config.splitting.random_state)

        if total_interactions < 5000 or unique_users < 100: # Condition for small dataset
            print("Small dataset detected. Using leave_one_out strategy.")
            # Use leave_one_out_strategy from config if specified, otherwise default to 'random'
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
                if not data_config.splitting.timestamp_col: raise ValueError("timestamp_col required for temporal split.")
                split_params['timestamp_col'] = data_config.splitting.timestamp_col
                split_params['train_ratio'] = data_config.splitting.train_ratio
            if data_config.splitting.strategy == 'leave_one_out':
                split_params['strategy'] = data_config.splitting.leave_one_out_strategy
            
            train_interactions_df, val_interactions_df = split_func(**split_params)


        split_stats = splitter.get_split_statistics(train_interactions_df, val_interactions_df)
        print(f"\nSplit statistics:")
        for key, value in split_stats.items(): print(f"  {key}: {value}")
        print(f"\nTraining interactions: {len(train_interactions_df)}")
        print(f"Validation interactions: {len(val_interactions_df)}")

        print("\nCreating dataset instances...")
        # This instance is primarily to fit encoders on the complete (potentially sampled) dataset
        full_dataset_for_encoders = MultimodalDataset(
            interactions_df=interactions_df,
            item_info_df=item_info_df,
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=False, # Not creating samples here for this instance
            negative_sampling_ratio=0,
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler,
            is_train_mode=False
        )
        # The encoders are now fitted inside full_dataset_for_encoders.__init__
        # And n_users/n_items are derived from these fitted encoders.
        print(f"Fitted encoders on full dataset:")
        print(f"  Number of users: {full_dataset_for_encoders.n_users}")
        print(f"  Number of items: {full_dataset_for_encoders.n_items}")

        # Create actual training dataset
        train_dataset = MultimodalDataset(
            interactions_df=train_interactions_df,
            item_info_df=item_info_df, # Full item info
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True, # Will be used by finalize_setup
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=data_config.text_augmentation,
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler, # Pass the fitted scaler
            is_train_mode=True
        )
        # Assign globally fitted encoders and counts
        train_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        train_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        train_dataset.n_users = full_dataset_for_encoders.n_users
        train_dataset.n_items = full_dataset_for_encoders.n_items
        train_dataset.finalize_setup() # Create samples now

        # Create actual validation dataset
        val_dataset = MultimodalDataset(
            interactions_df=val_interactions_df, # This might be empty
            item_info_df=item_info_df, # Full item info
            image_folder=data_config.image_folder,
            vision_model_name=model_config.vision_model,
            language_model_name=model_config.language_model,
            create_negative_samples=True, # Will be used by finalize_setup
            negative_sampling_ratio=data_config.negative_sampling_ratio,
            text_augmentation_config=TextAugmentationConfig(enabled=False),
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler, # Pass the fitted scaler
            is_train_mode=False
        )
        # Assign globally fitted encoders and counts
        val_dataset.user_encoder = full_dataset_for_encoders.user_encoder
        val_dataset.item_encoder = full_dataset_for_encoders.item_encoder
        val_dataset.n_users = full_dataset_for_encoders.n_users
        val_dataset.n_items = full_dataset_for_encoders.n_items
        val_dataset.finalize_setup() # Create samples now
        
        print(f"\nDataset sizes after final setup:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=training_config.num_workers, pin_memory=True)
        # Handle empty val_loader if val_dataset is empty
        val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=training_config.num_workers, pin_memory=True) if len(val_dataset) > 0 else None


        print("\nInitializing model...")
        model_class_to_use = PretrainedMultimodalRecommender
        if hasattr(model_config, 'model_class') and model_config.model_class == 'enhanced':
            model_class_to_use = EnhancedMultimodalRecommender
            print("Using EnhancedMultimodalRecommender")
        else:
            print("Using PretrainedMultimodalRecommender")
        
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
        if model_class_to_use == EnhancedMultimodalRecommender:
            model_params.update({
                'use_cross_modal_attention': model_config.use_cross_modal_attention,
                'cross_modal_attention_weight': model_config.cross_modal_attention_weight
            })
        model = model_class_to_use(**model_params).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        trainer = Trainer(model=model, device=device, checkpoint_dir=config.checkpoint_dir, use_contrastive=model_config.use_contrastive)
        trainer.criterion.contrastive_weight = training_config.contrastive_weight
        trainer.criterion.bce_weight = training_config.bce_weight

        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        print("\nStarting training...")
        training_params_for_trainer = { # Renamed to avoid conflict
            'train_loader': train_loader,
            'val_loader': val_loader, # Pass val_loader, Trainer handles if it's None
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
        train_losses, val_losses = trainer.train(**training_params_for_trainer)

        trainer.save_checkpoint('final_model.pth')
        print(f"\nSaved final model to {config.checkpoint_dir}/final_model.pth")
        encoders_dir = Path(config.checkpoint_dir) / 'encoders'
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with open(encoders_dir / 'user_encoder.pkl', 'wb') as f: pickle.dump(full_dataset_for_encoders.user_encoder, f)
        with open(encoders_dir / 'item_encoder.pkl', 'wb') as f: pickle.dump(full_dataset_for_encoders.item_encoder, f)
        print(f"Saved encoders to {encoders_dir}")

        if train_losses and val_losses : # Only plot if training happened and losses were recorded
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Validation Loss'); plt.legend()
            results_fig_dir = Path(config.results_dir) / 'figures'; results_fig_dir.mkdir(parents=True, exist_ok=True)
            plot_path = results_fig_dir / 'training_curves.png'; plt.savefig(plot_path)
            print(f"\nSaved training curves to {plot_path}")
            if args.use_wandb and wandb.run is not None:
                try: wandb.log({"training_validation_loss_curves": wandb.Image(str(plot_path))})
                except Exception as e: print(f"Warning: Failed to log training curves to wandb: {e}")
        else:
            print("\nSkipping plotting training curves as training might have been skipped or no validation occurred.")


        config_save_path = Path(config.results_dir) / 'training_run_config.yaml'; config.to_yaml(str(config_save_path))
        print(f"Saved effective configuration to {config_save_path}")
        if args.use_wandb and wandb.run is not None:
            try: wandb.save(str(config_save_path))
            except Exception as e: print(f"Warning: Failed to save config to wandb: {e}")
        print("\nTraining completed!")
    finally:
        if args.use_wandb and wandb.run is not None:
            print("Finishing wandb run...")
            wandb.finish()

if __name__ == '__main__':
    main()