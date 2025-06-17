#!/usr/bin/env python3
"""
Hyperparameter optimization script using Optuna.

This script performs automated hyperparameter search for the multimodal recommender system
by wrapping the training process and using Optuna's Bayesian optimization.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import optuna
from optuna import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import yaml
import logging
import torch

# Add the parent directory to the path to import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from scripts.train import run_training  # Import run_training instead of main


def create_objective(base_config_path: str, args: argparse.Namespace):
    """
    Creates the objective function for Optuna optimization.
    
    Args:
        base_config_path: Path to the base configuration file
        args: Command line arguments
        
    Returns:
        The objective function that Optuna will optimize
    """
    
    def objective(trial: Trial) -> float:
        """
        Objective function that Optuna optimizes.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            The metric value to optimize (lower is better for loss)
        """
        # Load base configuration
        config = Config.from_yaml(base_config_path)
        
        # Define the available model choices for each modality
        # 1. Define the complete, static set of choices for each modality.
        vision_model_choices = ['clip', 'resnet', 'dino', 'convnext', None]
        language_model_choices = ['sentence-bert', 'clip', 'bert-base', 'distilbert', None]

        # 2. Suggest from the static lists for both models.
        # Now, both categorical suggestions use a fixed list of choices for every trial.
        config.model.vision_model = trial.suggest_categorical(
            'vision_model', vision_model_choices
        )
        config.model.language_model = trial.suggest_categorical(
            'language_model', language_model_choices
        )

        # 3. Validate the combination and prune the trial if it is invalid.
        # This check enforces the rule that at least one of the models must be active.
        if config.model.vision_model is None and config.model.language_model is None:
            # If both suggested models are None, the combination is invalid.
            # We prune the trial, telling Optuna to stop this run and start a new one.
            raise optuna.TrialPruned("Both vision and language models cannot be None.")


        # Suggest hyperparameters using Optuna
        # Training hyperparameters
        config.training.learning_rate = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True
        )
        config.training.batch_size = trial.suggest_categorical(
            'batch_size', [16, 32, 64, 128]
        )
        config.training.weight_decay = trial.suggest_float(
            'weight_decay', 1e-6, 1e-2, log=True
        )
        config.training.gradient_clip = trial.suggest_float(
            'gradient_clip', 0.5, 5.0
        )
        config.model.num_attention_heads = trial.suggest_categorical(
            'num_attention_heads', [2, 4, 8]
        )
        
        # Model hyperparameters
        config.model.embedding_dim = trial.suggest_categorical(
            'embedding_dim', [64, 128, 256, 512]
        )
        
        config.model.fusion_type = trial.suggest_categorical(
            'fusion_type', ['concatenate', 'attention', 'gated']
        )

         # Dropout rates
        config.model.dropout_rate = trial.suggest_float(
            'dropout_rate', 0.1, 0.5
        )
        config.model.attention_dropout = trial.suggest_float(
            'attention_dropout', 0.0, 0.3
        )
        
        # Hidden layers configuration
        fusion_hidden_configs = [
            "256, 128",
            "512, 256",
            "512, 256, 128",
            "256, 128, 64",
            "128, 64",
            "512",
            "256"
        ]
        # Optuna suggests a string
        chosen_config_str = trial.suggest_categorical(
            'fusion_hidden_dims', fusion_hidden_configs
        )
        # Convert the chosen string back to a list of integers
        config.model.fusion_hidden_dims = [int(x) for x in chosen_config_str.split(',')]

        # Projection hidden dimension 
        config.model.projection_hidden_dim = trial.suggest_categorical(
            'projection_hidden_dim', [None, 128, 256, 512]
        )
        
        # Activation functions
        config.model.fusion_activation = trial.suggest_categorical(
            'fusion_activation', ['relu', 'gelu', 'tanh', 'leaky_relu']
        )
        
        # Batch normalization
        config.model.use_batch_norm = trial.suggest_categorical(
            'use_batch_norm', [True, False]
        )
        
        # Contrastive learning configuration 
        config.model.use_contrastive = trial.suggest_categorical(
            'use_contrastive', [True, False]
        )
        
        ## --- Contrastive Learning Section ---
        # Suggest whether to use contrastive learning.
        config.model.use_contrastive = trial.suggest_categorical(
            'use_contrastive', [True, False]
        )

        # ALWAYS suggest all parameters. The training script will be responsible
        # for ignoring them if they are not needed (e.g., if use_contrastive is False).
        config.model.contrastive_temperature = trial.suggest_float(
            'contrastive_temperature', 0.01, 0.5, log=True
        )
        config.training.contrastive_weight = trial.suggest_float(
            'contrastive_weight', 0.01, 1.0
        )
        config.training.bce_weight = trial.suggest_float(
            'bce_weight', 0.5, 1.0
        )

        # --- Optimizer Section ---
        # Suggest the optimizer type.
        config.training.optimizer_type = trial.suggest_categorical(
            'optimizer_type', ['adam', 'adamw', 'sgd']
        )

        # ALWAYS suggest the Adam-specific parameters. The trainer will only use them
        # if the optimizer is 'adam' or 'adamw'.
        config.training.adam_beta1 = trial.suggest_float('adam_beta1', 0.8, 0.99)
        config.training.adam_beta2 = trial.suggest_float('adam_beta2', 0.9, 0.999)
        config.training.adam_eps = trial.suggest_float('adam_eps', 1e-9, 1e-7, log=True)

        # --- Learning Rate Scheduler Section ---
        # Suggest whether to use a learning rate scheduler.
        config.training.use_lr_scheduler = trial.suggest_categorical(
            'use_lr_scheduler', [True, False]
        )

        # ALWAYS suggest the scheduler-specific parameters. The trainer will only use them
        # if use_lr_scheduler is True.
        config.training.lr_scheduler_type = trial.suggest_categorical(
            'lr_scheduler_type', ['reduce_on_plateau', 'cosine', 'step']
        )
        config.training.lr_scheduler_factor = trial.suggest_float(
            'lr_scheduler_factor', 0.1, 0.9
        )

        
        # Create unique directories for this trial
        trial_dir = Path(args.output_dir) / f"trial_{trial.number}"
        config.checkpoint_dir = str(trial_dir / "checkpoints")
        config.results_dir = str(trial_dir / "results")
        
        # Save trial configuration
        trial_config_path = trial_dir / "config.yaml"
        trial_config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(trial_config_path))
        
        # Create args namespace for run_training
        train_args = argparse.Namespace(
            config=str(trial_config_path),
            device=args.device,
            resume=None,
            use_wandb=args.use_wandb,
            wandb_project=f"{args.wandb_project}_optuna" if args.use_wandb else None,
            wandb_entity=args.wandb_entity if args.use_wandb else None,
            wandb_run_name=f"trial_{trial.number}" if args.use_wandb else None,
            verbose=args.verbose if hasattr(args, 'verbose') else False,
            # Add trial information to args
            trial_info={
                'trial_number': trial.number,
                'trial_params': trial.params,
                'study_name': args.study_name,
                'optimization_direction': args.direction,
                'target_metric': args.optimize_metric
            }
        )
        
        try:
            # Run training and capture the results
            print(f"\n{'='*60}")
            print(f"Starting Trial {trial.number}")
            print(f"Hyperparameters: {trial.params}")
            print(f"{'='*60}\n")
            
            # Run the training using run_training function
            results = run_training(config, train_args)
            
            # Extract the metric to optimize
            if args.optimize_metric == 'val_loss':
                best_metric = results.get('best_val_loss', float('inf'))
            elif args.optimize_metric in results.get('all_best_metrics', {}):
                # Use the all_best_metrics dictionary if available
                best_metric = results['all_best_metrics'][args.optimize_metric]
            elif f'best_{args.optimize_metric}' in results:
                # Try with 'best_' prefix
                best_metric = results[f'best_{args.optimize_metric}']
            else:
                # Fallback to metadata if available
                metadata = results.get('metadata', {})
                best_metric = metadata.get(f'best_{args.optimize_metric}', float('inf'))
                if best_metric == float('inf'):
                    print(f"Warning: Metric {args.optimize_metric} not found. Using val_loss.")
                    best_metric = results.get('best_val_loss', float('inf'))
            
            # Report intermediate values for pruning
            val_losses = results.get('val_losses', [])
            for epoch, val_loss in enumerate(val_losses):
                trial.report(val_loss, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
            
            print(f"\nTrial {trial.number} completed. Best {args.optimize_metric}: {best_metric}")
            
            # Save trial summary
            trial_summary = {
                'trial_number': trial.number,
                'best_metric': best_metric,
                'metric_name': args.optimize_metric,
                'params': trial.params,
                'epochs_completed': results.get('epochs_completed', 0),
                'training_time': results.get('training_time', 0),
                'all_best_metrics': results.get('all_best_metrics', {})
            }
            
            trial_summary_path = trial_dir / 'trial_summary.json'
            with open(trial_summary_path, 'w') as f:
                json.dump(trial_summary, f, indent=2)
            
            return best_metric
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return worst possible value on error
            return float('inf') if args.direction == 'minimize' else float('-inf')
    
    return objective


def main():
    """Main function to run hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization for multimodal recommender'
    )
    parser.add_argument(
        '--config', type=str, default='configs/simple_config.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--n_trials', type=int, default=100,
        help='Number of trials to run'
    )
    parser.add_argument(
        '--study_name', type=str, default=None,
        help='Name for the Optuna study (default: auto-generated)'
    )
    parser.add_argument(
        '--storage', type=str, default=None,
        help='Database URL for distributed optimization (e.g., sqlite:///study.db)'
    )
    parser.add_argument(
        '--direction', type=str, default='minimize',
        choices=['minimize', 'maximize'],
        help='Direction of optimization'
    )
    parser.add_argument(
        '--optimize_metric', type=str, default='val_loss',
        help='Metric to optimize (e.g., val_loss, val_accuracy, val_f1_score)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='optuna_trials',
        help='Directory to save trial results'
    )
    parser.add_argument(
        '--device', type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--use_wandb', action='store_true',
        help='Enable Weights & Biases logging for trials'
    )
    parser.add_argument(
        '--wandb_project', type=str, default='MultimodalRecommender',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb_entity', type=str, default=None,
        help='Weights & Biases entity'
    )
    parser.add_argument(
        '--pruning', action='store_true',
        help='Enable trial pruning based on intermediate values'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume an existing study'
    )
    parser.add_argument(
        '--parallel', action='store_true',
        help='Enable parallel trial execution (requires storage)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Generate study name if not provided
    if args.study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.study_name = f"multimodal_rec_study_{timestamp}"
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save study configuration
    study_config = vars(args)
    study_config_path = Path(args.output_dir) / 'study_config.json'
    with open(study_config_path, 'w') as f:
        json.dump(study_config, f, indent=2)
    
    print(f"\nStarting Optuna hyperparameter optimization")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Optimization direction: {args.direction}")
    print(f"Metric to optimize: {args.optimize_metric}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Pruning enabled: {args.pruning}")
    print(f"Parallel execution: {args.parallel}")
    
    # Create sampler and pruner
    sampler = TPESampler(seed=42)  # Tree-structured Parzen Estimator
    pruner = MedianPruner() if args.pruning else None
    
    # --- START OF CORRECTION ---
    # Simplified and corrected study creation/loading logic.
    # This single call to 'create_study' handles all cases correctly.
    # The 'load_if_exists' flag, set by '--resume', ensures that Optuna
    # will either load an existing study with the specified name or create a
    # new one, preventing the KeyError.
    print(f"\nCreating or resuming study '{args.study_name}' from storage: {args.storage}")
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction=args.direction,
        load_if_exists=args.resume
    )

    if args.resume and len(study.trials) > 0:
        print(f"Successfully resumed study with {len(study.trials)} existing trials.")
    else:
        print("Starting a new study or resuming an empty one.")
    # --- END OF CORRECTION ---
    
    # Create objective function
    objective = create_objective(args.config, args)
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=1 if not args.parallel else -1,  # Use all cores if parallel
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if len(study.trials) > 0 and study.best_trial:
        print(f"\nBest trial:")
        best_trial = study.best_trial
        print(f"  Number: {best_trial.number}")
        print(f"  Value ({args.optimize_metric}): {best_trial.value:.6f}")
        print(f"\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Save best hyperparameters
        best_params_path = Path(args.output_dir) / 'best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump({
                'trial_number': best_trial.number,
                'value': best_trial.value,
                'params': best_trial.params,
                'datetime': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nBest parameters saved to {best_params_path}")
        
        # Save full study results
        study_results_path = Path(args.output_dir) / 'study_results.json'
        study_df = study.trials_dataframe()
        study_df.to_json(study_results_path, orient='records', indent=2)
        print(f"Full study results saved to {study_results_path}")
        
        # Load best trial's full results if available
        best_trial_dir = Path(args.output_dir) / f"trial_{best_trial.number}"
        best_trial_summary_path = best_trial_dir / 'trial_summary.json'
        if best_trial_summary_path.exists():
            with open(best_trial_summary_path, 'r') as f:
                best_trial_summary = json.load(f)
            
            print(f"\nBest trial training summary:")
            print(f"  Epochs completed: {best_trial_summary.get('epochs_completed', 'N/A')}")
            print(f"  Training time: {best_trial_summary.get('training_time', 0)/3600:.2f} hours")
            
            if 'all_best_metrics' in best_trial_summary:
                print(f"\nAll metrics from best trial:")
                for metric, value in best_trial_summary['all_best_metrics'].items():
                    print(f"  {metric}: {value:.6f}")
        
        # Generate visualization if optuna-dashboard is available
        try:
            import optuna.visualization as vis
            
            # Plot optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_html(Path(args.output_dir) / 'optimization_history.html')
            
            # Plot parameter importance
            if len(study.trials) > 5:  # Need at least a few trials for importance
                fig = vis.plot_param_importances(study)
                fig.write_html(Path(args.output_dir) / 'param_importances.html')
            
            # Plot parallel coordinate
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(Path(args.output_dir) / 'parallel_coordinate.html')
            
            print(f"\nVisualizations saved to {args.output_dir}")
        except ImportError:
            print("\nNote: Install plotly for visualization support: pip install plotly")
        except Exception as e:
            print(f"\nWarning: Could not generate visualizations: {e}")
        
        # Create configuration file with best parameters
        print("\nCreating configuration with best parameters...")
        best_config = Config.from_yaml(args.config)
        
        # Apply best parameters
        for param_name, param_value in best_trial.params.items():
            if param_name == 'vision_model':
                best_config.model.vision_model = param_value
            elif param_name == 'language_model':
                best_config.model.language_model = param_value
            elif param_name == 'learning_rate':
                best_config.training.learning_rate = param_value
            elif param_name == 'batch_size':
                best_config.training.batch_size = param_value
            elif param_name == 'weight_decay':
                best_config.training.weight_decay = param_value
            elif param_name == 'gradient_clip':
                best_config.training.gradient_clip = param_value
            elif param_name == 'embedding_dim':
                best_config.model.embedding_dim = param_value
            elif param_name == 'num_attention_heads':
                best_config.model.num_attention_heads = param_value
            elif param_name == 'fusion_type':
                best_config.model.fusion_type = param_value
            elif param_name == 'dropout_rate':
                best_config.model.dropout_rate = param_value
            elif param_name == 'attention_dropout':
                best_config.model.attention_dropout = param_value
            elif param_name == 'fusion_hidden_dims':
                best_config.model.fusion_hidden_dims = [int(x) for x in param_value.split(',')]
            elif param_name == 'projection_hidden_dim':
                best_config.model.projection_hidden_dim = param_value
            elif param_name == 'fusion_activation':
                best_config.model.fusion_activation = param_value
            elif param_name == 'use_batch_norm':
                best_config.model.use_batch_norm = param_value
            elif param_name == 'use_contrastive':
                best_config.model.use_contrastive = param_value
            elif param_name == 'contrastive_temperature':
                best_config.model.contrastive_temperature = param_value
            elif param_name == 'contrastive_weight':
                best_config.training.contrastive_weight = param_value
            elif param_name == 'bce_weight':
                best_config.training.bce_weight = param_value
            elif param_name == 'optimizer_type':
                best_config.training.optimizer_type = param_value
            elif param_name == 'adam_beta1':
                best_config.training.adam_beta1 = param_value
            elif param_name == 'adam_beta2':
                best_config.training.adam_beta2 = param_value
            elif param_name == 'adam_eps':
                best_config.training.adam_eps = param_value
            elif param_name == 'use_lr_scheduler':
                best_config.training.use_lr_scheduler = param_value
            elif param_name == 'lr_scheduler_type':
                best_config.training.lr_scheduler_type = param_value
            elif param_name == 'lr_scheduler_factor':
                best_config.training.lr_scheduler_factor = param_value
        
        best_config_path = Path(args.output_dir) / 'best_config.yaml'
        best_config.to_yaml(str(best_config_path))
        print(f"Best configuration saved to {best_config_path}")
        
        # Provide instructions for using the best model
        best_checkpoint_dir = Path(args.output_dir) / f"trial_{best_trial.number}" / "checkpoints"
        model_combo = f"{best_config.model.vision_model}_{best_config.model.language_model}"
        best_checkpoint_path = best_checkpoint_dir / model_combo / "best_model.pth"

        if best_checkpoint_path.exists():
            print(f"\nTo use the best model:")
            print(f"  Config: {best_config_path}")
            print(f"  Checkpoint: {best_checkpoint_path}")
            print(f"\nEvaluate with:")
            print(f"  python scripts/evaluate.py --config {best_config_path} --checkpoint_name best_model.pth")
    else:
        print("\nNo successful trials completed.")


if __name__ == '__main__':
    main()