#!/usr/bin/env python
"""
A command-line script for evaluating the performance of recommender models.

This script orchestrates the evaluation process by:
- Loading a model configuration and a trained model checkpoint.
- Initializing a specified recommender (e.g., multimodal, popularity baseline).
- Loading the appropriate training and test datasets.
- Executing a specified evaluation task (e.g., retrieval, ranking).
- Reporting performance metrics and saving the results to a JSON file.
"""
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from tqdm import tqdm
from typing import Optional, Any
import random
import warnings

# Add the project's root directory to the system path to allow importing from 'src'.
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import MultimodalRecommender
from src.inference.recommender import Recommender as MultimodalModelRecommender
from src.evaluation.tasks import EvaluationTask, create_evaluator
from src.inference.baseline_recommenders import (
    RandomRecommender, PopularityRecommender,
    ItemKNNRecommender, UserKNNRecommender,
    BaselineRecommender 
)
from src.data.processors.numerical_processor import NumericalProcessor 

# Imports the SimpleFeatureCache for efficient data handling during evaluation.
try:
    from src.data.simple_cache import SimpleFeatureCache
except ImportError:
    # Provides a fallback in-memory cache if the main class is not available.
    class SimpleFeatureCache:
        def __init__(self, *args, **kwargs):
            self.cache = {}
            print("Using fallback SimpleFeatureCache")
        def get(self, item_id): return self.cache.get(item_id)
        def set(self, item_id, features): self.cache[item_id] = features
        def print_stats(self): print(f"Simple cache: {len(self.cache)} items")


def find_model_checkpoint(config: Config, checkpoint_name: str = 'best_model.pth') -> Path:
    """    
    Locates the model checkpoint file using a prioritized search strategy.

    It first searches in the model-specific directory (e.g., 'checkpoints/resnet_bert/'),
    then falls back to the base checkpoint directory and tries alternative common names.

    Args:
        config: The main configuration object containing directory paths.
        checkpoint_name: The preferred name of the checkpoint file.

    Returns:
        A pathlib.Path object pointing to the found checkpoint file.
        
    Raises:
        FileNotFoundError: If no suitable checkpoint file is found.
    """
    # Constructs the path for the model-specific checkpoint directory.
    model_combo = f"{config.model.vision_model}_{config.model.language_model}"
    model_specific_dir = Path(config.checkpoint_dir) / model_combo
    
    # Defines a prioritized list of checkpoint filenames to search for.
    checkpoint_names = ['best_model.pth', 'final_model.pth', 'last_model.pth']
    
    # Searches for the checkpoint in the model-specific directory first.
    for name in checkpoint_names:
        path = model_specific_dir / name
        if path.exists():
            print(f"✓ Found checkpoint in model-specific directory: {path}")
            return path
            
    # Falls back to searching in the base checkpoint directory for backward compatibility.
    for name in checkpoint_names:
        path = Path(config.checkpoint_dir) / name
        if path.exists():
            print(f"✓ Found checkpoint in base directory (fallback): {path}")
            return path
            
    # If no preferred checkpoint is found, searches for alternative common names.
    alternative_names = ['model.pth', 'checkpoint.pth']
    for alt_name in alternative_names:
        alt_model_path = model_specific_dir / alt_name
        if alt_model_path.exists():
            print(f"✓ Found alternative checkpoint in model-specific directory: {alt_model_path}")
            return alt_model_path
        
        alt_base_path = Path(config.checkpoint_dir) / alt_name
        if alt_base_path.exists():
            print(f"✓ Found alternative checkpoint in base directory: {alt_base_path}")
            return alt_base_path
    
    # Raises an error if no checkpoint is found after all search attempts.
    raise FileNotFoundError(
        f"Model checkpoint not found. Searched for {checkpoint_names + alternative_names} in:\n"
        f"  → Model-specific dir: {model_specific_dir}\n"
        f"  → Base dir: {config.checkpoint_dir}"
    )


def find_encoders(config: Config) -> Path:
    """
    Locates the directory containing the user and item encoder files.

    It searches in a prioritized list of locations: first a shared 'encoders'
    directory, then the base checkpoint directory, and finally the model-specific
    directory as a last resort.

    Args:
        config: The main configuration object.
        
    Returns:
        A pathlib.Path object for the directory containing the encoders.
        
    Raises:
        FileNotFoundError: If the encoder files are not found in any of the
                           searched locations.
    """
    # Defines the primary path for shared encoders.
    shared_encoders_path = Path(config.checkpoint_dir) / 'encoders'
    
    if shared_encoders_path.exists():
        user_encoder_path = shared_encoders_path / 'user_encoder.pkl'
        item_encoder_path = shared_encoders_path / 'item_encoder.pkl'
        
        if user_encoder_path.exists() and item_encoder_path.exists():
            print(f"✓ Found encoders in shared directory: {shared_encoders_path}")
            return shared_encoders_path
    
    # Falls back to searching in the base checkpoint directory.
    base_encoders_path = Path(config.checkpoint_dir)
    user_encoder_path = base_encoders_path / 'user_encoder.pkl'
    item_encoder_path = base_encoders_path / 'item_encoder.pkl'
    
    if user_encoder_path.exists() and item_encoder_path.exists():
        print(f"✓ Found encoders in base directory (backward compatibility): {base_encoders_path}")
        return base_encoders_path
    
    # As a last resort, checks the model-specific directory.
    model_combo = f"{config.model.vision_model}_{config.model.language_model}"
    model_encoders_path = Path(config.checkpoint_dir) / model_combo
    user_encoder_path = model_encoders_path / 'user_encoder.pkl'
    item_encoder_path = model_encoders_path / 'item_encoder.pkl'
    
    if user_encoder_path.exists() and item_encoder_path.exists():
        print(f"✓ Found encoders in model-specific directory: {model_encoders_path}")
        return model_encoders_path
    
    raise FileNotFoundError(
        f"Encoders not found. Searched locations:\n"
        f"  → Shared directory: {shared_encoders_path}\n"
        f"  → Base directory: {base_encoders_path}\n"
        f"  → Model-specific directory: {model_encoders_path}\n"
        f"  Ensure training or the 'extract_encoders.py' script has been run."
    )


def create_recommender(
    recommender_type: str, 
    dataset: MultimodalDataset, 
    model: Optional[torch.nn.Module] = None, 
    device: Optional[torch.device] = None,    
    history_interactions_df: Optional[pd.DataFrame] = None,
    config_obj: Optional[Config] = None,
    simple_cache_instance: Optional[SimpleFeatureCache] = None
):
    """
    A factory function to instantiate and return a specific recommender object.

    Args:
        recommender_type: The type of recommender to create (e.g., 'multimodal',
                          'popularity').
        dataset: The dataset object, used by all recommenders for data access.
        model: The trained PyTorch model, required for 'multimodal' type.
        device: The PyTorch device, required for 'multimodal' type.
        history_interactions_df: DataFrame of training interactions, used by
                                 baseline models to establish user history.
        config_obj: The main configuration object.
        simple_cache_instance: An instance of the feature cache.

    Returns:
        An instantiated recommender object ready for evaluation.

    Raises:
        ValueError: If an unknown recommender type is provided or if required
                    arguments for a specific type are missing.
    """
    # Retrieves k-neighbors from config for KNN-based recommenders.
    k_neighbors_val = 50 
    if config_obj and hasattr(config_obj, 'model') and hasattr(config_obj.model, 'k_neighbors'): 
        k_neighbors_val = config_obj.model.k_neighbors

    # Instantiates the specified recommender class.
    if recommender_type == 'multimodal':
        if model is None or config_obj is None:
            raise ValueError("Model and Config object are required for multimodal recommender")
        return MultimodalModelRecommender(
            model, dataset, device,
            cache_max_items=config_obj.data.cache_config.max_memory_items,
            cache_dir=config_obj.data.cache_config.cache_directory if simple_cache_instance else None,
            cache_to_disk=config_obj.data.cache_config.use_disk
        )
    elif recommender_type == 'random':
        return RandomRecommender(dataset, device, history_interactions_df=history_interactions_df)
    elif recommender_type == 'popularity':
        return PopularityRecommender(dataset, device, history_interactions_df=history_interactions_df)
    elif recommender_type == 'item_knn':
        return ItemKNNRecommender(dataset, device, k_neighbors=k_neighbors_val, history_interactions_df=history_interactions_df)
    elif recommender_type == 'user_knn':
        return UserKNNRecommender(dataset, device, k_neighbors=k_neighbors_val, history_interactions_df=history_interactions_df)
    else:
        raise ValueError(f"Unknown recommender type: {recommender_type}")


def main():
    """
    Main function to execute the full evaluation pipeline.
    """
    # Suppresses specific warnings from the transformers library during model loading.
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.clip.modeling_clip")
    
    # Configures and parses command-line arguments.
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument('--config', type=str, default='configs/simple_config.yaml', help='Path to configuration file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data CSV file')
    parser.add_argument('--train_data', type=str, help='Path to training data CSV file for user history')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for evaluation')
    parser.add_argument('--recommender_type', type=str, default='multimodal', choices=['multimodal', 'random', 'popularity', 'item_knn', 'user_knn'], help='Recommender to evaluate')
    parser.add_argument('--eval_task', type=str, default='retrieval', choices=['retrieval', 'ranking'], help='Evaluation task')
    parser.add_argument('--save_predictions', type=str, default=None, help='Path to save user-level predictions')
    parser.add_argument('--warmup_recommender_cache', action='store_true', help="Warm-up the Recommender's feature cache")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for evaluation')
    parser.add_argument('--use_sampling', action='store_true', default=True, help='Use negative sampling for faster evaluation')
    parser.add_argument('--no_sampling', dest='use_sampling', action='store_false', help='Disable negative sampling')
    parser.add_argument('--num_negatives', type=int, default=100, help='Number of negative samples per positive item')
    parser.add_argument('--sampling_strategy', type=str, default='random', choices=['random', 'popularity', 'popularity_inverse'], help='Negative sampling strategy')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pth', help='Name of checkpoint file to load')
    args = parser.parse_args()

    # Loads configuration and sets up the device.
    config_obj = Config.from_yaml(args.config)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Loads the datasets required for evaluation.
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    train_df = None
    if args.train_data:
        print(f"Loading training data from {args.train_data}...")
        train_df = pd.read_csv(args.train_data)
    elif args.eval_task == 'retrieval':
        print(f"Note: --train_data not provided. User history for filtering seen items will be empty.")

    # Loads item metadata and interactions for initializing the dataset object.
    item_info_df = pd.read_csv(config_obj.data.processed_item_info_path)
    interactions_df_for_dataset_init = pd.read_csv(config_obj.data.processed_interactions_path)

    # Determines which numerical features to use by checking the saved scaler.
    numerical_scaler = None
    numerical_features_for_dataset = config_obj.data.numerical_features_cols
    scaler_path = Path(config_obj.data.scaler_path)
    if scaler_path.exists():
        print(f"Loading numerical scaler from {scaler_path}...")
        num_processor = NumericalProcessor()
        if num_processor.load_scaler(scaler_path) and num_processor.fitted_columns:
            print(f"  → Scaler was fitted on {len(num_processor.fitted_columns)} columns: {num_processor.fitted_columns}")
            numerical_features_for_dataset = num_processor.fitted_columns
            numerical_scaler = num_processor.scaler
    else:
        print(f"  → Warning: Scaler file not found at {scaler_path}. Numerical features will not be scaled.")
    print(f"✅ Using {len(numerical_features_for_dataset)} numerical features: {numerical_features_for_dataset}")

    # Determines the correct image folder to use (raw or processed).
    effective_image_folder = config_obj.data.processed_image_destination_folder or config_obj.data.image_folder

    # Initializes the feature cache if enabled in the configuration.
    simple_cache_instance = None
    cache_config = config_obj.data.cache_config
    if cache_config.enabled:
        simple_cache_instance = SimpleFeatureCache(
            vision_model=config_obj.model.vision_model,
            language_model=config_obj.model.language_model,
            base_cache_dir=cache_config.cache_directory,
            max_memory_items=cache_config.max_memory_items,
            use_disk=cache_config.use_disk
        )
        simple_cache_instance.print_stats()

    # Creates a dataset instance primarily for fitting the encoders.
    dataset_for_encoders = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init,
        item_info_df=item_info_df,
        image_folder=effective_image_folder,
        vision_model_name=config_obj.model.vision_model,
        language_model_name=config_obj.model.language_model,
        create_negative_samples=False,
        cache_features=cache_config.enabled,
        cache_max_items=cache_config.max_memory_items,
        cache_dir=cache_config.cache_directory,
        cache_to_disk=cache_config.use_disk,
        numerical_feat_cols=numerical_features_for_dataset,
        numerical_normalization_method=config_obj.data.numerical_normalization_method,
        numerical_scaler=numerical_scaler
    )
    
    # Loads the pre-fitted user and item encoders.
    try:
        encoders_dir = find_encoders(config_obj)
        with open(encoders_dir / 'user_encoder.pkl', 'rb') as f:
            dataset_for_encoders.user_encoder = pickle.load(f)
        with open(encoders_dir / 'item_encoder.pkl', 'rb') as f:
            dataset_for_encoders.item_encoder = pickle.load(f)
        dataset_for_encoders.n_users = len(dataset_for_encoders.user_encoder.classes_)
        dataset_for_encoders.n_items = len(dataset_for_encoders.item_encoder.classes_)
        print(f"Loaded encoders: {dataset_for_encoders.n_users} users, {dataset_for_encoders.n_items} items.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initializes the model if the multimodal recommender is being evaluated.
    model_instance_multimodal = None
    if args.recommender_type == 'multimodal':
        print("Initializing multimodal model...")
        model_instance_multimodal = MultimodalRecommender(
            n_users=dataset_for_encoders.n_users,
            n_items=dataset_for_encoders.n_items,
            num_numerical_features=len(numerical_features_for_dataset), # Use the validated number of features
            embedding_dim=config_obj.model.embedding_dim,
            vision_model_name=config_obj.model.vision_model,         
            language_model_name=config_obj.model.language_model,     
            freeze_vision=config_obj.model.freeze_vision,
            freeze_language=config_obj.model.freeze_language,
            use_contrastive=config_obj.model.use_contrastive,
            dropout_rate=0.0,  # No dropout during evaluation
            num_attention_heads=config_obj.model.num_attention_heads,
            attention_dropout=0.0, # No dropout during evaluation
            use_batch_norm=config_obj.model.use_batch_norm,
            fusion_hidden_dims=config_obj.model.fusion_hidden_dims,
            fusion_activation=config_obj.model.fusion_activation,
            projection_hidden_dim=config_obj.model.projection_hidden_dim,
            final_activation=config_obj.model.final_activation,
            init_method=config_obj.model.init_method,
            contrastive_temperature=config_obj.model.contrastive_temperature
        ).to(device)
                
        # Loads the trained weights from the checkpoint file.
        try:
            checkpoint_path = find_model_checkpoint(config_obj, args.checkpoint_name)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_instance_multimodal.load_state_dict(checkpoint['model_state_dict'])
            model_instance_multimodal.eval()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Creates the recommender instance to be evaluated.
    recommender_instance = create_recommender(
        args.recommender_type,
        dataset=dataset_for_encoders, 
        model=model_instance_multimodal,
        device=device,
        history_interactions_df=train_df, 
        config_obj=config_obj,
        simple_cache_instance=simple_cache_instance
    )

    # Optionally warms up the feature cache before evaluation begins.
    if args.recommender_type == 'multimodal' and args.warmup_recommender_cache:
        print("Warming up Recommender's cache...")
        all_item_ids_for_warmup = dataset_for_encoders.item_encoder.classes_
        for item_id_to_warm in tqdm(all_item_ids_for_warmup[:1000], desc="Warming cache"):
            if hasattr(recommender_instance, '_get_item_features'):
                recommender_instance._get_item_features(item_id_to_warm) 
        if hasattr(recommender_instance, 'print_cache_stats'): recommender_instance.print_cache_stats()

    # Initializes the appropriate evaluator for the specified task.
    task_map = {'retrieval': EvaluationTask.TOP_K_RETRIEVAL, 'ranking': EvaluationTask.TOP_K_RANKING}
    eval_task_enum_val = task_map.get(args.eval_task)
    
    evaluator = create_evaluator(
        task=eval_task_enum_val, recommender=recommender_instance, test_data=test_df,
        config=config_obj, train_data=train_df,
        use_sampling=(args.use_sampling if args.eval_task == 'retrieval' else False),
        num_negatives=args.num_negatives, sampling_strategy=args.sampling_strategy,
        num_workers=args.num_workers
    )
    
    # Executes the evaluation and prints a summary of the results.
    print(f"\nStarting evaluation for task: {evaluator.task_name}...")
    results = evaluator.evaluate()

    # Optionally saves user-level predictions to a file.
    if args.save_predictions and 'predictions' in results:
        predictions_data = results.pop('predictions')
        save_path = Path(config_obj.results_dir) / args.save_predictions
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensures predictions are JSON-serializable.
        serializable_predictions = {str(user): [{'item_id': str(item), 'score': float(score)} for item, score in recs]
                                    for user, recs in predictions_data.items()}
        with open(save_path, 'w') as f:
            json.dump(serializable_predictions, f, indent=2)
        print(f"\n✓ User-level predictions saved to {save_path}")

    # Saves the final evaluation metrics to a JSON file.
    results['evaluation_metadata'] = {
        'task': evaluator.task_name, 'recommender_type': args.recommender_type,
        'top_k': config_obj.recommendation.top_k, 'test_file': args.test_data,
        'model_combination': f"{config_obj.model.vision_model}_{config_obj.model.language_model}",
        'checkpoint_used': args.checkpoint_name
    }
    evaluator.print_summary(results)

    output_path = Path(config_obj.results_dir) / args.output 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        # Custom JSON encoder to handle NumPy data types.
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        json.dump(results, f, indent=2, cls=NpEncoder)

    print(f"\nEvaluation results saved to {output_path}")

if __name__ == '__main__':
    main()