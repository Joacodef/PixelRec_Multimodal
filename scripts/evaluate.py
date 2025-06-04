# scripts/evaluate.py
#!/usr/bin/env python
"""
Evaluate model performance
"""
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from tqdm import tqdm
from typing import Optional # Add Optional

# Add parent directory
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
# Ensure correct model class is imported if needed for multimodal
from src.models.multimodal import PretrainedMultimodalRecommender, EnhancedMultimodalRecommender 
from src.inference.recommender import Recommender as MultimodalModelRecommender # Renamed to avoid conflict
from src.evaluation.metrics import calculate_precision_at_k, calculate_recall_at_k, calculate_ndcg
from src.evaluation.tasks import EvaluationTask, create_evaluator
from src.inference.baseline_recommenders import (
    RandomRecommender, PopularityRecommender,
    ItemKNNRecommender, UserKNNRecommender,
    BaselineRecommender # Import Base class if needed for type hinting
)
from src.data.image_cache import SharedImageCache 

# Modified create_recommender function
def create_recommender(
    recommender_type: str, 
    dataset: MultimodalDataset, 
    model: Optional[torch.nn.Module] = None, # Changed from 'Any'
    device: Optional[torch.device] = None,    # Changed from 'Any'
    history_interactions_df: Optional[pd.DataFrame] = None,
    config_obj: Optional[Config] = None # Added config_obj for baseline params if needed
) -> BaselineRecommender | MultimodalModelRecommender: # Type hint for return
    """Create the specified type of recommender"""

    k_neighbors_val = 50 # Default, can be made configurable if needed
    if config_obj and hasattr(config_obj, 'model') and hasattr(config_obj.model, 'k_neighbors'): # Example if k_neighbors was in config
        k_neighbors_val = config_obj.model.k_neighbors


    if recommender_type == 'multimodal':
        if model is None:
            raise ValueError("Model required for multimodal recommender")
        return MultimodalModelRecommender(model, dataset, device) # Use renamed import

    elif recommender_type == 'random':
        return RandomRecommender(dataset, device, history_interactions_df=history_interactions_df)

    elif recommender_type == 'popularity':
        return PopularityRecommender(dataset, device, history_interactions_df=history_interactions_df)

    elif recommender_type == 'item_knn':
        # ItemKNNRecommender's k_neighbors can be made configurable if desired
        return ItemKNNRecommender(dataset, device, k_neighbors=k_neighbors_val, history_interactions_df=history_interactions_df)

    elif recommender_type == 'user_knn':
        # UserKNNRecommender's k_neighbors can be made configurable
        return UserKNNRecommender(dataset, device, k_neighbors=k_neighbors_val, history_interactions_df=history_interactions_df)

    else:
        raise ValueError(f"Unknown recommender type: {recommender_type}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test data CSV file (user_id, item_id interactions)'
    )
    parser.add_argument(
        '--train_data',
        type=str, # This will be used for history_interactions_df
        help='Path to training data CSV file (crucial for defining "seen" history for baselines and "novelty" for evaluators)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Path to save evaluation results JSON file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation (cuda or cpu)'
    )
    parser.add_argument(
        '--recommender_type',
        type=str,
        default='multimodal',
        choices=['multimodal', 'random', 'popularity', 'item_knn', 'user_knn'],
        help='Type of recommender to evaluate'
    )
    parser.add_argument(
        '--eval_task',
        type=str,
        default='retrieval', # Changed from 'topk_retrieval' to match enum better potentially
        choices=['retrieval', 'ranking', 'next_item', 'cold_user', 'cold_item', 'beyond_accuracy', 'legacy'],
        help='Evaluation task to perform (use "legacy" for old evaluation function)'
    )

    args = parser.parse_args()

    config_obj = Config.from_yaml(args.config)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    train_df = None
    if args.train_data:
        print(f"Loading training data from {args.train_data} (this will be used for baseline history)...")
        train_df = pd.read_csv(args.train_data)
    else:
        # For retrieval tasks, train_data is essential for defining novelty correctly.
        # And for baselines to have a fair "seen" history.
        if args.eval_task in ['retrieval', 'next_item', 'cold_user', 'cold_item', 'beyond_accuracy']:
            print(f"Warning: --train_data not provided. For '{args.eval_task}' task, evaluation of novelty might be skewed, and baselines will use full dataset history.")


    print("Loading item info and interaction data for dataset initialization...")
    item_info_df = pd.read_csv(config_obj.data.processed_item_info_path)
    # Dataset for global encoders and general item info should still use full interactions
    interactions_df_for_dataset_init = pd.read_csv(config_obj.data.processed_interactions_path)

    print("Creating dataset instance for global encoders...")
    numerical_scaler = None
    if config_obj.data.numerical_normalization_method in ['standardization', 'min_max']:
        scaler_path = Path(config_obj.data.scaler_path)
        if scaler_path.exists():
            print(f"Loading numerical scaler from {scaler_path}...")
            with open(scaler_path, 'rb') as f:
                numerical_scaler = pickle.load(f)
        # else: Scaler might be fitted by train script, or not used if 'none' or 'log1p'

    effective_image_folder = config_obj.data.image_folder
    if hasattr(config_obj.data, 'offline_image_compression') and \
       config_obj.data.offline_image_compression.enabled and \
       hasattr(config_obj.data, 'processed_image_destination_folder') and \
       config_obj.data.processed_image_destination_folder:
        effective_image_folder = config_obj.data.processed_image_destination_folder

    shared_image_cache_eval = None
    if config_obj.data.cache_processed_images: # Use the flag from config
        cache_config_eval = config_obj.data.image_cache_config
        cache_directory_path_eval = Path(cache_config_eval.cache_directory)

        print(f"Initializing SharedImageCache for evaluation:")
        print(f"  Strategy: {cache_config_eval.strategy}")
        print(f"  Cache directory: {cache_directory_path_eval}")

        shared_image_cache_eval = SharedImageCache(
            cache_path=cache_directory_path_eval,
            max_memory_items=cache_config_eval.max_memory_items, # Can be smaller for eval if memory is tight
            strategy=cache_config_eval.strategy
        )
        shared_image_cache_eval.load_from_disk() # Crucial: loads metadata about disk cache
        shared_image_cache_eval.print_stats()
    else:
        print("Image caching is disabled for evaluation.")

    # This dataset is primarily for fitting global encoders and providing item metadata
    dataset_for_encoders = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init, # Use full interactions here
        item_info_df=item_info_df,
        image_folder=effective_image_folder,
        vision_model_name=config_obj.model.vision_model,
        language_model_name=config_obj.model.language_model,
        create_negative_samples=False, # Not needed for encoder fitting context
        numerical_feat_cols=config_obj.data.numerical_features_cols,
        numerical_normalization_method=config_obj.data.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        cache_processed_images=config_obj.data.cache_processed_images, # Use config for caching
        shared_image_cache=shared_image_cache_eval
    )
    dataset_for_encoders.finalize_setup() # This fits the encoders

    encoders_dir = Path(config_obj.checkpoint_dir) / 'encoders'
    print(f"Loading encoders from {encoders_dir}...")
    try:
        with open(encoders_dir / 'user_encoder.pkl', 'rb') as f:
            dataset_for_encoders.user_encoder = pickle.load(f)
        with open(encoders_dir / 'item_encoder.pkl', 'rb') as f:
            dataset_for_encoders.item_encoder = pickle.load(f)
        dataset_for_encoders.n_users = len(dataset_for_encoders.user_encoder.classes_)
        dataset_for_encoders.n_items = len(dataset_for_encoders.item_encoder.classes_)
    except FileNotFoundError:
        print(f"Error: Encoders not found in {encoders_dir}. Run training or extract_encoders.py.")
        sys.exit(1)
    print(f"Loaded encoders: {dataset_for_encoders.n_users} users, {dataset_for_encoders.n_items} items.")

    model_instance_multimodal = None # Renamed to avoid confusion
    if args.recommender_type == 'multimodal':
        print("Initializing multimodal model...")
        model_class_to_use = PretrainedMultimodalRecommender
        if hasattr(config_obj.model, 'model_class') and config_obj.model.model_class == 'enhanced':
            model_class_to_use = EnhancedMultimodalRecommender
        
        model_params = {
            'n_users': dataset_for_encoders.n_users,
            'n_items': dataset_for_encoders.n_items,
            'embedding_dim': config_obj.model.embedding_dim,
            'vision_model_name': config_obj.model.vision_model,
            'language_model_name': config_obj.model.language_model,
            'use_contrastive': config_obj.model.use_contrastive, # Keep as per config
            'dropout_rate': 0.0, # Typically 0 for eval
             # Add other necessary model_config parameters from your Config dataclass
            'freeze_vision': config_obj.model.freeze_vision,
            'freeze_language': config_obj.model.freeze_language,
            'num_attention_heads': config_obj.model.num_attention_heads,
            'attention_dropout': 0.0, # Typically 0 for eval
            'fusion_hidden_dims': config_obj.model.fusion_hidden_dims,
            'fusion_activation': config_obj.model.fusion_activation,
            'use_batch_norm': config_obj.model.use_batch_norm,
            'projection_hidden_dim': config_obj.model.projection_hidden_dim,
            'final_activation': config_obj.model.final_activation,
            'init_method': config_obj.model.init_method,
            'contrastive_temperature': config_obj.model.contrastive_temperature,
        }
        if model_class_to_use == EnhancedMultimodalRecommender:
             model_params.update({
                'use_cross_modal_attention': config_obj.model.use_cross_modal_attention,
                'cross_modal_attention_weight': config_obj.model.cross_modal_attention_weight
            })

        model_instance_multimodal = model_class_to_use(**model_params).to(device)
        
        checkpoint_path_str = 'best_model.pth' # Default
        if hasattr(config_obj, 'eval_checkpoint_name'): # If you add this to config
            checkpoint_path_str = config_obj.eval_checkpoint_name
            
        checkpoint_path = Path(config_obj.checkpoint_dir) / checkpoint_path_str
        if not checkpoint_path.exists():
             checkpoint_path = Path(config_obj.checkpoint_dir) / 'final_model.pth' # Fallback
        if not checkpoint_path.exists():
            print(f"Error: Multimodal model checkpoint not found at {Path(config_obj.checkpoint_dir) / checkpoint_path_str} or final_model.pth.")
            sys.exit(1)
        
        print(f"Loading multimodal model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_instance_multimodal.load_state_dict(checkpoint['model_state_dict'])
        model_instance_multimodal.eval()

    # Create recommender instance, passing train_df as history_interactions_df
    # The 'dataset_for_encoders' is passed to baselines for access to global item lists, encoders, etc.
    # The 'train_df' (if provided) is passed as 'history_interactions_df' to restrict their "seen" history.
    recommender_instance = create_recommender(
        args.recommender_type,
        dataset=dataset_for_encoders, # Baselines use this for item lists, encoders
        model=model_instance_multimodal, # Only for multimodal type
        device=device,
        history_interactions_df=train_df, # This is the crucial change for baselines
        config_obj=config_obj 
    )

    print("\nStarting evaluation...")
    
    results = {}
    
    task_map = {
        'retrieval': EvaluationTask.TOP_K_RETRIEVAL,
        'ranking': EvaluationTask.TOP_K_RANKING,
        'next_item': EvaluationTask.NEXT_ITEM_PREDICTION,
        'cold_user': EvaluationTask.COLD_START_USER,
        'cold_item': EvaluationTask.COLD_START_ITEM,
        'beyond_accuracy': EvaluationTask.BEYOND_ACCURACY
    }
    eval_task_enum_val = task_map.get(args.eval_task)
    if not eval_task_enum_val:
        raise ValueError(f"Unknown eval_task: {args.eval_task}")
    
    print(f"Creating evaluator for task: {eval_task_enum_val.value}")
    # The evaluator needs train_df to define "novelty" correctly.
    evaluator = create_evaluator(
        task=eval_task_enum_val,
        recommender=recommender_instance,
        test_data=test_df,
        config=config_obj, # Pass the full config object
        train_data=train_df # Crucial for defining novelty and for evaluators like ColdStart
    )
    
    print(f"Task: {evaluator.task_name}")
    print(f"Filter seen items (evaluator's perspective for ground truth): {evaluator.filter_seen}")
    
    results = evaluator.evaluate()
    results['evaluation_metadata'] = {
        'task': evaluator.task_name,
        'filter_seen_evaluator_perspective': evaluator.filter_seen,
        'recommender_type': args.recommender_type,
        'top_k': config_obj.recommendation.top_k, # Assuming this is in your main config
        'test_file': args.test_data,
        'train_file_used_for_history': args.train_data if args.train_data else "None (baselines used full dataset history via dataset_for_encoders)"
    }
    evaluator.print_summary(results)

    output_path = Path(config_obj.results_dir) / args.output 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        # Custom serializer for numpy types if they appear
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        json.dump(results, f, indent=2, cls=NpEncoder)

    print(f"\nEvaluation results saved to {output_path}")

if __name__ == '__main__':
    main()