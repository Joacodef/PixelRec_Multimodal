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
from typing import Optional, Any, Union # Added Any for ProcessedFeatureCache if import fails
import random
import warnings

# Add parent directory
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender, EnhancedMultimodalRecommender 
from src.inference.recommender import Recommender as MultimodalModelRecommender
from src.evaluation.tasks import EvaluationTask, create_evaluator
from src.inference.baseline_recommenders import (
    RandomRecommender, PopularityRecommender,
    ItemKNNRecommender, UserKNNRecommender,
    BaselineRecommender 
)
from src.data.image_cache import SharedImageCache
# Import the new ProcessedFeatureCache
# Adjust the import path if you saved ProcessedFeatureCache elsewhere
try:
    from src.data.feature_cache import ProcessedFeatureCache
except ImportError:
    ProcessedFeatureCache = None # Fallback if not found, will print warning later
    print("Warning: ProcessedFeatureCache class could not be imported from src.data.feature_cache.")
    print("Ensure the file exists and the path is correct for optimal performance.")


def create_recommender(
    recommender_type: str, 
    dataset: MultimodalDataset, 
    model: Optional[torch.nn.Module] = None, 
    device: Optional[torch.device] = None,    
    history_interactions_df: Optional[pd.DataFrame] = None,
    config_obj: Optional[Config] = None,
    # Add parameter for the new cache
    processed_feature_cache_instance: Optional[Any] = None # Use Any for now
) -> Union[BaselineRecommender, MultimodalModelRecommender]: # Use Union for broader compatibility
    """Create the specified type of recommender"""

    k_neighbors_val = 50 
    if config_obj and hasattr(config_obj, 'model') and hasattr(config_obj.model, 'k_neighbors'): 
        k_neighbors_val = config_obj.model.k_neighbors


    if recommender_type == 'multimodal':
        if model is None:
            raise ValueError("Model required for multimodal recommender")
        # Pass the new cache to the MultimodalModelRecommender
        return MultimodalModelRecommender(
            model, 
            dataset, 
            device, 
            processed_feature_cache=processed_feature_cache_instance
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

    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.clip.modeling_clip")
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
        type=str, 
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
        default='retrieval', 
        choices=['retrieval', 'ranking', 'next_item', 'cold_user', 'cold_item', 'beyond_accuracy'],
        help='Evaluation task to perform'
    )
    # Argument to control warm-up of Recommender's L1 cache
    parser.add_argument(
        '--warmup_recommender_l1_cache',
        action='store_true',
        help="Enable warm-up of the Recommender's internal L1 feature cache. Uses more RAM upfront but can be faster if L2 (ProcessedFeatureCache) is disk-based."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel workers for evaluation'
    )
    parser.add_argument(
        '--use_parallel',
        action='store_true',
        help='Use parallel processing for multimodal evaluation'
    )
    parser.add_argument(
        '--parallel_chunk_size',
        type=int,
        default=5000,
        help='Chunk size for parallel processing'
    )

    parser.add_argument(
        '--use_sampling',
        action='store_true',
        default=True,  # Enable by default
        help='Use negative sampling for evaluation (much faster)'
        )
    parser.add_argument(
        '--no_sampling',
        dest='use_sampling',
        action='store_false',
        help='Disable negative sampling (use full evaluation)'
    )
    parser.add_argument(
        '--num_negatives',
        type=int,
        default=100,
        help='Number of negative samples per positive item (default: 100)'
    )
    parser.add_argument(
        '--sampling_strategy',
        type=str,
        default='random',
        choices=['random', 'popularity', 'popularity_inverse'],
        help='Strategy for negative sampling'
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
        if args.eval_task in ['retrieval', 'next_item', 'cold_user', 'cold_item', 'beyond_accuracy']:
            print(f"Warning: --train_data not provided. For '{args.eval_task}' task, evaluation of novelty might be skewed, and baselines will use full dataset history.")


    print("Loading item info and interaction data for dataset initialization...")
    item_info_df = pd.read_csv(config_obj.data.processed_item_info_path)
    interactions_df_for_dataset_init = pd.read_csv(config_obj.data.processed_interactions_path)

    print("Creating dataset instance for global encoders...")
    numerical_scaler = None
    if config_obj.data.numerical_normalization_method in ['standardization', 'min_max']:
        scaler_path = Path(config_obj.data.scaler_path)
        if scaler_path.exists():
            print(f"Loading numerical scaler from {scaler_path}...")
            with open(scaler_path, 'rb') as f:
                numerical_scaler = pickle.load(f)

    effective_image_folder = config_obj.data.image_folder
    if hasattr(config_obj.data, 'offline_image_compression') and \
       config_obj.data.offline_image_compression.enabled and \
       hasattr(config_obj.data, 'processed_image_destination_folder') and \
       config_obj.data.processed_image_destination_folder:
        effective_image_folder = config_obj.data.processed_image_destination_folder

    # Initialize SharedImageCache (for image tensors)
    shared_image_cache_eval = None
    if config_obj.data.cache_processed_images:
        cache_config_eval = config_obj.data.image_cache_config
        cache_directory_path_eval = Path(cache_config_eval.cache_directory)
        print(f"Initializing SharedImageCache for evaluation (image tensors):")
        print(f"  Strategy: {cache_config_eval.strategy}")
        print(f"  Cache directory: {cache_directory_path_eval}")
        shared_image_cache_eval = SharedImageCache(
            cache_path=str(cache_directory_path_eval), # Ensure path is string
            max_memory_items=cache_config_eval.max_memory_items,
            strategy=cache_config_eval.strategy
        )
        shared_image_cache_eval.load_from_disk() 
        shared_image_cache_eval.print_stats()
    else:
        print("SharedImageCache (for image tensors) is disabled for evaluation.")

    # Initialize ProcessedFeatureCache (for non-image features like text tokens, numericals)
    processed_feature_cache_instance = None
    if ProcessedFeatureCache and hasattr(config_obj.data, 'processed_features_cache_config'):
        pfc_config = config_obj.data.processed_features_cache_config
        pfc_cache_dir = Path(pfc_config.cache_directory)
        print(f"Initializing ProcessedFeatureCache for evaluation (non-image features):")
        print(f"  Strategy: {pfc_config.strategy}")
        print(f"  Cache directory: {pfc_cache_dir}")
        processed_feature_cache_instance = ProcessedFeatureCache(
            cache_path=str(pfc_cache_dir), # Ensure path is string
            max_memory_items=pfc_config.max_memory_items,
            strategy=pfc_config.strategy
        )
        processed_feature_cache_instance.load_from_disk_meta() # Scans disk for metadata
        processed_feature_cache_instance.print_stats()
        
        # Optional: Precomputation/Warm-up for ProcessedFeatureCache if configured
        # This would typically be done by a separate script or integrated into train/preprocess.
        # If pfc_config.precompute_at_startup is true, you might add logic here to call
        # a precomputation method for ProcessedFeatureCache if it doesn't have many items on disk.
        # For now, we assume it's pre-populated or will populate on-the-fly.

    else:
        print("ProcessedFeatureCache (for non-image features) is not configured or class not imported.")


    dataset_for_encoders = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init,
        item_info_df=item_info_df,
        image_folder=effective_image_folder,
        vision_model_name=config_obj.model.vision_model,
        language_model_name=config_obj.model.language_model,
        create_negative_samples=False, 
        numerical_feat_cols=config_obj.data.numerical_features_cols,
        numerical_normalization_method=config_obj.data.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        cache_processed_images=config_obj.data.cache_processed_images,
        shared_image_cache=shared_image_cache_eval # For image tensors
    )
    dataset_for_encoders.finalize_setup() 

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

    model_instance_multimodal = None
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
            'use_contrastive': config_obj.model.use_contrastive,
            'dropout_rate': 0.0, 
            'freeze_vision': config_obj.model.freeze_vision,
            'freeze_language': config_obj.model.freeze_language,
            'num_attention_heads': config_obj.model.num_attention_heads,
            'attention_dropout': 0.0,
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
        
        checkpoint_path_str = 'best_model.pth' 
        if hasattr(config_obj, 'eval_checkpoint_name'): 
            checkpoint_path_str = config_obj.eval_checkpoint_name
            
        checkpoint_path = Path(config_obj.checkpoint_dir) / checkpoint_path_str
        if not checkpoint_path.exists():
             checkpoint_path = Path(config_obj.checkpoint_dir) / 'final_model.pth' 
        if not checkpoint_path.exists():
            print(f"Error: Multimodal model checkpoint not found at {Path(config_obj.checkpoint_dir) / checkpoint_path_str} or final_model.pth.")
            sys.exit(1)
        
        print(f"Loading multimodal model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_instance_multimodal.load_state_dict(checkpoint['model_state_dict'])
        model_instance_multimodal.eval()

    recommender_instance = create_recommender(
        args.recommender_type,
        dataset=dataset_for_encoders, 
        model=model_instance_multimodal, 
        device=device,
        history_interactions_df=train_df, 
        config_obj=config_obj,
        processed_feature_cache_instance=processed_feature_cache_instance
    )

    # Set up parallel processing for multimodal recommender if requested
    if args.recommender_type == 'multimodal' and args.use_parallel:
        print(f"Using parallel evaluation with {args.num_workers} workers")
        
        # Check if the parallel method exists
        if not hasattr(recommender_instance, 'get_recommendations_parallel'):
            print("Warning: Parallel recommendation method not implemented. Falling back to sequential.")
            args.use_parallel = False
        else:
            # Override the get_recommendations method to use parallel version
            original_get_recommendations = recommender_instance.get_recommendations
            
            def parallel_get_recommendations(user_id, top_k=10, filter_seen=True, candidates=None):
                return recommender_instance.get_recommendations_parallel(
                    user_id=user_id,
                    top_k=top_k,
                    filter_seen=filter_seen,
                    candidates=candidates,
                    num_workers=args.num_workers,
                    chunk_size=args.parallel_chunk_size
                )
            
            recommender_instance.get_recommendations = parallel_get_recommendations

    # Warm up cache if requested (only do this once)
    if args.recommender_type == 'multimodal' and args.warmup_recommender_l1_cache:
        print("Warming up Recommender's L1 internal item_features_cache...")
        
        if hasattr(dataset_for_encoders, 'item_encoder') and \
           hasattr(dataset_for_encoders.item_encoder, 'classes_') and \
           dataset_for_encoders.item_encoder.classes_ is not None:
            
            all_item_ids_for_warmup = dataset_for_encoders.item_encoder.classes_
            print(f"Attempting to warm up Recommender L1 cache for {len(all_item_ids_for_warmup)} items.")
            
            # Use parallel pre-loading if available and parallel mode is enabled
            if args.use_parallel and hasattr(recommender_instance, 'preload_features_parallel'):
                print("Pre-loading features in parallel...")
                recommender_instance.preload_features_parallel(
                    all_item_ids_for_warmup,
                    num_workers=args.num_workers,
                    chunk_size=1000
                )
            else:
                # Sequential warming
                for item_id_to_warm in tqdm(all_item_ids_for_warmup, desc="Warming Recommender L1 cache"):
                    try:
                        recommender_instance._get_item_features(item_id_to_warm) 
                    except Exception as e:
                        # Only print first few errors to avoid spam
                        if all_item_ids_for_warmup.tolist().index(item_id_to_warm) < 5:
                            print(f"Warning: Error warming L1 cache for item {item_id_to_warm}: {e}")
            
            # Print cache statistics
            print(f"\nCache warming completed:")
            print(f"L1 cache size: {len(recommender_instance.item_features_cache)} items")
            
            if processed_feature_cache_instance:
                print("\nProcessedFeatureCache (L2) stats:")
                processed_feature_cache_instance.print_stats()
            
            if shared_image_cache_eval:
                print("\nSharedImageCache (images) stats:")
                shared_image_cache_eval.print_stats()
        else:
            print("Warning: Could not get all item IDs to warm up Recommender's L1 cache.")

    if args.use_parallel and args.recommender_type != 'multimodal':
        print("Warning: Parallel mode only supported for multimodal recommender. Ignoring --use_parallel.")
        args.use_parallel = False

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

    if args.eval_task == 'retrieval' and args.use_sampling:
        print(f"Using negative sampling: {args.num_negatives} negatives per positive, strategy: {args.sampling_strategy}")

    evaluator = create_evaluator(
        task=eval_task_enum_val,
        recommender=recommender_instance,
        test_data=test_df,
        config=config_obj,
        train_data=train_df,
        use_sampling=args.use_sampling if args.eval_task == 'retrieval' else False,
        num_negatives=args.num_negatives,
        sampling_strategy=args.sampling_strategy
    )
        
    print(f"Task: {evaluator.task_name}")
    print(f"Filter seen items (evaluator's perspective for ground truth): {evaluator.filter_seen}")
    
    results = evaluator.evaluate()
    results['evaluation_metadata'] = {
        'task': evaluator.task_name,
        'filter_seen_evaluator_perspective': evaluator.filter_seen,
        'recommender_type': args.recommender_type,
        'top_k': config_obj.recommendation.top_k, 
        'test_file': args.test_data,
        'train_file_used_for_history': args.train_data if args.train_data else "None",
        'l1_cache_warmup_enabled': args.warmup_recommender_l1_cache,
        'parallel_evaluation': args.use_parallel,
        'num_workers': args.num_workers if args.use_parallel else 0,
        'parallel_chunk_size': args.parallel_chunk_size if args.use_parallel else 0
    }
    evaluator.print_summary(results)

    output_path = Path(config_obj.results_dir) / args.output 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
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