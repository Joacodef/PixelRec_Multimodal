# scripts/evaluate.py - Simplified without cross-modal attention
#!/usr/bin/env python
"""
Evaluate model performance with simplified architecture
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

# Add parent directory
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

# Use simplified cache instead of old cache system
try:
    from src.data.simple_cache import SimpleFeatureCache
except ImportError:
    # Create a minimal cache if the file doesn't exist
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


def create_recommender(
    recommender_type: str, 
    dataset: MultimodalDataset, 
    model: Optional[torch.nn.Module] = None, 
    device: Optional[torch.device] = None,    
    history_interactions_df: Optional[pd.DataFrame] = None,
    config_obj: Optional[Config] = None,
    simple_cache_instance: Optional[SimpleFeatureCache] = None
):
    """Create the specified type of recommender with simplified cache"""

    k_neighbors_val = 50 
    if config_obj and hasattr(config_obj, 'model') and hasattr(config_obj.model, 'k_neighbors'): 
        k_neighbors_val = config_obj.model.k_neighbors

    if recommender_type == 'multimodal':
        if model is None:
            raise ValueError("Model required for multimodal recommender")
        if config_obj is None:
            raise ValueError("Config object required for multimodal recommender to determine num_numerical_features")
        
        # Pass config_obj to the Recommender which will instantiate the MultimodalRecommender
        # The Recommender itself will need to be modified or it needs to pass num_numerical_features
        # For now, assuming the Recommender class itself might handle passing this to the model it instantiates.
        # If Recommender directly takes a pre-initialized model (as it seems to do),
        # then the model must be initialized with num_numerical_features *before* calling create_recommender.
        return MultimodalModelRecommender( # This is src.inference.recommender.Recommender
            model, 
            dataset, 
            device,
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
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.clip.modeling_clip")
    
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument('--config', type=str, default='configs/simple_config.yaml', help='Path to configuration file')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data CSV file')
    parser.add_argument('--train_data', type=str, help='Path to training data CSV file')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Path to save evaluation results JSON file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation')
    parser.add_argument('--recommender_type', type=str, default='multimodal', choices=['multimodal', 'random', 'popularity', 'item_knn', 'user_knn'], help='Type of recommender to evaluate')
    parser.add_argument('--eval_task', type=str, default='retrieval', choices=['retrieval', 'ranking'], help='Evaluation task to perform')
    parser.add_argument('--save_predictions', type=str, default=None, help='Path to save user-level predictions JSON file (relative to results_dir)')
    parser.add_argument('--warmup_recommender_cache', action='store_true', help="Enable warm-up of the Recommender's cache")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for evaluation')
    parser.add_argument('--use_parallel', action='store_true', help='Use parallel processing for multimodal evaluation')
    parser.add_argument('--parallel_chunk_size', type=int, default=5000, help='Chunk size for parallel processing')
    parser.add_argument('--use_sampling', action='store_true', default=True, help='Use negative sampling for evaluation (much faster)')
    parser.add_argument('--no_sampling', dest='use_sampling', action='store_false', help='Disable negative sampling (use full evaluation)')
    parser.add_argument('--num_negatives', type=int, default=100, help='Number of negative samples per positive item (default: 100)')
    parser.add_argument('--sampling_strategy', type=str, default='random', choices=['random', 'popularity', 'popularity_inverse'], help='Strategy for negative sampling')

    args = parser.parse_args()

    config_obj = Config.from_yaml(args.config)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    # Remove complex task validation
    train_df = None
    if args.train_data:
        print(f"Loading training data from {args.train_data}...")
        train_df = pd.read_csv(args.train_data)
    else:
        if args.eval_task == 'retrieval':
            print(f"Note: --train_data not provided for '{args.eval_task}' task. Using filter_seen=False.")

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

    # Initialize simplified cache using new config structure
    simple_cache_instance = None
    cache_config = config_obj.data.cache_config
    
    if cache_config.enabled:
        print(f"Initializing SimpleFeatureCache:")
        print(f"  Max memory items: {cache_config.max_memory_items}")
        print(f"  Cache directory: {cache_config.cache_directory}")
        print(f"  Use disk: {cache_config.use_disk}")
        
        simple_cache_instance = SimpleFeatureCache(
            vision_model=config_obj.model.vision_model,
            language_model=config_obj.model.language_model,
            base_cache_dir=cache_config.cache_directory,
            max_memory_items=cache_config.max_memory_items,
            use_disk=cache_config.use_disk
        )
        simple_cache_instance.print_stats()
    else:
        print("Feature caching is disabled")

    # Create dataset with simplified cache parameters using new config structure
    cache_config = config_obj.data.cache_config
    dataset_for_encoders = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init,
        item_info_df=item_info_df,
        image_folder=effective_image_folder,
        vision_model_name=config_obj.model.vision_model,
        language_model_name=config_obj.model.language_model,
        create_negative_samples=False,
        # Simplified cache parameters using new config structure
        cache_features=cache_config.enabled,
        cache_max_items=cache_config.max_memory_items,
        cache_dir=cache_config.cache_directory,
        cache_to_disk=cache_config.use_disk,
        # Keep some existing parameters for compatibility
        numerical_feat_cols=config_obj.data.numerical_features_cols,
        numerical_normalization_method=config_obj.data.numerical_normalization_method,
        numerical_scaler=numerical_scaler
    )
    
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
        
        # Simplified model initialization - no more complex model class selection
        print("Using MultimodalRecommender")
        
        model_params = {
            'n_users': dataset_for_encoders.n_users,
            'n_items': dataset_for_encoders.n_items,
            'num_numerical_features': len(config_obj.data.numerical_features_cols), # Pass the correct number of features
            'embedding_dim': config_obj.model.embedding_dim,
            'vision_model_name': config_obj.model.vision_model,
            'language_model_name': config_obj.model.language_model,
            'use_contrastive': config_obj.model.use_contrastive,
            'dropout_rate': 0.0,  # No dropout during evaluation
            'freeze_vision': config_obj.model.freeze_vision,
            'freeze_language': config_obj.model.freeze_language,
            'num_attention_heads': config_obj.model.num_attention_heads,
            'attention_dropout': 0.0,  # No dropout during evaluation
            'fusion_hidden_dims': config_obj.model.fusion_hidden_dims,
            'fusion_activation': config_obj.model.fusion_activation,
            'use_batch_norm': config_obj.model.use_batch_norm,
            'projection_hidden_dim': config_obj.model.projection_hidden_dim,
            'final_activation': config_obj.model.final_activation,
            'init_method': config_obj.model.init_method,
            'contrastive_temperature': config_obj.model.contrastive_temperature,
        }

        model_instance_multimodal = MultimodalRecommender(**model_params).to(device)
        
        checkpoint_path_str = 'best_model.pth' 
        if hasattr(config_obj, 'eval_checkpoint_name'): 
            checkpoint_path_str = config_obj.eval_checkpoint_name
            
        checkpoint_path = Path(config_obj.checkpoint_dir) / checkpoint_path_str
        if not checkpoint_path.exists():
             checkpoint_path = Path(config_obj.checkpoint_dir) / 'final_model.pth' 
        if not checkpoint_path.exists():
            print(f"Error: Multimodal model checkpoint not found.")
            sys.exit(1)
        
        print(f"Loading multimodal model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_instance_multimodal.load_state_dict(checkpoint['model_state_dict'])
        model_instance_multimodal.eval()

    recommender_instance = create_recommender(
        args.recommender_type,
        dataset=dataset_for_encoders, 
        model=model_instance_multimodal, # Pass the already initialized model
        device=device,
        history_interactions_df=train_df, 
        config_obj=config_obj,
        simple_cache_instance=simple_cache_instance
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

    # Warm up cache if requested
    if args.recommender_type == 'multimodal' and args.warmup_recommender_cache:
        print("Warming up Recommender's cache...")
        
        if hasattr(dataset_for_encoders, 'item_encoder') and \
           hasattr(dataset_for_encoders.item_encoder, 'classes_') and \
           dataset_for_encoders.item_encoder.classes_ is not None:
            
            all_item_ids_for_warmup = dataset_for_encoders.item_encoder.classes_
            print(f"Warming up cache for {len(all_item_ids_for_warmup)} items.")
            
            # Simple cache warmup - just call _get_item_features for each item
            for item_id_to_warm in tqdm(all_item_ids_for_warmup[:1000], desc="Warming cache"):  # Limit to 1000 for speed
                try:
                    if hasattr(recommender_instance, '_get_item_features'):
                        recommender_instance._get_item_features(item_id_to_warm) 
                except Exception as e:
                    if all_item_ids_for_warmup.tolist().index(item_id_to_warm) < 5:
                        print(f"Warning: Error warming cache for item {item_id_to_warm}: {e}")
            
            # Print cache statistics
            print(f"Cache warming completed")
            if hasattr(recommender_instance, 'print_cache_stats'):
                recommender_instance.print_cache_stats()
            elif simple_cache_instance:
                simple_cache_instance.print_stats()
        else:
            print("Warning: Could not get all item IDs to warm up cache.")

    if args.use_parallel and args.recommender_type != 'multimodal':
        print("Warning: Parallel mode only supported for multimodal recommender. Ignoring --use_parallel.")
        args.use_parallel = False

    print("\nStarting evaluation...")
    
    # Simplified task mapping
    task_map = {
        'retrieval': EvaluationTask.TOP_K_RETRIEVAL,
        'ranking': EvaluationTask.TOP_K_RANKING
    }
    eval_task_enum_val = task_map.get(args.eval_task)
    if not eval_task_enum_val:
        raise ValueError(f"Unknown eval_task: {args.eval_task}. Available: {list(task_map.keys())}")
    
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
        sampling_strategy=args.sampling_strategy,
        num_workers=args.num_workers
    )
        
    print(f"Task: {evaluator.task_name}")
    print(f"Filter seen items: {evaluator.filter_seen}")
    
    results = evaluator.evaluate()

    # Save predictions if requested
    if args.save_predictions:
        if 'predictions' in results:
            predictions_data = results.pop('predictions')
            
            # Ensure path is relative to results_dir
            save_path = Path(config_obj.results_dir) / args.save_predictions
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to a serializable format (list of dicts)
            serializable_predictions = {}
            for user, recs in predictions_data.items():
                if isinstance(recs, list) and all(isinstance(rec, tuple) for rec in recs):
                    serializable_predictions[str(user)] = [{'item_id': str(item), 'score': float(score)} for item, score in recs]
                else:
                    serializable_predictions[str(user)] = recs

            with open(save_path, 'w') as f:
                json.dump(serializable_predictions, f, indent=2)
            print(f"\n✓ User-level predictions saved to {save_path}")
        else:
            print("\n⚠️  Warning: --save_predictions was specified, but the evaluator did not return predictions.")

    results['evaluation_metadata'] = {
        'task': evaluator.task_name,
        'filter_seen_evaluator_perspective': evaluator.filter_seen,
        'recommender_type': args.recommender_type,
        'top_k': config_obj.recommendation.top_k, 
        'test_file': args.test_data,
        'train_file_used_for_history': args.train_data if args.train_data else "None",
        'cache_warmup_enabled': args.warmup_recommender_cache,
        'parallel_evaluation': args.use_parallel,
        'num_workers': args.num_workers if args.use_parallel else 0,
        'parallel_chunk_size': args.parallel_chunk_size if args.use_parallel else 0,
        'simplified_architecture': True  # New metadata field
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