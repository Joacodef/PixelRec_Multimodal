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

# Add parent directory
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.inference.recommender import Recommender
# Import metric calculation functions from the source module
from src.evaluation.metrics import calculate_precision_at_k, calculate_recall_at_k, calculate_ndcg
from src.inference.baseline_recommenders import (
    RandomRecommender, PopularityRecommender, 
    ItemKNNRecommender, UserKNNRecommender
)

def create_recommender(recommender_type, dataset, model=None, device=None):
    """Create the specified type of recommender"""
    
    if recommender_type == 'multimodal':
        if model is None:
            raise ValueError("Model required for multimodal recommender")
        from src.inference.recommender import Recommender
        return Recommender(model, dataset, device)
    
    elif recommender_type == 'random':
        return RandomRecommender(dataset, device)
    
    elif recommender_type == 'popularity':
        return PopularityRecommender(dataset, device)
    
    elif recommender_type == 'item_knn':
        return ItemKNNRecommender(dataset, device)
    
    elif recommender_type == 'user_knn':
        return UserKNNRecommender(dataset, device)
    
    else:
        raise ValueError(f"Unknown recommender type: {recommender_type}")


def evaluate_recommendations(recommender, test_data, config):
    """Evaluate recommendation quality"""
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],    
        'ndcg_at_k': [],      
        'coverage': set(),
    }
    
    top_k = int(config.recommendation.top_k)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get unique users and items
    unique_users_in_test = test_data['user_id'].unique()
    unique_items = test_data['item_id'].unique()
    
    # Configuration for efficiency
    max_candidates = min(500, len(unique_items))  # Limit candidates per user
    batch_size = 256  # Larger batch size for scoring
    
    # Pre-cache all item features for efficiency
    print(f"Pre-caching features for {len(unique_items)} items...")
    items_to_cache = unique_items if len(unique_items) <= 1000 else np.random.choice(unique_items, 1000, replace=False)
    for item in tqdm(items_to_cache, desc="Caching item features"):
        _ = recommender._get_item_features(item)
    print(f"Cached {len(recommender.item_features_cache)} item features")
    
    # Monkey-patch the recommender to use larger batch size
    if hasattr(recommender, '_score_items'):
        original_score_items = recommender._score_items
        def _score_items_with_larger_batch(user_idx, items, batch_size=256):
            return original_score_items(user_idx, items, batch_size)
        recommender._score_items = _score_items_with_larger_batch
    
    # Limit evaluation to a subset of users for efficiency
    max_users_to_evaluate = min(100, len(unique_users_in_test))
    if len(unique_users_in_test) > max_users_to_evaluate:
        print(f"Sampling {max_users_to_evaluate} users from {len(unique_users_in_test)} for evaluation")
        users_to_evaluate = np.random.choice(unique_users_in_test, max_users_to_evaluate, replace=False)
    else:
        users_to_evaluate = unique_users_in_test
    
    print(f"Evaluating on {len(users_to_evaluate)} users with top_k={top_k}...")
    
    # Pre-select a pool of candidate items that will be used for all users
    # This ensures more consistent evaluation and better use of caching
    global_candidate_pool = np.random.choice(
        unique_items,
        size=min(max_candidates * 2, len(unique_items)),  # Larger pool for diversity
        replace=False
    ).tolist()
    
    for user_id in tqdm(users_to_evaluate, desc="Evaluating users"):
        # Get ground truth relevant items for the user
        ground_truth_items = set(
            test_data[test_data['user_id'] == user_id]['item_id']
        )
        
        if not ground_truth_items:
            continue
        
        # Select candidates for this user from the global pool
        # Mix some ground truth items with random candidates for better evaluation
        user_candidates = list(ground_truth_items)  # Include ground truth
        
        # Add random items from the global pool
        additional_candidates = [item for item in global_candidate_pool if item not in ground_truth_items]
        n_additional = min(max_candidates - len(user_candidates), len(additional_candidates))
        if n_additional > 0:
            user_candidates.extend(np.random.choice(additional_candidates, n_additional, replace=False))
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            user_id,
            top_k=top_k,
            filter_seen=True,
            candidates=user_candidates
        )
        
        if not recommendations:
            metrics['precision_at_k'].append(0.0)
            metrics['recall_at_k'].append(0.0)
            metrics['ndcg_at_k'].append(0.0)
            continue
        
        recommended_item_ids = [item_id for item_id, _ in recommendations]
        
        # Calculate metrics using imported functions
        precision = calculate_precision_at_k(recommended_item_ids, ground_truth_items, top_k)
        metrics['precision_at_k'].append(precision)
        
        recall = calculate_recall_at_k(recommended_item_ids, ground_truth_items, top_k)
        metrics['recall_at_k'].append(recall)
        
        ndcg = calculate_ndcg(recommended_item_ids, ground_truth_items, top_k)
        metrics['ndcg_at_k'].append(ndcg)
        
        # Update coverage with recommended items
        metrics['coverage'].update(recommended_item_ids)
    
    # Aggregate metrics
    mean_precision = np.mean(metrics['precision_at_k']) if metrics['precision_at_k'] else 0.0
    mean_recall = np.mean(metrics['recall_at_k']) if metrics['recall_at_k'] else 0.0
    mean_ndcg = np.mean(metrics['ndcg_at_k']) if metrics['ndcg_at_k'] else 0.0
    
    all_possible_items_in_test = test_data['item_id'].unique()
    catalog_coverage = len(metrics['coverage']) / len(all_possible_items_in_test) if all_possible_items_in_test.size > 0 else 0.0
    
    results = {
        f'precision_at_{top_k}': mean_precision,
        f'recall_at_{top_k}': mean_recall,
        f'ndcg_at_{top_k}': mean_ndcg,
        'catalog_coverage': catalog_coverage,
        'n_users_evaluated': len(users_to_evaluate),
        'total_items_recommended_unique': len(metrics['coverage']),
        'n_candidates_per_user': max_candidates,
        'items_in_cache': len(recommender.item_features_cache)
    }
    
    return results

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
        '--output', 
        type=str, 
        default='results/evaluation_results.json',
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
        
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config) 
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    # Initialize model and recommender
    print("Loading item info and interaction data for dataset initialization...")
    item_info_df = pd.read_csv(config.data.processed_item_info_path)
    interactions_df_for_dataset_init = pd.read_csv(config.data.processed_interactions_path)

    if config.data.sample_size: 
        interactions_df_for_dataset_init = interactions_df_for_dataset_init.sample(
            n=min(config.data.sample_size, len(interactions_df_for_dataset_init)),
            random_state=42 
        )

    print("Creating dataset instance...")
    
    # Load numerical scaler if needed
    numerical_scaler = None
    if config.data.numerical_normalization_method in ['standardization', 'min_max']:
        scaler_path = Path(config.data.scaler_path)
        if scaler_path.exists():
            print(f"Loading numerical scaler from {scaler_path}...")
            with open(scaler_path, 'rb') as f:
                numerical_scaler = pickle.load(f)
        else:
            print(f"Warning: Numerical scaler not found at {scaler_path}. Proceeding without scaling.")
    
    dataset = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init, 
        item_info_df=item_info_df,
        image_folder=config.data.image_folder,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        create_negative_samples=False,
        numerical_feat_cols=config.data.numerical_features_cols,
        numerical_normalization_method=config.data.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        cache_processed_images=False  # Disable caching for evaluation
    )
    
    # Load encoders
    encoders_dir = Path(config.checkpoint_dir) / 'encoders' #
    print(f"Loading encoders from {encoders_dir}...")
    try:
        with open(encoders_dir / 'user_encoder.pkl', 'rb') as f:
            dataset.user_encoder = pickle.load(f)
        with open(encoders_dir / 'item_encoder.pkl', 'rb') as f:
            dataset.item_encoder = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Encoders not found in {encoders_dir}. Make sure training was completed and encoders were saved.")
        sys.exit(1)

    dataset.n_users = len(dataset.user_encoder.classes_)
    dataset.n_items = len(dataset.item_encoder.classes_)
    print(f"Loaded encoders: {dataset.n_users} users, {dataset.n_items} items.")

    if args.recommender_type == 'multimodal':
        print("Initializing model...")
        model = PretrainedMultimodalRecommender(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embedding_dim=config.model.embedding_dim, 
            vision_model_name=config.model.vision_model, 
            language_model_name=config.model.language_model, 
            freeze_vision=config.model.freeze_vision, 
            freeze_language=config.model.freeze_language, 
            use_contrastive=config.model.use_contrastive, 
            dropout_rate=0.0 
        ).to(device) 
        
        # Load checkpoint
        checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth' 
        if not checkpoint_path.exists():
            checkpoint_path = Path(config.checkpoint_dir) / 'final_model.pth' 
            if not checkpoint_path.exists():
                print(f"Error: Model checkpoint not found at {Path(config.checkpoint_dir) / 'best_model.pth'} or {checkpoint_path}.")
                sys.exit(1)
                
        print(f"Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() 
        
        print("Initializing recommender...")
        recommender = Recommender(
            model=model,
            dataset=dataset, 
            device=device
        ) 
    else:
        # For baselines, we don't need to load a model
        model = None
        recommender = create_recommender(args.recommender_type, dataset, None, device)
        
    print("\nStarting evaluation...")
    results = evaluate_recommendations(recommender, test_df, config)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {output_path}")
    print("--- Evaluation Summary ---")
    print(json.dumps(results, indent=2))
    print("--------------------------")

if __name__ == '__main__':
    main()