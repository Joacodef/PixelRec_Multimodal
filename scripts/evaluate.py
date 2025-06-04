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
# Import Recommender specifically for type checking if needed
from src.inference.recommender import Recommender as MultimodalRecommender
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
        # Use the specific import for clarity
        return MultimodalRecommender(model, dataset, device)

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
    """Evaluate recommendation quality without performance limits on scope."""
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'ndcg_at_k': [],
        'coverage': set(),
    }

    top_k = int(config.recommendation.top_k)

    # Set random seed for reproducibility (good practice for consistent runs)
    np.random.seed(42)

    # Get unique users and items from the test data
    unique_users_in_test = test_data['user_id'].unique()
    unique_items_in_test = test_data['item_id'].unique() # All unique items in the test set

    # Batch size for internal scoring loops (multimodal recommender)
    # This is an efficiency parameter for how items are processed, not a limit on what is processed.
    # It was previously hardcoded and used in monkey-patching.
    # Keeping it here for clarity if used internally by the recommender or its setup.
    scoring_batch_size = 256

    # Pre-cache all item features for efficiency (ONLY for multimodal recommender)
    if isinstance(recommender, MultimodalRecommender):
        print(f"Pre-caching features for all {len(unique_items_in_test)} unique items in test data (multimodal recommender)...")
        # Cache features for all unique items present in the test data
        items_to_cache = unique_items_in_test
        for item_id_to_cache in tqdm(items_to_cache, desc="Caching item features"):
            _ = recommender._get_item_features(item_id_to_cache)
        print(f"Cached {len(recommender.item_features_cache)} item features")

        # Monkey-patch the recommender's _score_items method to use the defined scoring_batch_size
        # This was part of the original script's logic for the multimodal case.
        if hasattr(recommender, '_score_items'):
            original_score_items = recommender._score_items
            def _score_items_with_custom_batch(user_idx_arg, items_arg, batch_size_arg=scoring_batch_size):
                return original_score_items(user_idx_arg, items_arg, batch_size_arg)
            recommender._score_items = _score_items_with_custom_batch

    else:
        print(f"Skipping feature pre-caching for {type(recommender).__name__}.")

    # Evaluate on ALL unique users in the test set
    users_to_evaluate = unique_users_in_test
    print(f"Evaluating on all {len(users_to_evaluate)} users in the test set with top_k={top_k}...")

    for user_id in tqdm(users_to_evaluate, desc="Evaluating users"):
        ground_truth_items = set(
            test_data[test_data['user_id'] == user_id]['item_id']
        )

        if not ground_truth_items:
            # Append 0.0 for users with no relevant items in test set, or handle as per desired logic
            metrics['precision_at_k'].append(0.0)
            metrics['recall_at_k'].append(0.0)
            metrics['ndcg_at_k'].append(0.0)
            continue

        # Get recommendations, considering all items in the dataset as candidates (None)
        # The recommender's get_recommendations method should handle `candidates=None`
        # by scoring all relevant items from its internal item set after filtering seen ones.
        recommendations = recommender.get_recommendations(
            user_id,
            top_k=top_k,
            filter_seen=True,
            candidates=None # This tells the recommender to consider all its known items
        )

        if not recommendations:
            metrics['precision_at_k'].append(0.0)
            metrics['recall_at_k'].append(0.0)
            metrics['ndcg_at_k'].append(0.0)
            continue

        recommended_item_ids = [rec_item_id for rec_item_id, _ in recommendations]

        precision = calculate_precision_at_k(recommended_item_ids, ground_truth_items, top_k)
        metrics['precision_at_k'].append(precision)

        recall = calculate_recall_at_k(recommended_item_ids, ground_truth_items, top_k)
        metrics['recall_at_k'].append(recall)

        ndcg = calculate_ndcg(recommended_item_ids, ground_truth_items, top_k)
        metrics['ndcg_at_k'].append(ndcg)

        metrics['coverage'].update(recommended_item_ids)

    mean_precision = np.mean(metrics['precision_at_k']) if metrics['precision_at_k'] else 0.0
    mean_recall = np.mean(metrics['recall_at_k']) if metrics['recall_at_k'] else 0.0
    mean_ndcg = np.mean(metrics['ndcg_at_k']) if metrics['ndcg_at_k'] else 0.0

    # Catalog coverage is calculated against all unique items *in the test set*
    # If you want coverage against the *entire dataset catalog*, use `len(recommender.dataset.item_encoder.classes_)`
    # For now, using unique_items_in_test as per previous context.
    catalog_coverage = len(metrics['coverage']) / len(unique_items_in_test) if len(unique_items_in_test) > 0 else 0.0

    results = {
        f'precision_at_{top_k}': mean_precision,
        f'recall_at_{top_k}': mean_recall,
        f'ndcg_at_{top_k}': mean_ndcg,
        'catalog_coverage': catalog_coverage,
        'n_users_evaluated': len(users_to_evaluate),
        'total_items_recommended_unique': len(metrics['coverage']),
        # 'n_candidates_per_user': 'all' # Max_candidates is no longer a fixed number for candidate generation for get_recommendations
    }
    if isinstance(recommender, MultimodalRecommender):
        results['items_in_cache_after_eval'] = len(recommender.item_features_cache)

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

    config_obj = Config.from_yaml(args.config) # Renamed to avoid conflict
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)

    print("Loading item info and interaction data for dataset initialization...")
    item_info_df = pd.read_csv(config_obj.data.processed_item_info_path)
    interactions_df_for_dataset_init = pd.read_csv(config_obj.data.processed_interactions_path)

    print("Creating dataset instance...")

    numerical_scaler = None
    if config_obj.data.numerical_normalization_method in ['standardization', 'min_max']:
        scaler_path = Path(config_obj.data.scaler_path)
        if scaler_path.exists():
            print(f"Loading numerical scaler from {scaler_path}...")
            with open(scaler_path, 'rb') as f:
                numerical_scaler = pickle.load(f)
        else:
            print(f"Warning: Numerical scaler not found at {scaler_path}. Proceeding without scaling.")

    # Use a consistent image folder path, effective_image_folder
    effective_image_folder = config_obj.data.image_folder
    if hasattr(config_obj.data, 'offline_image_compression') and \
       config_obj.data.offline_image_compression.enabled and \
       hasattr(config_obj.data, 'processed_image_destination_folder') and \
       config_obj.data.processed_image_destination_folder:
        effective_image_folder = config_obj.data.processed_image_destination_folder


    dataset = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init,
        item_info_df=item_info_df,
        image_folder=effective_image_folder, # Use effective image folder
        vision_model_name=config_obj.model.vision_model,
        language_model_name=config_obj.model.language_model,
        create_negative_samples=False, # No negative sampling for evaluation dataset context
        numerical_feat_cols=config_obj.data.numerical_features_cols,
        numerical_normalization_method=config_obj.data.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        cache_processed_images=False # Caching during eval dataset init is not the focus; recommender might do its own
    )
    dataset.finalize_setup() # To fit encoders if not already done by some other means

    encoders_dir = Path(config_obj.checkpoint_dir) / 'encoders'
    print(f"Loading encoders from {encoders_dir}...")
    try:
        with open(encoders_dir / 'user_encoder.pkl', 'rb') as f:
            dataset.user_encoder = pickle.load(f)
        with open(encoders_dir / 'item_encoder.pkl', 'rb') as f:
            dataset.item_encoder = pickle.load(f)
        # Manually update n_users and n_items after loading encoders
        dataset.n_users = len(dataset.user_encoder.classes_)
        dataset.n_items = len(dataset.item_encoder.classes_)

    except FileNotFoundError:
        print(f"Error: Encoders not found in {encoders_dir}. Make sure training was completed and encoders were saved, or run extract_encoders.py.")
        sys.exit(1)


    print(f"Loaded encoders: {dataset.n_users} users, {dataset.n_items} items.")

    model_instance = None # Renamed to avoid conflict
    if args.recommender_type == 'multimodal':
        print("Initializing model...")
        # Determine model class (Pretrained or Enhanced)
        model_class_to_use = MultimodalRecommender # Default to Pretrained
        if hasattr(config_obj.model, 'model_class') and config_obj.model.model_class == 'enhanced':
            from src.models.multimodal import EnhancedMultimodalRecommender
            model_class_to_use = EnhancedMultimodalRecommender
            print("Using EnhancedMultimodalRecommender for evaluation.")
        else:
            # Ensure PretrainedMultimodalRecommender is imported if not default
            from src.models.multimodal import PretrainedMultimodalRecommender
            model_class_to_use = PretrainedMultimodalRecommender
            print("Using PretrainedMultimodalRecommender for evaluation.")

        # Prepare model parameters based on configuration
        model_params = {
            'n_users': dataset.n_users,
            'n_items': dataset.n_items,
            'embedding_dim': config_obj.model.embedding_dim,
            'vision_model_name': config_obj.model.vision_model,
            'language_model_name': config_obj.model.language_model,
            'freeze_vision': config_obj.model.freeze_vision,
            'freeze_language': config_obj.model.freeze_language,
            'use_contrastive': config_obj.model.use_contrastive,
            'dropout_rate': 0.0,  # Typically no dropout during evaluation
            'num_attention_heads': config_obj.model.num_attention_heads,
            'attention_dropout': 0.0, # No dropout
            'fusion_hidden_dims': config_obj.model.fusion_hidden_dims,
            'fusion_activation': config_obj.model.fusion_activation,
            'use_batch_norm': config_obj.model.use_batch_norm, # BN is usually kept for eval
            'projection_hidden_dim': config_obj.model.projection_hidden_dim,
            'final_activation': config_obj.model.final_activation,
            'init_method': config_obj.model.init_method, # Less relevant for loading state_dict
            'contrastive_temperature': config_obj.model.contrastive_temperature
        }
        if model_class_to_use == EnhancedMultimodalRecommender:
             model_params.update({
                'use_cross_modal_attention': config_obj.model.use_cross_modal_attention,
                'cross_modal_attention_weight': config_obj.model.cross_modal_attention_weight
            })

        model_instance = model_class_to_use(**model_params).to(device)

        checkpoint_path = Path(config_obj.checkpoint_dir) / 'best_model.pth'
        if not checkpoint_path.exists():
            checkpoint_path = Path(config_obj.checkpoint_dir) / 'final_model.pth'
            if not checkpoint_path.exists():
                print(f"Error: Model checkpoint not found at {Path(config_obj.checkpoint_dir) / 'best_model.pth'} or {checkpoint_path}.")
                sys.exit(1)

        print(f"Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.eval()

        print("Initializing multimodal recommender...")
        recommender_instance = MultimodalRecommender( # Use the specific alias
            model=model_instance,
            dataset=dataset,
            device=device
        )
    else:
        # For baselines, model_instance remains None
        recommender_instance = create_recommender(args.recommender_type, dataset, None, device) # model_instance is None

    print("\nStarting evaluation...")
    results = evaluate_recommendations(recommender_instance, test_df, config_obj) # Pass config_obj

    output_path = Path(config_obj.results_dir) / args.output # Use config_obj
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to {output_path}")
    print("--- Evaluation Summary ---")
    print(json.dumps(results, indent=2))
    print("--------------------------")

if __name__ == '__main__':
    main()