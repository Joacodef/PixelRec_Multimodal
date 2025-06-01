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

# Add parent directory
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.inference.recommender import Recommender
# Import metric calculation functions from the source module
from src.evaluation.metrics import calculate_precision_at_k, calculate_recall_at_k, calculate_ndcg


def evaluate_recommendations(recommender, test_data, config):
    """Evaluate recommendation quality"""
    metrics = {
        'precision_at_k': [], 
        'recall_at_k': [],    
        'ndcg_at_k': [],      
        'coverage': set(),
    }
    
    top_k = int(config.recommendation.top_k)

    unique_users_in_test = test_data['user_id'].unique()
    # Limit evaluation to a subset of users for efficiency, if necessary
    users_to_evaluate = unique_users_in_test[:100] 
    
    print(f"Evaluating on {len(users_to_evaluate)} users with top_k={top_k}...")

    for user_id in users_to_evaluate:
        # Get ground truth relevant items for the user
        ground_truth_items = set(
            test_data[test_data['user_id'] == user_id]['item_id']
        )
        
        if not ground_truth_items:
            continue

        # Get recommendations from the model
        recommendations = recommender.get_recommendations(
            user_id, 
            top_k=top_k,
            filter_seen=True 
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
        'total_items_recommended_unique': len(metrics['coverage'])
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
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config) #
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    # Initialize model and recommender
    print("Loading item info and interaction data for dataset initialization...")
    item_info_df = pd.read_csv(config.data.item_info_path) #
    
    interactions_df_for_dataset_init = pd.read_csv(config.data.interactions_path) #
    if config.data.sample_size: 
        interactions_df_for_dataset_init = interactions_df_for_dataset_init.sample(
            n=min(config.data.sample_size, len(interactions_df_for_dataset_init)),
            random_state=42 
        )

    print("Creating dataset instance...")
    dataset = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init, 
        item_info_df=item_info_df,
        image_folder=config.data.image_folder, #
        vision_model_name=config.model.vision_model, #
        language_model_name=config.model.language_model, #
        create_negative_samples=False 
    ) #
    
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

    print("Initializing model...")
    model = PretrainedMultimodalRecommender(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=config.model.embedding_dim, #
        vision_model_name=config.model.vision_model, #
        language_model_name=config.model.language_model, #
        freeze_vision=config.model.freeze_vision, 
        freeze_language=config.model.freeze_language, 
        use_contrastive=config.model.use_contrastive, #
        dropout_rate=0.0 
    ).to(device) #
    
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
    ) #
    
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