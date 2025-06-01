#!/usr/bin/env python
"""
Evaluate model performance
"""
import argparse
import torch
import pandas as pd
import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score # These are not used
from pathlib import Path
import json
import pickle # Added for loading encoders

# Add parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.inference.recommender import Recommender
# from src.evaluation.metrics import calculate_ndcg, calculate_map # These are not used in the current evaluate_recommendations

def evaluate_recommendations(recommender, test_data, config):
    """Evaluate recommendation quality"""
    metrics = {
        'precision_at_k': [], # Renamed for clarity
        'recall_at_k': [],    # Renamed for clarity
        'ndcg_at_k': [],      # Added NDCG
        # 'map_at_k': [], # MAP is typically calculated over all relevant items, not just top-k
        'coverage': set(),
        # 'diversity': [] # Diversity calculation would require item embeddings
    }
    
    # Ensure top_k is an integer
    top_k = int(config.recommendation.top_k)

    # Use a limited number of users for evaluation if specified, otherwise all users in test_data
    # Limiting to 100 users for efficiency as in the original file, can be changed.
    unique_users_in_test = test_data['user_id'].unique()
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
            filter_seen=True # Assuming training data might be implicitly known by recommender if not using a strict test set
        )
        
        if not recommendations:
            metrics['precision_at_k'].append(0.0)
            metrics['recall_at_k'].append(0.0)
            metrics['ndcg_at_k'].append(0.0)
            continue
            
        recommended_item_ids = [item_id for item_id, _ in recommendations]
        
        # Calculate metrics
        # Relevance score for each recommended item (1 if relevant, 0 otherwise)
        relevance_scores = [1 if item_id in ground_truth_items else 0 for item_id in recommended_item_ids]
        
        # Precision@k
        if recommended_item_ids: # Avoid division by zero
            precision = sum(relevance_scores) / len(recommended_item_ids)
        else:
            precision = 0.0
        metrics['precision_at_k'].append(precision)
        
        # Recall@k
        if ground_truth_items: # Avoid division by zero
            recall = sum(relevance_scores) / len(ground_truth_items)
        else:
            recall = 0.0
        metrics['recall_at_k'].append(recall)

        # NDCG@k
        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, rel_score in enumerate(relevance_scores):
            dcg += rel_score / np.log2(i + 2) # i+2 because ranks are 1-based, log is 0-indexed
        
        # IDCG (Ideal Discounted Cumulative Gain)
        ideal_relevance_scores = sorted(relevance_scores, reverse=True) # In this binary case, it's just number of hits
        idcg = 0.0
        for i, rel_score in enumerate(ideal_relevance_scores[:top_k]): # Consider only top_k for ideal order
             idcg += rel_score / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics['ndcg_at_k'].append(ndcg)
        
        # Coverage: add recommended items to the set of all recommended items
        metrics['coverage'].update(recommended_item_ids)
    
    # Aggregate metrics
    # Calculate mean of list-based metrics, handling cases where no recommendations were made for any user
    mean_precision = np.mean(metrics['precision_at_k']) if metrics['precision_at_k'] else 0.0
    mean_recall = np.mean(metrics['recall_at_k']) if metrics['recall_at_k'] else 0.0
    mean_ndcg = np.mean(metrics['ndcg_at_k']) if metrics['ndcg_at_k'] else 0.0
    
    # Calculate overall catalog coverage
    # Total unique items in the test set that could have been recommended
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
    config = Config.from_yaml(args.config)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    # --- Initialize model and recommender ---
    print("Loading item info and interaction data for dataset initialization...")
    item_info_df = pd.read_csv(config.data.item_info_path)
    
    # For dataset n_users/n_items initialization, it's good to use the same interactions file as in training,
    # or ensure encoders handle all users/items. Here, we load encoders later.
    # We use a minimal interactions_df for dataset creation if encoders are to be loaded.
    # Alternatively, if full training interactions are available, use them.
    # For simplicity and consistency with generate_recommendations.py, we can load the configured interactions.
    interactions_df_for_dataset_init = pd.read_csv(config.data.interactions_path)
    if config.data.sample_size: # Apply sampling if configured, similar to training script
        interactions_df_for_dataset_init = interactions_df_for_dataset_init.sample(
            n=min(config.data.sample_size, len(interactions_df_for_dataset_init)),
            random_state=42 # Consistent random state
        )

    print("Creating dataset instance...")
    dataset = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init, # Used for initial setup
        item_info_df=item_info_df,
        image_folder=config.data.image_folder,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        create_negative_samples=False # No negative sampling needed for evaluation setup
    )
    
    # Load encoders
    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    print(f"Loading encoders from {encoders_dir}...")
    try:
        with open(encoders_dir / 'user_encoder.pkl', 'rb') as f:
            dataset.user_encoder = pickle.load(f)
        with open(encoders_dir / 'item_encoder.pkl', 'rb') as f:
            dataset.item_encoder = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Encoders not found in {encoders_dir}. Make sure training was completed and encoders were saved.")
        sys.exit(1)

    # Update n_users and n_items from loaded encoders, as the model was trained with these
    dataset.n_users = len(dataset.user_encoder.classes_)
    dataset.n_items = len(dataset.item_encoder.classes_)
    print(f"Loaded encoders: {dataset.n_users} users, {dataset.n_items} items.")

    print("Initializing model...")
    model = PretrainedMultimodalRecommender(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=config.model.embedding_dim,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        freeze_vision=config.model.freeze_vision, # Usually true for eval if not fine-tuned
        freeze_language=config.model.freeze_language, # Usually true for eval
        use_contrastive=config.model.use_contrastive,
        dropout_rate=0.0 # Dropout is typically disabled during evaluation
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth' # Or 'final_model.pth'
    if not checkpoint_path.exists():
        checkpoint_path = Path(config.checkpoint_dir) / 'final_model.pth' # Try final if best doesn't exist
        if not checkpoint_path.exists():
            print(f"Error: Model checkpoint not found at {Path(config.checkpoint_dir) / 'best_model.pth'} or {checkpoint_path}.")
            sys.exit(1)
            
    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode
    
    print("Initializing recommender...")
    recommender = Recommender(
        model=model,
        dataset=dataset, # The dataset here provides access to item info, image paths, encoders etc.
        device=device
    )
    # --- End of model and recommender initialization ---
    
    print("\nStarting evaluation...")
    results = evaluate_recommendations(recommender, test_df, config)
    
    # Ensure output directory exists
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