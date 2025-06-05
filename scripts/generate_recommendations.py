#!/usr/bin/env python
"""
Script for generating recommendations
"""
import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import pickle
import json
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender # Alias for MultimodalRecommender
from src.inference.recommender import Recommender


def load_model_and_data(config: Config, device: torch.device):
    """Load trained model and dataset"""
    # Load data
    print("Loading data...")
    item_info_df = pd.read_csv(config.data.processed_item_info_path)
    interactions_df = pd.read_csv(config.data.processed_interactions_path)
    
    # Create dataset (without negative sampling for inference)
    dataset = MultimodalDataset(
        interactions_df,
        item_info_df,
        config.data.image_folder,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        create_negative_samples=False,
        # Pass relevant data config parts for dataset initialization
        numerical_feat_cols=config.data.numerical_features_cols,
        numerical_normalization_method=config.data.numerical_normalization_method,
        cache_features=config.data.cache_config.enabled,
        cache_max_items=config.data.cache_config.max_memory_items,
        cache_dir=config.data.cache_config.cache_directory,
        cache_to_disk=config.data.cache_config.use_disk
    )
    
    # Load encoders
    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    with open(encoders_dir / 'user_encoder.pkl', 'rb') as f:
        dataset.user_encoder = pickle.load(f)
    with open(encoders_dir / 'item_encoder.pkl', 'rb') as f:
        dataset.item_encoder = pickle.load(f)
    # Ensure n_users and n_items are set after loading external encoders
    dataset.n_users = len(dataset.user_encoder.classes_)
    dataset.n_items = len(dataset.item_encoder.classes_)

    
    # Initialize model
    print("Loading model...")
    model = PretrainedMultimodalRecommender( # This is MultimodalRecommender
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        num_numerical_features=len(config.data.numerical_features_cols), # Pass the correct number of features
        embedding_dim=config.model.embedding_dim,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        use_contrastive=config.model.use_contrastive,
        dropout_rate=config.model.dropout_rate, # Ensure all relevant model_config fields are passed
        freeze_vision=config.model.freeze_vision,
        freeze_language=config.model.freeze_language,
        num_attention_heads=config.model.num_attention_heads,
        attention_dropout=config.model.attention_dropout,
        fusion_hidden_dims=config.model.fusion_hidden_dims,
        fusion_activation=config.model.fusion_activation,
        use_batch_norm=config.model.use_batch_norm,
        projection_hidden_dim=config.model.projection_hidden_dim,
        final_activation=config.model.final_activation,
        init_method=config.model.init_method,
        contrastive_temperature=config.model.contrastive_temperature
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth'
    if not checkpoint_path.exists():
        checkpoint_path = Path(config.checkpoint_dir) / 'final_model.pth'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {config.checkpoint_dir}/best_model.pth or final_model.pth")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    
    return model, dataset


def generate_recommendations_for_users(
    users: List[str],
    recommender: Recommender,
    config: Config,
    use_diversity: bool = False # This flag's functionality depends on get_diverse_recommendations
):
    """Generate recommendations for a list of users"""
    results = {}
    
    for user_id in users:
        print(f"\nGenerating recommendations for user: {user_id}")
        
        if use_diversity:
            # Generate diverse recommendations
            # This method needs to be implemented in src.inference.recommender.Recommender
            if hasattr(recommender, 'get_diverse_recommendations'):
                recommendations_data = recommender.get_diverse_recommendations(
                    user_id=user_id,
                    top_k=config.recommendation.top_k,
                    diversity_weight=config.recommendation.diversity_weight,
                    novelty_weight=config.recommendation.novelty_weight,
                    filter_seen=config.recommendation.filter_seen
                )
                # Assuming get_diverse_recommendations returns a tuple (recommendations, metrics)
                # where recommendations is a list of dicts
                if isinstance(recommendations_data, tuple) and len(recommendations_data) == 2:
                    recommendations, metrics = recommendations_data
                    user_results = {
                        'recommendations': recommendations, # Already in desired format
                        'metrics': {k: float(v) for k, v in metrics.items()}
                    }
                    # Print metrics
                    print("Novelty Metrics:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")

                else: # Fallback if structure is different or method not fully implemented
                    print("Warning: get_diverse_recommendations did not return expected (recommendations, metrics) tuple. Falling back to standard recommendations.")
                    recommendations = recommender.get_recommendations(
                        user_id=user_id, top_k=config.recommendation.top_k, filter_seen=config.recommendation.filter_seen
                    )
                    user_results = {'recommendations': [{'item_id': item_id, 'score': float(score)} for item_id, score in recommendations]}

            else:
                print("Warning: --use_diversity flag is set, but Recommender.get_diverse_recommendations is not implemented. Falling back to standard recommendations.")
                recommendations = recommender.get_recommendations(
                    user_id=user_id, top_k=config.recommendation.top_k, filter_seen=config.recommendation.filter_seen
                )
                user_results = {'recommendations': [{'item_id': item_id, 'score': float(score)} for item_id, score in recommendations]}

        else:
            # Generate standard recommendations
            recommendations = recommender.get_recommendations(
                user_id=user_id,
                top_k=config.recommendation.top_k,
                filter_seen=config.recommendation.filter_seen
            )
            
            user_results = {
                'recommendations': [
                    {'item_id': item_id, 'score': float(score)}
                    for item_id, score in recommendations
                ]
            }
        
        # Print recommendations
        print("Top Recommendations:")
        for i, rec in enumerate(user_results['recommendations'][:5], 1):
            print(f"  {i}. {rec['item_id']} (score: {rec.get('score', 'N/A'):.4f})") # Use .get for score
        
        results[user_id] = user_results
    
    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate recommendations')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/simple_config.yaml', # Corrected typo from simplet_config.yaml
        help='Path to configuration file'
    )
    parser.add_argument(
        '--users',
        type=str,
        nargs='+',
        help='User IDs to generate recommendations for'
    )
    parser.add_argument(
        '--user_file',
        type=str,
        help='File containing user IDs (one per line)'
    )
    parser.add_argument(
        '--sample_users',
        type=int,
        help='Number of random users to sample'
    )
    parser.add_argument(
        '--use_diversity',
        action='store_true',
        help='Use diversity-aware recommendation algorithm (requires Recommender.get_diverse_recommendations to be implemented)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='recommendations.json',
        help='Output file for recommendations'
    )
    parser.add_argument(
        '--embeddings_cache',
        type=str,
        help='Path to pre-computed embeddings cache (Note: current Recommender uses simple in-memory feature cache)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    device = torch.device(args.device)
    
    # Load model and data
    model, dataset = load_model_and_data(config, device)
    
    # Initialize recommender
    recommender = Recommender(
        model, 
        dataset, 
        device,
        cache_max_items=config.data.cache_config.max_memory_items,
        cache_dir=config.data.cache_config.cache_directory,
        cache_to_disk=config.data.cache_config.use_disk
    )
    
    # Load embeddings cache if provided - Note: current Recommender uses simple_cache for features, not separate embeddings.
    # This argument might be for a different caching mechanism not fully integrated with the current Recommender.
    if args.embeddings_cache:
        if hasattr(recommender, 'load_embeddings_cache'):
            recommender.load_embeddings_cache(args.embeddings_cache)
        else:
            print(f"Warning: --embeddings_cache provided, but recommender does not have 'load_embeddings_cache' method. Current Recommender uses internal feature caching.")

    
    # Determine which users to generate recommendations for
    if args.users:
        users = args.users
    elif args.user_file:
        with open(args.user_file, 'r') as f:
            users = [line.strip() for line in f if line.strip()]
    elif args.sample_users:
        all_users = dataset.user_encoder.classes_
        if len(all_users) < args.sample_users:
            print(f"Warning: Requested sample_users ({args.sample_users}) is more than available users ({len(all_users)}). Using all available users.")
            users = all_users.tolist()
        else:
            users = pd.Series(all_users).sample(n=args.sample_users, random_state=42).tolist()
    else:
        # Default: generate for a few sample users if encoders are not empty
        users = dataset.user_encoder.classes_[:5].tolist() if len(dataset.user_encoder.classes_) > 0 else []

    if not users:
        print("No users specified or found to generate recommendations for. Exiting.")
        return
    
    print(f"\nGenerating recommendations for {len(users)} users...")
    
    # Generate recommendations
    results = generate_recommendations_for_users(
        users=users,
        recommender=recommender,
        config=config,
        use_diversity=args.use_diversity
    )
    
    # Save results
    output_path = Path(config.results_dir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved recommendations to {output_path}")
    
    # Save embeddings cache - Note: current Recommender does not have save_embeddings_cache.
    # It uses its internal feature_cache which is not explicitly saved here.
    if not args.embeddings_cache:
        if hasattr(recommender, 'save_embeddings_cache'):
            cache_path = Path(config.results_dir) / 'embeddings_cache.pkl'
            recommender.save_embeddings_cache(cache_path)
        # else:
            # The current Recommender's simple_cache is not designed to be saved with save_embeddings_cache
            # print("Note: Recommender does not have 'save_embeddings_cache'. Internal feature cache not saved by this argument.")


if __name__ == '__main__':
    main()