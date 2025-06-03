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
from src.models.multimodal import PretrainedMultimodalRecommender
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
        create_negative_samples=False
    )
    
    # Load encoders
    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    with open(encoders_dir / 'user_encoder.pkl', 'rb') as f:
        dataset.user_encoder = pickle.load(f)
    with open(encoders_dir / 'item_encoder.pkl', 'rb') as f:
        dataset.item_encoder = pickle.load(f)
    
    # Initialize model
    print("Loading model...")
    model = PretrainedMultimodalRecommender(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=config.model.embedding_dim,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        use_contrastive=config.model.use_contrastive,
        dropout_rate=config.model.dropout_rate
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    
    return model, dataset


def generate_recommendations_for_users(
    users: List[str],
    recommender: Recommender,
    config: Config,
    use_diversity: bool = False
):
    """Generate recommendations for a list of users"""
    results = {}
    
    for user_id in users:
        print(f"\nGenerating recommendations for user: {user_id}")
        
        if use_diversity:
            # Generate diverse recommendations
            recommendations, metrics = recommender.get_diverse_recommendations(
                user_id=user_id,
                top_k=config.recommendation.top_k,
                diversity_weight=config.recommendation.diversity_weight,
                novelty_weight=config.recommendation.novelty_weight,
                filter_seen=config.recommendation.filter_seen
            )
            
            # Format results
            user_results = {
                'recommendations': [
                    {
                        'item_id': rec['item_id'],
                        'score': float(rec['score']),
                        'popularity': float(rec['popularity'])
                    }
                    for rec in recommendations
                ],
                'metrics': {k: float(v) for k, v in metrics.items()}
            }
            
            # Print metrics
            print("Novelty Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
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
            print(f"  {i}. {rec['item_id']} (score: {rec['score']:.4f})")
        
        results[user_id] = user_results
    
    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate recommendations')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
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
        help='Use diversity-aware recommendation algorithm'
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
        help='Path to pre-computed embeddings cache'
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
    recommender = Recommender(model, dataset, device)
    
    # Load embeddings cache if provided
    if args.embeddings_cache:
        recommender.load_embeddings_cache(args.embeddings_cache)
    
    # Determine which users to generate recommendations for
    if args.users:
        users = args.users
    elif args.user_file:
        with open(args.user_file, 'r') as f:
            users = [line.strip() for line in f if line.strip()]
    elif args.sample_users:
        all_users = dataset.user_encoder.classes_
        users = pd.Series(all_users).sample(n=args.sample_users, random_state=42).tolist()
    else:
        # Default: generate for a few sample users
        users = dataset.user_encoder.classes_[:5]
    
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
    
    # Save embeddings cache
    if not args.embeddings_cache:
        cache_path = Path(config.results_dir) / 'embeddings_cache.pkl'
        recommender.save_embeddings_cache(cache_path)


if __name__ == '__main__':
    main()