#!/usr/bin/env python
"""
A command-line script for generating personalized recommendations.

This script loads a trained multimodal recommender model and associated data
to generate top-K recommendations for a specified set of users. It supports
various methods for user selection (individual IDs, file-based, random sampling)
and saves the output in a structured JSON format.
"""
import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import pickle
import json
from typing import List, Optional

# Add the project's root directory to the system path to allow importing from 'src'.
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.inference.recommender import Recommender


def find_model_checkpoint(config: Config) -> Path:
    """
    Finds the model checkpoint file using a prioritized search strategy.

    This function first searches in the model-specific directory (e.g.,
    'checkpoints/resnet_sentence-bert/'), then falls back to the base
    checkpoint directory to ensure backward compatibility. It tries a list of
    common checkpoint names in order of preference.

    Args:
        config: The configuration object containing directory paths and model names.

    Returns:
        A pathlib.Path object pointing to the found checkpoint file.

    Raises:
        FileNotFoundError: If no suitable checkpoint file can be found in any of
                           the searched locations.
    """
    # Defines the model-specific subdirectory name based on the configuration.
    model_combo = f"{config.model.vision_model}_{config.model.language_model}"
    model_specific_dir = Path(config.checkpoint_dir) / model_combo
    
    # Defines a prioritized list of checkpoint filenames to search for.
    checkpoint_names = ['best_model.pth', 'final_model.pth', 'last_model.pth']
    
    # First, search in the model-specific directory.
    for name in checkpoint_names:
        path = model_specific_dir / name
        if path.exists():
            print(f"✓ Found checkpoint in model-specific directory: {path}")
            return path
            
    # As a fallback, search in the base checkpoint directory.
    for name in checkpoint_names:
        path = Path(config.checkpoint_dir) / name
        if path.exists():
            print(f"✓ Found checkpoint in base directory (fallback): {path}")
            return path
            
    # If no preferred checkpoint is found, raises an error with helpful debug info.
    raise FileNotFoundError(
        f"Model checkpoint not found. Searched for {checkpoint_names} in:\n"
        f"  → Model-specific dir: {model_specific_dir}\n"
        f"  → Base dir: {config.checkpoint_dir}"
    )


def load_model_and_data(config: Config, device: torch.device):
    """
    Loads and prepares all necessary assets for inference.

    This function handles loading the processed datasets, user/item encoders,
    and the trained model weights into the model architecture defined by the
    configuration.

    Args:
        config: The main configuration object.
        device: The torch.device to which the model will be moved.

    Returns:
        A tuple containing the initialized and loaded PyTorch model and the
        fully configured dataset instance.
    """
    # Loads the processed item metadata and user interaction data.
    print("Loading data...")
    item_info_df = pd.read_csv(config.data.processed_item_info_path)
    interactions_df = pd.read_csv(config.data.processed_interactions_path)
    
    # Creates a dataset instance. For inference, negative sampling is disabled.
    dataset = MultimodalDataset(
        interactions_df,
        item_info_df,
        config.data.image_folder,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        create_negative_samples=False,
        numerical_feat_cols=config.data.numerical_features_cols,
        numerical_normalization_method=config.data.numerical_normalization_method,
        cache_features=config.data.cache_config.enabled,
        cache_max_items=config.data.cache_config.max_memory_items,
        cache_dir=config.data.cache_config.cache_directory,
        cache_to_disk=config.data.cache_config.use_disk
    )
    
    # Loads the pre-fitted user and item encoders.
    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    with open(encoders_dir / 'user_encoder.pkl', 'rb') as f:
        dataset.user_encoder = pickle.load(f)
    with open(encoders_dir / 'item_encoder.pkl', 'rb') as f:
        dataset.item_encoder = pickle.load(f)
    
    # Recalculates user and item counts based on the loaded encoders.
    dataset.n_users = len(dataset.user_encoder.classes_)
    dataset.n_items = len(dataset.item_encoder.classes_)
    
    # Initializes the model architecture based on the configuration.
    print("Loading model...")
    model = PretrainedMultimodalRecommender(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        num_numerical_features=len(config.data.numerical_features_cols),
        embedding_dim=config.model.embedding_dim,
        vision_model_name=config.model.vision_model,
        language_model_name=config.model.language_model,
        use_contrastive=config.model.use_contrastive,
        dropout_rate=config.model.dropout_rate,
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
    
    # Locates and loads the saved model weights into the initialized architecture.
    checkpoint_path = find_model_checkpoint(config)
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
    """
    Generates and formats recommendations for a list of specified users.

    This function iterates through a list of user IDs, calls the recommender's
    recommendation generation method, and formats the output into a structured
    dictionary.

    Args:
        users: A list of user IDs for whom to generate recommendations.
        recommender: An initialized Recommender object.
        config: The main configuration object.
        use_diversity: A flag to indicate whether to use a diversity-aware
                       recommendation algorithm (if implemented).

    Returns:
        A dictionary where keys are user IDs and values are dictionaries
        containing the list of formatted recommendations.
    """
    results = {}
    
    # Iterates through each user to generate recommendations.
    for user_id in users:
        print(f"\nGenerating recommendations for user: {user_id}")
        
        # Branch for diversity-aware recommendations (if implemented).
        if use_diversity:
            if hasattr(recommender, 'get_diverse_recommendations'):
                recommendations_data = recommender.get_diverse_recommendations(
                    user_id=user_id,
                    top_k=config.recommendation.top_k,
                    diversity_weight=config.recommendation.diversity_weight,
                    novelty_weight=config.recommendation.novelty_weight,
                    filter_seen=config.recommendation.filter_seen
                )
                recommendations, metrics = recommendations_data
                user_results = {
                    'recommendations': recommendations,
                    'metrics': {k: float(v) for k, v in metrics.items()}
                }
            else:
                # Falls back to standard recommendations if the diversity method is not implemented.
                print("Warning: Diversity method not implemented. Falling back to standard recommendations.")
                recommendations = recommender.get_recommendations(
                    user_id=user_id, top_k=config.recommendation.top_k, filter_seen=config.recommendation.filter_seen
                )
                user_results = {'recommendations': [{'item_id': item_id, 'score': float(score)} for item_id, score in recommendations]}
        # Branch for standard top-K recommendations.
        else:
            recommendations = recommender.get_recommendations(
                user_id=user_id,
                top_k=config.recommendation.top_k,
                filter_seen=config.recommendation.filter_seen
            )
            # Formats the raw recommendations into a list of dictionaries.
            user_results = {
                'recommendations': [
                    {'item_id': item_id, 'score': float(score)}
                    for item_id, score in recommendations
                ]
            }
        
        # Prints the top 5 recommendations to the console for immediate feedback.
        print("Top Recommendations:")
        for i, rec in enumerate(user_results['recommendations'][:5], 1):
            print(f"  {i}. {rec['item_id']} (score: {rec.get('score', 0.0):.4f})")
        
        results[user_id] = user_results
    
    return results


def main():
    """
    Main entry point for the recommendation generation script.

    This function orchestrates the entire process, including parsing command-line
    arguments, loading all necessary components, determining the target users,
    triggering the recommendation generation, and saving the final results.
    """
    # Sets up an argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(description='Generate recommendations using a trained model.')
    parser.add_argument('--config', type=str, default='configs/simple_config.yaml', help='Path to configuration file.')
    parser.add_argument('--users', type=str, nargs='+', help='A list of user IDs to generate recommendations for.')
    parser.add_argument('--user_file', type=str, help='Path to a file containing user IDs, one per line.')
    parser.add_argument('--sample_users', type=int, help='Number of random users to sample from the dataset.')
    parser.add_argument('--use_diversity', action='store_true', help='Use a diversity-aware recommendation algorithm.')
    parser.add_argument('--output', type=str, default='recommendations.json', help='Name of the output JSON file.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for inference.')
    args = parser.parse_args()
    
    # Loads the configuration and initializes the model and data.
    config = Config.from_yaml(args.config)
    device = torch.device(args.device)
    model, dataset = load_model_and_data(config, device)
    
    # Initializes the main Recommender object for inference.
    recommender = Recommender(
        model, dataset, device,
        cache_max_items=config.data.cache_config.max_memory_items,
        cache_dir=config.data.cache_config.cache_directory,
        cache_to_disk=config.data.cache_config.use_disk
    )
    
    # Determines the final list of users to generate recommendations for.
    if args.users:
        users = args.users
    elif args.user_file:
        with open(args.user_file, 'r') as f:
            users = [line.strip() for line in f if line.strip()]
    elif args.sample_users:
        all_users = dataset.user_encoder.classes_
        if len(all_users) < args.sample_users:
            users = all_users.tolist()
        else:
            users = pd.Series(all_users).sample(n=args.sample_users, random_state=42).tolist()
    else:
        # Defaults to the first 5 users if no other option is specified.
        users = dataset.user_encoder.classes_[:5].tolist() if len(dataset.user_encoder.classes_) > 0 else []

    if not users:
        print("No users specified or found. Exiting.")
        return
    
    print(f"\nGenerating recommendations for {len(users)} users...")
    
    # Calls the main generation function.
    results = generate_recommendations_for_users(
        users=users,
        recommender=recommender,
        config=config,
        use_diversity=args.use_diversity
    )
    
    # Saves the results to a JSON file in the configured results directory.
    output_path = Path(config.results_dir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved recommendations to {output_path}")

# Ensures the main() function is called only when the script is executed directly.
if __name__ == '__main__':
    main()