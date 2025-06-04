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
from src.inference.recommender import Recommender as MultimodalRecommender
from src.evaluation.metrics import calculate_precision_at_k, calculate_recall_at_k, calculate_ndcg
from src.evaluation.tasks import EvaluationTask, create_evaluator  # NEW IMPORT
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


# KEEP YOUR ORIGINAL evaluate_recommendations FUNCTION AS A FALLBACK
def evaluate_recommendations(recommender, test_data, config):
    """Original evaluation function - kept for backward compatibility"""
    # ... (keep your existing implementation) ...
    pass


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
        type=str,
        help='Path to training data CSV file (optional, uses recommender data if not provided)'
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
    parser.add_argument(
        '--eval_task',
        type=str,
        default='retrieval',
        choices=['retrieval', 'ranking', 'next_item', 'cold_user', 'cold_item', 'beyond_accuracy', 'legacy'],
        help='Evaluation task to perform (use "legacy" for old evaluation function)'
    )

    args = parser.parse_args()

    config_obj = Config.from_yaml(args.config)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    # Load training data if provided
    train_df = None
    if args.train_data:
        print(f"Loading training data from {args.train_data}...")
        train_df = pd.read_csv(args.train_data)

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

    # Use a consistent image folder path
    effective_image_folder = config_obj.data.image_folder
    if hasattr(config_obj.data, 'offline_image_compression') and \
       config_obj.data.offline_image_compression.enabled and \
       hasattr(config_obj.data, 'processed_image_destination_folder') and \
       config_obj.data.processed_image_destination_folder:
        effective_image_folder = config_obj.data.processed_image_destination_folder

    dataset = MultimodalDataset(
        interactions_df=interactions_df_for_dataset_init,
        item_info_df=item_info_df,
        image_folder=effective_image_folder,
        vision_model_name=config_obj.model.vision_model,
        language_model_name=config_obj.model.language_model,
        create_negative_samples=False,
        numerical_feat_cols=config_obj.data.numerical_features_cols,
        numerical_normalization_method=config_obj.data.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        cache_processed_images=False
    )
    dataset.finalize_setup()

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

    model_instance = None
    if args.recommender_type == 'multimodal':
        print("Initializing model...")
        # Determine model class
        from src.models.multimodal import PretrainedMultimodalRecommender, EnhancedMultimodalRecommender
        
        if hasattr(config_obj.model, 'model_class') and config_obj.model.model_class == 'enhanced':
            model_class_to_use = EnhancedMultimodalRecommender
            print("Using EnhancedMultimodalRecommender for evaluation.")
        else:
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
            'dropout_rate': 0.0,  # No dropout during evaluation
            'num_attention_heads': config_obj.model.num_attention_heads,
            'attention_dropout': 0.0,
            'fusion_hidden_dims': config_obj.model.fusion_hidden_dims,
            'fusion_activation': config_obj.model.fusion_activation,
            'use_batch_norm': config_obj.model.use_batch_norm,
            'projection_hidden_dim': config_obj.model.projection_hidden_dim,
            'final_activation': config_obj.model.final_activation,
            'init_method': config_obj.model.init_method,
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
                print(f"Error: Model checkpoint not found.")
                sys.exit(1)

        print(f"Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.eval()

    # Create recommender instance
    recommender_instance = create_recommender(args.recommender_type, dataset, model_instance, device)

    print("\nStarting evaluation...")
    
    # NEW: Use task-based evaluation or legacy
    if args.eval_task == 'legacy':
        # Use original evaluation function
        print("Using legacy evaluation function...")
        results = evaluate_recommendations(recommender_instance, test_df, config_obj)
    else:
        # Map string to enum
        task_map = {
            'retrieval': EvaluationTask.TOP_K_RETRIEVAL,
            'ranking': EvaluationTask.TOP_K_RANKING,
            'next_item': EvaluationTask.NEXT_ITEM_PREDICTION,
            'cold_user': EvaluationTask.COLD_START_USER,
            'cold_item': EvaluationTask.COLD_START_ITEM,
            'beyond_accuracy': EvaluationTask.BEYOND_ACCURACY
        }
        eval_task = task_map[args.eval_task]
        
        # Create the appropriate evaluator
        print(f"Creating evaluator for task: {eval_task.value}")
        evaluator = create_evaluator(
            task=eval_task,
            recommender=recommender_instance,
            test_data=test_df,
            config=config_obj,
            train_data=train_df
        )
        
        print(f"Task: {evaluator.task_name}")
        print(f"Filter seen items: {evaluator.filter_seen}")
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Add metadata
        results['evaluation_metadata'] = {
            'task': evaluator.task_name,
            'filter_seen': evaluator.filter_seen,
            'recommender_type': args.recommender_type,
            'top_k': config_obj.recommendation.top_k,
            'test_file': args.test_data
        }
        
        # Print summary
        evaluator.print_summary(results)

    # Save results
    output_path = Path(config_obj.results_dir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to {output_path}")
    if args.eval_task == 'legacy':
        print("--- Evaluation Summary ---")
        print(json.dumps(results, indent=2))
        print("--------------------------")

if __name__ == '__main__':
    main()