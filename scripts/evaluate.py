#!/usr/bin/env python
"""
Evaluate model performance
"""
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
import json

# Add parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.inference.recommender import Recommender
from src.evaluation.metrics import calculate_ndcg, calculate_map

def evaluate_recommendations(recommender, test_data, config):
    """Evaluate recommendation quality"""
    metrics = {
        'precision@k': [],
        'recall@k': [],
        'ndcg@k': [],
        'map@k': [],
        'coverage': set(),
        'diversity': []
    }
    
    users = test_data['user_id'].unique()[:100]  # Sample for efficiency
    
    for user_id in users:
        # Get ground truth
        ground_truth = set(
            test_data[test_data['user_id'] == user_id]['item_id']
        )
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            user_id, 
            top_k=config.recommendation.top_k,
            filter_seen=True
        )
        
        if not recommendations:
            continue
            
        recommended_items = [item_id for item_id, _ in recommendations]
        
        # Calculate metrics
        hits = [1 if item in ground_truth else 0 for item in recommended_items]
        
        # Precision@k
        precision = sum(hits) / len(hits) if hits else 0
        metrics['precision@k'].append(precision)
        
        # Recall@k
        recall = sum(hits) / len(ground_truth) if ground_truth else 0
        metrics['recall@k'].append(recall)
        
        # Coverage
        metrics['coverage'].update(recommended_items)
    
    # Aggregate metrics
    results = {
        'precision@k': np.mean(metrics['precision@k']),
        'recall@k': np.mean(metrics['recall@k']),
        'coverage': len(metrics['coverage']) / len(test_data['item_id'].unique()),
        'n_users_evaluated': len(users)
    }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default_config.yaml')
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--output', default='evaluation_results.json')
    
    args = parser.parse_args()
    
    # Load config and model
    config = Config.from_yaml(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    test_df = pd.read_csv(args.test_data)
    
    # Initialize model and recommender
    # ... (similar to generate_recommendations.py)
    
    # Evaluate
    results = evaluate_recommendations(recommender, test_df, config)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()