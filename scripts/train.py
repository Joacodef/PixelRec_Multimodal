#!/usr/bin/env python
"""
Training script for the multimodal recommender system
"""
import argparse
import yaml
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ModelConfig, TrainingConfig, DataConfig
from src.data.dataset import MultimodalDataset
from src.models.multimodal import PretrainedMultimodalRecommender
from src.training.trainer import train_model

def main():
    parser = argparse.ArgumentParser(description='Train multimodal recommender')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create configurations
    model_config = ModelConfig(**config['model'])
    training_config = TrainingConfig(**config['training'])
    data_config = DataConfig(**config['data'])
    
    # Train model
    # ... (training code here)

if __name__ == '__main__':
    main()