# scripts/create_splits.py
#!/usr/bin/env python
"""Create standardized splits for evaluation"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.splitting import DataSplitter

import pandas as pd
import json
import hashlib

def create_splits(config_path, output_dir="data/evaluation_splits"):
    """
    Create standardized train/validation/test splits for fair comparison.
    The split ratios and other parameters are read from the provided configuration file.
    """
    
    # Load configuration
    config = Config.from_yaml(config_path)
    splitting_config = config.data.splitting # Access the splitting configuration directly
    
    # Load processed interactions data
    interactions_df = pd.read_csv(config.data.processed_interactions_path)
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data splitter with configured random state
    splitter = DataSplitter(random_state=splitting_config.random_state)
    
    # Calculate intermediate ratios for the two-step stratified split
    # The first split separates out the test set
    # The ratio for train_val is (train_final_ratio + val_final_ratio)
    total_train_val_ratio = splitting_config.train_final_ratio + splitting_config.val_final_ratio
    
    # Basic validation for ratios
    if total_train_val_ratio <= 0:
        raise ValueError("Sum of train_final_ratio and val_final_ratio must be greater than zero.")
    if splitting_config.test_final_ratio < 0 or splitting_config.test_final_ratio >= 1:
        raise ValueError("test_final_ratio must be between 0 and 1 (exclusive of 1).")
    
    # Check if ratios sum to 1.0 and normalize if not (due to floating point errors)
    sum_of_all_ratios = splitting_config.train_final_ratio + splitting_config.val_final_ratio + splitting_config.test_final_ratio
    if abs(sum_of_all_ratios - 1.0) > 1e-6:
        print(f"Warning: Configured train_final_ratio ({splitting_config.train_final_ratio}), "
              f"val_final_ratio ({splitting_config.val_final_ratio}), and "
              f"test_final_ratio ({splitting_config.test_final_ratio}) do not sum to 1.0 (sum is {sum_of_all_ratios}). Normalizing...")
        if sum_of_all_ratios > 0:
            splitting_config.train_final_ratio /= sum_of_all_ratios
            splitting_config.val_final_ratio /= sum_of_all_ratios
            splitting_config.test_final_ratio /= sum_of_all_ratios
            total_train_val_ratio = splitting_config.train_final_ratio + splitting_config.val_final_ratio
        else: # Handle case where all ratios are zero
            print("All configured split ratios are zero. Defaulting to 60/20/20 split.")
            splitting_config.train_final_ratio = 0.6
            splitting_config.val_final_ratio = 0.2
            splitting_config.test_final_ratio = 0.2
            total_train_val_ratio = 0.8


    # First split: Separate out the test set from the rest
    train_val_df, test_df = splitter.stratified_split(
        interactions_df, 
        train_ratio=total_train_val_ratio, # This is the proportion of data going to train_val
        min_interactions_per_user=splitting_config.min_interactions_per_user # Use configurable min interactions
    )
    
    # Second split: Divide the train_val_df into train and validation sets
    # The train_ratio for this step is train_final_ratio / (train_final_ratio + val_final_ratio)
    # This ensures the correct proportion of train and val within the train_val_df
    train_ratio_for_second_split = (splitting_config.train_final_ratio / total_train_val_ratio) if total_train_val_ratio > 0 else 0.0
    
    train_df, val_df = splitter.stratified_split(
        train_val_df,
        train_ratio=train_ratio_for_second_split,
        min_interactions_per_user=splitting_config.min_interactions_per_user # Use configurable min interactions
    )
    
    # Save the generated splits to CSV files
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    # Create and save metadata about the split
    metadata = {
        "creation_date": pd.Timestamp.now().isoformat(),
        "config_file": config_path,
        "total_interactions": len(interactions_df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_users": train_df['user_id'].nunique(),
        "val_users": val_df['user_id'].nunique(),
        "test_users": test_df['user_id'].nunique(),
        "train_items": train_df['item_id'].nunique(),
        "val_items": val_df['item_id'].nunique(),
        "test_items": test_df['item_id'].nunique(),
        "data_hash": hashlib.md5(interactions_df.to_csv().encode()).hexdigest(),
        "split_ratios_configured": { # Record the configured ratios in metadata
            "train_final_ratio": splitting_config.train_final_ratio,
            "val_final_ratio": splitting_config.val_final_ratio,
            "test_final_ratio": splitting_config.test_final_ratio
        },
        "random_state_used": splitting_config.random_state,
        "min_interactions_per_user_used": splitting_config.min_interactions_per_user
    }
    
    with open(output_path / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created evaluation splits in {output_path}")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    args = parser.parse_args()
    
    create_evaluation_splits(args.config)