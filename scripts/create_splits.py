#!/usr/bin/env python
"""
Creates standardized and reproducible data splits for training and evaluation.

This script takes a processed interactions dataset and splits it into training,
validation, and test sets based on parameters defined in a configuration file.
It supports optional down-sampling of the dataset and ensures data quality by
filtering for user and item activity levels. The output includes the split data
in CSV format and a metadata file in JSON format detailing the split process.
"""
import sys
from pathlib import Path
import pandas as pd
import json
import hashlib
from typing import List, Optional

# Add the project's root directory to the system path to allow importing local modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.splitting import DataSplitter


def filter_by_activity(
    interactions_df: pd.DataFrame, 
    min_user_interactions: int, 
    min_item_interactions: int
) -> pd.DataFrame:
    """
    Filters an interactions DataFrame to ensure data density.

    This function removes users and items that do not meet the specified minimum
    interaction count thresholds. Filtering is applied sequentially, first for
    items and then for users.

    Args:
        interactions_df: The input DataFrame with 'user_id' and 'item_id' columns.
        min_user_interactions: The minimum number of interactions a user must
                               have to be included in the output.
        min_item_interactions: The minimum number of interactions an item must
                               have to be included in the output.

    Returns:
        A new DataFrame containing only interactions from users and items that
        meet the specified activity thresholds.
    """
    # Filters out items that have fewer than the specified minimum interactions.
    if min_item_interactions > 0:
        item_counts = interactions_df['item_id'].value_counts()
        active_items = item_counts[item_counts >= min_item_interactions].index
        interactions_df = interactions_df[interactions_df['item_id'].isin(active_items)]
        print(f"Filtered by item activity (min {min_item_interactions}): {len(interactions_df)} interactions, {interactions_df['item_id'].nunique()} items remain.")

    # Filters out users that have fewer than the specified minimum interactions.
    if min_user_interactions > 0:
        user_counts = interactions_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_user_interactions].index
        interactions_df = interactions_df[interactions_df['user_id'].isin(active_users)]
        print(f"Filtered by user activity (min {min_user_interactions}): {len(interactions_df)} interactions, {interactions_df['user_id'].nunique()} users remain.")
    
    return interactions_df


def create_splits(config_path: str, sample_n: Optional[int] = None):
    """
    Orchestrates the data splitting process based on a configuration file.

    This function loads data, applies activity filtering and optional sampling,
    performs a stratified split into training, validation, and test sets, and
    saves the outputs to disk along with a metadata file.

    Args:
        config_path: The file path to the YAML configuration file that governs
                     the splitting process (e.g., ratios, paths, criteria).
        sample_n: If provided, the number of interactions to randomly sample
                  from the dataset before filtering and splitting. If None, the
                  full dataset is used.
    """
    # Loads the main configuration object from the specified YAML file.
    config = Config.from_yaml(config_path)
    # Retrieves the specific configuration section for data splitting.
    splitting_config = config.data.splitting
    
    # Loads the processed interactions data, which is the input for splitting.
    interactions_df = pd.read_csv(config.data.processed_interactions_path)
    original_total_interactions = len(interactions_df)

    # If a sample size is provided, randomly sample the interactions DataFrame.
    if sample_n is not None:
        if sample_n > 0 and sample_n <= len(interactions_df):
            print(f"Sampling {sample_n} interactions from the dataset (original size: {len(interactions_df)})...")
            interactions_df = interactions_df.sample(n=sample_n, random_state=splitting_config.random_state).reset_index(drop=True)
            print(f"Dataset size after sampling: {len(interactions_df)} interactions.")
        elif sample_n > len(interactions_df):
            print(f"Warning: Requested sample_n ({sample_n}) is larger than the dataset size ({len(interactions_df)}). Using the full dataset.")
        else:
            print("Warning: Invalid sample_n value. Using the full dataset.")
    
    # Ensures that the target directory for the output splits exists.
    output_path = Path(config.data.split_data_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filters the dataset to include only users and items with sufficient activity.
    interactions_df = filter_by_activity(
        interactions_df,
        min_user_interactions=splitting_config.min_interactions_per_user,
        min_item_interactions=splitting_config.min_interactions_per_item
    )

    # Halts execution if no data remains after the filtering step.
    if interactions_df.empty:
        print("Error: No interactions remaining after activity filtering (and potential sampling). Cannot create splits.")
        sys.exit(1)
    
    # Initializes the data splitter with a fixed random state for reproducibility.
    splitter = DataSplitter(random_state=splitting_config.random_state)
    
    # Calculates the combined ratio for the training and validation sets.
    total_train_val_ratio = splitting_config.train_final_ratio + splitting_config.val_final_ratio
    
    # Validates that the configured split ratios are mathematically sound.
    if total_train_val_ratio <= 0:
        raise ValueError("Sum of train_final_ratio and val_final_ratio must be greater than zero.")
    if splitting_config.test_final_ratio < 0 or splitting_config.test_final_ratio >= 1:
        raise ValueError("test_final_ratio must be between 0 and 1 (exclusive of 1).")
    
    # Normalizes the split ratios if they do not sum to 1.0.
    sum_of_all_ratios = splitting_config.train_final_ratio + splitting_config.val_final_ratio + splitting_config.test_final_ratio
    if abs(sum_of_all_ratios - 1.0) > 1e-6:
        print(f"Warning: Configured split ratios do not sum to 1.0. Normalizing...")
        if sum_of_all_ratios > 0:
            splitting_config.train_final_ratio /= sum_of_all_ratios
            splitting_config.val_final_ratio /= sum_of_all_ratios
            splitting_config.test_final_ratio /= sum_of_all_ratios
            total_train_val_ratio = splitting_config.train_final_ratio + splitting_config.val_final_ratio
        else:
            # Defaults to a 60/20/20 split if all configured ratios are zero.
            print("All configured split ratios are zero. Defaulting to 60/20/20 split.")
            splitting_config.train_final_ratio, splitting_config.val_final_ratio, splitting_config.test_final_ratio = 0.6, 0.2, 0.2
            total_train_val_ratio = 0.8

    # First split: Separates the test set from the combined training and validation data.
    train_val_df, test_df = splitter.stratified_split(
        interactions_df, 
        train_ratio=total_train_val_ratio,
        min_interactions_per_user=splitting_config.min_interactions_per_user
    )
    
    # Second split: Divides the combined data into final training and validation sets.
    train_ratio_for_second_split = (splitting_config.train_final_ratio / total_train_val_ratio) if total_train_val_ratio > 0 else 0.0
    train_df, val_df = splitter.stratified_split(
        train_val_df,
        train_ratio=train_ratio_for_second_split,
        min_interactions_per_user=splitting_config.min_interactions_per_user
    )
    
    # Saves the final data splits to CSV files.
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    # Compiles and saves a metadata file that documents the entire splitting process.
    metadata = {
        "creation_date": pd.Timestamp.now().isoformat(),
        "config_file": config_path,
        "original_total_interactions_before_sampling": original_total_interactions, 
        "requested_sample_n": sample_n if sample_n is not None else "Not Sampled", 
        "interactions_after_sampling_before_filtering": len(interactions_df) if sample_n is not None else original_total_interactions, 
        "total_interactions_after_activity_filtering": len(interactions_df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_users": train_df['user_id'].nunique(),
        "val_users": val_df['user_id'].nunique(),
        "test_users": test_df['user_id'].nunique(),
        "train_items": train_df['item_id'].nunique(),
        "val_items": val_df['item_id'].nunique(),
        "test_items": test_df['item_id'].nunique(),
        "data_hash_after_sampling_and_filtering": hashlib.md5(interactions_df.to_csv().encode()).hexdigest(),
        "split_ratios_configured": {
            "train_final_ratio": splitting_config.train_final_ratio,
            "val_final_ratio": splitting_config.val_final_ratio,
            "test_final_ratio": splitting_config.test_final_ratio
        },
        "random_state_used": splitting_config.random_state,
        "min_interactions_per_user_used": splitting_config.min_interactions_per_user,
        "min_interactions_per_item_used": splitting_config.min_interactions_per_item
    }
    
    with open(output_path / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created evaluation splits in {output_path}")
    print(json.dumps(metadata, indent=2))

    # Performs an optional validation to check for user and item overlap between splits.
    if splitting_config.validate_no_leakage:
        print("\nPerforming leakage validation...")
        train_test_stats = splitter.get_split_statistics(train_df, test_df)
        print(f"Train-Test Overlap: Users - {train_test_stats['user_overlap']} ({train_test_stats['user_overlap_ratio']:.2%}), Items - {train_test_stats['item_overlap']} ({train_test_stats['item_overlap_ratio']:.2%})")

        train_val_stats = splitter.get_split_statistics(train_df, val_df)
        print(f"Train-Validation Overlap: Users - {train_val_stats['user_overlap']} ({train_val_stats['user_overlap_ratio']:.2%}), Items - {train_val_stats['item_overlap']} ({train_val_stats['item_overlap_ratio']:.2%})")
        
        val_test_stats = splitter.get_split_statistics(val_df, test_df)
        print(f"Validation-Test Overlap: Users - {val_test_stats['user_overlap']} ({val_test_stats['user_overlap_ratio']:.2%}), Items - {val_test_stats['item_overlap']} ({val_test_stats['item_overlap_ratio']:.2%})")

        # For stratified splits, user and item overlap between sets is expected,
        # as the goal is to distribute a user's interactions across the splits.
        if train_test_stats['user_overlap_ratio'] < 1.0 or train_test_stats['item_overlap_ratio'] < 1.0:
            print("Info: Some users/items in test set might not be present in the training set, or vice-versa. This is typical for stratified splits.")
        
        print("Leakage validation completed.")

if __name__ == "__main__":
    # Sets up an argument parser to handle command-line execution of the script.
    import argparse
    parser = argparse.ArgumentParser(description="Create standardized splits for evaluation")
    parser.add_argument('--config', type=str, default='configs/simple_config.yaml',
                        help='Path to configuration file (e.g., configs/simple_config.yaml)')

    parser.add_argument('--sample_n', type=int, default=None,
                        help='Optional: Number of random interactions to sample from the dataset before splitting.')

    args = parser.parse_args()
    
    # Calls the main splitting function with the parsed arguments.
    create_splits(args.config, args.sample_n)