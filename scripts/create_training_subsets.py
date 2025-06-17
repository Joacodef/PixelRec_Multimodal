import pandas as pd
import yaml
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config

def create_subsets(config_path: str):
    """
    Loads the full training data and creates smaller, stratified subsets for HPO.
    """
    print("--- Creating Stratified Training Subsets for Hyperparameter Optimization ---")
    cfg = Config.from_yaml(config_path)
    
    full_train_path = Path(cfg.data.train_data_path)
    if not full_train_path.exists():
        print(f"Error: Full training file not found at {full_train_path}")
        print("Please run scripts/create_splits.py first.")
        return

    print(f"Loading full training data from: {full_train_path}")
    df_train_full = pd.read_csv(full_train_path)

    # --- Stratification by Timestamp ---
    # To ensure subsets have a similar temporal distribution, we bin the timestamps.
    # This turns the continuous timestamp into a categorical feature for stratification.
    print("Binning timestamps for stratification...")
    # Ensure timestamp is in datetime format
    df_train_full['timestamp'] = pd.to_datetime(df_train_full['timestamp'])
    # Create 10 bins (e.g., deciles) based on the timestamp distribution.
    df_train_full['time_bin'] = pd.qcut(df_train_full['timestamp'], q=10, labels=False, duplicates='drop')

    # --- Create the 50% Subset ---
    print("Creating 50% subset...")
    # First, split off the 50% subset from the full dataset.
    _, df_train_50 = train_test_split(
        df_train_full,
        test_size=0.5, # 50% of the data
        random_state=cfg.data.splitting.random_state, # Use same random state for consistency
        stratify=df_train_full['time_bin'] # Ensure timestamp distribution is preserved
    )

    # --- Create the 20% Subset FROM THE 50% SUBSET ---
    # This ensures that the 20% set is a true subset of the 50% set.
    print("Creating 20% subset (from the 50% subset)...")
    # The new test_size is 0.4 because 0.2 / 0.5 = 0.4
    _, df_train_20 = train_test_split(
        df_train_50,
        test_size=0.4, # 40% of the 50% subset gives 20% of the original
        random_state=cfg.data.splitting.random_state,
        stratify=df_train_50['time_bin']
    )

    # --- Create the 5% Subset FROM THE 20% SUBSET ---
    # This ensures that the 5% set is a true subset of the 20% set.
    print("Creating 5% subset (from the 20% subset)...")
    # The new test_size is 0.25 because 0.05 / 0.20 = 0.25
    _, df_train_05 = train_test_split(
        df_train_20,
        test_size=0.25, # 25% of the 20% subset gives 5% of the original
        random_state=cfg.data.splitting.random_state,
        stratify=df_train_20['time_bin']
    )

    # --- Save the Subsets ---
    # Get the directory where the main splits are located
    splits_dir = full_train_path.parent

    path_50_percent = splits_dir / "train_50_percent.csv"
    path_20_percent = splits_dir / "train_20_percent.csv"
    path_05_percent = splits_dir / "train_05_percent.csv"

    # We don't need the 'time_bin' column in the final files.
    df_train_50.drop(columns=['time_bin'], inplace=True)
    df_train_20.drop(columns=['time_bin'], inplace=True)
    df_train_05.drop(columns=['time_bin'], inplace=True)

    df_train_50.to_csv(path_50_percent, index=False)
    df_train_20.to_csv(path_20_percent, index=False)
    df_train_05.to_csv(path_05_percent, index=False)

    print("\n--- Subsets Created Successfully ---")
    print(f"Full training set size: {len(df_train_full)}")
    print(f"50% subset saved to: {path_50_percent} (size: {len(df_train_50)})")
    print(f"20% subset saved to: {path_20_percent} (size: {len(df_train_20)})")
    print(f"5% subset saved to: {path_05_percent} (size: {len(df_train_05)})")

    # --- Verification of Timestamp Stratification ---
    print("\n--- Verifying Timestamp Stratification ---")
    original_time_dist = df_train_full['timestamp'].dt.to_period('M').value_counts(normalize=True).sort_index()
    subset_05_time_dist = pd.read_csv(path_05_percent)['timestamp']
    subset_05_time_dist = pd.to_datetime(subset_05_time_dist).dt.to_period('M').value_counts(normalize=True).sort_index()

    # Align indices and fill missing values with 0 for comparison
    all_months = original_time_dist.index.union(subset_05_time_dist.index)
    original_aligned = original_time_dist.reindex(all_months, fill_value=0)
    subset_05_aligned = subset_05_time_dist.reindex(all_months, fill_value=0)

    # Calculate absolute difference between distributions
    diff = (original_aligned - subset_05_aligned).abs().sum()

    print(f"Original (Full Training) Timestamp Distribution (first 5):\n{original_time_dist.head()}")
    print(f"\n5% Subset Timestamp Distribution (first 5):\n{subset_05_aligned.head()}")
    print(f"\nAbsolute sum of differences in monthly timestamp distribution: {diff:.4f}")

    if diff < 0.1: # A small threshold for difference, can be adjusted
        print("Timestamp stratification appears to be working correctly (difference is small).")
    else:
        print("Warning: Large difference in timestamp distribution, stratification might not be effective.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create training data subsets for HPO.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main configuration file.')
    args = parser.parse_args()
    create_subsets(args.config)