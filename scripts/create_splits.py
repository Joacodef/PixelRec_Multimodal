# scripts/create_splits.py
"""
Data Splitting Script
"""
import pandas as pd
import yaml
import argparse
from pathlib import Path
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.data.processors.data_filter import DataFilter
from src.data.splitting import create_robust_splits, DataSplitter

def main(config_path: str):
    """
    Main function to execute the data splitting process.
    """
    cfg = Config.from_yaml(config_path)
    
    try:
        interactions_df = pd.read_csv(cfg.data.processed_interactions_path)
    except FileNotFoundError:
        print(f"Error: Processed interactions file not found at {cfg.data.processed_interactions_path}")
        return

    # Initialize DataFilter and apply activity filtering
    data_filter = DataFilter()
    min_user_interactions = cfg.data.splitting.min_interactions_per_user
    min_item_interactions = cfg.data.splitting.min_interactions_per_item
    
    print("Filtering data by minimum interactions...")
    filtered_df = data_filter.filter_by_activity(
        interactions_df,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions
    )
    
    if filtered_df.empty:
        print("No data left after filtering. Please check your interaction thresholds.")
        return

    # If stratification is requested on a column not in the interactions DataFrame,
    # attempt to merge it from the item metadata (item_info.csv).
    stratify_col = cfg.data.splitting.stratify_by
    if stratify_col and stratify_col not in filtered_df.columns:
        print(f"Stratification column '{stratify_col}' not in interactions, attempting to merge from item info.")
        try:
            item_info_path = Path(cfg.data.processed_item_info_path)
            if not item_info_path.exists():
                raise FileNotFoundError(f"Processed item info file not found at {item_info_path}")

            item_info_df = pd.read_csv(item_info_path)
            
            if stratify_col in item_info_df.columns:
                # Select only the item ID and the stratification column for the merge
                item_info_subset = item_info_df[['item_id', stratify_col]]
                
                # Perform a left merge to add the stratification column to the interactions
                filtered_df = pd.merge(filtered_df, item_info_subset, on='item_id', how='left')
                print(f"Successfully merged '{stratify_col}' from item info for stratification.")

                # Check for any interactions that did not have a corresponding item
                if filtered_df[stratify_col].isnull().any():
                    print(f"Warning: Null values are present in '{stratify_col}' after merge.")
            else:
                print(f"Warning: Stratification column '{stratify_col}' not in '{item_info_path}'. Proceeding without stratification.")
                cfg.data.splitting.stratify_by = None
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Could not merge stratification column '{stratify_col}' due to an error: {e}. Proceeding without stratification.")
            cfg.data.splitting.stratify_by = None

    # Prepare parameters and execute the data split
    split_params = {
        'split_strategy': cfg.data.splitting.strategy,
        'random_state': cfg.data.splitting.random_state,
        'train_ratio': cfg.data.splitting.train_final_ratio,
        'val_ratio': cfg.data.splitting.val_final_ratio,
        'test_ratio': cfg.data.splitting.test_final_ratio,
        'stratify_by': cfg.data.splitting.stratify_by,
        'min_interactions_per_user': min_user_interactions,
        'min_interactions_per_item': min_item_interactions
    }
    
    splits = create_robust_splits(filtered_df, **split_params)

    # Save the generated splits to CSV files
    output_dir = Path(cfg.data.split_data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = DataSplitter(random_state=cfg.data.splitting.random_state)
    
    if len(splits) == 3:
        train_df, val_df, test_df = splits
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        stats = splitter.get_split_statistics(train_df, val_df, test_df)
    else:
        train_df, val_df = splits
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        stats = splitter.get_split_statistics(train_df, val_df)

    print("\nSplit Statistics:")
    print(yaml.dump(stats, sort_keys=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data splits for the recommender system.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)