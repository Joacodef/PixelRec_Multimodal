#!/usr/bin/env python
"""
Preprocess raw data for training.
This script performs initial cleaning, validation, and filtering of the raw data
and saves the processed versions. Numerical scaling fitting is handled in the training script.
"""
import argparse
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.preprocessing import (
    remove_html_tags,
    normalize_unicode_text,
    is_image_corrupted,
    check_image_dimensions
)

def validate_and_filter_images(
    item_df: pd.DataFrame,
    image_folder_path: Path,
    img_val_config: 'ImageValidationConfig' # Use string for forward reference from src.config
) -> pd.DataFrame:
    """Validate images and filter items with invalid/missing images."""
    valid_item_ids = []
    print("Validating images...")
    for item_id in tqdm(item_df['item_id'], desc="Validating images"):
        image_found = False
        for ext in img_val_config.allowed_extensions:
            image_path = image_folder_path / f"{item_id}{ext}"
            if image_path.exists():
                if img_val_config.check_corrupted and is_image_corrupted(str(image_path)):
                    # print(f"Warning: Image {image_path} is corrupted. Skipping item {item_id}.")
                    continue
                if not check_image_dimensions(str(image_path), img_val_config.min_width, img_val_config.min_height):
                    # print(f"Warning: Image {image_path} does not meet dimension requirements. Skipping item {item_id}.")
                    continue
                valid_item_ids.append(item_id)
                image_found = True
                break
        # if not image_found:
            # print(f"Warning: No valid image found for item {item_id} with allowed extensions. Skipping item.")
    
    original_count = len(item_df)
    filtered_df = item_df[item_df['item_id'].isin(valid_item_ids)].copy()
    print(f"Image validation: {len(filtered_df)} items remaining out of {original_count} after image validation.")
    return filtered_df

def clean_text_fields(df: pd.DataFrame, text_cols: List[str], clean_config: 'OfflineTextCleaningConfig') -> pd.DataFrame:
    """Clean specified text fields in the DataFrame."""
    print("Cleaning text data...")
    for col in tqdm(text_cols, desc="Cleaning text fields"):
        if col in df.columns:
            # Ensure column is string type
            df[col] = df[col].astype(str).fillna('')
            if clean_config.remove_html:
                df[col] = df[col].apply(remove_html_tags)
            if clean_config.normalize_unicode:
                df[col] = df[col].apply(normalize_unicode_text)
            if clean_config.to_lowercase:
                df[col] = df[col].str.lower()
            df[col] = df[col].str.strip()
    return df

def filter_interactions_by_valid_items(interactions_df: pd.DataFrame, valid_item_ids: set) -> pd.DataFrame:
    """Filter interactions to only include those with valid items."""
    original_count = len(interactions_df)
    filtered_df = interactions_df[interactions_df['item_id'].isin(valid_item_ids)].copy()
    print(f"Interaction filtering: {len(filtered_df)} interactions remaining out of {original_count} after filtering by valid items.")
    return filtered_df

def filter_by_activity(interactions_df: pd.DataFrame, min_user_interactions: int = 5, min_item_interactions: int = 3) -> pd.DataFrame:
    """Filter users and items by minimum number of interactions."""
    # Filter by item interactions first
    if min_item_interactions > 0:
        item_counts = interactions_df['item_id'].value_counts()
        active_items = item_counts[item_counts >= min_item_interactions].index
        interactions_df = interactions_df[interactions_df['item_id'].isin(active_items)]
        print(f"Filtered by item activity (min {min_item_interactions}): {len(interactions_df)} interactions, {interactions_df['item_id'].nunique()} items remain.")

    # Filter by user interactions
    if min_user_interactions > 0:
        user_counts = interactions_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_user_interactions].index
        interactions_df = interactions_df[interactions_df['user_id'].isin(active_users)]
        print(f"Filtered by user activity (min {min_user_interactions}): {len(interactions_df)} interactions, {interactions_df['user_id'].nunique()} users remain.")

    # Iterative filtering (optional, apply a few times if desired)
    # For simplicity, one pass each is often sufficient.
    return interactions_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw data for training.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)
    data_config = config.data
    print(f"Loaded configuration from {args.config}")

    # Define paths
    raw_item_info_path = Path(data_config.item_info_path)
    raw_interactions_path = Path(data_config.interactions_path)
    image_folder_path = Path(data_config.image_folder)

    processed_item_info_path = Path(data_config.processed_item_info_path)
    processed_interactions_path = Path(data_config.processed_interactions_path)

    # Create output directories
    processed_item_info_path.parent.mkdir(parents=True, exist_ok=True)
    processed_interactions_path.parent.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"\nLoading raw item info from {raw_item_info_path}...")
    item_info_df = pd.read_csv(raw_item_info_path)
    print(f"Loading raw interactions from {raw_interactions_path}...")
    interactions_df = pd.read_csv(raw_interactions_path)

    # --- Item Data Preprocessing ---
    # Clean text fields in item_info_df
    item_text_cols = ['title', 'tag', 'description'] # Adjust as per your item_info columns
    item_info_df = clean_text_fields(item_info_df, item_text_cols, data_config.offline_text_cleaning)

    # Validate images and filter item_info_df
    item_info_df = validate_and_filter_images(item_info_df, image_folder_path, data_config.offline_image_validation)
    valid_item_ids = set(item_info_df['item_id'])

    if not valid_item_ids:
        print("Error: No valid items remaining after image validation. Exiting.")
        sys.exit(1)

    # --- Interaction Data Preprocessing ---
    # Filter interactions based on valid items
    interactions_df = filter_interactions_by_valid_items(interactions_df, valid_item_ids)
    
    if interactions_df.empty:
        print("Error: No interactions remaining after filtering by valid items. Exiting.")
        sys.exit(1)

    # Filter by user and item activity levels
    # These min_interactions can also be part of config if desired
    interactions_df = filter_by_activity(interactions_df, min_user_interactions=5, min_item_interactions=3)
    
    if interactions_df.empty:
        print("Error: No interactions remaining after activity filtering. Exiting.")
        sys.exit(1)
        
    # Ensure item_info_df only contains items present in the final interactions_df
    final_interacting_item_ids = set(interactions_df['item_id'].unique())
    item_info_df = item_info_df[item_info_df['item_id'].isin(final_interacting_item_ids)].copy()


    # Save processed data
    print(f"\nSaving processed item info to {processed_item_info_path}...")
    item_info_df.to_csv(processed_item_info_path, index=False)
    print(f"Saving processed interactions to {processed_interactions_path}...")
    interactions_df.to_csv(processed_interactions_path, index=False)

    print("\nPreprocessing completed!")
    print(f"Final processed dataset: {len(interactions_df)} interactions")
    print(f"Number of unique users: {interactions_df['user_id'].nunique()}")
    print(f"Number of unique items: {interactions_df['item_id'].nunique()}")
    print(f"Item info rows: {len(item_info_df)}")

if __name__ == '__main__':
    main()