# Path: scripts/preprocess_data.py
#!/usr/bin/env python
"""
Preprocess raw data for training.
This script performs initial cleaning, validation, filtering, and optional compression
of raw data, saving processed versions.
"""
import argparse
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional
import os
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the specific dataclass types needed for type hinting
from src.config import Config, OfflineImageCompressionConfig, ImageValidationConfig, OfflineTextCleaningConfig
from src.data.preprocessing import (
    remove_html_tags,
    normalize_unicode_text,
    is_image_corrupted,
    check_image_dimensions
)

def process_and_save_image(
    original_image_path: Path,
    destination_image_path: Path,
    # Corrected type hints:
    compression_config: OfflineImageCompressionConfig,
    img_val_config: ImageValidationConfig
) -> bool:
    """
    Processes a single image: validates, optionally compresses/resizes, and saves it.
    Returns True if processed and saved successfully, False otherwise.
    """
    destination_image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not original_image_path.exists():
            return False
        # Use attributes from the img_val_config object
        if img_val_config.check_corrupted and is_image_corrupted(str(original_image_path)):
            return False
        if not check_image_dimensions(str(original_image_path), img_val_config.min_width, img_val_config.min_height):
            return False

        compress_this_image = False
        # Use attributes from the compression_config object
        if compression_config.enabled:
            original_filesize_kb = original_image_path.stat().st_size / 1024
            if original_filesize_kb > compression_config.compress_if_kb_larger_than:
                compress_this_image = True

        if compress_this_image:
            with Image.open(original_image_path) as img:
                img = img.convert("RGB")

                if compression_config.resize_if_pixels_larger_than and \
                   compression_config.resize_target_longest_edge:
                    if img.width > compression_config.resize_if_pixels_larger_than[0] or \
                       img.height > compression_config.resize_if_pixels_larger_than[1]:
                        current_longest_edge = max(img.width, img.height)
                        scale_factor = compression_config.resize_target_longest_edge / current_longest_edge
                        new_width = int(img.width * scale_factor)
                        new_height = int(img.height * scale_factor)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                file_extension = destination_image_path.suffix.lower()
                if file_extension in ['.jpg', '.jpeg']:
                    img.save(destination_image_path, quality=compression_config.target_quality, optimize=True)
                elif file_extension == '.png':
                    img.save(destination_image_path, compress_level=6) 
                else: 
                    img.save(destination_image_path)
        else:
            shutil.copy2(original_image_path, destination_image_path)
        return True
    except Exception as e:
        # print(f"Error processing image {original_image_path}: {e}")
        return False


def validate_and_filter_items_with_processing(
    item_df: pd.DataFrame,
    original_image_folder_path: Path,
    processed_image_dest_folder_path: Path,
    # Corrected type hints:
    img_val_config: ImageValidationConfig,
    compression_config: OfflineImageCompressionConfig
) -> pd.DataFrame:
    """
    Validates original images, processes them (copies or compresses/resizes)
    to a new destination, and filters items if their processed image
    is not successfully created.
    """
    valid_item_ids_with_processed_image = []
    print("Validating and processing images...")
    processed_image_dest_folder_path.mkdir(parents=True, exist_ok=True)

    for item_id in tqdm(item_df['item_id'], desc="Processing and validating images"):
        image_processed_successfully = False
        original_image_found_path = None
        # Use attributes from img_val_config object
        for ext in img_val_config.allowed_extensions:
            potential_original_path = original_image_folder_path / f"{item_id}{ext}"
            if potential_original_path.exists():
                original_image_found_path = potential_original_path
                break
        
        if original_image_found_path:
            destination_path = processed_image_dest_folder_path / original_image_found_path.name
            
            if process_and_save_image(
                original_image_found_path,
                destination_path,
                compression_config, # Pass the actual config object
                img_val_config      # Pass the actual config object
            ):
                image_processed_successfully = True
                valid_item_ids_with_processed_image.append(item_id)
        
        if not image_processed_successfully and not original_image_found_path:
            pass
        elif not image_processed_successfully and original_image_found_path:
            pass
            
    original_count = len(item_df)
    filtered_df = item_df[item_df['item_id'].isin(valid_item_ids_with_processed_image)].copy()
    print(f"Image processing and validation: {len(filtered_df)} items remaining out of {original_count}.")
    return filtered_df

# Ensure clean_text_fields also uses the correct type hint if not already
def clean_text_fields(df: pd.DataFrame, text_cols: List[str], clean_config: OfflineTextCleaningConfig) -> pd.DataFrame:
    print("Cleaning text data...")
    for col in tqdm(text_cols, desc="Cleaning text fields"):
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('')
            if clean_config.remove_html:
                df[col] = df[col].apply(remove_html_tags)
            if clean_config.normalize_unicode:
                df[col] = df[col].apply(normalize_unicode_text)
            if clean_config.to_lowercase:
                df[col] = df[col].str.lower()
            df[col] = df[col].str.strip()
    return df


def filter_interactions_by_valid_items(interactions_df: pd.DataFrame, valid_item_ids: set) -> pd.DataFrame: #
    """Filter interactions to only include those with valid items.""" #
    original_count = len(interactions_df) #
    filtered_df = interactions_df[interactions_df['item_id'].isin(valid_item_ids)].copy() #
    print(f"Interaction filtering: {len(filtered_df)} interactions remaining out of {original_count} after filtering by valid items.") #
    return filtered_df #

def filter_by_activity(interactions_df: pd.DataFrame, min_user_interactions: int = 5, min_item_interactions: int = 3) -> pd.DataFrame: #
    """Filter users and items by minimum number of interactions.""" #
    # Filter by item interactions first #
    if min_item_interactions > 0: #
        item_counts = interactions_df['item_id'].value_counts() #
        active_items = item_counts[item_counts >= min_item_interactions].index #
        interactions_df = interactions_df[interactions_df['item_id'].isin(active_items)] #
        print(f"Filtered by item activity (min {min_item_interactions}): {len(interactions_df)} interactions, {interactions_df['item_id'].nunique()} items remain.") #

    # Filter by user interactions #
    if min_user_interactions > 0: #
        user_counts = interactions_df['user_id'].value_counts() #
        active_users = user_counts[user_counts >= min_user_interactions].index #
        interactions_df = interactions_df[interactions_df['user_id'].isin(active_users)] #
        print(f"Filtered by user activity (min {min_user_interactions}): {len(interactions_df)} interactions, {interactions_df['user_id'].nunique()} users remain.") #
    return interactions_df #


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw data for training.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    data_config = config.data # This data_config is an instance of DataConfig
    print(f"Loaded configuration from {args.config}")

    raw_item_info_path = Path(data_config.item_info_path)
    raw_interactions_path = Path(data_config.interactions_path)
    original_image_folder_path = Path(data_config.image_folder)

    processed_image_dest_folder = Path(data_config.processed_image_destination_folder)
    # processed_image_dest_folder.mkdir(parents=True, exist_ok=True) # This is handled in process_and_save_image

    processed_item_info_path = Path(data_config.processed_item_info_path)
    processed_interactions_path = Path(data_config.processed_interactions_path)

    processed_item_info_path.parent.mkdir(parents=True, exist_ok=True)
    processed_interactions_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading raw item info from {raw_item_info_path}...")
    item_info_df = pd.read_csv(raw_item_info_path)
    print(f"Loading raw interactions from {raw_interactions_path}...")
    interactions_df = pd.read_csv(raw_interactions_path)

    item_text_cols = ['title', 'tag', 'description']
    # Pass the actual config objects from data_config instance
    item_info_df = clean_text_fields(item_info_df, item_text_cols, data_config.offline_text_cleaning)

    item_info_df = validate_and_filter_items_with_processing(
        item_info_df,
        original_image_folder_path,
        processed_image_dest_folder,
        data_config.offline_image_validation,      # Pass the instance
        data_config.offline_image_compression    # Pass the instance
    )
    valid_item_ids = set(item_info_df['item_id'])

    if not valid_item_ids:
        print("Error: No valid items remaining after image processing and validation. Exiting.")
        sys.exit(1)

    interactions_df = filter_interactions_by_valid_items(interactions_df, valid_item_ids)
    
    if interactions_df.empty:
        print("Error: No interactions remaining after filtering by valid items. Exiting.")
        sys.exit(1)

    interactions_df = filter_by_activity(
        interactions_df,
        min_user_interactions=data_config.splitting.min_interactions_per_user,
        min_item_interactions=data_config.splitting.min_interactions_per_item
    )
    
    if interactions_df.empty:
        print("Error: No interactions remaining after activity filtering. Exiting.")
        sys.exit(1)
        
    final_interacting_item_ids = set(interactions_df['item_id'].unique())
    item_info_df = item_info_df[item_info_df['item_id'].isin(final_interacting_item_ids)].copy()

    print(f"\nSaving processed item info to {processed_item_info_path}...")
    item_info_df.to_csv(processed_item_info_path, index=False)
    print(f"Saving processed interactions to {processed_interactions_path}...")
    interactions_df.to_csv(processed_interactions_path, index=False)

    print("\nPreprocessing completed!")
    print(f"Processed images are saved in: {processed_image_dest_folder}")
    print(f"Final processed dataset: {len(interactions_df)} interactions")
    print(f"Number of unique users: {interactions_df['user_id'].nunique()}")
    print(f"Number of unique items: {interactions_df['item_id'].nunique()}")
    print(f"Item info rows: {len(item_info_df)}")

if __name__ == '__main__':
    main()