# Path: scripts/preprocess_data.py
#!/usr/bin/env python
"""
Preprocess raw data for training.
This script performs initial cleaning, validation, filtering, optional compression
of raw data, saving processed versions, AND pre-computes and caches 
non-image features (tokenized text, numerical) to disk.
"""
import argparse
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional, Any
import os
from PIL import Image
import torch # Added for torch.save/load with ProcessedFeatureCache

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config, OfflineImageCompressionConfig, ImageValidationConfig, OfflineTextCleaningConfig
from src.data.preprocessing import (
    remove_html_tags,
    normalize_unicode_text
    # is_image_corrupted, # Defined locally below for clarity
    # check_image_dimensions # Defined locally below for clarity
)
# Import the new ProcessedFeatureCache and MultimodalDataset
try:
    from src.data.feature_cache import ProcessedFeatureCache
except ImportError:
    ProcessedFeatureCache = None
    print("WARNING: ProcessedFeatureCache class could not be imported from src.data.feature_cache.")
    print("Ensure the file exists and the path is correct if you intend to cache non-image features.")

from src.data.dataset import MultimodalDataset # For accessing tokenizers and feature processing logic

# Local definitions for image validation functions if not imported or to ensure they are the ones used
def is_image_corrupted(image_path: str) -> bool:
    """Checks if an image file is corrupted by trying to open it."""
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path) # Must re-open after verify
        img.load() # Try to load the image data
        return False
    except Exception:
        return True

def check_image_dimensions(image_path: str, min_width: int, min_height: int) -> bool:
    """Checks if an image meets minimum dimension requirements."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width >= min_width and height >= min_height
    except Exception:
        return False # If image can't be opened, it fails the check


def process_and_save_image(
    original_image_path: Path,
    destination_image_path: Path,
    compression_config: OfflineImageCompressionConfig,
    img_val_config: ImageValidationConfig
) -> bool:
    """
    Processes a single image: validates, optionally compresses/resizes, and saves it.
    Returns True if processed and saved successfully, False otherwise.
    If the processed image already exists at the destination, it skips processing.
    """
    destination_image_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_image_path.exists():
        return True # Skip if already processed

    try:
        if not original_image_path.exists():
            return False
        if img_val_config.check_corrupted and is_image_corrupted(str(original_image_path)):
            return False
        if not check_image_dimensions(str(original_image_path), img_val_config.min_width, img_val_config.min_height):
            return False

        compress_this_image = False
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
    img_val_config: ImageValidationConfig,
    compression_config: OfflineImageCompressionConfig
) -> pd.DataFrame:
    valid_item_ids_with_processed_image = []
    print("Validating and processing images (e.g., compression/resizing)...")
    processed_image_dest_folder_path.mkdir(parents=True, exist_ok=True)

    for item_id in tqdm(item_df['item_id'].astype(str), desc="Processing and validating images"):
        original_image_found_path = None
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
                compression_config,
                img_val_config
            ):
                valid_item_ids_with_processed_image.append(item_id)
            
    original_count = len(item_df)
    filtered_df = item_df[item_df['item_id'].astype(str).isin(valid_item_ids_with_processed_image)].copy()
    print(f"Image processing and validation: {len(filtered_df)} items remaining out of {original_count}.")
    return filtered_df

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

def filter_interactions_by_valid_items(interactions_df: pd.DataFrame, valid_item_ids: set) -> pd.DataFrame:
    original_count = len(interactions_df)
    # Ensure item_id in interactions_df is compared as string if valid_item_ids are strings
    filtered_df = interactions_df[interactions_df['item_id'].astype(str).isin(list(map(str, valid_item_ids)))].copy()
    print(f"Interaction filtering: {len(filtered_df)} interactions remaining out of {original_count} after filtering by valid items.")
    return filtered_df

def filter_by_activity(interactions_df: pd.DataFrame, min_user_interactions: int = 5, min_item_interactions: int = 3) -> pd.DataFrame:
    if min_item_interactions > 0:
        item_counts = interactions_df['item_id'].value_counts()
        active_items = item_counts[item_counts >= min_item_interactions].index
        interactions_df = interactions_df[interactions_df['item_id'].isin(active_items)].copy()
        print(f"Filtered by item activity (min {min_item_interactions}): {len(interactions_df)} interactions, {interactions_df['item_id'].nunique()} items remain.")

    if min_user_interactions > 0:
        user_counts = interactions_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_user_interactions].index
        interactions_df = interactions_df[interactions_df['user_id'].isin(active_users)].copy()
        print(f"Filtered by user activity (min {min_user_interactions}): {len(interactions_df)} interactions, {interactions_df['user_id'].nunique()} users remain.")
    return interactions_df

def fit_and_save_scaler(
    item_info_df: pd.DataFrame, 
    numerical_cols: List[str], 
    method: str, 
    scaler_path: Path
) -> Optional[Any]:
    """Fits and saves a numerical scaler if method requires it and path is provided."""
    if not numerical_cols or method in ['none', 'log1p'] or not scaler_path:
        print(f"Numerical scaler fitting skipped (method: {method}, no_cols: {not numerical_cols}, no_path: {not scaler_path}).")
        return None
    
    data_to_scale = item_info_df[numerical_cols].fillna(0).values
    if method == 'standardization':
        scaler = StandardScaler()
    elif method == 'min_max':
        scaler = MinMaxScaler()
    else:
        print(f"Unknown scaling method for fitting: {method}. No scaler fitted.")
        return None
    
    print(f"Fitting {method} scaler on {len(data_to_scale)} samples from item_info_df...")
    scaler.fit(data_to_scale)
    
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Numerical scaler saved to {scaler_path}")
    return scaler

def load_scaler(scaler_path: Path) -> Optional[Any]:
    """Loads a numerical scaler from path if it exists."""
    if scaler_path and scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Numerical scaler loaded from {scaler_path}")
        return scaler
    print(f"Warning: Scaler not found at {scaler_path}. Proceeding without pre-loaded scaler.")
    return None

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw data and cache non-image features.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    data_config = config.data
    print(f"Loaded configuration from {args.config}")

    raw_item_info_path = Path(data_config.item_info_path)
    raw_interactions_path = Path(data_config.interactions_path)
    original_image_folder_path = Path(data_config.image_folder)

    processed_image_dest_folder = Path(data_config.processed_image_destination_folder)
    processed_item_info_path = Path(data_config.processed_item_info_path)
    processed_interactions_path = Path(data_config.processed_interactions_path)
    scaler_save_path = Path(data_config.scaler_path) # From config

    processed_item_info_path.parent.mkdir(parents=True, exist_ok=True)
    processed_interactions_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading raw item info from {raw_item_info_path}...")
    item_info_df = pd.read_csv(raw_item_info_path)
    item_info_df['item_id'] = item_info_df['item_id'].astype(str) # Ensure item_id is string
    
    print(f"Loading raw interactions from {raw_interactions_path}...")
    interactions_df = pd.read_csv(raw_interactions_path)
    interactions_df['item_id'] = interactions_df['item_id'].astype(str) # Ensure item_id is string
    interactions_df['user_id'] = interactions_df['user_id'].astype(str) 


    item_text_cols = ['title', 'tag', 'description'] # Assuming these are the text columns
    item_info_df = clean_text_fields(item_info_df, item_text_cols, data_config.offline_text_cleaning)

    item_info_df = validate_and_filter_items_with_processing(
        item_info_df,
        original_image_folder_path,
        processed_image_dest_folder,
        data_config.offline_image_validation,
        data_config.offline_image_compression
    )
    valid_item_ids = set(item_info_df['item_id'].astype(str))

    if not valid_item_ids:
        print("Error: No valid items remaining after image processing and validation. Exiting.")
        sys.exit(1)

    interactions_df = filter_interactions_by_valid_items(interactions_df, valid_item_ids)
    
    if interactions_df.empty:
        print("Error: No interactions remaining after filtering by valid items. Exiting.")
        sys.exit(1)

    # Activity filtering uses min_interactions from the splitting config section
    interactions_df = filter_by_activity(
        interactions_df,
        min_user_interactions=data_config.splitting.min_interactions_per_user,
        min_item_interactions=data_config.splitting.min_interactions_per_item
    )
    
    if interactions_df.empty:
        print("Error: No interactions remaining after activity filtering. Exiting.")
        sys.exit(1)
        
    final_interacting_item_ids = set(interactions_df['item_id'].unique())
    item_info_df = item_info_df[item_info_df['item_id'].astype(str).isin(list(map(str,final_interacting_item_ids)))].copy()

    # Fit and save numerical scaler using the finalized item_info_df
    numerical_scaler_instance = None
    if data_config.numerical_features_cols:
        if scaler_save_path.exists():
            print(f"Scaler already exists at {scaler_save_path}. Loading it.")
            numerical_scaler_instance = load_scaler(scaler_save_path)
        else:
            print(f"Fitting and saving numerical scaler to {scaler_save_path}...")
            numerical_scaler_instance = fit_and_save_scaler(
                item_info_df,
                data_config.numerical_features_cols,
                data_config.numerical_normalization_method,
                scaler_save_path
            )
    else:
        print("No numerical_features_cols defined in config. Skipping scaler fitting.")


    print(f"\nSaving processed item info to {processed_item_info_path}...")
    item_info_df.to_csv(processed_item_info_path, index=False)
    print(f"Saving processed interactions to {processed_interactions_path}...")
    interactions_df.to_csv(processed_interactions_path, index=False)

    # --- New Section: Precompute and Cache Non-Image Features ---
    if ProcessedFeatureCache and hasattr(data_config, 'processed_features_cache_config'):
        pfc_config = data_config.processed_features_cache_config
        pfc_cache_dir = Path(pfc_config.cache_directory)
        print(f"\nInitializing ProcessedFeatureCache for precomputation (non-image features):")
        print(f"  Strategy: {pfc_config.strategy}")
        print(f"  Cache directory: {pfc_cache_dir}")

        feature_cache_instance = ProcessedFeatureCache(
            cache_path=str(pfc_cache_dir),
            max_memory_items=pfc_config.max_memory_items, # For precomputation, disk is primary
            strategy=pfc_config.strategy
        )
        feature_cache_instance.load_from_disk_meta() # Check existing cache

        # Create a minimal MultimodalDataset instance to use its feature processing methods
        # This dataset instance does not need to do negative sampling or have full interactions
        # It primarily needs the item_info_df, tokenizers, and numerical processing settings.
        print("Creating temporary Dataset instance for feature extraction logic...")
        temp_dataset_for_feature_extraction = MultimodalDataset(
            interactions_df=interactions_df, # Pass final interactions, mainly for item_encoder consistency if used by dataset methods
            item_info_df=item_info_df,       # Use the final processed item_info
            image_folder=str(processed_image_dest_folder), # Path to already processed images
            vision_model_name=config.model.vision_model, # Needed for CLIP tokenizer if vision is CLIP
            language_model_name=config.model.language_model,
            create_negative_samples=False,
            text_augmentation_config=None, # No augmentation during precomputation
            numerical_feat_cols=data_config.numerical_features_cols,
            numerical_normalization_method=data_config.numerical_normalization_method,
            numerical_scaler=numerical_scaler_instance, # Use the scaler fitted/loaded above
            is_train_mode=False,
            cache_processed_images=False, # Not caching images with this temp dataset's own cache
            shared_image_cache=None       # Not using SharedImageCache here
        )
        # Important: Finalize setup for the temp_dataset to ensure its internal components like
        # encoders (if any part of its feature processing relies on item_idx) and
        # pre-processing of numerical features within its item_info copy are set up.
        temp_dataset_for_feature_extraction.finalize_setup()


        print(f"Precomputing and caching non-image features for {len(item_info_df)} items...")
        for index, item_row in tqdm(item_info_df.iterrows(), total=len(item_info_df), desc="Caching non-image features"):
            item_id = str(item_row['item_id'])
            
            # Check if already cached to avoid reprocessing (optional, cache.get() handles this too)
            # if feature_cache_instance.get(item_id) is not None:
            #     continue

            try:
                # Use dataset methods to get consistently processed features
                # item_info_series_for_item = temp_dataset_for_feature_extraction._get_item_info(item_id) # Already have item_row as series
                text_content = temp_dataset_for_feature_extraction._get_item_text(item_row) # Pass series directly

                main_tokenizer_max_len = 128
                if hasattr(temp_dataset_for_feature_extraction.tokenizer, 'model_max_length') and \
                   temp_dataset_for_feature_extraction.tokenizer.model_max_length:
                    main_tokenizer_max_len = temp_dataset_for_feature_extraction.tokenizer.model_max_length

                text_tokens = temp_dataset_for_feature_extraction.tokenizer(
                    text_content,
                    padding='max_length',
                    truncation=True,
                    max_length=main_tokenizer_max_len,
                    return_tensors='pt'
                )
                
                # Get numerical features using the dataset's method which handles normalization
                numerical_features_tensor = temp_dataset_for_feature_extraction._get_item_numerical_features(item_id, item_row)

                non_image_feature_dict = {
                    'text_input_ids': text_tokens['input_ids'].squeeze(0),
                    'text_attention_mask': text_tokens['attention_mask'].squeeze(0),
                    'numerical_features': numerical_features_tensor
                }

                if temp_dataset_for_feature_extraction.clip_tokenizer_for_contrastive:
                    clip_tokens = temp_dataset_for_feature_extraction.clip_tokenizer_for_contrastive(
                        text_content,
                        padding='max_length',
                        truncation=True,
                        max_length=77, # Standard CLIP max length
                        return_tensors='pt'
                    )
                    non_image_feature_dict['clip_text_input_ids'] = clip_tokens['input_ids'].squeeze(0)
                    non_image_feature_dict['clip_text_attention_mask'] = clip_tokens['attention_mask'].squeeze(0)
                
                feature_cache_instance.set(item_id, non_image_feature_dict)
            except Exception as e:
                print(f"Error processing or caching features for item {item_id}: {e}")
        
        print("Non-image feature caching complete.")
        feature_cache_instance.print_stats()
    else:
        print("\nProcessedFeatureCache not configured or class not available. Skipping non-image feature caching.")


    print("\nPreprocessing and feature caching (if enabled) completed!")
    print(f"Processed images are saved in: {processed_image_dest_folder}")
    print(f"Final processed interactions dataset: {len(interactions_df)} interactions")
    print(f"Number of unique users in interactions: {interactions_df['user_id'].nunique()}")
    print(f"Number of unique items in interactions: {interactions_df['item_id'].nunique()}")
    print(f"Final item info rows: {len(item_info_df)}")
    if numerical_scaler_instance:
        print(f"Numerical scaler has been fitted/loaded and saved to {scaler_save_path}.")
    if ProcessedFeatureCache and hasattr(data_config, 'processed_features_cache_config'):
        print(f"Cached non-image features are stored in: {data_config.processed_features_cache_config.cache_directory}")


if __name__ == '__main__':
    main()