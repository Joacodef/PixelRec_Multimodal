#!/usr/bin/env python
"""
A command-line script for pre-computing and caching multimodal features.

This utility iterates through all items in the dataset, processes their
visual, textual, and numerical data to generate feature tensors, and saves
these features to disk. Pre-caching significantly accelerates the model
training process by removing the need to compute these features on-the-fly
during each epoch.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import pickle
import torch
from tqdm import tqdm
import time

# Add the project's root directory to the system path to allow importing from 'src'.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset



def precompute_features_cache(config_path: str, force_recompute: bool = False, max_items: int = None):
    """
    Orchestrates the feature pre-computation and caching process.

    This function iterates through all items defined in the item metadata file,
    generates their complete multimodal features (image, text, numerical), and
    saves each item's feature set as an individual file to a model-specific
    cache directory on disk.

    Args:
        config_path: The file path to the main YAML configuration file.
        force_recompute: If True, existing cached items will be recomputed and
                         overwritten.
        max_items: An optional integer to limit processing to the first N items,
                   which is useful for debugging or quick tests.
    """
    
    print("🔄 PRECOMPUTING MULTIMODAL FEATURES CACHE")
    print("=" * 60)
    
    # Loads the main configuration object from the specified YAML file.
    config = Config.from_yaml(config_path)
    data_config = config.data
    model_config = config.model
    
    print(f"Configuration:")
    print(f"  Vision model: {model_config.vision_model}")
    print(f"  Language model: {model_config.language_model}")
    
    # Loads the processed data files.
    print(f"\nLoading data...")
    item_info_df = pd.read_csv(data_config.processed_item_info_path)
    interactions_df = pd.read_csv(data_config.processed_interactions_path)
    
    print(f"  Items: {len(item_info_df):,}")
    print(f"  Interactions: {len(interactions_df):,}")
    
    # If max_items is specified, reduces the DataFrame to that number of items.
    if max_items and max_items < len(item_info_df):
        print(f"  Limiting to {max_items:,} items for testing")
        item_info_df = item_info_df.head(max_items)

    
    # Determines which image folder to use (raw or processed).
    effective_image_folder = data_config.processed_image_destination_folder or data_config.image_folder
    print(f"  Image folder: {effective_image_folder}")
    
    # Loads the pre-fitted numerical scaler if it exists.
    numerical_scaler = None
    scaler_path = Path(data_config.scaler_path)
    if scaler_path.is_file():
        print(f"  Loading numerical scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            # Extracts the scaler object from the saved dictionary.
            if isinstance(scaler_data, dict) and 'scaler' in scaler_data:
                numerical_scaler = scaler_data['scaler']
    else:
        print(f"  ⚠️  Numerical scaler not found at {scaler_path}")
    
    print(f"\nInitializing dataset for feature processing...")
    
    # Creates a minimal DataFrame of interactions to satisfy the Dataset constructor.
    # The actual interactions are not needed for item feature processing.
    dummy_interactions = pd.DataFrame({'item_id': item_info_df['item_id'], 'user_id': 'dummy_user'})
    
    # Initializes the MultimodalDataset, which will also initialize the SimpleFeatureCache.
    # We enable caching here so the dataset can manage it.
    dataset = MultimodalDataset(
        interactions_df=dummy_interactions,
        item_info_df=item_info_df,
        image_folder=effective_image_folder,
        vision_model_name=model_config.vision_model,
        language_model_name=model_config.language_model,
        create_negative_samples=False,
        cache_features=True,
        cache_to_disk=True,
        cache_dir=data_config.cache_config.cache_directory,
        numerical_feat_cols=data_config.numerical_features_cols,
        numerical_normalization_method=data_config.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        is_train_mode=False 
    )
    
    print(f"✅ Dataset initialized, starting feature computation...")
    
    successful, failed, skipped = 0, 0, 0
    start_time = time.time()
    
    print(f"\nProcessing {len(item_info_df):,} items...")
    
     # Iterate through unique item IDs and delegate caching to the cache object.
    for item_id in tqdm(item_info_df['item_id'].unique(), desc="Computing features"):
        try:
            # The _get_item_features method contains all the necessary processing logic.
            features = dataset._get_item_features(str(item_id))
            if features:
                # The cache's 'set' method correctly handles force_recompute and file saving.
                dataset.feature_cache.set(str(item_id), features, force_recompute=force_recompute)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed <= 5: print(f"\n⚠️  Error processing item {item_id}: {e}")

    # Prints a final summary of the caching process.
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 60)
    print("✅ CACHE PRECOMPUTATION COMPLETED")
    print(f"  Successfully processed: {successful:,}, Failed: {failed:,}")
    if successful > 0:
        # Get the cache directory path from the dataset's feature_cache object.
        print(f"  Location: {dataset.feature_cache.cache_dir}")
    if failed > 0:
        print(f"⚠️  {failed:,} items failed. Check logs for details.")

# This block allows the script to be executed from the command line.
if __name__ == "__main__":
    # Sets up an argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(description="Precompute and cache multimodal features.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation of all items, overwriting existing cache.')
    parser.add_argument('--max_items', type=int, default=None, help='Limit the number of items to process (for testing).')
    args = parser.parse_args()

    # Calls the main function with the parsed arguments.
    precompute_features_cache(args.config, args.force_recompute, args.max_items)