# scripts/precompute_cache.py
#!/usr/bin/env python
"""
Simple script to precompute multimodal features cache
Saves all features for each item in a single dictionary file: cache/<vision_model>_<language_model>/
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import pickle
import torch
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset


def get_cache_directory(vision_model: str, language_model: str, base_path: str = "cache") -> Path:
    """Get cache directory for specific model combination"""
    cache_name = f"{vision_model}_{language_model}"
    cache_dir = Path(base_path) / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def precompute_features_cache(config_path: str, force_recompute: bool = False, max_items: int = None):
    """
    Precompute and cache all multimodal features
    
    Args:
        config_path: Path to configuration file
        force_recompute: Whether to recompute existing cached items
        max_items: Maximum number of items to process (for testing)
    """
    
    print("🔄 PRECOMPUTING MULTIMODAL FEATURES CACHE")
    print("=" * 60)
    
    # Load configuration
    config = Config.from_yaml(config_path)
    data_config = config.data
    model_config = config.model
    
    print(f"Configuration:")
    print(f"  Vision model: {model_config.vision_model}")
    print(f"  Language model: {model_config.language_model}")
    
    # Setup cache directory using the path from config
    base_cache_path = Path(data_config.cache_config.cache_directory)
    cache_dir = get_cache_directory(model_config.vision_model, model_config.language_model, base_path=str(base_cache_path))
    print(f"  Cache directory: {cache_dir}")
    
    # Load data
    print(f"\nLoading data...")
    item_info_df = pd.read_csv(data_config.processed_item_info_path)
    interactions_df = pd.read_csv(data_config.processed_interactions_path)
    
    print(f"  Items: {len(item_info_df):,}")
    print(f"  Interactions: {len(interactions_df):,}")
    
    if max_items and max_items < len(item_info_df):
        print(f"  Limiting to {max_items:,} items for testing")
        item_info_df = item_info_df.head(max_items)
    
    if force_recompute:
        print(f"  Clearing existing cache...")
        for cache_file in cache_dir.glob("*.pt"):
            cache_file.unlink()
        print(f"  ✅ Cache cleared")
    
    effective_image_folder = data_config.processed_image_destination_folder or data_config.image_folder
    print(f"  Image folder: {effective_image_folder}")
    
    # Load numerical scaler if needed
    numerical_scaler = None
    scaler_path = Path(data_config.scaler_path)
    if scaler_path.exists():
        print(f"  Loading numerical scaler from {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            # FIX: Extract the scaler object from the dictionary
            if isinstance(scaler_data, dict) and 'scaler' in scaler_data:
                numerical_scaler = scaler_data['scaler']
                print("  → Extracted scaler object from saved dictionary.")
            else:
                numerical_scaler = scaler_data # Fallback for old format
    else:
        print(f"  ⚠️  Numerical scaler not found at {scaler_path}")
    
    print(f"\nInitializing dataset for feature processing...")
    
    sample_interactions = pd.DataFrame({'user_id': ['temp'], 'item_id': [item_info_df.iloc[0]['item_id']]})
    
    dataset = MultimodalDataset(
        interactions_df=sample_interactions,
        item_info_df=item_info_df,
        image_folder=effective_image_folder,
        vision_model_name=model_config.vision_model,
        language_model_name=model_config.language_model,
        create_negative_samples=False,
        cache_features=False,
        numerical_feat_cols=data_config.numerical_features_cols,
        numerical_normalization_method=data_config.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        is_train_mode=False
    )
    
    print(f"✅ Dataset initialized, starting feature computation...")
    
    successful, failed, skipped = 0, 0, 0
    start_time = time.time()
    
    print(f"\nProcessing {len(item_info_df):,} items...")
    
    for idx, (_, item_row) in enumerate(tqdm(item_info_df.iterrows(), total=len(item_info_df), desc="Computing features")):
        item_id = str(item_row['item_id'])
        cache_file = cache_dir / f"{item_id}.pt"
        
        if not force_recompute and cache_file.exists():
            skipped += 1
            continue
        
        try:
            features = dataset._process_item_features(item_id)
            if features:
                torch.save(features, cache_file)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed <= 5: print(f"\n⚠️  Error processing item {item_id}: {e}")

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 60)
    print("✅ CACHE PRECOMPUTATION COMPLETED")
    print(f"  Successfully cached: {successful:,}, Failed: {failed:,}, Skipped: {skipped:,}")
    if successful > 0:
        print(f"  Location: {cache_dir}")
    if failed > 0:
        print(f"⚠️  {failed:,} items failed. Check logs for details.")