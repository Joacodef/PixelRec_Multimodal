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
    
    # Setup cache directory
    cache_dir = get_cache_directory(model_config.vision_model, model_config.language_model)
    print(f"  Cache directory: {cache_dir}")
    
    # Load data
    print(f"\nLoading data...")
    item_info_df = pd.read_csv(data_config.processed_item_info_path)
    interactions_df = pd.read_csv(data_config.processed_interactions_path)
    
    print(f"  Items: {len(item_info_df):,}")
    print(f"  Interactions: {len(interactions_df):,}")
    
    # Limit items for testing
    if max_items and max_items < len(item_info_df):
        print(f"  Limiting to {max_items:,} items for testing")
        item_info_df = item_info_df.head(max_items)
    
    # Clear existing cache if force recompute
    if force_recompute:
        print(f"  Clearing existing cache...")
        for cache_file in cache_dir.glob("*.pt"):
            cache_file.unlink()
        print(f"  ✅ Cache cleared")
    
    # Setup image folder
    effective_image_folder = data_config.image_folder
    if (hasattr(data_config, 'offline_image_compression') and 
        data_config.offline_image_compression.enabled and 
        hasattr(data_config, 'processed_image_destination_folder') and 
        data_config.processed_image_destination_folder):
        effective_image_folder = data_config.processed_image_destination_folder
    
    print(f"  Image folder: {effective_image_folder}")
    
    # Load numerical scaler if needed
    numerical_scaler = None
    if data_config.numerical_normalization_method in ['standardization', 'min_max']:
        scaler_path = Path(data_config.scaler_path)
        if scaler_path.exists():
            print(f"  Loading numerical scaler from {scaler_path}")
            with open(scaler_path, 'rb') as f:
                numerical_scaler = pickle.load(f)
        else:
            print(f"  ⚠️  Numerical scaler not found at {scaler_path}")
    
    # Create minimal dataset for feature processing
    print(f"\nInitializing dataset for feature processing...")
    print(f"  (This will download models if not cached - may take a few minutes)")
    
    # Create minimal interactions for dataset initialization
    sample_interactions = pd.DataFrame({
        'user_id': ['temp_user'],
        'item_id': [item_info_df.iloc[0]['item_id']]
    })
    
    dataset = MultimodalDataset(
        interactions_df=sample_interactions,
        item_info_df=item_info_df,
        image_folder=effective_image_folder,
        vision_model_name=model_config.vision_model,
        language_model_name=model_config.language_model,
        create_negative_samples=False,
        cache_features=False,  # Don't use cache during precomputation
        numerical_feat_cols=data_config.numerical_features_cols,
        numerical_normalization_method=data_config.numerical_normalization_method,
        numerical_scaler=numerical_scaler,
        is_train_mode=False
    )
    
    print(f"✅ Dataset initialized, starting feature computation...")
    
    # Process each item
    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    print(f"\nProcessing {len(item_info_df):,} items...")
    
    for idx, (_, item_row) in enumerate(tqdm(item_info_df.iterrows(), 
                                           total=len(item_info_df), 
                                           desc="Computing features")):
        item_id = str(item_row['item_id'])
        cache_file = cache_dir / f"{item_id}.pt"
        
        # Skip if already cached (unless force recompute)
        if not force_recompute and cache_file.exists():
            skipped += 1
            continue
        
        try:
            # Compute all features for this item
            features = dataset._process_item_features(item_id)
            
            if features is not None:
                # Save features to cache file
                torch.save(features, cache_file)
                successful += 1
            else:
                failed += 1
                if failed <= 5:  # Only print first few errors
                    print(f"\n⚠️  Failed to process item {item_id}")
                
        except Exception as e:
            failed += 1
            if failed <= 5:  # Only print first few errors
                print(f"\n⚠️  Error processing item {item_id}: {e}")
        
        # Print progress every 1000 items
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta_seconds = (len(item_info_df) - idx - 1) / rate
            eta_minutes = eta_seconds / 60
            
            print(f"\n  Progress: {idx + 1:,}/{len(item_info_df):,} "
                  f"({rate:.1f} items/sec, ETA: {eta_minutes:.1f}m)")
            print(f"  Success: {successful:,}, Failed: {failed:,}, Skipped: {skipped:,}")
    
    # Final statistics
    elapsed = time.time() - start_time
    total_processed = successful + failed
    
    print(f"\n" + "=" * 60)
    print("✅ CACHE PRECOMPUTATION COMPLETED")
    print("=" * 60)
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"📊 Results:")
    print(f"   Successfully cached: {successful:,}")
    print(f"   Failed: {failed:,}")
    print(f"   Skipped (already cached): {skipped:,}")
    if total_processed > 0:
        print(f"   Processing rate: {total_processed/elapsed:.1f} items/sec")
    
    # Storage statistics
    cache_files = list(cache_dir.glob("*.pt"))
    if cache_files:
        total_size = sum(f.stat().st_size for f in cache_files) / (1024*1024)
        avg_size = total_size / len(cache_files) * 1024  # KB per file
        print(f"💾 Storage:")
        print(f"   Files: {len(cache_files):,}")
        print(f"   Total size: {total_size:.1f} MB")
        print(f"   Average file size: {avg_size:.1f} KB")
        
        # Sample a cached file to show structure
        sample_file = cache_files[0]
        try:
            sample_features = torch.load(sample_file, map_location='cpu')
            print(f"📄 Sample feature structure ({sample_file.name}):")
            for key, value in sample_features.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"   {key}: {type(value)}")
        except Exception as e:
            print(f"   Could not sample features: {e}")
    
    if failed > 0:
        print(f"\n⚠️  {failed:,} items failed to process. Common issues:")
        print("   - Missing or corrupted image files")
        print("   - Incomplete item info data")
        print("   - Insufficient disk space")
        print("   - Memory limitations")
    
    print(f"\n🎯 Cache ready!")
    print(f"   Location: {cache_dir}")
    print(f"   Update your config to use this cache:")
    print(f"   cache_directory: '{cache_dir}'")
    print(f"\n🚀 Training will now be much faster with precomputed features!")


def main():
    parser = argparse.ArgumentParser(description="Precompute multimodal features cache")
    parser.add_argument('--config', type=str, default='configs/simple_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--force', action='store_true',
                        help='Recompute features even if already cached')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Maximum number of items to process (for testing)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: process only 100 items')
    
    args = parser.parse_args()
    
    if args.test:
        args.max_items = 100
        print("🧪 Test mode: processing only 100 items")
    
    precompute_features_cache(
        config_path=args.config,
        force_recompute=args.force,
        max_items=args.max_items
    )


if __name__ == "__main__":
    main()