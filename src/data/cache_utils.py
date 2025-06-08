# src/data/cache_utils.py
"""
Provides command-line and programmatic utilities for managing feature caches.

This module contains a set of functions to list, inspect, and clear the
on-disk feature caches that are generated during data preprocessing and training.
It helps maintain an organized cache directory and provides visibility into the
storage used by different model configurations.
"""
from pathlib import Path
import torch
import shutil
from typing import Dict, List, Tuple


def list_available_caches(base_cache_dir: str = "cache") -> Dict[str, Dict]:
    """
    Scans the base cache directory and lists all valid, non-empty cache subdirectories.

    A valid cache directory is expected to be named using the format
    '<vision_model>_<language_model>'. The function returns detailed information
    for each valid cache found.

    Args:
        base_cache_dir (str): The path to the base directory containing all caches.

    Returns:
        Dict[str, Dict]: A dictionary where keys are the cache names (e.g.,
                         'resnet_sentence-bert') and values are dictionaries
                         containing details such as model names, file count,
                         and total size.
    """
    cache_dir = Path(base_cache_dir)
    available_caches = {}
    
    if not cache_dir.exists():
        return available_caches
    
    # Iterates through all subdirectories in the base cache directory.
    for model_dir in cache_dir.iterdir():
        # A directory is considered a potential cache if it is a directory and its name contains an underscore.
        if model_dir.is_dir() and '_' in model_dir.name:
            try:
                # Parses the vision and language model names from the directory name.
                parts = model_dir.name.split('_')
                if len(parts) >= 2:
                    vision_model = parts[0]
                    language_model = '_'.join(parts[1:])
                    
                    # Counts the number of PyTorch cache files (.pt) in the directory.
                    cache_files = list(model_dir.glob("*.pt"))
                    
                    # Only includes non-empty caches in the result.
                    if cache_files:
                        # Calculates the total size of the cache in megabytes.
                        total_size = sum(f.stat().st_size for f in cache_files) / (1024*1024)
                        
                        # Loads a sample file to inspect the structure of the cached features.
                        sample_features = {}
                        try:
                            sample_data = torch.load(cache_files[0], map_location='cpu')
                            if isinstance(sample_data, dict):
                                sample_features = {k: str(v.shape) if hasattr(v, 'shape') else str(type(v)) 
                                                 for k, v in sample_data.items()}
                        except:
                            pass
                        
                        available_caches[model_dir.name] = {
                            'vision_model': vision_model,
                            'language_model': language_model,
                            'path': str(model_dir),
                            'files': len(cache_files),
                            'size_mb': total_size,
                            'sample_features': sample_features
                        }
                        
            except Exception as e:
                print(f"Warning: Error processing cache directory {model_dir}: {e}")
    
    return available_caches


def print_cache_summary(base_cache_dir: str = "cache"):
    """
    Prints a formatted, human-readable summary of all available caches.

    This function calls `list_available_caches` and presents the information
    in a structured format to the console.

    Args:
        base_cache_dir (str): The path to the base directory containing all caches.
    """
    
    print("📁 AVAILABLE FEATURE CACHES")
    print("=" * 50)
    
    caches = list_available_caches(base_cache_dir)
    
    if not caches:
        print("📭 No caches found")
        print(f"   Looked in: {Path(base_cache_dir).absolute()}")
        return
    
    total_size = 0
    total_files = 0
    
    for cache_name, info in caches.items():
        print(f"\n📂 {cache_name}")
        print(f"   Vision: {info['vision_model']}")
        print(f"   Language: {info['language_model']}")
        print(f"   Files: {info['files']:,}")
        print(f"   Size: {info['size_mb']:.1f} MB")
        
        if info['sample_features']:
            print(f"   Features: {', '.join(info['sample_features'].keys())}")
        
        total_size += info['size_mb']
        total_files += info['files']
    
    print(f"\n📊 TOTAL: {len(caches)} caches, {total_files:,} files, {total_size:.1f} MB")


def get_cache_path(vision_model: str, language_model: str, base_cache_dir: str = "cache") -> Path:
    """
    Constructs the standardized path for a model-specific cache directory.

    Args:
        vision_model (str): The name of the vision model.
        language_model (str): The name of the language model.
        base_cache_dir (str): The base directory for all caches.

    Returns:
        Path: A pathlib.Path object for the model-specific cache directory.
    """
    cache_name = f"{vision_model}_{language_model}"
    return Path(base_cache_dir) / cache_name


def cache_exists(vision_model: str, language_model: str, base_cache_dir: str = "cache") -> bool:
    """
    Checks if a valid, non-empty cache exists for a given model combination.

    Args:
        vision_model (str): The name of the vision model.
        language_model (str): The name of the language model.
        base_cache_dir (str): The base directory for all caches.

    Returns:
        bool: True if a non-empty cache exists for the model combination, False otherwise.
    """
    cache_path = get_cache_path(vision_model, language_model, base_cache_dir)
    return cache_path.exists() and any(cache_path.glob("*.pt"))


def get_cache_stats(vision_model: str, language_model: str, base_cache_dir: str = "cache") -> Dict:
    """
    Retrieves statistics for a specific model combination's cache.

    Args:
        vision_model (str): The name of the vision model.
        language_model (str): The name of the language model.
        base_cache_dir (str): The base directory for all caches.

    Returns:
        Dict: A dictionary containing statistics like file count, size, and existence.
    """
    cache_path = get_cache_path(vision_model, language_model, base_cache_dir)
    
    if not cache_path.exists():
        return {"exists": False}
    
    cache_files = list(cache_path.glob("*.pt"))
    
    if not cache_files:
        return {"exists": True, "files": 0, "size_mb": 0}
    
    total_size = sum(f.stat().st_size for f in cache_files) / (1024*1024)
    
    return {
        "exists": True,
        "files": len(cache_files),
        "size_mb": total_size,
        "path": str(cache_path)
    }


def clear_cache(vision_model: str, language_model: str, base_cache_dir: str = "cache") -> bool:
    """
    Deletes the cache directory for a specific model combination.

    Args:
        vision_model (str): The name of the vision model.
        language_model (str): The name of the language model.
        base_cache_dir (str): The base directory for all caches.

    Returns:
        bool: True if the cache was found and cleared, False otherwise.
    """
    cache_path = get_cache_path(vision_model, language_model, base_cache_dir)
    
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"✅ Cleared cache: {cache_path}")
        return True
    else:
        print(f"📭 Cache not found: {cache_path}")
        return False


def clear_all_caches(base_cache_dir: str = "cache") -> int:
    """
    Deletes all subdirectories within the base cache directory.

    Args:
        base_cache_dir (str): The base directory for all caches.

    Returns:
        int: The number of cache directories that were cleared.
    """
    cache_dir = Path(base_cache_dir)
    
    if not cache_dir.exists():
        print(f"📭 Cache directory not found: {cache_dir}")
        return 0
    
    cleared = 0
    for model_dir in cache_dir.iterdir():
        if model_dir.is_dir():
            shutil.rmtree(model_dir)
            print(f"✅ Cleared: {model_dir.name}")
            cleared += 1
    
    # Removes the base directory itself if it has become empty.
    if not any(cache_dir.iterdir()):
        cache_dir.rmdir()
        print(f"✅ Removed empty cache directory: {cache_dir}")
    
    return cleared


if __name__ == "__main__":
    """
    Defines the command-line interface for the cache management script.

    This block allows the script to be executed directly from the command line
    with arguments to perform actions like listing, clearing, or viewing stats
    for the feature caches.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage feature caches")
    parser.add_argument("--list", action="store_true", help="List available caches")
    parser.add_argument("--clear", type=str, help="Clear cache for a model combination (format: vision_language)")
    parser.add_argument("--clear_all", action="store_true", help="Clear all caches")
    parser.add_argument("--stats", type=str, help="Show stats for a model combination (format: vision_language)")
    parser.add_argument("--cache_dir", default="cache", help="Base cache directory")
    
    args = parser.parse_args()
    
    if args.list:
        print_cache_summary(args.cache_dir)
    
    elif args.clear:
        if '_' in args.clear:
            vision, language = args.clear.split('_', 1)
            clear_cache(vision, language, args.cache_dir)
        else:
            print("❌ Format should be: vision_language (e.g., resnet_sentence-bert)")
    
    elif args.clear_all:
        response = input("⚠️  Clear ALL caches? (y/N): ")
        if response.lower() == 'y':
            cleared = clear_all_caches(args.cache_dir)
            print(f"✅ Cleared {cleared} caches")
    
    elif args.stats:
        if '_' in args.stats:
            vision, language = args.stats.split('_', 1)
            stats = get_cache_stats(vision, language, args.cache_dir)
            print(f"📊 Cache stats for {vision} + {language}:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        else:
            print("❌ Format should be: vision_language (e.g., resnet_sentence-bert)")
    
    else:
        print_cache_summary(args.cache_dir)