# Suggested path: src/data/feature_cache.py (or a similar appropriate location)

import torch
from pathlib import Path
from typing import Optional, Dict, List, Any
import os
from tqdm import tqdm
from collections import OrderedDict
import threading
import time # For potential use in precomputation, similar to image_cache

class ProcessedFeatureCache:
    """
    Cache for processed item features (e.g., tokenized text, numerical features).
    Supports disk persistence and an optional in-memory LRU cache.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        max_memory_items: int = 5000,
        strategy: str = 'hybrid'
    ):
        """
        Initializes the ProcessedFeatureCache.

        Args:
            cache_path: Directory to store cached feature tensor files.
            max_memory_items: Maximum number of feature sets to keep in the in-memory LRU cache.
            strategy: Caching strategy ('hybrid', 'disk', 'memory', 'disabled').
                      'hybrid': Uses both disk and memory LRU cache.
                      'disk': Uses only disk cache; items loaded on demand.
                      'memory': Uses only memory LRU cache (not persistent unless saved).
                      'disabled': Caching is turned off.
        """
        self.cache_dir: Optional[Path] = Path(cache_path) if cache_path else None
        self.strategy: str = strategy.lower()

        # Adjust max_memory_items based on strategy
        if self.strategy == 'disk':
            self.max_memory_items: int = 0  # No memory cache for disk-only
        elif self.strategy == 'disabled':
            self.max_memory_items = 0
            self.cache_dir = None  # No disk cache either
        else:
            self.max_memory_items = max_memory_items

        # LRU cache for in-memory feature management
        self.memory_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        # Lock for thread-safe operations on the cache
        self._lock: Optional[threading.Lock] = None # Initialized lazily

        # Statistics
        self.disk_hits: int = 0
        self.memory_hits: int = 0
        self.cache_misses: int = 0

        if self.strategy == 'disabled':
            print("ProcessedFeatureCache: Caching is DISABLED.")
        elif self.strategy == 'disk' and self.cache_dir:
            print(f"ProcessedFeatureCache: Using DISK-ONLY caching at {self.cache_dir}.")
        elif self.strategy == 'memory':
            print(f"ProcessedFeatureCache: Using MEMORY-ONLY caching (max {self.max_memory_items} items).")
        elif self.strategy == 'hybrid' and self.cache_dir:
            print(f"ProcessedFeatureCache: Using HYBRID caching: disk at {self.cache_dir}, memory limit {self.max_memory_items}.")
        elif (self.strategy == 'disk' or self.strategy == 'hybrid') and not self.cache_dir:
            print(f"ProcessedFeatureCache: Strategy is '{self.strategy}' but cache_path is None. Caching will be ineffective for disk operations.")
            # Fallback to behave like 'disabled' or 'memory' depending on max_memory_items
            if self.max_memory_items == 0 : self.strategy = 'disabled' # Effectively
            else: self.strategy = 'memory' # Effectively
            print(f"ProcessedFeatureCache: Fallback strategy to '{self.strategy}'.")


    @property
    def lock(self) -> threading.Lock:
        """
        Provides thread-safe access to the lock, initializing it if it doesn't exist.
        Ensures lock is created in the current process context, important for multiprocessing.
        """
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    def get(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieves cached processed features for a given item_id.
        Checks memory cache first, then disk cache, based on the configured strategy.

        Args:
            item_id: The unique identifier for the item.

        Returns:
            A dictionary of feature tensors if found in cache, otherwise None.
        """
        if self.strategy == 'disabled':
            self.cache_misses +=1 # Still count as miss if queried
            return None

        with self.lock:
            # 1. Check memory cache (if strategy allows)
            if self.strategy in ['hybrid', 'memory'] and self.max_memory_items > 0:
                if item_id in self.memory_cache:
                    # Move to end (most recently used)
                    features_dict = self.memory_cache.pop(item_id)
                    self.memory_cache[item_id] = features_dict
                    self.memory_hits += 1
                    return features_dict

            # 2. Try loading from disk (if strategy allows and cache_dir is set)
            if self.strategy in ['hybrid', 'disk'] and self.cache_dir:
                feature_file_path = self.cache_dir / f"{item_id}.pt"
                if feature_file_path.exists():
                    try:
                        features_dict = torch.load(feature_file_path, map_location='cpu')
                        # Add to memory cache if hybrid strategy and memory cache has space
                        if self.strategy == 'hybrid' and self.max_memory_items > 0:
                            self._add_to_memory_cache(item_id, features_dict)
                        self.disk_hits += 1
                        return features_dict
                    except Exception as e:
                        # Log error if a cached file cannot be loaded.
                        print(f"ProcessedFeatureCache: Warning - Could not load cached feature file {feature_file_path}: {e}")
                        # Optionally, remove corrupted file: os.remove(feature_file_path)

            self.cache_misses += 1
            return None

    def set(self, item_id: str, features_dict: Dict[str, torch.Tensor]) -> None:
        """
        Stores processed features for an item in the cache.
        Saves to disk and/or memory based on the configured strategy.

        Args:
            item_id: The unique identifier for the item.
            features_dict: A dictionary of feature tensors to cache.
        """
        if self.strategy == 'disabled':
            return

        # 1. Save to disk (if strategy allows and cache_dir is set)
        if self.strategy in ['hybrid', 'disk'] and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            feature_file_path = self.cache_dir / f"{item_id}.pt"
            try:
                torch.save(features_dict, feature_file_path)
            except Exception as e:
                 # Log error if features cannot be saved to disk.
                print(f"ProcessedFeatureCache: Warning - Could not save feature file to {feature_file_path}: {e}")

        # 2. Add to memory cache (if strategy allows and memory cache has space)
        if self.strategy in ['hybrid', 'memory'] and self.max_memory_items > 0:
            with self.lock:
                self._add_to_memory_cache(item_id, features_dict)

    def _add_to_memory_cache(self, item_id: str, features_dict: Dict[str, torch.Tensor]) -> None:
        """
        Adds an item's features to the in-memory LRU cache and evicts if over capacity.
        This method should be called within a lock.

        Args:
            item_id: The item's unique identifier.
            features_dict: The dictionary of features to cache.
        """
        # Remove if already exists (to update its position as most recently used)
        if item_id in self.memory_cache:
            self.memory_cache.pop(item_id)

        self.memory_cache[item_id] = features_dict

        # Evict oldest items if the cache exceeds its maximum size
        while len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False) # Pop the oldest item

    def load_from_disk_meta(self) -> None:
        """
        Scans the cache directory to count existing feature files.
        This can be used to understand cache population without loading all data into memory.
        """
        if self.strategy != 'disabled' and self.cache_dir and self.cache_dir.exists():
            try:
                cached_files = list(self.cache_dir.glob("*.pt"))
                print(f"ProcessedFeatureCache: Found {len(cached_files)} cached feature files on disk in {self.cache_dir}. Items will be loaded on demand.")
            except Exception as e:
                print(f"ProcessedFeatureCache: Error scanning cache directory {self.cache_dir}: {e}")
        elif self.strategy != 'disabled' and self.cache_dir:
            print(f"ProcessedFeatureCache: Cache directory {self.cache_dir} not found. Starting with an empty disk cache.")

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves statistics about cache usage.

        Returns:
            A dictionary containing cache statistics (hits, misses, hit rate, etc.).
        """
        total_requests = self.memory_hits + self.disk_hits + self.cache_misses
        hit_rate = (self.memory_hits + self.disk_hits) / max(1, total_requests)
        memory_hit_rate = self.memory_hits / max(1, total_requests)
        disk_hit_rate = self.disk_hits / max(1, total_requests)

        return {
            'memory_items': len(self.memory_cache),
            'max_memory_items': self.max_memory_items,
            'memory_hits': self.memory_hits,
            'disk_hits': self.disk_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'overall_hit_rate': hit_rate,
            'memory_hit_rate_contribution': memory_hit_rate,
            'disk_hit_rate_contribution': disk_hit_rate,
            'cache_directory': str(self.cache_dir) if self.cache_dir else "N/A",
            'strategy': self.strategy
        }

    def print_stats(self) -> None:
        """Prints formatted cache statistics to the console."""
        stats = self.get_stats()
        print("\n=== ProcessedFeatureCache Statistics ===")
        print(f"  Strategy: {stats['strategy']}")
        print(f"  Cache Directory: {stats['cache_directory']}")
        print(f"  Memory Items: {stats['memory_items']} / {stats['max_memory_items']}")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Overall Hit Rate: {stats['overall_hit_rate']:.2%}")
        print(f"    Memory Hits: {stats['memory_hits']} (Contribution: {stats['memory_hit_rate_contribution']:.1%})")
        print(f"    Disk Hits: {stats['disk_hits']} (Contribution: {stats['disk_hit_rate_contribution']:.1%})")
        print(f"    Cache Misses: {stats['cache_misses']}")
        print("=" * 36)

    def clear_memory_cache(self) -> None:
        """Clears the in-memory LRU cache. Disk cache remains intact."""
        with self.lock:
            self.memory_cache.clear()
        print("ProcessedFeatureCache: In-memory cache cleared. Disk cache remains.")

    def clear_disk_cache(self) -> None:
        """
        Deletes all .pt files from the disk cache directory.
        Warning: This operation is irreversible.
        """
        if self.strategy != 'disabled' and self.cache_dir and self.cache_dir.exists():
            print(f"ProcessedFeatureCache: WARNING - Attempting to clear disk cache at {self.cache_dir}.")
            confirmation = input("Are you sure you want to delete all *.pt files in this directory? (yes/no): ")
            if confirmation.lower() == 'yes':
                deleted_count = 0
                error_count = 0
                for file_path in self.cache_dir.glob("*.pt"):
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"ProcessedFeatureCache: Error deleting file {file_path}: {e}")
                        error_count +=1
                print(f"ProcessedFeatureCache: Cleared {deleted_count} files from disk cache. {error_count} errors.")
            else:
                print("ProcessedFeatureCache: Disk cache clearing aborted by user.")
        elif self.strategy != 'disabled':
            print(f"ProcessedFeatureCache: Disk cache directory {self.cache_dir} not found or not configured. Nothing to clear.")


    def __getstate__(self) -> Dict:
        """
        Prepares the object's state for pickling by removing unpicklable attributes like locks.
        """
        state = self.__dict__.copy()
        state.pop('_lock', None) # Remove lock before pickling
        return state

    def __setstate__(self, state: Dict) -> None:
        """
        Restores the object's state after unpickling.
        The lock will be lazily re-initialized when the .lock property is accessed.
        """
        self.__dict__.update(state)
        self._lock = None # Ensure lock is None, to be re-created by the property