# src/data/simple_cache.py
"""
Refactored simple feature caching system with model-specific path construction.
"""
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict
# import pickle # Not directly used for lock, but torch.save/load use it.
import threading


class SimpleFeatureCache:
    """
    Simple unified cache for all item features (images + text + numerical).
    Constructs a model-specific cache path: base_cache_dir/vision_model_language_model/
    Supports optional disk persistence with LRU memory management and is
    compatible with multiprocessing DataLoaders.
    """

    def __init__(
        self,
        vision_model: str,              # Name of the vision model
        language_model: str,            # Name of the language model
        base_cache_dir: str = "cache",  # Base directory for all model caches
        max_memory_items: int = 1000,
        use_disk: bool = False
    ):
        """
        Initializes the feature cache.

        Args:
            vision_model: Name of the vision model (e.g., 'resnet', 'clip').
            language_model: Name of the language model (e.g., 'sentence-bert').
            base_cache_dir: Base directory where model-specific cache folders will be created.
            max_memory_items: Maximum number of items to keep in the in-memory cache.
            use_disk: Boolean flag to enable or disable persisting the cache to disk.
        """
        self.vision_model = vision_model
        self.language_model = language_model
        self.base_cache_dir = Path(base_cache_dir)

        # Construct the model-specific cache directory path
        cache_name = f"{self.vision_model}_{self.language_model}"
        self.cache_dir: Path = self.base_cache_dir / cache_name

        self.max_memory_items = max_memory_items
        self.use_disk = use_disk

        # In-memory LRU cache
        self.memory_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        # Threading lock for thread-safe access
        self._lock = threading.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0

        if self.use_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # print(f"SimpleFeatureCache: Using disk cache at {self.cache_dir}") # For debugging
        # else:
            # print(f"SimpleFeatureCache: Memory-only cache (max {max_memory_items} items) for {cache_name}") # For debugging

    def __getstate__(self) -> Dict[str, Any]:
        # Exclude lock for pickling
        state = self.__dict__.copy()
        if '_lock' in state:
            del state['_lock']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Re-initialize lock after unpickling
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def get(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        # Retrieves cached features for a given item_id.
        with self._lock:
            if item_id in self.memory_cache:
                features = self.memory_cache.pop(item_id)
                self.memory_cache[item_id] = features  # Move to end (most recently used)
                self.hits += 1
                return features

            if self.use_disk: # self.cache_dir is correctly model-specific
                cache_file = self.cache_dir / f"{item_id}.pt"
                if cache_file.exists():
                    try:
                        features = torch.load(cache_file, map_location='cpu')
                        self._add_to_memory(item_id, features) # Add to memory cache
                        self.hits += 1
                        return features
                    except Exception as e:
                        print(f"Warning: SimpleFeatureCache could not load {cache_file}: {e}")
            
            self.misses += 1
            return None

    def set(self, item_id: str, features: Dict[str, torch.Tensor]) -> None:
        # Caches features for a given item_id.
        with self._lock:
            self._add_to_memory(item_id, features)

            if self.use_disk: # self.cache_dir is correctly model-specific
                cache_file = self.cache_dir / f"{item_id}.pt"
                try:
                    torch.save(features, cache_file)
                except Exception as e:
                    print(f"Warning: SimpleFeatureCache could not save {cache_file}: {e}")

    def _add_to_memory(self, item_id: str, features: Dict[str, torch.Tensor]) -> None:
        # Adds an item to the memory cache, handling LRU eviction.
        # Assumes caller holds the lock.
        if item_id in self.memory_cache:
            self.memory_cache.pop(item_id)
        self.memory_cache[item_id] = features
        while len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False) # Evict oldest

    def clear(self) -> None:
        # Clears all items from the in-memory cache.
        with self._lock:
            self.memory_cache.clear()
            # Note: This does not clear the disk cache.
            # A separate method would be needed for that if desired.

    def stats(self) -> Dict[str, Any]:
        # Returns statistics about the cache.
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
            return {
                'memory_items': len(self.memory_cache),
                'max_memory_items': self.max_memory_items,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'use_disk': self.use_disk,
                'cache_directory_active': str(self.cache_dir) # The model-specific path
            }

    def print_stats(self) -> None:
        # Prints cache statistics.
        stats = self.stats()
        print(f"SimpleFeatureCache Stats ({self.vision_model}_{self.language_model}): "
              f"{stats['memory_items']}/{stats['max_memory_items']} in-memory. "
              f"Hit rate: {stats['hit_rate']:.2%}. "
              f"Disk: {'Enabled' if stats['use_disk'] else 'Disabled'} at {stats['cache_directory_active']}.")