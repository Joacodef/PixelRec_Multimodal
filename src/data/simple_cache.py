# src/data/simple_cache.py
"""
This module contains the SimpleFeatureCache class, which is designed to store and
retrieve pre-computed feature tensors. It supports both an in-memory cache with a
Least Recently Used (LRU) eviction policy and optional persistence to disk.
The cache paths are constructed to be model-specific, preventing conflicts
between different experimental configurations. The class is also thread-safe,
making it suitable for use with multiprocessing DataLoaders.
"""
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict
import threading


class SimpleFeatureCache:
    """
    A unified cache for item features with model-specific storage paths.

    This class manages the caching of processed item features, including image,
    text, and numerical tensors. It uses an in-memory OrderedDict to implement an
    LRU eviction policy and can optionally write features to disk for persistence
    across different runs. It is designed to be thread-safe.
    """

    def __init__(
        self,
        vision_model: str,
        language_model: str,
        base_cache_dir: str = "cache",
        max_memory_items: int = 1000,
        use_disk: bool = False
    ):
        """
        Initializes the feature cache.

        Args:
            vision_model (str): The name of the vision model being used (e.g., 'resnet').
            language_model (str): The name of the language model being used (e.g., 'sentence-bert').
            base_cache_dir (str): The parent directory for the cache. This can be a base path
                                  or the full model-specific path.
            max_memory_items (int): The maximum number of items to retain in the in-memory cache.
            use_disk (bool): A flag to enable or disable persisting the cache to disk.
        """
        self.vision_model = vision_model
        self.language_model = language_model
        self.max_memory_items = max_memory_items
        self.use_disk = use_disk

        cache_name = f"vision_{vision_model or 'none'}_lang_{language_model or 'none'}"
        provided_path = Path(base_cache_dir)

        # This logic prevents creating nested cache directories.
        # It checks if the provided path already ends with the model-specific name.
        if provided_path.name == cache_name:
            # If the provided path is already model-specific, use it directly.
            self.cache_dir = provided_path
        else:
            # Otherwise, create the model-specific subdirectory within the provided base path.
            self.cache_dir = provided_path / cache_name

        # The base directory is the parent of the final active cache directory.
        self.base_cache_dir = self.cache_dir.parent
        
        # An OrderedDict is used for the in-memory cache to efficiently manage the LRU policy.
        self.memory_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        # A threading lock ensures that cache operations are atomic and thread-safe.
        self._lock = threading.Lock()

        # Initializes statistics to track cache performance.
        self.hits = 0
        self.misses = 0

        # Creates the on-disk cache directory if disk persistence is enabled.
        if self.use_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepares the cache's state for serialization (pickling).

        This method is called by pickle. It excludes the threading lock from the
        state dictionary, as lock objects cannot be pickled.

        Returns:
            Dict[str, Any]: A dictionary representing the object's state.
        """
        state = self.__dict__.copy()
        if '_lock' in state:
            del state['_lock']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores the cache's state after deserialization (unpickling).

        This method is called by pickle. It restores the object's attributes and
        re-initializes the threading lock, ensuring the unpickled object is
        thread-safe.

        Args:
            state (Dict[str, Any]): The dictionary of state to restore.
        """
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def get(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieves cached features for a given item ID.

        The method first checks the in-memory cache. If the item is found, it is
        moved to the end of the OrderedDict to mark it as most recently used. If
        not in memory and disk usage is enabled, it checks for a corresponding
        file on disk.

        Args:
            item_id (str): The unique identifier for the item.

        Returns:
            Optional[Dict[str, torch.Tensor]]: A dictionary of feature tensors
                                               if the item is found in the cache,
                                               otherwise None.
        """
        with self._lock:
            # First, check the in-memory cache for the item.
            if item_id in self.memory_cache:
                # Pop and re-insert the item to move it to the end (most recently used).
                features = self.memory_cache.pop(item_id)
                self.memory_cache[item_id] = features
                self.hits += 1
                return features

            # If not in memory, check the disk cache if enabled.
            if self.use_disk:
                cache_file = self.cache_dir / f"{item_id}.pt"
                if cache_file.exists():
                    try:
                        features = torch.load(cache_file, map_location='cpu')
                        # Load the item from disk into the in-memory cache.
                        self._add_to_memory(item_id, features)
                        self.hits += 1
                        return features
                    except Exception as e:
                        print(f"Warning: SimpleFeatureCache could not load {cache_file}: {e}")
            
            # If the item is not found in memory or on disk, it is a cache miss.
            self.misses += 1
            return None

    def set(self, item_id: str, features: Dict[str, torch.Tensor], force_recompute: bool = False) -> None:
        """
        Caches the features for a given item ID.

        This method adds the features to the in-memory cache and, if disk
        persistence is enabled, saves them to a file.

        Args:
            item_id (str): The unique identifier for the item.
            features (Dict[str, torch.Tensor]): A dictionary of the item's
                                                 processed feature tensors.
            force_recompute (bool): If True, forces the cache to overwrite an
                                    existing file on disk.
        """
        with self._lock:
            self._add_to_memory(item_id, features)

            if self.use_disk:
                cache_file = self.cache_dir / f"{item_id}.pt"
                # Only save to disk if forcing recompute or if the file doesn't already exist.
                if force_recompute or not cache_file.exists():
                    try:
                        torch.save(features, cache_file)
                    except Exception as e:
                        print(f"Warning: SimpleFeatureCache could not save {cache_file}: {e}")


    def _add_to_memory(self, item_id: str, features: Dict[str, torch.Tensor]) -> None:
        """
        A helper method to add an item to the in-memory cache and manage eviction.

        This method assumes that the caller already holds the thread lock. It adds
        the new item and, if the cache exceeds its maximum size, evicts the least
        recently used item.

        Args:
            item_id (str): The identifier of the item to add.
            features (Dict[str, torch.Tensor]): The feature dictionary for the item.
        """
        # If the item already exists, remove it first to update its position.
        if item_id in self.memory_cache:
            self.memory_cache.pop(item_id)
        self.memory_cache[item_id] = features
        # If the cache size exceeds the maximum, evict the oldest item.
        while len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False)

    def clear(self) -> None:
        """
        Clears all items from the in-memory cache.

        Note: This method does not affect any files saved to the disk cache.
        """
        with self._lock:
            self.memory_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """
        Returns a dictionary of current cache performance statistics.

        Returns:
            Dict[str, Any]: A dictionary containing statistics such as hit rate,
                            item counts, and configuration settings.
        """
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
                'cache_directory_active': str(self.cache_dir)
            }

    def print_stats(self) -> None:
        """Prints a formatted summary of the cache's current statistics."""
        stats = self.stats()
        print("Cache Status:")
        print(f"  - Model Combination: {self.vision_model}_{self.language_model}")
        print(f"  - Active Directory: {stats['cache_directory_active']}")
        print(f"  - In-Memory: {stats['memory_items']}/{stats['max_memory_items']} items")
        print(f"  - Performance: {stats['hits']} hits, {stats['misses']} misses (Hit Rate: {stats['hit_rate']:.2%})")
        print(f"  - Disk Persistence: {'Enabled' if stats['use_disk'] else 'Disabled'}")