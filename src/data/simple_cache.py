# src/data/simple_cache.py
"""
Simplified feature caching system with pickling support for multiprocessing.
"""
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict
import pickle # Retained if used for disk cache logic, not for the lock itself
import threading


class SimpleFeatureCache:
    """
    Simple unified cache for all item features (images + text + numerical).
    Supports optional disk persistence with LRU memory management and is
    compatible with multiprocessing DataLoaders by correctly handling its lock.
    """
    
    def __init__(
        self, 
        max_memory_items: int = 1000,
        cache_dir: Optional[str] = None,
        use_disk: bool = False
    ):
        """
        Initializes the feature cache.

        Args:
            max_memory_items: Maximum number of items to keep in the in-memory cache.
            cache_dir: Directory path for disk-based cache. Required if use_disk is True.
            use_disk: Boolean flag to enable or disable persisting the cache to disk.
        """
        self.max_memory_items = max_memory_items
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_disk = use_disk
        
        # In-memory LRU cache using OrderedDict to track item usage order.
        self.memory_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        # Threading lock for ensuring thread-safe access to cache structures.
        self._lock = threading.Lock()
        
        # Statistics for cache performance monitoring.
        self.hits = 0
        self.misses = 0
        
        if self.use_disk and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # print(f"SimpleFeatureCache: Using disk cache at {self.cache_dir}") # Optional: for debugging
        # else:
            # print(f"SimpleFeatureCache: Memory-only cache (max {max_memory_items} items)") # Optional: for debugging

    def __getstate__(self) -> Dict[str, Any]:
        # Custom method to define the state of the object for pickling.
        # The '_lock' attribute is excluded as thread locks are not picklable.
        state = self.__dict__.copy()
        if '_lock' in state:
            del state['_lock']
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Custom method to restore the state of the object after unpickling.
        # The '_lock' attribute is re-initialized as a new lock instance.
        self.__dict__.update(state)
        self._lock = threading.Lock()
    
    def get(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        # Retrieves cached features for a given item_id.
        # Checks memory cache first, then disk cache if enabled.
        # Updates LRU status for memory cache hits.
        with self._lock: # Ensures thread-safe access to cache.
            # Check memory cache.
            if item_id in self.memory_cache:
                # Move item to the end (most recently used) to maintain LRU order.
                features = self.memory_cache.pop(item_id)
                self.memory_cache[item_id] = features
                self.hits += 1
                return features
            
            # Try disk cache if enabled and item not in memory.
            if self.use_disk and self.cache_dir:
                cache_file = self.cache_dir / f"{item_id}.pt" # Assumes PyTorch tensors are saved.
                if cache_file.exists():
                    try:
                        # Loads features from disk. map_location='cpu' is safer for general use.
                        features = torch.load(cache_file, map_location='cpu')
                        # Add loaded features to memory cache.
                        self._add_to_memory(item_id, features)
                        self.hits += 1
                        return features
                    except Exception as e:
                        # Logs a warning if loading from disk fails.
                        print(f"Warning: SimpleFeatureCache could not load {cache_file}: {e}")
            
            self.misses += 1
            return None # Returns None if item is not found in cache.
    
    def set(self, item_id: str, features: Dict[str, torch.Tensor]) -> None:
        # Caches features for a given item_id.
        # Adds to memory cache and saves to disk if enabled.
        with self._lock: # Ensures thread-safe access.
            # Add to memory cache, respecting LRU policy.
            self._add_to_memory(item_id, features)
            
            # Save to disk cache if enabled.
            if self.use_disk and self.cache_dir:
                cache_file = self.cache_dir / f"{item_id}.pt"
                try:
                    torch.save(features, cache_file)
                except Exception as e:
                    # Logs a warning if saving to disk fails.
                    print(f"Warning: SimpleFeatureCache could not save {cache_file}: {e}")
    
    def _add_to_memory(self, item_id: str, features: Dict[str, torch.Tensor]) -> None:
        # Internal method to add an item to the memory cache, handling LRU eviction.
        # This method assumes it's called within a context that already holds the lock.
        
        # Remove item if it already exists to update its position (as most recently used).
        if item_id in self.memory_cache:
            self.memory_cache.pop(item_id)
        
        # Add item to the end of the OrderedDict.
        self.memory_cache[item_id] = features
        
        # Evict oldest items if memory cache exceeds its maximum size.
        while len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False) # popitem(last=False) removes in FIFO order (oldest).
    
    def clear(self) -> None:
        # Clears all items from the in-memory cache.
        with self._lock: # Ensures thread-safe operation.
            self.memory_cache.clear()
            # Optionally, one might also clear disk cache here if desired,
            # but current implementation only clears memory.
    
    def stats(self) -> Dict[str, Any]:
        # Returns statistics about the cache's current state and performance.
        with self._lock: # Ensures consistent reading of stats.
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
            
            return {
                'memory_items': len(self.memory_cache),
                'max_memory_items': self.max_memory_items,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'use_disk': self.use_disk,
                'cache_dir': str(self.cache_dir) if self.cache_dir else None
            }
    
    def print_stats(self) -> None:
        # Prints a summary of cache statistics to the console.
        stats = self.stats() # Retrieves current statistics.
        print(f"SimpleFeatureCache Stats: {stats['memory_items']}/{stats['max_memory_items']} in-memory items. "
              f"Hit rate: {stats['hit_rate']:.2%} ({stats['hits']} hits, {stats['misses']} misses). "
              f"Disk usage: {'Enabled' if stats['use_disk'] else 'Disabled'}.")