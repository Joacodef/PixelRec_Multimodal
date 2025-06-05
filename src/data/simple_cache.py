# src/data/simple_cache.py
"""
Simplified feature caching system
"""
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict
import pickle
import threading


class SimpleFeatureCache:
    """
    Simple unified cache for all item features (images + text + numerical).
    Supports optional disk persistence with LRU memory management.
    """
    
    def __init__(
        self, 
        max_memory_items: int = 1000,
        cache_dir: Optional[str] = None,
        use_disk: bool = False
    ):
        """
        Args:
            max_memory_items: Maximum items to keep in memory
            cache_dir: Directory for disk cache (optional)
            use_disk: Whether to persist to disk
        """
        self.max_memory_items = max_memory_items
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_disk = use_disk
        
        # Simple LRU cache
        self.memory_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        self._lock = threading.Lock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        
        if self.use_disk and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"SimpleFeatureCache: Using disk cache at {self.cache_dir}")
        else:
            print(f"SimpleFeatureCache: Memory-only cache (max {max_memory_items} items)")
    
    def get(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached features for an item"""
        with self._lock:
            # Check memory cache first
            if item_id in self.memory_cache:
                # Move to end (most recently used)
                features = self.memory_cache.pop(item_id)
                self.memory_cache[item_id] = features
                self.hits += 1
                return features
            
            # Try disk cache if enabled
            if self.use_disk and self.cache_dir:
                cache_file = self.cache_dir / f"{item_id}.pt"
                if cache_file.exists():
                    try:
                        features = torch.load(cache_file, map_location='cpu')
                        # Add to memory cache
                        self._add_to_memory(item_id, features)
                        self.hits += 1
                        return features
                    except Exception as e:
                        print(f"Warning: Could not load {cache_file}: {e}")
            
            self.misses += 1
            return None
    
    def set(self, item_id: str, features: Dict[str, torch.Tensor]) -> None:
        """Cache features for an item"""
        with self._lock:
            # Add to memory cache
            self._add_to_memory(item_id, features)
            
            # Save to disk if enabled
            if self.use_disk and self.cache_dir:
                cache_file = self.cache_dir / f"{item_id}.pt"
                try:
                    torch.save(features, cache_file)
                except Exception as e:
                    print(f"Warning: Could not save {cache_file}: {e}")
    
    def _add_to_memory(self, item_id: str, features: Dict[str, torch.Tensor]) -> None:
        """Add item to memory cache with LRU eviction"""
        # Remove if already exists
        if item_id in self.memory_cache:
            del self.memory_cache[item_id]
        
        # Add to end (most recent)
        self.memory_cache[item_id] = features
        
        # Evict oldest items if over limit
        while len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear memory cache"""
        with self._lock:
            self.memory_cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(1, total_requests)
        
        return {
            'memory_items': len(self.memory_cache),
            'max_memory_items': self.max_memory_items,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'use_disk': self.use_disk
        }
    
    def print_stats(self) -> None:
        """Print cache statistics"""
        stats = self.stats()
        print(f"Cache Stats: {stats['memory_items']}/{stats['max_memory_items']} items, "
              f"Hit rate: {stats['hit_rate']:.2%} ({stats['hits']} hits, {stats['misses']} misses)")


# Backwards compatibility - can be removed after migration
SharedImageCache = SimpleFeatureCache
ProcessedFeatureCache = SimpleFeatureCache