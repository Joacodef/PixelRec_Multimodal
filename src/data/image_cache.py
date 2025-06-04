# src/data/image_cache.py
"""
Shared image cache manager for multimodal dataset
"""
import torch
from pathlib import Path
from typing import Optional, Dict, List
import os
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
import threading
import time

class SharedImageCache:
    """
    Memory-efficient image cache that:
    1. Saves all processed images to disk
    2. Keeps only a subset in memory (LRU cache)  
    3. Loads from disk when needed
    """
    
    def __init__(self, cache_path: Optional[str] = None, max_memory_items: int = 1000, strategy: str = 'hybrid'):
        """
        Args:
            cache_path: Directory to store cached tensor files
            max_memory_items: Maximum number of items to keep in memory
            strategy: Caching strategy ('hybrid', 'disk', 'memory', 'disabled')
        """
        self.cache_dir = Path(cache_path) if cache_path else None
        self.strategy = strategy.lower()
        
        # Adjust max_memory_items based on strategy
        if self.strategy == 'disk':
            self.max_memory_items = 0  # No memory cache for disk-only
        elif self.strategy == 'disabled':
            self.max_memory_items = 0
            self.cache_dir = None  # No disk cache either
        else:
            self.max_memory_items = max_memory_items
        
        # LRU cache for memory management
        self.memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()  # Thread safety
        
        # Statistics
        self.disk_hits = 0
        self.memory_hits = 0
        self.cache_misses = 0
        
        # Legacy compatibility - keep reference to memory cache as 'cache'
        self.cache = self.memory_cache
        
        if self.strategy == 'disabled':
            print("Image caching is DISABLED")
        elif self.strategy == 'disk':
            print(f"Using DISK-ONLY caching at {self.cache_dir}")
        elif self.strategy == 'memory':
            print(f"Using MEMORY-ONLY caching (max {self.max_memory_items} items)")
        else:  # hybrid
            print(f"Using HYBRID caching: disk at {self.cache_dir}, memory limit {self.max_memory_items}")
        
        
    def get(self, item_id: str) -> Optional[torch.Tensor]:
        """Get cached image tensor, loading from disk if needed"""
        if self.strategy == 'disabled':
            return None
        
        with self._lock:
            # Check memory cache first (if strategy allows)
            if self.strategy in ['hybrid', 'memory'] and item_id in self.memory_cache:
                # Move to end (most recently used)
                tensor = self.memory_cache.pop(item_id)
                self.memory_cache[item_id] = tensor
                self.memory_hits += 1
                return tensor
            
            # Try loading from disk (if strategy allows)
            if self.strategy in ['hybrid', 'disk'] and self.cache_dir:
                tensor_path = self.cache_dir / f"{item_id}.pt"
                if tensor_path.exists():
                    try:
                        tensor = torch.load(tensor_path, map_location='cpu')
                        # Add to memory cache if strategy allows
                        if self.strategy == 'hybrid':
                            self._add_to_memory_cache(item_id, tensor)
                        self.disk_hits += 1
                        return tensor
                    except Exception as e:
                        print(f"Warning: Could not load cached tensor {tensor_path}: {e}")
            
            self.cache_misses += 1
            return None
    
    def set(self, item_id: str, tensor: torch.Tensor):
        """Set cached image tensor (saves to disk and optionally keeps in memory)"""
        if self.strategy == 'disabled':
            return
        
        # Save to disk if strategy allows
        if self.strategy in ['hybrid', 'disk'] and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tensor_path = self.cache_dir / f"{item_id}.pt"
            try:
                torch.save(tensor, tensor_path)
            except Exception as e:
                print(f"Warning: Could not save tensor to {tensor_path}: {e}")
        
        # Add to memory cache if strategy allows
        if self.strategy in ['hybrid', 'memory']:
            with self._lock:
                self._add_to_memory_cache(item_id, tensor)
    
    def _add_to_memory_cache(self, item_id: str, tensor: torch.Tensor):
        """Add item to memory cache with LRU eviction"""
        # Remove if already exists (to update position)
        if item_id in self.memory_cache:
            del self.memory_cache[item_id]
        
        # Add to end (most recent)
        self.memory_cache[item_id] = tensor
        
        # Evict oldest items if over limit
        while len(self.memory_cache) > self.max_memory_items:
            oldest_item = next(iter(self.memory_cache))
            del self.memory_cache[oldest_item]
    
    def load_from_disk(self):
        """Load information about cached files on disk (but don't load into memory)"""
        if self.cache_dir and self.cache_dir.exists():
            cached_files = list(self.cache_dir.glob("*.pt"))
            print(f"Found {len(cached_files)} cached images on disk. Will load on demand to save memory.")
        else:
            if self.cache_dir:
                print(f"Cache directory {self.cache_dir} not found. Starting with empty cache.")
    
    def precompute_all_images(
        self,
        item_ids: List[str],
        image_folder: str,
        image_processor,
        force_recompute: bool = False,
        batch_size: int = 100
    ):
        """
        Precompute images and save to disk in batches.
        Does NOT load all into memory at once to avoid memory issues.
        """
        if not self.cache_dir:
            print("Warning: No cache directory specified. Cannot precompute to disk.")
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter items that need processing
        items_to_process = []
        if force_recompute:
            items_to_process = item_ids
        else:
            for item_id in item_ids:
                tensor_path = self.cache_dir / f"{item_id}.pt"
                if not tensor_path.exists():
                    items_to_process.append(item_id)
        
        if not items_to_process:
            print("All images already cached on disk.")
            return
        
        print(f"Processing {len(items_to_process)} images to disk cache (batch size: {batch_size})...")
        
        # Get placeholder size
        placeholder_size = self._get_placeholder_size(image_processor)
        
        # Process in batches to avoid memory buildup
        processed_count = 0
        for i in tqdm(range(0, len(items_to_process), batch_size), desc="Processing image batches"):
            batch_items = items_to_process[i:i + batch_size]
            batch_tensors = {}
            
            # Process batch
            for item_id in batch_items:
                tensor = self._process_single_image(item_id, image_folder, image_processor, placeholder_size)
                if tensor is not None:
                    batch_tensors[item_id] = tensor
            
            # Save batch to disk
            for item_id, tensor in batch_tensors.items():
                tensor_path = self.cache_dir / f"{item_id}.pt"
                try:
                    torch.save(tensor, tensor_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Error saving {item_id}: {e}")
            
            # Clear batch from memory
            del batch_tensors
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
        
        print(f"Successfully processed {processed_count} images to {self.cache_dir}")
    
    def _process_single_image(self, item_id: str, image_folder: str, image_processor, placeholder_size: tuple) -> Optional[torch.Tensor]:
        """Process a single image and return tensor"""
        base_path = os.path.join(image_folder, str(item_id))
        image_path = None
        
        # Find image file
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.webp', '.WEBP']:
            current_path = f"{base_path}{ext}"
            if os.path.exists(current_path):
                image_path = current_path
                break
        
        try:
            if image_path is None:
                raise FileNotFoundError(f"Image for {item_id} not found")
            
            image = Image.open(image_path).convert('RGB')
            processed_output = image_processor(images=image, return_tensors='pt')
            
            # Extract tensor
            if isinstance(processed_output, dict) and 'pixel_values' in processed_output:
                img_tensor = processed_output['pixel_values']
                if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.squeeze(0)
            elif torch.is_tensor(processed_output):
                img_tensor = processed_output
                if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.squeeze(0)
            else:
                raise ValueError("Unexpected processor output format")
            
            # Ensure correct format
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
            elif img_tensor.ndim == 3 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            
            if not (img_tensor.ndim == 3 and img_tensor.shape[0] == 3):
                img_tensor = torch.zeros(3, placeholder_size[0], placeholder_size[1])
            
            return img_tensor
            
        except Exception as e:
            # Return placeholder on error
            return torch.zeros(3, placeholder_size[0], placeholder_size[1])
    
    def _get_placeholder_size(self, image_processor) -> tuple:
        """Get placeholder size from image processor"""
        placeholder_size = (224, 224)
        try:
            if hasattr(image_processor, 'size'):
                processor_size = image_processor.size
                if isinstance(processor_size, dict) and 'shortest_edge' in processor_size:
                    size_val = processor_size['shortest_edge']
                    placeholder_size = (size_val, size_val)
                elif isinstance(processor_size, dict) and 'height' in processor_size:
                    placeholder_size = (processor_size['height'], processor_size['width'])
                elif isinstance(processor_size, (tuple, list)) and len(processor_size) >= 2:
                    placeholder_size = (processor_size[0], processor_size[1])
                elif isinstance(processor_size, int):
                    placeholder_size = (processor_size, processor_size)
        except Exception:
            pass
        return placeholder_size
    
    def save_to_disk(self):
        """
        Save current in-memory cache to disk.
        Note: In the new implementation, items are saved immediately when set(),
        so this method mainly exists for compatibility.
        """
        if self.cache_dir and self.memory_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving {len(self.memory_cache)} in-memory items to disk...")
            
            saved_count = 0
            for item_id, tensor in tqdm(self.memory_cache.items(), desc="Saving to disk"):
                tensor_path = self.cache_dir / f"{item_id}.pt"
                try:
                    torch.save(tensor, tensor_path)
                    saved_count += 1
                except Exception as e:
                    print(f"Warning: Could not save cached image for item_id {item_id}: {e}")
            
            print(f"Saved {saved_count} items to {self.cache_dir}")
        elif not self.memory_cache:
            print("In-memory cache is empty. Nothing to save.")
        else:
            print("Cache directory not specified. Cannot save to disk.")
    
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        total_requests = self.memory_hits + self.disk_hits + self.cache_misses
        hit_rate = (self.memory_hits + self.disk_hits) / max(1, total_requests)
        
        return {
            'memory_items': len(self.memory_cache),
            'max_memory_items': self.max_memory_items,
            'memory_hits': self.memory_hits,
            'disk_hits': self.disk_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_hit_rate': self.memory_hits / max(1, total_requests),
            'disk_hit_rate': self.disk_hits / max(1, total_requests)
        }
    
    def clear_memory_cache(self):
        """Clear memory cache but keep disk cache"""
        with self._lock:
            self.memory_cache.clear()
            print("Cleared memory cache. Disk cache remains intact.")
    
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print(f"\n=== Cache Statistics ===")
        print(f"Memory items: {stats['memory_items']}/{stats['max_memory_items']}")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print(f"  Memory hits: {stats['memory_hits']} ({stats['memory_hit_rate']:.1%})")
        print(f"  Disk hits: {stats['disk_hits']} ({stats['disk_hit_rate']:.1%})")
        print(f"  Cache misses: {stats['cache_misses']}")