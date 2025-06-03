# src/data/image_cache.py
"""
Shared image cache manager for multimodal dataset
"""
import torch
from pathlib import Path
import pickle
from typing import Dict, Optional
import os
from tqdm import tqdm
from PIL import Image


class SharedImageCache:
    """Manages a shared cache of processed images that can be used across dataset instances"""
    
    def __init__(self, cache_path: Optional[str] = None):
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        
    def load_from_disk(self):
        """Load cache from disk if available"""
        if self.cache_path and self.cache_path.exists():
            print(f"Loading image cache from {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"Loaded {len(self.cache)} cached images")
            
    def save_to_disk(self):
        """Save cache to disk"""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving image cache to {self.cache_path}")
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Saved {len(self.cache)} cached images")
            
    def get(self, item_id: str) -> Optional[torch.Tensor]:
        """Get cached image tensor"""
        return self.cache.get(item_id)
        
    def set(self, item_id: str, tensor: torch.Tensor):
        """Set cached image tensor"""
        self.cache[item_id] = tensor
        
    def precompute_all_images(
        self, 
        item_ids: list, 
        image_folder: str,
        image_processor,
        force_recompute: bool = False
    ):
        """Precompute all images in the dataset"""
        if not force_recompute and self.cache:
            print(f"Cache already contains {len(self.cache)} images. Skipping precomputation.")
            return
            
        print(f"Precomputing {len(item_ids)} images...")
        
        # Determine placeholder size
        placeholder_size = (224, 224)
        try:
            if hasattr(image_processor, 'size'):
                processor_size = image_processor.size
                if isinstance(processor_size, dict) and 'shortest_edge' in processor_size:
                    size_val = processor_size['shortest_edge']
                    placeholder_size = (size_val, size_val)
                elif isinstance(processor_size, (tuple, list)) and len(processor_size) >= 2:
                    placeholder_size = (processor_size[0], processor_size[1])
                elif isinstance(processor_size, int):
                    placeholder_size = (processor_size, processor_size)
        except Exception:
            pass
            
        for item_id in tqdm(item_ids, desc="Processing images"):
            if item_id in self.cache and not force_recompute:
                continue
                
            # Find image path
            base_path = os.path.join(image_folder, str(item_id))
            image_path_to_load = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                current_path = f"{base_path}{ext}"
                if os.path.exists(current_path):
                    image_path_to_load = current_path
                    break
            
            # Load and process image
            try:
                if image_path_to_load is None:
                    raise FileNotFoundError(f"Image for {item_id} not found.")
                image = Image.open(image_path_to_load).convert('RGB')
            except Exception:
                image = Image.new('RGB', placeholder_size, color='grey')
            
            # Process image
            try:
                processed_output = image_processor(images=image, return_tensors='pt')
            except Exception:
                self.cache[item_id] = torch.zeros(3, placeholder_size[0], placeholder_size[1])
                continue
                
            # Extract tensor
            image_tensor = None
            if isinstance(processed_output, dict) and 'pixel_values' in processed_output:
                image_tensor = processed_output['pixel_values']
                if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
                    image_tensor = image_tensor.squeeze(0)
            elif torch.is_tensor(processed_output):
                image_tensor = processed_output
                if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
                    image_tensor = image_tensor.squeeze(0)
                    
            if image_tensor is None:
                self.cache[item_id] = torch.zeros(3, placeholder_size[0], placeholder_size[1])
                continue
                
            # Handle tensor dimensions
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.repeat(3, 1, 1)
                
            if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
                self.cache[item_id] = torch.zeros(3, placeholder_size[0], placeholder_size[1])
            else:
                self.cache[item_id] = image_tensor