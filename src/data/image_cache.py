# src/data/image_cache.py
"""
Shared image cache manager for multimodal dataset
"""
import torch
from pathlib import Path
from typing import Optional, Dict, List
import os
from tqdm import tqdm # Ensure tqdm is imported
from PIL import Image
from collections.abc import Mapping

class SharedImageCache:
    """Manages a shared cache of processed images that can be used across dataset instances"""
    
    def __init__(self, cache_path: Optional[str] = None): # cache_path now refers to a directory
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_dir = Path(cache_path) if cache_path else None # Treat cache_path as a directory
        
    def load_from_disk(self):
        """Load cache from disk if available by loading individual tensor files."""
        if self.cache_dir and self.cache_dir.exists() and self.cache_dir.is_dir():
            # This print should be outside tqdm or use tqdm.write if tqdm is active elsewhere
            print(f"Loading image cache from directory {self.cache_dir}")
            loaded_count = 0
            item_files = [f for f in self.cache_dir.iterdir() if f.suffix == '.pt']
            for item_file in tqdm(item_files, desc="Loading cached images"):
                item_id = item_file.stem # Get item_id from filename without extension
                try:
                    if item_id not in self.cache: # Only load if not already in memory
                        tensor = torch.load(item_file, map_location=torch.device('cpu')) # Load to CPU first
                        self.cache[item_id] = tensor
                    loaded_count += 1
                except Exception as e:
                    # Use tqdm.write if this method could be called when a tqdm bar is active
                    print(f"Warning: Could not load cached image {item_file}: {e}") 
            print(f"Loaded/verified {loaded_count} cached images from {self.cache_dir}. In-memory cache size: {len(self.cache)}")
        else:
            if self.cache_dir:
                print(f"Cache directory {self.cache_dir} not found or is not a directory. Starting with an empty cache.")

    def precompute_all_images(
        self,
        item_ids: List[str],
        image_folder: str,
        image_processor, # This is the HuggingFace image processor
        force_recompute: bool = False
    ):
        """
        Precompute images, save them progressively to disk if cache_dir is set,
        and also populate the in-memory self.cache for current session use.
        """
        tqdm.write(f"--- precompute_all_images started ---") # Diagnostic print
        tqdm.write(f"Initial self.cache_dir: {str(self.cache_dir if self.cache_dir else 'None')}")
        tqdm.write(f"force_recompute: {force_recompute}")

        if self.cache_dir:
            tqdm.write(f"Precomputing images progressively to directory: {self.cache_dir}")
            self.cache_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        else:
            tqdm.write("WARNING: self.cache_dir is None. Images will only be precomputed into memory, not saved progressively to disk.")

        placeholder_size = (224, 224) # Default
        try:
            if hasattr(image_processor, 'size'):
                processor_size = image_processor.size 
                if isinstance(processor_size, dict) and 'shortest_edge' in processor_size:
                    size_val = processor_size['shortest_edge']; placeholder_size = (size_val, size_val)
                elif isinstance(processor_size, dict) and 'height' in processor_size and 'width' in processor_size:
                    placeholder_size = (processor_size['height'], processor_size['width'])
                elif isinstance(processor_size, (tuple, list)) and len(processor_size) >= 2:
                    placeholder_size = (processor_size[0], processor_size[1])
                elif isinstance(processor_size, int):
                    placeholder_size = (processor_size, processor_size)
        except Exception:
            pass 
            
        processed_in_loop_count = 0
        saved_to_disk_count = 0
        skipped_count = 0

        for i, item_id in enumerate(tqdm(item_ids, desc="Processing images for cache")):
            tensor_path = self.cache_dir / f"{item_id}.pt" if self.cache_dir else None
            
            # Debug print for tensor_path (prints for first 5 items and then every 1000th)
            if i < 5 or (i % 1000 == 0) : 
                 tqdm.write(f"Loop item {item_id}: self.cache_dir is '{str(self.cache_dir)}', tensor_path is '{str(tensor_path)}'")

            if not force_recompute:
                if item_id in self.cache:
                    if i < 5 or (i % 1000 == 0) : tqdm.write(f"Skipping {item_id}: Already in memory cache.")
                    skipped_count += 1
                    continue
                if tensor_path and tensor_path.exists():
                    if i < 5 or (i % 1000 == 0) : tqdm.write(f"Skipping {item_id}: File already exists at {str(tensor_path)}.")
                    skipped_count += 1
                    continue
            
            # --- Image loading and processing logic ---
            base_path = os.path.join(image_folder, str(item_id))
            image_path_to_load = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.webp', '.WEBP']:
                current_path = f"{base_path}{ext}"
                if os.path.exists(current_path): 
                    image_path_to_load = current_path
                    break
            
            img_tensor: Optional[torch.Tensor] = None
            try:
                if image_path_to_load is None: 
                    raise FileNotFoundError(f"Image for {item_id} not found in {image_folder} (tried common extensions).")
                image = Image.open(image_path_to_load).convert('RGB')
                processed_output = image_processor(images=image, return_tensors='pt')
                
                if (isinstance(processed_output, Mapping) or isinstance(processed_output, dict)) and \
                    hasattr(processed_output, 'get') and processed_output.get('pixel_values') is not None:
                    img_tensor = processed_output['pixel_values'] # Or processed_output.get('pixel_values')
                    if img_tensor.ndim == 4 and img_tensor.shape[0] == 1: 
                        img_tensor = img_tensor.squeeze(0)
                elif torch.is_tensor(processed_output):
                    img_tensor = processed_output
                    if img_tensor.ndim == 4 and img_tensor.shape[0] == 1: 
                        img_tensor = img_tensor.squeeze(0)
                else:
                    tqdm.write(f"UNEXPECTED PROCESSED_OUTPUT for item {item_id}: type={type(processed_output)}, content='{str(processed_output)[:500]}'") # Print type and first 500 chars of content
                    raise ValueError("Image processor output not in expected format (dict with 'pixel_values' or a tensor).") 

                if img_tensor.ndim == 2: 
                    img_tensor = img_tensor.unsqueeze(0).repeat(3,1,1)
                elif img_tensor.ndim == 3 and img_tensor.shape[0] == 1: 
                    img_tensor = img_tensor.repeat(3,1,1)

                if not (img_tensor.ndim == 3 and img_tensor.shape[0] == 3):
                    if i < 5 or (i % 1000 == 0): tqdm.write(f"Warning: Processed image for {item_id} has unexpected shape {img_tensor.shape}. Using placeholder.")
                    img_tensor = torch.zeros(3, placeholder_size[0], placeholder_size[1])
            except Exception as e_proc:
                if i < 5 or (i % 1000 == 0): tqdm.write(f"Error processing item {item_id}: {e_proc}. Using placeholder.")
                img_tensor = torch.zeros(3, placeholder_size[0], placeholder_size[1])
            # --- End Image loading and processing logic ---

            self.cache[item_id] = img_tensor 
            processed_in_loop_count += 1
            
            if tensor_path: # Check if disk saving is applicable
                try:
                    # Debug print before attempting to save
                    if i % 1000 == 0 : tqdm.write(f"Attempting to save tensor for {item_id} to {str(tensor_path.resolve())}")
                    torch.save(img_tensor, tensor_path)
                    if i % 1000 == 0 : tqdm.write(f"Successfully saved tensor for {item_id}.")
                    saved_to_disk_count += 1
                except Exception as e_save:
                    tqdm.write(f"ERROR SAVING TENSOR for {item_id} to {str(tensor_path.resolve())}: {e_save}")
            elif self.cache_dir is None and processed_in_loop_count == 1: # Only print this specific debug once
                tqdm.write(f"DEBUG: tensor_path is None for {item_id} because self.cache_dir is None. Not saving to disk.")
        
        tqdm.write(f"--- precompute_all_images finished ---")
        tqdm.write(f"Items processed in loop: {processed_in_loop_count}. Items successfully saved to disk in loop: {saved_to_disk_count}.")
        if skipped_count > 0:
            tqdm.write(f"Skipped processing for {skipped_count} items (already cached or on disk).")


    def save_to_disk(self):
        """
        Saves the current in-memory cache to disk by saving each tensor individually.
        """
        if self.cache_dir and self.cache: # Only if cache_dir is set and cache is not empty
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving/Updating in-memory image cache to directory {self.cache_dir} ({len(self.cache)} items)")
            
            saved_count = 0
            for item_id, tensor in tqdm(self.cache.items(), desc="Flushing in-memory cache to disk"):
                tensor_path = self.cache_dir / f"{item_id}.pt"
                try:
                    torch.save(tensor, tensor_path) # This will overwrite if file exists
                    saved_count += 1
                except Exception as e:
                    print(f"Warning: Could not save cached image for item_id {item_id} to {tensor_path}: {e}")
            print(f"Saved/Updated {saved_count} cached images to {self.cache_dir} from in-memory items.")
        elif not self.cache:
            print("In-memory image cache is empty. Nothing to save to disk.")
        else: # self.cache_dir is None
             print("Cache directory not specified. Skipping saving in-memory image cache to disk.")
            
    def get(self, item_id: str) -> Optional[torch.Tensor]:
        """Get cached image tensor from in-memory cache"""
        return self.cache.get(item_id)
        
    def set(self, item_id: str, tensor: torch.Tensor):
        """Set cached image tensor in the in-memory cache.
           Note: This does not automatically save to disk. Call save_to_disk() if persistence is needed.
        """
        self.cache[item_id] = tensor