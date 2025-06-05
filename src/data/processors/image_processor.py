# src/data/processors/image_processor.py
"""
Modular image processing for validation, compression, and filtering
"""
from pathlib import Path
from typing import List, Set, Tuple
from PIL import Image
import shutil
from tqdm import tqdm

from ..preprocessing import is_image_corrupted, check_image_dimensions
from ...config import OfflineImageCompressionConfig, ImageValidationConfig


class ImageProcessor:
    """Handles image validation, compression, and filtering operations"""
    
    def __init__(
        self, 
        compression_config: OfflineImageCompressionConfig,
        validation_config: ImageValidationConfig
    ):
        self.compression_config = compression_config
        self.validation_config = validation_config
    
    def process_single_image(
        self,
        source_path: Path,
        dest_path: Path
    ) -> bool:
        """
        Process a single image: validate, optionally compress/resize, and save.
        
        Args:
            source_path: Original image file path
            dest_path: Destination path for processed image
            
        Returns:
            bool: True if successfully processed, False otherwise
        """
        # Skip if already processed
        if dest_path.exists():
            return True
            
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Validation checks
            if not source_path.exists():
                return False
                
            if self.validation_config.check_corrupted:
                if is_image_corrupted(str(source_path)):
                    return False
                    
            if not check_image_dimensions(
                str(source_path), 
                self.validation_config.min_width, 
                self.validation_config.min_height
            ):
                return False
            
            # Determine if compression is needed
            should_compress = self._should_compress_image(source_path)
            
            if should_compress:
                self._compress_and_save(source_path, dest_path)
            else:
                # Just copy the file
                shutil.copy2(source_path, dest_path)
                
            return True
            
        except Exception as e:
            # Optionally log the error
            return False
    
    def process_items_images(
        self,
        item_ids: List[str],
        source_folder: Path,
        dest_folder: Path
    ) -> Set[str]:
        """
        Process images for multiple items.
        
        Args:
            item_ids: List of item IDs to process
            source_folder: Source image folder
            dest_folder: Destination folder for processed images
            
        Returns:
            Set of item IDs that were successfully processed
        """
        dest_folder.mkdir(parents=True, exist_ok=True)
        valid_item_ids = set()
        
        print(f"Processing images for {len(item_ids)} items...")
        
        for item_id in tqdm(item_ids, desc="Processing images"):
            source_path = self._find_image_for_item(item_id, source_folder)
            
            if source_path:
                dest_path = dest_folder / source_path.name
                
                if self.process_single_image(source_path, dest_path):
                    valid_item_ids.add(item_id)
        
        print(f"Successfully processed {len(valid_item_ids)} out of {len(item_ids)} images")
        return valid_item_ids
    
    def _find_image_for_item(self, item_id: str, source_folder: Path) -> Path:
        """Find image file for given item ID"""
        for ext in self.validation_config.allowed_extensions:
            potential_path = source_folder / f"{item_id}{ext}"
            if potential_path.exists():
                return potential_path
        return None
    
    def _should_compress_image(self, image_path: Path) -> bool:
        """Determine if image should be compressed based on file size"""
        if not self.compression_config.enabled:
            return False
            
        file_size_kb = image_path.stat().st_size / 1024
        return file_size_kb > self.compression_config.compress_if_kb_larger_than
    
    def _compress_and_save(self, source_path: Path, dest_path: Path):
        """Compress and save image with configured settings"""
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            
            # Resize if needed
            if (self.compression_config.resize_if_pixels_larger_than and 
                self.compression_config.resize_target_longest_edge):
                
                if (img.width > self.compression_config.resize_if_pixels_larger_than[0] or 
                    img.height > self.compression_config.resize_if_pixels_larger_than[1]):
                    
                    current_longest = max(img.width, img.height)
                    scale_factor = self.compression_config.resize_target_longest_edge / current_longest
                    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save with compression
            file_ext = dest_path.suffix.lower()
            if file_ext in ['.jpg', '.jpeg']:
                img.save(dest_path, quality=self.compression_config.target_quality, optimize=True)
            elif file_ext == '.png':
                img.save(dest_path, compress_level=6)
            else:
                img.save(dest_path)