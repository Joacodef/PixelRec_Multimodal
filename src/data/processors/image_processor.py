# src/data/processors/image_processor.py
"""
Modular image processing for both offline validation/compression and 
online loading and transformation for the model.
"""
import os
import shutil
from pathlib import Path
from typing import List, Set, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor

from ..preprocessing import is_image_corrupted, check_image_dimensions
from ...config import (
    OfflineImageCompressionConfig, 
    ImageValidationConfig, 
    ImageAugmentationConfig, 
    MODEL_CONFIGS
)


class ImageProcessor:
    """Handles both offline and online image processing tasks."""

    def __init__(
        self,
        # Parameters for online mode (used by Dataset)
        model_name: Optional[str] = None,
        augmentation_config: Optional[ImageAugmentationConfig] = None,
        is_train: bool = False,
        # Parameters for offline mode (used by preprocessing scripts)
        compression_config: Optional[OfflineImageCompressionConfig] = None,
        validation_config: Optional[ImageValidationConfig] = None
    ):
        """
        Initializes the ImageProcessor for either online or offline use.

        Args:
            model_name (Optional[str]): The key for the vision model configuration (online mode).
            augmentation_config (Optional[ImageAugmentationConfig]): Augmentation config (online mode).
            is_train (bool): Flag for training mode (online mode).
            compression_config (Optional[OfflineImageCompressionConfig]): Compression config (offline mode).
            validation_config (Optional[ImageValidationConfig]): Validation config (offline mode).
        """
        # Store all configurations
        self.compression_config = compression_config
        self.validation_config = validation_config
        self.augmentation_config = augmentation_config
        self.is_train = is_train

        # Initialize components for ONLINE mode if model_name is provided
        if model_name:
            self.config = MODEL_CONFIGS['vision'].get(model_name)
            if not self.config:
                raise ValueError(f"Configuration for vision model '{model_name}' not found.")
            self.feature_extractor = AutoImageProcessor.from_pretrained(self.config['name'])
        else:
            self.config = None
            self.feature_extractor = None

    # --- Methods for Online Processing (used by Dataset) ---

    def load_and_transform_image(self, image_path: str) -> torch.Tensor:
        """Loads an image from a path, applies transformations, and returns a tensor."""
        if not self.feature_extractor:
            raise RuntimeError("ImageProcessor not initialized for online mode. Provide 'model_name'.")
        
        if not os.path.exists(image_path):
            return self.get_placeholder_tensor()
            
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = self.feature_extractor(images=image, return_tensors="pt")
            return processed_image['pixel_values'].squeeze(0)
        except Exception:
            return self.get_placeholder_tensor()

    def get_placeholder_tensor(self) -> torch.Tensor:
        """Creates a placeholder (zero) tensor for an image."""
        if not self.config:
            raise RuntimeError("ImageProcessor not initialized for online mode. Provide 'model_name'.")
        size = self.config.get('input_size', (224, 224))
        return torch.zeros(3, size[0], size[1])

    # --- Methods for Offline Processing (used by scripts) ---

    def process_items_images(self, item_ids: List[str], source_folder: Path, dest_folder: Path) -> Set[str]:
        """Process images for multiple items."""
        if not self.validation_config:
            raise RuntimeError("ImageProcessor not initialized for offline mode. Provide 'validation_config'.")
            
        dest_folder.mkdir(parents=True, exist_ok=True)
        valid_item_ids = set()
        for item_id in tqdm(item_ids, desc="Processing images"):
            source_path = self._find_image_for_item(item_id, source_folder)
            if source_path:
                dest_path = dest_folder / source_path.name
                if self._process_single_image(source_path, dest_path):
                    valid_item_ids.add(item_id)
        return valid_item_ids
    
    def _process_single_image(self, source_path: Path, dest_path: Path) -> bool:
        """Process a single image: validate, optionally compress/resize, and save."""
        if dest_path.exists():
            return True
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not source_path.exists() or (self.validation_config.check_corrupted and is_image_corrupted(str(source_path))):
                return False
            if not check_image_dimensions(str(source_path), self.validation_config.min_width, self.validation_config.min_height):
                return False
            if self.compression_config and self._should_compress_image(source_path):
                self._compress_and_save(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)
            return True
        except Exception:
            return False

    def _find_image_for_item(self, item_id: str, source_folder: Path) -> Optional[Path]:
        """Find image file for given item ID."""
        for ext in self.validation_config.allowed_extensions:
            potential_path = source_folder / f"{item_id}{ext}"
            if potential_path.exists():
                return potential_path
        return None

    def _should_compress_image(self, image_path: Path) -> bool:
        """Determine if image should be compressed based on file size."""
        if not self.compression_config or not self.compression_config.enabled:
            return False
        file_size_kb = image_path.stat().st_size / 1024
        return file_size_kb > self.compression_config.compress_if_kb_larger_than

    def _compress_and_save(self, source_path: Path, dest_path: Path):
        """Compress and save image with configured settings."""
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            if (self.compression_config.resize_if_pixels_larger_than and 
                max(img.width, img.height) > self.compression_config.resize_target_longest_edge):
                scale = self.compression_config.resize_target_longest_edge / max(img.width, img.height)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            file_ext = dest_path.suffix.lower()
            if file_ext in ['.jpg', '.jpeg']:
                img.save(dest_path, quality=self.compression_config.target_quality, optimize=True)
            else:
                img.save(dest_path)