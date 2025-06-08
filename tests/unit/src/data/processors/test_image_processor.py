import unittest
from pathlib import Path
import shutil
import sys
from PIL import Image

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from src.data.processors.image_processor import ImageProcessor
from src.config import OfflineImageCompressionConfig, ImageValidationConfig

class TestImageProcessor(unittest.TestCase):
    """Unit tests for the ImageProcessor class."""

    def setUp(self):
        """Set up temporary directories and dummy images."""
        self.test_dir = Path("test_temp_image_processor")
        self.source_dir = self.test_dir / "source"
        self.dest_dir = self.test_dir / "dest"
        self.test_dir.mkdir()
        self.source_dir.mkdir()
        self.dest_dir.mkdir()

        # Create dummy images
        Image.new('RGB', (200, 200)).save(self.source_dir / "valid_item.jpg")
        Image.new('RGB', (30, 30)).save(self.source_dir / "small_item.jpg")
        Image.new('RGB', (200, 200)).save(self.source_dir / "unsupported.gif")
        (self.source_dir / "corrupted_item.jpg").write_text("not an image")

        # Configurations
        self.validation_config = ImageValidationConfig(
            check_corrupted=True,
            min_width=50,
            min_height=50,
            allowed_extensions=['.jpg']
        )
        self.compression_config = OfflineImageCompressionConfig(enabled=False)
        
        self.processor = ImageProcessor(self.compression_config, self.validation_config)

    def tearDown(self):
        """Clean up temporary directories."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_process_single_image_success(self):
        """Tests successful processing of a valid image."""
        source_path = self.source_dir / "valid_item.jpg"
        dest_path = self.dest_dir / "valid_item.jpg"
        result = self.processor.process_single_image(source_path, dest_path)
        self.assertTrue(result)
        self.assertTrue(dest_path.exists())

    def test_process_single_image_corrupted(self):
        """Tests that a corrupted image is not processed."""
        source_path = self.source_dir / "corrupted_item.jpg"
        dest_path = self.dest_dir / "corrupted_item.jpg"
        result = self.processor.process_single_image(source_path, dest_path)
        self.assertFalse(result)
        self.assertFalse(dest_path.exists())

    def test_process_single_image_too_small(self):
        """Tests that an image with dimensions below the minimum is not processed."""
        source_path = self.source_dir / "small_item.jpg"
        dest_path = self.dest_dir / "small_item.jpg"
        result = self.processor.process_single_image(source_path, dest_path)
        self.assertFalse(result)
        self.assertFalse(dest_path.exists())
    
    def test_find_image_for_item(self):
        """Tests that the correct image file can be found for an item ID."""
        found_path = self.processor._find_image_for_item("valid_item", self.source_dir)
        self.assertEqual(found_path, self.source_dir / "valid_item.jpg")

        # Test with unsupported extension
        not_found_path = self.processor._find_image_for_item("unsupported", self.source_dir)
        self.assertIsNone(not_found_path)

    def test_process_items_images_batch(self):
        """Tests the batch processing of images for a list of item IDs."""
        item_ids = ["valid_item", "small_item", "nonexistent_item", "corrupted_item"]
        valid_ids = self.processor.process_items_images(item_ids, self.source_dir, self.dest_dir)

        self.assertEqual(valid_ids, {"valid_item"})
        self.assertTrue((self.dest_dir / "valid_item.jpg").exists())
        self.assertFalse((self.dest_dir / "small_item.jpg").exists())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)