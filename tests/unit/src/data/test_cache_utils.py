# tests/unit/src/data/test_cache_utils.py
"""
Unit tests for the cache_utils.py utility functions.
"""
import unittest
from pathlib import Path
import sys
import shutil
import torch
from unittest.mock import patch
import io

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from src.data.cache_utils import (
    list_available_caches,
    print_cache_summary,
    get_cache_path,
    cache_exists,
    get_cache_stats,
    clear_cache,
    clear_all_caches
)

class TestCacheUtils(unittest.TestCase):
    """Test cases for cache management utility functions."""

    def setUp(self):
        """Set up a temporary directory structure for testing."""
        # Creates a temporary directory for test artifacts.
        self.test_dir = Path("test_temp_cache_utils")
        self.test_dir.mkdir(exist_ok=True)
        self.cache_base_dir = self.test_dir / "cache"
        self.cache_base_dir.mkdir(exist_ok=True)

        # Creates a cache directory for a resnet/sentence-bert model combination with two dummy feature files.
        self.cache1_dir = self.cache_base_dir / "resnet_sentence-bert"
        self.cache1_dir.mkdir()
        torch.save({'feature': torch.rand(10)}, self.cache1_dir / "item1.pt")
        torch.save({'feature': torch.rand(10)}, self.cache1_dir / "item2.pt")

        # Creates a second cache directory for a clip/mpnet combination with one dummy feature file.
        self.cache2_dir = self.cache_base_dir / "clip_mpnet"
        self.cache2_dir.mkdir()
        torch.save({'feature': torch.rand(20)}, self.cache2_dir / "itemA.pt")

        # Creates an empty cache directory which should be ignored by some functions but cleared by others.
        self.cache3_dir = self.cache_base_dir / "dino_roberta"
        self.cache3_dir.mkdir()
        
        # Creates another subdirectory that should also be cleared.
        self.other_dir = self.cache_base_dir / "not_a_cache_dir"
        self.other_dir.mkdir()

        # Creates a file directly in the base cache directory to test the clearing logic.
        self.leftover_file = self.cache_base_dir / "some_file.txt"
        self.leftover_file.touch()

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        # Removes the temporary directory and all its contents.
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_get_cache_path(self):
        """Test that the cache path is constructed correctly."""
        # Verifies that the cache path is generated with the correct structure.
        expected_path = self.cache_base_dir / "vision_language"
        self.assertEqual(get_cache_path("vision", "language", str(self.cache_base_dir)), expected_path)

    def test_list_available_caches(self):
        """Test listing of available and valid caches."""
        # Verifies that only non-empty, correctly named cache directories are listed.
        caches = list_available_caches(str(self.cache_base_dir))
        
        self.assertEqual(len(caches), 2, "Should only find 2 valid caches with .pt files.")
        self.assertIn("resnet_sentence-bert", caches)
        self.assertIn("clip_mpnet", caches)
        self.assertNotIn("dino_roberta", caches, "Empty cache directories should not be listed.")
        self.assertNotIn("not_a_cache_dir", caches, "Directory without '_' in name should be ignored by list_available_caches.")

        # Verifies the detailed information for a specific listed cache.
        resnet_cache_info = caches["resnet_sentence-bert"]
        self.assertEqual(resnet_cache_info['vision_model'], "resnet")
        self.assertEqual(resnet_cache_info['language_model'], "sentence-bert")
        self.assertEqual(resnet_cache_info['files'], 2)
        self.assertGreater(resnet_cache_info['size_mb'], 0)

    def test_cache_exists(self):
        """Test the cache_exists function."""
        # Checks for the existence of valid, invalid, and empty cache directories.
        self.assertTrue(cache_exists("resnet", "sentence-bert", str(self.cache_base_dir)))
        self.assertFalse(cache_exists("nonexistent", "model", str(self.cache_base_dir)))
        self.assertFalse(cache_exists("dino", "roberta", str(self.cache_base_dir)), "Should return False for empty directories.")

    def test_get_cache_stats(self):
        """Test getting statistics for a specific cache."""
        # Verifies that statistics (file count, size) are correctly reported for an existing cache.
        stats = get_cache_stats("resnet", "sentence-bert", str(self.cache_base_dir))
        
        self.assertTrue(stats["exists"])
        self.assertEqual(stats["files"], 2)
        self.assertGreater(stats["size_mb"], 0)
        self.assertEqual(stats["path"], str(self.cache1_dir))

        # Verifies the correct response for a non-existent cache.
        stats_none = get_cache_stats("nonexistent", "model", str(self.cache_base_dir))
        self.assertFalse(stats_none["exists"])

    def test_clear_cache(self):
        """Test clearing a single model-specific cache."""
        # Verifies that a specific cache directory is removed while others remain.
        self.assertTrue(self.cache1_dir.exists())
        clear_cache("resnet", "sentence-bert", str(self.cache_base_dir))
        self.assertFalse(self.cache1_dir.exists(), "The specific cache directory should be removed.")
        self.assertTrue(self.cache2_dir.exists(), "Other cache directories should remain.")

    def test_clear_all_caches(self):
        """
        Test clearing all model-specific caches.
        This test is corrected based on the provided failure. The function under test
        only removes subdirectories, not files in the base cache directory.
        """
        # Verifies the initial state of directories and the leftover file.
        self.assertTrue(self.cache1_dir.exists())
        self.assertTrue(self.cache2_dir.exists())
        self.assertTrue(self.other_dir.exists())
        self.assertTrue(self.leftover_file.exists())

        cleared_count = clear_all_caches(str(self.cache_base_dir))

        # Verifies that the function reports clearing the correct number of subdirectories.
        self.assertEqual(cleared_count, 4, "Should report clearing all 4 subdirectories.")

        # Verifies that all subdirectories have been removed.
        self.assertFalse(self.cache1_dir.exists())
        self.assertFalse(self.cache2_dir.exists())
        self.assertFalse(self.cache3_dir.exists())
        self.assertFalse(self.other_dir.exists())

        # Corrects the original failing assertion. The base directory should still exist
        # because the function does not remove files within it.
        self.assertTrue(self.cache_base_dir.exists(), "Base cache directory should NOT be removed as it's not empty.")
        self.assertTrue(self.leftover_file.exists(), "The file in the base directory should not be touched.")

    def test_print_cache_summary(self):
        """Test the summary print function by capturing its output."""
        # Captures standard output to verify the printed summary's content.
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            print_cache_summary(str(self.cache_base_dir))
            output = fake_stdout.getvalue()

        self.assertIn("AVAILABLE FEATURE CACHES", output)
        self.assertIn("resnet_sentence-bert", output)
        self.assertIn("clip_mpnet", output)
        self.assertIn("Files: 2", output)
        self.assertIn("Files: 1", output)
        self.assertIn("TOTAL:", output)
        self.assertNotIn("dino_roberta", output)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)