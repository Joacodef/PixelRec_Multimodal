import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import torch
import pickle
import json
from unittest.mock import patch, MagicMock

# Add the project root to the path to allow importing from scripts and src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.checkpoint_manager import (
    scan_checkpoints,
    extract_model_info_from_checkpoint,
    organize_checkpoints,
    create_checkpoint_info
)

# Mock the pandas import within the script since it's only used for one function
# that we won't be testing directly in this unit test.
# This prevents needing pandas as a dependency just for this test file if it wasn't already.
try:
    # Attempt to import pandas to see if it's available
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Mock pd.Timestamp.now() if pandas is not available
if not HAS_PANDAS:
    # Create a mock object that simulates pd.Timestamp.now()
    class MockTimestamp:
        def __init__(self, *args, **kwargs):
            pass
        def __str__(self):
            return "2023-01-01T12:00:00"

    # Patch the 'pd' module inside the checkpoint_manager script
    mock_pd = MagicMock()
    mock_pd.Timestamp.now.return_value = MockTimestamp()
    sys.modules['scripts.checkpoint_manager.pd'] = mock_pd


class TestCheckpointManager(unittest.TestCase):
    """Test cases for the checkpoint_manager.py script."""

    def setUp(self):
        """Set up a temporary directory structure for testing."""
        self.test_dir = Path("test_temp_checkpoints")
        self.test_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.test_dir / "models" / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # --- Create Dummy Files ---

        # 1. Unorganized checkpoint with metadata
        model_config_1 = {'vision_model': 'resnet', 'language_model': 'sentence-bert'}
        torch.save({'model_config': model_config_1}, self.checkpoints_dir / "unorganized_with_meta.pth")

        # 2. Unorganized checkpoint without metadata
        torch.save({'model_state': [1, 2, 3]}, self.checkpoints_dir / "unorganized_no_meta.pth")

        # 3. Shared encoder files in the root
        with open(self.checkpoints_dir / "user_encoder.pkl", "wb") as f:
            pickle.dump({'user_map': {'a': 0}}, f)
        with open(self.checkpoints_dir / "item_encoder.pkl", "wb") as f:
            pickle.dump({'item_map': {'x': 0}}, f)

        # 4. An already organized checkpoint
        organized_dir = self.checkpoints_dir / "clip_mpnet"
        organized_dir.mkdir()
        torch.save({'model_state': [4, 5, 6]}, organized_dir / "already_organized.pth")

        # 5. An unknown file type
        (self.checkpoints_dir / "some_other_file.txt").write_text("info")


    def tearDown(self):
        """Clean up the temporary directory after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_scan_checkpoints(self):
        """Test that the directory is scanned and categorized correctly."""
        scan_results = scan_checkpoints(str(self.checkpoints_dir))

        # Check unorganized models
        self.assertEqual(len(scan_results['models']), 2)
        model_names = {p.name for p in scan_results['models']}
        self.assertIn("unorganized_with_meta.pth", model_names)
        self.assertIn("unorganized_no_meta.pth", model_names)

        # Check encoders
        self.assertEqual(len(scan_results['encoders']), 2)
        encoder_names = {p.name for p in scan_results['encoders']}
        self.assertIn("user_encoder.pkl", encoder_names)
        self.assertIn("item_encoder.pkl", encoder_names)

        # Check already organized files
        self.assertEqual(len(scan_results['organized']), 1)
        self.assertEqual(scan_results['organized'][0].name, "already_organized.pth")

        # Check unknown files
        self.assertEqual(len(scan_results['unknown']), 1)
        self.assertEqual(scan_results['unknown'][0].name, "some_other_file.txt")

    def test_extract_model_info_from_checkpoint(self):
        """Test extraction of model combination from a checkpoint file."""
        # Test with a file that has metadata
        info = extract_model_info_from_checkpoint(self.checkpoints_dir / "unorganized_with_meta.pth")
        self.assertIsNotNone(info)
        self.assertEqual(info, ('resnet', 'sentence-bert'))

        # Test with a file that has no metadata
        info_none = extract_model_info_from_checkpoint(self.checkpoints_dir / "unorganized_no_meta.pth")
        self.assertIsNone(info_none)

    def test_organize_checkpoints_dry_run(self):
        """Test the organization logic in dry-run mode (no files moved)."""
        organize_checkpoints(str(self.checkpoints_dir), dry_run=True)

        # Verify that no files were actually moved
        self.assertTrue((self.checkpoints_dir / "unorganized_with_meta.pth").exists())
        self.assertTrue((self.checkpoints_dir / "user_encoder.pkl").exists())
        self.assertFalse((self.checkpoints_dir / "resnet_sentence-bert").exists())
        self.assertFalse((self.checkpoints_dir / "encoders").exists())

    def test_organize_checkpoints_actual_run(self):
        """Test the actual file moving logic of the organization script."""
        organize_checkpoints(str(self.checkpoints_dir), dry_run=False)

        # Define expected new paths
        organized_model_dir = self.checkpoints_dir / "resnet_sentence-bert"
        organized_model_path = organized_model_dir / "unorganized_with_meta.pth"
        shared_encoders_dir = self.checkpoints_dir / "encoders"
        organized_user_encoder_path = shared_encoders_dir / "user_encoder.pkl"
        organized_item_encoder_path = shared_encoders_dir / "item_encoder.pkl"

        # Verify that files were moved correctly
        self.assertTrue(organized_model_dir.exists())
        self.assertTrue(organized_model_path.exists())
        self.assertTrue(shared_encoders_dir.exists())
        self.assertTrue(organized_user_encoder_path.exists())
        self.assertTrue(organized_item_encoder_path.exists())

        # Verify that original files are gone
        self.assertFalse((self.checkpoints_dir / "unorganized_with_meta.pth").exists())
        self.assertFalse((self.checkpoints_dir / "user_encoder.pkl").exists())

        # Verify that untouched files remain
        self.assertTrue((self.checkpoints_dir / "unorganized_no_meta.pth").exists())
        self.assertTrue((self.checkpoints_dir / "clip_mpnet" / "already_organized.pth").exists())

    @unittest.skipIf(not HAS_PANDAS, "Pandas not installed, skipping info file creation test")
    def test_create_checkpoint_info_file(self):
        """Test the creation of the JSON summary file."""
        # First, organize the checkpoints to have something to summarize
        organize_checkpoints(str(self.checkpoints_dir), dry_run=False)

        # Create the info file
        create_checkpoint_info(str(self.checkpoints_dir))

        # Verify the JSON file was created
        info_file_path = self.checkpoints_dir / "checkpoint_info.json"
        self.assertTrue(info_file_path.exists())

        # Load and validate the JSON content
        with open(info_file_path, "r") as f:
            info_data = json.load(f)

        # Check top-level keys
        self.assertIn("organization_type", info_data)
        self.assertIn("models", info_data)
        self.assertIn("shared_files", info_data)
        self.assertIn("summary", info_data)

        # Check models section
        self.assertIn("resnet_sentence-bert", info_data["models"])
        self.assertEqual(len(info_data["models"]["resnet_sentence-bert"]["checkpoints"]), 1)
        self.assertEqual(info_data["models"]["resnet_sentence-bert"]["checkpoints"][0]["filename"], "unorganized_with_meta.pth")
        
        # Check shared files section
        self.assertEqual(len(info_data["shared_files"]), 2)
        shared_filenames = {f["filename"] for f in info_data["shared_files"]}
        self.assertIn("user_encoder.pkl", shared_filenames)
        self.assertIn("item_encoder.pkl", shared_filenames)

        # Check summary
        self.assertEqual(info_data["summary"]["total_model_combinations"], 2) # resnet_sentence-bert and clip_mpnet
        self.assertEqual(info_data["summary"]["total_checkpoints"], 2)
        self.assertEqual(info_data["summary"]["total_shared_files"], 2)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)