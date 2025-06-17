# tests/unit/src/data/test_simple_cache.py
"""
Unit tests for the SimpleFeatureCache class.
"""
import unittest
from pathlib import Path
import sys
import shutil
import torch
import pickle
import threading
import time

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.data.simple_cache import SimpleFeatureCache

class TestSimpleFeatureCache(unittest.TestCase):
    """Test cases for the SimpleFeatureCache functionality."""

    def setUp(self):
        """Set up a temporary directory for each test."""
        # Creates a temporary directory to store cache files for the duration of a single test.
        self.test_dir = Path("test_temp_simple_cache")
        self.test_dir.mkdir(exist_ok=True)
        self.base_cache_dir = self.test_dir / "cache"
        # Defines standard model names for consistent testing.
        self.vision_model = "resnet"
        self.language_model = "bert"

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        # Removes the temporary directory and all its contents to ensure test isolation.
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_cache_directory_creation(self):
        """Tests that the model-specific cache directory is created only when use_disk is True."""
        # Verifies that the cache directory is created on disk when disk usage is enabled.
        cache = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            base_cache_dir=str(self.base_cache_dir),
            use_disk=True
        )
        self.assertTrue(cache.cache_dir.exists())
        self.assertTrue(cache.cache_dir.is_dir())
        expected_path = self.base_cache_dir / f"vision_{self.vision_model or 'none'}_lang_{self.language_model or 'none'}"
        self.assertEqual(cache.cache_dir, expected_path)
        
        # Verifies that the cache directory is not created when disk usage is disabled.
        shutil.rmtree(self.base_cache_dir)
        cache_mem_only = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            base_cache_dir=str(self.base_cache_dir),
            use_disk=False
        )
        self.assertFalse(cache_mem_only.cache_dir.exists())

    def test_set_and_get_memory_only(self):
        """Tests the basic set and get functionality for an in-memory cache."""
        # Initializes a cache that operates only in memory.
        cache = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            use_disk=False
        )
        features = {'data': torch.tensor([1, 2, 3])}
        
        # Verifies that an item can be added and then retrieved correctly.
        cache.set('item1', features)
        retrieved = cache.get('item1')
        
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(features['data'], retrieved['data']))
        self.assertFalse(self.base_cache_dir.exists(), "Disk cache should not be created.")

    def test_set_and_get_disk_enabled(self):
        """Tests set and get functionality with disk persistence enabled."""
        # Initializes a cache with disk support.
        cache = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            base_cache_dir=str(self.base_cache_dir),
            use_disk=True
        )
        features = {'data': torch.tensor([4, 5, 6])}
        
        # Verifies that setting an item also creates a corresponding file on disk.
        cache.set('item2', features)
        cache_file = cache.cache_dir / "item2.pt"
        self.assertTrue(cache_file.exists())
        
        # Verifies that the item can be retrieved correctly.
        retrieved = cache.get('item2')
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(features['data'], retrieved['data']))

    def test_get_from_disk_when_not_in_memory(self):
        """Tests that features are loaded from disk if not present in memory."""
        # Creates a first cache instance to write an item to disk.
        cache1 = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            base_cache_dir=str(self.base_cache_dir),
            use_disk=True
        )
        features = {'data': torch.tensor([7, 8, 9])}
        cache1.set('item3', features)
        
        # Creates a second instance with an empty memory cache, pointing to the same disk location.
        cache2 = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            base_cache_dir=str(self.base_cache_dir),
            use_disk=True
        )
        self.assertNotIn('item3', cache2.memory_cache)
        
        # Verifies that getting the item loads it from disk into memory.
        retrieved = cache2.get('item3')
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(features['data'], retrieved['data']))
        self.assertIn('item3', cache2.memory_cache)

    def test_lru_eviction_policy(self):
        """Tests that the Least Recently Used (LRU) item is evicted when cache is full."""
        # Initializes a cache with a small memory capacity.
        cache = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            max_memory_items=2,
            use_disk=False
        )
        
        # Adds items to fill and exceed the cache capacity.
        cache.set('item1', {'data': torch.tensor(1)})
        cache.set('item2', {'data': torch.tensor(2)})
        cache.set('item3', {'data': torch.tensor(3)}) # This should evict 'item1'
        
        # Verifies that the oldest item ('item1') was evicted.
        self.assertIsNone(cache.get('item1'))
        self.assertIsNotNone(cache.get('item2'))
        self.assertIsNotNone(cache.get('item3'))
        self.assertEqual(len(cache.memory_cache), 2)

    def test_lru_update_on_get(self):
        """Tests that accessing an item makes it the most recently used."""
        cache = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            max_memory_items=2,
            use_disk=False
        )
        
        cache.set('itemA', {'data': torch.tensor(10)})
        cache.set('itemB', {'data': torch.tensor(11)})
        
        # Accesses 'itemA', making it the most recent.
        cache.get('itemA')
        
        # Adds a new item, which should evict 'itemB' (now the oldest).
        cache.set('itemC', {'data': torch.tensor(12)})
        
        self.assertIsNone(cache.get('itemB'))
        self.assertIsNotNone(cache.get('itemA'))
        self.assertIsNotNone(cache.get('itemC'))

    def test_cache_stats(self):
        """Tests that cache statistics (hits, misses, hit rate) are tracked correctly."""
        cache = SimpleFeatureCache(
            vision_model=self.vision_model,
            language_model=self.language_model,
            max_memory_items=5
        )
        
        # Simulates a series of cache hits and misses.
        cache.get('miss1') # Miss
        cache.set('hit1', {'data': torch.tensor(1)})
        cache.get('hit1')  # Hit
        cache.get('miss2') # Miss
        cache.get('hit1')  # Hit
        
        stats = cache.stats()
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 2)
        self.assertEqual(stats['memory_items'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 0.5)

    def test_pickling_and_unpickling(self):
        """Tests that the cache object can be pickled and unpickled, preserving state."""
        # Verifies that the cache can be serialized, which is essential for use with multiprocessing.
        cache = SimpleFeatureCache(vision_model="test", language_model="pickle", max_memory_items=10)
        cache.set('item_p', {'data': torch.tensor(100)})
        
        # Simulates sending the cache object to another process.
        pickled_cache = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_cache)
        
        # Verifies that the state is restored correctly.
        self.assertIsNotNone(unpickled_cache.get('item_p'))
        self.assertTrue(hasattr(unpickled_cache, '_lock'), "Lock should be re-initialized after unpickling.")
        
    def test_multithreading_safety(self):
        """Tests basic thread safety of set and get operations."""
        # Verifies that concurrent access to the cache does not lead to race conditions or errors.
        num_threads = 10
        iterations = 50
        expected_size = num_threads * iterations

        # Corrected: Initialize cache with enough capacity to hold all items from all threads.
        # This tests thread-safe addition without interference from the LRU eviction policy.
        cache = SimpleFeatureCache(
            vision_model="thread", 
            language_model="safe", 
            max_memory_items=expected_size
        )

        def worker(thread_id):
            for i in range(iterations):
                item_id = f"item_{thread_id}_{i}"
                cache.set(item_id, {'id': thread_id, 'val': i})
                retrieved = cache.get(item_id)
                self.assertIsNotNone(retrieved)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verifies that all items were added correctly despite concurrent access.
        self.assertEqual(len(cache.memory_cache), expected_size)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)