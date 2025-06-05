# src/data/processors/feature_cache_processor.py
"""
Modular feature caching processor for precomputing and storing features
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm

try:
    from ..simple_cache import SimpleFeatureCache as ProcessedFeatureCache
except ImportError:
    ProcessedFeatureCache = None


class FeatureCacheProcessor:
    """Handles feature caching operations for precomputed features"""
    
    def __init__(self, cache_config: Optional[Any] = None):
        self.cache_config = cache_config
        self.feature_cache = None
        
        if cache_config and ProcessedFeatureCache:
            self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize the feature cache with configuration"""
        cache_dir = Path(self.cache_config.cache_directory)
        
        print(f"Initializing ProcessedFeatureCache:")
        print(f"  Strategy: {self.cache_config.strategy}")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Max memory items: {self.cache_config.max_memory_items}")
        
        self.feature_cache = ProcessedFeatureCache(
            cache_path=str(cache_dir),
            max_memory_items=self.cache_config.max_memory_items,
            strategy=self.cache_config.strategy
        )
        
        # Load existing cache metadata
        self.feature_cache.load_from_disk_meta()
    
    def precompute_features(
        self,
        item_info_df: pd.DataFrame,
        dataset_instance: Any,
        force_recompute: bool = False
    ) -> bool:
        """
        Precompute and cache non-image features for all items.
        
        Args:
            item_info_df: DataFrame with item information
            dataset_instance: Dataset instance for feature processing
            force_recompute: Whether to recompute existing cached features
            
        Returns:
            True if successful, False otherwise
        """
        if not self.feature_cache:
            print("Feature cache not initialized. Skipping precomputation.")
            return False
        
        print(f"Precomputing features for {len(item_info_df)} items...")
        
        success_count = 0
        error_count = 0
        
        for index, item_row in tqdm(
            item_info_df.iterrows(), 
            total=len(item_info_df), 
            desc="Caching features"
        ):
            item_id = str(item_row['item_id'])
            
            # Skip if already cached (unless forcing recompute)
            if not force_recompute and self.feature_cache.get(item_id) is not None:
                success_count += 1
                continue
            
            try:
                # Extract features using dataset methods
                features = self._extract_item_features(item_row, dataset_instance)
                
                if features:
                    self.feature_cache.set(item_id, features)
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"Error processing features for item {item_id}: {e}")
                error_count += 1
        
        print(f"Feature caching completed: {success_count} successful, {error_count} errors")
        
        if hasattr(self.feature_cache, 'print_stats'):
            self.feature_cache.print_stats()
        
        return error_count == 0
    
    def _extract_item_features(
        self, 
        item_row: pd.Series, 
        dataset_instance: Any
    ) -> Optional[dict]:
        """
        Extract features for a single item using dataset methods.
        
        Args:
            item_row: Item information as pandas Series
            dataset_instance: Dataset instance for feature processing
            
        Returns:
            Dictionary of extracted features or None if error
        """
        try:
            # Get text content
            text_content = dataset_instance._get_item_text(item_row)
            
            # Tokenize main text
            main_tokenizer_max_len = getattr(
                dataset_instance.tokenizer, 'model_max_length', 128
            )
            
            text_tokens = dataset_instance.tokenizer(
                text_content,
                padding='max_length',
                truncation=True,
                max_length=main_tokenizer_max_len,
                return_tensors='pt'
            )
            
            # Get numerical features
            item_id = str(item_row['item_id'])
            numerical_features = dataset_instance._get_item_numerical_features(
                item_id, item_row
            )
            
            # Prepare feature dictionary
            features = {
                'text_input_ids': text_tokens['input_ids'].squeeze(0),
                'text_attention_mask': text_tokens['attention_mask'].squeeze(0),
                'numerical_features': numerical_features
            }
            
            # Add CLIP tokens if available
            if hasattr(dataset_instance, 'clip_tokenizer_for_contrastive') and \
               dataset_instance.clip_tokenizer_for_contrastive:
                
                clip_tokens = dataset_instance.clip_tokenizer_for_contrastive(
                    text_content,
                    padding='max_length',
                    truncation=True,
                    max_length=77,  # Standard CLIP max length
                    return_tensors='pt'
                )
                
                features['clip_text_input_ids'] = clip_tokens['input_ids'].squeeze(0)
                features['clip_text_attention_mask'] = clip_tokens['attention_mask'].squeeze(0)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.feature_cache:
            return {"cache_enabled": False}
        
        if hasattr(self.feature_cache, 'print_stats'):
            # Return basic stats
            return {
                "cache_enabled": True,
                "cache_type": type(self.feature_cache).__name__
            }
        
        return {"cache_enabled": True}
    
    def clear_cache(self):
        """Clear the feature cache"""
        if self.feature_cache and hasattr(self.feature_cache, 'clear'):
            self.feature_cache.clear()
            print("Feature cache cleared")