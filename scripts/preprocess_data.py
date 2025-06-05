#!/usr/bin/env python
"""
Simplified preprocessing script using modular processors
"""
import argparse
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.processors.image_processor import ImageProcessor
from src.data.processors.text_processor import TextProcessor
from src.data.processors.numerical_processor import NumericalProcessor
from src.data.processors.data_filter import DataFilter
from src.data.processors.feature_cache_processor import FeatureCacheProcessor


class PreprocessingPipeline:
    """Main preprocessing pipeline using modular processors"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data
        
        # Initialize processors
        self.image_processor = ImageProcessor(
            self.data_config.offline_image_compression,
            self.data_config.offline_image_validation
        )
        self.text_processor = TextProcessor(
            self.data_config.offline_text_cleaning
        )
        self.numerical_processor = NumericalProcessor()
        self.data_filter = DataFilter()
        self.feature_cache_processor = FeatureCacheProcessor(
            getattr(self.data_config, 'processed_features_cache_config', None)
        )
        
        # Define text columns (can be made configurable)
        self.text_columns = ['title', 'tag', 'description']
    
    def run_full_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        print("=" * 60)
        print("Starting Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Load raw data
        print("\n1. Loading raw data...")
        item_info_df, interactions_df = self._load_raw_data()
        
        # Step 2: Clean text data
        print("\n2. Cleaning text data...")
        item_info_df = self._clean_text_data(item_info_df)
        
        # Step 3: Process and validate images
        print("\n3. Processing and validating images...")
        valid_item_ids = self._process_images(item_info_df)
        
        if not valid_item_ids:
            print("ERROR: No valid items after image processing!")
            sys.exit(1)
        
        # Step 4: Filter data by valid items
        print("\n4. Filtering data by valid items...")
        item_info_df, interactions_df = self._filter_by_valid_items(
            item_info_df, interactions_df, valid_item_ids
        )
        
        # Step 5: Filter by activity levels
        print("\n5. Filtering by activity levels...")
        interactions_df = self._filter_by_activity(interactions_df)
        
        if interactions_df.empty:
            print("ERROR: No interactions remaining after filtering!")
            sys.exit(1)
        
        # Step 6: Align item info with final interactions
        print("\n6. Aligning item info with interactions...")
        item_info_df = self._align_item_info(item_info_df, interactions_df)
        
        # Step 7: Process numerical features
        print("\n7. Processing numerical features...")
        self._process_numerical_features(item_info_df)
        
        # Step 8: Save processed data
        print("\n8. Saving processed data...")
        self._save_processed_data(item_info_df, interactions_df)
        
        # Step 9: Cache features (optional)
        print("\n9. Caching features...")
        self._cache_features_if_enabled(item_info_df)
        
        # Step 10: Print summary
        self._print_summary(item_info_df, interactions_df)
        
        print("\n" + "=" * 60)
        print("Preprocessing Pipeline Completed Successfully!")
        print("=" * 60)
    
    def _load_raw_data(self):
        """Load raw item info and interactions data"""
        item_info_path = Path(self.data_config.item_info_path)
        interactions_path = Path(self.data_config.interactions_path)
        
        print(f"Loading item info from: {item_info_path}")
        item_info_df = pd.read_csv(item_info_path)
        item_info_df['item_id'] = item_info_df['item_id'].astype(str)
        
        print(f"Loading interactions from: {interactions_path}")
        interactions_df = pd.read_csv(interactions_path)
        interactions_df['item_id'] = interactions_df['item_id'].astype(str)
        interactions_df['user_id'] = interactions_df['user_id'].astype(str)
        
        print(f"Loaded {len(item_info_df)} items and {len(interactions_df)} interactions")
        return item_info_df, interactions_df
    
    def _clean_text_data(self, item_info_df):
        """Clean text columns in item info"""
        return self.text_processor.clean_dataframe_text_columns(
            item_info_df, self.text_columns
        )
    
    def _process_images(self, item_info_df):
        """Process and validate images, return valid item IDs"""
        source_folder = Path(self.data_config.image_folder)
        dest_folder = Path(self.data_config.processed_image_destination_folder)
        
        item_ids = item_info_df['item_id'].astype(str).tolist()
        
        valid_item_ids = self.image_processor.process_items_images(
            item_ids, source_folder, dest_folder
        )
        
        return valid_item_ids
    
    def _filter_by_valid_items(self, item_info_df, interactions_df, valid_item_ids):
        """Filter dataframes to only include valid items"""
        # Filter item info
        original_item_count = len(item_info_df)
        item_info_df = item_info_df[
            item_info_df['item_id'].astype(str).isin(valid_item_ids)
        ].copy()
        
        print(f"Item info filtering: {len(item_info_df)} items remaining "
              f"out of {original_item_count}")
        
        # Filter interactions
        interactions_df = self.data_filter.filter_interactions_by_valid_items(
            interactions_df, valid_item_ids
        )
        
        return item_info_df, interactions_df
    
    def _filter_by_activity(self, interactions_df):
        """Filter interactions by user and item activity"""
        return self.data_filter.filter_by_activity(
            interactions_df,
            min_user_interactions=self.data_config.splitting.min_interactions_per_user,
            min_item_interactions=self.data_config.splitting.min_interactions_per_item
        )
    
    def _align_item_info(self, item_info_df, interactions_df):
        """Align item info with final interactions"""
        return self.data_filter.align_item_info_with_interactions(
            item_info_df, interactions_df
        )
    
    def _process_numerical_features(self, item_info_df):
        """Process numerical features and save scaler"""
        numerical_cols = self.data_config.numerical_features_cols
        method = self.data_config.numerical_normalization_method
        scaler_path = Path(self.data_config.scaler_path)
        
        if not numerical_cols:
            print("No numerical columns specified. Skipping scaler processing.")
            return
        
        # Check if scaler already exists
        if scaler_path.exists():
            print(f"Loading existing scaler from {scaler_path}")
            self.numerical_processor.load_scaler(scaler_path)
        else:
            print(f"Fitting new scaler with method: {method}")
            self.numerical_processor.fit_scaler(item_info_df, numerical_cols, method)
            self.numerical_processor.save_scaler(scaler_path)
        
        # Print scaler info
        scaler_info = self.numerical_processor.get_scaler_info()
        print(f"Scaler info: {scaler_info}")
    
    def _save_processed_data(self, item_info_df, interactions_df):
        """Save processed data to configured paths"""
        # Ensure output directories exist
        item_info_path = Path(self.data_config.processed_item_info_path)
        interactions_path = Path(self.data_config.processed_interactions_path)
        
        item_info_path.parent.mkdir(parents=True, exist_ok=True)
        interactions_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        print(f"Saving processed item info to: {item_info_path}")
        item_info_df.to_csv(item_info_path, index=False)
        
        print(f"Saving processed interactions to: {interactions_path}")
        interactions_df.to_csv(interactions_path, index=False)
    
    def _cache_features_if_enabled(self, item_info_df):
        """Cache features if caching is enabled"""
        if not hasattr(self.data_config, 'processed_features_cache_config'):
            print("Feature caching not configured. Skipping.")
            return
        
        try:
            # Create temporary dataset for feature extraction
            from src.data.dataset import MultimodalDataset
            
            print("Creating temporary dataset for feature extraction...")
            temp_interactions = pd.DataFrame({
                'user_id': ['temp_user'],
                'item_id': [item_info_df.iloc[0]['item_id']]
            })
            
            temp_dataset = MultimodalDataset(
                interactions_df=temp_interactions,
                item_info_df=item_info_df,
                image_folder=str(self.data_config.processed_image_destination_folder),
                vision_model_name=self.config.model.vision_model,
                language_model_name=self.config.model.language_model,
                create_negative_samples=False,
                numerical_feat_cols=self.data_config.numerical_features_cols,
                numerical_normalization_method=self.data_config.numerical_normalization_method,
                numerical_scaler=self.numerical_processor.scaler,
                is_train_mode=False,
                cache_features=False
            )
            
            # Finalize dataset setup
            temp_dataset.finalize_setup()
            
            # Precompute features
            success = self.feature_cache_processor.precompute_features(
                item_info_df, temp_dataset
            )
            
            if success:
                print("Feature caching completed successfully")
            else:
                print("Feature caching completed with some errors")
                
        except Exception as e:
            print(f"Error during feature caching: {e}")
            print("Continuing without feature caching...")
    
    def _print_summary(self, item_info_df, interactions_df):
        """Print preprocessing summary"""
        print(f"""
            Preprocessing Summary:
            ---------------------
            ✓ Final item count: {len(item_info_df)}
            ✓ Final interaction count: {len(interactions_df)}
            ✓ Unique users: {interactions_df['user_id'].nunique()}
            ✓ Unique items in interactions: {interactions_df['item_id'].nunique()}
            ✓ Processed images directory: {self.data_config.processed_image_destination_folder}
            ✓ Numerical scaler: {self.numerical_processor.get_scaler_info()['scaler_type'] if self.numerical_processor.scaler else 'None'}
                    """)


def main():
    """Main function for preprocessing"""
    parser = argparse.ArgumentParser(description="Modular data preprocessing pipeline")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/simple_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-caching',
        action='store_true',
        help='Skip feature caching step'
    )
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing of all images and features'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Modify config based on arguments
    if args.skip_caching:
        # Disable caching in config
        if hasattr(config.data, 'processed_features_cache_config'):
            delattr(config.data, 'processed_features_cache_config')
        print("Feature caching disabled by --skip-caching flag")
    
    # Create and run preprocessing pipeline
    pipeline = PreprocessingPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()