#!/usr/bin/env python
"""
A comprehensive data preprocessing script for the multimodal recommender system.

This script executes a sequential pipeline to clean, validate, filter, and
transform raw interaction and item data into a state suitable for model
training and evaluation. It leverages a series of modular processors to handle
specific tasks such as text cleaning, image validation and compression,
numerical feature scaling, and data filtering based on activity levels.
The entire process is driven by a YAML configuration file, ensuring that
all preprocessing steps are reproducible and configurable.
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.processors.image_processor import ImageProcessor
from src.data.processors.text_processor import TextProcessor
from src.data.processors.numerical_processor import NumericalProcessor
from src.data.processors.data_filter import DataFilter
from src.data.processors.feature_cache_processor import FeatureCacheProcessor


class PreprocessingPipeline:
    """
    Orchestrates the entire data preprocessing workflow.

    This class encapsulates the logic for loading raw data, applying a series
    of processing and validation steps using modular processors, and saving
    the final, cleaned datasets. Each step is designed to be a distinct
    function, making the pipeline easy to understand and maintain.
    """

    def __init__(self, config: Config):
        """
        Initializes the PreprocessingPipeline with all necessary processors.

        Args:
            config: A Config object loaded from a YAML file, containing all
                    necessary paths and settings for the preprocessing steps.
        """
        self.config = config
        self.data_config = config.data
        
        # Initializes all modular processors with their respective configurations.
        self.image_processor = ImageProcessor(
            compression_config=config.data.image_compression_config,
            validation_config=config.data.image_validation_config
        )
        self.text_processor = TextProcessor(
            cleaning_config=config.data.text_cleaning_config
        )
        self.numerical_processor = NumericalProcessor()
        self.data_filter = DataFilter()
        self.feature_cache_processor = FeatureCacheProcessor(
            getattr(self.data_config, 'processed_features_cache_config', None)
        )
        
        # Defines the columns that will be subjected to text cleaning.
        self.text_columns = ['title', 'tag', 'description']
    
    def run_full_pipeline(self):
        """
        Executes the complete preprocessing pipeline in a defined order.

        This method serves as the main entry point for the class, calling each
        preprocessing step sequentially. It handles the flow of data from one
        step to the next, ensuring that all transformations are applied
        correctly before the final data is saved.
        """
        print("=" * 60)
        print("Starting Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Load raw data from the paths specified in the config.
        print("\n1. Loading raw data...")
        item_info_df, interactions_df = self._load_raw_data()
        
        # Step 2: Clean and normalize text fields in the item metadata.
        print("\n2. Cleaning text data...")
        item_info_df = self._clean_text_data(item_info_df)
        
        # Step 3: Validate images, copy valid ones, and get the list of valid item IDs.
        print("\n3. Processing and validating images...")
        valid_item_ids = self._process_images(item_info_df)
        
        # Halts execution if no items have valid images, as they are crucial for the model.
        if not valid_item_ids:
            print("ERROR: No valid items after image processing!")
            sys.exit(1)
        
        # Step 4: Filter both item and interaction dataframes to keep only items with valid images.
        print("\n4. Filtering data by valid items...")
        item_info_df, interactions_df = self._filter_by_valid_items(
            item_info_df, interactions_df, valid_item_ids
        )
        
        # Step 5: Filter out users and items that do not meet minimum interaction counts.
        print("\n5. Filtering by activity levels...")
        interactions_df = self._filter_by_activity(interactions_df)
        
        # Halts execution if no interactions remain after activity filtering.
        if interactions_df.empty:
            print("ERROR: No interactions remaining after filtering!")
            sys.exit(1)
        
        # Step 6: Ensure the item metadata only contains items present in the final interaction set.
        print("\n6. Aligning item info with interactions...")
        item_info_df = self._align_item_info(item_info_df, interactions_df)

        # Step 7: Group rare tags to ensure robust stratified splitting.
        print("\n7. Grouping rare tags...")
        item_info_df = self._group_rare_tags(item_info_df)
        
        # Step 8: Fit a scaler on the numerical features and save it for later use.
        print("\n8. Processing numerical features...")
        self._process_numerical_features(item_info_df)
        
        # Step 9: Save the final, processed dataframes to disk.
        print("\n9. Saving processed data...")
        self._save_processed_data(item_info_df, interactions_df)
        
        # Step 10: Pre-compute and cache features if enabled in the configuration.
        print("\n10. Caching features...")
        self._cache_features_if_enabled(item_info_df)
        
        # Step 11: Print a summary of the final dataset statistics.
        self._print_summary(item_info_df, interactions_df)
        
        print("\n" + "=" * 60)
        print("Preprocessing Pipeline Completed Successfully!")
        print("=" * 60)
    
    def _load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the raw item info and interactions data from CSV files.

        This function reads the raw data files specified in the configuration,
        ensures that key ID columns are treated as strings, and handles potential
        NaN values in numerical columns by filling them with zero.

        Returns:
            A tuple containing two pandas DataFrames:
            - The raw item information.
            - The raw user-item interactions.
        """
        item_info_path = Path(self.data_config.item_info_path)
        interactions_path = Path(self.data_config.interactions_path)
        
        print(f"Loading item info from: {item_info_path}")
        item_info_df = pd.read_csv(item_info_path)
        item_info_df['item_id'] = item_info_df['item_id'].astype(str)
        
        print(f"Loading interactions from: {interactions_path}")
        interactions_df = pd.read_csv(interactions_path)
        interactions_df['item_id'] = interactions_df['item_id'].astype(str)
        interactions_df['user_id'] = interactions_df['user_id'].astype(str)
        
        # Checks for and handles missing values in numerical feature columns.
        print("\nChecking for NaN values in numerical columns...")
        for col in self.data_config.numerical_features_cols:
            if col in item_info_df.columns:
                nan_count = item_info_df[col].isna().sum()
                if nan_count > 0:
                    print(f"WARNING: {nan_count} NaN values found in column '{col}'")
                    # Fills missing numerical values with 0.
                    item_info_df[col] = item_info_df[col].fillna(0)
                    print(f"Filled NaN values in '{col}' with 0")

        print(f"Loaded {len(item_info_df)} items and {len(interactions_df)} interactions")
        
        return item_info_df, interactions_df
    
    def _clean_text_data(self, item_info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the text columns of the item metadata DataFrame.

        This function utilizes the TextProcessor to apply cleaning operations
        like removing HTML tags, normalizing unicode, and converting text to
        lowercase, based on the settings in the configuration file.

        Args:
            item_info_df: The item metadata DataFrame with raw text columns.

        Returns:
            The item metadata DataFrame with cleaned text columns.
        """
        # First, handle missing values and ensure string type for the 'tag' column.
        if 'tag' in item_info_df.columns:
            print("Cleaning 'tag' column: Filling NaN with 'unknown'.")
            item_info_df['tag'] = item_info_df['tag'].fillna('unknown').astype(str)
        
        # Then, apply general text cleaning (lowercase, etc.) to all text columns.
        return self.text_processor.clean_dataframe_text_columns(
            item_info_df, self.text_columns
        )
    def _process_images(self, item_info_df: pd.DataFrame) -> set:
        """
        Processes and validates all images corresponding to the items.

        This function uses the ImageProcessor to check each item's image for
        corruption and dimension requirements. Valid images are optionally
        compressed and copied to a processed images directory.

        Args:
            item_info_df: The DataFrame containing the list of item IDs.

        Returns:
            A set of item IDs that have a valid, processed image.
        """
        source_folder = Path(self.data_config.image_folder)
        dest_folder = Path(self.data_config.processed_image_destination_folder)
        
        item_ids = item_info_df['item_id'].astype(str).tolist()
        
        valid_item_ids = self.image_processor.process_items_images(
            item_ids, source_folder, dest_folder
        )
        
        return valid_item_ids
    
    def _filter_by_valid_items(self, item_info_df: pd.DataFrame, interactions_df: pd.DataFrame, valid_item_ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters both item and interaction DataFrames based on a set of valid item IDs.

        This step ensures that both datasets only contain entries corresponding
        to items that have successfully passed the image validation stage.

        Args:
            item_info_df: The DataFrame of item metadata.
            interactions_df: The DataFrame of user-item interactions.
            valid_item_ids: A set of item IDs that are considered valid.

        Returns:
            A tuple containing the filtered item_info_df and interactions_df.
        """
        # Filters the item metadata DataFrame.
        original_item_count = len(item_info_df)
        item_info_df = item_info_df[
            item_info_df['item_id'].astype(str).isin(valid_item_ids)
        ].copy()
        
        print(f"Item info filtering: {len(item_info_df)} items remaining "
              f"out of {original_item_count}")
        
        # Filters the interactions DataFrame using the DataFilter processor.
        interactions_df = self.data_filter.filter_interactions_by_valid_items(
            interactions_df, valid_item_ids
        )
        
        return item_info_df, interactions_df
    
    def _filter_by_activity(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the interactions DataFrame by user and item activity levels.

        This function uses the DataFilter processor to remove users and items
        that do not meet the minimum interaction thresholds defined in the
        configuration, which helps in reducing data sparsity.

        Args:
            interactions_df: The interactions DataFrame to be filtered.

        Returns:
            The filtered interactions DataFrame.
        """
        return self.data_filter.filter_by_activity(
            interactions_df,
            min_user_interactions=self.data_config.splitting.min_interactions_per_user,
            min_item_interactions=self.data_config.splitting.min_interactions_per_item
        )
    
    def _align_item_info(self, item_info_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns the item metadata with the final set of interactions.

        After filtering interactions, this function ensures that the item_info
        DataFrame only contains metadata for items that are still present in
        the interactions dataset, preventing unused item data.

        Args:
            item_info_df: The current item metadata DataFrame.
            interactions_df: The final, filtered interactions DataFrame.

        Returns:
            The aligned item metadata DataFrame.
        """
        return self.data_filter.align_item_info_with_interactions(
            item_info_df, interactions_df
        )
    
    def _process_numerical_features(self, item_info_df: pd.DataFrame):
        """
        Fits a numerical feature scaler and saves it to disk.

        This function uses the NumericalProcessor to fit a scaler (e.g.,
        StandardScaler) on the specified numerical feature columns from the
        item data. The fitted scaler is then saved to a file so that the
        exact same transformation can be applied during model training and inference.

        Args:
            item_info_df: The final, filtered DataFrame of item metadata.
        """
        numerical_cols = self.data_config.numerical_features_cols
        method = self.data_config.numerical_normalization_method
        scaler_path = Path(self.data_config.scaler_path)
        
        # Skips the process if no numerical columns are specified in the config.
        if not numerical_cols:
            print("No numerical columns specified. Skipping scaler processing.")
            return

        # Ensures all numerical columns have no NaN values before scaling.
        for col in numerical_cols:
            if col in item_info_df.columns:
                # Fills any remaining NaNs with 0.
                item_info_df[col] = item_info_df[col].fillna(0)
        
        # Fits and saves the scaler unless the method is 'none'.
        if self.data_config.numerical_normalization_method != 'none':
            # Checks if a scaler already exists to avoid re-fitting unnecessarily.
            if scaler_path.exists():
                print(f"Loading existing scaler from {scaler_path}")
                self.numerical_processor.load_scaler(scaler_path)
            else:
                print(f"Fitting new scaler with method: {method}")
                self.numerical_processor.fit_scaler(item_info_df, numerical_cols, method)
                self.numerical_processor.save_scaler(scaler_path)
        
        # Prints information about the scaler that was used or loaded.
        scaler_info = self.numerical_processor.get_scaler_info()
        print(f"Scaler info: {scaler_info}")
    
    def _save_processed_data(self, item_info_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Saves the processed DataFrames to their final destination paths.

        This function ensures the output directories exist and then writes the
        cleaned and filtered item metadata and interaction data to CSV files.

        Args:
            item_info_df: The final processed item metadata DataFrame.
            interactions_df: The final processed interactions DataFrame.
        """
        item_info_path = Path(self.data_config.processed_item_info_path)
        interactions_path = Path(self.data_config.processed_interactions_path)
        
        # Ensures that the directories for the output files exist.
        item_info_path.parent.mkdir(parents=True, exist_ok=True)
        interactions_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Saves the final DataFrames to CSV files.
        print(f"Saving processed item info to: {item_info_path}")
        item_info_df.to_csv(item_info_path, index=False)
        
        print(f"Saving processed interactions to: {interactions_path}")
        interactions_df.to_csv(interactions_path, index=False)
    
    def _cache_features_if_enabled(self, item_info_df: pd.DataFrame):
        """
        Pre-computes and caches features if enabled in the configuration.

        This function initializes a temporary dataset instance to process and
        cache multimodal features, which can significantly speed up subsequent
        model training.

        Args:
            item_info_df: The final processed item metadata DataFrame.
        """
        if not hasattr(self.data_config, 'processed_features_cache_config'):
            print("Feature caching not configured. Skipping.")
            return
        
        try:
            # A temporary dataset instance is created solely for feature extraction logic.
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

            # The FeatureCacheProcessor handles the actual computation and saving.
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

    def _print_summary(self, item_info_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Prints a final summary of the preprocessing results.

        This function displays key statistics about the final datasets, such as
        the number of items, interactions, and unique users, providing a quick
        overview of the preprocessing outcome.

        Args:
            item_info_df: The final processed item metadata DataFrame.
            interactions_df: The final processed interactions DataFrame.
        """
        # Retrieves the scaler type for the summary.
        scaler_type = self.numerical_processor.get_scaler_info()['scaler_type'] if self.numerical_processor.scaler else 'None'
        
        # Formats and prints the final statistics.
        summary_text = f"""
            Preprocessing Summary:
            ---------------------
            ✓ Final item count: {len(item_info_df)}
            ✓ Final interaction count: {len(interactions_df)}
            ✓ Unique users: {interactions_df['user_id'].nunique()}
            ✓ Unique items in interactions: {interactions_df['item_id'].nunique()}
            ✓ Processed images directory: {self.data_config.processed_image_destination_folder}
            ✓ Numerical scaler: {scaler_type}
        """
        print(summary_text)

    def _group_rare_tags(self, item_info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups infrequent tags into a single 'rare_tag' category.

        This step is crucial for ensuring that stratified splitting does not fail
        due to classes (tags) with too few members. It operates on the final,
        filtered item data before it's saved.

        Args:
            item_info_df: The aligned and filtered item metadata DataFrame.

        Returns:
            The item metadata DataFrame with rare tags grouped together.
        """
        # Get the frequency threshold from config, with a safe default of 10.
        threshold_config = getattr(self.data_config.splitting, 'tag_grouping_threshold', None)
        
        # If no threshold is set in the config, skip this step.
        if threshold_config is None:
            print("tag_grouping_threshold not set in config. Skipping tag grouping.")
            return item_info_df
        
        threshold = int(threshold_config)
        print(f"Grouping tags that appear less than {threshold} times.")

        # Calculate how many times each tag appears.
        tag_counts = item_info_df['tag'].value_counts()

        # Identify which tags are rare.
        rare_tags = tag_counts[tag_counts < threshold].index

        if len(rare_tags) > 0:
            # Replace the rare tags with a single, unified category.
            item_info_df.loc[item_info_df['tag'].isin(rare_tags), 'tag'] = 'rare_tag'
            print(f"Grouped {len(rare_tags)} rare tags into a single 'rare_tag' category.")
        else:
            print("No rare tags found below the threshold.")
            
        return item_info_df

def main(cli_args: Optional[List[str]] = None):
    """
    Main function to execute the preprocessing pipeline from the command line.

    This function parses command-line arguments, loads the specified
    configuration, and runs the entire preprocessing pipeline. It allows for
    overriding certain configuration settings via flags.
    """
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
    
    args = parser.parse_args(cli_args)
    
    # Loads the configuration from the specified YAML file.
    config = Config.from_yaml(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Modifies the loaded configuration based on command-line arguments.
    if args.skip_caching:
        # Dynamically removes the caching configuration if the flag is set.
        if hasattr(config.data, 'processed_features_cache_config'):
            delattr(config.data, 'processed_features_cache_config')
        print("Feature caching disabled by --skip-caching flag")
    
    # Creates an instance of the pipeline and runs it.
    pipeline = PreprocessingPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()