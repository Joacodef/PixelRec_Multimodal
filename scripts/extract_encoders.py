#!/usr/bin/env python
"""
A utility script for generating and saving user, item, and tag ID encoders.

This script reads the complete processed interaction and item data to create
and fit scikit-learn LabelEncoder objects. These encoders map unique user,
item, and tag IDs to continuous integer indices, which is a required
preprocessing step for the model's embedding layers. The fitted encoders are
then saved to disk for consistent use during training, evaluation, and inference.
"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import argparse

# Add the project's root directory to the system path to allow importing from 'src'.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset

def main(config_path: str):
    """
    Initializes a dataset to fit user, item, and tag encoders and saves them to disk.

    This function orchestrates the encoder creation process. It loads the system
    configuration, reads the full processed datasets, and then instantiates a
    MultimodalDataset object. The dataset's initialization logic automatically
    fits LabelEncoders on all user, item, and specified categorical feature IDs
    present in the data. Finally, it serializes these fitted encoders and saves
    them to the specified directory for later use.

    Args:
        config_path: The file path to the YAML configuration file which contains
                     the paths to the processed data and the output directory.

    Side-effects:
        - Creates an 'encoders' subdirectory within the configured checkpoint directory.
        - Saves 'user_encoder.pkl', 'item_encoder.pkl', and 'tag_encoder.pkl'
          to this directory.
    """
    
    # Loads the main configuration object from the specified YAML file.
    config = Config.from_yaml(config_path)
    data_config = config.data
    
    print("Loading data to create and fit encoders...")
    
    # Loads the complete processed datasets to ensure the encoders are aware of
    # all unique users, items, and tags that exist in the system.
    item_info_df = pd.read_csv(data_config.processed_item_info_path)
    interactions_df = pd.read_csv(data_config.processed_interactions_path)
    
    # Instantiates the dataset object. Its primary purpose in this context is to
    # leverage its internal logic that fits the LabelEncoder instances.
    print("Creating dataset to fit encoders...")
    dataset = MultimodalDataset(
        interactions_df=interactions_df,
        item_info_df=item_info_df,
        image_folder=data_config.image_folder,
        vision_model_name='clip',
        language_model_name='sentence-bert',
        create_negative_samples=False,
        cache_features=False,
        # CHANGED: Pass the categorical features config to enable tag encoding.
        categorical_feat_cols=data_config.categorical_features_cols
    )
    
    # Defines and creates the output directory where the encoder files will be saved.
    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    encoders_dir.mkdir(parents=True, exist_ok=True)
    
    # Serializes and saves the fitted encoders to disk using pickle.
    print(f"Saving encoders to {encoders_dir}...")
    with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.user_encoder, f)
    with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.item_encoder, f)

    # Checks for and saves the tag encoder if it was created.
    if hasattr(dataset, 'tag_encoder'):
        with open(encoders_dir / 'tag_encoder.pkl', 'wb') as f:
            pickle.dump(dataset.tag_encoder, f)

    # Prints a final confirmation message with statistics about the created encoders.
    print(f"Successfully saved encoders!")
    print(f"Number of unique users encoded: {dataset.n_users}")
    print(f"Number of unique items encoded: {dataset.n_items}")
    
    # --- START OF MODIFICATION ---
    if hasattr(dataset, 'n_tags'):
        print(f"Number of unique tags encoded: {dataset.n_tags}")
    # --- END OF MODIFICATION ---

# This block allows the script to be executed from the command line.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and save user, item, and tag encoders from the dataset.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file used for training.')
    args = parser.parse_args()
    
    main(args.config)