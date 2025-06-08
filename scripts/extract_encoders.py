#!/usr/bin/env python
"""
A utility script for generating and saving user and item ID encoders.

This script reads the complete processed interaction and item data to create
and fit scikit-learn LabelEncoder objects. These encoders map unique user and
item IDs to continuous integer indices, which is a required preprocessing step
for the model's embedding layers. The fitted encoders are then saved to disk
for consistent use during training, evaluation, and inference.
"""

import sys
from pathlib import Path
import pandas as pd
import pickle

# Add the project's root directory to the system path to allow importing from 'src'.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset

def extract_encoders(config_path: str):
    """
    Initializes a dataset to fit user and item encoders and saves them to disk.

    This function orchestrates the encoder creation process. It loads the system
    configuration, reads the full processed datasets, and then instantiates a
    MultimodalDataset object. The dataset's initialization logic automatically
    fits LabelEncoders on all user and item IDs present in the data. Finally,
    it serializes these fitted encoders and saves them to the specified
    directory for later use.

    Args:
        config_path: The file path to the YAML configuration file which contains
                     the paths to the processed data and the output directory.

    Side-effects:
        - Creates an 'encoders' subdirectory within the configured checkpoint directory.
        - Saves 'user_encoder.pkl' and 'item_encoder.pkl' to this directory.
    """
    
    # Loads the main configuration object from the specified YAML file.
    config = Config.from_yaml(config_path)
    data_config = config.data
    
    print("Loading data to create and fit encoders...")
    
    # Loads the complete processed datasets to ensure the encoders are aware of
    # all unique users and items that exist in the system.
    item_info_df = pd.read_csv(data_config.processed_item_info_path)
    interactions_df = pd.read_csv(data_config.processed_interactions_path)
    
    # Instantiates the dataset object. Its primary purpose in this context is to
    # leverage its internal logic that fits the LabelEncoder instances.
    # Other dataset parameters are set to minimal values as they are not
    # relevant for the encoding process.
    print("Creating dataset to fit encoders...")
    dataset = MultimodalDataset(
        interactions_df=interactions_df,
        item_info_df=item_info_df,
        image_folder=data_config.image_folder,
        vision_model_name='clip',
        language_model_name='sentence-bert',
        create_negative_samples=False,
        cache_features=False
    )
    
    # Defines and creates the output directory where the encoder files will be saved.
    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    encoders_dir.mkdir(parents=True, exist_ok=True)
    
    # Serializes and saves the fitted user and item encoders to disk using pickle.
    print(f"Saving encoders to {encoders_dir}...")
    with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.user_encoder, f)
    with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.item_encoder, f)
    
    # Prints a final confirmation message with statistics about the created encoders.
    print(f"Successfully saved encoders!")
    print(f"Number of unique users encoded: {dataset.n_users}")
    print(f"Number of unique items encoded: {dataset.n_items}")

# This block allows the script to be executed from the command line.
if __name__ == '__main__':
    # Sets up an argument parser to handle command-line inputs.
    import argparse
    parser = argparse.ArgumentParser(description='Extract and save user and item encoders from the dataset.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file used for training.')
    args = parser.parse_args()
    
    # Calls the main function with the provided configuration path.
    extract_encoders(args.config)