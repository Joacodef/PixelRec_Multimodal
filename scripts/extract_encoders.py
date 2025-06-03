# Save this as extract_encoders.py and run it to extract encoders from your trained model

import sys
from pathlib import Path
import pandas as pd
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import MultimodalDataset

def extract_encoders(config_path: str):
    """Extract encoders by recreating the dataset with the same data"""
    
    # Load configuration
    config = Config.from_yaml(config_path)
    data_config = config.data
    
    print("Loading data to recreate encoders...")
    
    # Load the same data used for training
    item_info_df = pd.read_csv(data_config.processed_item_info_path)
    interactions_df = pd.read_csv(data_config.processed_interactions_path)
    
    # Create dataset to fit encoders
    print("Creating dataset to fit encoders...")
    dataset = MultimodalDataset(
        interactions_df=interactions_df,
        item_info_df=item_info_df,
        image_folder=data_config.image_folder,  # Doesn't matter for encoders
        vision_model_name='clip',  # Doesn't matter for encoders
        language_model_name='sentence-bert',  # Doesn't matter for encoders
        create_negative_samples=False,
        cache_processed_images=False
    )
    
    # Save encoders
    encoders_dir = Path(config.checkpoint_dir) / 'encoders'
    encoders_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving encoders to {encoders_dir}...")
    with open(encoders_dir / 'user_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.user_encoder, f)
    with open(encoders_dir / 'item_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.item_encoder, f)
    
    print(f"Successfully saved encoders!")
    print(f"Number of users: {dataset.n_users}")
    print(f"Number of items: {dataset.n_items}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract encoders from training data')
    parser.add_argument('--config', type=str, required=True, help='Path to config file used for training')
    args = parser.parse_args()
    
    extract_encoders(args.config)