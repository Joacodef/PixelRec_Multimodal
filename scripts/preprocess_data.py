#!/usr/bin/env python
"""
Preprocess raw data for training
"""
import argparse
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

def validate_images(item_info_df, image_folder):
    """Validate that images exist for all items"""
    missing_images = []
    for item_id in tqdm(item_info_df['item_id'], desc="Validating images"):
        image_path = Path(image_folder) / f"{item_id}.jpg"
        if not image_path.exists():
            # Check alternative extensions
            found = False
            for ext in ['.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                if (Path(image_folder) / f"{item_id}{ext}").exists():
                    found = True
                    break
            if not found:
                missing_images.append(item_id)
    
    return missing_images

def clean_text_data(df):
    """Clean text fields"""
    text_fields = ['title', 'tag', 'description']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].fillna('')
            df[field] = df[field].astype(str).str.strip()
    return df

def filter_inactive_users(interactions_df, min_interactions=5):
    """Filter users with too few interactions"""
    user_counts = interactions_df['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index
    return interactions_df[interactions_df['user_id'].isin(active_users)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--item_info', required=True)
    parser.add_argument('--interactions', required=True)
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_user_interactions', type=int, default=5)
    parser.add_argument('--min_item_interactions', type=int, default=3)
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    item_info_df = pd.read_csv(args.item_info)
    interactions_df = pd.read_csv(args.interactions)
    
    # Clean data
    print("Cleaning data...")
    item_info_df = clean_text_data(item_info_df)
    
    # Filter interactions
    print("Filtering interactions...")
    interactions_df = filter_inactive_users(
        interactions_df, 
        args.min_user_interactions
    )
    
    # Validate images
    print("Validating images...")
    missing_images = validate_images(item_info_df, args.image_folder)
    if missing_images:
        print(f"Warning: {len(missing_images)} items have missing images")
        item_info_df = item_info_df[~item_info_df['item_id'].isin(missing_images)]
        interactions_df = interactions_df[~interactions_df['item_id'].isin(missing_images)]
    
    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    item_info_df.to_csv(output_dir / 'item_info_processed.csv', index=False)
    interactions_df.to_csv(output_dir / 'interactions_processed.csv', index=False)
    
    print(f"Processed data saved to {output_dir}")
    print(f"Final dataset: {len(interactions_df)} interactions, "
          f"{interactions_df['user_id'].nunique()} users, "
          f"{interactions_df['item_id'].nunique()} items")

if __name__ == '__main__':
    main()