# scripts/create_evaluation_splits.py
#!/usr/bin/env python
"""Create standardized splits for evaluation"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.splitting import DataSplitter

import pandas as pd
import json
import hashlib

def create_evaluation_splits(config_path, output_dir="data/evaluation_splits"):
    """Create standardized splits for fair comparison"""
    
    config = Config.from_yaml(config_path)
    
    # Load processed data
    interactions_df = pd.read_csv(config.data.processed_interactions_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    splitter = DataSplitter(random_state=42)
    
    # 60/20/20 split
    train_val_df, test_df = splitter.stratified_split(
        interactions_df, 
        train_ratio=0.8,
        min_interactions_per_user=5
    )
    
    train_df, val_df = splitter.stratified_split(
        train_val_df,
        train_ratio=0.75,
        min_interactions_per_user=3
    )
    
    # Save splits
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    # Create metadata
    metadata = {
        "creation_date": pd.Timestamp.now().isoformat(),
        "config_file": config_path,
        "total_interactions": len(interactions_df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_users": train_df['user_id'].nunique(),
        "val_users": val_df['user_id'].nunique(),
        "test_users": test_df['user_id'].nunique(),
        "train_items": train_df['item_id'].nunique(),
        "val_items": val_df['item_id'].nunique(),
        "test_items": test_df['item_id'].nunique(),
        "data_hash": hashlib.md5(interactions_df.to_csv().encode()).hexdigest()
    }
    
    with open(output_path / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created evaluation splits in {output_path}")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    args = parser.parse_args()
    
    create_evaluation_splits(args.config)