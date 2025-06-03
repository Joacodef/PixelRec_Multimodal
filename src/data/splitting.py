# src/data/splitting.py
"""
Improved data splitting strategies for recommender systems
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Literal
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict


class DataSplitter:
    """Handles various data splitting strategies for recommender systems"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    def user_based_split(
        self,
        interactions_df: pd.DataFrame,
        train_ratio: float = 0.8,
        min_interactions_per_user: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by users - ensures no user overlap between train/val.
        Best for evaluating performance on completely new users.
        
        Args:
            interactions_df: DataFrame with user_id, item_id columns
            train_ratio: Fraction of users for training
            min_interactions_per_user: Minimum interactions required per user
            
        Returns:
            train_df, val_df
        """
        # Filter users with minimum interactions
        user_counts = interactions_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions_per_user].index
        
        if len(valid_users) == 0:
            raise ValueError(f"No users have >= {min_interactions_per_user} interactions")
        
        filtered_df = interactions_df[interactions_df['user_id'].isin(valid_users)]
        unique_users = filtered_df['user_id'].unique()
        
        # Split users
        train_users, val_users = train_test_split(
            unique_users, 
            train_size=train_ratio, 
            random_state=self.random_state
        )
        
        train_df = filtered_df[filtered_df['user_id'].isin(train_users)]
        val_df = filtered_df[filtered_df['user_id'].isin(val_users)]
        
        return train_df, val_df
    
    def item_based_split(
        self,
        interactions_df: pd.DataFrame,
        train_ratio: float = 0.8,
        min_interactions_per_item: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by items - ensures no item overlap between train/val.
        Best for evaluating performance on completely new items.
        """
        # Filter items with minimum interactions
        item_counts = interactions_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions_per_item].index
        
        if len(valid_items) == 0:
            raise ValueError(f"No items have >= {min_interactions_per_item} interactions")
        
        filtered_df = interactions_df[interactions_df['item_id'].isin(valid_items)]
        unique_items = filtered_df['item_id'].unique()
        
        # Split items
        train_items, val_items = train_test_split(
            unique_items,
            train_size=train_ratio,
            random_state=self.random_state
        )
        
        train_df = filtered_df[filtered_df['item_id'].isin(train_items)]
        val_df = filtered_df[filtered_df['item_id'].isin(val_items)]
        
        return train_df, val_df
    
    def temporal_split(
        self,
        interactions_df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by time - train on older interactions, validate on newer ones.
        Most realistic for production systems.
        """
        if timestamp_col not in interactions_df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")
        
        # Sort by timestamp
        sorted_df = interactions_df.sort_values(timestamp_col)
        
        # Split by time
        split_idx = int(len(sorted_df) * train_ratio)
        train_df = sorted_df.iloc[:split_idx]
        val_df = sorted_df.iloc[split_idx:]
        
        return train_df, val_df
    
    def leave_one_out_split(
        self,
        interactions_df: pd.DataFrame,
        strategy: Literal['random', 'latest'] = 'random'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Leave-one-out split - hold out one interaction per user for validation.
        Standard approach in many recommender system papers.
        
        Args:
            strategy: 'random' to sample random interaction, 'latest' for most recent
        """
        train_data = []
        val_data = []
        
        # Group by user
        user_groups = interactions_df.groupby('user_id')
        
        for user_id, user_interactions in user_groups:
            user_df = user_interactions.copy()
            
            if len(user_df) < 2:
                # If user has only 1 interaction, put in training
                train_data.append(user_df)
                continue
            
            if strategy == 'random':
                # Random sample for validation
                val_idx = user_df.sample(n=1, random_state=self.random_state).index
            else:  # latest
                # Most recent interaction for validation
                if 'timestamp' in user_df.columns:
                    val_idx = user_df.loc[user_df['timestamp'].idxmax()].name
                    val_idx = [val_idx] if not isinstance(val_idx, list) else val_idx
                else:
                    # If no timestamp, take last row
                    val_idx = [user_df.index[-1]]
            
            val_data.append(user_df.loc[val_idx])
            train_data.append(user_df.drop(val_idx))
        
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        
        return train_df, val_df
    
    def stratified_split(
        self,
        interactions_df: pd.DataFrame,
        train_ratio: float = 0.8,
        min_interactions_per_user: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stratified split - ensures each user has interactions in both train and val.
        Good balance between realism and avoiding cold-start issues.
        """
        train_data = []
        val_data = []
        
        # Group by user
        user_groups = interactions_df.groupby('user_id')
        
        print(f"Stratified split: Processing {len(user_groups)} users...")
        users_with_enough_interactions = 0
        
        for user_id, user_interactions in user_groups:
            user_df = user_interactions.copy()
            
            if len(user_df) < min_interactions_per_user:
                # Put all interactions in training if too few
                train_data.append(user_df)
                continue
            
            users_with_enough_interactions += 1
            
            # Split user's interactions
            n_train = max(1, int(len(user_df) * train_ratio))
            # Ensure at least 1 interaction goes to validation
            n_train = min(n_train, len(user_df) - 1)
            
            # Randomly sample for training
            train_indices = user_df.sample(
                n=n_train, 
                random_state=self.random_state
            ).index
            
            train_data.append(user_df.loc[train_indices])
            val_data.append(user_df.drop(train_indices))
        
        print(f"Users with >= {min_interactions_per_user} interactions: {users_with_enough_interactions}")
        
        # Handle edge cases
        if not train_data:
            raise ValueError("No data available for training after filtering")
        
        train_df = pd.concat(train_data, ignore_index=True)
        
        if not val_data:
            print(f"Warning: No users have >= {min_interactions_per_user} interactions. "
                  f"Using simple random split instead.")
            # Fallback to simple random split
            train_df = interactions_df.sample(
                frac=train_ratio, 
                random_state=self.random_state
            )
            val_df = interactions_df.drop(train_df.index)
        else:
            val_df = pd.concat(val_data, ignore_index=True)
        
        return train_df, val_df
    
    def simple_random_split(
        self,
        interactions_df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple random split - good for very small datasets where stratified approaches fail.
        WARNING: Can cause data leakage as same users/items may appear in both splits.
        """
        train_df = interactions_df.sample(
            frac=train_ratio, 
            random_state=self.random_state
        )
        val_df = interactions_df.drop(train_df.index)
        
        return train_df, val_df
    
    def mixed_split(
        self,
        interactions_df: pd.DataFrame,
        cold_user_ratio: float = 0.1,
        cold_item_ratio: float = 0.1,
        train_ratio: float = 0.8
    ) -> Dict[str, pd.DataFrame]:
        """
        Creates multiple evaluation scenarios:
        - Warm users & warm items (standard)
        - Cold users & warm items  
        - Warm users & cold items
        - Cold users & cold items
        
        Returns dict with keys: 'train', 'val_warm', 'val_cold_user', 'val_cold_item', 'val_cold_both'
        """
        # Get user and item statistics
        user_interactions = interactions_df.groupby('user_id').size()
        item_interactions = interactions_df.groupby('item_id').size()
        
        # Define "cold" users and items (those with fewer interactions)
        cold_user_threshold = user_interactions.quantile(cold_user_ratio)
        cold_item_threshold = item_interactions.quantile(cold_item_ratio)
        
        cold_users = user_interactions[user_interactions <= cold_user_threshold].index
        cold_items = item_interactions[item_interactions <= cold_item_threshold].index
        
        warm_users = user_interactions[user_interactions > cold_user_threshold].index
        warm_items = item_interactions[item_interactions > cold_item_threshold].index
        
        # Create interaction subsets
        def get_subset(users, items):
            return interactions_df[
                interactions_df['user_id'].isin(users) & 
                interactions_df['item_id'].isin(items)
            ]
        
        warm_warm = get_subset(warm_users, warm_items)
        cold_warm = get_subset(cold_users, warm_items)
        warm_cold = get_subset(warm_users, cold_items)
        cold_cold = get_subset(cold_users, cold_items)
        
        # Split warm-warm into train/val
        if len(warm_warm) > 0:
            train_df, val_warm = self.stratified_split(warm_warm, train_ratio)
        else:
            # Fallback if no warm-warm interactions
            train_df, val_warm = self.simple_random_split(interactions_df, train_ratio)
        
        return {
            'train': train_df,
            'val_warm': val_warm,
            'val_cold_user': cold_warm,
            'val_cold_item': warm_cold,
            'val_cold_both': cold_cold
        }
    
    def get_split_statistics(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame
    ) -> Dict[str, any]:
        """Calculate statistics about the split"""
        # Check if train_df is not empty and has 'user_id' and 'item_id' columns
        train_users = set(train_df['user_id'].unique()) if not train_df.empty and 'user_id' in train_df.columns else set()
        train_items = set(train_df['item_id'].unique()) if not train_df.empty and 'item_id' in train_df.columns else set()
        
        # Check if val_df is not empty and has 'user_id' and 'item_id' columns
        val_users = set(val_df['user_id'].unique()) if not val_df.empty and 'user_id' in val_df.columns else set()
        val_items = set(val_df['item_id'].unique()) if not val_df.empty and 'item_id' in val_df.columns else set()
        
        return {
            'train_interactions': len(train_df),
            'val_interactions': len(val_df),
            'train_users': len(train_users),
            'train_items': len(train_items),
            'val_users': len(val_users),
            'val_items': len(val_items),
            'user_overlap': len(train_users & val_users),
            'item_overlap': len(train_items & val_items),
            'user_overlap_ratio': len(train_users & val_users) / len(val_users) if len(val_users) > 0 else 0,
            'item_overlap_ratio': len(train_items & val_items) / len(val_items) if len(val_items) > 0 else 0
        }


# Helper function for backward compatibility
def create_robust_splits(
    interactions_df: pd.DataFrame,
    split_strategy: str = 'stratified',
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation splits using the specified strategy
    
    Args:
        interactions_df: DataFrame with interactions
        split_strategy: One of 'user', 'item', 'temporal', 'stratified', 'leave_one_out', 'simple_random'
        **kwargs: Additional arguments for the splitting strategy
    """
    splitter = DataSplitter(random_state=kwargs.get('random_state', 42))
    
    # Filter kwargs based on the splitting strategy
    if split_strategy == 'user':
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['train_ratio', 'min_interactions_per_user']}
        return splitter.user_based_split(interactions_df, **valid_kwargs)
    
    elif split_strategy == 'item':
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['train_ratio', 'min_interactions_per_item']}
        return splitter.item_based_split(interactions_df, **valid_kwargs)
    
    elif split_strategy == 'temporal':
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['timestamp_col', 'train_ratio']}
        return splitter.temporal_split(interactions_df, **valid_kwargs)
    
    elif split_strategy == 'stratified':
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['train_ratio', 'min_interactions_per_user']}
        return splitter.stratified_split(interactions_df, **valid_kwargs)
    
    elif split_strategy == 'leave_one_out':
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['strategy']}
        return splitter.leave_one_out_split(interactions_df, **valid_kwargs)
    
    elif split_strategy == 'simple_random':
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['train_ratio']}
        return splitter.simple_random_split(interactions_df, **valid_kwargs)
    
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}. "
                        f"Available options: 'user', 'item', 'temporal', 'stratified', "
                        f"'leave_one_out', 'simple_random'")