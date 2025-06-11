# src/data/splitting.py
"""
Provides a collection of data splitting strategies tailored for recommender systems.

This module contains the DataSplitter class, which implements various methods
for dividing interaction data into training, validation, and test sets. Each
strategy serves a different evaluation purpose, from evaluating performance on
new users and items (cold-start) to simulating a production environment with
temporal splits.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Literal, Union
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict


class DataSplitter:
    """
    A class that encapsulates various data splitting strategies.

    This class provides a suite of methods for splitting recommender system
    datasets. It is initialized with a random state to ensure that all
    splitting operations are reproducible.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initializes the DataSplitter.

        Args:
            random_state (int): The seed for the random number generator to ensure
                                that all splits are deterministic and reproducible.
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    def stratified_temporal_split(
        self,
        interactions_df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        timestamp_col: str = 'timestamp',
        stratify_by: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically, then stratifies future interactions.
        This method ensures the training set is older, and the validation/test
        sets are newer, maintaining user overlap. The returned DataFrames will
        only contain core interaction columns.
        """
        if timestamp_col not in interactions_df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found.")
        if stratify_by and stratify_by not in interactions_df.columns:
            raise ValueError(f"Stratification column '{stratify_by}' not found.")

        # --- This logic is now clean and correct ---
        # It does not add any 'tag' columns to the final output.
        
        # 1. Chronological split
        sorted_df = interactions_df.sort_values(by=timestamp_col).reset_index(drop=True)
        train_end_idx = int(len(sorted_df) * train_ratio)
        train_df = sorted_df.iloc[:train_end_idx]
        future_interactions = sorted_df.iloc[train_end_idx:]

        # 2. Ensure user overlap
        train_users = set(train_df['user_id'].unique())
        future_interactions = future_interactions[future_interactions['user_id'].isin(train_users)]
        
        if future_interactions.empty:
            raise ValueError("No interactions left for validation/test after ensuring user overlap.")

        # 3. Stratified split of future interactions
        test_size = test_ratio / (val_ratio + test_ratio)
        stratify_col_data = future_interactions[stratify_by] if stratify_by else None

        try:
            val_df, test_df = train_test_split(
                future_interactions,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_col_data
            )
        except ValueError as e:
            print(f"Warning: Stratified split failed: {e}. Falling back to random split.")
            val_df, test_df = train_test_split(
                future_interactions,
                test_size=test_size,
                random_state=self.random_state
            )

        # Ensure only core columns are returned, preventing any mix-ups.
        core_columns = ['user_id', 'item_id', 'timestamp']
        return train_df[core_columns], val_df[core_columns], test_df[core_columns]


    def user_based_split(
        self,
        interactions_df: pd.DataFrame,
        train_ratio: float = 0.8,
        min_interactions_per_user: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data by users to prevent user overlap between sets.

        This method is ideal for evaluating a model's ability to generalize to
        entirely new users (a user-level cold-start scenario).

        Args:
            interactions_df (pd.DataFrame): The DataFrame of user-item interactions.
            train_ratio (float): The proportion of users to allocate to the training set.
            min_interactions_per_user (int): The minimum number of interactions
                                             a user must have to be included.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training
                                               and validation DataFrames.
        """
        user_counts = interactions_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions_per_user].index
        
        if len(valid_users) == 0:
            raise ValueError(f"No users have >= {min_interactions_per_user} interactions")
        
        filtered_df = interactions_df[interactions_df['user_id'].isin(valid_users)]
        unique_users = filtered_df['user_id'].unique()
        
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
        Splits data by items to prevent item overlap between sets.

        This method is used for evaluating a model's ability to recommend
        entirely new items (an item-level cold-start scenario).

        Args:
            interactions_df (pd.DataFrame): The DataFrame of user-item interactions.
            train_ratio (float): The proportion of items to allocate to the training set.
            min_interactions_per_item (int): The minimum number of interactions
                                             an item must have to be included.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training
                                               and validation DataFrames.
        """
        item_counts = interactions_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions_per_item].index
        
        if len(valid_items) == 0:
            raise ValueError(f"No items have >= {min_interactions_per_item} interactions")
        
        filtered_df = interactions_df[interactions_df['item_id'].isin(valid_items)]
        unique_items = filtered_df['item_id'].unique()
        
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
        Splits data based on time.

        This method sorts interactions by a timestamp and splits them into a
        training set of older data and a validation set of newer data. This is
        often the most realistic evaluation scenario as it mimics a production
        environment where a model predicts future interactions based on past behavior.

        Args:
            interactions_df (pd.DataFrame): The interactions DataFrame, which must
                                            contain a timestamp column.
            timestamp_col (str): The name of the timestamp column.
            train_ratio (float): The proportion of the timeline to use for training.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training
                                               and validation DataFrames.
        """
        if timestamp_col not in interactions_df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")
        
        sorted_df = interactions_df.sort_values(timestamp_col)
        
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
        Performs a leave-one-out split, holding out one item per user.

        This is a common evaluation strategy in academic literature where for each
        user, a single interaction is held out for the validation set, and the
        model is trained on the rest of that user's interactions.

        Args:
            interactions_df (pd.DataFrame): The DataFrame of user-item interactions.
            strategy ('random' or 'latest'): Determines which item to hold out.
                                            'random' selects a random interaction.
                                            'latest' selects the most recent one
                                            (requires a 'timestamp' column).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training
                                               and validation DataFrames.
        """
        train_data = []
        val_data = []
        
        user_groups = interactions_df.groupby('user_id')
        
        for user_id, user_interactions in user_groups:
            user_df = user_interactions.copy()
            
            if len(user_df) < 2:
                train_data.append(user_df)
                continue
            
            if strategy == 'random':
                val_idx = user_df.sample(n=1, random_state=self.random_state).index
            else:
                if 'timestamp' in user_df.columns:
                    val_idx = user_df.loc[user_df['timestamp'].idxmax()].name
                    val_idx = [val_idx] if not isinstance(val_idx, list) else val_idx
                else:
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
        Performs a stratified split on a per-user basis.

        This method ensures that for each user with enough interactions, their
        interaction history is split between the training and validation sets
        according to the specified ratio. This is useful for evaluating how well
        a model can rank items for known users.

        Args:
            interactions_df (pd.DataFrame): The DataFrame of user-item interactions.
            train_ratio (float): The proportion of each user's interactions to
                                 allocate to the training set.
            min_interactions_per_user (int): The minimum number of interactions
                                             a user must have to be included
                                             in the stratified split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training
                                               and validation DataFrames.
        """
        train_data = []
        val_data = []
        
        user_groups = interactions_df.groupby('user_id')
        
        print(f"Stratified split: Processing {len(user_groups)} users...")
        users_with_enough_interactions = 0
        
        for user_id, user_interactions in user_groups:
            user_df = user_interactions.copy()
            
            if len(user_df) < min_interactions_per_user:
                train_data.append(user_df)
                continue
            
            users_with_enough_interactions += 1
            
            n_train = max(1, int(len(user_df) * train_ratio))
            n_train = min(n_train, len(user_df) - 1)
            
            train_indices = user_df.sample(
                n=n_train, 
                random_state=self.random_state
            ).index
            
            train_data.append(user_df.loc[train_indices])
            val_data.append(user_df.drop(train_indices))
        
        print(f"Users with >= {min_interactions_per_user} interactions: {users_with_enough_interactions}")
        
        if not train_data:
            raise ValueError("No data available for training after filtering")
        
        train_df = pd.concat(train_data, ignore_index=True)
        
        if not val_data:
            print(f"Warning: No users have >= {min_interactions_per_user} interactions. "
                  f"Using simple random split instead.")
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
        Performs a simple random split of interactions.

        This method randomly samples a fraction of the entire interactions
        DataFrame for training and uses the rest for validation. It does not
        guarantee user or item disjointness and should be used with caution.

        Args:
            interactions_df (pd.DataFrame): The DataFrame of interactions.
            train_ratio (float): The fraction of interactions for the training set.

        Returns:
            A tuple of training and validation DataFrames.
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
        Creates multiple validation sets for different cold-start scenarios.

        This advanced method splits the data to create validation sets for:
        - Warm users and warm items (standard evaluation)
        - Cold users (new users) and warm items
        - Warm users and cold items (new items)
        - Cold users and cold items (new users and new items)

        Args:
            interactions_df (pd.DataFrame): The DataFrame of interactions.
            cold_user_ratio (float): The quantile of user activity to define "cold" users.
            cold_item_ratio (float): The quantile of item activity to define "cold" items.
            train_ratio (float): The ratio for the training/validation split of the warm-warm set.

        Returns:
            A dictionary containing the training DataFrame
                                     and the various validation DataFrames.
        """
        user_interactions = interactions_df.groupby('user_id').size()
        item_interactions = interactions_df.groupby('item_id').size()
        
        cold_user_threshold = user_interactions.quantile(cold_user_ratio)
        cold_item_threshold = item_interactions.quantile(cold_item_ratio)
        
        cold_users = user_interactions[user_interactions <= cold_user_threshold].index
        cold_items = item_interactions[item_interactions <= cold_item_threshold].index
        
        warm_users = user_interactions[user_interactions > cold_user_threshold].index
        warm_items = item_interactions[item_interactions > cold_item_threshold].index
        
        def get_subset(users, items):
            return interactions_df[
                interactions_df['user_id'].isin(users) & 
                interactions_df['item_id'].isin(items)
            ]
        
        warm_warm = get_subset(warm_users, warm_items)
        cold_warm = get_subset(cold_users, warm_items)
        warm_cold = get_subset(warm_users, cold_items)
        cold_cold = get_subset(cold_users, cold_items)
        
        if len(warm_warm) > 0:
            train_df, val_warm = self.stratified_split(warm_warm, train_ratio)
        else:
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
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        Calculates and returns statistics about a train/validation/test split.
    
        Args:
            train_df (pd.DataFrame): The training DataFrame.
            val_df (pd.DataFrame): The validation DataFrame.
            test_df (Optional[pd.DataFrame]): The test DataFrame.
    
        Returns:
            A dictionary containing statistics such as interaction counts,
            unique user/item counts, and overlap ratios.
        """
        train_users = set(train_df['user_id'].unique())
        train_items = set(train_df['item_id'].unique())
        val_users = set(val_df['user_id'].unique())
        val_items = set(val_df['item_id'].unique())
    
        stats = {
            'train_interactions': len(train_df),
            'val_interactions': len(val_df),
            'train_users': len(train_users),
            'train_items': len(train_items),
            'val_users': len(val_users),
            'val_items': len(val_items),
            'user_overlap_val': len(train_users & val_users),
            'item_overlap_val': len(train_items & val_items),
            'user_overlap_ratio_val': len(train_users & val_users) / len(val_users) if val_users else 0,
            'item_overlap_ratio_val': len(train_items & val_items) / len(val_items) if val_items else 0
        }
    
        if test_df is not None:
            test_users = set(test_df['user_id'].unique())
            test_items = set(test_df['item_id'].unique())
            stats.update({
                'test_interactions': len(test_df),
                'test_users': len(test_users),
                'test_items': len(test_items),
                'user_overlap_test': len(train_users & test_users),
                'item_overlap_test': len(train_items & test_items),
                'user_overlap_ratio_test': len(train_users & test_users) / len(test_users) if test_users else 0,
                'item_overlap_ratio_test': len(train_items & test_items) / len(test_items) if test_items else 0
            })
            
        return stats


def create_robust_splits(
    interactions_df: pd.DataFrame,
    split_strategy: str = 'stratified',
    **kwargs
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    A factory function to create train/validation splits using a named strategy.

    Args:
        interactions_df (pd.DataFrame): The DataFrame of interactions to split.
        split_strategy (str): The name of the splitting strategy to use.
        **kwargs: Additional arguments to be passed to the chosen splitting method.

    Returns:
        A tuple of training and validation/test DataFrames. The number of returned
        DataFrames depends on the chosen strategy.
    """
    random_state = kwargs.get('random_state', 42)
    splitter = DataSplitter(random_state=random_state)
    
    if split_strategy == 'stratified_temporal':
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['train_ratio', 'val_ratio', 'test_ratio', 'timestamp_col', 'stratify_by']}
        return splitter.stratified_temporal_split(interactions_df, **valid_kwargs)

    elif split_strategy == 'user':
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
                        f"'leave_one_out', 'simple_random', 'stratified_temporal'")