# src/data/processors/data_filter.py
"""
Modular data filtering for interactions and items
"""
import pandas as pd
from typing import Set, Tuple


class DataFilter:
    """Handles data filtering operations for interactions and items"""
    
    @staticmethod
    def filter_interactions_by_valid_items(
        interactions_df: pd.DataFrame, 
        valid_item_ids: Set[str]
    ) -> pd.DataFrame:
        """
        Filter interactions to only include valid items.
        
        Args:
            interactions_df: Input interactions DataFrame
            valid_item_ids: Set of valid item IDs
            
        Returns:
            Filtered interactions DataFrame
        """
        original_count = len(interactions_df)
        
        # Ensure item_id comparison consistency
        filtered_df = interactions_df[
            interactions_df['item_id'].astype(str).isin([str(x) for x in valid_item_ids])
        ].copy()
        
        print(f"Interaction filtering: {len(filtered_df)} interactions remaining "
              f"out of {original_count} after filtering by valid items")
        
        return filtered_df
    
    @staticmethod
    def filter_by_activity(
        interactions_df: pd.DataFrame,
        min_user_interactions: int = 5,
        min_item_interactions: int = 3
    ) -> pd.DataFrame:
        """
        Filter interactions by user and item activity levels.
        
        Args:
            interactions_df: Input interactions DataFrame
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            
        Returns:
            Filtered interactions DataFrame
        """
        df_filtered = interactions_df.copy()
        
        # Filter by item activity first
        if min_item_interactions > 0:
            item_counts = df_filtered['item_id'].value_counts()
            active_items = item_counts[item_counts >= min_item_interactions].index
            df_filtered = df_filtered[df_filtered['item_id'].isin(active_items)].copy()
            
            print(f"Filtered by item activity (min {min_item_interactions}): "
                  f"{len(df_filtered)} interactions, "
                  f"{df_filtered['item_id'].nunique()} items remain")
        
        # Filter by user activity
        if min_user_interactions > 0:
            user_counts = df_filtered['user_id'].value_counts()
            active_users = user_counts[user_counts >= min_user_interactions].index
            df_filtered = df_filtered[df_filtered['user_id'].isin(active_users)].copy()
            
            print(f"Filtered by user activity (min {min_user_interactions}): "
                  f"{len(df_filtered)} interactions, "
                  f"{df_filtered['user_id'].nunique()} users remain")
        
        return df_filtered
    
    @staticmethod
    def align_item_info_with_interactions(
        item_info_df: pd.DataFrame,
        interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter item_info to only include items present in interactions.
        
        Args:
            item_info_df: Item information DataFrame
            interactions_df: Interactions DataFrame
            
        Returns:
            Filtered item_info DataFrame
        """
        interacting_item_ids = set(interactions_df['item_id'].unique())
        
        original_count = len(item_info_df)
        filtered_df = item_info_df[
            item_info_df['item_id'].astype(str).isin([str(x) for x in interacting_item_ids])
        ].copy()
        
        print(f"Item info alignment: {len(filtered_df)} items remaining "
              f"out of {original_count} after filtering by interactions")
        
        return filtered_df
    
    @staticmethod
    def get_filtering_stats(
        original_interactions: pd.DataFrame,
        filtered_interactions: pd.DataFrame,
        original_items: pd.DataFrame,
        filtered_items: pd.DataFrame
    ) -> dict:
        """
        Get comprehensive filtering statistics.
        
        Returns:
            Dictionary with filtering statistics
        """
        stats = {
            "interactions": {
                "original": len(original_interactions),
                "filtered": len(filtered_interactions),
                "retention_rate": len(filtered_interactions) / len(original_interactions)
            },
            "users": {
                "original": original_interactions['user_id'].nunique(),
                "filtered": filtered_interactions['user_id'].nunique(),
                "retention_rate": (filtered_interactions['user_id'].nunique() / 
                                 original_interactions['user_id'].nunique())
            },
            "items": {
                "original": len(original_items),
                "filtered": len(filtered_items),
                "retention_rate": len(filtered_items) / len(original_items)
            }
        }
        
        return stats