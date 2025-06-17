# src/data/processors/numerical_processor.py
"""
Modular numerical feature processing for both offline scaling 
and online feature extraction for the model.
"""
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Optional, Any, Tuple, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NumericalProcessor:
    """Handles both offline and online numerical feature processing and scaling."""

    def __init__(
        self,
        # Parameters for online mode (used by Dataset)
        numerical_cols: Optional[List[str]] = None,
        normalization_method: str = 'none',
        scaler: Optional[Any] = None
    ):
        """
        Initializes the NumericalProcessor for either online or offline use.

        Args:
            numerical_cols (Optional[List[str]]): List of columns to process (online mode).
            normalization_method (str): Normalization method to use (online mode).
            scaler (Optional[Any]): A pre-fitted scikit-learn scaler (online mode).
        """
        self.numerical_cols = numerical_cols or []
        self.normalization_method = normalization_method
        self.scaler = scaler
        self.fitted_columns = getattr(scaler, 'feature_names_in_', None)

    # --- Methods for Online Processing (used by Dataset) ---
    def get_scaler_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary with information about the fitted scaler.

        Returns:
            Dict[str, Any]: A dictionary containing the scaler type and fitted columns.
        """
        if not self.scaler:
            return {
                "scaler_type": "None",
                "fitted_columns": []
            }
        
        return {
            "scaler_type": type(self.scaler).__name__,
            "fitted_columns": self.fitted_columns or []
        }
    

    def get_features(self, item_info_row: pd.Series) -> torch.Tensor:
        """
        Extracts and processes numerical features from an item's metadata row.
        ...
        """
        if not self.numerical_cols:
            return torch.empty(0, dtype=torch.float32)

        features = item_info_row.get(self.numerical_cols, pd.Series(0.0, index=self.numerical_cols))
        features = features.fillna(0).values.astype(np.float32).reshape(1, -1)
        
        # Apply scaling if a scaler is present
        if self.scaler and self.normalization_method in ['standardization', 'min_max']:
            features = self.scaler.transform(features)
        # Apply log transform if specified
        elif self.normalization_method == 'log1p':
            features = np.log1p(features)
        
        return torch.tensor(features, dtype=torch.float32).squeeze(0)

    def get_placeholder_tensor(self) -> torch.Tensor:
        """
        Creates a placeholder (zero) tensor for numerical features.

        Returns:
            torch.Tensor: A zero tensor with length equal to the number of numerical features.
        """
        return torch.zeros(len(self.numerical_cols), dtype=torch.float32)
        
    # --- Methods for Offline Processing (used by scripts) ---

    def fit_scaler(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        method: str = 'standardization'
    ) -> Optional[Any]:
        """
        Fit a scaler on numerical columns.
        
        Args:
            df: DataFrame containing numerical features.
            numerical_columns: List of column names to scale.
            method: Scaling method ('standardization', 'min_max', 'log1p', 'none').
            
        Returns:
            Fitted scaler object or None.
        """
        if not numerical_columns or method in ['none', 'log1p']:
            return None
        
        data_to_scale = df[numerical_columns].fillna(0).values
        
        if method == 'standardization':
            self.scaler = StandardScaler()
        elif method == 'min_max':
            self.scaler = MinMaxScaler()
        else:
            return None
        
        self.scaler.fit(data_to_scale)
        self.fitted_columns = numerical_columns.copy()
        
        return self.scaler
    
    def transform_features(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        method: str = 'standardization'
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Transform numerical features using the fitted scaler.
        
        Args:
            df: Input DataFrame.
            numerical_columns: Columns to transform.
            method: Transformation method.
            
        Returns:
            Tuple of (original_df, transformed_features_array).
        """
        if not numerical_columns or method == 'none':
            return df, df[numerical_columns].fillna(0).values

        features = df[numerical_columns].fillna(0).values
        
        if method in ['standardization', 'min_max']:
            # The check for the scaler should happen here, specifically for
            # methods that require it.
            if self.scaler:
                transformed_features = self.scaler.transform(features)
            else:
                # If no scaler exists for a scaling method, return original features.
                transformed_features = features
        elif method == 'log1p':
            transformed_features = np.log1p(features)
        else:
            transformed_features = features
        
        return df, transformed_features

    def save_scaler(self, scaler_path: Path) -> bool:
        """Save fitted scaler and its column names to disk."""
        if self.scaler is None:
            return False
        
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'columns': self.fitted_columns}, f)
        return True
    
    def load_scaler(self, scaler_path: Path) -> bool:
        """Load scaler and its column names from disk."""
        if not scaler_path.exists():
            return False
        
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            if isinstance(scaler_data, dict):
                self.scaler = scaler_data.get('scaler')
                self.fitted_columns = scaler_data.get('columns')
            else:
                self.scaler = scaler_data
                self.fitted_columns = None
        return True