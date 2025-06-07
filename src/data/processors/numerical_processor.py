# src/data/processors/numerical_processor.py
"""
Modular numerical feature processing and scaling
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


class NumericalProcessor:
    """Handles numerical feature processing and scaling operations"""
    
    def __init__(self):
        self.scaler = None
        self.fitted_columns = None
    
    def fit_scaler(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        method: str = 'standardization'
    ) -> Optional[Any]:
        """
        Fit a scaler on numerical columns.
        
        Args:
            df: DataFrame containing numerical features
            numerical_columns: List of column names to scale
            method: Scaling method ('standardization', 'min_max', 'log1p', 'none')
            
        Returns:
            Fitted scaler object or None
        """
        if not numerical_columns or method in ['none', 'log1p']:
            print(f"Scaler fitting skipped for method: {method}")
            return None
        
        # Validate columns exist
        missing_cols = [col for col in numerical_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Skipping scaler fitting.")
            return None
        
        # Prepare data
        data_to_scale = df[numerical_columns].fillna(0).values
        
        # Create scaler
        if method == 'standardization':
            self.scaler = StandardScaler()
        elif method == 'min_max':
            self.scaler = MinMaxScaler()
        else:
            print(f"Unknown scaling method: {method}")
            return None
        
        # Fit scaler
        print(f"Fitting {method} scaler on {len(data_to_scale)} samples...")
        self.scaler.fit(data_to_scale)
        self.fitted_columns = numerical_columns.copy()
        
        return self.scaler
    
    def transform_features(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        method: str = 'standardization',
        scaler: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Transform numerical features using specified method.
        
        Args:
            df: Input DataFrame
            numerical_columns: Columns to transform
            method: Transformation method
            scaler: Pre-fitted scaler (optional)
            
        Returns:
            Tuple of (original_df, transformed_features_array)
        """
        if not numerical_columns or method == 'none':
            return df, np.array([])
        
        # Validate columns
        missing_cols = [col for col in numerical_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            available_cols = [col for col in numerical_columns if col in df.columns]
            if not available_cols:
                return df, np.array([])
            numerical_columns = available_cols
        
        # Prepare data
        features = df[numerical_columns].fillna(0).values
        
        # Apply transformation
        if method == 'log1p':
            if np.any(features < 0):
                print("Warning: log1p applied to negative values")
            transformed_features = np.log1p(features)
        elif method in ['standardization', 'min_max'] and scaler is not None:
            transformed_features = scaler.transform(features)
        elif method in ['standardization', 'min_max'] and self.scaler is not None:
            transformed_features = self.scaler.transform(features)
        else:
            print(f"No scaler available for method {method}. Using original features.")
            transformed_features = features
        
        return df, transformed_features
    
    def save_scaler(self, scaler_path: Path) -> bool:
        """
        Save fitted scaler and its column names to disk.
        
        Args:
            scaler_path: Path to save the scaler
            
        Returns:
            True if successful, False otherwise
        """
        if self.scaler is None:
            print("No scaler to save")
            return False
        
        try:
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scaler_path, 'wb') as f:
                # Save both the scaler and the columns it was fitted on
                pickle.dump({'scaler': self.scaler, 'columns': self.fitted_columns}, f)
            print(f"Scaler and column info saved to {scaler_path}")
            return True
        except Exception as e:
            print(f"Error saving scaler: {e}")
            return False
    
    def load_scaler(self, scaler_path: Path) -> bool:
        """
        Load scaler and its column names from disk.
        
        Args:
            scaler_path: Path to load the scaler from
            
        Returns:
            True if successful, False otherwise
        """
        if not scaler_path.exists():
            print(f"Scaler not found at {scaler_path}")
            return False
        
        try:
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
                # Handle both new (dict) and old (direct scaler) formats
                if isinstance(scaler_data, dict):
                    self.scaler = scaler_data.get('scaler')
                    self.fitted_columns = scaler_data.get('columns')
                else:
                    self.scaler = scaler_data
                    self.fitted_columns = None # Old format, columns are unknown
            print(f"Scaler loaded from {scaler_path}")
            return True
        except Exception as e:
            print(f"Error loading scaler: {e}")
            return False
    
    def get_scaler_info(self) -> dict:
        """Get information about the current scaler"""
        if self.scaler is None:
            return {"scaler": None, "fitted_columns": None}
        
        scaler_type = type(self.scaler).__name__
        return {
            "scaler_type": scaler_type,
            "fitted_columns": self.fitted_columns,
            "n_features": len(self.fitted_columns) if self.fitted_columns else 'Unknown'
        }