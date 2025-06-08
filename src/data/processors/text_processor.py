# src/data/processors/text_processor.py
"""
Modular text processing for cleaning and normalization
"""
import pandas as pd
from typing import List
from tqdm import tqdm
import re  # Import the regular expression module

from ..preprocessing import remove_html_tags, normalize_unicode_text
from ...config import OfflineTextCleaningConfig


class TextProcessor:
    """Handles text cleaning and normalization operations"""
    
    def __init__(self, config: OfflineTextCleaningConfig):
        self.config = config
    
    def clean_text_field(self, text: str) -> str:
        """
        Clean a single text field according to configuration.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ''
        
        # Remove HTML tags
        if self.config.remove_html:
            text = remove_html_tags(text)
        
        # Normalize Unicode
        if self.config.normalize_unicode:
            text = normalize_unicode_text(text)
        
        # Convert to lowercase
        if self.config.to_lowercase:
            text = text.lower()
        
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_dataframe_text_columns(
        self, 
        df: pd.DataFrame, 
        text_columns: List[str]
    ) -> pd.DataFrame:
        """
        Clean text columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names to clean
            
        Returns:
            DataFrame with cleaned text columns
        """
        df_cleaned = df.copy()
        
        print(f"Cleaning {len(text_columns)} text columns...")
        
        for col in tqdm(text_columns, desc="Cleaning text fields"):
            if col in df_cleaned.columns:
                # Fill NaN values and ensure string type
                df_cleaned[col] = df_cleaned[col].astype(str).fillna('')
                
                # Apply cleaning function
                df_cleaned[col] = df_cleaned[col].apply(self.clean_text_field)
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")
        
        return df_cleaned
    
    def get_combined_text(
        self, 
        row: pd.Series, 
        text_columns: List[str], 
        separator: str = " "
    ) -> str:
        """
        Combine multiple text columns into a single text string.
        
        Args:
            row: DataFrame row (pandas Series)
            text_columns: List of column names to combine
            separator: String to use for joining
            
        Returns:
            Combined text string
        """
        texts = []
        for col in text_columns:
            if col in row.index and pd.notna(row[col]):
                text = str(row[col]).strip()
                if text:
                    texts.append(text)
        
        return separator.join(texts)