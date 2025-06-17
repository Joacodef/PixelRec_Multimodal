# src/data/processors/text_processor.py
"""
Modular text processing for both offline cleaning and normalization, 
and online tokenization for model input.
"""
import pandas as pd
import torch
import re
from tqdm import tqdm
from typing import List, Dict, Optional
from transformers import AutoTokenizer

from ..preprocessing import remove_html_tags, normalize_unicode_text
from ...config import (
    OfflineTextCleaningConfig, 
    TextAugmentationConfig, 
    MODEL_CONFIGS
)


class TextProcessor:
    """Handles both offline text cleaning and online tokenization operations."""

    def __init__(
        self,
        # Parameters for online mode (used by Dataset)
        model_name: Optional[str] = None,
        augmentation_config: Optional[TextAugmentationConfig] = None,
        # Parameters for offline mode (used by preprocessing scripts)
        cleaning_config: Optional[OfflineTextCleaningConfig] = None
    ):
        """
        Initializes the TextProcessor for either online or offline use.

        Args:
            model_name (Optional[str]): The key for the language model config (online mode).
            augmentation_config (Optional[TextAugmentationConfig]): Augmentation config (online mode).
            cleaning_config (Optional[OfflineTextCleaningConfig]): Cleaning config (offline mode).
        """
        self.cleaning_config = cleaning_config
        self.augmentation_config = augmentation_config
        
        # Initialize components for ONLINE mode if model_name is provided
        if model_name:
            self.online_config = MODEL_CONFIGS['language'].get(model_name)
            if not self.online_config:
                raise ValueError(f"Configuration for language model '{model_name}' not found.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.online_config['name'])
            self.max_length = self.tokenizer.model_max_length
        else:
            self.online_config = None
            self.tokenizer = None
            self.max_length = None

    # --- Methods for Online Processing (used by Dataset) ---

    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a text string for model input.

        Args:
            text (str): The input text.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'input_ids' and 'attention_mask'.
        """
        if not self.tokenizer:
            raise RuntimeError("TextProcessor not initialized for online mode. Provide 'model_name'.")

        # Augmentation logic would be applied here.
        
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'text_input_ids': inputs['input_ids'].squeeze(0),
            'text_attention_mask': inputs['attention_mask'].squeeze(0)
        }

    def get_placeholder_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Creates placeholder (zero) tensors for tokenized text.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with placeholder 'input_ids' and 'attention_mask'.
        """
        if not self.max_length:
            raise RuntimeError("TextProcessor not initialized for online mode. Provide 'model_name'.")
            
        return {
            'text_input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'text_attention_mask': torch.zeros(self.max_length, dtype=torch.long)
        }

    # --- Methods for Offline Processing (used by scripts) ---

    def clean_text_field(self, text: str) -> str:
        """
        Clean a single text field according to the cleaning configuration.
        
        Args:
            text: Input text string.
            
        Returns:
            Cleaned text string.
        """
        if not self.cleaning_config:
            raise RuntimeError("TextProcessor not initialized for offline mode. Provide 'cleaning_config'.")

        if not isinstance(text, str):
            text = str(text) if text is not None else ''
        
        if self.cleaning_config.remove_html:
            text = remove_html_tags(text)
        
        if self.cleaning_config.normalize_unicode:
            text = normalize_unicode_text(text)
        
        if self.cleaning_config.to_lowercase:
            text = text.lower()
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def clean_dataframe_text_columns(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Clean specified text columns in a DataFrame.
        
        Args:
            df: Input DataFrame.
            text_columns: List of column names to clean.
            
        Returns:
            DataFrame with cleaned text columns.
        """
        df_cleaned = df.copy()
        for col in tqdm(text_columns, desc="Cleaning text fields"):
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(str).fillna('')
                df_cleaned[col] = df_cleaned[col].apply(self.clean_text_field)
        return df_cleaned
    
    def get_combined_text(self, row: pd.Series, text_columns: List[str], separator: str = " ") -> str:
        """
        Combine multiple text columns from a row into a single string.
        
        Args:
            row: DataFrame row (pandas Series).
            text_columns: List of column names to combine.
            separator: String to use for joining.
            
        Returns:
            Combined text string.
        """
        texts = []
        for col in text_columns:
            if col in row.index and pd.notna(row[col]):
                text = str(row[col]).strip()
                if text:
                    texts.append(text)
        return separator.join(texts)