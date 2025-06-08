# src/data/preprocessing.py
"""
Provides a collection of standalone utility functions for data preprocessing.

This module contains reusable functions for common data cleaning, augmentation,
and validation tasks, such as text augmentation, numerical feature scaling,
HTML tag removal, Unicode normalization, and image validation. These functions
are designed to be independent and can be integrated into various data
processing pipelines.
"""
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import unicodedata
import re
from PIL import Image
import os
from typing import Optional, Any, Tuple

def augment_text(text: str, augmentation_type: str = 'random_delete', delete_prob: float = 0.1, swap_prob: float = 0.1) -> str:
    """
    Applies simple augmentation techniques to a given text string.

    This function can perform random word deletion or swapping of adjacent words
    to create variations of the input text, which can be useful for training
    more robust models.

    Args:
        text (str): The input string to be augmented.
        augmentation_type (str): The type of augmentation to perform.
                                 Options: 'random_delete', 'random_swap', 'none'.
        delete_prob (float): The probability of deleting each word when using
                             the 'random_delete' strategy.
        swap_prob (float): The probability of swapping adjacent words when
                           using the 'random_swap' strategy.

    Returns:
        str: The augmented text string. If the input text is empty or the
             augmentation type is 'none', the original text is returned.
    """
    words = text.split()
    if not words or augmentation_type == 'none':
        return text

    if augmentation_type == 'random_delete':
        augmented_words = [word for word in words if random.random() > delete_prob]
        return " ".join(augmented_words)

    elif augmentation_type == 'random_swap':
        augmented_words = list(words)
        for i in range(len(augmented_words) - 1):
            if random.random() < swap_prob:
                augmented_words[i], augmented_words[i+1] = augmented_words[i+1], augmented_words[i]
        return " ".join(augmented_words)

    else:
        return text


def normalize_features(features: np.ndarray, method: str = 'standardization', scaler: Optional[Any] = None) -> tuple[np.ndarray, Optional[Any]]:
    """
    Scales numerical features using a specified normalization method.

    This function can either fit a new scaler to the provided data or apply a
    pre-fitted scaler. It supports standard scaling, min-max scaling, and
    logarithmic transformation.

    Args:
        features (np.ndarray): A 2D NumPy array of numerical features.
        method (str): The normalization method to apply.
                      Options: 'standardization', 'min_max', 'log1p', 'none'.
        scaler (Optional[Any]): A pre-fitted scikit-learn scaler object. If one
                                is not provided, a new scaler will be fitted to
                                the data.

    Returns:
        A tuple containing:
            - np.ndarray: The array of normalized features.
            - Optional[Any]: The scaler object that was used for the transformation.
                             This is None if the method is 'log1p' or 'none'.
    """
    if not isinstance(features, np.ndarray) or features.size == 0 or method == 'none':
        return features, None

    # Reshapes 1D arrays to 2D for compatibility with scikit-learn scalers.
    if features.ndim == 1:
        features_reshaped = features.reshape(-1, 1)
    else:
        features_reshaped = features

    fitted_scaler = scaler

    if method == 'standardization':
        if scaler is None:
            fitted_scaler = StandardScaler()
            normalized_features = fitted_scaler.fit_transform(features_reshaped)
        else:
            normalized_features = fitted_scaler.transform(features_reshaped)

    elif method == 'min_max':
        if scaler is None:
            fitted_scaler = MinMaxScaler()
            normalized_features = fitted_scaler.fit_transform(features_reshaped)
        else:
            normalized_features = fitted_scaler.transform(features_reshaped)

    elif method == 'log1p':
        if np.any(features_reshaped < 0):
            print("Warning: log1p transform applied to data with negative values. Results might be NaN.")
        normalized_features = np.log1p(features_reshaped)
        fitted_scaler = None

    else:
        # Returns original features if the method is unknown or 'none'.
        print(f"Warning: Unknown or 'none' normalization method '{method}'. Returning original features.")
        return features, None

    return normalized_features, fitted_scaler


def remove_html_tags(text: str) -> str:
    """
    Removes all HTML tags from a given string using regular expressions.

    Args:
        text (str): The input string, which may contain HTML tags.

    Returns:
        str: The cleaned string with all HTML tags removed.
    """
    if not isinstance(text, str):
        return text
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def normalize_unicode_text(text: str) -> str:
    """
    Normalizes Unicode characters in a string to a standard representation.

    This function uses the NFKC (Normalization Form Compatibility Composition)
    standard to handle various Unicode characters, ensuring consistency.

    Args:
        text (str): The input string.

    Returns:
        str: The normalized string.
    """
    if not isinstance(text, str):
        return text
    return unicodedata.normalize('NFKC', text)


def is_image_corrupted(image_path: str) -> bool:
    """
    Checks if an image file is corrupted by attempting to open and load it.

    This provides a reliable way to verify image integrity beyond just checking
    the file extension.

    Args:
        image_path (str): The path to the image file.

    Returns:
        bool: True if the image is corrupted or cannot be read, False otherwise.
    """
    try:
        img = Image.open(image_path)
        # Verifies the file header and integrity.
        img.verify()
        # Re-opens the image and attempts to load the pixel data into memory.
        img = Image.open(image_path)
        img.load()
        return False
    except Exception:
        return True


def check_image_dimensions(image_path: str, min_width: int, min_height: int) -> bool:
    """
    Verifies if an image meets specified minimum dimension requirements.

    Args:
        image_path (str): The path to the image file.
        min_width (int): The minimum required width in pixels.
        min_height (int): The minimum required height in pixels.

    Returns:
        bool: True if the image meets the dimension criteria, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width >= min_width and height >= min_height
    except Exception:
        # Returns False if the image cannot be opened.
        return False