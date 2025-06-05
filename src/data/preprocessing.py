# src/data/preprocessing.py
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
    Augments text using simple techniques.

    Args:
        text: The input string to augment.
        augmentation_type: Type of augmentation ('random_delete', 'random_swap', 'none').
        delete_prob: Probability of deleting a word (if 'random_delete' is used).
        swap_prob: Probability of swapping adjacent words (if 'random_swap' is used).

    Returns:
        The augmented string.
    """
    words = text.split()
    if not words or augmentation_type == 'none':
        return text

    if augmentation_type == 'random_delete':
        augmented_words = [word for word in words if random.random() > delete_prob]
        if not augmented_words:
            return text
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
    Normalizes numerical features. Can fit a new scaler or use a pre-fitted one.

    Args:
        features: A NumPy array of numerical features (2D).
        method: Normalization method ('standardization', 'min_max', 'log1p', 'none').
        scaler: A pre-fitted scaler object (e.g., from sklearn.preprocessing).
                If None and method requires fitting, a new scaler will be fitted.

    Returns:
        A tuple containing:
            - A NumPy array with normalized features.
            - The scaler object used (fitted or passed-in). None if method is 'log1p' or 'none'.
    """
    if not isinstance(features, np.ndarray) or features.size == 0 or method == 'none':
        return features, None

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
        fitted_scaler = None # log1p does not have a scikit-learn style scaler here

    else:
        print(f"Warning: Unknown or 'none' normalization method '{method}'. Returning original features.")
        return features, None

    return normalized_features, fitted_scaler


def remove_html_tags(text: str) -> str:
    """Removes HTML tags from a string."""
    if not isinstance(text, str):
        return text
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def normalize_unicode_text(text: str) -> str:
    """Normalizes unicode characters in a string."""
    if not isinstance(text, str):
        return text
    return unicodedata.normalize('NFKC', text)


def is_image_corrupted(image_path: str) -> bool:
    """Checks if an image file is corrupted by trying to open it."""
    try:
        img = Image.open(image_path)
        img.verify()  # Verify is a superficial check, load() is more thorough
        img = Image.open(image_path)
        img.load()    # Try to load the image data
        return False
    except Exception:
        return True


def check_image_dimensions(image_path: str, min_width: int, min_height: int) -> bool:
    """Checks if an image meets minimum dimension requirements."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width >= min_width and height >= min_height
    except Exception:
        return False # If image can't be opened, it fails the check