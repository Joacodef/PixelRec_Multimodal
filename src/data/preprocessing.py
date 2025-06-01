import numpy as np
import random
from sklearn.preprocessing import StandardScaler # For Z-score normalization

def augment_text(text: str, augmentation_type: str = 'random_delete', delete_prob: float = 0.1, swap_prob: float = 0.1) -> str:
    """
    Augments text using simple techniques.

    Args:
        text: The input string to augment.
        augmentation_type: Type of augmentation ('random_delete', 'random_swap', 'none').
                           More types can be added.
        delete_prob: Probability of deleting a word (if 'random_delete' is used).
        swap_prob: Probability of swapping adjacent words (if 'random_swap' is used).

    Returns:
        The augmented string.
    """
    words = text.split()
    if len(words) == 0:
        return text

    if augmentation_type == 'random_delete':
        # Randomly delete words with a given probability
        augmented_words = [word for word in words if random.random() > delete_prob]
        if not augmented_words: # Ensure not to return empty string if all words deleted
            return text 
        return " ".join(augmented_words)

    elif augmentation_type == 'random_swap':
        # Randomly swap adjacent words
        augmented_words = list(words) # Create a mutable copy
        for i in range(len(augmented_words) - 1):
            if random.random() < swap_prob:
                # Swap words at i and i+1
                augmented_words[i], augmented_words[i+1] = augmented_words[i+1], augmented_words[i]
        return " ".join(augmented_words)
    
    elif augmentation_type == 'none':
        return text

    else:
        # Default to no augmentation if type is unknown
        # More advanced techniques like back-translation or synonym replacement
        # could be implemented here using libraries like NLTK, spaCy, or transformers.
        # For example, using a pre-trained model for synonym replacement:
        # from nltk.corpus import wordnet
        # ... implementation ...
        # Or using back-translation with a library like `transformers`:
        # from transformers import MarianMTModel, MarianTokenizer
        # ... implementation ...
        # For now, return original text if type is not recognized
        return text


def normalize_features(features: np.ndarray, method: str = 'standardization') -> np.ndarray:
    """
    Normalizes numerical features.

    Args:
        features: A NumPy array of numerical features.
                  Assumes features are in columns if 2D.
                  If 1D, it's treated as a single feature.
        method: Normalization method ('standardization', 'min_max', 'log1p').

    Returns:
        A NumPy array with normalized features.
        Returns original features if method is unknown or not applicable.
    """
    if not isinstance(features, np.ndarray) or features.size == 0:
        return features # Return as is if not a numpy array or empty

    # Reshape 1D array to 2D for consistent processing with sklearn scalers
    if features.ndim == 1:
        features_reshaped = features.reshape(-1, 1)
    else:
        features_reshaped = features

    if method == 'standardization':
        # Z-score normalization (mean=0, std=1)
        # This is generally robust.
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_reshaped)
    
    elif method == 'min_max':
        # Min-Max scaling (scales to a range, typically [0, 1])
        # Sensitive to outliers.
        min_val = np.min(features_reshaped, axis=0)
        max_val = np.max(features_reshaped, axis=0)
        # Avoid division by zero if max_val == min_val for any feature
        range_val = max_val - min_val
        # Set range_val to 1 where it's 0 to prevent division by zero (feature is constant)
        range_val[range_val == 0] = 1.0 
        normalized_features = (features_reshaped - min_val) / range_val
        
    elif method == 'log1p':
        # Log transform (log(1+x))
        # Useful for skewed data.
        # The dataset.py already uses this for specific features.
        # This option is provided here for general use if needed.
        if np.any(features_reshaped < 0):
            print("Warning: log1p transform applied to data with negative values. Results might be NaN.")
        normalized_features = np.log1p(features_reshaped)
        
    else:
        print(f"Warning: Unknown normalization method '{method}'. Returning original features.")
        return features # Return original if method is unknown

    # If original was 1D, return 1D
    if features.ndim == 1 and normalized_features.shape[1] == 1:
        return normalized_features.flatten()
    return normalized_features