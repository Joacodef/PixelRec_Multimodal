"""
Recommendation generation and inference
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import pickle
from pathlib import Path
import pandas as pd

from ..data.dataset import MultimodalDataset
from ..evaluation.novelty import NoveltyMetrics, DiversityCalculator 
try:
    from ..data.feature_cache import ProcessedFeatureCache
except ImportError:
    ProcessedFeatureCache = None 
    print("Warning: ProcessedFeatureCache could not be imported. Ensure it's defined and path is correct.")


class Recommender:
    """Class for generating recommendations"""
    
    def __init__(
        self,
        model: nn.Module,
        dataset: MultimodalDataset,
        device: torch.device,
        item_embeddings_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        processed_feature_cache: Optional[Any] = None
    ) -> None:
        """
        Initialize recommender.
        
        Args:
            model: Trained model
            dataset: Dataset with encoders and data
            device: Device for inference
            item_embeddings_cache: Pre-computed item features
        """
        self.model: nn.Module = model
        self.dataset: MultimodalDataset = dataset
        self.device: torch.device = device
        # This L1 cache stores the fully assembled features for an item.
        self.item_features_cache: Dict[str, Dict[str, torch.Tensor]] = item_embeddings_cache or {}
        
        # This L2 cache is for non-image features (text tokens, numerical), potentially disk-backed.
        self.processed_feature_cache = processed_feature_cache
        
        # Initialize metrics calculators
        item_popularity: Dict[str, float] = self.dataset.get_item_popularity()
        user_history: List[Tuple[str, str]] = []
        if 'user_id' in self.dataset.interactions.columns and 'item_id' in self.dataset.interactions.columns:
            user_history = [
                (row['user_id'], row['item_id']) 
                for _, row in self.dataset.interactions.iterrows()
            ]
        
        self.novelty_metrics: NoveltyMetrics = NoveltyMetrics(item_popularity, user_history)
        
        # Put model in eval mode
        self.model.eval()
    
    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-k recommendations for a user.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_seen: Whether to filter already seen items
            candidates: Optional list of candidate items
            
        Returns:
            List of (item_id, score) tuples
        """
        # Get user index
        try:
            user_idx: int = self.dataset.user_encoder.transform([user_id])[0]
        except Exception as e:
            print(f"User {user_id} not found in training data or encoder error: {e}")
            return []
        
        # Get candidate items
        if candidates is None:
            candidates = list(self.dataset.item_encoder.classes_)
        
        # Filter seen items if requested
        if filter_seen:
            user_items_history: set = self.dataset.get_user_history(user_id)
            candidates = [item for item in candidates if item not in user_items_history]
        
        # Score all candidate items
        scores: List[Tuple[str, float]] = self._score_items(user_idx, candidates)
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_diverse_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        diversity_weight: float = 0.3,
        novelty_weight: float = 0.2,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Get diverse recommendations with novelty considerations.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            diversity_weight: Weight for diversity (0-1)
            novelty_weight: Weight for novelty (0-1)
            filter_seen: Whether to filter seen items
            candidates: Optional candidate items
            
        Returns:
            Tuple of (recommendations, novelty_metrics)
        """
        # Get user index
        try:
            user_idx: int = self.dataset.user_encoder.transform([user_id])[0]
        except Exception as e:
            print(f"User {user_id} not found in training data or encoder error: {e}")
            return [], {}
        
        # Get candidate items
        if candidates is None:
            candidates = list(self.dataset.item_encoder.classes_)[:1000]
        
        # Filter seen items
        if filter_seen:
            user_items_history: set = self.dataset.get_user_history(user_id)
            candidates = [item for item in candidates if item not in user_items_history]
        
        if not candidates:
            print(f"No candidates left for user {user_id} after filtering.")
            return [], {}

        # Score items with embeddings
        scored_items: List[Dict[str, Any]] = self._score_items_with_embeddings(user_idx, candidates)
        
        if not scored_items:
            print(f"No items could be scored for user {user_id}.")
            return [], {}

        # Initial ranking by score
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        
        # Rerank considering diversity and novelty
        final_recommendations: List[Dict[str, Any]] = self._rerank_for_diversity(
            scored_items,
            top_k,
            diversity_weight,
            novelty_weight
        )
        
        # Calculate metrics
        rec_ids: List[str] = [r['item_id'] for r in final_recommendations]
        metrics: Dict[str, float] = self.novelty_metrics.calculate_metrics(rec_ids, user_id)
        
        return final_recommendations, metrics
    
    def _score_items(
        self, 
        user_idx: int, 
        items: List[str],
        batch_size: int = 256
    ) -> List[Tuple[str, float]]:
        """Score a list of items for a user using batched processing"""
        scores: List[Tuple[str, float]] = []
        
        # Process items in batches for efficiency
        with torch.no_grad():
            for i in tqdm(range(0, len(items), batch_size), desc="Scoring items"):
                batch_items: List[str] = items[i:i + batch_size]
                batch_scores: List[Tuple[str, float]] = self._score_item_batch(user_idx, batch_items)
                scores.extend(batch_scores)
        
        return scores
    
    def _score_item_batch(self, user_idx: int, item_ids: List[str]) -> List[Tuple[str, float]]:
        """Score a batch of items efficiently"""
        batch_size: int = len(item_ids)
        
        # Prepare batch tensors
        user_indices: torch.Tensor = torch.full((batch_size,), user_idx, dtype=torch.long).to(self.device)
        item_indices_list: List[int] = [] # Renamed to avoid confusion with item_indices_tensor
        valid_items: List[str] = []
        # valid_indices: List[int] = [] # Not strictly used later, can be removed if not needed

        # Get item indices and filter invalid items
        for idx, item_id in enumerate(item_ids):
            try:
                item_idx_val: int = self.dataset.item_encoder.transform([item_id])[0]
                item_indices_list.append(item_idx_val)
                valid_items.append(item_id)
                # valid_indices.append(idx)
            except Exception: # More general exception
                # print(f"Warning: Item ID {item_id} not found in item_encoder. Skipping.")
                continue
        
        if not item_indices_list:
            return []
        
        # Create tensors for valid items
        item_indices_tensor: torch.Tensor = torch.tensor(item_indices_list, dtype=torch.long).to(self.device)
        user_indices_valid: torch.Tensor = user_indices[:len(item_indices_list)] # Adjust batch size for user_indices
        
        # Batch process features
        batch_images: List[torch.Tensor] = []
        batch_text_input_ids: List[torch.Tensor] = [] # Changed key
        batch_text_attention_masks: List[torch.Tensor] = [] # Changed key
        batch_numerical: List[torch.Tensor] = []
        # For CLIP features, if your model uses them directly in this scoring path
        batch_clip_text_input_ids: List[torch.Tensor] = []
        batch_clip_text_attention_masks: List[torch.Tensor] = []
        
        # Get expected dimensions from model or dataset defaults
        expected_num_features: int = 7 
        if hasattr(self.dataset, 'numerical_feat_cols') and self.dataset.numerical_feat_cols:
            expected_num_features = len(self.dataset.numerical_feat_cols)
        elif hasattr(self.model, 'numerical_projection') and hasattr(self.model.numerical_projection, 'in_features'): # For nn.Linear
             expected_num_features = self.model.numerical_projection.in_features
        elif hasattr(self.model, 'numerical_projection') and isinstance(self.model.numerical_projection, nn.Sequential) and self.model.numerical_projection and hasattr(self.model.numerical_projection[0], 'in_features'): # For Sequential
            expected_num_features = self.model.numerical_projection[0].in_features


        img_height, img_width = (224, 224) # Default
        if hasattr(self.dataset.image_processor, 'size'):
            proc_size = self.dataset.image_processor.size
            if isinstance(proc_size, dict) and 'shortest_edge' in proc_size:
                img_height = img_width = proc_size['shortest_edge']
            elif isinstance(proc_size, (tuple, list)) and len(proc_size) >= 2:
                img_height, img_width = proc_size[0], proc_size[1]
            elif isinstance(proc_size, int):
                img_height = img_width = proc_size


        main_tokenizer_max_len = getattr(self.dataset.tokenizer, 'model_max_length', 128)
        clip_tokenizer_max_len = 77 # Standard CLIP max length
        
        first_valid_features_processed = False

        for item_id in valid_items:
            features: Optional[Dict[str, torch.Tensor]] = self._get_item_features(item_id)
            
            if features is None: # Fallback if _get_item_features fails for an item
                print(f"Warning: Could not retrieve features for item {item_id}. Using placeholders.")
                # Use updated keys for placeholder features
                current_features = {
                    'image': torch.zeros(3, img_height, img_width, dtype=torch.float),
                    'text_input_ids': torch.zeros(main_tokenizer_max_len, dtype=torch.long),
                    'text_attention_mask': torch.zeros(main_tokenizer_max_len, dtype=torch.long),
                    'numerical_features': torch.zeros(expected_num_features, dtype=torch.float32)
                }
                if self.dataset.clip_tokenizer_for_contrastive:
                    current_features['clip_text_input_ids'] = torch.zeros(clip_tokenizer_max_len, dtype=torch.long)
                    current_features['clip_text_attention_mask'] = torch.zeros(clip_tokenizer_max_len, dtype=torch.long)
            else:
                current_features = features

            # Update dynamic dimensions from the first successfully processed valid item's features
            if not first_valid_features_processed and current_features:
                if 'image' in current_features and current_features['image'].ndim == 3: # C, H, W
                    img_height, img_width = current_features['image'].shape[1], current_features['image'].shape[2]
                if 'text_input_ids' in current_features: # Use new key
                    main_tokenizer_max_len = current_features['text_input_ids'].shape[0]
                if 'numerical_features' in current_features:
                    expected_num_features = current_features['numerical_features'].shape[0]
                if 'clip_text_input_ids' in current_features:
                    clip_tokenizer_max_len = current_features['clip_text_input_ids'].shape[0]
                first_valid_features_processed = True
            
            # Ensure numerical features have correct shape if they were somehow empty from _get_item_features
            if 'numerical_features' not in current_features or current_features['numerical_features'].numel() == 0:
                 current_features['numerical_features'] = torch.zeros(expected_num_features, dtype=torch.float32)
            if 'image' not in current_features: # Should not happen if _get_item_features is robust
                 current_features['image'] = torch.zeros(3, img_height, img_width, dtype=torch.float)


            batch_images.append(current_features['image'])
            batch_text_input_ids.append(current_features.get('text_input_ids', torch.zeros(main_tokenizer_max_len, dtype=torch.long))) # Use new key
            batch_text_attention_masks.append(current_features.get('text_attention_mask', torch.zeros(main_tokenizer_max_len, dtype=torch.long))) # Use new key
            batch_numerical.append(current_features['numerical_features'])
            
            if self.dataset.clip_tokenizer_for_contrastive: # Check if model will need these
                batch_clip_text_input_ids.append(current_features.get('clip_text_input_ids', torch.zeros(clip_tokenizer_max_len, dtype=torch.long)))
                batch_clip_text_attention_masks.append(current_features.get('clip_text_attention_mask', torch.zeros(clip_tokenizer_max_len, dtype=torch.long)))

        # Stack all features
        images_tensor: torch.Tensor = torch.stack(batch_images).to(self.device)
        text_input_ids_tensor: torch.Tensor = torch.stack(batch_text_input_ids).to(self.device) # Changed key
        text_attention_masks_tensor: torch.Tensor = torch.stack(batch_text_attention_masks).to(self.device) # Changed key
        numerical_tensor: torch.Tensor = torch.stack(batch_numerical).to(self.device)
        
        model_input_args = {
            'user_idx': user_indices_valid,
            'item_idx': item_indices_tensor,
            'image': images_tensor,
            'text_input_ids': text_input_ids_tensor, # Pass to model with this key
            'text_attention_mask': text_attention_masks_tensor, # Pass to model with this key
            'numerical_features': numerical_tensor
        }
        # Add CLIP inputs if model expects them for this path (usually for contrastive loss or if embeddings are returned)
        # The Recommender's get_recommendations doesn't typically ask for return_embeddings=True for this path
        # but _score_items_with_embeddings does. So we prepare them if available.
        if batch_clip_text_input_ids and batch_clip_text_attention_masks:
             model_input_args['clip_text_input_ids'] = torch.stack(batch_clip_text_input_ids).to(self.device)
             model_input_args['clip_text_attention_mask'] = torch.stack(batch_clip_text_attention_masks).to(self.device)
        
        # Get batch predictions
        with torch.no_grad():
            # Check if model expects return_embeddings (usually false for simple scoring)
            # The base get_recommendations path does not set return_embeddings=True
            if 'return_embeddings' in self.model.forward.__code__.co_varnames:
                 model_input_args['return_embeddings'] = False # Explicitly false for scoring path
            
            output_val = self.model(**model_input_args)
            
            # Handle if model returns tuple (e.g. output, embeddings)
            if isinstance(output_val, tuple):
                batch_scores_tensor = output_val[0].squeeze()
            else:
                batch_scores_tensor = output_val.squeeze()
        
        # Handle single item case for scores
        if batch_scores_tensor.dim() == 0:
            batch_scores_tensor = batch_scores_tensor.unsqueeze(0)
        
        # Pair items with scores
        results: List[Tuple[str, float]] = []
        for item_id, score_val in zip(valid_items, batch_scores_tensor.cpu().numpy()):
            results.append((item_id, float(score_val)))
        
        return results
    
    def _score_single_item(self, user_idx: int, item_id: str) -> float:
        """Score a single item for a user"""
        item_features: Optional[Dict[str, torch.Tensor]] = self._get_item_features(item_id)
        if item_features is None:
            print(f"Warning: Could not get features for item {item_id} in _score_single_item. Returning -inf.")
            return -float('inf')
        
        try:
            item_idx_arr: np.ndarray = self.dataset.item_encoder.transform([item_id])
            if len(item_idx_arr) == 0: 
                print(f"Warning: Item {item_id} not in item_encoder during _score_single_item. Returning -inf.")
                return -float('inf')
            item_idx_val: int = item_idx_arr[0]
        except Exception:
            print(f"Warning: Error transforming item_id {item_id} in _score_single_item. Returning -inf.")
            return -float('inf')
        
        user_tensor: torch.Tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        item_tensor: torch.Tensor = torch.tensor([item_idx_val], dtype=torch.long).to(self.device)
        
        model_input_args = {
            'user_idx': user_tensor,
            'item_idx': item_tensor,
            'image': item_features['image'].unsqueeze(0).to(self.device),
            'text_input_ids': item_features['text_input_ids'].unsqueeze(0).to(self.device), # Changed key
            'text_attention_mask': item_features['text_attention_mask'].unsqueeze(0).to(self.device), # Changed key
            'numerical_features': item_features['numerical_features'].unsqueeze(0).to(self.device)
        }

        if 'clip_text_input_ids' in item_features and 'clip_text_attention_mask' in item_features:
            model_input_args['clip_text_input_ids'] = item_features['clip_text_input_ids'].unsqueeze(0).to(self.device)
            model_input_args['clip_text_attention_mask'] = item_features['clip_text_attention_mask'].unsqueeze(0).to(self.device)

        if 'return_embeddings' in self.model.forward.__code__.co_varnames:
            model_input_args['return_embeddings'] = False


        output_val = self.model(**model_input_args)
        
        if isinstance(output_val, tuple):
            score = output_val[0].item()
        else:
            score = output_val.item()
            
        return score
    
    def _score_items_with_embeddings(
        self, 
        user_idx: int, 
        items: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Score items and return with embeddings and other info using batched processing"""
        scored_items_list: List[Dict[str, Any]] = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(items), batch_size), desc="Scoring items with embeddings"):
                batch_items: List[str] = items[i:i + batch_size]
                batch_results: List[Dict[str, Any]] = self._score_item_batch_with_embeddings(user_idx, batch_items)
                scored_items_list.extend(batch_results)
        
        return scored_items_list

    def _score_item_batch_with_embeddings(self, user_idx: int, item_ids: List[str]) -> List[Dict[str, Any]]:
        """Score a batch of items and return embeddings efficiently"""
        batch_size: int = len(item_ids)
        results: List[Dict[str, Any]] = []
        
        # Prepare batch tensors
        user_indices: torch.Tensor = torch.full((batch_size,), user_idx, dtype=torch.long).to(self.device)
        item_indices: List[int] = []
        valid_items: List[str] = []
        valid_indices: List[int] = []
        
        # Get item indices and filter invalid items
        for idx, item_id in enumerate(item_ids):
            try:
                item_idx: int = self.dataset.item_encoder.transform([item_id])[0]
                item_indices.append(item_idx)
                valid_items.append(item_id)
                valid_indices.append(idx)
            except:
                continue
        
        if not item_indices:
            return []
        
        # Create tensors for valid items
        item_indices_tensor: torch.Tensor = torch.tensor(item_indices, dtype=torch.long).to(self.device)
        user_indices_valid: torch.Tensor = user_indices[:len(item_indices)]
        
        # Batch process features
        batch_images: List[torch.Tensor] = []
        batch_text_ids: List[torch.Tensor] = []
        batch_text_masks: List[torch.Tensor] = []
        batch_numerical: List[torch.Tensor] = []
        
        # Get the expected number of numerical features
        expected_num_features: int = 7  # Default value from config
        if hasattr(self.dataset, 'numerical_feat_cols') and self.dataset.numerical_feat_cols:
            expected_num_features = len(self.dataset.numerical_feat_cols)
        elif hasattr(self.model, 'numerical_projection') and hasattr(self.model.numerical_projection[0], 'in_features'):
            expected_num_features = self.model.numerical_projection[0].in_features
        
        for item_id in valid_items:
            features: Optional[Dict[str, torch.Tensor]] = self._get_item_features(item_id)
            if features is None:
                # Create dummy features with correct dimensions
                img_height: int = 224
                img_width: int = 224
                if hasattr(self.dataset, 'image_processor'):
                    try:
                        if hasattr(self.dataset.image_processor, 'size'):
                            processor_size = self.dataset.image_processor.size
                            if isinstance(processor_size, dict):
                                if 'height' in processor_size:
                                    img_height = processor_size['height']
                                    img_width = processor_size['width']
                                elif 'shortest_edge' in processor_size:
                                    img_height = img_width = processor_size['shortest_edge']
                            elif isinstance(processor_size, (int, float)):
                                img_height = img_width = int(processor_size)
                    except:
                        pass
                
                # Get text sequence length from tokenizer
                max_length: int = 128
                if hasattr(self.dataset, 'tokenizer') and hasattr(self.dataset.tokenizer, 'model_max_length'):
                    max_length = min(self.dataset.tokenizer.model_max_length, 512)
                    
                features = {
                    'image': torch.zeros(3, img_height, img_width),
                    'text_ids': torch.zeros(max_length, dtype=torch.long),
                    'text_mask': torch.zeros(max_length, dtype=torch.long),
                    'numerical': torch.zeros(expected_num_features)
                }
            
            batch_images.append(features['image'])
            batch_text_ids.append(features['text_ids'])
            batch_text_masks.append(features['text_mask'])
            batch_numerical.append(features['numerical'])
        
        # Stack all features
        images_tensor: torch.Tensor = torch.stack(batch_images).to(self.device)
        text_ids_tensor: torch.Tensor = torch.stack(batch_text_ids).to(self.device)
        text_masks_tensor: torch.Tensor = torch.stack(batch_text_masks).to(self.device)
        numerical_tensor: torch.Tensor = torch.stack(batch_numerical).to(self.device)
        
        # Get batch predictions with embeddings
        with torch.no_grad():
            output_tuple: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = self.model(
                user_idx=user_indices_valid,
                item_idx=item_indices_tensor,
                image=images_tensor,
                text_input_ids=text_ids_tensor,
                text_attention_mask=text_masks_tensor,
                numerical_features=numerical_tensor,
                return_embeddings=True
            )
        
        # Unpack the output tuple
        if isinstance(output_tuple, tuple) and len(output_tuple) >= 4:
            batch_scores: torch.Tensor = output_tuple[0]
            vision_embeddings: torch.Tensor = output_tuple[3]
        else:
            batch_scores = output_tuple if not isinstance(output_tuple, tuple) else output_tuple[0]
            vision_embeddings = torch.randn(len(valid_items), self.model.embedding_dim).to(self.device)
        
        # Handle single item case for scores
        if batch_scores.dim() == 0:
            batch_scores = batch_scores.unsqueeze(0)
        
        # Get item popularity from NoveltyMetrics
        popularity_dict: Dict[str, float] = self.novelty_metrics.item_popularity
        
        # Build results with all required information
        for i, (item_id, score) in enumerate(zip(valid_items, batch_scores.cpu().numpy())):
            result: Dict[str, Any] = {
                'item_id': item_id,
                'score': float(score),
                'embedding': vision_embeddings[i].cpu().numpy() if i < len(vision_embeddings) else np.zeros(self.model.embedding_dim),
                'popularity': popularity_dict.get(item_id, 0.0)
            }
            results.append(result)
        
        return results

    def _get_item_features(self, item_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get preprocessed features for an item.
        Uses L1 RAM cache (self.item_features_cache) for fully assembled features.
        Uses L2 disk/hybrid cache (self.processed_feature_cache) for non-image features.
        Image tensors are fetched via self.dataset._load_and_process_image (SharedImageCache).
        """
        # Check L1 RAM cache for fully assembled features first
        if item_id in self.item_features_cache:
            return self.item_features_cache[item_id]

        non_image_features: Optional[Dict[str, torch.Tensor]] = None
        
        # Try to get non-image features from the L2 processed_feature_cache
        if self.processed_feature_cache:
            non_image_features = self.processed_feature_cache.get(item_id)

        if non_image_features is None:
            # L2 cache miss for non-image features, so process them
            try:
                item_info_series = self.dataset._get_item_info(item_id)
                text_content = self.dataset._get_item_text(item_info_series)

                # Main tokenizer (consistent with MultimodalDataset.__getitem__)
                main_tokenizer_max_len = 128
                if hasattr(self.dataset.tokenizer, 'model_max_length') and self.dataset.tokenizer.model_max_length:
                    main_tokenizer_max_len = self.dataset.tokenizer.model_max_length
                
                text_tokens = self.dataset.tokenizer(
                    text_content,
                    padding='max_length',
                    truncation=True,
                    max_length=main_tokenizer_max_len,
                    return_tensors='pt'
                )

                # Numerical features (consistent with MultimodalDataset._get_item_numerical_features)
                # This directly calls the method that handles pre-processed or raw numerical features.
                numerical_features_tensor = self.dataset._get_item_numerical_features(item_id, item_info_series)
                
                # Ensure numerical features have the correct shape if empty or not preprocessed,
                # mimicking logic from original _get_item_features if necessary.
                # This fallback might be redundant if _get_item_numerical_features is robust.
                if numerical_features_tensor.numel() == 0 and self.dataset.numerical_feat_cols:
                    expected_size = len(self.dataset.numerical_feat_cols)
                    if hasattr(self.model, 'numerical_projection'):
                        if isinstance(self.model.numerical_projection, nn.Sequential) and self.model.numerical_projection:
                            first_layer = self.model.numerical_projection[0]
                            if hasattr(first_layer, 'in_features'):
                                expected_size = first_layer.in_features
                        elif hasattr(self.model.numerical_projection, 'in_features'):
                            expected_size = self.model.numerical_projection.in_features
                    numerical_features_tensor = torch.zeros(expected_size, dtype=torch.float32)
                elif not self.dataset.numerical_feat_cols: # No numerical features configured
                    numerical_features_tensor = torch.empty(0, dtype=torch.float32)


                non_image_features = {
                    'text_input_ids': text_tokens['input_ids'].squeeze(0),
                    'text_attention_mask': text_tokens['attention_mask'].squeeze(0),
                    'numerical_features': numerical_features_tensor
                }

                # Add CLIP specific tokens if the dataset's CLIP tokenizer is available
                # (consistent with MultimodalDataset.__getitem__)
                if self.dataset.clip_tokenizer_for_contrastive:
                    clip_tokens = self.dataset.clip_tokenizer_for_contrastive(
                        text_content,
                        padding='max_length',
                        truncation=True,
                        max_length=77, # Standard CLIP max length
                        return_tensors='pt'
                    )
                    non_image_features['clip_text_input_ids'] = clip_tokens['input_ids'].squeeze(0)
                    non_image_features['clip_text_attention_mask'] = clip_tokens['attention_mask'].squeeze(0)
                
                # Store these processed non-image features in the L2 cache
                if self.processed_feature_cache:
                    self.processed_feature_cache.set(item_id, non_image_features)

            except Exception as e:
                print(f"Error processing non-image features for item {item_id}: {e}")
                # Fallback: create empty/default non-image features if processing fails
                # to allow the process to continue, though this item might be scored poorly.
                main_tokenizer_max_len = getattr(self.dataset.tokenizer, 'model_max_length', 128)
                num_num_feats = len(self.dataset.numerical_feat_cols) if self.dataset.numerical_feat_cols else 0
                if num_num_feats == 0 and hasattr(self.model, 'num_numerical_features'): # from multimodal.py
                    num_num_feats = self.model.num_numerical_features

                non_image_features = {
                    'text_input_ids': torch.zeros(main_tokenizer_max_len, dtype=torch.long),
                    'text_attention_mask': torch.zeros(main_tokenizer_max_len, dtype=torch.long),
                    'numerical_features': torch.zeros(num_num_feats, dtype=torch.float32)
                }
                if self.dataset.clip_tokenizer_for_contrastive:
                    non_image_features['clip_text_input_ids'] = torch.zeros(77, dtype=torch.long)
                    non_image_features['clip_text_attention_mask'] = torch.zeros(77, dtype=torch.long)
        
        # Get image tensor (this uses SharedImageCache via the dataset)
        try:
            image_tensor = self.dataset._load_and_process_image(item_id)
        except Exception as e:
            print(f"Error loading image tensor for item {item_id} (will use placeholder): {e}")
            # Determine placeholder size dynamically if possible
            placeholder_size = (224, 224) # Default
            if hasattr(self.dataset.image_processor, 'size'):
                proc_size = self.dataset.image_processor.size
                if isinstance(proc_size, dict) and 'shortest_edge' in proc_size:
                    placeholder_size = (proc_size['shortest_edge'], proc_size['shortest_edge'])
                elif isinstance(proc_size, (tuple, list)) and len(proc_size) >= 2:
                    placeholder_size = (proc_size[0], proc_size[1])
                elif isinstance(proc_size, int):
                    placeholder_size = (proc_size, proc_size)
            image_tensor = torch.zeros(3, placeholder_size[0], placeholder_size[1], dtype=torch.float)


        # Combine non-image features (from L2 cache or freshly processed) and image tensor
        # Ensure non_image_features is not None before trying to merge
        if non_image_features is None: # Should not happen if error handling above is correct
            print(f"Critical Error: non_image_features is None for item {item_id} before combining. Using placeholder non_image_features.")
            main_tokenizer_max_len = getattr(self.dataset.tokenizer, 'model_max_length', 128)
            num_num_feats = len(self.dataset.numerical_feat_cols) if self.dataset.numerical_feat_cols else 0
            if num_num_feats == 0 and hasattr(self.model, 'num_numerical_features'):
                num_num_feats = self.model.num_numerical_features
            non_image_features = {
                'text_input_ids': torch.zeros(main_tokenizer_max_len, dtype=torch.long),
                'text_attention_mask': torch.zeros(main_tokenizer_max_len, dtype=torch.long),
                'numerical_features': torch.zeros(num_num_feats, dtype=torch.float32)
            }
            if self.dataset.clip_tokenizer_for_contrastive:
                non_image_features['clip_text_input_ids'] = torch.zeros(77, dtype=torch.long)
                non_image_features['clip_text_attention_mask'] = torch.zeros(77, dtype=torch.long)

        all_features = {**non_image_features, 'image': image_tensor}
        
        # Store the fully assembled features in the L1 RAM cache
        self.item_features_cache[item_id] = all_features
        
        return all_features
    
    def _rerank_for_diversity(
        self,
        scored_items: List[Dict[str, Any]],
        top_k: int,
        diversity_weight: float,
        novelty_weight: float
    ) -> List[Dict[str, Any]]:
        """Rerank items considering diversity and novelty using MMR-like approach."""
        final_recommendations: List[Dict[str, Any]] = []
        selected_embeddings: List[np.ndarray] = []
        
        # Calculate total weight for relevance once
        relevance_weight: float = 1.0 - diversity_weight - novelty_weight
        if relevance_weight < 0:
            print("Warning: Sum of diversity and novelty weights exceeds 1.0. Clamping relevance_weight.")
            total_dynamic_weight: float = diversity_weight + novelty_weight
            if total_dynamic_weight > 0:
                diversity_weight /= total_dynamic_weight
                novelty_weight /= total_dynamic_weight
            relevance_weight = 0.0

        if not scored_items:
            return []
            
        # Max popularity for normalization
        all_popularities: List[float] = [s['popularity'] for s in scored_items if 'popularity' in s]
        if not all_popularities:
             max_popularity: float = 1.0
        else:
            max_popularity = max(all_popularities) + 1.0

        # Make a copy of scored_items to modify during reranking
        candidate_pool: List[Dict[str, Any]] = list(scored_items)

        while len(final_recommendations) < top_k and candidate_pool:
            best_item_for_current_step: Optional[Dict[str, Any]] = None
            max_combined_score_for_current_step: float = -float('inf')
            best_item_idx_in_pool: int = -1

            for i, item in enumerate(candidate_pool):
                # Base relevance score
                current_item_score: float = relevance_weight * item['score']
                
                # Diversity score
                if selected_embeddings and diversity_weight > 0:
                    item_embedding: Optional[np.ndarray] = item.get('embedding')
                    if item_embedding is not None:
                        similarities: List[float] = [
                            1 - (np.dot(item_embedding.flatten(), s_emb.flatten()) / 
                                 (np.linalg.norm(item_embedding.flatten()) * np.linalg.norm(s_emb.flatten()) + 1e-9))
                            for s_emb in selected_embeddings
                        ]
                        diversity_contribution: float = np.mean(similarities) if similarities else 0
                        current_item_score += diversity_weight * diversity_contribution

                # Novelty score
                if novelty_weight > 0:
                    item_popularity_val: float = item.get('popularity', 0.0)
                    normalized_popularity: float = item_popularity_val / max_popularity
                    novelty_score_val: float = 1.0 - normalized_popularity 
                    current_item_score += novelty_weight * novelty_score_val
                
                if current_item_score > max_combined_score_for_current_step:
                    max_combined_score_for_current_step = current_item_score
                    best_item_for_current_step = item
                    best_item_idx_in_pool = i
            
            if best_item_for_current_step is not None:
                final_recommendations.append(best_item_for_current_step)
                if 'embedding' in best_item_for_current_step and best_item_for_current_step['embedding'] is not None:
                    selected_embeddings.append(best_item_for_current_step['embedding'])
                candidate_pool.pop(best_item_idx_in_pool)
            else:
                break
                
        return final_recommendations

    def load_embeddings_cache(self, cache_path: str) -> None:
        """Loads the embeddings cache from a pickle file."""
        try:
            with open(cache_path, 'rb') as f:
                self.item_features_cache = pickle.load(f)
            print(f"Embeddings cache loaded from {cache_path}")
        except FileNotFoundError:
            print(f"Warning: Embeddings cache file not found at {cache_path}. Starting with an empty cache.")
            self.item_features_cache = {}
        except Exception as e:
            print(f"Error loading embeddings cache from {cache_path}: {e}")
            self.item_features_cache = {}

    def save_embeddings_cache(self, cache_path: str) -> None:
        """Saves the current embeddings cache to a pickle file."""
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.item_features_cache, f)
            print(f"Embeddings cache saved to {cache_path}")
        except Exception as e:
            print(f"Error saving embeddings cache to {cache_path}: {e}")

    def load_item_features_cache(self, cache_path: str) -> None:
        """Alias for load_embeddings_cache for backward compatibility."""
        self.load_embeddings_cache(cache_path)
    
    def save_item_features_cache(self, cache_path: str) -> None:
        """Alias for save_embeddings_cache for backward compatibility."""
        self.save_embeddings_cache(cache_path)