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
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp

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
        candidates: Optional[List[str]] = None,
        batch_size: int = 256 
    ) -> List[Tuple[str, float]]:
        """Get top-k recommendations for a user.
        
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
        scores: List[Tuple[str, float]] = self._score_items(
            user_idx, 
            candidates, 
            batch_size=batch_size,
            show_progress=len(candidates) > 5000
        )
            
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
        batch_size: int = 256,  
        show_progress: bool = True
    ) -> List[Tuple[str, float]]:
        """Score a list of items for a user using batched processing"""
        scores: List[Tuple[str, float]] = []
        
        # Process items in batches for efficiency
        with torch.no_grad():
            iterator = range(0, len(items), batch_size)
            if show_progress and len(items) > 5000:  # Only show progress for large item sets
                iterator = tqdm(iterator, desc="Scoring items", unit="batch")
            
            for i in iterator:
                batch_items: List[str] = items[i:i + batch_size]
                batch_scores: List[Tuple[str, float]] = self._score_item_batch(user_idx, batch_items)
                scores.extend(batch_scores)
        
        return scores
    
    def _score_item_batch(self, user_idx: int, item_ids: List[str]) -> List[Tuple[str, float]]:
        """Score a batch of items efficiently using batch feature loading"""
        batch_size: int = len(item_ids)
        
        # Get valid item indices
        item_indices_list: List[int] = []
        valid_items: List[str] = []
        
        for item_id in item_ids:
            try:
                item_idx_val: int = self.dataset.item_encoder.transform([item_id])[0]
                item_indices_list.append(item_idx_val)
                valid_items.append(item_id)
            except Exception:
                continue
        
        if not item_indices_list:
            return []
        
        # Batch load all features at once
        features_batch = self._get_batch_features(valid_items)
        
        # Prepare tensors for model
        user_indices: torch.Tensor = torch.full((len(valid_items),), user_idx, dtype=torch.long).to(self.device)
        item_indices_tensor: torch.Tensor = torch.tensor(item_indices_list, dtype=torch.long).to(self.device)
        
        # Stack features
        batch_images = []
        batch_text_input_ids = []
        batch_text_attention_masks = []
        batch_numerical = []
        batch_clip_text_input_ids = []
        batch_clip_text_attention_masks = []
        
        for features in features_batch:
            if features is not None:
                batch_images.append(features['image'])
                batch_text_input_ids.append(features['text_input_ids'])
                batch_text_attention_masks.append(features['text_attention_mask'])
                batch_numerical.append(features['numerical_features'])
                
                if 'clip_text_input_ids' in features:
                    batch_clip_text_input_ids.append(features['clip_text_input_ids'])
                    batch_clip_text_attention_masks.append(features['clip_text_attention_mask'])
        
        # Convert to tensors
        images_tensor = torch.stack(batch_images).to(self.device)
        text_input_ids_tensor = torch.stack(batch_text_input_ids).to(self.device)
        text_attention_masks_tensor = torch.stack(batch_text_attention_masks).to(self.device)
        numerical_tensor = torch.stack(batch_numerical).to(self.device)
        
        # Prepare model input
        model_input_args = {
            'user_idx': user_indices,
            'item_idx': item_indices_tensor,
            'image': images_tensor,
            'text_input_ids': text_input_ids_tensor,
            'text_attention_mask': text_attention_masks_tensor,
            'numerical_features': numerical_tensor
        }
        
        if batch_clip_text_input_ids:
            model_input_args['clip_text_input_ids'] = torch.stack(batch_clip_text_input_ids).to(self.device)
            model_input_args['clip_text_attention_mask'] = torch.stack(batch_clip_text_attention_masks).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            if 'return_embeddings' in self.model.forward.__code__.co_varnames:
                model_input_args['return_embeddings'] = False
            
            output_val = self.model(**model_input_args)
            
            if isinstance(output_val, tuple):
                batch_scores_tensor = output_val[0].squeeze()
            else:
                batch_scores_tensor = output_val.squeeze()
        
        # Handle single item case
        if batch_scores_tensor.dim() == 0:
            batch_scores_tensor = batch_scores_tensor.unsqueeze(0)
        
        # Return results
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
        batch_size: int = 256
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

                # Numerical features
                numerical_features_tensor = self.dataset._get_item_numerical_features(item_id, item_info_series)
                
                # Ensure numerical features have the correct shape if empty
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
                elif not self.dataset.numerical_feat_cols:
                    numerical_features_tensor = torch.empty(0, dtype=torch.float32)

                non_image_features = {
                    'text_input_ids': text_tokens['input_ids'].squeeze(0),
                    'text_attention_mask': text_tokens['attention_mask'].squeeze(0),
                    'numerical_features': numerical_features_tensor
                }

                # Add CLIP specific tokens if available
                if self.dataset.clip_tokenizer_for_contrastive:
                    clip_tokens = self.dataset.clip_tokenizer_for_contrastive(
                        text_content,
                        padding='max_length',
                        truncation=True,
                        max_length=77,
                        return_tensors='pt'
                    )
                    non_image_features['clip_text_input_ids'] = clip_tokens['input_ids'].squeeze(0)
                    non_image_features['clip_text_attention_mask'] = clip_tokens['attention_mask'].squeeze(0)
                
                # Store in L2 cache
                if self.processed_feature_cache:
                    self.processed_feature_cache.set(item_id, non_image_features)

            except Exception as e:
                print(f"Error processing non-image features for item {item_id}: {e}")
                # Create fallback features
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
        
        # Get image tensor
        try:
            image_tensor = self.dataset._load_and_process_image(item_id)
        except Exception as e:
            print(f"Error loading image tensor for item {item_id}: {e}")
            # Create placeholder
            placeholder_size = (224, 224)
            if hasattr(self.dataset.image_processor, 'size'):
                proc_size = self.dataset.image_processor.size
                if isinstance(proc_size, dict) and 'shortest_edge' in proc_size:
                    placeholder_size = (proc_size['shortest_edge'], proc_size['shortest_edge'])
                elif isinstance(proc_size, (tuple, list)) and len(proc_size) >= 2:
                    placeholder_size = (proc_size[0], proc_size[1])
                elif isinstance(proc_size, int):
                    placeholder_size = (proc_size, proc_size)
            image_tensor = torch.zeros(3, placeholder_size[0], placeholder_size[1], dtype=torch.float)

        # Combine all features
        all_features = {**non_image_features, 'image': image_tensor}
        
        # Store in L1 cache
        self.item_features_cache[item_id] = all_features
        
        return all_features


    def _get_batch_features(self, item_ids: List[str]) -> List[Dict[str, torch.Tensor]]:
        """
        Get features for multiple items efficiently using batch processing.
        This method minimizes redundant operations by processing all uncached items together.
        """
        features_list = []
        uncached_items = []
        uncached_indices = []
        
        # Step 1: Check L1 cache first
        for idx, item_id in enumerate(item_ids):
            if item_id in self.item_features_cache:
                features_list.append(self.item_features_cache[item_id])
            else:
                uncached_items.append(item_id)
                uncached_indices.append(idx)
                features_list.append(None)  # Placeholder
        
        # If all items were cached, return immediately
        if not uncached_items:
            return features_list
        
        # Step 2: Check L2 cache for non-image features
        non_image_features_map = {}
        items_needing_processing = []
        
        if self.processed_feature_cache:
            for item_id in uncached_items:
                cached_features = self.processed_feature_cache.get(item_id)
                if cached_features is not None:
                    non_image_features_map[item_id] = cached_features
                else:
                    items_needing_processing.append(item_id)
        else:
            items_needing_processing = uncached_items
        
        # Step 3: Batch process text and numerical features for items not in L2 cache
        if items_needing_processing:
            # Batch get item info
            item_info_batch = []
            for item_id in items_needing_processing:
                try:
                    item_info = self.dataset._get_item_info(item_id)
                    item_info_batch.append((item_id, item_info))
                except Exception as e:
                    print(f"Error getting item info for {item_id}: {e}")
                    item_info_batch.append((item_id, None))
            
            # Batch process text
            text_contents = []
            valid_item_ids = []
            for item_id, item_info in item_info_batch:
                if item_info is not None:
                    text_content = self.dataset._get_item_text(item_info)
                    text_contents.append(text_content)
                    valid_item_ids.append((item_id, item_info))
            
            if text_contents:
                # Batch tokenize all texts at once
                main_tokenizer_max_len = getattr(self.dataset.tokenizer, 'model_max_length', 128)
                
                text_tokens_batch = self.dataset.tokenizer(
                    text_contents,
                    padding='max_length',
                    truncation=True,
                    max_length=main_tokenizer_max_len,
                    return_tensors='pt'
                )
                
                # CLIP tokenization if needed
                clip_tokens_batch = None
                if self.dataset.clip_tokenizer_for_contrastive:
                    clip_tokens_batch = self.dataset.clip_tokenizer_for_contrastive(
                        text_contents,
                        padding='max_length',
                        truncation=True,
                        max_length=77,
                        return_tensors='pt'
                    )
                
                # Process each item's features
                for idx, (item_id, item_info) in enumerate(valid_item_ids):
                    # Get numerical features
                    numerical_features = self.dataset._get_item_numerical_features(item_id, item_info)
                    
                    # Assemble non-image features
                    features_dict = {
                        'text_input_ids': text_tokens_batch['input_ids'][idx],
                        'text_attention_mask': text_tokens_batch['attention_mask'][idx],
                        'numerical_features': numerical_features
                    }
                    
                    if clip_tokens_batch is not None:
                        features_dict['clip_text_input_ids'] = clip_tokens_batch['input_ids'][idx]
                        features_dict['clip_text_attention_mask'] = clip_tokens_batch['attention_mask'][idx]
                    
                    non_image_features_map[item_id] = features_dict
                    
                    # Cache in L2 if available
                    if self.processed_feature_cache:
                        self.processed_feature_cache.set(item_id, features_dict)
        
        # Step 4: Load images (can potentially be parallelized)
        image_tensors_map = {}
        for item_id in uncached_items:
            try:
                image_tensor = self.dataset._load_and_process_image(item_id)
                image_tensors_map[item_id] = image_tensor
            except Exception as e:
                print(f"Error loading image for {item_id}: {e}")
                # Create placeholder
                placeholder_size = (224, 224)
                if hasattr(self.dataset.image_processor, 'size'):
                    proc_size = self.dataset.image_processor.size
                    if isinstance(proc_size, dict) and 'shortest_edge' in proc_size:
                        placeholder_size = (proc_size['shortest_edge'], proc_size['shortest_edge'])
                image_tensors_map[item_id] = torch.zeros(3, placeholder_size[0], placeholder_size[1], dtype=torch.float)
        
        # Step 5: Assemble complete features and update caches
        for idx, item_id in zip(uncached_indices, uncached_items):
            # Get non-image features
            non_image_feats = non_image_features_map.get(item_id)
            if non_image_feats is None:
                # Create fallback features
                main_tokenizer_max_len = getattr(self.dataset.tokenizer, 'model_max_length', 128)
                num_features = len(self.dataset.numerical_feat_cols) if self.dataset.numerical_feat_cols else 7
                
                non_image_feats = {
                    'text_input_ids': torch.zeros(main_tokenizer_max_len, dtype=torch.long),
                    'text_attention_mask': torch.zeros(main_tokenizer_max_len, dtype=torch.long),
                    'numerical_features': torch.zeros(num_features, dtype=torch.float32)
                }
                
                if self.dataset.clip_tokenizer_for_contrastive:
                    non_image_feats['clip_text_input_ids'] = torch.zeros(77, dtype=torch.long)
                    non_image_feats['clip_text_attention_mask'] = torch.zeros(77, dtype=torch.long)
            
            # Get image tensor
            image_tensor = image_tensors_map.get(item_id, torch.zeros(3, 224, 224, dtype=torch.float))
            
            # Combine all features
            complete_features = {**non_image_feats, 'image': image_tensor}
            
            # Update L1 cache
            self.item_features_cache[item_id] = complete_features
            
            # Update features list
            features_list[idx] = complete_features
        
        return features_list

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

    def get_recommendations_parallel(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidates: Optional[List[str]] = None,
        num_workers: int = 4,
        chunk_size: int = 5000,
        show_progress: bool = False # Added parameter to control internal tqdm
    ) -> List[Tuple[str, float]]:
        """Get top-k recommendations using parallel processing."""
        # Get user index
        try:
            user_idx: int = self.dataset.user_encoder.transform([user_id])[0]
        except Exception as e:
            # print(f"User {user_id} not found in training data or encoder error: {e}") # Commenting out for cleaner output during normal runs
            return []
        
        # Get candidate items
        current_candidates: List[str]
        if candidates is None:
            if hasattr(self.dataset, 'item_encoder') and hasattr(self.dataset.item_encoder, 'classes_') and self.dataset.item_encoder.classes_ is not None:
                current_candidates = list(self.dataset.item_encoder.classes_)
            else:
                # print(f"Warning: Item encoder or classes not available for user {user_id}. Returning empty recommendations.") # Cleaner output
                return []
        else:
            current_candidates = list(candidates)
        
        # Filter seen items if requested
        if filter_seen:
            user_items_history: set = self.dataset.get_user_history(user_id)
            current_candidates = [item for item in current_candidates if item not in user_items_history]
        
        if not current_candidates:
            return []
        
        # Score items in parallel
        scores: List[Tuple[str, float]] = self._score_items_parallel(
            user_idx, 
            current_candidates, 
            num_workers=num_workers,
            chunk_size=chunk_size,
            show_progress=show_progress # Pass the flag here
        )
            
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _score_items_parallel(
        self,
        user_idx: int,
        items: List[str],
        num_workers: int = 8,
        chunk_size: int = 5000,
        show_progress: bool = True # Added parameter
    ) -> List[Tuple[str, float]]:
        """Score items using parallel processing with controlled tqdm."""
        
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        all_scores: List[Tuple[str, float]] = []
        
        def process_chunk(chunk_items: List[str]) -> List[Tuple[str, float]]:
            # _score_item_batch_optimized calls _score_item_batch, which does not have its own tqdm.
            # So, no need to pass show_progress further down from here for tqdm control.
            return self._score_item_batch_optimized(user_idx, chunk_items)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            iterable_chunks = as_completed(future_to_chunk)
            if show_progress: # Only show this tqdm if explicitly asked
                iterable_chunks = tqdm(
                    iterable_chunks,
                    total=len(chunks),
                    desc="Processing chunks", 
                    leave=False,  # Cleans up the bar after completion
                    position=1    # For nested progress bars, assumes outer is 0
                )

            for future in iterable_chunks:
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_scores = future.result()
                    all_scores.extend(chunk_scores)
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx} during parallel scoring: {e}")
        
        return all_scores
    
    def _score_item_batch_optimized(
        self, 
        user_idx: int, 
        item_ids: List[str],
        batch_size: int = 256
    ) -> List[Tuple[str, float]]:
        """
        Optimized batch scoring with sub-batching for GPU efficiency.
        This method processes a chunk of items in smaller GPU batches.
        """
        all_scores = []
        
        # Process in sub-batches for GPU
        for i in range(0, len(item_ids), batch_size):
            sub_batch = item_ids[i:i + batch_size]
            scores = self._score_item_batch(user_idx, sub_batch)
            all_scores.extend(scores)
        
        return all_scores


class ParallelScoringDataset(torch.utils.data.Dataset):
    """Dataset for parallel scoring of items"""
    
    def __init__(
        self,
        user_idx: int,
        item_ids: List[str],
        recommender: 'Recommender',
        batch_feature_loading: bool = True
    ):
        self.user_idx = user_idx
        self.item_ids = item_ids
        self.recommender = recommender
        self.batch_feature_loading = batch_feature_loading
        
        # Pre-compute item indices
        self.valid_items = []
        self.item_indices = []
        
        for item_id in item_ids:
            try:
                item_idx = recommender.dataset.item_encoder.transform([item_id])[0]
                self.valid_items.append(item_id)
                self.item_indices.append(item_idx)
            except:
                continue
    
    def __len__(self):
        return len(self.valid_items)
    
    def __getitem__(self, idx):
        item_id = self.valid_items[idx]
        item_idx = self.item_indices[idx]
        
        # Get features (will use cache if available)
        features = self.recommender._get_item_features(item_id)
        
        if features is None:
            # Return placeholder
            features = self._get_placeholder_features()
        
        return {
            'item_id': item_id,
            'item_idx': item_idx,
            'user_idx': self.user_idx,
            **features
        }
    
    def _get_placeholder_features(self):
        """Get placeholder features with correct dimensions"""
        # Implementation similar to what's in _get_item_features
        return {
            'image': torch.zeros(3, 224, 224),
            'text_input_ids': torch.zeros(128, dtype=torch.long),
            'text_attention_mask': torch.zeros(128, dtype=torch.long),
            'numerical_features': torch.zeros(7)
        }
    
    def preload_features_parallel(
        self,
        item_ids: List[str],
        num_workers: int = 4,
        chunk_size: int = 1000
    ) -> None:
        """
        Pre-load features for items in parallel to warm up caches.
        This is especially useful before evaluation.
        """
        # Split items into chunks
        chunks = [item_ids[i:i + chunk_size] for i in range(0, len(item_ids), chunk_size)]
        
        def load_chunk_features(chunk_items: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
            """Load features for a chunk of items"""
            chunk_features = {}
            
            # Use the batch feature loader
            features_list = self._get_batch_features(chunk_items)
            
            for item_id, features in zip(chunk_items, features_list):
                if features is not None:
                    chunk_features[item_id] = features
            
            return chunk_features
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(load_chunk_features, chunk) for chunk in chunks]
            
            with tqdm(total=len(chunks), desc="Pre-loading features") as pbar:
                for future in as_completed(futures):
                    try:
                        chunk_features = future.result()
                        # Update L1 cache
                        self.item_features_cache.update(chunk_features)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error loading chunk: {e}")
                        pbar.update(1)
        
        print(f"Pre-loaded features for {len(self.item_features_cache)} items")

    def _score_items_with_dataloader(
        self,
        user_idx: int,
        items: List[str],
        batch_size: int = 256,
        num_workers: int = 4
    ) -> List[Tuple[str, float]]:
        """Score items using PyTorch DataLoader for true parallel processing"""
        
        # Create dataset
        dataset = ParallelScoringDataset(user_idx, items, self)
        
        # Create DataLoader with multiple workers
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        all_scores = []
        
        # Process batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Scoring batches"):
                # Move batch to device
                batch_user_idx = batch['user_idx'].to(self.device)
                batch_item_idx = batch['item_idx'].to(self.device)
                batch_images = batch['image'].to(self.device)
                batch_text_ids = batch['text_input_ids'].to(self.device)
                batch_text_masks = batch['text_attention_mask'].to(self.device)
                batch_numerical = batch['numerical_features'].to(self.device)
                
                # Prepare model input
                model_input = {
                    'user_idx': batch_user_idx,
                    'item_idx': batch_item_idx,
                    'image': batch_images,
                    'text_input_ids': batch_text_ids,
                    'text_attention_mask': batch_text_masks,
                    'numerical_features': batch_numerical
                }
                
                # Add CLIP inputs if available
                if 'clip_text_input_ids' in batch:
                    model_input['clip_text_input_ids'] = batch['clip_text_input_ids'].to(self.device)
                    model_input['clip_text_attention_mask'] = batch['clip_text_attention_mask'].to(self.device)
                
                # Get predictions
                output = self.model(**model_input)
                if isinstance(output, tuple):
                    scores = output[0].squeeze()
                else:
                    scores = output.squeeze()
                
                # Collect results
                item_ids = batch['item_id']
                for item_id, score in zip(item_ids, scores.cpu().numpy()):
                    all_scores.append((item_id, float(score)))
        
        return all_scores


class ParallelScoringDataset(torch.utils.data.Dataset):
    """Dataset for parallel scoring of items"""
    
    def __init__(
        self,
        user_idx: int,
        item_ids: List[str],
        recommender: 'Recommender',
        batch_feature_loading: bool = True
    ):
        self.user_idx = user_idx
        self.item_ids = item_ids
        self.recommender = recommender
        self.batch_feature_loading = batch_feature_loading
        
        # Pre-compute item indices
        self.valid_items = []
        self.item_indices = []
        
        for item_id in item_ids:
            try:
                item_idx = recommender.dataset.item_encoder.transform([item_id])[0]
                self.valid_items.append(item_id)
                self.item_indices.append(item_idx)
            except:
                continue
    
    def __len__(self):
        return len(self.valid_items)
    
    def __getitem__(self, idx):
        item_id = self.valid_items[idx]
        item_idx = self.item_indices[idx]
        
        # Get features (will use cache if available)
        features = self.recommender._get_item_features(item_id)
        
        if features is None:
            # Return placeholder
            features = self._get_placeholder_features()
        
        return {
            'item_id': item_id,
            'item_idx': item_idx,
            'user_idx': self.user_idx,
            **features
        }
    
    def _get_placeholder_features(self):
        """Get placeholder features with correct dimensions"""
        return {
            'image': torch.zeros(3, 224, 224),
            'text_input_ids': torch.zeros(128, dtype=torch.long),
            'text_attention_mask': torch.zeros(128, dtype=torch.long),
            'numerical_features': torch.zeros(7)
        }

def _score_items_with_dataloader(
    self,
    user_idx: int,
    items: List[str],
    batch_size: int = 256,
    num_workers: int = 4
) -> List[Tuple[str, float]]:
    """Score items using PyTorch DataLoader for true parallel processing"""
    
    # Create dataset
    dataset = ParallelScoringDataset(user_idx, items, self)
    
    # Create DataLoader with multiple workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    all_scores = []
    
    # Process batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scoring batches"):
            # Move batch to device
            batch_user_idx = batch['user_idx'].to(self.device)
            batch_item_idx = batch['item_idx'].to(self.device)
            batch_images = batch['image'].to(self.device)
            batch_text_ids = batch['text_input_ids'].to(self.device)
            batch_text_masks = batch['text_attention_mask'].to(self.device)
            batch_numerical = batch['numerical_features'].to(self.device)
            
            # Prepare model input
            model_input = {
                'user_idx': batch_user_idx,
                'item_idx': batch_item_idx,
                'image': batch_images,
                'text_input_ids': batch_text_ids,
                'text_attention_mask': batch_text_masks,
                'numerical_features': batch_numerical
            }
            
            # Add CLIP inputs if available
            if 'clip_text_input_ids' in batch:
                model_input['clip_text_input_ids'] = batch['clip_text_input_ids'].to(self.device)
                model_input['clip_text_attention_mask'] = batch['clip_text_attention_mask'].to(self.device)
            
            # Get predictions
            output = self.model(**model_input)
            if isinstance(output, tuple):
                scores = output[0].squeeze()
            else:
                scores = output.squeeze()
            
            # Collect results
            item_ids = batch['item_id']
            for item_id, score in zip(item_ids, scores.cpu().numpy()):
                all_scores.append((item_id, float(score)))
    
    return all_scores