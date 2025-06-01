# src/interpretability/explain.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from captum.attr import IntegratedGradients, GradientShap, Saliency

class RecommendationExplainer:
    """Explain recommendations using various interpretability methods"""
    
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.model.eval()
        
    def get_feature_importance(
        self,
        user_id: str,
        item_id: str,
        method: str = 'integrated_gradients'
    ) -> Dict[str, np.ndarray]:
        """Get feature importance for a recommendation"""
        
        # Prepare input
        user_idx = self.dataset.user_encoder.transform([user_id])[0]
        item_features = self.dataset._get_item_features(item_id)
        
        # Create input tensors
        inputs = {
            'user_idx': torch.tensor([user_idx]),
            'item_idx': torch.tensor([self.dataset.item_encoder.transform([item_id])[0]]),
            'image': item_features['image'].unsqueeze(0),
            'text_input_ids': item_features['text_ids'].unsqueeze(0),
            'text_attention_mask': item_features['text_mask'].unsqueeze(0),
            'numerical_features': item_features['numerical'].unsqueeze(0)
        }
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get baseline (average features)
        baseline = self._get_baseline_inputs(inputs)
        
        # Apply attribution method
        if method == 'integrated_gradients':
            attributions = self._integrated_gradients(inputs, baseline)
        elif method == 'gradient_shap':
            attributions = self._gradient_shap(inputs, baseline)
        else:
            attributions = self._saliency(inputs)
            
        return attributions
    
    def _integrated_gradients(self, inputs: Dict, baseline: Dict) -> Dict[str, np.ndarray]:
        """Apply Integrated Gradients"""
        ig = IntegratedGradients(self._forward_func)
        
        attributions = {}
        
        # Get attributions for each input modality
        for key in ['image', 'numerical_features']:
            if key in inputs:
                attr = ig.attribute(
                    inputs[key],
                    baselines=baseline[key],
                    additional_forward_args=(inputs, key),
                    n_steps=50
                )
                attributions[key] = attr.cpu().numpy()
                
        return attributions
    
    def _forward_func(self, input_tensor, all_inputs, input_key):
        """Forward function for attribution methods"""
        # Replace the specific input with the tensor being analyzed
        forward_inputs = all_inputs.copy()
        forward_inputs[input_key] = input_tensor
        
        return self.model(**forward_inputs)
    
    def visualize_attention_weights(
        self,
        user_id: str,
        item_id: str,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Visualize attention weights from the model"""
        
        # Get model attention weights
        with torch.no_grad():
            # Prepare inputs
            user_idx = self.dataset.user_encoder.transform([user_id])[0]
            item_features = self.dataset._get_item_features(item_id)
            
            # Forward pass with attention capture
            user_emb = self.model.user_embedding(torch.tensor([user_idx]))
            item_emb = self.model.item_embedding(
                torch.tensor([self.dataset.item_encoder.transform([item_id])[0]])
            )
            
            # Get all embeddings
            vision_emb = self.model.vision_projection(
                self.model._get_vision_features(item_features['image'].unsqueeze(0))
            )
            language_emb = self.model.language_projection(
                self.model._get_language_features(
                    item_features['text_ids'].unsqueeze(0),
                    item_features['text_mask'].unsqueeze(0)
                )
            )
            numerical_emb = self.model.numerical_projection(
                item_features['numerical'].unsqueeze(0)
            )
            
            # Stack features
            features = torch.stack(
                [user_emb, item_emb, vision_emb, language_emb, numerical_emb],
                dim=1
            )
            
            # Get attention weights
            _, attention_weights = self.model.attention(
                features, features, features, need_weights=True
            )
            
        # Visualize
        attention_matrix = attention_weights[0].cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        feature_names = ['User', 'Item', 'Vision', 'Language', 'Numerical']
        
        sns.heatmap(
            attention_matrix,
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=True,
            fmt='.3f',
            cmap='Blues'
        )
        plt.title(f'Attention Weights for User {user_id} - Item {item_id}')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        return {'attention_weights': attention_matrix}
    
    def generate_textual_explanation(
        self,
        user_id: str,
        recommendations: List[Tuple[str, float]],
        top_k_features: int = 3
    ) -> List[Dict[str, any]]:
        """Generate natural language explanations for recommendations"""
        
        explanations = []
        
        for item_id, score in recommendations:
            # Get feature importance
            importance = self.get_feature_importance(user_id, item_id)
            
            # Get item info
            item_info = self.dataset.item_info.loc[item_id]
            
            # Identify top contributing factors
            factors = []
            
            # Check numerical features importance
            if 'numerical_features' in importance:
                num_importance = importance['numerical_features'].squeeze()
                top_indices = np.argsort(np.abs(num_importance))[-top_k_features:]
                
                for idx in top_indices:
                    feature_name = self.dataset.numerical_feat_cols[idx]
                    feature_value = item_info[feature_name]
                    importance_score = num_importance[idx]
                    
                    if importance_score > 0:
                        factors.append(f"high {feature_name.replace('_', ' ')} ({feature_value})")
                    else:
                        factors.append(f"moderate {feature_name.replace('_', ' ')}")
            
            # Create explanation
            explanation = {
                'item_id': item_id,
                'score': score,
                'title': item_info.get('title', 'Unknown'),
                'explanation': f"Recommended because of {', '.join(factors[:2])} and {'its' if len(factors) > 2 else 'the'} relevance to your interests.",
                'contributing_factors': factors
            }
            
            explanations.append(explanation)
            
        return explanations


