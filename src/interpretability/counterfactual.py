# src/interpretability/counterfactual.py

class CounterfactualExplainer:
    """Generate counterfactual explanations"""
    
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        
    def find_minimal_change(
        self,
        user_id: str,
        item_id: str,
        target_score: float = 0.8,
        max_iterations: int = 100
    ) -> Dict[str, any]:
        """Find minimal changes needed to achieve target recommendation score"""
        
        # Get current score
        current_score = self._get_score(user_id, item_id)
        
        if current_score >= target_score:
            return {
                'current_score': current_score,
                'target_score': target_score,
                'changes_needed': None,
                'message': 'Target score already achieved'
            }
        
        # Get current features
        item_info = self.dataset.item_info.loc[item_id].copy()
        original_features = {col: item_info[col] 
                           for col in self.dataset.numerical_feat_cols}
        
        # Optimization loop
        best_changes = {}
        best_score = current_score
        
        for iteration in range(max_iterations):
            # Try changing each feature
            for feature in self.dataset.numerical_feat_cols:
                # Increase feature value
                test_info = item_info.copy()
                test_info[feature] *= 1.1  # 10% increase
                
                # Get new score
                new_score = self._get_score_with_modified_item(
                    user_id, item_id, test_info
                )
                
                if new_score > best_score:
                    best_score = new_score
                    best_changes[feature] = test_info[feature]
                    
                if best_score >= target_score:
                    break
                    
            if best_score >= target_score:
                break
                
            # Update item_info with best changes
            for feature, value in best_changes.items():
                item_info[feature] = value
                
        return {
            'current_score': current_score,
            'achieved_score': best_score,
            'target_score': target_score,
            'changes_needed': best_changes,
            'original_features': original_features
        }