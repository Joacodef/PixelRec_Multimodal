# src/deployment/ab_testing.py

import random
from typing import Dict, List, Callable
import numpy as np

class ABTestFramework:
    """A/B testing framework for recommendation models"""
    
    def __init__(self, experiments_config: Dict[str, Dict]):
        """
        Initialize A/B test framework
        
        Args:
            experiments_config: Dict with experiment configurations
                {
                    "experiment_name": {
                        "models": {"variant_a": model_a, "variant_b": model_b},
                        "traffic_split": {"variant_a": 0.5, "variant_b": 0.5},
                        "metrics": ["ctr", "conversion_rate"],
                        "min_samples": 1000
                    }
                }
        """
        self.experiments = experiments_config
        self.results = {exp: {"variant_results": {}} for exp in experiments_config}
        
    def assign_variant(self, user_id: str, experiment: str) -> str:
        """Assign user to experiment variant"""
        if experiment not in self.experiments:
            raise ValueError(f"Experiment {experiment} not found")
            
        # Use consistent hashing for user assignment
        hash_value = int(hashlib.md5(f"{user_id}{experiment}".encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 100) / 100.0
        
        # Assign based on traffic split
        cumulative_prob = 0.0
        traffic_split = self.experiments[experiment]["traffic_split"]
        
        for variant, prob in traffic_split.items():
            cumulative_prob += prob
            if normalized_hash < cumulative_prob:
                return variant
                
        # Fallback (shouldn't reach here)
        return list(traffic_split.keys())[-1]
    
    def get_recommendations(
        self,
        user_id: str,
        experiment: str,
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, any]:
        """Get recommendations using assigned variant"""
        variant = self.assign_variant(user_id, experiment)
        model = self.experiments[experiment]["models"][variant]
        
        # Get recommendations from the assigned model
        recommendations = model.get_recommendations(user_id, top_k, **kwargs)
        
        # Log the interaction
        self._log_interaction(experiment, variant, user_id, recommendations)
        
        return {
            "recommendations": recommendations,
            "variant": variant,
            "experiment": experiment
        }
    
    def _log_interaction(
        self,
        experiment: str,
        variant: str,
        user_id: str,
        recommendations: List
    ):
        """Log interaction for analysis"""
        if variant not in self.results[experiment]["variant_results"]:
            self.results[experiment]["variant_results"][variant] = {
                "interactions": [],
                "metrics": {}
            }
            
        self.results[experiment]["variant_results"][variant]["interactions"].append({
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations
        })
    
    def calculate_significance(
        self,
        experiment: str,
        metric: str,
        confidence_level: float = 0.95
    ) -> Dict[str, any]:
        """Calculate statistical significance for experiment"""
        from scipy import stats
        
        if experiment not in self.results:
            raise ValueError(f"No results for experiment {experiment}")
            
        variant_results = self.results[experiment]["variant_results"]
        
        if len(variant_results) != 2:
            raise ValueError("Significance testing requires exactly 2 variants")
            
        variants = list(variant_results.keys())
        
        # Get metric values for each variant
        values_a = variant_results[variants[0]]["metrics"].get(metric, [])
        values_b = variant_results[variants[1]]["metrics"].get(metric, [])
        
        if not values_a or not values_b:
            return {"significant": False, "message": "Insufficient data"}
            
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        
        # Calculate effect size (Cohen's d)
        mean_a, mean_b = np.mean(values_a), np.mean(values_b)
        std_a, std_b = np.std(values_a), np.std(values_b)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
        
        return {
            "significant": p_value < (1 - confidence_level),
            "p_value": p_value,
            "effect_size": effect_size,
            "variant_a_mean": mean_a,
            "variant_b_mean": mean_b,
            "variant_a": variants[0],
            "variant_b": variants[1],
            "improvement": ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0
        }