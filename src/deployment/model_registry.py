# src/deployment/model_registry.py

import json
import hashlib
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, Optional, List

class ModelRegistry:
    """Manage model versions and deployments"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "active_model": None}
    
    def _save_metadata(self):
        """Save registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(
        self,
        model_path: str,
        config: Dict,
        metrics: Dict,
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """Register a new model version"""
        
        # Generate model ID
        timestamp = datetime.now().isoformat()
        config_str = json.dumps(config, sort_keys=True)
        model_id = hashlib.md5(f"{timestamp}{config_str}".encode()).hexdigest()[:8]
        
        # Create model directory
        model_dir = self.registry_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Copy model files
        shutil.copy2(model_path, model_dir / "model.pth")
        
        # Save config
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save metadata
        self.metadata["models"][model_id] = {
            "id": model_id,
            "timestamp": timestamp,
            "description": description,
            "config": config,
            "metrics": metrics,
            "tags": tags or [],
            "status": "registered"
        }
        
        self._save_metadata()
        return model_id
    
    def promote_model(self, model_id: str, environment: str = "production"):
        """Promote model to production"""
        if model_id not in self.metadata["models"]:
            raise ValueError(f"Model {model_id} not found")
            
        self.metadata["models"][model_id]["status"] = environment
        self.metadata["active_model"] = model_id
        self._save_metadata()
        
    def get_active_model(self) -> Optional[str]:
        """Get currently active model"""
        return self.metadata.get("active_model")
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """Compare multiple model versions"""
        comparison_data = []
        
        for model_id in model_ids:
            if model_id in self.metadata["models"]:
                model_info = self.metadata["models"][model_id]
                comparison_data.append({
                    "model_id": model_id,
                    "timestamp": model_info["timestamp"],
                    **model_info["metrics"],
                    "status": model_info["status"]
                })
                
        return pd.DataFrame(comparison_data)