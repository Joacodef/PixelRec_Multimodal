# src/deployment/feature_store.py

class FeatureStore:
    """Centralized feature storage and retrieval"""
    
    def __init__(self, storage_backend: str = "redis"):
        self.storage_backend = storage_backend
        self.cache = {}
        
        if storage_backend == "redis":
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            
    def store_features(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, any],
        ttl: Optional[int] = 3600
    ):
        """Store features for an entity"""
        key = f"{entity_type}:{entity_id}"
        
        if self.storage_backend == "redis":
            serialized = json.dumps(features)
            self.redis_client.setex(key, ttl, serialized)
        else:
            self.cache[key] = features
            
    def get_features(
        self,
        entity_type: str,
        entity_id: str
    ) -> Optional[Dict[str, any]]:
        """Retrieve features for an entity"""
        key = f"{entity_type}:{entity_id}"
        
        if self.storage_backend == "redis":
            serialized = self.redis_client.get(key)
            if serialized:
                return json.loads(serialized)
        else:
            return self.cache.get(key)
            
        return None
    
    def batch_get_features(
        self,
        entity_type: str,
        entity_ids: List[str]
    ) -> Dict[str, Dict[str, any]]:
        """Retrieve features for multiple entities"""
        results = {}
        
        if self.storage_backend == "redis":
            pipe = self.redis_client.pipeline()
            keys = [f"{entity_type}:{entity_id}" for entity_id in entity_ids]
            
            for key in keys:
                pipe.get(key)
                
            responses = pipe.execute()
            
            for entity_id, response in zip(entity_ids, responses):
                if response:
                    results[entity_id] = json.loads(response)
        else:
            for entity_id in entity_ids:
                key = f"{entity_type}:{entity_id}"
                if key in self.cache:
                    results[entity_id] = self.cache[key]
                    
        return results