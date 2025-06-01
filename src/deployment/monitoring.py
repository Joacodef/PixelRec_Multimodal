# src/deployment/monitoring.py

import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Callable
import threading
import warnings

class ModelMonitor:
    """Real-time monitoring for recommendation models"""
    
    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: Dict[str, float] = None
    ):
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            "latency_p95": 1.0,  # 1 second
            "error_rate": 0.01,   # 1%
            "score_drift": 0.1    # 10% drift in average score
        }
        
        # Metrics storage
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.feature_stats = defaultdict(lambda: deque(maxlen=window_size))
        
        # Baseline statistics
        self.baseline_stats = {}
        self.alerts = []
        
        # Background monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def start_monitoring(self, check_interval: int = 60):
        """Start background monitoring"""
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,)
        )
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
    def _monitoring_loop(self, check_interval: int):
        """Background monitoring loop"""
        while not self.stop_monitoring.is_set():
            self.check_health()
            time.sleep(check_interval)
            
    def log_prediction(
        self,
        user_id: str,
        predictions: List[float],
        latency: float,
        features: Optional[Dict] = None,
        error: Optional[Exception] = None
    ):
        """Log a prediction event"""
        timestamp = time.time()
        
        # Log latency
        self.latencies.append(latency)
        
        # Log predictions
        self.predictions.extend(predictions)
        
        # Log errors
        self.errors.append(1 if error else 0)
        
        # Log feature statistics
        if features:
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    self.feature_stats[feature_name].append(value)
                    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate current metrics"""
        metrics = {}
        
        # Latency metrics
        if self.latencies:
            latencies = list(self.latencies)
            metrics["latency_mean"] = np.mean(latencies)
            metrics["latency_p50"] = np.percentile(latencies, 50)
            metrics["latency_p95"] = np.percentile(latencies, 95)
            metrics["latency_p99"] = np.percentile(latencies, 99)
            
        # Error rate
        if self.errors:
            metrics["error_rate"] = sum(self.errors) / len(self.errors)
            
        # Prediction statistics
        if self.predictions:
            predictions = list(self.predictions)
            metrics["prediction_mean"] = np.mean(predictions)
            metrics["prediction_std"] = np.std(predictions)
            
        # Feature drift
        for feature_name, values in self.feature_stats.items():
            if values and feature_name in self.baseline_stats:
                current_mean = np.mean(list(values))
                baseline_mean = self.baseline_stats[feature_name]["mean"]
                drift = abs(current_mean - baseline_mean) / (baseline_mean + 1e-6)
                metrics[f"drift_{feature_name}"] = drift
                
        return metrics
    
    def set_baseline(self, baseline_data: Dict[str, List[float]]):
        """Set baseline statistics for drift detection"""
        self.baseline_stats = {}
        
        for feature_name, values in baseline_data.items():
            self.baseline_stats[feature_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
            
    def check_health(self) -> Dict[str, any]:
        """Check model health and raise alerts"""
        metrics = self.calculate_metrics()
        health_status = {"healthy": True, "alerts": []}
        
        # Check latency
        if "latency_p95" in metrics:
            if metrics["latency_p95"] > self.alert_thresholds["latency_p95"]:
                alert = {
                    "type": "high_latency",
                    "message": f"P95 latency ({metrics['latency_p95']:.3f}s) exceeds threshold",
                    "severity": "warning"
                }
                health_status["alerts"].append(alert)
                self.alerts.append(alert)
                
        # Check error rate
        if "error_rate" in metrics:
            if metrics["error_rate"] > self.alert_thresholds["error_rate"]:
                alert = {
                    "type": "high_error_rate",
                    "message": f"Error rate ({metrics['error_rate']:.2%}) exceeds threshold",
                    "severity": "critical"
                }
                health_status["alerts"].append(alert)
                health_status["healthy"] = False
                self.alerts.append(alert)
                
        # Check prediction drift
        if "prediction_mean" in metrics and "prediction_mean" in self.baseline_stats:
            baseline_mean = self.baseline_stats["prediction_mean"]["mean"]
            current_mean = metrics["prediction_mean"]
            drift = abs(current_mean - baseline_mean) / (baseline_mean + 1e-6)
            
            if drift > self.alert_thresholds["score_drift"]:
                alert = {
                    "type": "score_drift",
                    "message": f"Prediction score drift ({drift:.2%}) detected",
                    "severity": "warning"
                }
                health_status["alerts"].append(alert)
                self.alerts.append(alert)
                
        return health_status