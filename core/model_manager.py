# core/model_manager.py

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import ML components with fallback handling
try:
    from core.ml_tracker import ml_tracker
except ImportError:
    ml_tracker = None

try:
    from core.drift_monitor import drift_monitor
except ImportError:
    drift_monitor = None

try:
    from core.online_learner import online_learner
except ImportError:
    online_learner = None


class ModelManager:
    """
    Automated model management system for continuous learning.
    
    Features:
    - Automated retraining triggers
    - Champion/challenger model promotion
    - Model versioning and rollback
    - Performance-based model selection
    - A/B testing framework
    """
    
    def __init__(self, 
                 retrain_threshold: float = 0.1,
                 min_samples_retrain: int = 100,
                 ab_test_duration_hours: int = 24,
                 significance_level: float = 0.05):
        
        self.retrain_threshold = retrain_threshold
        self.min_samples_retrain = min_samples_retrain
        self.ab_test_duration_hours = ab_test_duration_hours
        self.significance_level = significance_level
        
        # Model registry
        self.registered_models = defaultdict(dict)
        self.model_performance = defaultdict(lambda: defaultdict(list))
        self.champion_models = defaultdict(str)
        self.challenger_models = defaultdict(str)
        
        # A/B testing
        self.active_ab_tests = defaultdict(dict)
        self.ab_test_results = defaultdict(list)
        
        # Retraining schedule
        self.last_retrain = defaultdict(float)
        self.retrain_queue = []
        
        # Performance baselines
        self.performance_baselines = defaultdict(dict)
        
        os.makedirs("models/registry", exist_ok=True)
        os.makedirs("logs/model_management", exist_ok=True)
    
    def register_model(self, 
                      symbol: str, 
                      model_name: str, 
                      model_version: str,
                      model_type: str = "ml",
                      performance_metrics: Optional[Dict] = None) -> bool:
        """Register a new model version"""
        
        model_key = f"{symbol}_{model_name}"
        
        self.registered_models[model_key][model_version] = {
            "symbol": symbol,
            "model_name": model_name,
            "version": model_version,
            "model_type": model_type,
            "registration_time": time.time(),
            "performance_metrics": performance_metrics or {},
            "status": "registered"
        }
        
        # Log to MLflow
        try:
            ml_tracker.register_model(model_name, model_version)
        except Exception:
            pass
        
        self._log_model_event(symbol, "model_registered", {
            "model_name": model_name,
            "version": model_version,
            "performance": performance_metrics
        })
        
        return True
    
    def set_champion(self, symbol: str, model_name: str, model_version: str) -> bool:
        """Set champion model for a symbol"""
        
        model_key = f"{symbol}_{model_name}"
        
        if model_key not in self.registered_models or model_version not in self.registered_models[model_key]:
            return False
        
        # Update previous champion to staging
        if symbol in self.champion_models:
            old_champion = self.champion_models[symbol]
            try:
                ml_tracker.transition_model_stage(old_champion.split("_v")[0], 
                                                old_champion.split("_v")[1], "Staging")
            except Exception:
                pass
        
        # Set new champion
        self.champion_models[symbol] = f"{model_name}_v{model_version}"
        self.registered_models[model_key][model_version]["status"] = "champion"
        
        # Update in MLflow
        try:
            ml_tracker.transition_model_stage(model_name, model_version, "Production")
        except Exception:
            pass
        
        self._log_model_event(symbol, "champion_promoted", {
            "model_name": model_name,
            "version": model_version
        })
        
        return True
    
    def set_challenger(self, symbol: str, model_name: str, model_version: str) -> bool:
        """Set challenger model for A/B testing"""
        
        model_key = f"{symbol}_{model_name}"
        
        if model_key not in self.registered_models or model_version not in self.registered_models[model_key]:
            return False
        
        self.challenger_models[symbol] = f"{model_name}_v{model_version}"
        self.registered_models[model_key][model_version]["status"] = "challenger"
        
        self._log_model_event(symbol, "challenger_set", {
            "model_name": model_name,
            "version": model_version
        })
        
        return True
    
    def start_ab_test(self, 
                     symbol: str, 
                     champion_model: str, 
                     challenger_model: str,
                     traffic_split: float = 0.5) -> str:
        """Start A/B test between champion and challenger"""
        
        test_id = f"ab_test_{symbol}_{int(time.time())}"
        
        self.active_ab_tests[symbol] = {
            "test_id": test_id,
            "champion": champion_model,
            "challenger": challenger_model,
            "traffic_split": traffic_split,
            "start_time": time.time(),
            "end_time": time.time() + (self.ab_test_duration_hours * 3600),
            "champion_results": [],
            "challenger_results": [],
            "status": "active"
        }
        
        self._log_model_event(symbol, "ab_test_started", {
            "test_id": test_id,
            "champion": champion_model,
            "challenger": challenger_model,
            "traffic_split": traffic_split
        })
        
        return test_id
    
    def record_ab_result(self, 
                        symbol: str, 
                        model_variant: str, 
                        performance_metric: float,
                        metadata: Optional[Dict] = None) -> None:
        """Record A/B test result"""
        
        if symbol not in self.active_ab_tests:
            return
        
        test = self.active_ab_tests[symbol]
        
        result = {
            "timestamp": time.time(),
            "performance": performance_metric,
            "metadata": metadata or {}
        }
        
        if model_variant == "champion":
            test["champion_results"].append(result)
        elif model_variant == "challenger":
            test["challenger_results"].append(result)
    
    def evaluate_ab_test(self, symbol: str) -> Optional[Dict]:
        """Evaluate A/B test and determine winner"""
        
        if symbol not in self.active_ab_tests:
            return None
        
        test = self.active_ab_tests[symbol]
        
        # Check if test is complete
        if time.time() < test["end_time"] and len(test["champion_results"]) < 30:
            return {"status": "ongoing", "message": "Test still running"}
        
        champion_performance = [r["performance"] for r in test["champion_results"]]
        challenger_performance = [r["performance"] for r in test["challenger_results"]]
        
        if len(champion_performance) < 10 or len(challenger_performance) < 10:
            return {"status": "insufficient_data", "message": "Not enough data for evaluation"}
        
        # Statistical significance test
        try:
            from scipy.stats import ttest_ind
            
            t_stat, p_value = ttest_ind(challenger_performance, champion_performance)
            
            champion_mean = np.mean(champion_performance)
            challenger_mean = np.mean(challenger_performance)
            
            improvement = (challenger_mean - champion_mean) / abs(champion_mean) if champion_mean != 0 else 0
            
            result = {
                "status": "completed",
                "champion_performance": {
                    "mean": champion_mean,
                    "std": np.std(champion_performance),
                    "count": len(champion_performance)
                },
                "challenger_performance": {
                    "mean": challenger_mean,
                    "std": np.std(challenger_performance),
                    "count": len(challenger_performance)
                },
                "statistical_significance": {
                    "p_value": p_value,
                    "significant": p_value < self.significance_level,
                    "improvement_pct": improvement * 100
                },
                "winner": "challenger" if (p_value < self.significance_level and challenger_mean > champion_mean) else "champion"
            }
            
            # Store result
            self.ab_test_results[symbol].append(result)
            test["status"] = "completed"
            test["result"] = result
            
            self._log_model_event(symbol, "ab_test_completed", result)
            
            return result
            
        except Exception as e:
            return {"status": "error", "message": f"Evaluation failed: {e}"}
    
    def should_retrain(self, symbol: str) -> Tuple[bool, str]:
        """Determine if model should be retrained"""
        
        # Check drift monitoring
        if drift_monitor:
            should_retrain_drift, drift_reason = drift_monitor.should_retrain(symbol)
            if should_retrain_drift:
                return True, f"drift_detected: {drift_reason}"
        
        # Check performance degradation
        if symbol in self.performance_baselines:
            recent_performance = self.get_recent_performance(symbol, days=7)
            baseline_performance = self.performance_baselines[symbol]
            
            for metric, baseline_value in baseline_performance.items():
                if metric in recent_performance:
                    current_value = recent_performance[metric]
                    degradation = (baseline_value - current_value) / abs(baseline_value)
                    
                    if degradation > self.retrain_threshold:
                        return True, f"performance_degradation: {metric} dropped {degradation:.2%}"
        
        # Check minimum samples since last retrain
        last_retrain_time = self.last_retrain.get(symbol, 0)
        if time.time() - last_retrain_time > 86400:  # 24 hours
            sample_count = self._get_sample_count_since(symbol, last_retrain_time)
            if sample_count >= self.min_samples_retrain:
                return True, f"scheduled_retrain: {sample_count} new samples"
        
        return False, "no_retrain_needed"
    
    def trigger_retrain(self, 
                       symbol: str, 
                       reason: str,
                       model_type: str = "ensemble") -> bool:
        """Trigger model retraining"""
        
        try:
            # Add to retrain queue
            retrain_task = {
                "symbol": symbol,
                "reason": reason,
                "model_type": model_type,
                "timestamp": time.time(),
                "status": "queued"
            }
            
            self.retrain_queue.append(retrain_task)
            
            self._log_model_event(symbol, "retrain_triggered", {
                "reason": reason,
                "model_type": model_type
            })
            
            return True
            
        except Exception as e:
            self._log_model_event(symbol, "retrain_failed", {
                "reason": reason,
                "error": str(e)
            })
            return False
    
    def get_recent_performance(self, symbol: str, days: int = 7) -> Dict[str, float]:
        """Get recent performance metrics for a symbol"""
        
        cutoff_time = time.time() - (days * 24 * 3600)
        
        # Get from online learner
        summary = online_learner.get_learning_summary(symbol) if online_learner else {}
        
        # Get from drift monitor
        drift_summary = drift_monitor.get_drift_summary(symbol, days) if drift_monitor else {}
        
        return {
            "model_mae": summary.get("models", {}).get("adaptive", {}).get("recent_mae", 0.0),
            "drift_rate": drift_summary.get("drift_rate", 0.0),
            "sample_count": summary.get("models", {}).get("adaptive", {}).get("count", 0)
        }
    
    def set_performance_baseline(self, symbol: str, metrics: Dict[str, float]):
        """Set performance baseline for comparison"""
        self.performance_baselines[symbol] = {
            **metrics,
            "timestamp": time.time()
        }
    
    def get_model_status(self, symbol: str) -> Dict[str, Any]:
        """Get current model status for symbol"""
        
        champion = self.champion_models.get(symbol, "none")
        challenger = self.challenger_models.get(symbol, "none")
        
        ab_test = self.active_ab_tests.get(symbol, {})
        ab_status = ab_test.get("status", "none")
        
        should_retrain, retrain_reason = self.should_retrain(symbol)
        
        return {
            "symbol": symbol,
            "champion_model": champion,
            "challenger_model": challenger,
            "ab_test_status": ab_status,
            "should_retrain": should_retrain,
            "retrain_reason": retrain_reason,
            "last_retrain": self.last_retrain.get(symbol, 0),
            "recent_performance": self.get_recent_performance(symbol)
        }
    
    def _get_sample_count_since(self, symbol: str, timestamp: float) -> int:
        """Get number of samples since timestamp"""
        # This would typically query the feature store or database
        # For now, return a mock count
        return max(0, int((time.time() - timestamp) / 60))  # 1 sample per minute
    
    def _log_model_event(self, symbol: str, event_type: str, data: Dict):
        """Log model management event"""
        
        log_entry = {
            "timestamp": time.time(),
            "symbol": symbol,
            "event_type": event_type,
            "data": data
        }
        
        log_file = f"logs/model_management/{symbol}_model_events.jsonl"
        
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Best effort logging
        
        # Also log to MLflow
        try:
            ml_tracker.log_metrics({
                f"{symbol}_model_event": 1.0
            })
        except Exception:
            pass
    
    def run_periodic_checks(self) -> Dict[str, Any]:
        """Run periodic model management checks"""
        
        results = {
            "timestamp": time.time(),
            "checks_performed": [],
            "actions_taken": []
        }
        
        # Check all symbols for retraining needs
        all_symbols = set(list(self.champion_models.keys()) + list(drift_monitor.baseline_performance.keys()))
        
        for symbol in all_symbols:
            should_retrain, reason = self.should_retrain(symbol)
            
            if should_retrain:
                success = self.trigger_retrain(symbol, reason)
                results["actions_taken"].append({
                    "symbol": symbol,
                    "action": "retrain_triggered",
                    "reason": reason,
                    "success": success
                })
            
            results["checks_performed"].append({
                "symbol": symbol,
                "should_retrain": should_retrain,
                "reason": reason
            })
        
        # Evaluate active A/B tests
        for symbol in list(self.active_ab_tests.keys()):
            if self.active_ab_tests[symbol]["status"] == "active":
                evaluation = self.evaluate_ab_test(symbol)
                
                if evaluation and evaluation["status"] == "completed":
                    results["actions_taken"].append({
                        "symbol": symbol,
                        "action": "ab_test_evaluated",
                        "winner": evaluation["winner"]
                    })
                    
                    # Auto-promote winner if significant improvement
                    if evaluation["winner"] == "challenger" and evaluation["statistical_significance"]["significant"]:
                        challenger_model = self.active_ab_tests[symbol]["challenger"]
                        model_name, version = challenger_model.split("_v")
                        
                        if self.set_champion(symbol, model_name, version):
                            results["actions_taken"].append({
                                "symbol": symbol,
                                "action": "champion_promoted",
                                "new_champion": challenger_model
                            })
        
        return results


# Global model manager instance
model_manager = ModelManager()