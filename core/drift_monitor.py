# core/drift_monitor.py

import os
import time
import json
import warnings
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

try:
    import whylogs as why
    from whylogs.core.dataset_profile import DatasetProfile
    WHYLOGS_AVAILABLE = True
except ImportError:
    WHYLOGS_AVAILABLE = False

warnings.filterwarnings('ignore')


class DriftMonitor:
    """
    Advanced drift monitoring for continuous learning systems.
    
    Features:
    - Feature drift detection using statistical tests
    - Label/target drift monitoring
    - Model performance drift tracking
    - Real-time alerts and notifications
    - Historical drift analysis and reporting
    - Integration with MLflow for drift logging
    """
    
    def __init__(self, 
                 reference_window_size: int = 1000,
                 detection_window_size: int = 100,
                 drift_threshold: float = 0.1,
                 enable_whylogs: bool = True,
                 enable_evidently: bool = True):
        
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.drift_threshold = drift_threshold
        
        self.enable_whylogs = enable_whylogs and WHYLOGS_AVAILABLE
        self.enable_evidently = enable_evidently and EVIDENTLY_AVAILABLE
        
        # Data storage
        self.reference_data = defaultdict(lambda: deque(maxlen=reference_window_size))
        self.current_data = defaultdict(lambda: deque(maxlen=detection_window_size))
        self.drift_history = defaultdict(list)
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_performance = defaultdict(dict)
        
        # Drift alerts
        self.drift_alerts = []
        self.alert_callbacks = []
        
        # Setup logging
        os.makedirs("logs/drift", exist_ok=True)
        
        if not (self.enable_whylogs or self.enable_evidently):
            print("⚠️ No drift detection libraries available - using statistical fallback")
    
    def set_reference_data(self, 
                          symbol: str, 
                          features: pd.DataFrame, 
                          targets: Optional[pd.Series] = None):
        """Set reference data for drift detection"""
        self.reference_data[f"{symbol}_features"] = deque(
            [features.iloc[i:i+1] for i in range(len(features))],
            maxlen=self.reference_window_size
        )
        
        if targets is not None:
            self.reference_data[f"{symbol}_targets"] = deque(
                targets.tolist(),
                maxlen=self.reference_window_size
            )
    
    def add_current_data(self, 
                        symbol: str, 
                        features: pd.DataFrame, 
                        targets: Optional[pd.Series] = None,
                        predictions: Optional[np.ndarray] = None,
                        performance_metrics: Optional[Dict] = None):
        """Add current data point for drift monitoring"""
        
        # Store features
        for i in range(len(features)):
            self.current_data[f"{symbol}_features"].append(features.iloc[i:i+1])
        
        # Store targets
        if targets is not None:
            if isinstance(targets, pd.Series):
                self.current_data[f"{symbol}_targets"].extend(targets.tolist())
            else:
                self.current_data[f"{symbol}_targets"].extend(targets)
        
        # Store predictions
        if predictions is not None:
            self.current_data[f"{symbol}_predictions"].extend(predictions.tolist())
        
        # Store performance metrics
        if performance_metrics:
            self.performance_history[symbol].append({
                "timestamp": time.time(),
                **performance_metrics
            })
        
        # Check for drift if we have enough data
        if len(self.current_data[f"{symbol}_features"]) >= self.detection_window_size:
            self.detect_drift(symbol)
    
    def detect_drift(self, symbol: str) -> Dict[str, Any]:
        """Detect various types of drift"""
        drift_results = {
            "timestamp": time.time(),
            "symbol": symbol,
            "feature_drift": {},
            "target_drift": {},
            "prediction_drift": {},
            "performance_drift": {},
            "overall_drift_detected": False
        }
        
        # Feature drift detection
        if self.enable_evidently:
            feature_drift = self._detect_feature_drift_evidently(symbol)
            drift_results["feature_drift"] = feature_drift
        elif self.enable_whylogs:
            feature_drift = self._detect_feature_drift_whylogs(symbol)
            drift_results["feature_drift"] = feature_drift
        else:
            feature_drift = self._detect_feature_drift_statistical(symbol)
            drift_results["feature_drift"] = feature_drift
        
        # Target drift detection
        target_drift = self._detect_target_drift(symbol)
        drift_results["target_drift"] = target_drift
        
        # Performance drift detection
        performance_drift = self._detect_performance_drift(symbol)
        drift_results["performance_drift"] = performance_drift
        
        # Overall drift assessment
        drift_results["overall_drift_detected"] = (
            feature_drift.get("drift_detected", False) or
            target_drift.get("drift_detected", False) or
            performance_drift.get("drift_detected", False)
        )
        
        # Store results
        self.drift_history[symbol].append(drift_results)
        
        # Trigger alerts if drift detected
        if drift_results["overall_drift_detected"]:
            self._trigger_drift_alert(drift_results)
        
        # Log drift results
        self._log_drift_results(drift_results)
        
        return drift_results
    
    def _detect_feature_drift_evidently(self, symbol: str) -> Dict:
        """Detect feature drift using Evidently"""
        try:
            # Prepare reference and current data
            ref_features = pd.concat(list(self.reference_data[f"{symbol}_features"]))
            curr_features = pd.concat(list(self.current_data[f"{symbol}_features"]))
            
            # Create drift report
            report = Report(metrics=[
                DataDriftPreset(),
                ColumnDriftMetric(column_name=col) for col in ref_features.columns
            ])
            
            report.run(reference_data=ref_features, current_data=curr_features)
            
            # Extract drift results
            drift_results = report.as_dict()
            
            return {
                "method": "evidently",
                "drift_detected": drift_results["metrics"][0]["result"]["dataset_drift"],
                "drift_score": drift_results["metrics"][0]["result"]["drift_share"],
                "drifted_features": [
                    col for col in ref_features.columns 
                    if any(m["result"].get("drift_detected", False) 
                          for m in drift_results["metrics"] 
                          if m.get("metric") == "ColumnDriftMetric" and 
                             m.get("column_name") == col)
                ],
                "detailed_results": drift_results
            }
            
        except Exception as e:
            print(f"⚠️ Evidently drift detection failed: {e}")
            return self._detect_feature_drift_statistical(symbol)
    
    def _detect_feature_drift_whylogs(self, symbol: str) -> Dict:
        """Detect feature drift using whylogs"""
        try:
            # Prepare data
            ref_features = pd.concat(list(self.reference_data[f"{symbol}_features"]))
            curr_features = pd.concat(list(self.current_data[f"{symbol}_features"]))
            
            # Create profiles
            ref_profile = why.log(ref_features).profile().view()
            curr_profile = why.log(curr_features).profile().view()
            
            # Compare profiles (simplified drift detection)
            drift_detected = False
            drifted_features = []
            
            for col in ref_features.columns:
                try:
                    ref_summary = ref_profile.get_column(col)
                    curr_summary = curr_profile.get_column(col)
                    
                    # Simple statistical comparison
                    if ref_summary and curr_summary:
                        ref_mean = ref_summary.get_metric("mean")
                        curr_mean = curr_summary.get_metric("mean")
                        ref_std = ref_summary.get_metric("stddev")
                        
                        if ref_mean and curr_mean and ref_std:
                            z_score = abs(curr_mean.value - ref_mean.value) / max(ref_std.value, 1e-8)
                            if z_score > 2.0:  # Simple threshold
                                drift_detected = True
                                drifted_features.append(col)
                except Exception:
                    continue
            
            return {
                "method": "whylogs",
                "drift_detected": drift_detected,
                "drift_score": len(drifted_features) / len(ref_features.columns),
                "drifted_features": drifted_features
            }
            
        except Exception as e:
            print(f"⚠️ WhyLogs drift detection failed: {e}")
            return self._detect_feature_drift_statistical(symbol)
    
    def _detect_feature_drift_statistical(self, symbol: str) -> Dict:
        """Statistical drift detection fallback"""
        try:
            ref_features = pd.concat(list(self.reference_data[f"{symbol}_features"]))
            curr_features = pd.concat(list(self.current_data[f"{symbol}_features"]))
            
            drift_detected = False
            drifted_features = []
            drift_scores = {}
            
            for col in ref_features.columns:
                if col in curr_features.columns:
                    # Kolmogorov-Smirnov test for distribution drift
                    from scipy.stats import ks_2samp
                    
                    ref_values = ref_features[col].dropna()
                    curr_values = curr_features[col].dropna()
                    
                    if len(ref_values) > 0 and len(curr_values) > 0:
                        ks_stat, p_value = ks_2samp(ref_values, curr_values)
                        drift_scores[col] = {"ks_statistic": ks_stat, "p_value": p_value}
                        
                        if p_value < 0.05:  # Significant drift
                            drift_detected = True
                            drifted_features.append(col)
            
            return {
                "method": "statistical",
                "drift_detected": drift_detected,
                "drift_score": len(drifted_features) / len(ref_features.columns),
                "drifted_features": drifted_features,
                "detailed_scores": drift_scores
            }
            
        except Exception as e:
            print(f"⚠️ Statistical drift detection failed: {e}")
            return {"method": "fallback", "drift_detected": False}
    
    def _detect_target_drift(self, symbol: str) -> Dict:
        """Detect target/label drift"""
        try:
            ref_targets = list(self.reference_data[f"{symbol}_targets"])
            curr_targets = list(self.current_data[f"{symbol}_targets"])
            
            if len(ref_targets) > 0 and len(curr_targets) > 0:
                from scipy.stats import ks_2samp
                ks_stat, p_value = ks_2samp(ref_targets, curr_targets)
                
                return {
                    "drift_detected": p_value < 0.05,
                    "ks_statistic": ks_stat,
                    "p_value": p_value,
                    "reference_mean": np.mean(ref_targets),
                    "current_mean": np.mean(curr_targets)
                }
            
        except Exception as e:
            print(f"⚠️ Target drift detection failed: {e}")
        
        return {"drift_detected": False}
    
    def _detect_performance_drift(self, symbol: str) -> Dict:
        """Detect model performance drift"""
        try:
            recent_performance = list(self.performance_history[symbol])[-50:]  # Last 50 data points
            
            if len(recent_performance) < 10:
                return {"drift_detected": False, "reason": "insufficient_data"}
            
            # Calculate recent performance metrics
            recent_metrics = {}
            for metric in ["win_rate", "avg_return", "sharpe_ratio"]:
                values = [p.get(metric, 0) for p in recent_performance if metric in p]
                if values:
                    recent_metrics[metric] = np.mean(values)
            
            # Compare with baseline
            if symbol in self.baseline_performance and recent_metrics:
                drift_detected = False
                performance_changes = {}
                
                for metric, recent_value in recent_metrics.items():
                    baseline_value = self.baseline_performance[symbol].get(metric, recent_value)
                    change = abs(recent_value - baseline_value) / max(abs(baseline_value), 1e-8)
                    performance_changes[metric] = {
                        "baseline": baseline_value,
                        "recent": recent_value,
                        "change_pct": change * 100
                    }
                    
                    if change > 0.2:  # 20% performance change threshold
                        drift_detected = True
                
                return {
                    "drift_detected": drift_detected,
                    "performance_changes": performance_changes
                }
            
        except Exception as e:
            print(f"⚠️ Performance drift detection failed: {e}")
        
        return {"drift_detected": False}
    
    def set_baseline_performance(self, symbol: str, metrics: Dict[str, float]):
        """Set baseline performance metrics"""
        self.baseline_performance[symbol] = metrics
    
    def _trigger_drift_alert(self, drift_results: Dict):
        """Trigger drift alert"""
        alert = {
            "timestamp": drift_results["timestamp"],
            "symbol": drift_results["symbol"],
            "alert_type": "drift_detected",
            "severity": self._calculate_drift_severity(drift_results),
            "details": drift_results
        }
        
        self.drift_alerts.append(alert)
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"⚠️ Alert callback failed: {e}")
        
        print(f"🚨 DRIFT ALERT: {alert['symbol']} - Severity: {alert['severity']}")
    
    def _calculate_drift_severity(self, drift_results: Dict) -> str:
        """Calculate drift severity level"""
        feature_drift_score = drift_results["feature_drift"].get("drift_score", 0)
        target_drift = drift_results["target_drift"].get("drift_detected", False)
        performance_drift = drift_results["performance_drift"].get("drift_detected", False)
        
        if performance_drift:
            return "critical"
        elif target_drift or feature_drift_score > 0.5:
            return "high"
        elif feature_drift_score > 0.2:
            return "medium"
        else:
            return "low"
    
    def _log_drift_results(self, drift_results: Dict):
        """Log drift results to file"""
        log_file = f"logs/drift/drift_{drift_results['symbol']}.jsonl"
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(drift_results, default=str) + '\n')
        except Exception:
            pass  # Best effort logging
    
    def add_alert_callback(self, callback):
        """Add callback function for drift alerts"""
        self.alert_callbacks.append(callback)
    
    def get_drift_summary(self, symbol: str, days: int = 7) -> Dict:
        """Get drift summary for the last N days"""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_drift = [
            d for d in self.drift_history[symbol] 
            if d["timestamp"] > cutoff_time
        ]
        
        if not recent_drift:
            return {"no_data": True}
        
        return {
            "total_checks": len(recent_drift),
            "drift_detected_count": sum(1 for d in recent_drift if d["overall_drift_detected"]),
            "drift_rate": sum(1 for d in recent_drift if d["overall_drift_detected"]) / len(recent_drift),
            "most_recent_drift": recent_drift[-1] if recent_drift else None,
            "feature_drift_frequency": sum(
                1 for d in recent_drift 
                if d["feature_drift"].get("drift_detected", False)
            ) / len(recent_drift),
            "performance_drift_frequency": sum(
                1 for d in recent_drift 
                if d["performance_drift"].get("drift_detected", False)
            ) / len(recent_drift)
        }
    
    def should_retrain(self, symbol: str) -> Tuple[bool, str]:
        """Determine if model should be retrained based on drift"""
        recent_summary = self.get_drift_summary(symbol, days=1)
        
        if recent_summary.get("no_data"):
            return False, "insufficient_data"
        
        # Retrain if high drift rate or recent critical drift
        if recent_summary["drift_rate"] > 0.7:
            return True, "high_drift_rate"
        
        most_recent = recent_summary.get("most_recent_drift")
        if most_recent and self._calculate_drift_severity(most_recent) == "critical":
            return True, "critical_drift_detected"
        
        if recent_summary["performance_drift_frequency"] > 0.5:
            return True, "performance_degradation"
        
        return False, "no_significant_drift"


# Global drift monitor instance
drift_monitor = DriftMonitor()