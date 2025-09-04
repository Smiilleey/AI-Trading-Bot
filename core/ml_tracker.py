# core/ml_tracker.py

import os
import time
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLTracker:
    """
    MLflow-based model tracking and versioning for continuous learning.
    
    Features:
    - Experiment tracking with automatic run management
    - Model versioning and registry
    - Performance metrics logging
    - Artifact storage (models, features, predictions)
    - Champion/challenger model comparison
    """
    
    def __init__(self, 
                 tracking_uri: str = "file:./mlruns",
                 experiment_name: str = "forex_trading_ml",
                 enable_tracking: bool = True):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.enable_tracking = enable_tracking and MLFLOW_AVAILABLE
        
        if self.enable_tracking:
            self._setup_mlflow()
        else:
            print("⚠️ MLflow not available - using fallback logging")
            self._setup_fallback()
    
    def _setup_mlflow(self):
        """Initialize MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            print(f"✅ MLflow tracking initialized: {self.tracking_uri}")
            
        except Exception as e:
            print(f"⚠️ MLflow setup failed: {e}")
            self.enable_tracking = False
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback logging when MLflow is not available"""
        self.fallback_dir = "logs/ml_tracking"
        os.makedirs(self.fallback_dir, exist_ok=True)
        self.current_run_id = str(uuid.uuid4())
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None) -> str:
        """Start a new MLflow run"""
        if self.enable_tracking:
            run = mlflow.start_run(run_name=run_name, tags=tags or {})
            return run.info.run_id
        else:
            self.current_run_id = str(uuid.uuid4())
            run_data = {
                "run_id": self.current_run_id,
                "run_name": run_name,
                "start_time": time.time(),
                "tags": tags or {}
            }
            self._log_to_fallback("run_start", run_data)
            return self.current_run_id
    
    def end_run(self):
        """End the current MLflow run"""
        if self.enable_tracking:
            mlflow.end_run()
        else:
            run_data = {
                "run_id": self.current_run_id,
                "end_time": time.time()
            }
            self._log_to_fallback("run_end", run_data)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        if self.enable_tracking:
            mlflow.log_params(params)
        else:
            self._log_to_fallback("params", params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.enable_tracking:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        else:
            data = {"metrics": metrics, "step": step, "timestamp": time.time()}
            self._log_to_fallback("metrics", data)
    
    def log_model(self, model, model_name: str, signature=None, input_example=None):
        """Log model artifact"""
        if self.enable_tracking:
            try:
                # Try to log as sklearn model first
                mlflow.sklearn.log_model(
                    model, 
                    model_name,
                    signature=signature,
                    input_example=input_example
                )
                return True
            except Exception:
                try:
                    # Fallback to generic python model
                    mlflow.pyfunc.log_model(
                        model_name,
                        python_model=model
                    )
                    return True
                except Exception as e:
                    print(f"⚠️ Failed to log model {model_name}: {e}")
                    return False
        else:
            # Save model locally
            model_path = os.path.join(self.fallback_dir, f"{model_name}_{self.current_run_id}.joblib")
            try:
                import joblib
                joblib.dump(model, model_path)
                self._log_to_fallback("model", {"name": model_name, "path": model_path})
                return True
            except Exception as e:
                print(f"⚠️ Failed to save model locally: {e}")
                return False
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file"""
        if self.enable_tracking:
            mlflow.log_artifact(local_path, artifact_path)
        else:
            # Copy to fallback directory
            import shutil
            dest_dir = os.path.join(self.fallback_dir, self.current_run_id, artifact_path or "")
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(local_path))
            shutil.copy2(local_path, dest_path)
    
    def register_model(self, model_name: str, model_version: Optional[str] = None) -> bool:
        """Register model in MLflow Model Registry"""
        if self.enable_tracking:
            try:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
                mlflow.register_model(model_uri, model_name)
                return True
            except Exception as e:
                print(f"⚠️ Failed to register model: {e}")
                return False
        else:
            self._log_to_fallback("model_registry", {
                "model_name": model_name,
                "version": model_version,
                "timestamp": time.time()
            })
            return True
    
    def get_model_versions(self, model_name: str) -> List[Dict]:
        """Get all versions of a registered model"""
        if self.enable_tracking:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                versions = client.search_model_versions(f"name='{model_name}'")
                return [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "creation_timestamp": v.creation_timestamp
                    }
                    for v in versions
                ]
            except Exception:
                return []
        else:
            # Return empty list for fallback
            return []
    
    def transition_model_stage(self, model_name: str, version: str, stage: str) -> bool:
        """Transition model to different stage (Staging, Production, Archived)"""
        if self.enable_tracking:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage
                )
                return True
            except Exception as e:
                print(f"⚠️ Failed to transition model stage: {e}")
                return False
        else:
            self._log_to_fallback("model_stage_transition", {
                "model_name": model_name,
                "version": version,
                "stage": stage,
                "timestamp": time.time()
            })
            return True
    
    def log_prediction_batch(self, 
                           predictions: np.ndarray,
                           actuals: Optional[np.ndarray] = None,
                           features: Optional[pd.DataFrame] = None,
                           metadata: Optional[Dict] = None):
        """Log a batch of predictions for monitoring"""
        batch_data = {
            "timestamp": time.time(),
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "metadata": metadata or {}
        }
        
        if actuals is not None:
            batch_data["actuals"] = actuals.tolist() if isinstance(actuals, np.ndarray) else actuals
            
        if features is not None:
            batch_data["features_summary"] = {
                "columns": list(features.columns),
                "shape": features.shape,
                "dtypes": features.dtypes.to_dict()
            }
        
        if self.enable_tracking:
            # Log as artifact
            batch_file = f"predictions_batch_{int(time.time())}.json"
            with open(batch_file, 'w') as f:
                json.dump(batch_data, f)
            self.log_artifact(batch_file)
            os.remove(batch_file)  # Clean up temp file
        else:
            self._log_to_fallback("predictions", batch_data)
    
    def log_performance_metrics(self, 
                              symbol: str,
                              timeframe: str,
                              win_rate: float,
                              avg_return: float,
                              sharpe_ratio: float,
                              max_drawdown: float,
                              total_trades: int):
        """Log trading performance metrics"""
        metrics = {
            f"{symbol}_{timeframe}_win_rate": win_rate,
            f"{symbol}_{timeframe}_avg_return": avg_return,
            f"{symbol}_{timeframe}_sharpe_ratio": sharpe_ratio,
            f"{symbol}_{timeframe}_max_drawdown": max_drawdown,
            f"{symbol}_{timeframe}_total_trades": total_trades
        }
        self.log_metrics(metrics)
    
    def _log_to_fallback(self, log_type: str, data: Dict):
        """Log data to fallback file system"""
        log_file = os.path.join(self.fallback_dir, f"{log_type}.jsonl")
        log_entry = {
            "timestamp": time.time(),
            "run_id": self.current_run_id,
            "data": data
        }
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Best effort logging
    
    def get_champion_model(self, model_name: str) -> Optional[str]:
        """Get the current champion (Production) model version"""
        versions = self.get_model_versions(model_name)
        for version in versions:
            if version["stage"] == "Production":
                return version["version"]
        return None
    
    def compare_models(self, model_name: str, metric_name: str = "win_rate") -> Dict:
        """Compare performance of different model versions"""
        if not self.enable_tracking:
            return {}
            
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            versions = self.get_model_versions(model_name)
            comparison = {}
            
            for version in versions:
                run = client.get_run(version["run_id"])
                metrics = run.data.metrics
                comparison[version["version"]] = {
                    "stage": version["stage"],
                    "metric_value": metrics.get(metric_name, 0.0),
                    "creation_time": version["creation_timestamp"]
                }
            
            return comparison
        except Exception:
            return {}


# Global tracker instance
ml_tracker = MLTracker()