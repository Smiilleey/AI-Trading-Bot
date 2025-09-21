# core/online_learner.py

import numpy as np
import pandas as pd
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime
import warnings

try:
    from river import bandit, linear_model, ensemble, metrics, preprocessing
    from river.utils import Rolling
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

warnings.filterwarnings('ignore')


class OnlineLearner:
    """
    Online learning system for continuous model adaptation.
    
    Features:
    - Multi-armed bandits for parameter optimization
    - Online regression/classification models
    - Contextual bandits for feature-aware decisions
    - Real-time model updates with forgetting
    - Performance tracking and adaptation
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 window_size: int = 1000,
                 enable_bandits: bool = True,
                 enable_river: bool = True):
        
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.enable_bandits = enable_bandits
        self.enable_river = enable_river and RIVER_AVAILABLE
        
        # Online models per symbol
        self.models = defaultdict(dict)
        self.bandits = defaultdict(dict)
        self.performance_metrics = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size)))
        
        # Parameter optimization
        self.parameter_bandits = defaultdict(dict)
        self.parameter_performance = defaultdict(lambda: defaultdict(list))
        
        # Context tracking
        self.context_history = defaultdict(lambda: deque(maxlen=window_size))
        self.reward_history = defaultdict(lambda: deque(maxlen=window_size))
        
        if not self.enable_river:
            print("⚠️ River not available - using simplified online learning")
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize online learning models"""
        if self.enable_river:
            self.default_model_config = {
                "linear": linear_model.LinearRegression(),
                "adaptive": linear_model.ALMAClassifier(),
                "passive_aggressive": linear_model.PAClassifier(),
                "ensemble": ensemble.AdaptiveRandomForestRegressor(n_models=5)
            }
        else:
            # Fallback models
            self.default_model_config = {
                "simple_average": None,  # Will implement simple averaging
                "ewma": None  # Exponentially weighted moving average
            }
    
    def get_or_create_model(self, symbol: str, model_type: str = "linear"):
        """Get or create online model for symbol"""
        if symbol not in self.models or model_type not in self.models[symbol]:
            if self.enable_river and model_type in self.default_model_config:
                if model_type == "linear":
                    self.models[symbol][model_type] = linear_model.LinearRegression()
                elif model_type == "adaptive":
                    self.models[symbol][model_type] = linear_model.ALMAClassifier()
                elif model_type == "passive_aggressive":
                    self.models[symbol][model_type] = linear_model.PAClassifier()
                elif model_type == "ensemble":
                    self.models[symbol][model_type] = ensemble.AdaptiveRandomForestRegressor(n_models=5)
            else:
                # Fallback model
                self.models[symbol][model_type] = SimpleOnlineModel()
        
        return self.models[symbol][model_type]
    
    def update_model(self, 
                    symbol: str, 
                    features: Dict[str, float], 
                    target: float,
                    model_type: str = "linear") -> Dict[str, Any]:
        """Update online model with new data point"""
        
        model = self.get_or_create_model(symbol, model_type)
        
        if self.enable_river and hasattr(model, 'learn_one'):
            # River model
            try:
                # Make prediction before learning
                prediction = model.predict_one(features)
                
                # Update model
                model.learn_one(features, target)
                
                # Calculate error
                error = abs(prediction - target) if prediction is not None else 0
                
                result = {
                    "prediction": prediction,
                    "target": target,
                    "error": error,
                    "model_type": model_type,
                    "timestamp": time.time()
                }
                
                # Track performance
                self.performance_metrics[symbol][model_type].append(error)
                
                return result
                
            except Exception as e:
                print(f"⚠️ River model update failed: {e}")
                return self._update_fallback_model(symbol, features, target, model_type)
        else:
            return self._update_fallback_model(symbol, features, target, model_type)
    
    def _update_fallback_model(self, 
                              symbol: str, 
                              features: Dict[str, float], 
                              target: float,
                              model_type: str) -> Dict[str, Any]:
        """Update fallback model"""
        model = self.get_or_create_model(symbol, model_type)
        
        prediction = model.predict(features)
        model.update(features, target)
        
        error = abs(prediction - target)
        self.performance_metrics[symbol][model_type].append(error)
        
        return {
            "prediction": prediction,
            "target": target,
            "error": error,
            "model_type": model_type,
            "timestamp": time.time()
        }
    
    def predict(self, 
                symbol: str, 
                features: Dict[str, float], 
                model_type: str = "linear") -> float:
        """Make prediction using online model"""
        
        model = self.get_or_create_model(symbol, model_type)
        
        if self.enable_river and hasattr(model, 'predict_one'):
            try:
                return model.predict_one(features) or 0.0
            except Exception:
                pass
        
        # Fallback prediction
        return model.predict(features) if hasattr(model, 'predict') else 0.0
    
    def setup_parameter_bandit(self, 
                              symbol: str, 
                              parameter_name: str, 
                              parameter_values: List[Any],
                              bandit_type: str = "ucb") -> None:
        """Setup multi-armed bandit for parameter optimization"""
        
        if not self.enable_bandits:
            return
        
        if self.enable_river:
            if bandit_type == "ucb":
                self.parameter_bandits[symbol][parameter_name] = bandit.UCB(
                    reward_obj=metrics.Mean(),
                    confidence=1.96,
                    seed=42
                )
            elif bandit_type == "epsilon_greedy":
                self.parameter_bandits[symbol][parameter_name] = bandit.EpsilonGreedy(
                    reward_obj=metrics.Mean(),
                    epsilon=0.1,
                    seed=42
                )
            elif bandit_type == "thompson":
                self.parameter_bandits[symbol][parameter_name] = bandit.ThompsonSampling(
                    reward_obj=metrics.Mean(),
                    seed=42
                )
        else:
            # Fallback bandit
            self.parameter_bandits[symbol][parameter_name] = SimpleBandit(parameter_values)
        
        # Store parameter values
        self.parameter_bandits[symbol][f"{parameter_name}_values"] = parameter_values
    
    def select_parameter(self, symbol: str, parameter_name: str) -> Any:
        """Select parameter value using bandit"""
        
        if symbol not in self.parameter_bandits or parameter_name not in self.parameter_bandits[symbol]:
            return None
        
        bandit_model = self.parameter_bandits[symbol][parameter_name]
        parameter_values = self.parameter_bandits[symbol][f"{parameter_name}_values"]
        
        if self.enable_river and hasattr(bandit_model, 'pull'):
            try:
                # River bandit
                arm_index = bandit_model.pull(list(range(len(parameter_values))))
                return parameter_values[arm_index]
            except Exception:
                pass
        
        # Fallback selection
        if hasattr(bandit_model, 'select'):
            return bandit_model.select()
        
        # Random fallback
        return np.random.choice(parameter_values)
    
    def update_parameter_reward(self, 
                               symbol: str, 
                               parameter_name: str, 
                               parameter_value: Any, 
                               reward: float) -> None:
        """Update bandit with reward for selected parameter"""
        
        if symbol not in self.parameter_bandits or parameter_name not in self.parameter_bandits[symbol]:
            return
        
        bandit_model = self.parameter_bandits[symbol][parameter_name]
        parameter_values = self.parameter_bandits[symbol][f"{parameter_name}_values"]
        
        try:
            arm_index = parameter_values.index(parameter_value)
        except ValueError:
            return
        
        if self.enable_river and hasattr(bandit_model, 'update'):
            try:
                bandit_model.update(arm_index, reward)
            except Exception:
                pass
        
        # Track performance
        self.parameter_performance[symbol][parameter_name].append({
            "parameter_value": parameter_value,
            "reward": reward,
            "timestamp": time.time()
        })
    
    def get_best_parameters(self, symbol: str) -> Dict[str, Any]:
        """Get currently best performing parameters"""
        best_params = {}
        
        for param_name in self.parameter_performance[symbol]:
            recent_performance = self.parameter_performance[symbol][param_name][-100:]  # Last 100
            
            if recent_performance:
                # Group by parameter value and calculate average reward
                param_rewards = defaultdict(list)
                for perf in recent_performance:
                    param_rewards[perf["parameter_value"]].append(perf["reward"])
                
                # Find best parameter value
                best_param = max(param_rewards.keys(), 
                               key=lambda x: np.mean(param_rewards[x]))
                best_params[param_name] = {
                    "value": best_param,
                    "avg_reward": np.mean(param_rewards[best_param]),
                    "count": len(param_rewards[best_param])
                }
        
        return best_params
    
    def create_contextual_bandit(self, 
                                symbol: str, 
                                actions: List[str],
                                context_features: List[str]) -> None:
        """Create contextual bandit for action selection"""
        
        if not self.enable_bandits:
            return
        
        if self.enable_river:
            # Use linear contextual bandit
            self.bandits[symbol]["contextual"] = bandit.LinUCB(
                alpha=0.15,
                l2_penalty=0.1
            )
        else:
            # Fallback contextual bandit
            self.bandits[symbol]["contextual"] = SimpleContextualBandit(actions)
        
        self.bandits[symbol]["actions"] = actions
        self.bandits[symbol]["context_features"] = context_features
    
    def select_action(self, 
                     symbol: str, 
                     context: Dict[str, float]) -> str:
        """Select action using contextual bandit"""
        
        if symbol not in self.bandits or "contextual" not in self.bandits[symbol]:
            return "default"
        
        bandit_model = self.bandits[symbol]["contextual"]
        actions = self.bandits[symbol]["actions"]
        
        if self.enable_river and hasattr(bandit_model, 'pull'):
            try:
                action_index = bandit_model.pull(actions, context)
                return actions[action_index] if isinstance(action_index, int) else action_index
            except Exception:
                pass
        
        # Fallback action selection
        if hasattr(bandit_model, 'select_action'):
            return bandit_model.select_action(context)
        
        return np.random.choice(actions)
    
    def update_action_reward(self, 
                            symbol: str, 
                            action: str, 
                            context: Dict[str, float], 
                            reward: float) -> None:
        """Update contextual bandit with reward"""
        
        if symbol not in self.bandits or "contextual" not in self.bandits[symbol]:
            return
        
        bandit_model = self.bandits[symbol]["contextual"]
        
        if self.enable_river and hasattr(bandit_model, 'update'):
            try:
                bandit_model.update(action, context, reward)
            except Exception:
                pass
        
        # Store context and reward for analysis
        self.context_history[symbol].append({
            "context": context,
            "action": action,
            "reward": reward,
            "timestamp": time.time()
        })
    
    def get_model_performance(self, symbol: str, model_type: str = "linear") -> Dict[str, float]:
        """Get performance metrics for online model"""
        
        if symbol not in self.performance_metrics or model_type not in self.performance_metrics[symbol]:
            return {"mae": 0.0, "count": 0}
        
        errors = list(self.performance_metrics[symbol][model_type])
        
        if not errors:
            return {"mae": 0.0, "count": 0}
        
        return {
            "mae": np.mean(errors),
            "std": np.std(errors),
            "count": len(errors),
            "recent_mae": np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors)
        }
    
    def should_switch_model(self, symbol: str) -> Tuple[bool, str]:
        """Determine if we should switch to a different model type"""
        
        if symbol not in self.performance_metrics:
            return False, "no_data"
        
        model_performances = {}
        for model_type in self.performance_metrics[symbol]:
            perf = self.get_model_performance(symbol, model_type)
            if perf["count"] >= 10:  # Minimum samples
                model_performances[model_type] = perf["recent_mae"]
        
        if len(model_performances) < 2:
            return False, "insufficient_models"
        
        # Find best performing model
        best_model = min(model_performances.keys(), key=lambda x: model_performances[x])
        current_model = "linear"  # Assume linear is current default
        
        if best_model != current_model:
            improvement = (model_performances[current_model] - model_performances[best_model]) / model_performances[current_model]
            if improvement > 0.1:  # 10% improvement threshold
                return True, best_model
        
        return False, "no_significant_improvement"
    
    def get_learning_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of online learning performance"""
        
        summary = {
            "symbol": symbol,
            "timestamp": time.time(),
            "models": {},
            "bandits": {},
            "best_parameters": self.get_best_parameters(symbol)
        }
        
        # Model performance
        for model_type in self.performance_metrics[symbol]:
            summary["models"][model_type] = self.get_model_performance(symbol, model_type)
        
        # Bandit performance
        if symbol in self.context_history:
            recent_actions = list(self.context_history[symbol])[-50:]  # Last 50 actions
            if recent_actions:
                action_rewards = defaultdict(list)
                for action_data in recent_actions:
                    action_rewards[action_data["action"]].append(action_data["reward"])
                
                summary["bandits"]["action_performance"] = {
                    action: {
                        "avg_reward": np.mean(rewards),
                        "count": len(rewards)
                    }
                    for action, rewards in action_rewards.items()
                }
        
        return summary


class SimpleOnlineModel:
    """Fallback online model when River is not available"""
    
    def __init__(self, learning_rate: float = 0.01, decay: float = 0.99):
        self.learning_rate = learning_rate
        self.decay = decay
        self.weights = defaultdict(float)
        self.bias = 0.0
        self.count = 0
    
    def predict(self, features: Dict[str, float]) -> float:
        prediction = self.bias
        for feature, value in features.items():
            prediction += self.weights[feature] * value
        return prediction
    
    def update(self, features: Dict[str, float], target: float):
        prediction = self.predict(features)
        error = target - prediction
        
        # Update bias
        self.bias += self.learning_rate * error
        
        # Update weights
        for feature, value in features.items():
            self.weights[feature] = self.weights[feature] * self.decay + self.learning_rate * error * value
        
        self.count += 1


class SimpleBandit:
    """Fallback bandit implementation"""
    
    def __init__(self, arms: List[Any]):
        self.arms = arms
        self.counts = defaultdict(int)
        self.rewards = defaultdict(list)
        self.epsilon = 0.1
    
    def select(self) -> Any:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.arms)
        
        # Select best arm
        if not self.rewards:
            return np.random.choice(self.arms)
        
        avg_rewards = {arm: np.mean(self.rewards[arm]) for arm in self.arms if self.rewards[arm]}
        if not avg_rewards:
            return np.random.choice(self.arms)
        
        return max(avg_rewards.keys(), key=lambda x: avg_rewards[x])
    
    def update(self, arm: Any, reward: float):
        self.counts[arm] += 1
        self.rewards[arm].append(reward)


class SimpleContextualBandit:
    """Fallback contextual bandit"""
    
    def __init__(self, actions: List[str]):
        self.actions = actions
        self.action_rewards = defaultdict(list)
        self.epsilon = 0.1
    
    def select_action(self, context: Dict[str, float]) -> str:
        if np.random.random() < self.epsilon or not self.action_rewards:
            return np.random.choice(self.actions)
        
        # Simple average-based selection
        avg_rewards = {}
        for action in self.actions:
            if self.action_rewards[action]:
                avg_rewards[action] = np.mean([r["reward"] for r in self.action_rewards[action]])
        
        if not avg_rewards:
            return np.random.choice(self.actions)
        
        return max(avg_rewards.keys(), key=lambda x: avg_rewards[x])
    
    def update(self, action: str, context: Dict[str, float], reward: float):
        self.action_rewards[action].append({
            "context": context,
            "reward": reward,
            "timestamp": time.time()
        })


# Global online learner instance
online_learner = OnlineLearner()