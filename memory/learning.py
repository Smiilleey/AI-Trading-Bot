# memory/learning.py

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from utils.config import ML_CONFIDENCE_THRESHOLD, ML_MIN_SAMPLES, ML_RETRAIN_INTERVAL, LEARNING_RATE, MEMORY_DECAY_RATE, PATTERN_WEIGHT_DECAY

MEMORY_FILE = "memory/pattern_memory.json"
MODEL_FILE = "memory/ml_models.joblib"
FEATURE_SCALER_FILE = "memory/feature_scaler.joblib"

class _CompatWrappersMixin:
    """
    Backward-compatibility + light stats helpers.
    """
    def suggest_confidence(self, symbol=None, features=None, **kwargs):
        try:
            return self.predict_confidence(symbol, features)
        except Exception:
            return 0.0

    def get_advanced_stats(self, symbol=None, **kwargs):
        try:
            stats = self.get_learning_stats()
        except Exception:
            stats = {}
        try:
            if symbol and getattr(self, 'memory', None):
                import numpy as np
                symbol_patterns = [p for p in self.memory.values()
                                   if isinstance(p, dict) and p.get('symbol') == symbol]
                stats['symbol_pattern_count'] = len(symbol_patterns)
                if symbol_patterns:
                    stats['symbol_avg_success'] = float(np.mean([p.get('success_rate', 0.0)
                                                                 for p in symbol_patterns]))
        except Exception:
            pass
        return stats

class AdvancedLearningEngine(_CompatWrappersMixin):
    """
    Advanced machine learning engine for continuous improvement:
    - Multi-model ensemble (Random Forest + Gradient Boosting)
    - Feature engineering from market data, patterns, and context
    - Continuous learning with adaptive confidence thresholds
    - Pattern memory with decay and importance weighting
    - Real-time model retraining and performance monitoring
    """
    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = memory_file
        self.memory = self._load_memory()
        self.unsaved_changes = 0
        self.save_threshold = 10
        
        # ML Models
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Performance tracking
        self.model_performance = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "last_retrain": None,
            "predictions_made": 0
        }
        
        # Load or initialize models
        self._load_or_initialize_models()
        
        # Pattern importance tracking
        self.pattern_importance = {}
        self.context_weights = {}
        
        # Rolling memory for continuous learning
        self.trade_history = deque(maxlen=500)
        
        # Clean up old patterns on initialization
        self._cleanup_old_patterns()

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return {}
        try:
            with open(self.memory_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=2)
            self.unsaved_changes = 0
        except Exception as e:
            print(f"Failed to save memory: {e}")

    def _cleanup_old_patterns(self):
        """Remove old patterns that are no longer relevant"""
        current_time = datetime.now()
        patterns_to_remove = []
        
        for pattern_id, pattern_data in self.memory.items():
            if "created" in pattern_data:
                try:
                    created_time = datetime.fromisoformat(pattern_data["created"])
                    age_days = (current_time - created_time).days
                    
                    # Remove patterns older than 90 days with low importance
                    if age_days > 90 and pattern_data.get("importance", 1.0) < 0.5:
                        patterns_to_remove.append(pattern_id)
                    
                    # Remove patterns older than 180 days regardless of importance
                    elif age_days > 180:
                        patterns_to_remove.append(pattern_id)
                        
                except (ValueError, TypeError):
                    # If date parsing fails, remove the pattern
                    patterns_to_remove.append(pattern_id)
        
        # Remove old patterns
        for pattern_id in patterns_to_remove:
            del self.memory[pattern_id]
        
        if patterns_to_remove:
            print(f"🧹 Cleaned up {len(patterns_to_remove)} old patterns")
            self._save_memory()

    def _load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        try:
            if os.path.exists(MODEL_FILE):
                models = joblib.load(MODEL_FILE)
                self.rf_model = models.get('rf_model')
                self.gb_model = models.get('gb_model')
                self.scaler = models.get('scaler', StandardScaler())
                self.label_encoder = models.get('label_encoder', LabelEncoder())
                self.feature_names = models.get('feature_names', [])
                print("✅ Loaded existing ML models")
            else:
                self._initialize_models()
                print("🆕 Initialized new ML models")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self._initialize_models()

    def _initialize_models(self):
        """Initialize new ML models"""
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            learning_rate=0.1
        )
        self.feature_names = []

    def _extract_features(self, market_context, signal_type, market_data):
        """Extract comprehensive features from market context"""
        features = []
        
        # Basic market features
        if market_data and "candles" in market_data and market_data["candles"]:
            candles = market_data["candles"]
            if len(candles) >= 20:
                # Convert to pandas DataFrame for analysis
                df = pd.DataFrame(candles)
                
                # Price momentum features
                df['returns'] = df['close'].pct_change()
                features.extend([
                    df['returns'].mean(),  # Average return
                    df['returns'].std(),   # Volatility
                    df['returns'].skew(),  # Skewness
                    df['returns'].tail(5).mean(),  # Recent momentum
                    df['returns'].tail(10).mean(), # Medium momentum
                ])
                
                # Volume features
                if 'tick_volume' in df.columns:
                    df['volume_ma'] = df['tick_volume'].rolling(5).mean()
                    features.extend([
                        df['tick_volume'].mean(),
                        df['tick_volume'].std(),
                        df['tick_volume'].tail(5).mean() / df['tick_volume'].tail(20).mean() if df['tick_volume'].tail(20).mean() > 0 else 1,
                    ])
                else:
                    features.extend([0, 0, 1])
                
                # Technical features
                df['high_low_ratio'] = df['high'] / df['low']
                df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
                features.extend([
                    df['high_low_ratio'].mean(),
                    df['close_position'].mean(),
                    df['close_position'].std(),
                ])
            else:
                features.extend([0] * 13)  # Fill with zeros if insufficient data
        else:
            features.extend([0] * 13)
        
        # Context features
        if market_context:
            # Volatility regime
            volatility_map = {"low": 0, "normal": 1, "high": 2}
            volatility = market_context.get("volatility_regime", "normal")
            features.append(volatility_map.get(volatility, 1))
            
            # Session context
            sessions = market_context.get("sessions", [])
            features.append(len(sessions))
            
            # MTF bias and confidence
            mtf_bias = market_context.get("mtf_bias", "neutral")
            bias_map = {"bearish": -1, "neutral": 0, "bullish": 1}
            features.extend([
                bias_map.get(mtf_bias, 0),
                market_context.get("mtf_confidence", 0.5),
                market_context.get("mtf_entry_ok", False) * 1.0,
                market_context.get("mtf_confluence_strength", 0.0),
            ])
            
            # Order flow features
            of = market_context.get("order_flow", {})
            features.extend([
                of.get("delta", 0),
                of.get("absorption_ratio", 0.5),
                of.get("participants_bias", 0.0),
            ])
        else:
            features.extend([1, 0, 0, 0.5, 0, 0, 0.5, 0.0])
        
        # Signal type
        signal_map = {"bullish": 1, "bearish": -1, "neutral": 0}
        features.append(signal_map.get(signal_type, 0))
        
        return np.array(features)

    def _prepare_training_data(self):
        """Prepare training data from memory"""
        if not self.memory or len(self.memory) < ML_MIN_SAMPLES:
            return None, None
        
        features_list = []
        labels = []
        
        for pattern_id, pattern_data in self.memory.items():
            if "features" in pattern_data and "outcomes" in pattern_data:
                features = pattern_data["features"]
                if len(features) > 0:
                    # Use pattern importance as sample weight
                    importance = pattern_data.get("importance", 1.0)
                    
                    # Create multiple samples based on outcomes
                    for outcome in pattern_data["outcomes"]:
                        features_list.append(features)
                        labels.append(1 if outcome["outcome"] == "win" else 0)
        
        if len(features_list) < ML_MIN_SAMPLES:
            return None, None
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Update feature names if needed
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        return X, y

    def _retrain_models(self):
        """Retrain ML models with current data"""
        X, y = self._prepare_training_data()
        if X is None or len(X) < ML_MIN_SAMPLES:
            return False
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            self.rf_model.fit(X_train_scaled, y_train)
            self.gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            rf_pred = self.rf_model.predict(X_test_scaled)
            gb_pred = self.gb_model.predict(X_test_scaled)
            
            self.model_performance.update({
                "accuracy": accuracy_score(y_test, rf_pred),
                "precision": precision_score(y_test, rf_pred, zero_division=0),
                "recall": recall_score(y_test, rf_pred, zero_division=0),
                "f1_score": f1_score(y_test, rf_pred, zero_division=0),
                "last_retrain": datetime.now().isoformat(),
            })
            
            # Save models
            self._save_models()
            print(f"✅ Models retrained. Accuracy: {self.model_performance['accuracy']:.2%}")
            return True
            
        except Exception as e:
            print(f"❌ Error retraining models: {e}")
            return False

    def _save_models(self):
        """Save trained models"""
        try:
            models = {
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            joblib.dump(models, MODEL_FILE)
        except Exception as e:
            print(f"Failed to save models: {e}")

    def predict_confidence(self, market_context, signal_type, market_data):
        """Predict confidence using ML models"""
        if not self.rf_model or not self.gb_model:
            return 0.7  # Default confidence
        
        try:
            features = self._extract_features(market_context, signal_type, market_data)
            if len(features) == 0:
                return 0.7
            
            # Ensure features match expected dimensions
            if hasattr(self.scaler, 'n_features_in_') and len(features) != self.scaler.n_features_in_:
                # Pad or truncate features to match
                if len(features) < self.scaler.n_features_in_:
                    features = np.pad(features, (0, self.scaler.n_features_in_ - len(features)), 'constant')
                else:
                    features = features[:self.scaler.n_features_in_]
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get predictions from both models
            rf_prob = self.rf_model.predict_proba(features_scaled)[0][1]  # Probability of win
            gb_prob = self.gb_model.predict_proba(features_scaled)[0][1]
            
            # Ensemble prediction (weighted average)
            confidence = (rf_prob * 0.6 + gb_prob * 0.4)
            
            # Apply ML confidence threshold
            if confidence < ML_CONFIDENCE_THRESHOLD:
                confidence *= 0.8  # Reduce confidence for low-probability signals
            
            self.model_performance["predictions_made"] += 1
            
            # Trigger retraining if needed
            if (self.model_performance["predictions_made"] % ML_RETRAIN_INTERVAL == 0 and 
                self.model_performance["predictions_made"] > 0):
                self._retrain_models()
            
            return min(confidence, 0.95)  # Cap at 95%
            
        except Exception as e:
            print(f"Error in confidence prediction: {e}")
            return 0.7

    def suggest_confidence(self, symbol: str, features: dict, **kwargs) -> float:
        """
        Enhanced wrapper for predict_confidence with flexible parameter handling.
        Supports both dict-based and direct parameter calls for maximum compatibility.
        """
        try:
            # Handle different parameter formats flexibly
            if isinstance(features, dict):
                # Dict-based call (current surgical patch format)
                signal_type = features.get("signal_type", "neutral")
                market_context = features.get("market_context", {})
                market_data = features.get("market_data", {})
            else:
                # Direct parameter call (legacy format support)
                signal_type = features if features else "neutral"
                market_context = kwargs.get("market_context", {})
                market_data = kwargs.get("market_data", {})
            
            # Enhanced ML prediction with fallback chain
            if market_context and market_data:
                try:
                    ml_confidence = self.predict_confidence(market_context, signal_type, market_data)
                    if ml_confidence and ml_confidence > 0:
                        return ml_confidence
                except Exception as ml_error:
                    print(f"ML prediction failed, using fallback: {ml_error}")
            
            # Intelligent fallback based on available data
            if signal_type in ["bullish", "bearish"]:
                # Check for pattern memory if available
                pattern_key = f"{symbol}_{signal_type}_{len(self.memory)}"
                if pattern_key in self.memory:
                    pattern = self.memory[pattern_key]
                    if pattern.get("success_rate", 0) > 0.6:
                        return min(pattern["success_rate"] * 1.1, 0.9)
                
                # Base confidence with signal type adjustment
                base_confidence = {
                    "bullish": 0.7,
                    "bearish": 0.7,
                    "neutral": 0.5
                }
                confidence = base_confidence.get(signal_type, 0.6)
                
                # Adjust based on market context if available
                if market_context:
                    volatility = market_context.get("volatility_regime", "normal")
                    if volatility == "high":
                        confidence *= 0.9  # Reduce confidence in high volatility
                    elif volatility == "low":
                        confidence *= 1.1  # Increase confidence in low volatility
                
                return min(confidence, 0.95)  # Cap at 95%
            
            return 0.6  # Default neutral confidence
            
        except Exception as e:
            print(f"Error in suggest_confidence: {e}")
            # Return safe default with error logging
            return 0.6

    def get_advanced_stats(self, symbol: str, **kwargs) -> dict:
        """
        Enhanced wrapper for get_learning_stats with comprehensive symbol analysis.
        Provides both basic stats and advanced analytics for informed decision making.
        """
        try:
            # Get base learning stats
            base_stats = self.get_learning_stats()
            
            # Enhanced symbol-specific analysis
            if symbol:
                # Find all patterns related to this symbol
                symbol_patterns = {k: v for k, v in self.memory.items() 
                                 if k.startswith(symbol) or symbol in k}
                
                if symbol_patterns:
                    # Calculate comprehensive symbol metrics
                    success_rates = [p.get("success_rate", 0) for p in symbol_patterns.values()]
                    importance_scores = [p.get("importance", 1.0) for p in symbol_patterns.values()]
                    pnl_values = [o.get("pnl", 0) for p in symbol_patterns.values() 
                                for o in p.get("outcomes", [])]
                    rr_values = [o.get("rr", 0) for p in symbol_patterns.values() 
                               for o in p.get("outcomes", [])]
                    
                    # Enhanced symbol statistics
                    symbol_stats = {
                        "symbol": symbol,
                        "pattern_count": len(symbol_patterns),
                        "total_patterns": len(self.memory),
                        "symbol_success_rate": np.mean(success_rates) if success_rates else 0,
                        "symbol_avg_importance": np.mean(importance_scores) if importance_scores else 1.0,
                        "symbol_confidence": np.mean(success_rates) * np.mean(importance_scores) if success_rates and importance_scores else 0,
                        "avg_pnl": np.mean(pnl_values) if pnl_values else 0,
                        "avg_rr": np.mean(rr_values) if rr_values else 0,
                        "best_pattern": max(symbol_patterns.values(), key=lambda x: x.get("success_rate", 0)) if symbol_patterns else None,
                        "recent_performance": self._get_recent_performance(symbol, symbol_patterns),
                        "market_regime_adaptation": self._get_market_regime_stats(symbol, symbol_patterns)
                    }
                else:
                    # No patterns for this symbol yet
                    symbol_stats = {
                        "symbol": symbol,
                        "pattern_count": 0,
                        "total_patterns": len(self.memory),
                        "symbol_success_rate": 0,
                        "symbol_avg_importance": 1.0,
                        "symbol_confidence": 0,
                        "avg_pnl": 0,
                        "avg_rr": 0,
                        "best_pattern": None,
                        "recent_performance": "insufficient_data",
                        "market_regime_adaptation": "insufficient_data"
                    }
                
                # Merge base stats with enhanced symbol stats
                return {**base_stats, **symbol_stats}
            else:
                # Return base stats if no symbol specified
                return base_stats
                
        except Exception as e:
            print(f"Error in get_advanced_stats: {e}")
            # Return safe fallback with error information
            return {
                "error": f"Failed to get stats: {e}",
                "symbol": symbol,
                "fallback_data": True,
                **self.model_performance
            }

    def record_result(self, pattern_id, outcome, pnl, rr):
        """Record pattern outcome for learning"""
        if pattern_id not in self.memory:
            self.memory[pattern_id] = {
                "created": datetime.now().isoformat(),
                "outcomes": [],
                "success_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_rr": 0.0,
                "importance": 1.0,
                "features": []
            }
        
        pattern = self.memory[pattern_id]
        pattern["outcomes"].append({
            "outcome": outcome,
            "pnl": pnl,
            "rr": rr,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update success rate
        wins = sum(1 for o in pattern["outcomes"] if o["outcome"] == "win")
        pattern["success_rate"] = wins / len(pattern["outcomes"])
        
        # Update average metrics
        if pattern["outcomes"]:
            pattern["avg_pnl"] = np.mean([o["pnl"] for o in pattern["outcomes"]])
            pattern["avg_rr"] = np.mean([o["rr"] for o in pattern["outcomes"]])
        
        # Update importance based on performance
        if pattern["success_rate"] > 0.6:
            pattern["importance"] *= 1.1
        elif pattern["success_rate"] < 0.4:
            pattern["importance"] *= 0.9
        
        # Apply decay
        pattern["importance"] *= PATTERN_WEIGHT_DECAY
        
        self.unsaved_changes += 1
        if self.unsaved_changes >= self.save_threshold:
            self._save_memory()

    def get_pattern_importance(self, pattern_id):
        """Get pattern importance score"""
        return self.memory.get(pattern_id, {}).get("importance", 1.0)

    def get_learning_stats(self):
        """Get learning statistics"""
        if not self.memory:
            return self.model_performance
        
        total_patterns = len(self.memory)
        avg_success = np.mean([p.get("success_rate", 0) for p in self.memory.values()])
        avg_importance = np.mean([p.get("importance", 1.0) for p in self.memory.values()])
        
        return {
            "total_patterns": total_patterns,
            "avg_success_rate": avg_success,
            "avg_importance": avg_importance,
            "last_retrain": self.model_performance["last_retrain"],
            "predictions_made": self.model_performance["predictions_made"],
            **self.model_performance
        }

    def _get_recent_performance(self, symbol: str, symbol_patterns: dict) -> str:
        """Get recent performance classification for a symbol"""
        try:
            if not symbol_patterns:
                return "insufficient_data"
            
            # Get recent outcomes (last 10)
            recent_outcomes = []
            for pattern in symbol_patterns.values():
                outcomes = pattern.get("outcomes", [])
                recent_outcomes.extend(outcomes[-3:])  # Last 3 from each pattern
            
            if not recent_outcomes:
                return "insufficient_data"
            
            # Calculate recent success rate
            recent_wins = sum(1 for o in recent_outcomes if o.get("outcome") == "win")
            recent_success_rate = recent_wins / len(recent_outcomes)
            
            # Classify performance
            if recent_success_rate >= 0.7:
                return "excellent"
            elif recent_success_rate >= 0.6:
                return "good"
            elif recent_success_rate >= 0.5:
                return "average"
            else:
                return "poor"
                
        except Exception as e:
            print(f"Error calculating recent performance: {e}")
            return "error"

    def _get_market_regime_stats(self, symbol: str, symbol_patterns: dict) -> str:
        """Get market regime adaptation statistics for a symbol"""
        try:
            if not symbol_patterns:
                return "insufficient_data"
            
            # Analyze performance across different market conditions
            regime_performance = {}
            for pattern in symbol_patterns.values():
                outcomes = pattern.get("outcomes", [])
                for outcome in outcomes:
                    # Extract market context if available
                    market_context = outcome.get("market_context", {})
                    volatility = market_context.get("volatility_regime", "normal")
                    
                    if volatility not in regime_performance:
                        regime_performance[volatility] = {"wins": 0, "total": 0}
                    
                    regime_performance[volatility]["total"] += 1
                    if outcome.get("outcome") == "win":
                        regime_performance[volatility]["wins"] += 1
            
            if not regime_performance:
                return "insufficient_data"
            
            # Calculate adaptation score
            adaptation_scores = []
            for regime, stats in regime_performance.items():
                if stats["total"] >= 3:  # Minimum samples for reliable stats
                    success_rate = stats["wins"] / stats["total"]
                    adaptation_scores.append(success_rate)
            
            if not adaptation_scores:
                return "insufficient_data"
            
            avg_adaptation = np.mean(adaptation_scores)
            
            # Classify adaptation
            if avg_adaptation >= 0.65:
                return "highly_adaptive"
            elif avg_adaptation >= 0.55:
                return "adaptive"
            elif avg_adaptation >= 0.45:
                return "moderately_adaptive"
            else:
                return "low_adaptation"
                
        except Exception as e:
            print(f"Error calculating market regime stats: {e}")
            return "error"

    def record_result(self, symbol, features, outcome: float):
        """
        Store outcome for continuous evolution. outcome: 1.0 win, 0.0 loss.
        """
        try:
            self.trade_history.append({
                'symbol': symbol,
                'features': features,
                'outcome': float(outcome)
            })
        except Exception:
            pass

    def force_save(self):
        """Force save memory and models"""
        self._save_memory()
        self._save_models()

    def cleanup(self):
        """Cleanup resources"""
        self.force_save()
