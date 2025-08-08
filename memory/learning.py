# memory/learning.py

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from utils.config import ML_CONFIDENCE_THRESHOLD, ML_MIN_SAMPLES, ML_RETRAIN_INTERVAL, LEARNING_RATE, MEMORY_DECAY_RATE, PATTERN_WEIGHT_DECAY

MEMORY_FILE = "memory/pattern_memory.json"
MODEL_FILE = "memory/ml_models.joblib"
FEATURE_SCALER_FILE = "memory/feature_scaler.joblib"

class AdvancedLearningEngine:
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
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def _save_models(self):
        """Save current models to disk"""
        try:
            os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
            models = {
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            joblib.dump(models, MODEL_FILE)
            print("💾 Models saved successfully")
        except Exception as e:
            print(f"❌ Failed to save models: {e}")

    def _extract_features(self, context, signal_type, market_data):
        """Extract comprehensive features for ML model"""
        features = {}
        
        # Basic signal features
        features['signal_type'] = signal_type
        features['confidence'] = context.get('confidence', 'unknown')
        
        # Market context features
        if 'volatility_regime' in context:
            features['volatility_regime'] = context['volatility_regime']
        if 'momentum_shift' in context:
            features['momentum_shift'] = context['momentum_shift']
        
        # Session features
        if 'session_context' in context:
            session_ctx = context['session_context']
            features['active_sessions_count'] = len(session_ctx.get('active_sessions', []))
            features['has_monday_expectation'] = 'monday_expectation' in session_ctx
            features['has_thursday_expectation'] = 'thursday_expectation' in session_ctx
        
        # Pattern features
        if 'situational_tags' in context:
            tags = context['situational_tags']
            features['friday_thursday_reversal'] = 'friday_thursday_reversal' in tags
            features['wednesday_monday_pullback'] = 'wednesday_monday_pullback' in tags
            features['high_volatility_regime'] = 'high_volatility_regime' in tags
            features['low_volatility_regime'] = 'low_volatility_regime' in tags
            features['london_ny_overlap'] = 'london_ny_overlap' in tags
        
        # Market data features (if available)
        if market_data and 'candles' in market_data:
            candles = market_data['candles']
            if len(candles) >= 5:
                recent_candles = candles[-5:]
                features['avg_range'] = np.mean([c['high'] - c['low'] for c in recent_candles])
                features['price_momentum'] = (recent_candles[-1]['close'] - recent_candles[0]['close']) / recent_candles[0]['close']
                features['volume_trend'] = np.mean([c.get('tick_volume', 0) for c in recent_candles])
        
        # Time-based features
        if 'timestamp' in context:
            try:
                dt = datetime.fromisoformat(context['timestamp'])
                features['hour'] = dt.hour
                features['day_of_week'] = dt.weekday()
                features['is_weekend'] = dt.weekday() >= 5
            except:
                pass
        
        return features

    def _prepare_training_data(self):
        """Prepare training data from memory"""
        if not self.memory:
            return None, None, None
        
        features_list = []
        labels = []
        
        for pair, records in self.memory.items():
            for record in records:
                if 'context' in record and 'outcome' in record:
                    features = self._extract_features(
                        record['context'], 
                        record.get('signal', 'unknown'), 
                        record.get('market_data', {})
                    )
                    
                    if features:
                        features_list.append(features)
                        labels.append(1 if record['outcome'] == 'win' else 0)
        
        if len(features_list) < ML_MIN_SAMPLES:
            return None, None, None
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(features_list)
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = self.label_encoder.fit_transform(df[col].astype(str))
        
        # Convert to numpy arrays
        X = df.values
        y = np.array(labels)
        
        return X, y, df.columns.tolist()

    def _retrain_models(self):
        """Retrain ML models with current data"""
        X, y, feature_names = self._prepare_training_data()
        
        if X is None or len(X) < ML_MIN_SAMPLES:
            print(f"⚠️ Insufficient data for retraining (need {ML_MIN_SAMPLES}, have {len(X) if X is not None else 0})")
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
            
            # Ensemble prediction (average of both models)
            ensemble_pred = ((rf_pred + gb_pred) / 2) > 0.5
            
            # Calculate metrics
            self.model_performance = {
                "accuracy": accuracy_score(y_test, ensemble_pred),
                "precision": precision_score(y_test, ensemble_pred, zero_division=0),
                "recall": recall_score(y_test, ensemble_pred, zero_division=0),
                "f1_score": f1_score(y_test, ensemble_pred, zero_division=0),
                "last_retrain": datetime.now().isoformat(),
                "predictions_made": self.model_performance.get("predictions_made", 0)
            }
            
            self.feature_names = feature_names
            self._save_models()
            
            print(f"✅ Models retrained successfully - Accuracy: {self.model_performance['accuracy']:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Model retraining failed: {e}")
            return False

    def predict_confidence(self, context, signal_type, market_data=None):
        """Predict confidence using ML models"""
        if not self.rf_model or not self.gb_model:
            return "unknown"
        
        try:
            # Extract features
            features = self._extract_features(context, signal_type, market_data)
            if not features:
                return "unknown"
            
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Handle categorical variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = self.label_encoder.transform(df[col].astype(str))
            
            # Scale features
            X = self.scaler.transform(df.values)
            
            # Get predictions
            rf_prob = self.rf_model.predict_proba(X)[0][1]
            gb_prob = self.gb_model.predict_proba(X)[0][1]
            
            # Ensemble probability
            ensemble_prob = (rf_prob + gb_prob) / 2
            
            # Convert to confidence level
            if ensemble_prob > ML_CONFIDENCE_THRESHOLD:
                return "high"
            elif ensemble_prob > 0.5:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            return "unknown"

    def record_result(self, pair, context, signal, outcome, rr, entry_time, tags=None, exit_time=None, pnl=None, market_data=None):
        """Record trade outcome with enhanced learning"""
        record = {
            "pair": pair,
            "context": context,
            "signal": signal,
            "outcome": outcome,
            "rr": rr,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "tags": tags or [],
            "pnl": pnl,
            "market_data": market_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if pair not in self.memory:
            self.memory[pair] = []
        self.memory[pair].append(record)
        self.unsaved_changes += 1
        
        # Update pattern importance
        self._update_pattern_importance(record)
        
        # Check if retraining is needed
        total_records = sum(len(records) for records in self.memory.values())
        if total_records % ML_RETRAIN_INTERVAL == 0:
            print(f"🔄 Retraining models after {total_records} records...")
            self._retrain_models()
        
        # Save periodically
        if self.unsaved_changes >= self.save_threshold:
            self._save_memory()

    def _update_pattern_importance(self, record):
        """Update pattern importance based on outcomes"""
        context = record.get('context', {})
        outcome = record.get('outcome', 'unknown')
        success = outcome == 'win'
        
        # Update situational tag importance
        if 'situational_tags' in context:
            for tag in context['situational_tags']:
                if tag not in self.pattern_importance:
                    self.pattern_importance[tag] = {'wins': 0, 'total': 0, 'weight': 1.0}
                
                self.pattern_importance[tag]['total'] += 1
                if success:
                    self.pattern_importance[tag]['wins'] += 1
                
                # Update weight based on success rate
                success_rate = self.pattern_importance[tag]['wins'] / self.pattern_importance[tag]['total']
                self.pattern_importance[tag]['weight'] = success_rate * PATTERN_WEIGHT_DECAY

    def suggest_confidence(self, pair, signal_type):
        """Enhanced confidence suggestion using ML models"""
        # First try ML prediction
        if self.rf_model and self.gb_model:
            # Get recent context for this pair
            pair_records = self.memory.get(pair, [])
            if pair_records:
                recent_context = pair_records[-1].get('context', {})
                return self.predict_confidence(recent_context, signal_type)
        
        # Fallback to historical win rate
        records = self.memory.get(pair, [])
        if not records:
            return "unknown"
        
        relevant = [r for r in records if r["signal"] == signal_type]
        if not relevant:
            return "low"
        
        wins = [r for r in relevant if r["outcome"] == "win"]
        win_rate = len(wins) / len(relevant)
        
        if win_rate >= 0.7:
            return "high"
        elif win_rate >= 0.5:
            return "medium"
        else:
            return "low"

    def get_advanced_stats(self, pair=None, signal_type=None):
        """Get comprehensive statistics including ML performance"""
        stats = self.get_pattern_stats(pair, signal_type)
        
        # Add ML performance metrics
        stats['ml_performance'] = self.model_performance
        stats['pattern_importance'] = self.pattern_importance
        
        # Add learning insights
        stats['total_patterns_learned'] = len(self.pattern_importance)
        stats['model_confidence'] = self.model_performance.get('accuracy', 0)
        
        return stats

    def get_pattern_stats(self, pair, signal_type=None):
        """Get basic pattern statistics"""
        records = self.memory.get(pair, []) if pair else []
        if not pair:
            # Aggregate all pairs
            all_records = []
            for pair_records in self.memory.values():
                all_records.extend(pair_records)
            records = all_records
        
        if signal_type:
            records = [r for r in records if r["signal"] == signal_type]
        
        total = len(records)
        if total == 0:
            return {"total": 0, "wins": 0, "win_rate": 0, "average_rr": 0, "tags": [], "streak": 0}
        
        wins = [r for r in records if r["outcome"] == "win"]
        avg_rr = sum(r.get("rr", 0) for r in records) / total
        tag_list = [tag for r in records for tag in r.get("tags", [])]
        
        # Calculate streak
        last_results = [r["outcome"] for r in records[-10:]]
        streak = 0
        for outcome in reversed(last_results):
            if outcome == "win":
                streak = streak + 1 if streak >= 0 else 1
            elif outcome == "loss":
                streak = streak - 1 if streak <= 0 else -1
            else:
                break
        
        return {
            "total": total,
            "wins": len(wins),
            "win_rate": len(wins) / total,
            "average_rr": avg_rr,
            "tags": list(set(tag_list)),
            "streak": streak
        }

    def force_save(self):
        """Force save all data"""
        if self.unsaved_changes > 0:
            self._save_memory()
        self._save_models()

    def __del__(self):
        """Ensure data is saved when object is destroyed"""
        try:
            self.force_save()
        except:
            pass

# Backward compatibility
LearningEngine = AdvancedLearningEngine
