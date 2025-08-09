# core/prophetic_ml.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import joblib
from collections import defaultdict

class PropheticMLEngine:
    """
    ML-enhanced cycle prediction engine:
    - Pattern recognition
    - Cycle prediction
    - Confidence scoring
    - Feature importance
    """
    def __init__(
        self,
        config: Optional[Dict] = None,
        model_path: Optional[str] = None
    ):
        self.config = config or {}
        self.model_path = model_path
        
        # Initialize models
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Performance tracking
        self.feature_importance = {}
        self.prediction_accuracy = defaultdict(list)
        self.confidence_scores = defaultdict(float)
        
        # Load or initialize models
        self._initialize_models()
        
    def predict_cycle(
        self,
        market_data: Dict,
        context: Dict
    ) -> Dict:
        """
        Predict market cycle:
        - Current phase
        - Next phase
        - Transition probability
        - Confidence score
        """
        # Extract features
        features = self._extract_features(
            market_data,
            context
        )
        
        # Scale features
        scaled_features = self.scaler.transform(
            features.reshape(1, -1)
        )
        
        # Get predictions
        rf_pred = self.rf_model.predict_proba(scaled_features)[0]
        gb_pred = self.gb_model.predict_proba(scaled_features)[0]
        
        # Combine predictions
        combined_pred = (rf_pred + gb_pred) / 2
        
        # Get predicted class
        pred_idx = np.argmax(combined_pred)
        pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Calculate confidence
        confidence = combined_pred[pred_idx]
        
        # Get feature importance
        importance = self._get_feature_importance(
            scaled_features
        )
        
        return {
            "current_phase": pred_class,
            "next_phase": self._predict_next_phase(
                pred_class,
                combined_pred
            ),
            "transition_prob": self._calculate_transition_prob(
                combined_pred
            ),
            "confidence": confidence,
            "feature_importance": importance
        }
        
    def train(
        self,
        training_data: List[Dict],
        labels: List[str]
    ):
        """
        Train ML models:
        - Feature extraction
        - Model training
        - Performance evaluation
        """
        # Prepare data
        features = []
        for data in training_data:
            features.append(
                self._extract_features(
                    data["market_data"],
                    data["context"]
                )
            )
            
        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train models
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            y_train = y[train_idx]
            X_val = X_scaled[val_idx]
            y_val = y[val_idx]
            
            # Train Random Forest
            self.rf_model.fit(X_train, y_train)
            
            # Train Gradient Boosting
            self.gb_model.fit(X_train, y_train)
            
            # Evaluate
            self._evaluate_performance(
                X_val,
                y_val
            )
            
        # Update feature importance
        self._update_feature_importance()
        
        # Save models
        if self.model_path:
            self._save_models()
            
    def update(
        self,
        market_data: Dict,
        context: Dict,
        actual_phase: str
    ):
        """
        Update models with new data:
        - Online learning
        - Performance tracking
        - Feature importance update
        """
        # Extract features
        features = self._extract_features(
            market_data,
            context
        )
        
        # Scale features
        scaled_features = self.scaler.transform(
            features.reshape(1, -1)
        )
        
        # Convert label
        y = self.label_encoder.transform([actual_phase])[0]
        
        # Update models
        self.rf_model.fit(
            scaled_features,
            [y]
        )
        
        self.gb_model.fit(
            scaled_features,
            [y]
        )
        
        # Update feature importance
        self._update_feature_importance()
        
    def _initialize_models(self):
        """Initialize or load ML models"""
        if self.model_path and self._models_exist():
            self._load_models()
        else:
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
    def _extract_features(
        self,
        market_data: Dict,
        context: Dict
    ) -> np.ndarray:
        """
        Extract features for ML:
        - Price patterns
        - Volume patterns
        - Cycle indicators
        - Context features
        """
        features = []
        
        # Price features
        prices = np.array(market_data["prices"])
        features.extend([
            np.mean(prices),
            np.std(prices),
            self._calculate_trend_strength(prices),
            self._calculate_volatility(prices)
        ])
        
        # Volume features
        volumes = np.array(market_data["volumes"])
        features.extend([
            np.mean(volumes),
            np.std(volumes),
            self._calculate_volume_trend(volumes)
        ])
        
        # Cycle features
        features.extend([
            self._calculate_cycle_momentum(prices),
            self._calculate_cycle_position(context),
            self._calculate_cycle_duration(context)
        ])
        
        # Context features
        features.extend([
            context.get("market_regime", 0),
            context.get("sentiment_score", 0),
            context.get("volatility_regime", 0)
        ])
        
        return np.array(features)
        
    def _predict_next_phase(
        self,
        current_phase: str,
        probabilities: np.ndarray
    ) -> str:
        """Predict next market phase"""
        # Get transition matrix
        transitions = self._get_transition_matrix()
        
        # Get current phase index
        current_idx = self.label_encoder.transform([current_phase])[0]
        
        # Calculate next phase probabilities
        next_probs = transitions[current_idx] * probabilities
        
        # Get most likely next phase
        next_idx = np.argmax(next_probs)
        
        return self.label_encoder.inverse_transform([next_idx])[0]
        
    def _calculate_transition_prob(
        self,
        probabilities: np.ndarray
    ) -> float:
        """Calculate phase transition probability"""
        # Get top two probabilities
        top_two = np.sort(probabilities)[-2:]
        
        # Calculate transition probability
        return 1 - (top_two[1] - top_two[0])
        
    def _get_feature_importance(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Get feature importance scores"""
        feature_names = [
            "price_mean",
            "price_std",
            "trend_strength",
            "volatility",
            "volume_mean",
            "volume_std",
            "volume_trend",
            "cycle_momentum",
            "cycle_position",
            "cycle_duration",
            "market_regime",
            "sentiment_score",
            "volatility_regime"
        ]
        
        # Get importance from both models
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_
        
        # Combine importance scores
        importance = (rf_importance + gb_importance) / 2
        
        return dict(zip(feature_names, importance))
        
    def _evaluate_performance(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Evaluate model performance"""
        # Get predictions
        rf_pred = self.rf_model.predict(X_val)
        gb_pred = self.gb_model.predict(X_val)
        
        # Calculate accuracy
        rf_accuracy = np.mean(rf_pred == y_val)
        gb_accuracy = np.mean(gb_pred == y_val)
        
        # Update tracking
        self.prediction_accuracy["rf"].append(rf_accuracy)
        self.prediction_accuracy["gb"].append(gb_accuracy)
        
        # Update confidence scores
        self.confidence_scores["rf"] = np.mean(
            self.prediction_accuracy["rf"][-100:]
        )
        self.confidence_scores["gb"] = np.mean(
            self.prediction_accuracy["gb"][-100:]
        )
        
    def _update_feature_importance(self):
        """Update feature importance tracking"""
        importance = self._get_feature_importance(
            np.zeros((1, 13))  # Dummy features for importance calculation
        )
        
        # Update tracking
        for feature, score in importance.items():
            if feature not in self.feature_importance:
                self.feature_importance[feature] = []
            self.feature_importance[feature].append(score)
            
    def _calculate_trend_strength(
        self,
        prices: np.ndarray
    ) -> float:
        """Calculate price trend strength"""
        # Implementation details...
        pass
        
    def _calculate_volatility(
        self,
        prices: np.ndarray
    ) -> float:
        """Calculate price volatility"""
        # Implementation details...
        pass
        
    def _calculate_volume_trend(
        self,
        volumes: np.ndarray
    ) -> float:
        """Calculate volume trend"""
        # Implementation details...
        pass
        
    def _calculate_cycle_momentum(
        self,
        prices: np.ndarray
    ) -> float:
        """Calculate cycle momentum"""
        # Implementation details...
        pass
        
    def _calculate_cycle_position(
        self,
        context: Dict
    ) -> float:
        """Calculate position in current cycle"""
        # Implementation details...
        pass
        
    def _calculate_cycle_duration(
        self,
        context: Dict
    ) -> float:
        """Calculate cycle duration"""
        # Implementation details...
        pass
        
    def _get_transition_matrix(self) -> np.ndarray:
        """Get phase transition probability matrix"""
        # Implementation details...
        pass
        
    def _models_exist(self) -> bool:
        """Check if saved models exist"""
        if not self.model_path:
            return False
            
        try:
            joblib.load(f"{self.model_path}/rf_model.joblib")
            joblib.load(f"{self.model_path}/gb_model.joblib")
            joblib.load(f"{self.model_path}/scaler.joblib")
            joblib.load(f"{self.model_path}/label_encoder.joblib")
            return True
        except:
            return False
            
    def _save_models(self):
        """Save models to disk"""
        if not self.model_path:
            return
            
        joblib.dump(self.rf_model, f"{self.model_path}/rf_model.joblib")
        joblib.dump(self.gb_model, f"{self.model_path}/gb_model.joblib")
        joblib.dump(self.scaler, f"{self.model_path}/scaler.joblib")
        joblib.dump(
            self.label_encoder,
            f"{self.model_path}/label_encoder.joblib"
        )
        
    def _load_models(self):
        """Load models from disk"""
        if not self.model_path:
            return
            
        self.rf_model = joblib.load(f"{self.model_path}/rf_model.joblib")
        self.gb_model = joblib.load(f"{self.model_path}/gb_model.joblib")
        self.scaler = joblib.load(f"{self.model_path}/scaler.joblib")
        self.label_encoder = joblib.load(
            f"{self.model_path}/label_encoder.joblib"
        )
        
    def cleanup(self):
        """Cleanup resources and save models"""
        try:
            if self.model_path:
                self._save_models()
                
            # Clear memory
            self.prediction_accuracy.clear()
            self.confidence_scores.clear()
            self.feature_importance.clear()
            
            # Release model resources
            self.rf_model = None
            self.gb_model = None
            self.scaler = None
            self.label_encoder = None
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
    def __del__(self):
        """Ensure cleanup on destruction"""
        try:
            self.cleanup()
        except:
            pass
