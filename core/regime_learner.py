import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import joblib
import os
from collections import defaultdict, deque

class RegimeLearner:
    """
    Advanced regime learning system that:
    1. Classifies market regimes (quiet, normal, trending, volatile)
    2. Learns optimal entry/exit thresholds for each regime
    3. Adapts thresholds based on reward feedback
    4. Integrates with IPDA phases and top-down analysis
    
    Key Features:
    - Multi-dimensional regime classification
    - Dynamic threshold optimization
    - Reward-based learning
    - Regime transition detection
    - Performance tracking per regime
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Regime classification models
        self.regime_classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        
        # Threshold optimization models (one per regime)
        self.threshold_models = {
            'quiet': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'normal': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'trending': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'volatile': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        
        # Regime transition detector
        self.transition_detector = KMeans(n_clusters=4, random_state=42)
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Regime-specific thresholds (learned)
        self.regime_thresholds = {
            'quiet': {
                'entry_confidence': 0.7,
                'exit_confidence': 0.5,
                'stop_multiplier': 1.2,
                'take_profit_multiplier': 1.5,
                'max_risk_per_trade': 0.01
            },
            'normal': {
                'entry_confidence': 0.6,
                'exit_confidence': 0.4,
                'stop_multiplier': 1.0,
                'take_profit_multiplier': 2.0,
                'max_risk_per_trade': 0.02
            },
            'trending': {
                'entry_confidence': 0.5,
                'exit_confidence': 0.3,
                'stop_multiplier': 0.8,
                'take_profit_multiplier': 3.0,
                'max_risk_per_trade': 0.03
            },
            'volatile': {
                'entry_confidence': 0.8,
                'exit_confidence': 0.6,
                'stop_multiplier': 1.5,
                'take_profit_multiplier': 1.2,
                'max_risk_per_trade': 0.015
            }
        }
        
        # Performance tracking per regime
        self.regime_performance = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'threshold_updates': 0
        })
        
        # Learning parameters
        self.learning_rate = config.get('regime_learning_rate', 0.01)
        self.min_samples_per_regime = config.get('min_samples_per_regime', 100)
        self.threshold_update_frequency = config.get('threshold_update_frequency', 50)
        
        # Memory for recent data
        self.recent_data = deque(maxlen=1000)
        self.recent_outcomes = deque(maxlen=1000)
        
        # Model persistence
        self.model_dir = "memory/regime_learner"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.is_trained = False
        
    def classify_regime(self, market_data: Dict, symbol: str = "") -> Dict[str, Any]:
        """
        Classify current market regime using multiple features.
        
        Args:
            market_data: Dictionary containing price, volume, and other market data
            symbol: Trading symbol
            
        Returns:
            Dictionary with regime classification and confidence
        """
        try:
            # Extract features for regime classification
            features = self._extract_regime_features(market_data, symbol)
            
            if not features:
                return {'regime': 'normal', 'confidence': 0.5, 'features': {}}
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict regime
            if self.is_trained:
                regime_proba = self.regime_classifier.predict_proba(features_scaled)[0]
                regime_classes = self.regime_classifier.classes_
                
                # Get most likely regime
                regime_idx = np.argmax(regime_proba)
                predicted_regime = regime_classes[regime_idx]
                confidence = regime_proba[regime_idx]
            else:
                # Fallback to rule-based classification
                predicted_regime, confidence = self._rule_based_regime_classification(features)
            
            # Detect regime transitions
            transition_info = self._detect_regime_transition(predicted_regime, features)
            
            return {
                'regime': predicted_regime,
                'confidence': confidence,
                'features': features,
                'transition': transition_info,
                'all_probabilities': dict(zip(regime_classes, regime_proba)) if self.is_trained else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error in regime classification: {e}")
            return {'regime': 'normal', 'confidence': 0.5, 'features': {}}
    
    def get_optimal_thresholds(self, regime: str, market_context: Dict = None) -> Dict[str, float]:
        """
        Get optimal entry/exit thresholds for the given regime.
        
        Args:
            regime: Market regime ('quiet', 'normal', 'trending', 'volatile')
            market_context: Additional market context for threshold adjustment
            
        Returns:
            Dictionary with optimal thresholds
        """
        try:
            # Get base thresholds for regime
            thresholds = self.regime_thresholds.get(regime, self.regime_thresholds['normal']).copy()
            
            # Apply learned adjustments if model is trained
            if self.is_trained and regime in self.threshold_models:
                if market_context:
                    # Extract context features for threshold adjustment
                    context_features = self._extract_context_features(market_context)
                    if context_features:
                        context_scaled = self.scaler.transform([context_features])
                        
                        # Get threshold adjustments from model
                        adjustments = self.threshold_models[regime].predict(context_scaled)[0]
                        
                        # Apply adjustments (assuming 5 threshold values)
                        if len(adjustments) >= 5:
                            thresholds['entry_confidence'] = max(0.1, min(0.9, 
                                thresholds['entry_confidence'] + adjustments[0] * 0.1))
                            thresholds['exit_confidence'] = max(0.1, min(0.9, 
                                thresholds['exit_confidence'] + adjustments[1] * 0.1))
                            thresholds['stop_multiplier'] = max(0.5, min(2.0, 
                                thresholds['stop_multiplier'] + adjustments[2] * 0.2))
                            thresholds['take_profit_multiplier'] = max(0.5, min(5.0, 
                                thresholds['take_profit_multiplier'] + adjustments[3] * 0.5))
                            thresholds['max_risk_per_trade'] = max(0.005, min(0.05, 
                                thresholds['max_risk_per_trade'] + adjustments[4] * 0.01))
            
            # Add regime-specific metadata
            thresholds['regime'] = regime
            thresholds['performance'] = self.regime_performance[regime]
            
            return thresholds
            
        except Exception as e:
            self.logger.error(f"Error getting optimal thresholds: {e}")
            return self.regime_thresholds.get(regime, self.regime_thresholds['normal'])
    
    def update_with_outcome(self, regime: str, trade_outcome: Dict, market_context: Dict = None):
        """
        Update regime learning with trade outcome.
        
        Args:
            regime: Market regime when trade was taken
            trade_outcome: Dictionary with trade results (pnl, rr, duration, etc.)
            market_context: Market context at time of trade
        """
        try:
            # Update performance tracking
            self._update_regime_performance(regime, trade_outcome)
            
            # Store data for learning
            if market_context:
                self.recent_data.append({
                    'regime': regime,
                    'context': market_context,
                    'outcome': trade_outcome,
                    'timestamp': pd.Timestamp.now()
                })
            
            # Trigger threshold update if enough samples
            if (self.regime_performance[regime]['total_trades'] % 
                self.threshold_update_frequency == 0):
                self._update_regime_thresholds(regime)
            
        except Exception as e:
            self.logger.error(f"Error updating regime with outcome: {e}")
    
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train regime classification and threshold optimization models.
        
        Args:
            training_data: DataFrame with features and outcomes
            
        Returns:
            Dictionary with training performance metrics
        """
        try:
            self.logger.info("Training regime learning models...")
            
            # Prepare features and targets
            feature_cols = [col for col in training_data.columns 
                          if col not in ['regime', 'outcome', 'timestamp', 'symbol']]
            X = training_data[feature_cols].fillna(0)
            y_regime = training_data['regime']
            y_outcome = training_data['outcome']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_regime_train, y_regime_test = train_test_split(
                X_scaled, y_regime, test_size=0.2, random_state=42, stratify=y_regime
            )
            
            # Train regime classifier
            self.regime_classifier.fit(X_train, y_regime_train)
            regime_score = self.regime_classifier.score(X_test, y_regime_test)
            
            # Train threshold models for each regime
            threshold_scores = {}
            for regime in self.threshold_models.keys():
                regime_mask = y_regime == regime
                if np.sum(regime_mask) > self.min_samples_per_regime:
                    X_regime = X_scaled[regime_mask]
                    y_regime_outcome = y_outcome[regime_mask]
                    
                    # Create threshold features (regime + context)
                    threshold_features = self._create_threshold_features(X_regime, regime)
                    
                    if len(threshold_features) > 10:  # Minimum samples for training
                        self.threshold_models[regime].fit(threshold_features, y_regime_outcome)
                        
                        # Evaluate threshold model
                        regime_test_mask = y_regime_test == regime
                        if np.sum(regime_test_mask) > 5:
                            X_regime_test = X_test[regime_test_mask]
                            threshold_test_features = self._create_threshold_features(X_regime_test, regime)
                            y_regime_test_outcome = y_outcome[regime_test_mask]
                            
                            score = self.threshold_models[regime].score(threshold_test_features, y_regime_test_outcome)
                            threshold_scores[regime] = score
                        else:
                            threshold_scores[regime] = 0.0
                    else:
                        threshold_scores[regime] = 0.0
                else:
                    threshold_scores[regime] = 0.0
            
            # Save models
            self._save_models()
            self.is_trained = True
            
            return {
                'regime_classifier_score': regime_score,
                'threshold_scores': threshold_scores,
                'overall_score': np.mean([regime_score] + list(threshold_scores.values())),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            self.logger.error(f"Error training regime models: {e}")
            return {'error': str(e)}
    
    def _extract_regime_features(self, market_data: Dict, symbol: str) -> List[float]:
        """Extract features for regime classification."""
        try:
            features = []
            
            # Price data
            candles = market_data.get('candles', [])
            if not candles or len(candles) < 20:
                return []
            
            prices = [float(c.get('close', 0)) for c in candles[-20:]]
            volumes = [float(c.get('volume', 0)) for c in candles[-20:]]
            highs = [float(c.get('high', 0)) for c in candles[-20:]]
            lows = [float(c.get('low', 0)) for c in candles[-20:]]
            
            # Price-based features
            features.extend([
                np.std(prices),  # Volatility
                np.mean(np.abs(np.diff(prices))),  # Average price change
                (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0,  # Total return
                np.corrcoef(range(len(prices)), prices)[0, 1] if len(prices) > 1 else 0,  # Trend strength
            ])
            
            # Volume-based features
            if volumes and any(v > 0 for v in volumes):
                features.extend([
                    np.mean(volumes),  # Average volume
                    np.std(volumes),   # Volume volatility
                    np.corrcoef(volumes, prices)[0, 1] if len(volumes) == len(prices) else 0,  # Volume-price correlation
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Range-based features
            ranges = [h - l for h, l in zip(highs, lows)]
            features.extend([
                np.mean(ranges),  # Average range
                np.std(ranges),   # Range volatility
                np.max(ranges) / np.mean(ranges) if np.mean(ranges) > 0 else 0,  # Range expansion
            ])
            
            # Time-based features
            current_hour = pd.Timestamp.now().hour
            features.extend([
                float(current_hour),  # Hour of day
                float(pd.Timestamp.now().dayofweek),  # Day of week
                1.0 if 7 <= current_hour <= 16 else 0.0,  # London session
                1.0 if 13 <= current_hour <= 22 else 0.0,  # New York session
            ])
            
            # Order flow features (if available)
            of_data = market_data.get('order_flow', {})
            features.extend([
                of_data.get('delta_momentum', 0.0),
                of_data.get('absorption_strength', 0.0),
                of_data.get('institutional_pressure', 0.0),
            ])
            
            # CISD features (if available)
            cisd_data = market_data.get('cisd', {})
            features.extend([
                cisd_data.get('score', 0.0),
                cisd_data.get('confidence', 0.0),
            ])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting regime features: {e}")
            return []
    
    def _rule_based_regime_classification(self, features: List[float]) -> Tuple[str, float]:
        """Fallback rule-based regime classification."""
        if len(features) < 10:
            return 'normal', 0.5
        
        volatility = features[0] if len(features) > 0 else 0.0
        trend_strength = abs(features[3]) if len(features) > 3 else 0.0
        volume_volatility = features[5] if len(features) > 5 else 0.0
        
        # Simple rule-based classification
        if volatility < 0.001 and trend_strength < 0.1:
            return 'quiet', 0.7
        elif volatility > 0.005 or volume_volatility > 1000:
            return 'volatile', 0.7
        elif trend_strength > 0.3:
            return 'trending', 0.7
        else:
            return 'normal', 0.6
    
    def _detect_regime_transition(self, current_regime: str, features: List[float]) -> Dict[str, Any]:
        """Detect if market is transitioning between regimes."""
        try:
            if len(self.recent_data) < 10:
                return {'transition': False, 'confidence': 0.0}
            
            # Get recent regime history
            recent_regimes = [data['regime'] for data in list(self.recent_data)[-10:]]
            
            # Check for regime stability
            regime_counts = {}
            for regime in recent_regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            most_common_regime = max(regime_counts, key=regime_counts.get)
            stability = regime_counts[most_common_regime] / len(recent_regimes)
            
            # Detect transition
            transition = (most_common_regime != current_regime) and (stability < 0.7)
            
            return {
                'transition': transition,
                'confidence': 1.0 - stability,
                'from_regime': most_common_regime,
                'to_regime': current_regime,
                'stability': stability
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting regime transition: {e}")
            return {'transition': False, 'confidence': 0.0}
    
    def _extract_context_features(self, market_context: Dict) -> List[float]:
        """Extract features from market context for threshold adjustment."""
        features = []
        
        # Basic context features
        features.extend([
            market_context.get('volatility', 0.0),
            market_context.get('trend_strength', 0.0),
            market_context.get('liquidity', 0.0),
            market_context.get('depth', 0.0),
            market_context.get('stress', 0.0),
        ])
        
        # Session features
        session = market_context.get('session', 'unknown')
        session_encoding = {
            'london': [1.0, 0.0, 0.0, 0.0],
            'newyork': [0.0, 1.0, 0.0, 0.0],
            'tokyo': [0.0, 0.0, 1.0, 0.0],
            'overlap': [0.0, 0.0, 0.0, 1.0]
        }
        features.extend(session_encoding.get(session, [0.0, 0.0, 0.0, 0.0]))
        
        return features
    
    def _create_threshold_features(self, X: np.ndarray, regime: str) -> np.ndarray:
        """Create features for threshold optimization model."""
        # Add regime encoding
        regime_encoding = {
            'quiet': [1.0, 0.0, 0.0, 0.0],
            'normal': [0.0, 1.0, 0.0, 0.0],
            'trending': [0.0, 0.0, 1.0, 0.0],
            'volatile': [0.0, 0.0, 0.0, 1.0]
        }
        
        regime_vec = np.array(regime_encoding.get(regime, [0.0, 0.0, 0.0, 0.0]))
        regime_matrix = np.tile(regime_vec, (X.shape[0], 1))
        
        # Combine original features with regime encoding
        return np.hstack([X, regime_matrix])
    
    def _update_regime_performance(self, regime: str, trade_outcome: Dict):
        """Update performance metrics for a regime."""
        perf = self.regime_performance[regime]
        
        pnl = trade_outcome.get('pnl', 0.0)
        perf['total_trades'] += 1
        perf['total_pnl'] += pnl
        
        if pnl > 0:
            perf['winning_trades'] += 1
        
        perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
        perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
    
    def _update_regime_thresholds(self, regime: str):
        """Update thresholds for a regime based on recent performance."""
        try:
            perf = self.regime_performance[regime]
            
            if perf['total_trades'] < 10:
                return
            
            # Adjust thresholds based on performance
            win_rate = perf['win_rate']
            avg_pnl = perf['avg_pnl']
            
            # Adjust entry confidence based on win rate
            if win_rate > 0.6:
                # High win rate - can be more aggressive
                self.regime_thresholds[regime]['entry_confidence'] *= 0.95
            elif win_rate < 0.4:
                # Low win rate - be more conservative
                self.regime_thresholds[regime]['entry_confidence'] *= 1.05
            
            # Adjust stop multiplier based on average PnL
            if avg_pnl > 0:
                # Profitable - can use tighter stops
                self.regime_thresholds[regime]['stop_multiplier'] *= 0.98
            else:
                # Losing - need wider stops
                self.regime_thresholds[regime]['stop_multiplier'] *= 1.02
            
            # Ensure thresholds stay within reasonable bounds
            self.regime_thresholds[regime]['entry_confidence'] = max(0.1, min(0.9, 
                self.regime_thresholds[regime]['entry_confidence']))
            self.regime_thresholds[regime]['stop_multiplier'] = max(0.5, min(2.0, 
                self.regime_thresholds[regime]['stop_multiplier']))
            
            perf['threshold_updates'] += 1
            
            self.logger.info(f"Updated thresholds for {regime} regime: {self.regime_thresholds[regime]}")
            
        except Exception as e:
            self.logger.error(f"Error updating regime thresholds: {e}")
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            # Save regime classifier
            classifier_path = os.path.join(self.model_dir, "regime_classifier.joblib")
            joblib.dump(self.regime_classifier, classifier_path)
            
            # Save threshold models
            for regime, model in self.threshold_models.items():
                model_path = os.path.join(self.model_dir, f"threshold_model_{regime}.joblib")
                joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            # Save thresholds
            thresholds_path = os.path.join(self.model_dir, "regime_thresholds.json")
            import json
            with open(thresholds_path, 'w') as f:
                json.dump(self.regime_thresholds, f, indent=2)
            
            self.logger.info("Regime learner models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving regime models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk."""
        try:
            # Load regime classifier
            classifier_path = os.path.join(self.model_dir, "regime_classifier.joblib")
            if os.path.exists(classifier_path):
                self.regime_classifier = joblib.load(classifier_path)
            
            # Load threshold models
            for regime in self.threshold_models.keys():
                model_path = os.path.join(self.model_dir, f"threshold_model_{regime}.joblib")
                if os.path.exists(model_path):
                    self.threshold_models[regime] = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load thresholds
            thresholds_path = os.path.join(self.model_dir, "regime_thresholds.json")
            if os.path.exists(thresholds_path):
                import json
                with open(thresholds_path, 'r') as f:
                    self.regime_thresholds = json.load(f)
            
            self.is_trained = True
            self.logger.info("Regime learner models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading regime models: {e}")
