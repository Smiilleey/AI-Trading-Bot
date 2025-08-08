# core/regime_detection.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from collections import defaultdict

@dataclass
class MarketRegime:
    regime_type: str
    confidence: float
    features: Dict
    transitions: Dict
    duration: int
    strength: float
    timestamp: datetime

class RegimeDetectionEngine:
    """
    Advanced regime detection engine:
    - Multi-factor analysis
    - Transition prediction
    - Adaptive strategies
    - Real-time monitoring
    """
    def __init__(
        self,
        lookback_window: int = 100,
        transition_threshold: float = 0.7,
        config: Optional[Dict] = None
    ):
        self.lookback_window = lookback_window
        self.transition_threshold = transition_threshold
        self.config = config or {}
        
        # Initialize models
        self.gmm = GaussianMixture(
            n_components=4,
            random_state=42
        )
        self.kmeans = KMeans(
            n_clusters=4,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # State tracking
        self.current_regime = None
        self.regime_history = []
        self.transition_matrix = None
        self.feature_importance = {}
        
    def detect_regime(
        self,
        market_data: Dict,
        additional_features: Optional[Dict] = None
    ) -> MarketRegime:
        """
        Detect current market regime:
        - Feature extraction
        - Model combination
        - Confidence scoring
        """
        # Extract features
        features = self._extract_features(
            market_data,
            additional_features
        )
        
        # Detect regime
        regime = self._detect_current_regime(features)
        
        # Calculate confidence
        confidence = self._calculate_regime_confidence(
            regime,
            features
        )
        
        # Analyze transitions
        transitions = self._analyze_regime_transitions(
            regime,
            features
        )
        
        # Create regime object
        current_regime = MarketRegime(
            regime_type=regime["type"],
            confidence=confidence,
            features=features,
            transitions=transitions,
            duration=regime["duration"],
            strength=regime["strength"],
            timestamp=datetime.now()
        )
        
        # Update state
        self._update_state(current_regime)
        
        return current_regime
        
    def predict_transition(
        self,
        current_regime: MarketRegime,
        horizon: int = 1
    ) -> Dict:
        """
        Predict regime transitions:
        - Transition probability
        - Next regime
        - Timing estimation
        """
        # Calculate transition probabilities
        probabilities = self._calculate_transition_probabilities(
            current_regime
        )
        
        # Predict next regime
        next_regime = self._predict_next_regime(
            current_regime,
            probabilities
        )
        
        # Estimate timing
        timing = self._estimate_transition_timing(
            current_regime,
            next_regime,
            horizon
        )
        
        return {
            "probabilities": probabilities,
            "next_regime": next_regime,
            "timing": timing
        }
        
    def get_regime_strategies(
        self,
        regime: MarketRegime
    ) -> Dict:
        """
        Get regime-specific strategies:
        - Strategy selection
        - Parameter adaptation
        - Risk adjustment
        """
        # Select strategies
        strategies = self._select_regime_strategies(regime)
        
        # Adapt parameters
        parameters = self._adapt_strategy_parameters(
            strategies,
            regime
        )
        
        # Adjust risk
        risk = self._adjust_risk_parameters(
            parameters,
            regime
        )
        
        return {
            "strategies": strategies,
            "parameters": parameters,
            "risk": risk
        }
        
    def _extract_features(
        self,
        market_data: Dict,
        additional_features: Optional[Dict]
    ) -> Dict:
        """
        Extract regime features:
        - Technical features
        - Volatility features
        - Flow features
        """
        features = {}
        
        # Extract price features
        price_features = self._extract_price_features(
            market_data
        )
        features["price"] = price_features
        
        # Extract volatility features
        vol_features = self._extract_volatility_features(
            market_data
        )
        features["volatility"] = vol_features
        
        # Extract volume features
        volume_features = self._extract_volume_features(
            market_data
        )
        features["volume"] = volume_features
        
        # Extract correlation features
        corr_features = self._extract_correlation_features(
            market_data
        )
        features["correlation"] = corr_features
        
        # Add additional features
        if additional_features:
            features.update(additional_features)
            
        return features
        
    def _detect_current_regime(
        self,
        features: Dict
    ) -> Dict:
        """
        Detect current regime:
        - Model combination
        - Regime classification
        - Duration analysis
        """
        # Prepare feature matrix
        X = self._prepare_feature_matrix(features)
        
        # Get GMM prediction
        gmm_regime = self._get_gmm_prediction(X)
        
        # Get KMeans prediction
        kmeans_regime = self._get_kmeans_prediction(X)
        
        # Combine predictions
        regime = self._combine_predictions(
            gmm_regime,
            kmeans_regime,
            features
        )
        
        # Calculate regime strength
        strength = self._calculate_regime_strength(
            regime,
            features
        )
        
        # Calculate duration
        duration = self._calculate_regime_duration(regime)
        
        return {
            "type": regime["type"],
            "probability": regime["probability"],
            "strength": strength,
            "duration": duration
        }
        
    def _calculate_regime_confidence(
        self,
        regime: Dict,
        features: Dict
    ) -> float:
        """
        Calculate regime confidence:
        - Model confidence
        - Feature stability
        - Historical accuracy
        """
        # Calculate model confidence
        model_confidence = regime["probability"]
        
        # Calculate feature stability
        feature_stability = self._calculate_feature_stability(
            features
        )
        
        # Calculate historical accuracy
        historical_accuracy = self._calculate_historical_accuracy(
            regime["type"]
        )
        
        # Combine confidence scores
        confidence = (
            model_confidence * 0.4 +
            feature_stability * 0.3 +
            historical_accuracy * 0.3
        )
        
        return confidence
        
    def _analyze_regime_transitions(
        self,
        regime: Dict,
        features: Dict
    ) -> Dict:
        """
        Analyze regime transitions:
        - Transition probability
        - Trigger analysis
        - Path analysis
        """
        # Calculate transition matrix
        if self.transition_matrix is None:
            self.transition_matrix = self._calculate_transition_matrix()
            
        # Analyze triggers
        triggers = self._analyze_transition_triggers(
            regime,
            features
        )
        
        # Analyze paths
        paths = self._analyze_transition_paths(
            regime,
            features
        )
        
        return {
            "matrix": self.transition_matrix,
            "triggers": triggers,
            "paths": paths
        }
        
    def _update_state(
        self,
        regime: MarketRegime
    ):
        """
        Update internal state:
        - History update
        - Model update
        - Feature update
        """
        # Update regime history
        self.regime_history.append(regime)
        
        # Update current regime
        self.current_regime = regime
        
        # Update transition matrix
        self._update_transition_matrix(regime)
        
        # Update feature importance
        self._update_feature_importance(regime)
        
    def _calculate_transition_probabilities(
        self,
        regime: MarketRegime
    ) -> Dict:
        """
        Calculate transition probabilities:
        - Base probability
        - Feature influence
        - Time influence
        """
        probabilities = {}
        
        # Get base probabilities
        base_probs = self._get_base_probabilities(regime)
        
        # Adjust for features
        feature_probs = self._adjust_for_features(
            base_probs,
            regime.features
        )
        
        # Adjust for time
        time_probs = self._adjust_for_time(
            feature_probs,
            regime.duration
        )
        
        # Calculate final probabilities
        for next_regime in self.config.get("regimes", []):
            probabilities[next_regime] = time_probs.get(
                next_regime,
                0.0
            )
            
        return probabilities
        
    def _predict_next_regime(
        self,
        current_regime: MarketRegime,
        probabilities: Dict
    ) -> Dict:
        """
        Predict next regime:
        - Most likely regime
        - Transition path
        - Confidence score
        """
        # Get most likely regime
        next_regime = max(
            probabilities.items(),
            key=lambda x: x[1]
        )
        
        # Calculate transition path
        path = self._calculate_transition_path(
            current_regime,
            next_regime[0]
        )
        
        # Calculate confidence
        confidence = self._calculate_transition_confidence(
            next_regime[1],
            path
        )
        
        return {
            "regime": next_regime[0],
            "probability": next_regime[1],
            "path": path,
            "confidence": confidence
        }
        
    def _estimate_transition_timing(
        self,
        current_regime: MarketRegime,
        next_regime: Dict,
        horizon: int
    ) -> Dict:
        """
        Estimate transition timing:
        - Time to transition
        - Transition window
        - Probability path
        """
        # Calculate base timing
        base_timing = self._calculate_base_timing(
            current_regime,
            next_regime
        )
        
        # Adjust for horizon
        adjusted_timing = self._adjust_timing_for_horizon(
            base_timing,
            horizon
        )
        
        # Calculate probability path
        prob_path = self._calculate_probability_path(
            current_regime,
            next_regime,
            adjusted_timing
        )
        
        return {
            "expected_time": adjusted_timing["expected"],
            "window": adjusted_timing["window"],
            "probability_path": prob_path
        }
        
    def _select_regime_strategies(
        self,
        regime: MarketRegime
    ) -> List[Dict]:
        """
        Select regime-specific strategies:
        - Strategy filtering
        - Performance analysis
        - Combination analysis
        """
        strategies = []
        
        # Get available strategies
        available = self.config.get("strategies", {})
        
        # Filter by regime
        regime_strategies = self._filter_strategies(
            available,
            regime
        )
        
        # Analyze performance
        performance = self._analyze_strategy_performance(
            regime_strategies,
            regime
        )
        
        # Select best combination
        selected = self._select_strategy_combination(
            performance,
            regime
        )
        
        return selected
        
    def _adapt_strategy_parameters(
        self,
        strategies: List[Dict],
        regime: MarketRegime
    ) -> Dict:
        """
        Adapt strategy parameters:
        - Parameter optimization
        - Regime adjustment
        - Risk scaling
        """
        parameters = {}
        
        for strategy in strategies:
            # Get base parameters
            base_params = strategy.get("parameters", {})
            
            # Optimize for regime
            optimized = self._optimize_parameters(
                base_params,
                regime
            )
            
            # Adjust for regime
            adjusted = self._adjust_parameters(
                optimized,
                regime
            )
            
            parameters[strategy["name"]] = adjusted
            
        return parameters
        
    def _adjust_risk_parameters(
        self,
        parameters: Dict,
        regime: MarketRegime
    ) -> Dict:
        """
        Adjust risk parameters:
        - Position sizing
        - Stop levels
        - Exposure limits
        """
        risk_params = {}
        
        for strategy, params in parameters.items():
            # Adjust position sizing
            position_size = self._adjust_position_size(
                params,
                regime
            )
            
            # Adjust stop levels
            stop_levels = self._adjust_stop_levels(
                params,
                regime
            )
            
            # Adjust exposure
            exposure = self._adjust_exposure_limits(
                params,
                regime
            )
            
            risk_params[strategy] = {
                "position_size": position_size,
                "stop_levels": stop_levels,
                "exposure": exposure
            }
            
        return risk_params
        
    def _extract_price_features(
        self,
        market_data: Dict
    ) -> Dict:
        """Extract price-based features"""
        # Implementation details...
        pass
        
    def _extract_volatility_features(
        self,
        market_data: Dict
    ) -> Dict:
        """Extract volatility-based features"""
        # Implementation details...
        pass
        
    def _extract_volume_features(
        self,
        market_data: Dict
    ) -> Dict:
        """Extract volume-based features"""
        # Implementation details...
        pass
        
    def _extract_correlation_features(
        self,
        market_data: Dict
    ) -> Dict:
        """Extract correlation-based features"""
        # Implementation details...
        pass
        
    def _prepare_feature_matrix(
        self,
        features: Dict
    ) -> np.ndarray:
        """Prepare feature matrix for models"""
        # Implementation details...
        pass
        
    def _get_gmm_prediction(
        self,
        X: np.ndarray
    ) -> Dict:
        """Get GMM model prediction"""
        # Implementation details...
        pass
        
    def _get_kmeans_prediction(
        self,
        X: np.ndarray
    ) -> Dict:
        """Get KMeans model prediction"""
        # Implementation details...
        pass
        
    def _combine_predictions(
        self,
        gmm_regime: Dict,
        kmeans_regime: Dict,
        features: Dict
    ) -> Dict:
        """Combine model predictions"""
        # Implementation details...
        pass
        
    def _calculate_regime_strength(
        self,
        regime: Dict,
        features: Dict
    ) -> float:
        """Calculate regime strength"""
        # Implementation details...
        pass
        
    def _calculate_regime_duration(
        self,
        regime: Dict
    ) -> int:
        """Calculate regime duration"""
        # Implementation details...
        pass
        
    def _calculate_feature_stability(
        self,
        features: Dict
    ) -> float:
        """Calculate feature stability"""
        # Implementation details...
        pass
        
    def _calculate_historical_accuracy(
        self,
        regime_type: str
    ) -> float:
        """Calculate historical accuracy"""
        # Implementation details...
        pass
        
    def _calculate_transition_matrix(self) -> np.ndarray:
        """Calculate regime transition matrix"""
        # Implementation details...
        pass
        
    def _analyze_transition_triggers(
        self,
        regime: Dict,
        features: Dict
    ) -> List[Dict]:
        """Analyze transition triggers"""
        # Implementation details...
        pass
        
    def _analyze_transition_paths(
        self,
        regime: Dict,
        features: Dict
    ) -> List[Dict]:
        """Analyze transition paths"""
        # Implementation details...
        pass
        
    def _update_transition_matrix(
        self,
        regime: MarketRegime
    ):
        """Update transition matrix"""
        # Implementation details...
        pass
        
    def _update_feature_importance(
        self,
        regime: MarketRegime
    ):
        """Update feature importance"""
        # Implementation details...
        pass
        
    def _get_base_probabilities(
        self,
        regime: MarketRegime
    ) -> Dict:
        """Get base transition probabilities"""
        # Implementation details...
        pass
        
    def _adjust_for_features(
        self,
        probabilities: Dict,
        features: Dict
    ) -> Dict:
        """Adjust probabilities for features"""
        # Implementation details...
        pass
        
    def _adjust_for_time(
        self,
        probabilities: Dict,
        duration: int
    ) -> Dict:
        """Adjust probabilities for time"""
        # Implementation details...
        pass
        
    def _calculate_transition_path(
        self,
        current_regime: MarketRegime,
        next_regime: str
    ) -> List[Dict]:
        """Calculate transition path"""
        # Implementation details...
        pass
        
    def _calculate_transition_confidence(
        self,
        probability: float,
        path: List[Dict]
    ) -> float:
        """Calculate transition confidence"""
        # Implementation details...
        pass
        
    def _calculate_base_timing(
        self,
        current_regime: MarketRegime,
        next_regime: Dict
    ) -> Dict:
        """Calculate base transition timing"""
        # Implementation details...
        pass
        
    def _adjust_timing_for_horizon(
        self,
        timing: Dict,
        horizon: int
    ) -> Dict:
        """Adjust timing for horizon"""
        # Implementation details...
        pass
        
    def _calculate_probability_path(
        self,
        current_regime: MarketRegime,
        next_regime: Dict,
        timing: Dict
    ) -> List[Dict]:
        """Calculate probability path"""
        # Implementation details...
        pass
        
    def _filter_strategies(
        self,
        strategies: Dict,
        regime: MarketRegime
    ) -> List[Dict]:
        """Filter strategies for regime"""
        # Implementation details...
        pass
        
    def _analyze_strategy_performance(
        self,
        strategies: List[Dict],
        regime: MarketRegime
    ) -> Dict:
        """Analyze strategy performance"""
        # Implementation details...
        pass
        
    def _select_strategy_combination(
        self,
        performance: Dict,
        regime: MarketRegime
    ) -> List[Dict]:
        """Select optimal strategy combination"""
        # Implementation details...
        pass
        
    def _optimize_parameters(
        self,
        parameters: Dict,
        regime: MarketRegime
    ) -> Dict:
        """Optimize strategy parameters"""
        # Implementation details...
        pass
        
    def _adjust_parameters(
        self,
        parameters: Dict,
        regime: MarketRegime
    ) -> Dict:
        """Adjust parameters for regime"""
        # Implementation details...
        pass
        
    def _adjust_position_size(
        self,
        parameters: Dict,
        regime: MarketRegime
    ) -> Dict:
        """Adjust position sizing"""
        # Implementation details...
        pass
        
    def _adjust_stop_levels(
        self,
        parameters: Dict,
        regime: MarketRegime
    ) -> Dict:
        """Adjust stop levels"""
        # Implementation details...
        pass
        
    def _adjust_exposure_limits(
        self,
        parameters: Dict,
        regime: MarketRegime
    ) -> Dict:
        """Adjust exposure limits"""
        # Implementation details...
        pass
