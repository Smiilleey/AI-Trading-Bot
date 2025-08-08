# core/market_regime.py

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class MarketState:
    regime: str  # trending, ranging, volatile
    trend_direction: str  # bullish, bearish, neutral
    volatility_state: str  # low, normal, high
    momentum_state: str  # accelerating, decelerating, neutral
    liquidity_state: str  # thin, normal, deep
    confidence: float  # 0 to 1
    timestamp: datetime

class MarketRegimeAnalyzer:
    """
    Advanced market regime detection:
    - Automatic regime classification
    - Volatility state tracking
    - Momentum analysis
    - Liquidity profiling
    - Regime transitions
    - Adaptive timeframe selection
    """
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        regime_window: int = 100,
        n_regimes: int = 3
    ):
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.regime_window = regime_window
        self.n_regimes = n_regimes
        
        # State tracking
        self.current_regime = None
        self.regime_history = []
        self.volatility_history = []
        self.regime_transitions = []
        
        # ML components
        self.scaler = StandardScaler()
        self.regime_classifier = KMeans(n_clusters=n_regimes)
        
        # Adaptive parameters
        self.volatility_thresholds = {
            "low": 0.0,
            "normal": 0.0,
            "high": 0.0
        }
        self.momentum_thresholds = {
            "weak": 0.0,
            "normal": 0.0,
            "strong": 0.0
        }
        
    def update(self, candles: List[Dict]) -> MarketState:
        """
        Update market regime analysis with new candles
        Returns current market state with regime classification
        """
        if len(candles) < self.regime_window:
            return self._default_state()
            
        # Extract price data
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        volumes = np.array([c.get("tick_volume", 0) for c in candles])
        
        # Calculate key metrics
        volatility = self._calculate_volatility(closes, highs, lows)
        trend = self._analyze_trend(closes)
        momentum = self._calculate_momentum(closes)
        liquidity = self._analyze_liquidity(volumes)
        
        # Detect regime
        regime = self._classify_regime(closes, volatility, momentum)
        
        # Update state
        state = MarketState(
            regime=regime["regime"],
            trend_direction=trend["direction"],
            volatility_state=volatility["state"],
            momentum_state=momentum["state"],
            liquidity_state=liquidity["state"],
            confidence=regime["confidence"],
            timestamp=datetime.utcnow()
        )
        
        self._update_history(state)
        return state
        
    def _calculate_volatility(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Dict:
        """
        Calculate multiple volatility metrics:
        - Standard deviation
        - Average True Range
        - Parkinson volatility
        - Garman-Klass volatility
        """
        # Standard deviation of returns
        returns = np.diff(np.log(closes))
        std_vol = np.std(returns[-self.volatility_window:]) * np.sqrt(252)
        
        # Average True Range
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))
        ranges = np.vstack([high_low, high_close, low_close])
        true_range = np.max(ranges, axis=0)
        atr = np.mean(true_range[-self.volatility_window:])
        
        # Parkinson volatility
        hl_vol = np.sqrt(
            1 / (4 * np.log(2)) *
            np.mean(np.log(highs/lows)**2) *
            252
        )
        
        # Determine volatility state
        current_vol = (std_vol + hl_vol) / 2
        self.volatility_history.append(current_vol)
        
        if len(self.volatility_history) > self.volatility_window:
            vol_mean = np.mean(self.volatility_history[-self.volatility_window:])
            vol_std = np.std(self.volatility_history[-self.volatility_window:])
            
            self.volatility_thresholds = {
                "low": vol_mean - vol_std,
                "normal": vol_mean,
                "high": vol_mean + vol_std
            }
        
        state = "normal"
        if current_vol > self.volatility_thresholds["high"]:
            state = "high"
        elif current_vol < self.volatility_thresholds["low"]:
            state = "low"
            
        return {
            "value": current_vol,
            "atr": atr,
            "state": state,
            "std_vol": std_vol,
            "parkinson_vol": hl_vol
        }
        
    def _analyze_trend(self, closes: np.ndarray) -> Dict:
        """
        Advanced trend analysis:
        - Linear regression
        - Moving average convergence
        - Price momentum
        - Trend strength
        """
        # Linear regression
        x = np.arange(len(closes[-self.trend_window:]))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, closes[-self.trend_window:])
        
        # Moving averages
        ma_short = np.mean(closes[-20:])
        ma_long = np.mean(closes[-50:])
        
        # Trend direction
        direction = "neutral"
        if slope > 0 and ma_short > ma_long:
            direction = "bullish"
        elif slope < 0 and ma_short < ma_long:
            direction = "bearish"
            
        # Trend strength
        strength = abs(r_value)
        
        return {
            "direction": direction,
            "strength": strength,
            "slope": slope,
            "r_squared": r_value**2,
            "ma_diff": ma_short - ma_long
        }
        
    def _calculate_momentum(self, closes: np.ndarray) -> Dict:
        """
        Calculate momentum indicators:
        - ROC (Rate of Change)
        - RSI
        - Moving average convergence
        """
        # Rate of Change
        roc = (closes[-1] / closes[-20] - 1) * 100
        
        # RSI
        returns = np.diff(closes)
        gains = np.maximum(returns, 0)
        losses = np.abs(np.minimum(returns, 0))
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Momentum state
        momentum = roc * (rsi / 50)
        state = "neutral"
        
        if momentum > self.momentum_thresholds["strong"]:
            state = "accelerating"
        elif momentum < self.momentum_thresholds["weak"]:
            state = "decelerating"
            
        return {
            "value": momentum,
            "state": state,
            "roc": roc,
            "rsi": rsi
        }
        
    def _analyze_liquidity(self, volumes: np.ndarray) -> Dict:
        """
        Analyze market liquidity:
        - Volume profile
        - Trade frequency
        - Bid-ask spread (if available)
        """
        recent_vol = np.mean(volumes[-20:])
        vol_std = np.std(volumes[-20:])
        
        state = "normal"
        if recent_vol > np.mean(volumes) + vol_std:
            state = "deep"
        elif recent_vol < np.mean(volumes) - vol_std:
            state = "thin"
            
        return {
            "state": state,
            "current_volume": recent_vol,
            "volume_std": vol_std
        }
        
    def _classify_regime(
        self,
        closes: np.ndarray,
        volatility: Dict,
        momentum: Dict
    ) -> Dict:
        """
        Classify market regime using multiple factors:
        - Price action patterns
        - Volatility state
        - Momentum characteristics
        - Recent regime history
        """
        # Prepare features for classification
        features = np.column_stack([
            np.diff(closes[-self.regime_window:]),  # Price changes
            [volatility["value"]] * (self.regime_window - 1),  # Volatility
            [momentum["value"]] * (self.regime_window - 1)  # Momentum
        ])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Classify regime
        labels = self.regime_classifier.fit_predict(scaled_features)
        current_label = labels[-1]
        
        # Interpret regime
        regime = "ranging"  # default
        confidence = 0.5
        
        # Analyze cluster characteristics
        cluster_volatility = volatility["value"]
        cluster_momentum = abs(momentum["value"])
        
        if cluster_volatility > self.volatility_thresholds["high"]:
            regime = "volatile"
            confidence = min(cluster_volatility / self.volatility_thresholds["high"], 1.0)
        elif cluster_momentum > self.momentum_thresholds["strong"]:
            regime = "trending"
            confidence = min(cluster_momentum / self.momentum_thresholds["strong"], 1.0)
            
        return {
            "regime": regime,
            "confidence": confidence,
            "cluster": int(current_label)
        }
        
    def _update_history(self, state: MarketState):
        """Update regime history and detect transitions"""
        self.regime_history.append(state)
        
        # Maintain history length
        if len(self.regime_history) > self.regime_window:
            self.regime_history.pop(0)
            
        # Detect regime transitions
        if len(self.regime_history) >= 2:
            prev_state = self.regime_history[-2]
            if prev_state.regime != state.regime:
                self.regime_transitions.append({
                    "from": prev_state.regime,
                    "to": state.regime,
                    "timestamp": state.timestamp
                })
                
    def _default_state(self) -> MarketState:
        """Return default state when insufficient data"""
        return MarketState(
            regime="undefined",
            trend_direction="neutral",
            volatility_state="normal",
            momentum_state="neutral",
            liquidity_state="normal",
            confidence=0.0,
            timestamp=datetime.utcnow()
        )
        
    def get_optimal_timeframe(self, state: MarketState) -> str:
        """
        Determine optimal timeframe based on current regime
        Returns: timeframe string (e.g., "M5", "M15", "H1")
        """
        if state.regime == "volatile":
            return "M5"  # Faster timeframe for volatile markets
        elif state.regime == "trending":
            return "H1"  # Longer timeframe for trends
        else:
            return "M15"  # Default timeframe for ranging
            
    def get_regime_summary(self) -> Dict:
        """Get comprehensive regime analysis summary"""
        if not self.regime_history:
            return {}
            
        current_state = self.regime_history[-1]
        
        return {
            "current_regime": current_state.regime,
            "confidence": current_state.confidence,
            "trend_direction": current_state.trend_direction,
            "volatility_state": current_state.volatility_state,
            "momentum_state": current_state.momentum_state,
            "liquidity_state": current_state.liquidity_state,
            "optimal_timeframe": self.get_optimal_timeframe(current_state),
            "recent_transitions": self.regime_transitions[-3:] if self.regime_transitions else [],
            "timestamp": current_state.timestamp.isoformat()
        }
        
    def get_trading_recommendations(self, state: MarketState) -> Dict:
        """
        Get regime-specific trading recommendations
        Returns strategy adjustments based on current regime
        """
        recommendations = {
            "position_size": 1.0,  # Default multiplier
            "stop_multiplier": 1.0,
            "target_multiplier": 1.0,
            "entry_type": "market",
            "exit_strategy": "fixed",
            "timeframe": "M15",
            "notes": []
        }
        
        if state.regime == "volatile":
            recommendations.update({
                "position_size": 0.7,  # Reduce position size
                "stop_multiplier": 1.5,  # Wider stops
                "target_multiplier": 1.2,  # Adjusted targets
                "entry_type": "limit",  # Prefer limit orders
                "exit_strategy": "trailing",
                "timeframe": "M5",
                "notes": ["Use limit orders", "Wider stops needed", "Quick profits"]
            })
        elif state.regime == "trending":
            recommendations.update({
                "position_size": 1.2,  # Increase position size
                "stop_multiplier": 1.0,
                "target_multiplier": 1.5,  # Larger targets
                "entry_type": "market",
                "exit_strategy": "trailing",
                "timeframe": "H1",
                "notes": ["Trail stops", "Pyramid entries", "Hold for trend"]
            })
        else:  # ranging
            recommendations.update({
                "position_size": 0.8,
                "stop_multiplier": 1.0,
                "target_multiplier": 1.0,
                "entry_type": "limit",
                "exit_strategy": "fixed",
                "timeframe": "M15",
                "notes": ["Trade range boundaries", "Quick exits", "Avoid breakouts"]
            })
            
        return recommendations
