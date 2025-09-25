# core/correlation_engine.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from collections import defaultdict

@dataclass
class CorrelationState:
    pair1: str
    pair2: str
    correlation: float
    strength: float  # correlation stability
    lag: int  # lead/lag relationship
    regime: str  # stable, breakdown, strengthening
    timestamp: datetime

class CrossPairAnalyzer:
    """
    Advanced cross-pair correlation analysis:
    - Dynamic correlation tracking
    - Lead/lag relationships
    - Correlation regime detection
    - Currency strength analysis
    - Correlation breakdown alerts
    - Entry/exit optimization
    """
    def __init__(
        self,
        correlation_window: int = 100,
        min_correlation: float = 0.5,
        update_interval: int = 20
    ):
        self.correlation_window = correlation_window
        self.min_correlation = min_correlation
        self.update_interval = update_interval
        
        # State tracking
        self.correlation_states = {}  # pair combo -> CorrelationState
        self.currency_strength = {}  # currency -> strength score
        self.regime_changes = []
        self.last_update = None
        
        # Performance tracking
        self.correlation_accuracy = defaultdict(list)  # pair combo -> historical accuracy
        
    def update(
        self,
        pair_data: Dict[str, List[Dict]],
        current_pair: str = None
    ) -> Dict:
        """
        Update correlation analysis with new data
        Returns correlation-based trading signals for current_pair
        """
        now = datetime.utcnow()
        
        # Check if update needed
        if (self.last_update and
            (now - self.last_update).seconds < self.update_interval):
            return self._get_last_analysis(current_pair)
            
        # Calculate all correlations
        pairs = list(pair_data.keys())
        for i, pair1 in enumerate(pairs):
            for pair2 in pairs[i+1:]:
                state = self._calculate_correlation(
                    pair_data[pair1],
                    pair_data[pair2],
                    pair1,
                    pair2
                )
                key = self._get_pair_key(pair1, pair2)
                self.correlation_states[key] = state
                
        # Update currency strength
        self._update_currency_strength(pair_data)
        
        # Generate analysis
        analysis = self._generate_analysis(current_pair)
        self.last_update = now
        
        return analysis
        
    def _calculate_correlation(
        self,
        data1: List[Dict],
        data2: List[Dict],
        pair1: str,
        pair2: str
    ) -> CorrelationState:
        """
        Calculate correlation state between two pairs
        Includes lead/lag analysis and regime detection
        """
        if len(data1) < self.correlation_window or len(data2) < self.correlation_window:
            return self._empty_correlation_state(pair1, pair2)
            
        # Extract close prices
        closes1 = np.array([c["close"] for c in data1[-self.correlation_window:]])
        closes2 = np.array([c["close"] for c in data2[-self.correlation_window:]])
        
        # Calculate returns
        returns1 = np.diff(np.log(closes1))
        returns2 = np.diff(np.log(closes2))
        
        # Base correlation
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        # Calculate correlation strength (stability)
        rolling_corr = []
        for i in range(20, len(returns1)):
            window_corr = np.corrcoef(
                returns1[i-20:i],
                returns2[i-20:i]
            )[0, 1]
            rolling_corr.append(window_corr)
            
        strength = 1 - np.std(rolling_corr)  # lower volatility = higher strength
        
        # Find lead/lag relationship
        max_lag = 10
        lag_correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(
                    returns1[:lag],
                    returns2[-lag:]
                )[0, 1]
            else:
                corr = np.corrcoef(
                    returns1[lag:],
                    returns2[:-lag] if lag > 0 else returns2
                )[0, 1]
            lag_correlations.append((lag, corr))
            
        # Find lag with highest correlation
        best_lag, max_corr = max(lag_correlations, key=lambda x: abs(x[1]))
        
        # Determine correlation regime
        regime = "stable"
        key = self._get_pair_key(pair1, pair2)
        
        if key in self.correlation_states:
            old_state = self.correlation_states[key]
            if abs(correlation - old_state.correlation) > 0.3:
                regime = "breakdown"
            elif abs(correlation) > abs(old_state.correlation) + 0.1:
                regime = "strengthening"
                
        return CorrelationState(
            pair1=pair1,
            pair2=pair2,
            correlation=correlation,
            strength=strength,
            lag=best_lag,
            regime=regime,
            timestamp=datetime.utcnow()
        )
        
    def _update_currency_strength(self, pair_data: Dict[str, List[Dict]]):
        """
        Calculate individual currency strength scores
        Uses weighted average of correlations and performance
        """
        currency_returns = defaultdict(list)
        
        # Calculate returns for each currency pair
        for pair, data in pair_data.items():
            if len(data) < 2:
                continue
                
            returns = np.diff(np.log([c["close"] for c in data[-self.correlation_window:]]))
            base, quote = pair[:3], pair[3:]
            
            # Add returns to both currencies (positive for base, negative for quote)
            currency_returns[base].extend(returns)
            currency_returns[quote].extend(-returns)
            
        # Calculate strength scores
        for currency, returns in currency_returns.items():
            # Combine recent performance and volatility
            recent_perf = np.mean(returns[-20:])  # Short-term performance
            volatility = np.std(returns)  # Risk adjustment
            
            self.currency_strength[currency] = {
                "strength": recent_perf / volatility if volatility > 0 else 0,
                "volatility": volatility,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    def _generate_analysis(self, current_pair: str = None) -> Dict:
        """
        Generate comprehensive correlation analysis
        Optionally focused on a specific pair
        """
        analysis = {
            "correlations": {},
            "currency_strength": self.currency_strength,
            "regime_changes": self.regime_changes[-5:],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add all correlation states
        for key, state in self.correlation_states.items():
            analysis["correlations"][key] = {
                "correlation": state.correlation,
                "strength": state.strength,
                "lag": state.lag,
                "regime": state.regime,
                "timestamp": state.timestamp.isoformat()
            }
            
        # Add focused analysis for current pair
        if current_pair:
            analysis["pair_analysis"] = self._analyze_pair(current_pair)
            
        return analysis
        
    def _analyze_pair(self, pair: str) -> Dict:
        """Detailed analysis for a specific pair"""
        base, quote = pair[:3], pair[3:]
        
        # Find relevant correlations
        relevant_correlations = []
        for key, state in self.correlation_states.items():
            if pair in [state.pair1, state.pair2]:
                relevant_correlations.append({
                    "pair": state.pair2 if state.pair1 == pair else state.pair1,
                    "correlation": state.correlation,
                    "strength": state.strength,
                    "lag": state.lag,
                    "regime": state.regime
                })
                
        # Sort by correlation strength
        relevant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "base_currency": {
                "currency": base,
                "strength": self.currency_strength.get(base, {"strength": 0})["strength"]
            },
            "quote_currency": {
                "currency": quote,
                "strength": self.currency_strength.get(quote, {"strength": 0})["strength"]
            },
            "correlated_pairs": relevant_correlations,
            "trading_implications": self._get_trading_implications(pair)
        }
        
    def _get_trading_implications(self, pair: str) -> Dict:
        """
        Generate trading recommendations based on correlations
        Returns entry/exit suggestions and risk adjustments
        """
        implications = {
            "entry_quality": 0.0,
            "exit_signals": [],
            "risk_adjustments": [],
            "correlated_movements": []
        }
        
        # Find strong correlations
        strong_correlations = [
            (key, state) for key, state in self.correlation_states.items()
            if abs(state.correlation) > self.min_correlation
            and (state.pair1 == pair or state.pair2 == pair)
        ]
        
        if not strong_correlations:
            return implications
            
        # Analyze correlation patterns
        for key, state in strong_correlations:
            other_pair = state.pair2 if state.pair1 == pair else state.pair1
            
            # Entry quality
            implications["entry_quality"] += (
                abs(state.correlation) * state.strength
            ) / len(strong_correlations)
            
            # Exit signals
            if state.regime == "breakdown":
                implications["exit_signals"].append(
                    f"Correlation breakdown with {other_pair}"
                )
                
            # Risk adjustments
            if state.correlation > 0.7:
                implications["risk_adjustments"].append(
                    f"Reduce position size due to high correlation with {other_pair}"
                )
                
            # Correlated movements
            implications["correlated_movements"].append({
                "pair": other_pair,
                "correlation": state.correlation,
                "lag": state.lag,
                "expected_impact": "same" if state.correlation > 0 else "opposite"
            })
            
        return implications
        
    def get_correlation_based_signals(
        self,
        pair: str,
        timeframe: str = "H1"
    ) -> Dict:
        """
        Generate trading signals based on correlation analysis
        Returns entry/exit suggestions with confidence levels
        """
        analysis = self._analyze_pair(pair)
        base, quote = pair[:3], pair[3:]
        
        signals = {
            "entry": None,
            "exit": False,
            "confidence": 0.0,
            "supporting_pairs": [],
            "risk_factors": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check currency strength
        base_strength = self.currency_strength.get(base, {"strength": 0})["strength"]
        quote_strength = self.currency_strength.get(quote, {"strength": 0})["strength"]
        strength_diff = base_strength - quote_strength
        
        # Generate directional bias
        if abs(strength_diff) > 0.5:
            signals["entry"] = "buy" if strength_diff > 0 else "sell"
            signals["confidence"] = min(abs(strength_diff), 1.0)
            
        # Find supporting pairs
        for corr in analysis["correlated_pairs"]:
            if abs(corr["correlation"]) > self.min_correlation:
                signals["supporting_pairs"].append({
                    "pair": corr["pair"],
                    "correlation": corr["correlation"],
                    "lag": corr["lag"]
                })
                
        # Check for exit signals
        if analysis["trading_implications"]["exit_signals"]:
            signals["exit"] = True
            signals["risk_factors"].extend(
                analysis["trading_implications"]["exit_signals"]
            )
            
        # Adjust confidence
        signals["confidence"] *= analysis["trading_implications"]["entry_quality"]
        
        return signals

    def get_intermarket_confirmation(self, pair: str, intermarket_data: Dict = None) -> Dict:
        """
        Get intermarket confirmation for USD pairs using DXY, US10Y, SPX
        Returns confirmation weights and directional bias
        """
        if not intermarket_data:
            return {"confirmation_weight": 0.0, "bias": "neutral", "confidence": 0.0}
        
        base, quote = pair[:3], pair[3:]
        
        # Only process USD pairs
        if base != "USD" and quote != "USD":
            return {"confirmation_weight": 0.0, "bias": "neutral", "confidence": 0.0}
        
        confirmation_score = 0.0
        total_weight = 0.0
        confirmations = []
        
        # DXY (Dollar Index) confirmation
        if "DXY" in intermarket_data:
            dxy_data = intermarket_data["DXY"]
            dxy_trend = dxy_data.get("trend", 0)  # -1 bearish, 0 neutral, 1 bullish
            dxy_strength = dxy_data.get("strength", 0.5)
            
            if base == "USD":  # USD is base currency
                # DXY up = USD strong = bullish for USD pairs
                dxy_confirmation = dxy_trend * dxy_strength
            else:  # USD is quote currency
                # DXY up = USD strong = bearish for non-USD pairs
                dxy_confirmation = -dxy_trend * dxy_strength
            
            confirmation_score += dxy_confirmation * 0.4  # 40% weight
            total_weight += 0.4
            confirmations.append({
                "instrument": "DXY",
                "confirmation": dxy_confirmation,
                "strength": dxy_strength,
                "trend": dxy_trend
            })
        
        # US10Y (10-Year Treasury) confirmation
        if "US10Y" in intermarket_data:
            us10y_data = intermarket_data["US10Y"]
            us10y_trend = us10y_data.get("trend", 0)
            us10y_strength = us10y_data.get("strength", 0.5)
            
            if base == "USD":
                # Rising yields = USD strong = bullish for USD pairs
                us10y_confirmation = us10y_trend * us10y_strength
            else:
                # Rising yields = USD strong = bearish for non-USD pairs
                us10y_confirmation = -us10y_trend * us10y_strength
            
            confirmation_score += us10y_confirmation * 0.3  # 30% weight
            total_weight += 0.3
            confirmations.append({
                "instrument": "US10Y",
                "confirmation": us10y_confirmation,
                "strength": us10y_strength,
                "trend": us10y_trend
            })
        
        # SPX (S&P 500) confirmation
        if "SPX" in intermarket_data:
            spx_data = intermarket_data["SPX"]
            spx_trend = spx_data.get("trend", 0)
            spx_strength = spx_data.get("strength", 0.5)
            
            if base == "USD":
                # Risk-on = USD weak = bearish for USD pairs
                spx_confirmation = -spx_trend * spx_strength
            else:
                # Risk-on = USD weak = bullish for non-USD pairs
                spx_confirmation = spx_trend * spx_strength
            
            confirmation_score += spx_confirmation * 0.3  # 30% weight
            total_weight += 0.3
            confirmations.append({
                "instrument": "SPX",
                "confirmation": spx_confirmation,
                "strength": spx_strength,
                "trend": spx_trend
            })
        
        # Normalize confirmation score
        if total_weight > 0:
            confirmation_score /= total_weight
        
        # Determine bias and confidence
        if confirmation_score > 0.3:
            bias = "bullish"
            confidence = min(abs(confirmation_score), 1.0)
        elif confirmation_score < -0.3:
            bias = "bearish"
            confidence = min(abs(confirmation_score), 1.0)
        else:
            bias = "neutral"
            confidence = 0.0
        
        return {
            "confirmation_weight": confirmation_score,
            "bias": bias,
            "confidence": confidence,
            "confirmations": confirmations,
            "total_weight": total_weight,
            "entry_gate_active": abs(confirmation_score) > 0.5 and confidence > 0.6
        }
        
    def _get_pair_key(self, pair1: str, pair2: str) -> str:
        """Generate consistent key for pair combination"""
        return f"{min(pair1, pair2)}_{max(pair1, pair2)}"
        
    def _empty_correlation_state(
        self,
        pair1: str,
        pair2: str
    ) -> CorrelationState:
        """Return empty correlation state"""
        return CorrelationState(
            pair1=pair1,
            pair2=pair2,
            correlation=0.0,
            strength=0.0,
            lag=0,
            regime="undefined",
            timestamp=datetime.utcnow()
        )
        
    def _get_last_analysis(self, current_pair: str = None) -> Dict:
        """Return last analysis if recent enough"""
        analysis = {
            "correlations": {},
            "currency_strength": self.currency_strength,
            "regime_changes": self.regime_changes[-5:],
            "timestamp": self.last_update.isoformat()
        }
        
        for key, state in self.correlation_states.items():
            analysis["correlations"][key] = {
                "correlation": state.correlation,
                "strength": state.strength,
                "lag": state.lag,
                "regime": state.regime,
                "timestamp": state.timestamp.isoformat()
            }
            
        if current_pair:
            analysis["pair_analysis"] = self._analyze_pair(current_pair)
            
        return analysis
