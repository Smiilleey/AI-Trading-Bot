# core/volatility_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class VolatilityEngine:
    """
    Advanced Volatility Engine for Squeeze/Expansion Analysis
    
    Features:
    - Bollinger Bands squeeze detection
    - Keltner Channels squeeze detection
    - Volatility expansion/contraction analysis
    - Volatility regime classification
    - Threshold modulation based on volatility state
    - Integration with signal engine for adaptive thresholds
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Volatility calculation parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.kc_period = config.get('kc_period', 20)
        self.kc_multiplier = config.get('kc_multiplier', 2.0)
        self.atr_period = config.get('atr_period', 14)
        
        # Squeeze detection parameters
        self.squeeze_threshold = config.get('squeeze_threshold', 0.1)  # 10% threshold
        self.expansion_threshold = config.get('expansion_threshold', 0.3)  # 30% threshold
        
        # Volatility regime thresholds
        self.regime_thresholds = {
            'low': 0.2,      # Below 20% of average
            'normal': 0.8,   # 20-80% of average
            'high': 1.2      # Above 120% of average
        }
        
        # Storage for volatility data
        self.volatility_history = defaultdict(lambda: deque(maxlen=1000))
        self.squeeze_states = defaultdict(dict)
        self.regime_states = defaultdict(str)
        
        # Performance tracking
        self.squeeze_signals = defaultdict(list)
        self.volatility_performance = defaultdict(lambda: {
            'total_signals': 0,
            'successful_signals': 0,
            'success_rate': 0.0,
            'avg_pnl': 0.0
        })
        
    def calculate_bollinger_bands(self, 
                                 prices: List[float], 
                                 period: int = None, 
                                 std_dev: float = None) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands for volatility analysis
        
        Args:
            prices: List of price values
            period: Period for moving average (default: self.bb_period)
            std_dev: Standard deviation multiplier (default: self.bb_std)
            
        Returns:
            Dictionary with Bollinger Bands data
        """
        try:
            if len(prices) < (period or self.bb_period):
                return {"valid": False, "error": "Insufficient data"}
            
            period = period or self.bb_period
            std_dev = std_dev or self.bb_std
            
            # Calculate moving average
            ma = np.mean(prices[-period:])
            
            # Calculate standard deviation
            std = np.std(prices[-period:])
            
            # Calculate bands
            upper_band = ma + (std_dev * std)
            lower_band = ma - (std_dev * std)
            
            # Calculate band width
            band_width = (upper_band - lower_band) / ma if ma > 0 else 0
            
            # Calculate %B (position within bands)
            current_price = prices[-1]
            percent_b = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
            
            return {
                "valid": True,
                "upper_band": upper_band,
                "middle_band": ma,
                "lower_band": lower_band,
                "band_width": band_width,
                "percent_b": percent_b,
                "std_dev": std,
                "period": period
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def calculate_keltner_channels(self, 
                                  candles: List[Dict], 
                                  period: int = None, 
                                  multiplier: float = None) -> Dict[str, Any]:
        """
        Calculate Keltner Channels for volatility analysis
        
        Args:
            candles: List of candle data with high, low, close
            period: Period for ATR calculation (default: self.kc_period)
            multiplier: ATR multiplier (default: self.kc_multiplier)
            
        Returns:
            Dictionary with Keltner Channels data
        """
        try:
            if len(candles) < (period or self.kc_period):
                return {"valid": False, "error": "Insufficient data"}
            
            period = period or self.kc_period
            multiplier = multiplier or self.kc_multiplier
            
            # Calculate ATR
            atr = self._calculate_atr(candles, period)
            
            # Calculate typical price
            typical_prices = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles[-period:]]
            ma = np.mean(typical_prices)
            
            # Calculate channels
            upper_channel = ma + (multiplier * atr)
            lower_channel = ma - (multiplier * atr)
            
            # Calculate channel width
            channel_width = (upper_channel - lower_channel) / ma if ma > 0 else 0
            
            return {
                "valid": True,
                "upper_channel": upper_channel,
                "middle_channel": ma,
                "lower_channel": lower_channel,
                "channel_width": channel_width,
                "atr": atr,
                "period": period
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def detect_volatility_squeeze(self, 
                                 symbol: str, 
                                 candles: List[Dict]) -> Dict[str, Any]:
        """
        Detect volatility squeeze using Bollinger Bands and Keltner Channels
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Dictionary with squeeze analysis
        """
        try:
            if len(candles) < max(self.bb_period, self.kc_period):
                return {"valid": False, "error": "Insufficient data"}
            
            # Extract prices
            prices = [c["close"] for c in candles]
            
            # Calculate Bollinger Bands
            bb_data = self.calculate_bollinger_bands(prices, self.bb_period, self.bb_std)
            if not bb_data["valid"]:
                return {"valid": False, "error": "Bollinger Bands calculation failed"}
            
            # Calculate Keltner Channels
            kc_data = self.calculate_keltner_channels(candles, self.kc_period, self.kc_multiplier)
            if not kc_data["valid"]:
                return {"valid": False, "error": "Keltner Channels calculation failed"}
            
            # Detect squeeze
            bb_width = bb_data["band_width"]
            kc_width = kc_data["channel_width"]
            
            # Squeeze occurs when BB width < KC width
            squeeze_ratio = bb_width / kc_width if kc_width > 0 else 1.0
            is_squeeze = squeeze_ratio < (1 - self.squeeze_threshold)
            
            # Determine squeeze strength
            if squeeze_ratio < 0.5:
                squeeze_strength = "extreme"
            elif squeeze_ratio < 0.7:
                squeeze_strength = "strong"
            elif squeeze_ratio < 0.9:
                squeeze_strength = "moderate"
            else:
                squeeze_strength = "weak"
            
            # Check for expansion
            is_expansion = squeeze_ratio > (1 + self.expansion_threshold)
            
            # Store squeeze state
            self.squeeze_states[symbol] = {
                "is_squeeze": is_squeeze,
                "squeeze_ratio": squeeze_ratio,
                "squeeze_strength": squeeze_strength,
                "is_expansion": is_expansion,
                "bb_width": bb_width,
                "kc_width": kc_width,
                "timestamp": datetime.now()
            }
            
            return {
                "valid": True,
                "is_squeeze": is_squeeze,
                "squeeze_ratio": squeeze_ratio,
                "squeeze_strength": squeeze_strength,
                "is_expansion": is_expansion,
                "bb_width": bb_width,
                "kc_width": kc_width,
                "bb_data": bb_data,
                "kc_data": kc_data
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def classify_volatility_regime(self, 
                                  symbol: str, 
                                  candles: List[Dict]) -> Dict[str, Any]:
        """
        Classify current volatility regime
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Dictionary with volatility regime analysis
        """
        try:
            if len(candles) < 50:
                return {"valid": False, "error": "Insufficient data for regime classification"}
            
            # Calculate recent volatility
            recent_prices = [c["close"] for c in candles[-20:]]
            recent_volatility = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
            
            # Calculate historical volatility
            historical_prices = [c["close"] for c in candles[-50:]]
            historical_volatility = np.std(historical_prices) / np.mean(historical_prices) if np.mean(historical_prices) > 0 else 0
            
            # Calculate volatility ratio
            volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0
            
            # Classify regime
            if volatility_ratio < self.regime_thresholds['low']:
                regime = "low"
                confidence = 0.9
            elif volatility_ratio < self.regime_thresholds['normal']:
                regime = "normal"
                confidence = 0.8
            elif volatility_ratio < self.regime_thresholds['high']:
                regime = "high"
                confidence = 0.8
            else:
                regime = "extreme"
                confidence = 0.9
            
            # Store regime state
            self.regime_states[symbol] = regime
            
            return {
                "valid": True,
                "regime": regime,
                "confidence": confidence,
                "volatility_ratio": volatility_ratio,
                "recent_volatility": recent_volatility,
                "historical_volatility": historical_volatility,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def calculate_threshold_modulation(self, 
                                     symbol: str, 
                                     base_threshold: float) -> Dict[str, Any]:
        """
        Calculate threshold modulation based on volatility state
        
        Args:
            symbol: Trading symbol
            base_threshold: Base entry threshold
            
        Returns:
            Dictionary with modulated threshold
        """
        try:
            # Get current squeeze state
            squeeze_state = self.squeeze_states.get(symbol, {})
            regime_state = self.regime_states.get(symbol, "normal")
            
            # Start with base threshold
            modulated_threshold = base_threshold
            modulation_factors = []
            
            # Apply squeeze modulation
            if squeeze_state.get("is_squeeze", False):
                squeeze_strength = squeeze_state.get("squeeze_strength", "weak")
                if squeeze_strength == "extreme":
                    # Lower threshold in extreme squeeze (more sensitive)
                    factor = 0.7
                    modulation_factors.append(f"extreme_squeeze: {factor}")
                elif squeeze_strength == "strong":
                    factor = 0.8
                    modulation_factors.append(f"strong_squeeze: {factor}")
                elif squeeze_strength == "moderate":
                    factor = 0.9
                    modulation_factors.append(f"moderate_squeeze: {factor}")
                else:
                    factor = 0.95
                    modulation_factors.append(f"weak_squeeze: {factor}")
                
                modulated_threshold *= factor
            
            # Apply expansion modulation
            if squeeze_state.get("is_expansion", False):
                # Higher threshold in expansion (less sensitive)
                factor = 1.2
                modulation_factors.append(f"expansion: {factor}")
                modulated_threshold *= factor
            
            # Apply regime modulation
            if regime_state == "low":
                # Lower threshold in low volatility (more sensitive)
                factor = 0.9
                modulation_factors.append(f"low_volatility: {factor}")
            elif regime_state == "high":
                # Higher threshold in high volatility (less sensitive)
                factor = 1.1
                modulation_factors.append(f"high_volatility: {factor}")
            elif regime_state == "extreme":
                # Much higher threshold in extreme volatility
                factor = 1.3
                modulation_factors.append(f"extreme_volatility: {factor}")
            
            # Ensure threshold stays within reasonable bounds
            modulated_threshold = max(0.1, min(0.95, modulated_threshold))
            
            return {
                "valid": True,
                "base_threshold": base_threshold,
                "modulated_threshold": modulated_threshold,
                "modulation_factors": modulation_factors,
                "squeeze_state": squeeze_state,
                "regime_state": regime_state
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_volatility_features(self, 
                               symbol: str, 
                               candles: List[Dict]) -> Dict[str, Any]:
        """
        Get volatility features for signal engine integration
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            
        Returns:
            Dictionary with volatility features
        """
        try:
            features = {}
            
            # Get squeeze analysis
            squeeze_analysis = self.detect_volatility_squeeze(symbol, candles)
            if squeeze_analysis["valid"]:
                features["volatility_squeeze"] = squeeze_analysis["is_squeeze"]
                features["squeeze_ratio"] = squeeze_analysis["squeeze_ratio"]
                features["squeeze_strength"] = squeeze_analysis["squeeze_strength"]
                features["volatility_expansion"] = squeeze_analysis["is_expansion"]
                features["bb_width"] = squeeze_analysis["bb_width"]
                features["kc_width"] = squeeze_analysis["kc_width"]
            
            # Get regime analysis
            regime_analysis = self.classify_volatility_regime(symbol, candles)
            if regime_analysis["valid"]:
                features["volatility_regime"] = regime_analysis["regime"]
                features["volatility_confidence"] = regime_analysis["confidence"]
                features["volatility_ratio"] = regime_analysis["volatility_ratio"]
                features["recent_volatility"] = regime_analysis["recent_volatility"]
                features["historical_volatility"] = regime_analysis["historical_volatility"]
            
            # Get threshold modulation
            base_threshold = 0.6  # Default base threshold
            threshold_analysis = self.calculate_threshold_modulation(symbol, base_threshold)
            if threshold_analysis["valid"]:
                features["volatility_modulated_threshold"] = threshold_analysis["modulated_threshold"]
                features["threshold_modulation_factors"] = threshold_analysis["modulation_factors"]
            
            return features
            
        except Exception as e:
            return {"volatility_error": str(e)}
    
    def _calculate_atr(self, 
                      candles: List[Dict], 
                      period: int) -> float:
        """Calculate Average True Range"""
        try:
            if len(candles) < period + 1:
                return 0.0
            
            true_ranges = []
            for i in range(1, min(len(candles), period + 1)):
                high = candles[i]["high"]
                low = candles[i]["low"]
                prev_close = candles[i-1]["close"]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            return np.mean(true_ranges) if true_ranges else 0.0
            
        except Exception:
            return 0.0
    
    def update_performance(self, 
                          symbol: str, 
                          signal_data: Dict, 
                          outcome: bool, 
                          pnl: float):
        """Update volatility performance tracking"""
        try:
            self.volatility_performance[symbol]["total_signals"] += 1
            if outcome:
                self.volatility_performance[symbol]["successful_signals"] += 1
            
            # Update success rate
            perf = self.volatility_performance[symbol]
            perf["success_rate"] = perf["successful_signals"] / perf["total_signals"]
            perf["avg_pnl"] = (perf["avg_pnl"] * (perf["total_signals"] - 1) + pnl) / perf["total_signals"]
            
        except Exception:
            pass
    
    def get_volatility_stats(self, symbol: str = None) -> Dict[str, Any]:
        """Get volatility engine statistics"""
        try:
            if symbol:
                return {
                    "symbol": symbol,
                    "performance": self.volatility_performance[symbol],
                    "squeeze_state": self.squeeze_states.get(symbol, {}),
                    "regime_state": self.regime_states.get(symbol, "unknown")
                }
            else:
                return {
                    "total_symbols": len(self.squeeze_states),
                    "performance": dict(self.volatility_performance),
                    "last_update": datetime.now().isoformat()
                }
        except Exception as e:
            return {"error": str(e)}