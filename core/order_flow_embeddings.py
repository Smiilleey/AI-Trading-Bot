import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

class OrderFlowEmbeddings:
    """
    Converts raw order flow data into dense, learnable embeddings for ML.
    
    Key transformations:
    - Delta sequences -> momentum/absorption patterns
    - Volume profiles -> institutional activity signatures  
    - Price-volume relationships -> market microstructure features
    - Time-based patterns -> session/regime indicators
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Embedding parameters
        self.sequence_length = config.get("of_sequence_length", 20)
        self.volume_buckets = config.get("of_volume_buckets", 10)
        self.price_levels = config.get("of_price_levels", 50)
        
        # Rolling windows for different timeframes
        self.delta_window = deque(maxlen=self.sequence_length)
        self.volume_window = deque(maxlen=self.sequence_length)
        self.price_window = deque(maxlen=self.sequence_length)
        
    def extract_embeddings(self, 
                          candles: List[Dict], 
                          order_flow_data: Dict,
                          market_context: Dict) -> Dict[str, Any]:
        """
        Extract comprehensive order flow embeddings from raw data.
        
        Returns:
            Dict containing various embedding vectors and features
        """
        try:
            embeddings = {}
            
            # 1. Delta sequence embeddings
            embeddings.update(self._extract_delta_embeddings(order_flow_data))
            
            # 2. Volume profile embeddings  
            embeddings.update(self._extract_volume_embeddings(order_flow_data, candles))
            
            # 3. Price-volume microstructure embeddings
            embeddings.update(self._extract_microstructure_embeddings(candles, order_flow_data))
            
            # 4. Institutional activity embeddings
            embeddings.update(self._extract_institutional_embeddings(order_flow_data, market_context))
            
            # 5. Time-based pattern embeddings
            embeddings.update(self._extract_temporal_embeddings(candles, market_context))
            
            # 6. Market regime embeddings
            embeddings.update(self._extract_regime_embeddings(market_context))
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error extracting order flow embeddings: {e}")
            return self._get_empty_embeddings()
    
    def _extract_delta_embeddings(self, order_flow_data: Dict) -> Dict[str, Any]:
        """Extract delta-based momentum and absorption patterns."""
        embeddings = {}
        
        # Raw delta values
        delta_values = order_flow_data.get("delta_values", [])
        if delta_values:
            delta_array = np.array(delta_values[-self.sequence_length:])
            
            # Basic delta statistics
            embeddings["delta_mean"] = float(np.mean(delta_array))
            embeddings["delta_std"] = float(np.std(delta_array))
            embeddings["delta_skew"] = float(self._safe_skew(delta_array))
            embeddings["delta_kurtosis"] = float(self._safe_kurtosis(delta_array))
            
            # Delta momentum patterns
            embeddings["delta_momentum"] = self._calculate_delta_momentum(delta_array)
            embeddings["delta_absorption"] = self._calculate_delta_absorption(delta_array)
            
            # Delta sequence patterns
            embeddings["delta_sequence"] = self._encode_delta_sequence(delta_array)
            
            # Delta regime indicators
            embeddings["delta_regime"] = self._classify_delta_regime(delta_array)
            
        return embeddings
    
    def _extract_volume_embeddings(self, order_flow_data: Dict, candles: List[Dict]) -> Dict[str, Any]:
        """Extract volume profile and institutional activity patterns."""
        embeddings = {}
        
        # Volume data
        volumes = [c.get("volume", 0) for c in candles[-self.sequence_length:]]
        if volumes:
            volume_array = np.array(volumes)
            
            # Volume statistics
            embeddings["volume_mean"] = float(np.mean(volume_array))
            embeddings["volume_std"] = float(np.std(volume_array))
            embeddings["volume_trend"] = self._calculate_volume_trend(volume_array)
            
            # Volume profile buckets
            embeddings["volume_profile"] = self._create_volume_profile(volume_array)
            
            # Volume-price relationship
            if len(candles) >= 2:
                price_changes = [abs(candles[i]["close"] - candles[i-1]["close"]) 
                               for i in range(1, len(candles))]
                embeddings["volume_price_correlation"] = self._calculate_volume_price_correlation(
                    volume_array[1:], price_changes)
        
        # Order flow volume data
        of_volumes = order_flow_data.get("volume_data", {})
        if of_volumes:
            embeddings["of_volume_imbalance"] = of_volumes.get("imbalance", 0.0)
            embeddings["of_volume_absorption"] = of_volumes.get("absorption", 0.0)
            embeddings["of_volume_exhaustion"] = of_volumes.get("exhaustion", 0.0)
        
        return embeddings
    
    def _extract_microstructure_embeddings(self, candles: List[Dict], order_flow_data: Dict) -> Dict[str, Any]:
        """Extract price-volume microstructure features."""
        embeddings = {}
        
        if len(candles) < 2:
            return embeddings
            
        # Price data
        prices = np.array([c["close"] for c in candles[-self.sequence_length:]])
        highs = np.array([c["high"] for c in candles[-self.sequence_length:]])
        lows = np.array([c["low"] for c in candles[-self.sequence_length:]])
        
        # Price statistics
        embeddings["price_volatility"] = float(np.std(prices))
        embeddings["price_range"] = float(np.max(highs) - np.min(lows))
        embeddings["price_efficiency"] = self._calculate_price_efficiency(prices)
        
        # Microstructure patterns
        embeddings["wick_ratio"] = self._calculate_wick_ratio(highs, lows, prices)
        embeddings["body_ratio"] = self._calculate_body_ratio(candles[-self.sequence_length:])
        
        # Order flow price impact
        of_data = order_flow_data.get("price_impact", {})
        if of_data:
            embeddings["price_impact_buy"] = of_data.get("buy_impact", 0.0)
            embeddings["price_impact_sell"] = of_data.get("sell_impact", 0.0)
            embeddings["price_impact_ratio"] = of_data.get("impact_ratio", 0.0)
        
        return embeddings
    
    def _extract_institutional_embeddings(self, order_flow_data: Dict, market_context: Dict) -> Dict[str, Any]:
        """Extract institutional activity and smart money patterns."""
        embeddings = {}
        
        # Institutional flow data
        inst_data = order_flow_data.get("institutional", {})
        if inst_data:
            embeddings["institutional_delta"] = inst_data.get("delta", 0.0)
            embeddings["institutional_volume"] = inst_data.get("volume", 0.0)
            embeddings["institutional_pressure"] = inst_data.get("pressure", 0.0)
            
            # Smart money indicators
            embeddings["smart_money_accumulation"] = inst_data.get("accumulation", 0.0)
            embeddings["smart_money_distribution"] = inst_data.get("distribution", 0.0)
            embeddings["smart_money_manipulation"] = inst_data.get("manipulation", 0.0)
        
        # Market context
        embeddings["market_depth"] = market_context.get("depth", 0.0)
        embeddings["market_liquidity"] = market_context.get("liquidity", 0.0)
        embeddings["market_stress"] = market_context.get("stress", 0.0)
        
        return embeddings
    
    def _extract_temporal_embeddings(self, candles: List[Dict], market_context: Dict) -> Dict[str, Any]:
        """Extract time-based patterns and session indicators."""
        embeddings = {}
        
        if not candles:
            return embeddings
            
        # Time-based features
        current_time = candles[-1].get("time", 0)
        
        # Session indicators (simplified)
        embeddings["session_london"] = 1.0 if 7 <= (current_time % 86400) // 3600 <= 16 else 0.0
        embeddings["session_newyork"] = 1.0 if 13 <= (current_time % 86400) // 3600 <= 22 else 0.0
        embeddings["session_tokyo"] = 1.0 if 0 <= (current_time % 86400) // 3600 <= 9 else 0.0
        embeddings["session_overlap"] = 1.0 if 13 <= (current_time % 86400) // 3600 <= 16 else 0.0
        
        # Day of week
        day_of_week = (current_time // 86400) % 7
        embeddings["day_of_week"] = float(day_of_week)
        
        # Hour of day
        hour_of_day = (current_time % 86400) // 3600
        embeddings["hour_of_day"] = float(hour_of_day)
        
        return embeddings
    
    def _extract_regime_embeddings(self, market_context: Dict) -> Dict[str, Any]:
        """Extract market regime and volatility embeddings."""
        embeddings = {}
        
        # Regime classification
        regime = market_context.get("regime", "unknown")
        regime_encoding = {
            "quiet": [1.0, 0.0, 0.0, 0.0],
            "normal": [0.0, 1.0, 0.0, 0.0], 
            "trending": [0.0, 0.0, 1.0, 0.0],
            "volatile": [0.0, 0.0, 0.0, 1.0]
        }
        embeddings["regime_encoding"] = regime_encoding.get(regime, [0.0, 0.0, 0.0, 0.0])
        
        # Volatility regime
        volatility = market_context.get("volatility", 0.0)
        embeddings["volatility_regime"] = self._classify_volatility_regime(volatility)
        
        # Trend strength
        trend_strength = market_context.get("trend_strength", 0.0)
        embeddings["trend_strength"] = float(trend_strength)
        
        return embeddings
    
    # Helper methods
    def _safe_skew(self, data: np.ndarray) -> float:
        """Calculate skewness safely."""
        try:
            from scipy.stats import skew
            return float(skew(data))
        except:
            return 0.0
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis safely."""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            return 0.0
    
    def _calculate_delta_momentum(self, delta_array: np.ndarray) -> float:
        """Calculate delta momentum indicator."""
        if len(delta_array) < 3:
            return 0.0
        
        # Simple momentum: recent delta vs earlier delta
        recent = np.mean(delta_array[-3:])
        earlier = np.mean(delta_array[:3])
        return float(recent - earlier)
    
    def _calculate_delta_absorption(self, delta_array: np.ndarray) -> float:
        """Calculate delta absorption indicator."""
        if len(delta_array) < 5:
            return 0.0
        
        # Look for high delta with low price movement (absorption)
        delta_magnitude = np.abs(delta_array)
        return float(np.mean(delta_magnitude))
    
    def _encode_delta_sequence(self, delta_array: np.ndarray) -> List[float]:
        """Encode delta sequence as normalized vector."""
        if len(delta_array) == 0:
            return [0.0] * self.sequence_length
        
        # Normalize and pad/truncate to sequence length
        normalized = delta_array / (np.std(delta_array) + 1e-8)
        
        if len(normalized) >= self.sequence_length:
            return normalized[-self.sequence_length:].tolist()
        else:
            padded = np.zeros(self.sequence_length)
            padded[-len(normalized):] = normalized
            return padded.tolist()
    
    def _classify_delta_regime(self, delta_array: np.ndarray) -> List[float]:
        """Classify delta into regime categories."""
        if len(delta_array) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        mean_delta = np.mean(delta_array)
        std_delta = np.std(delta_array)
        
        # Regime classification based on delta characteristics
        if std_delta < 0.1:
            return [1.0, 0.0, 0.0, 0.0]  # Low activity
        elif mean_delta > 0.5:
            return [0.0, 1.0, 0.0, 0.0]  # Strong buying
        elif mean_delta < -0.5:
            return [0.0, 0.0, 1.0, 0.0]  # Strong selling
        else:
            return [0.0, 0.0, 0.0, 1.0]  # Balanced/choppy
    
    def _calculate_volume_trend(self, volume_array: np.ndarray) -> float:
        """Calculate volume trend indicator."""
        if len(volume_array) < 3:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(volume_array))
        slope = np.polyfit(x, volume_array, 1)[0]
        return float(slope)
    
    def _create_volume_profile(self, volume_array: np.ndarray) -> List[float]:
        """Create volume profile buckets."""
        if len(volume_array) == 0:
            return [0.0] * self.volume_buckets
        
        # Create histogram buckets
        hist, _ = np.histogram(volume_array, bins=self.volume_buckets)
        normalized = hist / (np.sum(hist) + 1e-8)
        return normalized.tolist()
    
    def _calculate_volume_price_correlation(self, volumes: np.ndarray, price_changes: List[float]) -> float:
        """Calculate correlation between volume and price changes."""
        if len(volumes) != len(price_changes) or len(volumes) < 2:
            return 0.0
        
        try:
            correlation = np.corrcoef(volumes, price_changes)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_price_efficiency(self, prices: np.ndarray) -> float:
        """Calculate price efficiency (actual move vs total range)."""
        if len(prices) < 2:
            return 0.0
        
        total_range = np.max(prices) - np.min(prices)
        actual_move = abs(prices[-1] - prices[0])
        
        if total_range == 0:
            return 1.0
        
        return float(actual_move / total_range)
    
    def _calculate_wick_ratio(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate average wick to body ratio."""
        if len(highs) != len(lows) or len(highs) != len(closes):
            return 0.0
        
        wick_ratios = []
        for i in range(len(highs)):
            high = highs[i]
            low = lows[i]
            close = closes[i]
            
            upper_wick = high - max(close, low)
            lower_wick = min(close, low) - low
            body = abs(close - low)
            
            if body > 0:
                total_wick = upper_wick + lower_wick
                wick_ratios.append(total_wick / body)
        
        return float(np.mean(wick_ratios)) if wick_ratios else 0.0
    
    def _calculate_body_ratio(self, candles: List[Dict]) -> float:
        """Calculate average body to total range ratio."""
        if not candles:
            return 0.0
        
        body_ratios = []
        for candle in candles:
            high = candle.get("high", 0)
            low = candle.get("low", 0)
            open_price = candle.get("open", 0)
            close = candle.get("close", 0)
            
            total_range = high - low
            body_size = abs(close - open_price)
            
            if total_range > 0:
                body_ratios.append(body_size / total_range)
        
        return float(np.mean(body_ratios)) if body_ratios else 0.0
    
    def _classify_volatility_regime(self, volatility: float) -> List[float]:
        """Classify volatility into regime categories."""
        if volatility < 0.5:
            return [1.0, 0.0, 0.0]  # Low volatility
        elif volatility < 1.5:
            return [0.0, 1.0, 0.0]  # Normal volatility
        else:
            return [0.0, 0.0, 1.0]  # High volatility
    
    def _get_empty_embeddings(self) -> Dict[str, Any]:
        """Return empty embeddings structure."""
        return {
            "delta_mean": 0.0,
            "delta_std": 0.0,
            "delta_skew": 0.0,
            "delta_kurtosis": 0.0,
            "delta_momentum": 0.0,
            "delta_absorption": 0.0,
            "delta_sequence": [0.0] * self.sequence_length,
            "delta_regime": [0.0, 0.0, 0.0, 0.0],
            "volume_mean": 0.0,
            "volume_std": 0.0,
            "volume_trend": 0.0,
            "volume_profile": [0.0] * self.volume_buckets,
            "volume_price_correlation": 0.0,
            "price_volatility": 0.0,
            "price_range": 0.0,
            "price_efficiency": 0.0,
            "wick_ratio": 0.0,
            "body_ratio": 0.0,
            "regime_encoding": [0.0, 0.0, 0.0, 0.0],
            "volatility_regime": [0.0, 0.0, 0.0],
            "trend_strength": 0.0
        }
