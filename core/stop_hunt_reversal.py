# core/stop_hunt_reversal.py

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class StopHuntState:
    """State tracking for stop-hunt reversal detection."""
    sweep_detected: bool
    absorption_detected: bool
    reclaim_detected: bool
    sweep_price: float
    absorption_volume: float
    reclaim_price: float
    timestamp: datetime
    confidence: float
    reversal_type: str  # "bullish" or "bearish"

class StopHuntReversalDetector:
    """
    Stop-hunt reversal confirmation requiring:
    1. Sweep: Price breaks through key level
    2. Absorption: Volume shows absorption at the level
    3. Reclaim: Price reclaims the level with conviction
    """
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
        self.stop_hunt_states = defaultdict(list)
        self.key_levels = defaultdict(list)  # Track key levels for each symbol
        self.volume_threshold_multiplier = 1.5  # Volume must be 1.5x average
        self.reclaim_threshold = 0.0005  # 5 pips reclaim threshold
        
    def detect_stop_hunt_reversal(self, 
                                candles: List[Dict], 
                                symbol: str,
                                order_flow_data: Dict = None) -> Dict:
        """
        Detect stop-hunt reversal pattern with sweep + absorption + reclaim.
        """
        if len(candles) < self.lookback_periods:
            return {"reversal_detected": False, "confidence": 0.0}
        
        # Get recent candles
        recent_candles = candles[-self.lookback_periods:]
        
        # 1. Detect sweep (price breaks through key level)
        sweep_result = self._detect_sweep(recent_candles, symbol)
        
        if not sweep_result["sweep_detected"]:
            return {"reversal_detected": False, "confidence": 0.0}
        
        # 2. Detect absorption (volume shows absorption at the level)
        absorption_result = self._detect_absorption(
            recent_candles, 
            sweep_result["sweep_price"],
            order_flow_data
        )
        
        if not absorption_result["absorption_detected"]:
            return {"reversal_detected": False, "confidence": 0.0}
        
        # 3. Detect reclaim (price reclaims the level with conviction)
        reclaim_result = self._detect_reclaim(
            recent_candles,
            sweep_result["sweep_price"],
            sweep_result["sweep_direction"]
        )
        
        if not reclaim_result["reclaim_detected"]:
            return {"reversal_detected": False, "confidence": 0.0}
        
        # Calculate overall confidence
        confidence = (
            sweep_result["confidence"] * 0.3 +
            absorption_result["confidence"] * 0.4 +
            reclaim_result["confidence"] * 0.3
        )
        
        # Determine reversal type
        reversal_type = "bullish" if sweep_result["sweep_direction"] == "down" else "bearish"
        
        # Create stop-hunt state
        state = StopHuntState(
            sweep_detected=True,
            absorption_detected=True,
            reclaim_detected=True,
            sweep_price=sweep_result["sweep_price"],
            absorption_volume=absorption_result["absorption_volume"],
            reclaim_price=reclaim_result["reclaim_price"],
            timestamp=datetime.utcnow(),
            confidence=confidence,
            reversal_type=reversal_type
        )
        
        # Store state
        self.stop_hunt_states[symbol].append(state)
        if len(self.stop_hunt_states[symbol]) > 100:
            self.stop_hunt_states[symbol] = self.stop_hunt_states[symbol][-100:]
        
        return {
            "reversal_detected": True,
            "confidence": confidence,
            "reversal_type": reversal_type,
            "sweep_price": sweep_result["sweep_price"],
            "reclaim_price": reclaim_result["reclaim_price"],
            "absorption_volume": absorption_result["absorption_volume"],
            "entry_gate_active": confidence > 0.7,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _detect_sweep(self, candles: List[Dict], symbol: str) -> Dict:
        """Detect price sweep through key level."""
        if len(candles) < 10:
            return {"sweep_detected": False, "confidence": 0.0}
        
        # Get key levels for this symbol
        key_levels = self._get_key_levels(candles, symbol)
        
        if not key_levels:
            return {"sweep_detected": False, "confidence": 0.0}
        
        # Check for sweeps in recent candles
        recent_candles = candles[-5:]  # Last 5 candles
        
        for i, candle in enumerate(recent_candles):
            high = candle["high"]
            low = candle["low"]
            
            # Check for resistance sweep (bullish reversal)
            for level in key_levels["resistance"]:
                if high > level["price"] and low < level["price"]:
                    # Price swept through resistance
                    sweep_strength = (high - level["price"]) / level["price"]
                    if sweep_strength > 0.0001:  # At least 1 pip sweep
                        return {
                            "sweep_detected": True,
                            "sweep_price": level["price"],
                            "sweep_direction": "up",
                            "confidence": min(1.0, sweep_strength * 1000),
                            "level_strength": level["strength"]
                        }
            
            # Check for support sweep (bearish reversal)
            for level in key_levels["support"]:
                if low < level["price"] and high > level["price"]:
                    # Price swept through support
                    sweep_strength = (level["price"] - low) / level["price"]
                    if sweep_strength > 0.0001:  # At least 1 pip sweep
                        return {
                            "sweep_detected": True,
                            "sweep_price": level["price"],
                            "sweep_direction": "down",
                            "confidence": min(1.0, sweep_strength * 1000),
                            "level_strength": level["strength"]
                        }
        
        return {"sweep_detected": False, "confidence": 0.0}
    
    def _detect_absorption(self, 
                          candles: List[Dict], 
                          sweep_price: float,
                          order_flow_data: Dict = None) -> Dict:
        """Detect volume absorption at the swept level."""
        if len(candles) < 5:
            return {"absorption_detected": False, "confidence": 0.0}
        
        # Find candles around the sweep price
        relevant_candles = []
        for candle in candles[-10:]:  # Check last 10 candles
            if (candle["low"] <= sweep_price <= candle["high"] or
                abs(candle["close"] - sweep_price) / sweep_price < 0.0005):  # Within 5 pips
                relevant_candles.append(candle)
        
        if not relevant_candles:
            return {"absorption_detected": False, "confidence": 0.0}
        
        # Calculate volume metrics
        volumes = [c.get("tick_volume", 0) for c in relevant_candles]
        avg_volume = np.mean(volumes) if volumes else 0
        
        # Check for high volume (absorption)
        max_volume = max(volumes) if volumes else 0
        volume_ratio = max_volume / avg_volume if avg_volume > 0 else 0
        
        # Check for price rejection (wick formation)
        price_rejection = False
        for candle in relevant_candles:
            if candle["low"] < sweep_price < candle["high"]:
                # Check for long wick (rejection)
                body_size = abs(candle["close"] - candle["open"])
                wick_size = min(candle["high"] - max(candle["open"], candle["close"]),
                               min(candle["open"], candle["close"]) - candle["low"])
                
                if wick_size > body_size * 2:  # Long wick relative to body
                    price_rejection = True
                    break
        
        # Determine absorption
        absorption_detected = (volume_ratio > self.volume_threshold_multiplier and 
                             price_rejection)
        
        confidence = 0.0
        if absorption_detected:
            confidence = min(1.0, (volume_ratio - 1) * 0.5 + 0.5)
        
        return {
            "absorption_detected": absorption_detected,
            "absorption_volume": max_volume,
            "volume_ratio": volume_ratio,
            "price_rejection": price_rejection,
            "confidence": confidence
        }
    
    def _detect_reclaim(self, 
                       candles: List[Dict], 
                       sweep_price: float,
                       sweep_direction: str) -> Dict:
        """Detect price reclaim of the swept level."""
        if len(candles) < 3:
            return {"reclaim_detected": False, "confidence": 0.0}
        
        # Get recent candles after the sweep
        recent_candles = candles[-3:]
        current_price = recent_candles[-1]["close"]
        
        # Check for reclaim based on sweep direction
        if sweep_direction == "up":
            # For resistance sweep, look for price to close below the level
            reclaim_detected = current_price < sweep_price
            reclaim_strength = (sweep_price - current_price) / sweep_price
        else:  # sweep_direction == "down"
            # For support sweep, look for price to close above the level
            reclaim_detected = current_price > sweep_price
            reclaim_strength = (current_price - sweep_price) / sweep_price
        
        # Check for conviction (strong move away from level)
        conviction = reclaim_strength > self.reclaim_threshold
        
        confidence = 0.0
        if reclaim_detected and conviction:
            confidence = min(1.0, reclaim_strength * 2000)  # Scale to 0-1
        
        return {
            "reclaim_detected": reclaim_detected and conviction,
            "reclaim_price": current_price,
            "reclaim_strength": reclaim_strength,
            "conviction": conviction,
            "confidence": confidence
        }
    
    def _get_key_levels(self, candles: List[Dict], symbol: str) -> Dict:
        """Get key support/resistance levels for stop-hunt detection."""
        if len(candles) < 20:
            return {"support": [], "resistance": []}
        
        # Use recent highs and lows as key levels
        recent_candles = candles[-20:]
        highs = [c["high"] for c in recent_candles]
        lows = [c["low"] for c in recent_candles]
        
        # Find significant levels (simplified approach)
        resistance_levels = []
        support_levels = []
        
        # Find resistance levels (recent highs)
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                resistance_levels.append({
                    "price": highs[i],
                    "strength": 0.7,  # Default strength
                    "touches": 1
                })
        
        # Find support levels (recent lows)
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
                lows[i] < lows[i-2] and lows[i] < lows[i+2]):
                support_levels.append({
                    "price": lows[i],
                    "strength": 0.7,  # Default strength
                    "touches": 1
                })
        
        # Update key levels for this symbol
        self.key_levels[symbol] = {
            "support": support_levels,
            "resistance": resistance_levels
        }
        
        return self.key_levels[symbol]
    
    def get_reversal_signals(self, symbol: str) -> Dict:
        """Get recent reversal signals for symbol."""
        recent_states = self.stop_hunt_states.get(symbol, [])
        
        if not recent_states:
            return {"reversals": [], "active_reversals": 0}
        
        # Get recent reversals (last 24 hours)
        now = datetime.utcnow()
        recent_reversals = [
            state for state in recent_states
            if (now - state.timestamp).total_seconds() < 86400  # 24 hours
        ]
        
        # Count active reversals
        active_reversals = len([
            r for r in recent_reversals 
            if r.confidence > 0.7
        ])
        
        return {
            "reversals": [
                {
                    "timestamp": state.timestamp.isoformat(),
                    "type": state.reversal_type,
                    "confidence": state.confidence,
                    "sweep_price": state.sweep_price,
                    "reclaim_price": state.reclaim_price
                }
                for state in recent_reversals
            ],
            "active_reversals": active_reversals,
            "reversal_rate": len(recent_reversals) / max(1, len(recent_states))
        }
    
    def update_key_levels(self, symbol: str, new_levels: Dict):
        """Update key levels for a symbol."""
        self.key_levels[symbol] = new_levels

# Global detector instance
stop_hunt_detector = StopHuntReversalDetector()

def detect_stop_hunt_reversal(candles: List[Dict], 
                            symbol: str,
                            order_flow_data: Dict = None) -> Dict:
    """Detect stop-hunt reversal pattern."""
    return stop_hunt_detector.detect_stop_hunt_reversal(candles, symbol, order_flow_data)

def get_reversal_signals(symbol: str) -> Dict:
    """Get recent reversal signals."""
    return stop_hunt_detector.get_reversal_signals(symbol)