# core/cisd_engine.py

import time
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
import math

class CISDEngine:
    """
    Advanced Change in State of Delivery (CISD) Engine:
    - Dynamic pattern detection with institutional context
    - CISD Pattern Memory for learning from outcomes
    - Smart Delay Validator for time-based confirmation
    - FVG sync detection for Fair Value Gap alignment
    - Time-Filtered CISD validation
    - Flow Tracker for institutional order flow
    - Divergence Scanner for market misalignments
    - Adaptive thresholds based on market regime
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # CISD Pattern Memory
        self.cisd_memory = deque(maxlen=1000)
        self.pattern_performance = defaultdict(list)
        self.success_rates = defaultdict(float)
        
        # Smart Delay Validator
        self.delay_thresholds = {
            "immediate": 0,      # Same candle
            "fast": 1,           # Next candle
            "normal": 2,         # 2 candles
            "slow": 3            # 3+ candles
        }
        self.delay_performance = defaultdict(lambda: defaultdict(list))
        
        # FVG Sync Detection
        self.fvg_memory = deque(maxlen=200)
        self.fvg_sync_threshold = 0.7
        self.fvg_performance = defaultdict(list)
        
        # Time-Filtered Validation
        self.time_windows = {
            "london_open": (8, 12),      # 08:00-12:00 UTC
            "ny_open": (13, 17),         # 13:00-17:00 UTC
            "asian_session": (0, 8),     # 00:00-08:00 UTC
            "london_close": (16, 20),    # 16:00-20:00 UTC
            "ny_close": (21, 1)          # 21:00-01:00 UTC
        }
        self.session_performance = defaultdict(lambda: defaultdict(list))
        
        # Flow Tracker
        self.flow_memory = deque(maxlen=500)
        self.institutional_thresholds = {
            "whale_order": 1000000,      # $1M+ orders
            "block_trade": 500000,       # $500K+ block trades
            "absorption_ratio": 0.3,     # 30% absorption threshold
            "flow_imbalance": 0.6        # 60% imbalance threshold
        }
        
        # Divergence Scanner
        self.divergence_memory = deque(maxlen=300)
        self.divergence_types = ["price_momentum", "volume_price", "rsi_price", "macd_price"]
        self.divergence_performance = defaultdict(list)
        
        # Market Regime Adaptation
        self.regime_thresholds = {
            "quiet": {"cisd_strength": 0.6, "delay_tolerance": 2},
            "normal": {"cisd_strength": 0.7, "delay_tolerance": 1},
            "trending": {"cisd_strength": 0.8, "delay_tolerance": 0},
            "volatile": {"cisd_strength": 0.9, "delay_tolerance": 3}
        }
        self.current_regime = "normal"
        
        # Performance Tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.learning_rate = 0.01

    def detect_cisd(
        self,
        candles: List[Dict],
        structure_data: Dict,
        order_flow_data: Dict,
        market_context: Dict,
        time_context: Dict
    ) -> Dict:
        """
        Main CISD detection with institutional-grade analysis
        """
        if len(candles) < 10:
            return self._create_cisd_response(False, "Insufficient data for CISD analysis")
        
        # Extract key data
        current_candle = candles[-1]
        prev_candles = candles[-10:-1]
        
        # Core CISD Detection
        cisd_patterns = self._detect_cisd_patterns(candles, structure_data)
        
        # Smart Delay Validation
        delay_validation = self._validate_smart_delay(candles, cisd_patterns)
        
        # FVG Sync Detection
        fvg_sync = self._detect_fvg_sync(candles, cisd_patterns)
        
        # Time-Filtered Validation
        time_validation = self._validate_time_filtered(cisd_patterns, time_context)
        
        # Flow Analysis
        flow_analysis = self._analyze_institutional_flow(order_flow_data, cisd_patterns)
        
        # Divergence Scanning
        divergence_scan = self._scan_divergences(candles, market_context)
        
        # Composite CISD Score
        cisd_score = self._calculate_composite_score(
            cisd_patterns, delay_validation, fvg_sync, 
            time_validation, flow_analysis, divergence_scan
        )
        
        # Regime Adaptation
        adapted_threshold = self._adapt_to_regime(cisd_score, market_context)
        
        # Optional OB/FVG retest gating for CHoCH contexts
        require_retest = bool(self.config.get("require_ob_fvg_retest", False))
        retest_ok = True
        if require_retest and structure_data and structure_data.get("event") == "CHoCH":
            # If FVG sync not detected, fall back to simple OB wick-zone retest heuristic
            retest_ok = fvg_sync.get("detected", False) or self._simple_ob_retest(candles)

        # Final CISD Decision
        cisd_valid = (cisd_score >= adapted_threshold) and retest_ok
        
        # Update Memory
        self._update_cisd_memory(cisd_patterns, cisd_valid, market_context)
        
        return self._create_cisd_response(
            cisd_valid, 
            cisd_score, 
            cisd_patterns, 
            delay_validation, 
            fvg_sync, 
            time_validation, 
            flow_analysis, 
            divergence_scan,
            adapted_threshold
        )

    def _simple_ob_retest(self, candles: List[Dict]) -> bool:
        """Lightweight OB retest heuristic to avoid conflicts.
        Detects last strong impulse candle and checks if price later retested its body.
        """
        if len(candles) < 6:
            return True  # don't block when insufficient data
        last5 = candles[-6:-1]
        # pick largest body candle as OB proxy
        def body(c):
            return abs(float(c.get("close",0)) - float(c.get("open",0)))
        ob = max(last5, key=body)
        ob_low = min(ob.get("open",0.0), ob.get("close",0.0))
        ob_high = max(ob.get("open",0.0), ob.get("close",0.0))
        # Did subsequent candle touch OB body?
        last = candles[-1]
        touched = (float(last.get("low",0.0)) <= ob_high and float(last.get("high",0.0)) >= ob_low)
        return touched
    
    def _detect_cisd_patterns(self, candles: List[Dict], structure_data: Dict) -> Dict:
        """
        Detect CISD patterns with institutional context awareness
        """
        patterns = {
            "reversal": False,
            "continuation": False,
            "breakout": False,
            "breakdown": False,
            "strength": 0.0,
            "confidence": 0.0,
            "context": {}
        }
        
        if len(candles) < 5:
            return patterns
        
        # Get recent candles
        c1, c2, c3, c4, c5 = candles[-5:]
        
        # Reversal Pattern Detection
        if self._detect_reversal_pattern(c1, c2, c3, c4, c5):
            patterns["reversal"] = True
            patterns["strength"] += 0.4
            patterns["context"]["reversal_type"] = "strong_reversal"
        
        # Continuation Pattern Detection
        if self._detect_continuation_pattern(c1, c2, c3, c4, c5):
            patterns["continuation"] = True
            patterns["strength"] += 0.3
            patterns["context"]["continuation_type"] = "momentum_continuation"
        
        # Breakout/Breakdown Detection
        breakout_info = self._detect_breakout_pattern(candles, structure_data)
        if breakout_info["detected"]:
            patterns["breakout"] = breakout_info["type"] == "breakout"
            patterns["breakdown"] = breakout_info["type"] == "breakdown"
            patterns["strength"] += breakout_info["strength"]
            patterns["context"]["breakout_context"] = breakout_info["context"]
        
        # Structure Alignment
        if structure_data:
            structure_score = self._analyze_structure_alignment(structure_data, patterns)
            patterns["strength"] += structure_score
            patterns["context"]["structure_alignment"] = structure_score
        
        # Normalize strength to 0-1 range
        patterns["strength"] = min(1.0, patterns["strength"])
        patterns["confidence"] = patterns["strength"] * 0.8 + 0.2  # Base confidence
        
        return patterns
    
    def _detect_reversal_pattern(self, c1, c2, c3, c4, c5) -> bool:
        """
        Detect strong reversal patterns with institutional logic
        """
        # Engulfing pattern
        if (c4["high"] > c3["high"] and c4["low"] < c3["low"] and
            c4["close"] > c4["open"] and c3["close"] < c3["open"]):
            return True
        
        # Hammer/Shooting Star with confirmation
        if (c3["high"] - c3["low"]) > 2 * abs(c3["close"] - c3["open"]):
            if c4["close"] > c3["high"] or c4["close"] < c3["low"]:
                return True
        
        # Three White Soldiers / Three Black Crows
        if (c3["close"] > c3["open"] and c4["close"] > c4["open"] and c5["close"] > c5["open"] and
            c4["close"] > c3["close"] and c5["close"] > c4["close"]):
            return True
        
        if (c3["close"] < c3["open"] and c4["close"] < c4["open"] and c5["close"] < c5["open"] and
            c4["close"] < c3["close"] and c5["close"] < c4["close"]):
            return True
        
        return False
    
    def _detect_continuation_pattern(self, c1, c2, c3, c4, c5) -> bool:
        """
        Detect continuation patterns with momentum confirmation
        """
        # Flag pattern
        if (c3["high"] < c2["high"] and c3["low"] > c2["low"] and
            c4["close"] > c3["high"] and c5["close"] > c4["close"]):
            return True
        
        # Pennant pattern
        if (c3["high"] < c2["high"] and c3["low"] > c2["low"] and
            c4["high"] < c3["high"] and c4["low"] > c3["low"] and
            c5["close"] > c4["high"]):
            return True
        
        return False
    
    def _detect_breakout_pattern(self, candles: List[Dict], structure_data: Dict) -> Dict:
        """
        Detect breakout/breakdown patterns with volume confirmation
        """
        result = {
            "detected": False,
            "type": None,
            "strength": 0.0,
            "context": {}
        }
        
        if len(candles) < 20:
            return result
        
        # Calculate recent range
        recent_high = max(c["high"] for c in candles[-20:])
        recent_low = min(c["low"] for c in candles[-20:])
        range_size = recent_high - recent_low
        
        current_candle = candles[-1]
        
        # Breakout detection
        if current_candle["close"] > recent_high + (range_size * 0.1):
            result["detected"] = True
            result["type"] = "breakout"
            result["strength"] = 0.6
            result["context"]["breakout_level"] = recent_high
            result["context"]["breakout_strength"] = "strong"
        
        # Breakdown detection
        elif current_candle["close"] < recent_low - (range_size * 0.1):
            result["detected"] = True
            result["type"] = "breakdown"
            result["strength"] = 0.6
            result["context"]["breakdown_level"] = recent_low
            result["context"]["breakdown_strength"] = "strong"
        
        return result
    
    def _analyze_structure_alignment(self, structure_data: Dict, patterns: Dict) -> float:
        """
        Analyze how well CISD patterns align with market structure
        """
        score = 0.0
        
        if structure_data.get("event") == "FLIP":
            if patterns["reversal"]:
                score += 0.3
            elif patterns["continuation"]:
                score += 0.1
        
        if structure_data.get("event") == "CHoCH":
            if patterns["breakout"] or patterns["breakdown"]:
                score += 0.3
            elif patterns["reversal"]:
                score += 0.2
            # Enhance with structural context flags if present
            if structure_data.get("momentum_reduction"):
                score += 0.05
            if structure_data.get("failure_swing"):
                score += 0.05
            if structure_data.get("post_choch_expansion"):
                score += 0.07
        
        if structure_data.get("event") == "BOS":
            if patterns["continuation"]:
                score += 0.2
        
        return score

    def _validate_smart_delay(self, candles: List[Dict], cisd_patterns: Dict) -> Dict:
        """
        Smart delay validation based on pattern type and market conditions
        """
        validation = {
            "validated": False,
            "delay_type": "unknown",
            "delay_candles": 0,
            "confidence": 0.0,
            "context": {}
        }
        
        if not cisd_patterns["strength"] > 0.5:
            return validation
        
        # Determine optimal delay based on pattern type
        if cisd_patterns["reversal"]:
            optimal_delay = "normal"  # 2 candles for reversal confirmation
        elif cisd_patterns["continuation"]:
            optimal_delay = "fast"    # 1 candle for continuation
        elif cisd_patterns["breakout"] or cisd_patterns["breakdown"]:
            optimal_delay = "immediate"  # Immediate for breakouts
        else:
            optimal_delay = "normal"
        
        # Check if delay requirement is met
        delay_candles = self._calculate_delay_candles(candles, cisd_patterns)
        required_delay = self.delay_thresholds[optimal_delay]
        
        if delay_candles >= required_delay:
            validation["validated"] = True
            validation["delay_type"] = optimal_delay
            validation["delay_candles"] = delay_candles
            validation["confidence"] = 0.8
            validation["context"]["delay_requirement"] = f"{required_delay} candles"
            validation["context"]["actual_delay"] = f"{delay_candles} candles"
        
        return validation
    
    def _calculate_delay_candles(self, candles: List[Dict], cisd_patterns: Dict) -> int:
        """
        Calculate how many candles have passed since CISD pattern formation
        """
        if len(candles) < 3:
            return 0
        
        # Look for the pattern formation candle
        for i in range(len(candles) - 3, 0, -1):
            if self._is_pattern_formation_candle(candles[i:i+3], cisd_patterns):
                return len(candles) - i - 1
        
        return 0
    
    def _is_pattern_formation_candle(self, three_candles: List[Dict], patterns: Dict) -> bool:
        """
        Check if three candles form the CISD pattern
        """
        if len(three_candles) < 3:
            return False
        
        c1, c2, c3 = three_candles
        
        # Simple pattern check - can be enhanced
        if patterns["reversal"]:
            return (c2["high"] > c1["high"] and c2["low"] < c1["low"])
        elif patterns["continuation"]:
            return (c2["high"] < c1["high"] and c2["low"] > c1["low"])
        
        return False
    
    def _detect_fvg_sync(self, candles: List[Dict], cisd_patterns: Dict) -> Dict:
        """
        Detect Fair Value Gap synchronization with CISD patterns
        """
        fvg_sync = {
            "detected": False,
            "sync_strength": 0.0,
            "fvg_count": 0,
            "alignment_score": 0.0,
            "context": {}
        }
        
        if len(candles) < 5:
            return fvg_sync
        
        # Detect Fair Value Gaps
        fvgs = self._find_fair_value_gaps(candles)
        
        if not fvgs:
            return fvg_sync
        
        # Check alignment with CISD patterns
        alignment_score = self._calculate_fvg_alignment(fvgs, cisd_patterns)
        
        if alignment_score > self.fvg_sync_threshold:
            fvg_sync["detected"] = True
            fvg_sync["sync_strength"] = alignment_score
            fvg_sync["fvg_count"] = len(fvgs)
            fvg_sync["alignment_score"] = alignment_score
            fvg_sync["context"]["fvg_details"] = fvgs
        
        return fvg_sync
    
    def _find_fair_value_gaps(self, candles: List[Dict]) -> List[Dict]:
        """
        Find Fair Value Gaps in the candle data
        """
        fvgs = []
        
        for i in range(1, len(candles) - 1):
            prev_candle = candles[i-1]
            current_candle = candles[i]
            next_candle = candles[i+1]
            
            # Bullish FVG
            if (current_candle["low"] > prev_candle["high"] and
                next_candle["low"] < current_candle["low"]):
                fvgs.append({
                    "type": "bullish",
                    "start": prev_candle["high"],
                    "end": current_candle["low"],
                    "strength": current_candle["low"] - prev_candle["high"],
                    "position": i
                })
            
            # Bearish FVG
            elif (current_candle["high"] < prev_candle["low"] and
                  next_candle["high"] > current_candle["high"]):
                fvgs.append({
                    "type": "bearish",
                    "start": current_candle["high"],
                    "end": prev_candle["low"],
                    "strength": prev_candle["low"] - current_candle["high"],
                    "position": i
                })
        
        return fvgs
    
    def _calculate_fvg_alignment(self, fvgs: List[Dict], cisd_patterns: Dict) -> float:
        """
        Calculate how well FVGs align with CISD patterns
        """
        if not fvgs or not cisd_patterns["strength"]:
            return 0.0
        
        alignment_score = 0.0
        
        for fvg in fvgs:
            if cisd_patterns["reversal"]:
                if fvg["type"] == "bullish" and cisd_patterns.get("context", {}).get("reversal_type") == "strong_reversal":
                    alignment_score += 0.3
                elif fvg["type"] == "bearish" and cisd_patterns.get("context", {}).get("reversal_type") == "strong_reversal":
                    alignment_score += 0.3
            
            if cisd_patterns["breakout"] and fvg["type"] == "bullish":
                alignment_score += 0.2
            
            if cisd_patterns["breakdown"] and fvg["type"] == "bearish":
                alignment_score += 0.2
        
        return min(1.0, alignment_score)
    
    def _validate_time_filtered(self, cisd_patterns: Dict, time_context: Dict) -> Dict:
        """
        Time-filtered CISD validation based on session context
        """
        time_validation = {
            "validated": False,
            "session_score": 0.0,
            "time_alignment": 0.0,
            "optimal_session": None,
            "context": {}
        }
        
        if not time_context:
            return time_validation
        
        current_hour = time_context.get("hour", 0)
        current_session = self._identify_session(current_hour)
        
        # Calculate session performance score
        session_score = self._calculate_session_score(current_session, cisd_patterns)
        
        # Calculate time alignment score
        time_alignment = self._calculate_time_alignment(current_hour, cisd_patterns)
        
        # Determine optimal session for this pattern type
        optimal_session = self._find_optimal_session(cisd_patterns)
        
        # Composite time validation score
        composite_score = (session_score * 0.6) + (time_alignment * 0.4)
        
        if composite_score > 0.6:
            time_validation["validated"] = True
            time_validation["session_score"] = session_score
            time_validation["time_alignment"] = time_alignment
            time_validation["optimal_session"] = optimal_session
            time_validation["context"]["current_session"] = current_session
            time_validation["context"]["composite_score"] = composite_score
        
        return time_validation
    
    def _identify_session(self, hour: int) -> str:
        """
        Identify current trading session
        """
        if 0 <= hour < 8:
            return "asian_session"
        elif 8 <= hour < 12:
            return "london_open"
        elif 13 <= hour < 17:
            return "ny_open"
        elif 16 <= hour < 20:
            return "london_close"
        elif 21 <= hour <= 23 or hour == 0:
            return "ny_close"
        else:
            return "unknown"
    
    def _calculate_session_score(self, session: str, cisd_patterns: Dict) -> float:
        """
        Calculate session performance score for CISD patterns
        """
        if session not in self.session_performance:
            return 0.5  # Default neutral score
        
        session_data = self.session_performance[session]
        if not session_data:
            return 0.5
        
        # Calculate success rate for this session
        success_rate = sum(1 for result in session_data if result["success"]) / len(session_data)
        
        # Adjust based on pattern type
        if cisd_patterns["reversal"]:
            return success_rate * 0.8 + 0.2
        elif cisd_patterns["continuation"]:
            return success_rate * 0.7 + 0.3
        else:
            return success_rate
    
    def _calculate_time_alignment(self, hour: int, cisd_patterns: Dict) -> float:
        """
        Calculate time alignment score for CISD patterns
        """
        # Simple time-based scoring - can be enhanced
        if 8 <= hour <= 17:  # London + NY overlap
            return 0.9
        elif 0 <= hour < 8:  # Asian session
            return 0.7
        elif 18 <= hour <= 23:  # Evening session
            return 0.6
        else:
            return 0.5
    
    def _find_optimal_session(self, cisd_patterns: Dict) -> str:
        """
        Find optimal trading session for CISD pattern type
        """
        if cisd_patterns["reversal"]:
            return "london_open"  # Reversals often happen at session opens
        elif cisd_patterns["breakout"] or cisd_patterns["breakdown"]:
            return "ny_open"  # Breakouts often happen during NY session
        elif cisd_patterns["continuation"]:
            return "asian_session"  # Continuations often happen during Asian session
        else:
            return "london_open"  # Default to London open
    
    def _analyze_institutional_flow(self, order_flow_data: Dict, cisd_patterns: Dict) -> Dict:
        """
        Analyze institutional order flow for CISD validation
        """
        flow_analysis = {
            "validated": False,
            "flow_strength": 0.0,
            "institutional_activity": "low",
            "absorption_score": 0.0,
            "imbalance_score": 0.0,
            "context": {}
        }
        
        if not order_flow_data:
            return flow_analysis
        
        # Extract flow metrics
        volume_total = order_flow_data.get("volume_total", 0)
        delta = order_flow_data.get("delta", 0)
        absorption = order_flow_data.get("absorption", False)
        
        if volume_total == 0:
            return flow_analysis
        
        # Calculate institutional activity level
        if volume_total > self.institutional_thresholds["whale_order"]:
            flow_analysis["institutional_activity"] = "high"
            flow_analysis["flow_strength"] += 0.4
        elif volume_total > self.institutional_thresholds["block_trade"]:
            flow_analysis["institutional_activity"] = "medium"
            flow_analysis["flow_strength"] += 0.2
        
        # Calculate absorption score
        if absorption:
            flow_analysis["absorption_score"] = 0.8
            flow_analysis["flow_strength"] += 0.3
        
        # Calculate imbalance score
        imbalance_ratio = abs(delta) / volume_total
        if imbalance_ratio > self.institutional_thresholds["flow_imbalance"]:
            flow_analysis["imbalance_score"] = imbalance_ratio
            flow_analysis["flow_strength"] += 0.3
        
        # Validate flow alignment with CISD patterns
        if flow_analysis["flow_strength"] > 0.5:
            flow_analysis["validated"] = True
            flow_analysis["context"]["flow_metrics"] = {
                "volume": volume_total,
                "delta": delta,
                "absorption": absorption,
                "imbalance_ratio": imbalance_ratio
            }
        
        return flow_analysis
    
    def _scan_divergences(self, candles: List[Dict], market_context: Dict) -> Dict:
        """
        Scan for price and indicator divergences
        """
        divergence_scan = {
            "detected": False,
            "divergence_types": [],
            "strength": 0.0,
            "confidence": 0.0,
            "context": {}
        }
        
        if len(candles) < 20:
            return divergence_scan
        
        # Price-Momentum Divergence
        price_momentum = self._detect_price_momentum_divergence(candles)
        if price_momentum["detected"]:
            divergence_scan["divergence_types"].append("price_momentum")
            divergence_scan["strength"] += price_momentum["strength"]
        
        # Volume-Price Divergence
        volume_price = self._detect_volume_price_divergence(candles)
        if volume_price["detected"]:
            divergence_scan["divergence_types"].append("volume_price")
            divergence_scan["strength"] += volume_price["strength"]
        
        # RSI Divergence (if available)
        if market_context.get("indicators", {}).get("rsi"):
            rsi_divergence = self._detect_rsi_divergence(candles, market_context["indicators"]["rsi"])
            if rsi_divergence["detected"]:
                divergence_scan["divergence_types"].append("rsi_price")
                divergence_scan["strength"] += rsi_divergence["strength"]
        
        # MACD Divergence (if available)
        if market_context.get("indicators", {}).get("macd"):
            macd_divergence = self._detect_macd_divergence(candles, market_context["indicators"]["macd"])
            if macd_divergence["detected"]:
                divergence_scan["divergence_types"].append("macd_price")
                divergence_scan["strength"] += macd_divergence["strength"]
        
        # Set overall divergence status
        if divergence_scan["divergence_types"]:
            divergence_scan["detected"] = True
            divergence_scan["confidence"] = min(1.0, divergence_scan["strength"] * 0.8)
            divergence_scan["context"]["divergence_details"] = {
                "price_momentum": price_momentum,
                "volume_price": volume_price
            }
        
        return divergence_scan
    
    def _detect_price_momentum_divergence(self, candles: List[Dict]) -> Dict:
        """
        Detect price-momentum divergence
        """
        result = {"detected": False, "strength": 0.0, "type": None}
        
        if len(candles) < 20:
            return result
        
        # Calculate price highs and momentum
        price_highs = []
        momentum_values = []
        
        for i in range(5, len(candles)):
            if candles[i]["high"] > candles[i-1]["high"] and candles[i]["high"] > candles[i-2]["high"]:
                price_highs.append((i, candles[i]["high"]))
                
                # Calculate momentum (rate of change)
                momentum = (candles[i]["close"] - candles[i-5]["close"]) / candles[i-5]["close"]
                momentum_values.append(momentum)
        
        if len(price_highs) < 2:
            return result
        
        # Check for divergence
        if len(price_highs) >= 2 and len(momentum_values) >= 2:
            price_trend = price_highs[-1][1] > price_highs[-2][1]
            momentum_trend = momentum_values[-1] > momentum_values[-2]
            
            if price_trend != momentum_trend:
                result["detected"] = True
                result["strength"] = 0.7
                result["type"] = "bearish" if price_trend and not momentum_trend else "bullish"
        
        return result
    
    def _detect_volume_price_divergence(self, candles: List[Dict]) -> Dict:
        """
        Detect volume-price divergence
        """
        result = {"detected": False, "strength": 0.0, "type": None}
        
        if len(candles) < 20:
            return result
        
        # Calculate price and volume trends
        price_trend = candles[-1]["close"] > candles[-10]["close"]
        
        # Simple volume trend (assuming tick_volume is available)
        recent_volume = sum(c.get("tick_volume", 0) for c in candles[-5:])
        earlier_volume = sum(c.get("tick_volume", 0) for c in candles[-15:-10])
        
        if earlier_volume == 0:
            return result
        
        volume_trend = recent_volume > earlier_volume
        
        # Check for divergence
        if price_trend != volume_trend:
            result["detected"] = True
            result["strength"] = 0.6
            result["type"] = "bearish" if price_trend and not volume_trend else "bullish"
        
        return result
    
    def _detect_rsi_divergence(self, candles: List[Dict], rsi_values: List[float]) -> Dict:
        """
        Detect RSI divergence (placeholder - needs RSI data)
        """
        return {"detected": False, "strength": 0.0, "type": None}
    
    def _detect_macd_divergence(self, candles: List[Dict], macd_values: List[float]) -> Dict:
        """
        Detect MACD divergence (placeholder - needs MACD data)
        """
        return {"detected": False, "strength": 0.0, "type": None}
    
    def _calculate_composite_score(
        self,
        cisd_patterns: Dict,
        delay_validation: Dict,
        fvg_sync: Dict,
        time_validation: Dict,
        flow_analysis: Dict,
        divergence_scan: Dict
    ) -> float:
        """
        Calculate composite CISD score from all components
        """
        score = 0.0
        weights = {
            "patterns": 0.3,
            "delay": 0.2,
            "fvg": 0.15,
            "time": 0.15,
            "flow": 0.1,
            "divergence": 0.1
        }
        
        # Pattern strength
        score += cisd_patterns["strength"] * weights["patterns"]
        
        # Delay validation
        if delay_validation["validated"]:
            score += delay_validation["confidence"] * weights["delay"]
        
        # FVG sync
        if fvg_sync["detected"]:
            score += fvg_sync["sync_strength"] * weights["fvg"]
        
        # Time validation
        if time_validation["validated"]:
            score += (time_validation["session_score"] + time_validation["time_alignment"]) / 2 * weights["time"]
        
        # Flow analysis
        if flow_analysis["validated"]:
            score += flow_analysis["flow_strength"] * weights["flow"]
        
        # Divergence scan
        if divergence_scan["detected"]:
            score += divergence_scan["confidence"] * weights["divergence"]
        
        return min(1.0, score)
    
    def _adapt_to_regime(self, cisd_score: float, market_context: Dict) -> float:
        """
        Adapt CISD threshold based on market regime
        """
        regime = market_context.get("regime", "normal")
        base_threshold = self.regime_thresholds[regime]["cisd_strength"]
        
        # Adjust based on volatility
        volatility = market_context.get("volatility", "normal")
        if volatility == "high":
            base_threshold += 0.1
        elif volatility == "low":
            base_threshold -= 0.1
        
        # Adjust based on trend strength
        trend_strength = market_context.get("trend_strength", 0.5)
        if trend_strength > 0.7:
            base_threshold -= 0.05  # Easier to confirm in strong trends
        elif trend_strength < 0.3:
            base_threshold += 0.05  # Harder to confirm in weak trends
        
        return max(0.5, min(0.95, base_threshold))
    
    def _update_cisd_memory(self, cisd_patterns: Dict, cisd_valid: bool, market_context: Dict):
        """
        Update CISD memory for learning and performance tracking
        """
        memory_entry = {
            "timestamp": time.time(),
            "patterns": cisd_patterns,
            "valid": cisd_valid,
            "market_context": market_context,
            "regime": market_context.get("regime", "unknown")
        }
        
        self.cisd_memory.append(memory_entry)
        
        # Update pattern performance
        pattern_type = "reversal" if cisd_patterns["reversal"] else "continuation"
        if pattern_type in ["reversal", "continuation"]:
            self.pattern_performance[pattern_type].append(cisd_valid)
            
            # Calculate success rate
            if len(self.pattern_performance[pattern_type]) > 10:
                success_rate = sum(self.pattern_performance[pattern_type][-10:]) / 10
                self.success_rates[pattern_type] = success_rate
    
    def _create_cisd_response(
        self,
        cisd_valid: bool,
        cisd_score: float = 0.0,
        cisd_patterns: Dict = None,
        delay_validation: Dict = None,
        fvg_sync: Dict = None,
        time_validation: Dict = None,
        flow_analysis: Dict = None,
        divergence_scan: Dict = None,
        adapted_threshold: float = 0.0
    ) -> Dict:
        """
        Create comprehensive CISD response
        """
        response = {
            "cisd_valid": cisd_valid,
            "cisd_score": cisd_score,
            "adapted_threshold": adapted_threshold,
            "confidence": "high" if cisd_score > 0.8 else "medium" if cisd_score > 0.6 else "low",
            "timestamp": time.time(),
            "components": {
                "patterns": cisd_patterns or {},
                "delay_validation": delay_validation or {},
                "fvg_sync": fvg_sync or {},
                "time_validation": time_validation or {},
                "flow_analysis": flow_analysis or {},
                "divergence_scan": divergence_scan or {}
            },
            "summary": {
                "pattern_strength": cisd_patterns["strength"] if cisd_patterns else 0.0,
                "delay_validated": delay_validation["validated"] if delay_validation else False,
                "fvg_synced": fvg_sync["detected"] if fvg_sync else False,
                "time_validated": time_validation["validated"] if time_validation else False,
                "flow_validated": flow_analysis["validated"] if flow_analysis else False,
                "divergence_detected": divergence_scan["detected"] if divergence_scan else False
            },
            "performance_metrics": {
                "total_signals": self.total_signals,
                "successful_signals": self.successful_signals,
                "success_rate": self.successful_signals / max(1, self.total_signals),
                "pattern_success_rates": dict(self.success_rates)
            }
        }
        
        return response
    
    def get_cisd_stats(self) -> Dict:
        """
        Get CISD performance statistics
        """
        return {
            "total_signals": self.total_signals,
            "successful_signals": self.successful_signals,
            "success_rate": self.successful_signals / max(1, self.total_signals),
            "pattern_success_rates": dict(self.success_rates),
            "regime_performance": dict(self.regime_thresholds),
            "memory_size": len(self.cisd_memory)
        }
    
    def update_performance(self, signal_id: str, outcome: bool, pnl: float = 0.0):
        """
        Update performance tracking for a CISD signal
        """
        self.total_signals += 1
        if outcome:
            self.successful_signals += 1
        
        # Update regime-specific performance
        if hasattr(self, 'current_regime'):
            regime = self.current_regime
            if regime not in self.regime_thresholds:
                self.regime_thresholds[regime] = {"cisd_strength": 0.7, "delay_tolerance": 1}
            
            # Adjust threshold based on performance
            if self.total_signals % 10 == 0:  # Every 10 signals
                success_rate = self.successful_signals / self.total_signals
                if success_rate < 0.4:
                    self.regime_thresholds[regime]["cisd_strength"] += 0.05
                elif success_rate > 0.7:
                    self.regime_thresholds[regime]["cisd_strength"] -= 0.02
                
                # Ensure thresholds stay in reasonable range
                self.regime_thresholds[regime]["cisd_strength"] = max(0.5, min(0.95, 
                    self.regime_thresholds[regime]["cisd_strength"]))
