# core/multi_timeframe.py

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.signal import argrelextrema
from collections import defaultdict

@dataclass
class FractalPattern:
    type: str  # "top" or "bottom"
    price: float
    timestamp: datetime
    timeframe: str
    strength: float  # 0 to 1
    confirmation: int  # number of confirming candles

@dataclass
class StructureLevel:
    type: str  # "support", "resistance", "swing_high", "swing_low"
    price: float
    timeframe: str
    strength: float
    touches: int
    last_touch: datetime
    fractal_confirmed: bool

class MultiTimeframeAnalyzer:
    """
    Advanced multi-timeframe analysis:
    - Fractal pattern detection
    - Nested support/resistance
    - Timeframe confluence
    - Structure mapping
    - Harmonic patterns
    - Institutional levels
    """
    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ["M5", "M15", "H1", "H4", "D1"]
        self.fractal_patterns = defaultdict(list)  # by timeframe
        self.structure_levels = defaultdict(list)  # by timeframe
        self.confluences = {}
        
        # Tracking
        self.last_update = None
        self.current_bias = "neutral"
        self.confidence = 0.0
        
    def analyze(self, candles_by_timeframe: Dict[str, List[Dict]]) -> Dict:
        """
        Perform complete multi-timeframe analysis
        Returns comprehensive analysis including fractals, structure, and confluences
        """
        if not all(tf in candles_by_timeframe for tf in self.timeframes):
            return self._empty_analysis()
            
        # Clear old patterns
        self._clear_old_patterns()
        
        # Analyze each timeframe
        for tf in self.timeframes:
            candles = candles_by_timeframe[tf]
            if len(candles) < 10:  # minimum required candles
                continue
                
            # Detect patterns
            self._detect_fractals(candles, tf)
            self._analyze_structure(candles, tf)
            
        # Find confluences across timeframes
        self._find_confluences()
        
        # Generate complete analysis
        return self._generate_analysis()
        
    def _detect_fractals(self, candles: List[Dict], timeframe: str):
        """
        Detect fractal patterns using advanced recognition
        Includes strength and confirmation metrics
        """
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        
        # Find local extrema
        high_idx = argrelextrema(highs, np.greater, order=2)[0]
        low_idx = argrelextrema(lows, np.less, order=2)[0]
        
        # Process highs
        for idx in high_idx:
            if idx + 2 >= len(candles):  # need confirmation
                continue
                
            # Calculate pattern strength
            strength = self._calculate_fractal_strength(candles, idx, "top")
            
            # Count confirming candles
            confirmation = 0
            for i in range(idx + 1, len(candles)):
                if candles[i]["high"] < highs[idx]:
                    confirmation += 1
                else:
                    break
                    
            pattern = FractalPattern(
                type="top",
                price=highs[idx],
                timestamp=datetime.fromisoformat(candles[idx]["time"]),
                timeframe=timeframe,
                strength=strength,
                confirmation=confirmation
            )
            self.fractal_patterns[timeframe].append(pattern)
            
        # Process lows
        for idx in low_idx:
            if idx + 2 >= len(candles):
                continue
                
            strength = self._calculate_fractal_strength(candles, idx, "bottom")
            
            confirmation = 0
            for i in range(idx + 1, len(candles)):
                if candles[i]["low"] > lows[idx]:
                    confirmation += 1
                else:
                    break
                    
            pattern = FractalPattern(
                type="bottom",
                price=lows[idx],
                timestamp=datetime.fromisoformat(candles[idx]["time"]),
                timeframe=timeframe,
                strength=strength,
                confirmation=confirmation
            )
            self.fractal_patterns[timeframe].append(pattern)
            
    def _calculate_fractal_strength(
        self,
        candles: List[Dict],
        idx: int,
        pattern_type: str
    ) -> float:
        """
        Calculate fractal pattern strength based on:
        - Price movement magnitude
        - Volume confirmation
        - Surrounding structure
        """
        if idx < 2 or idx + 2 >= len(candles):
            return 0.0
            
        strength = 0.0
        
        # Price movement
        if pattern_type == "top":
            height = candles[idx]["high"] - min(
                candles[i]["low"] for i in range(idx-2, idx+3)
            )
            avg_range = np.mean([
                c["high"] - c["low"] for c in candles[idx-2:idx+3]
            ])
            strength += min(height / avg_range, 2.0) * 0.4  # 40% weight
            
        else:  # bottom
            height = max(
                candles[i]["high"] for i in range(idx-2, idx+3)
            ) - candles[idx]["low"]
            avg_range = np.mean([
                c["high"] - c["low"] for c in candles[idx-2:idx+3]
            ])
            strength += min(height / avg_range, 2.0) * 0.4
            
        # Volume confirmation
        vol_idx = candles[idx].get("tick_volume", 0)
        avg_vol = np.mean([
            c.get("tick_volume", 0) for c in candles[idx-2:idx+3]
        ])
        if vol_idx > avg_vol:
            strength += min(vol_idx / avg_vol, 2.0) * 0.3  # 30% weight
            
        # Clean structure (no overlapping wicks)
        if pattern_type == "top":
            clean = all(
                candles[i]["high"] < candles[idx]["high"]
                for i in range(idx-2, idx+3)
                if i != idx
            )
        else:
            clean = all(
                candles[i]["low"] > candles[idx]["low"]
                for i in range(idx-2, idx+3)
                if i != idx
            )
            
        if clean:
            strength += 0.3  # 30% weight
            
        return min(strength, 1.0)
        
    def _analyze_structure(self, candles: List[Dict], timeframe: str):
        """
        Analyze market structure including:
        - Support/resistance levels
        - Swing points
        - Level strength and touches
        """
        # Clear old levels
        self.structure_levels[timeframe] = [
            level for level in self.structure_levels[timeframe]
            if (datetime.utcnow() - level.last_touch).days < 30
        ]
        
        # Find new levels
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        
        # Detect swings using zigzag
        swing_points = self._detect_swings(highs, lows)
        
        for point in swing_points:
            price = point["price"]
            is_high = point["type"] == "high"
            
            # Check if level exists
            existing = None
            for level in self.structure_levels[timeframe]:
                if abs(level.price - price) / price < 0.0001:  # 0.01% tolerance
                    existing = level
                    break
                    
            if existing:
                existing.touches += 1
                existing.last_touch = datetime.fromisoformat(
                    candles[point["index"]]["time"]
                )
                existing.strength = min(
                    existing.strength + 0.1,
                    1.0
                )
            else:
                level = StructureLevel(
                    type="resistance" if is_high else "support",
                    price=price,
                    timeframe=timeframe,
                    strength=0.5,  # initial strength
                    touches=1,
                    last_touch=datetime.fromisoformat(
                        candles[point["index"]]["time"]
                    ),
                    fractal_confirmed=False
                )
                self.structure_levels[timeframe].append(level)
                
    def _detect_swings(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        threshold: float = 0.0003
    ) -> List[Dict]:
        """
        Detect swing points using zigzag algorithm
        Returns list of swing points with prices and types
        """
        points = []
        
        # Initialize
        current_type = "high" if highs[0] > highs[1] else "low"
        last_price = highs[0] if current_type == "high" else lows[0]
        last_idx = 0
        
        for i in range(1, len(highs)):
            if current_type == "high":
                # Looking for a low
                if lows[i] < last_price * (1 - threshold):
                    points.append({
                        "type": "high",
                        "price": last_price,
                        "index": last_idx
                    })
                    current_type = "low"
                    last_price = lows[i]
                    last_idx = i
            else:
                # Looking for a high
                if highs[i] > last_price * (1 + threshold):
                    points.append({
                        "type": "low",
                        "price": last_price,
                        "index": last_idx
                    })
                    current_type = "high"
                    last_price = highs[i]
                    last_idx = i
                    
        return points
        
    def _find_confluences(self):
        """
        Find price areas with confluence across timeframes
        Includes both structure and fractal confluences
        """
        all_levels = []
        
        # Gather all levels
        for tf in self.timeframes:
            # Add structure levels
            for level in self.structure_levels[tf]:
                all_levels.append({
                    "price": level.price,
                    "type": level.type,
                    "timeframe": tf,
                    "strength": level.strength
                })
                
            # Add fractal levels
            for pattern in self.fractal_patterns[tf]:
                all_levels.append({
                    "price": pattern.price,
                    "type": "fractal_" + pattern.type,
                    "timeframe": tf,
                    "strength": pattern.strength
                })
                
        # Find confluences
        confluences = []
        processed = set()
        
        for i, level in enumerate(all_levels):
            if i in processed:
                continue
                
            # Find nearby levels
            cluster = [level]
            price = level["price"]
            
            for j, other in enumerate(all_levels):
                if j != i and j not in processed:
                    if abs(other["price"] - price) / price < 0.0005:  # 0.05% tolerance
                        cluster.append(other)
                        processed.add(j)
                        
            if len(cluster) > 1:  # Real confluence needs multiple levels
                confluences.append({
                    "price": np.mean([l["price"] for l in cluster]),
                    "strength": np.mean([l["strength"] for l in cluster]),
                    "timeframes": list(set(l["timeframe"] for l in cluster)),
                    "types": list(set(l["type"] for l in cluster))
                })
                
        self.confluences = {
            "levels": confluences,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _generate_analysis(self) -> Dict:
        """Generate comprehensive multi-timeframe analysis"""
        if not self.confluences:
            return self._empty_analysis()
            
        # Calculate overall market bias
        bias = self._calculate_bias()
        
        return {
            "bias": bias["direction"],
            "confidence": bias["confidence"],
            "fractals": {
                tf: [
                    {
                        "type": p.type,
                        "price": p.price,
                        "strength": p.strength,
                        "confirmation": p.confirmation,
                        "timestamp": p.timestamp.isoformat()
                    }
                    for p in patterns
                ]
                for tf, patterns in self.fractal_patterns.items()
            },
            "structure": {
                tf: [
                    {
                        "type": l.type,
                        "price": l.price,
                        "strength": l.strength,
                        "touches": l.touches,
                        "last_touch": l.last_touch.isoformat()
                    }
                    for l in levels
                ]
                for tf, levels in self.structure_levels.items()
            },
            "confluences": self.confluences,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _calculate_bias(self) -> Dict:
        """
        Calculate overall market bias based on:
        - Fractal patterns
        - Structure breaks
        - Confluence zones
        """
        bias_score = 0.0
        total_weight = 0.0
        
        # Weight higher timeframes more
        tf_weights = {
            "M5": 0.5,
            "M15": 1.0,
            "H1": 2.0,
            "H4": 3.0,
            "D1": 4.0
        }
        
        # Analyze fractals
        for tf, patterns in self.fractal_patterns.items():
            weight = tf_weights.get(tf, 1.0)
            recent_patterns = [
                p for p in patterns
                if (datetime.utcnow() - p.timestamp).hours < 24
            ]
            
            for pattern in recent_patterns:
                if pattern.type == "top":
                    bias_score -= pattern.strength * weight
                else:
                    bias_score += pattern.strength * weight
                total_weight += weight
                
        # Analyze structure breaks
        for tf, levels in self.structure_levels.items():
            weight = tf_weights.get(tf, 1.0)
            for level in levels:
                if level.type == "support" and level.strength > 0.7:
                    bias_score += level.strength * weight
                elif level.type == "resistance" and level.strength > 0.7:
                    bias_score -= level.strength * weight
                total_weight += weight
                
        # Normalize bias score
        if total_weight > 0:
            bias_score /= total_weight
            
        # Convert to direction and confidence
        direction = "neutral"
        if bias_score > 0.2:
            direction = "bullish"
        elif bias_score < -0.2:
            direction = "bearish"
            
        confidence = min(abs(bias_score), 1.0)
        
        return {
            "direction": direction,
            "confidence": confidence,
            "score": bias_score
        }
        
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            "bias": "neutral",
            "confidence": 0.0,
            "fractals": {},
            "structure": {},
            "confluences": {
                "levels": [],
                "timestamp": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _clear_old_patterns(self):
        """Remove old patterns and clean up memory"""
        now = datetime.utcnow()
        
        for tf in self.timeframes:
            self.fractal_patterns[tf] = [
                p for p in self.fractal_patterns[tf]
                if (now - p.timestamp).days < 7
            ]
            
    def get_trading_recommendations(self) -> Dict:
        """
        Get actionable trading recommendations based on
        multi-timeframe analysis
        """
        analysis = self._generate_analysis()
        
        recommendations = {
            "entry_zones": [],
            "stop_zones": [],
            "target_zones": [],
            "bias": analysis["bias"],
            "confidence": analysis["confidence"],
            "timeframe_alignment": {},
            "notes": []
        }
        
        # Find entry zones from confluences
        if analysis["confluences"]["levels"]:
            for conf in analysis["confluences"]["levels"]:
                if conf["strength"] > 0.7:
                    recommendations["entry_zones"].append({
                        "price": conf["price"],
                        "strength": conf["strength"],
                        "timeframes": conf["timeframes"]
                    })
                    
        # Analyze timeframe alignment
        for tf in self.timeframes:
            if tf in analysis["fractals"]:
                fractals = analysis["fractals"][tf]
                recent_fractals = [
                    f for f in fractals
                    if (datetime.utcnow() - datetime.fromisoformat(f["timestamp"])).hours < 24
                ]
                
                if recent_fractals:
                    tops = sum(1 for f in recent_fractals if f["type"] == "top")
                    bottoms = len(recent_fractals) - tops
                    
                    if tops > bottoms:
                        recommendations["timeframe_alignment"][tf] = "bearish"
                    elif bottoms > tops:
                        recommendations["timeframe_alignment"][tf] = "bullish"
                    else:
                        recommendations["timeframe_alignment"][tf] = "neutral"
                        
        # Add trading notes
        if analysis["confidence"] > 0.7:
            recommendations["notes"].append(
                f"Strong {analysis['bias']} bias across timeframes"
            )
            
        if len(recommendations["entry_zones"]) > 2:
            recommendations["notes"].append(
                "Multiple strong confluence zones - prefer highest strength"
            )
            
        return recommendations
