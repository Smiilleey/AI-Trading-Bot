# core/situational_analysis.py

from datetime import datetime, timedelta
import numpy as np

class SituationalAnalyzer:
    """
    Advanced institutional situational logic layer:
    - Detects special weekly patterns (Friday/Monday, Wednesday/Thursday, etc.)
    - Analyzes market microstructure and session transitions
    - Outputs symbolic context tags and dashboard notes
    - Continuously learns from market behavior patterns
    """
    def __init__(self):
        self.pattern_memory = {}
        self.session_transitions = {}
        
    def analyze(self, candles):
        results = {
            "day_bias": None,
            "notes": [],
            "situational_tags": [],
            "session_context": {},
            "volatility_regime": "normal",
            "momentum_shift": False
        }

        if len(candles) < 5:
            results["notes"].append("Not enough historical data (need 5+ daily candles)")
            return results

        # Enhanced pattern detection
        self._analyze_weekly_patterns(candles, results)
        self._analyze_volatility_regime(candles, results)
        self._analyze_session_transitions(candles, results)
        self._detect_momentum_shifts(candles, results)

        return results
    
    def _analyze_weekly_patterns(self, candles, results):
        """Enhanced weekly pattern analysis with learning capabilities"""
        if len(candles) < 5:
            return
            
        # Assume candles are daily and ordered oldest to newest
        monday = candles[-5]
        wednesday = candles[-3]
        thursday = candles[-2]
        friday = candles[-1]

        # Pattern 1: Friday's high < Thursday's high → Monday likely to revisit Friday's low
        if friday["high"] < thursday["high"]:
            results["day_bias"] = "expect_monday_to_revisit_friday_low"
            results["notes"].append("Friday's high < Thursday's high: Monday may sweep Friday's low")
            results["situational_tags"].append("friday_thursday_reversal")
            results["session_context"]["monday_expectation"] = "sweep_friday_low"

        # Pattern 2: Wednesday's high < Monday's high → Thursday likely to revisit Wednesday's low
        if wednesday["high"] < monday["high"]:
            results["day_bias"] = "expect_thursday_to_revisit_wed_low"
            results["notes"].append("Wednesday's high < Monday's high: Thursday may sweep Wednesday's low")
            results["situational_tags"].append("wednesday_monday_pullback")
            results["session_context"]["thursday_expectation"] = "sweep_wednesday_low"

        # Pattern 3: Monday gap analysis
        if len(candles) >= 6:
            prev_friday = candles[-6]
            monday_gap = monday["open"] - prev_friday["close"]
            if abs(monday_gap) > (prev_friday["high"] - prev_friday["low"]) * 0.3:
                results["situational_tags"].append("significant_monday_gap")
                results["session_context"]["gap_direction"] = "up" if monday_gap > 0 else "down"

    def _analyze_volatility_regime(self, candles, results):
        """Detect volatility regime changes"""
        if len(candles) < 10:
            return
            
        # Calculate rolling volatility
        highs = [c["high"] for c in candles[-10:]]
        lows = [c["low"] for c in candles[-10:]]
        ranges = [h - l for h, l in zip(highs, lows)]
        avg_range = np.mean(ranges)
        current_range = ranges[-1]
        
        if current_range > avg_range * 1.5:
            results["volatility_regime"] = "high"
            results["situational_tags"].append("high_volatility_regime")
        elif current_range < avg_range * 0.7:
            results["volatility_regime"] = "low"
            results["situational_tags"].append("low_volatility_regime")

    def _analyze_session_transitions(self, candles, results):
        """Analyze session transition patterns"""
        if len(candles) < 3:
            return
            
        # Detect session boundary patterns
        current_time = datetime.fromisoformat(candles[-1]["time"])
        hour = current_time.hour
        
        # London-New York overlap
        if 13 <= hour <= 17:
            results["session_context"]["active_sessions"] = ["London", "New York"]
            results["situational_tags"].append("london_ny_overlap")
        elif 8 <= hour <= 12:
            results["session_context"]["active_sessions"] = ["London"]
            results["situational_tags"].append("london_session")
        elif 0 <= hour <= 4:
            results["session_context"]["active_sessions"] = ["Asia"]
            results["situational_tags"].append("asia_session")

    def _detect_momentum_shifts(self, candles, results):
        """Detect momentum shift patterns"""
        if len(candles) < 3:
            return
            
        # Simple momentum detection
        closes = [c["close"] for c in candles[-3:]]
        momentum = (closes[-1] - closes[0]) / closes[0]
        
        if abs(momentum) > 0.002:  # 0.2% move
            results["momentum_shift"] = True
            direction = "bullish" if momentum > 0 else "bearish"
            results["situational_tags"].append(f"momentum_shift_{direction}")
            results["session_context"]["momentum_direction"] = direction

    def learn_pattern(self, pattern_type, success_rate, context):
        """Learn from pattern outcomes to improve future analysis"""
        if pattern_type not in self.pattern_memory:
            self.pattern_memory[pattern_type] = {
                "success_count": 0,
                "total_count": 0,
                "contexts": []
            }
        
        self.pattern_memory[pattern_type]["total_count"] += 1
        if success_rate > 0.5:  # Successful pattern
            self.pattern_memory[pattern_type]["success_count"] += 1
        
        self.pattern_memory[pattern_type]["contexts"].append(context)
        
        # Keep only recent contexts to avoid memory bloat
        if len(self.pattern_memory[pattern_type]["contexts"]) > 100:
            self.pattern_memory[pattern_type]["contexts"] = self.pattern_memory[pattern_type]["contexts"][-50:]
