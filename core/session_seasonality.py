# core/session_seasonality.py

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import os

class SessionSeasonalityModel:
    """
    Session and seasonality expectancy model:
    - Time-of-day priors for different sessions
    - Day-of-week seasonality patterns
    - Historical performance by session/time
    - Dynamic threshold adjustment based on seasonality
    """
    
    def __init__(self, data_file: str = None):
        self.data_file = data_file or "data/session_seasonality.json"
        self.session_data = {}
        self.seasonality_data = {}
        self.performance_history = defaultdict(list)
        self.load_historical_data()
        
        # Session definitions (UTC hours)
        self.sessions = {
            "sydney": {"start": 22, "end": 6, "weight": 0.3, "characteristics": ["low_volatility", "range_bound"]},
            "tokyo": {"start": 0, "end": 8, "weight": 0.4, "characteristics": ["moderate_volatility", "trending"]},
            "london": {"start": 8, "end": 12, "weight": 0.8, "characteristics": ["high_volatility", "breakout_prone"]},
            "ny": {"start": 13, "end": 17, "weight": 0.9, "characteristics": ["highest_volatility", "news_driven"]},
            "overlap": {"start": 13, "end": 12, "weight": 0.95, "characteristics": ["maximum_volatility", "institutional_flow"]}
        }
        
        # Day-of-week patterns
        self.dow_patterns = {
            "monday": {"volatility": 0.8, "trend_strength": 0.6, "gap_frequency": 0.7},
            "tuesday": {"volatility": 0.9, "trend_strength": 0.8, "gap_frequency": 0.3},
            "wednesday": {"volatility": 1.0, "trend_strength": 0.9, "gap_frequency": 0.2},
            "thursday": {"volatility": 1.1, "trend_strength": 0.8, "gap_frequency": 0.1},
            "friday": {"volatility": 1.2, "trend_strength": 0.7, "gap_frequency": 0.4},
            "saturday": {"volatility": 0.1, "trend_strength": 0.0, "gap_frequency": 0.0},
            "sunday": {"volatility": 0.3, "trend_strength": 0.2, "gap_frequency": 0.6}
        }
    
    def load_historical_data(self):
        """Load historical session performance data."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.session_data = data.get("session_data", {})
                    self.seasonality_data = data.get("seasonality_data", {})
            else:
                self._initialize_default_data()
        except Exception as e:
            print(f"Error loading seasonality data: {e}")
            self._initialize_default_data()
    
    def _initialize_default_data(self):
        """Initialize with default seasonality patterns."""
        # Default session performance data
        self.session_data = {
            "sydney": {"win_rate": 0.52, "avg_rr": 1.8, "volatility": 0.6},
            "tokyo": {"win_rate": 0.55, "avg_rr": 1.9, "volatility": 0.7},
            "london": {"win_rate": 0.58, "avg_rr": 2.1, "volatility": 0.9},
            "ny": {"win_rate": 0.61, "avg_rr": 2.3, "volatility": 1.1},
            "overlap": {"win_rate": 0.63, "avg_rr": 2.5, "volatility": 1.3}
        }
        
        # Default day-of-week data
        self.seasonality_data = {
            "monday": {"win_rate": 0.48, "avg_rr": 1.6, "volatility": 0.8},
            "tuesday": {"win_rate": 0.52, "avg_rr": 1.8, "volatility": 0.9},
            "wednesday": {"win_rate": 0.55, "avg_rr": 2.0, "volatility": 1.0},
            "thursday": {"win_rate": 0.58, "avg_rr": 2.1, "volatility": 1.1},
            "friday": {"win_rate": 0.50, "avg_rr": 1.9, "volatility": 1.2}
        }
    
    def get_current_session(self, now: datetime = None) -> str:
        """Get current trading session."""
        if now is None:
            now = datetime.utcnow()
        
        current_hour = now.hour
        
        # Check for session overlaps first
        if 13 <= current_hour <= 12:  # London-NY overlap
            return "overlap"
        
        # Check individual sessions
        for session, details in self.sessions.items():
            if session == "overlap":
                continue
            start = details["start"]
            end = details["end"]
            
            if start > end:  # Crosses midnight
                if current_hour >= start or current_hour <= end:
                    return session
            else:  # Normal session
                if start <= current_hour <= end:
                    return session
        
        return "sydney"  # Default fallback
    
    def get_session_characteristics(self, session: str) -> Dict:
        """Get characteristics of a trading session."""
        return self.sessions.get(session, {
            "start": 0, "end": 24, "weight": 0.5,
            "characteristics": ["unknown"]
        })
    
    def get_day_of_week_factor(self, now: datetime = None) -> Dict:
        """Get day-of-week seasonality factor."""
        if now is None:
            now = datetime.utcnow()
        
        dow = now.strftime("%A").lower()
        return self.dow_patterns.get(dow, {
            "volatility": 1.0, "trend_strength": 0.5, "gap_frequency": 0.0
        })
    
    def calculate_session_expectancy(self, symbol: str, now: datetime = None) -> Dict:
        """Calculate session-based expectancy for trading."""
        if now is None:
            now = datetime.utcnow()
        
        current_session = self.get_current_session(now)
        session_char = self.get_session_characteristics(current_session)
        dow_factor = self.get_day_of_week_factor(now)
        
        # Get historical performance for this session
        session_perf = self.session_data.get(current_session, {
            "win_rate": 0.5, "avg_rr": 1.5, "volatility": 1.0
        })
        
        # Calculate expectancy
        expectancy = session_perf["win_rate"] * session_perf["avg_rr"] - (1 - session_perf["win_rate"])
        
        # Apply day-of-week adjustment
        dow_adjustment = self.dow_patterns.get(now.strftime("%A").lower(), {})
        volatility_multiplier = dow_adjustment.get("volatility", 1.0)
        trend_multiplier = dow_adjustment.get("trend_strength", 1.0)
        
        # Calculate threshold adjustment
        base_threshold = 0.6
        session_weight = session_char["weight"]
        threshold_adjustment = base_threshold * (1 - session_weight * 0.2)  # Lower threshold for better sessions
        
        # Calculate position sizing adjustment
        volatility_factor = session_perf["volatility"] * volatility_multiplier
        sizing_adjustment = 1.0 / max(volatility_factor, 0.5)  # Reduce size in high volatility
        
        return {
            "session": current_session,
            "expectancy": expectancy,
            "win_rate": session_perf["win_rate"],
            "avg_rr": session_perf["avg_rr"],
            "volatility_factor": volatility_factor,
            "threshold_adjustment": threshold_adjustment,
            "sizing_adjustment": sizing_adjustment,
            "characteristics": session_char["characteristics"],
            "dow_factors": dow_factor,
            "confidence": min(1.0, session_weight * 1.2)
        }
    
    def get_optimal_entry_times(self, symbol: str) -> List[Dict]:
        """Get optimal entry times based on historical performance."""
        optimal_times = []
        
        for session, perf in self.session_data.items():
            if session == "overlap":
                continue
            
            session_char = self.get_session_characteristics(session)
            expectancy = perf["win_rate"] * perf["avg_rr"] - (1 - perf["win_rate"])
            
            if expectancy > 0.3:  # Only include positive expectancy sessions
                optimal_times.append({
                    "session": session,
                    "start_hour": session_char["start"],
                    "end_hour": session_char["end"],
                    "expectancy": expectancy,
                    "win_rate": perf["win_rate"],
                    "avg_rr": perf["avg_rr"],
                    "weight": session_char["weight"],
                    "recommended": expectancy > 0.5
                })
        
        # Sort by expectancy
        optimal_times.sort(key=lambda x: x["expectancy"], reverse=True)
        return optimal_times
    
    def update_performance(self, symbol: str, session: str, outcome: Dict):
        """Update performance history for learning."""
        key = f"{symbol}_{session}"
        self.performance_history[key].append({
            "timestamp": datetime.utcnow().isoformat(),
            "outcome": outcome
        })
        
        # Keep only last 1000 records
        if len(self.performance_history[key]) > 1000:
            self.performance_history[key] = self.performance_history[key][-1000:]
        
        # Update session data with new performance
        self._update_session_data(symbol, session, outcome)
    
    def _update_session_data(self, symbol: str, session: str, outcome: Dict):
        """Update session performance data with new outcome."""
        if session not in self.session_data:
            self.session_data[session] = {"win_rate": 0.5, "avg_rr": 1.5, "volatility": 1.0}
        
        # Simple exponential moving average update
        alpha = 0.1  # Learning rate
        current = self.session_data[session]
        
        is_win = outcome.get("pnl", 0) > 0
        rr = outcome.get("rr", 1.0)
        
        # Update win rate
        current["win_rate"] = (1 - alpha) * current["win_rate"] + alpha * (1 if is_win else 0)
        
        # Update average RR
        current["avg_rr"] = (1 - alpha) * current["avg_rr"] + alpha * rr
        
        # Update volatility (simplified)
        volatility = abs(outcome.get("pnl", 0)) / 100  # Normalize
        current["volatility"] = (1 - alpha) * current["volatility"] + alpha * volatility
    
    def get_seasonality_signals(self, symbol: str, now: datetime = None) -> Dict:
        """Get seasonality-based trading signals."""
        if now is None:
            now = datetime.utcnow()
        
        expectancy = self.calculate_session_expectancy(symbol, now)
        current_session = expectancy["session"]
        
        signals = {
            "session": current_session,
            "expectancy": expectancy["expectancy"],
            "confidence": expectancy["confidence"],
            "threshold_modifier": expectancy["threshold_adjustment"],
            "sizing_modifier": expectancy["sizing_adjustment"],
            "volatility_regime": "high" if expectancy["volatility_factor"] > 1.2 else "normal",
            "entry_gate": expectancy["expectancy"] > 0.2,
            "recommendation": self._get_session_recommendation(expectancy)
        }
        
        return signals
    
    def _get_session_recommendation(self, expectancy: Dict) -> str:
        """Get trading recommendation based on session expectancy."""
        if expectancy["expectancy"] > 0.5:
            return "strong_buy"
        elif expectancy["expectancy"] > 0.2:
            return "buy"
        elif expectancy["expectancy"] > -0.2:
            return "neutral"
        else:
            return "avoid"
    
    def save_data(self):
        """Save updated seasonality data."""
        try:
            data = {
                "session_data": self.session_data,
                "seasonality_data": self.seasonality_data,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving seasonality data: {e}")

# Global seasonality model instance
seasonality_model = SessionSeasonalityModel()

def get_session_expectancy(symbol: str, now: datetime = None) -> Dict:
    """Get session expectancy for symbol."""
    return seasonality_model.calculate_session_expectancy(symbol, now)

def get_seasonality_signals(symbol: str, now: datetime = None) -> Dict:
    """Get seasonality-based signals."""
    return seasonality_model.get_seasonality_signals(symbol, now)

def update_session_performance(symbol: str, session: str, outcome: Dict):
    """Update session performance for learning."""
    seasonality_model.update_performance(symbol, session, outcome)