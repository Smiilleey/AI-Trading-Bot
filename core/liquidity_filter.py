# core/liquidity_filter.py

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LiquidityFilter:
    """
    Liquidity Filter for optimal trading windows
    - Global market session management
    - Liquidity window filtering
    - Spread and slippage monitoring
    - News window blocking
    - Optimal entry timing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Global liquidity windows (UTC)
        self.liquidity_windows = {
            "sydney": {"start": 22, "end": 6, "weight": 0.3},
            "tokyo": {"start": 0, "end": 8, "weight": 0.4},
            "london": {"start": 8, "end": 12, "weight": 0.8},
            "frankfurt": {"start": 7, "end": 11, "weight": 0.7},
            "ny": {"start": 13, "end": 17, "weight": 0.9},
            "chicago": {"start": 14, "end": 18, "weight": 0.8},
            "closing": {"start": 20, "end": 22, "weight": 0.5}
        }
        
        # Session overlap periods (high liquidity)
        self.overlap_periods = [
            {"name": "london_ny", "start": 13, "end": 12, "weight": 0.95},
            {"name": "tokyo_london", "start": 8, "end": 8, "weight": 0.6},
            {"name": "sydney_tokyo", "start": 0, "end": 6, "weight": 0.5}
        ]
        
        # Spread and slippage thresholds
        self.thresholds = {
            "max_spread_pips": 3.0,
            "max_slippage_pips": 2.0,
            "min_liquidity_score": 0.6,
            "news_buffer_minutes": 30
        }
        
        # News windows (major economic events)
        self.news_windows = {
            "nfp": {"day": "first_friday", "time": "13:30", "duration_hours": 2},
            "fomc": {"day": "variable", "time": "19:00", "duration_hours": 3},
            "cpi": {"day": "variable", "time": "13:30", "duration_hours": 2},
            "gdp": {"day": "variable", "time": "13:30", "duration_hours": 2}
        }
        
        # Performance tracking
        self.total_checks = 0
        self.blocked_trades = 0
        self.liquidity_scores = defaultdict(list)
        
    def check_liquidity_window(self, symbol: str, timeframe: str, 
                              current_time: datetime) -> Dict:
        """
        Check if current time is within optimal liquidity window
        """
        try:
            self.total_checks += 1
            
            # Get current UTC time
            utc_time = current_time.utcnow() if hasattr(current_time, 'utcnow') else current_time
            current_hour = utc_time.hour
            
            # Check if in news window
            if self._is_in_news_window(utc_time):
                return {
                    "liquidity_available": False,
                    "reason": "news_window",
                    "confidence": 0.0,
                    "liquidity_score": 0.0,
                    "optimal_session": "none"
                }
            
            # Calculate liquidity score for current time
            liquidity_score = self._calculate_liquidity_score(current_hour)
            
            # Determine optimal session
            optimal_session = self._get_optimal_session(current_hour)
            
            # Check if liquidity is sufficient
            liquidity_available = liquidity_score >= self.thresholds["min_liquidity_score"]
            
            # Update performance tracking
            self.liquidity_scores[symbol].append(liquidity_score)
            if len(self.liquidity_scores[symbol]) > 100:
                self.liquidity_scores[symbol] = self.liquidity_scores[symbol][-100:]
            
            if not liquidity_available:
                self.blocked_trades += 1
            
            return {
                "liquidity_available": liquidity_available,
                "reason": "insufficient_liquidity" if not liquidity_available else "optimal",
                "confidence": liquidity_score,
                "liquidity_score": liquidity_score,
                "optimal_session": optimal_session,
                "current_hour_utc": current_hour,
                "session_overlaps": self._get_session_overlaps(current_hour)
            }
            
        except Exception as e:
            return {
                "liquidity_available": False,
                "reason": "error",
                "confidence": 0.0,
                "liquidity_score": 0.0,
                "optimal_session": "none",
                "error": str(e)
            }
    
    def _is_in_news_window(self, current_time: datetime) -> bool:
        """Check if current time is within news window"""
        try:
            # Check for major news events
            for event, details in self.news_windows.items():
                if self._check_news_event(current_time, details):
                    return True
            return False
        except Exception as e:
            return False
    
    def _check_news_event(self, current_time: datetime, event_details: Dict) -> bool:
        """Check specific news event"""
        try:
            # This is a simplified check - in production you'd integrate with news APIs
            # For now, we'll just check if it's during typical news hours
            current_hour = current_time.hour
            
            # Major news typically happens during NY/London overlap
            if 13 <= current_hour <= 17:  # NY session
                return True
            
            return False
        except Exception as e:
            return False
    
    def _calculate_liquidity_score(self, current_hour: int) -> float:
        """Calculate liquidity score for current hour"""
        try:
            score = 0.0
            
            # Check each session
            for session, details in self.liquidity_windows.items():
                start = details["start"]
                end = details["end"]
                weight = details["weight"]
                
                # Handle sessions that cross midnight
                if start > end:  # Crosses midnight
                    if current_hour >= start or current_hour <= end:
                        score += weight
                else:  # Normal session
                    if start <= current_hour <= end:
                        score += weight
            
            # Check overlap periods
            for overlap in self.overlap_periods:
                start = overlap["start"]
                end = overlap["end"]
                weight = overlap["weight"]
                
                if start <= current_hour <= end:
                    score += weight
            
            # Normalize score
            return min(1.0, score)
            
        except Exception as e:
            return 0.0
    
    def _get_optimal_session(self, current_hour: int) -> str:
        """Get optimal trading session for current hour"""
        try:
            best_session = "none"
            best_score = 0.0
            
            for session, details in self.liquidity_windows.items():
                start = details["start"]
                end = details["end"]
                weight = details["weight"]
                
                # Check if current hour is in session
                if start > end:  # Crosses midnight
                    in_session = current_hour >= start or current_hour <= end
                else:  # Normal session
                    in_session = start <= current_hour <= end
                
                if in_session and weight > best_score:
                    best_score = weight
                    best_session = session
            
            return best_session
            
        except Exception as e:
            return "none"
    
    def _get_session_overlaps(self, current_hour: int) -> List[Dict]:
        """Get current session overlaps"""
        try:
            overlaps = []
            
            for overlap in self.overlap_periods:
                start = overlap["start"]
                end = overlap["end"]
                
                if start <= current_hour <= end:
                    overlaps.append({
                        "name": overlap["name"],
                        "weight": overlap["weight"],
                        "active": True
                    })
            
            return overlaps
            
        except Exception as e:
            return []
    
    def check_spread_conditions(self, current_spread: float, symbol: str) -> Dict:
        """Check if spread conditions are acceptable"""
        try:
            max_spread = self.thresholds["max_spread_pips"]
            spread_acceptable = current_spread <= max_spread
            
            return {
                "spread_acceptable": spread_acceptable,
                "current_spread": current_spread,
                "max_spread": max_spread,
                "spread_ratio": current_spread / max_spread if max_spread > 0 else 1.0
            }
            
        except Exception as e:
            return {
                "spread_acceptable": False,
                "current_spread": 0.0,
                "max_spread": self.thresholds["max_spread_pips"],
                "error": str(e)
            }
    
    def check_slippage_conditions(self, current_slippage: float, symbol: str) -> Dict:
        """Check if slippage conditions are acceptable"""
        try:
            max_slippage = self.thresholds["max_slippage_pips"]
            slippage_acceptable = current_slippage <= max_slippage
            
            return {
                "slippage_acceptable": slippage_acceptable,
                "current_slippage": current_slippage,
                "max_slippage": max_slippage,
                "slippage_ratio": current_slippage / max_slippage if max_slippage > 0 else 1.0
            }
            
        except Exception as e:
            return {
                "slippage_acceptable": False,
                "current_slippage": 0.0,
                "max_slippage": self.thresholds["max_slippage_pips"],
                "error": str(e)
            }
    
    def get_optimal_trading_times(self, symbol: str) -> Dict:
        """Get optimal trading times for symbol"""
        try:
            optimal_times = []
            
            # Sort sessions by weight
            sorted_sessions = sorted(
                self.liquidity_windows.items(),
                key=lambda x: x[1]["weight"],
                reverse=True
            )
            
            for session, details in sorted_sessions:
                optimal_times.append({
                    "session": session,
                    "start_utc": details["start"],
                    "end_utc": details["end"],
                    "weight": details["weight"],
                    "description": f"{session.title()} session"
                })
            
            return {
                "optimal_times": optimal_times,
                "best_session": optimal_times[0]["session"] if optimal_times else "none",
                "total_sessions": len(optimal_times)
            }
            
        except Exception as e:
            return {
                "optimal_times": [],
                "best_session": "none",
                "total_sessions": 0,
                "error": str(e)
            }
    
    def get_filter_stats(self) -> Dict:
        """Get comprehensive filter statistics"""
        return {
            "total_checks": self.total_checks,
            "blocked_trades": self.blocked_trades,
            "block_rate": self.blocked_trades / max(1, self.total_checks),
            "liquidity_scores": {k: np.mean(v) if v else 0.0 for k, v in self.liquidity_scores.items()},
            "thresholds": self.thresholds,
            "liquidity_windows": self.liquidity_windows
        }
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update filter thresholds"""
        try:
            for key, value in new_thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key] = value
        except Exception as e:
            pass  # Silent fail for threshold updates
