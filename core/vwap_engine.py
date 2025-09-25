# core/vwap_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class VWAPEngine:
    """
    Advanced VWAP Engine for Anchored and Session VWAP Analysis
    
    Features:
    - Anchored VWAP calculation from key levels (session opens, swing points, news events)
    - Session VWAP calculation for different trading sessions
    - VWAP confluence detection across multiple timeframes
    - VWAP-based entry gates and exit signals
    - Integration with existing signal engine architecture
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # VWAP calculation parameters
        self.vwap_periods = {
            'session': 24,      # 24 hours for session VWAP
            'anchored': 100,    # 100 periods for anchored VWAP
            'short': 20,        # 20 periods for short-term VWAP
            'long': 200         # 200 periods for long-term VWAP
        }
        
        # Session definitions (UTC hours)
        self.sessions = {
            'london': {'start': 8, 'end': 12, 'weight': 0.8},
            'newyork': {'start': 13, 'end': 17, 'weight': 0.9},
            'tokyo': {'start': 0, 'end': 8, 'weight': 0.4},
            'sydney': {'start': 22, 'end': 6, 'weight': 0.3},
            'overlap_london_ny': {'start': 13, 'end': 12, 'weight': 0.95}
        }
        
        # VWAP storage
        self.anchored_vwaps = defaultdict(dict)  # symbol -> {anchor_point: vwap_data}
        self.session_vwaps = defaultdict(dict)   # symbol -> {session: vwap_data}
        self.vwap_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Confluence tracking
        self.confluence_zones = defaultdict(list)
        self.last_confluence_update = defaultdict(float)
        
        # Performance tracking
        self.vwap_signals = defaultdict(list)
        self.vwap_performance = defaultdict(lambda: {
            'total_signals': 0,
            'successful_signals': 0,
            'success_rate': 0.0,
            'avg_pnl': 0.0
        })
        
    def calculate_anchored_vwap(self, 
                               symbol: str, 
                               candles: List[Dict], 
                               anchor_point: str = "session_open",
                               anchor_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate anchored VWAP from specified anchor point
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            anchor_point: Type of anchor ("session_open", "swing_high", "swing_low", "news_event")
            anchor_time: Specific time to anchor from (if None, uses latest session open)
            
        Returns:
            Dictionary with VWAP data and analysis
        """
        try:
            if not candles or len(candles) < 20:
                return {"valid": False, "error": "Insufficient data"}
            
            # Find anchor point
            anchor_idx = self._find_anchor_point(candles, anchor_point, anchor_time)
            if anchor_idx is None:
                return {"valid": False, "error": "Anchor point not found"}
            
            # Calculate VWAP from anchor point
            vwap_data = self._calculate_vwap_from_point(candles, anchor_idx)
            
            # Store anchored VWAP
            self.anchored_vwaps[symbol][anchor_point] = {
                "vwap": vwap_data["vwap"],
                "upper_band": vwap_data["upper_band"],
                "lower_band": vwap_data["lower_band"],
                "anchor_time": candles[anchor_idx]["time"],
                "anchor_price": candles[anchor_idx]["close"],
                "periods": len(candles) - anchor_idx,
                "timestamp": datetime.now()
            }
            
            return {
                "valid": True,
                "vwap": vwap_data["vwap"],
                "upper_band": vwap_data["upper_band"],
                "lower_band": vwap_data["lower_band"],
                "anchor_point": anchor_point,
                "anchor_time": candles[anchor_idx]["time"],
                "periods": len(candles) - anchor_idx,
                "deviation": vwap_data["deviation"],
                "trend": vwap_data["trend"]
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def calculate_session_vwap(self, 
                              symbol: str, 
                              candles: List[Dict], 
                              session: str = "current") -> Dict[str, Any]:
        """
        Calculate session VWAP for specified session
        
        Args:
            symbol: Trading symbol
            candles: List of candle data
            session: Session name ("london", "newyork", "tokyo", "sydney", "current")
            
        Returns:
            Dictionary with session VWAP data
        """
        try:
            if not candles or len(candles) < 10:
                return {"valid": False, "error": "Insufficient data"}
            
            # Determine session
            if session == "current":
                session = self._get_current_session()
            
            if session not in self.sessions:
                return {"valid": False, "error": f"Unknown session: {session}"}
            
            # Filter candles for session
            session_candles = self._filter_candles_for_session(candles, session)
            if not session_candles:
                return {"valid": False, "error": "No candles in session"}
            
            # Calculate session VWAP
            vwap_data = self._calculate_vwap_from_candles(session_candles)
            
            # Store session VWAP
            self.session_vwaps[symbol][session] = {
                "vwap": vwap_data["vwap"],
                "upper_band": vwap_data["upper_band"],
                "lower_band": vwap_data["lower_band"],
                "session": session,
                "candle_count": len(session_candles),
                "timestamp": datetime.now()
            }
            
            return {
                "valid": True,
                "vwap": vwap_data["vwap"],
                "upper_band": vwap_data["upper_band"],
                "lower_band": vwap_data["lower_band"],
                "session": session,
                "candle_count": len(session_candles),
                "deviation": vwap_data["deviation"],
                "trend": vwap_data["trend"]
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def detect_vwap_confluence(self, 
                              symbol: str, 
                              current_price: float,
                              tolerance: float = 0.001) -> Dict[str, Any]:
        """
        Detect VWAP confluence zones where multiple VWAPs align
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            tolerance: Price tolerance for confluence (0.001 = 0.1%)
            
        Returns:
            Dictionary with confluence analysis
        """
        try:
            confluence_zones = []
            confluence_strength = 0.0
            
            # Check anchored VWAPs
            for anchor_point, vwap_data in self.anchored_vwaps[symbol].items():
                vwap_price = vwap_data["vwap"]
                if abs(vwap_price - current_price) / current_price <= tolerance:
                    confluence_zones.append({
                        "type": "anchored",
                        "anchor_point": anchor_point,
                        "vwap": vwap_price,
                        "deviation": abs(vwap_price - current_price) / current_price,
                        "weight": 0.8  # Higher weight for anchored VWAPs
                    })
                    confluence_strength += 0.8
            
            # Check session VWAPs
            for session, vwap_data in self.session_vwaps[symbol].items():
                vwap_price = vwap_data["vwap"]
                if abs(vwap_price - current_price) / current_price <= tolerance:
                    confluence_zones.append({
                        "type": "session",
                        "session": session,
                        "vwap": vwap_price,
                        "deviation": abs(vwap_price - current_price) / current_price,
                        "weight": self.sessions[session]["weight"]
                    })
                    confluence_strength += self.sessions[session]["weight"]
            
            # Calculate confluence metrics
            total_confluence = len(confluence_zones)
            avg_deviation = np.mean([zone["deviation"] for zone in confluence_zones]) if confluence_zones else 1.0
            
            # Determine confluence quality
            if total_confluence >= 3:
                quality = "high"
            elif total_confluence >= 2:
                quality = "medium"
            elif total_confluence >= 1:
                quality = "low"
            else:
                quality = "none"
            
            # Store confluence data
            self.confluence_zones[symbol] = confluence_zones
            self.last_confluence_update[symbol] = datetime.now().timestamp()
            
            return {
                "valid": True,
                "confluence_zones": confluence_zones,
                "total_confluence": total_confluence,
                "confluence_strength": confluence_strength,
                "quality": quality,
                "avg_deviation": avg_deviation,
                "current_price": current_price,
                "tolerance": tolerance
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def generate_vwap_entry_gate(self, 
                                symbol: str, 
                                current_price: float,
                                direction: str = "buy") -> Dict[str, Any]:
        """
        Generate VWAP-based entry gate for signal validation
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            direction: Trade direction ("buy" or "sell")
            
        Returns:
            Dictionary with entry gate decision
        """
        try:
            # Get confluence analysis
            confluence = self.detect_vwap_confluence(symbol, current_price)
            
            if not confluence["valid"]:
                return {"gate_open": False, "reason": "VWAP analysis failed"}
            
            # Check if we have sufficient VWAP data
            if confluence["total_confluence"] == 0:
                return {"gate_open": False, "reason": "No VWAP confluence"}
            
            # Analyze VWAP positioning
            vwap_analysis = self._analyze_vwap_positioning(symbol, current_price, direction)
            
            # Determine entry gate
            gate_open = False
            confidence = 0.0
            reasons = []
            
            # High confluence = strong gate
            if confluence["quality"] == "high":
                gate_open = True
                confidence = 0.9
                reasons.append(f"High VWAP confluence ({confluence['total_confluence']} zones)")
            
            # Medium confluence with good positioning
            elif confluence["quality"] == "medium" and vwap_analysis["positioning_score"] > 0.7:
                gate_open = True
                confidence = 0.7
                reasons.append(f"Medium VWAP confluence with good positioning")
            
            # Low confluence but excellent positioning
            elif confluence["quality"] == "low" and vwap_analysis["positioning_score"] > 0.9:
                gate_open = True
                confidence = 0.6
                reasons.append(f"Low confluence but excellent VWAP positioning")
            
            # Add positioning reasons
            if vwap_analysis["reasons"]:
                reasons.extend(vwap_analysis["reasons"])
            
            return {
                "gate_open": gate_open,
                "confidence": confidence,
                "reasons": reasons,
                "confluence": confluence,
                "vwap_analysis": vwap_analysis,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {"gate_open": False, "reason": f"VWAP gate error: {str(e)}"}
    
    def _find_anchor_point(self, 
                          candles: List[Dict], 
                          anchor_point: str, 
                          anchor_time: Optional[datetime] = None) -> Optional[int]:
        """Find the index of the anchor point in candles"""
        try:
            if anchor_point == "session_open":
                # Find latest session open (8 AM UTC for London)
                for i in range(len(candles) - 1, -1, -1):
                    candle_time = datetime.fromisoformat(candles[i]["time"])
                    if candle_time.hour == 8 and candle_time.minute == 0:
                        return i
                # Fallback to 24 hours ago
                return max(0, len(candles) - 24)
            
            elif anchor_point == "swing_high":
                # Find highest high in last 50 candles
                highs = [c["high"] for c in candles[-50:]]
                max_high = max(highs)
                for i in range(len(candles) - 50, len(candles)):
                    if candles[i]["high"] == max_high:
                        return i
            
            elif anchor_point == "swing_low":
                # Find lowest low in last 50 candles
                lows = [c["low"] for c in candles[-50:]]
                min_low = min(lows)
                for i in range(len(candles) - 50, len(candles)):
                    if candles[i]["low"] == min_low:
                        return i
            
            elif anchor_point == "news_event" and anchor_time:
                # Find candle closest to news event time
                target_timestamp = anchor_time.timestamp()
                min_diff = float('inf')
                best_idx = None
                for i, candle in enumerate(candles):
                    candle_time = datetime.fromisoformat(candle["time"])
                    diff = abs(candle_time.timestamp() - target_timestamp)
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = i
                return best_idx
            
            return None
            
        except Exception:
            return None
    
    def _calculate_vwap_from_point(self, 
                                  candles: List[Dict], 
                                  start_idx: int) -> Dict[str, Any]:
        """Calculate VWAP from specific starting point"""
        try:
            vwap_candles = candles[start_idx:]
            return self._calculate_vwap_from_candles(vwap_candles)
        except Exception:
            return {"vwap": 0.0, "upper_band": 0.0, "lower_band": 0.0, "deviation": 0.0, "trend": "neutral"}
    
    def _calculate_vwap_from_candles(self, candles: List[Dict]) -> Dict[str, Any]:
        """Calculate VWAP from list of candles"""
        try:
            if not candles:
                return {"vwap": 0.0, "upper_band": 0.0, "lower_band": 0.0, "deviation": 0.0, "trend": "neutral"}
            
            # Calculate VWAP
            total_volume = sum(c.get("tick_volume", 1) for c in candles)
            if total_volume == 0:
                total_volume = len(candles)
            
            vwap = sum(c.get("tick_volume", 1) * (c["high"] + c["low"] + c["close"]) / 3 for c in candles) / total_volume
            
            # Calculate standard deviation
            prices = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles]
            deviation = np.std(prices)
            
            # Calculate bands
            upper_band = vwap + (2 * deviation)
            lower_band = vwap - (2 * deviation)
            
            # Determine trend
            if len(prices) >= 2:
                trend = "bullish" if prices[-1] > prices[0] else "bearish" if prices[-1] < prices[0] else "neutral"
            else:
                trend = "neutral"
            
            return {
                "vwap": vwap,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "deviation": deviation,
                "trend": trend
            }
            
        except Exception:
            return {"vwap": 0.0, "upper_band": 0.0, "lower_band": 0.0, "deviation": 0.0, "trend": "neutral"}
    
    def _filter_candles_for_session(self, 
                                   candles: List[Dict], 
                                   session: str) -> List[Dict]:
        """Filter candles for specific session"""
        try:
            session_info = self.sessions[session]
            start_hour = session_info["start"]
            end_hour = session_info["end"]
            
            session_candles = []
            for candle in candles:
                candle_time = datetime.fromisoformat(candle["time"])
                hour = candle_time.hour
                
                # Handle sessions that cross midnight
                if start_hour > end_hour:
                    if hour >= start_hour or hour <= end_hour:
                        session_candles.append(candle)
                else:
                    if start_hour <= hour <= end_hour:
                        session_candles.append(candle)
            
            return session_candles
            
        except Exception:
            return []
    
    def _get_current_session(self) -> str:
        """Get current trading session"""
        current_hour = datetime.now().hour
        
        # Check overlaps first
        if 13 <= current_hour <= 12:  # London-NY overlap
            return "overlap_london_ny"
        
        # Check individual sessions
        for session, info in self.sessions.items():
            if session == "overlap_london_ny":
                continue
            start = info["start"]
            end = info["end"]
            
            if start > end:  # Crosses midnight
                if current_hour >= start or current_hour <= end:
                    return session
            else:  # Normal session
                if start <= current_hour <= end:
                    return session
        
        return "london"  # Default fallback
    
    def _analyze_vwap_positioning(self, 
                                 symbol: str, 
                                 current_price: float, 
                                 direction: str) -> Dict[str, Any]:
        """Analyze VWAP positioning for entry decision"""
        try:
            positioning_score = 0.0
            reasons = []
            
            # Check anchored VWAPs
            for anchor_point, vwap_data in self.anchored_vwaps[symbol].items():
                vwap = vwap_data["vwap"]
                upper_band = vwap_data["upper_band"]
                lower_band = vwap_data["lower_band"]
                
                if direction == "buy":
                    # For buy: price should be near or below VWAP, not too far below lower band
                    if lower_band <= current_price <= vwap:
                        positioning_score += 0.8
                        reasons.append(f"Price near {anchor_point} VWAP (buy zone)")
                    elif current_price < lower_band:
                        positioning_score += 0.3
                        reasons.append(f"Price below {anchor_point} VWAP lower band (oversold)")
                    elif vwap < current_price <= upper_band:
                        positioning_score += 0.5
                        reasons.append(f"Price above {anchor_point} VWAP (trend continuation)")
                
                else:  # sell
                    # For sell: price should be near or above VWAP, not too far above upper band
                    if vwap <= current_price <= upper_band:
                        positioning_score += 0.8
                        reasons.append(f"Price near {anchor_point} VWAP (sell zone)")
                    elif current_price > upper_band:
                        positioning_score += 0.3
                        reasons.append(f"Price above {anchor_point} VWAP upper band (overbought)")
                    elif lower_band <= current_price < vwap:
                        positioning_score += 0.5
                        reasons.append(f"Price below {anchor_point} VWAP (trend continuation)")
            
            # Check session VWAPs
            for session, vwap_data in self.session_vwaps[symbol].items():
                vwap = vwap_data["vwap"]
                upper_band = vwap_data["upper_band"]
                lower_band = vwap_data["lower_band"]
                
                if direction == "buy":
                    if lower_band <= current_price <= vwap:
                        positioning_score += 0.6
                        reasons.append(f"Price near {session} session VWAP (buy zone)")
                else:
                    if vwap <= current_price <= upper_band:
                        positioning_score += 0.6
                        reasons.append(f"Price near {session} session VWAP (sell zone)")
            
            # Normalize score
            max_possible = len(self.anchored_vwaps[symbol]) + len(self.session_vwaps[symbol])
            if max_possible > 0:
                positioning_score = min(1.0, positioning_score / max_possible)
            
            return {
                "positioning_score": positioning_score,
                "reasons": reasons,
                "current_price": current_price,
                "direction": direction
            }
            
        except Exception as e:
            return {"positioning_score": 0.0, "reasons": [f"Positioning analysis error: {str(e)}"]}
    
    def get_vwap_features(self, 
                         symbol: str, 
                         current_price: float) -> Dict[str, Any]:
        """
        Get VWAP features for signal engine integration
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Dictionary with VWAP features
        """
        try:
            features = {}
            
            # Get confluence analysis
            confluence = self.detect_vwap_confluence(symbol, current_price)
            
            # Basic VWAP features
            features["vwap_confluence_count"] = confluence.get("total_confluence", 0)
            features["vwap_confluence_strength"] = confluence.get("confluence_strength", 0.0)
            features["vwap_confluence_quality"] = confluence.get("quality", "none")
            features["vwap_avg_deviation"] = confluence.get("avg_deviation", 1.0)
            
            # Anchored VWAP features
            if symbol in self.anchored_vwaps:
                for anchor_point, vwap_data in self.anchored_vwaps[symbol].items():
                    vwap = vwap_data["vwap"]
                    features[f"vwap_{anchor_point}"] = vwap
                    features[f"vwap_{anchor_point}_deviation"] = abs(vwap - current_price) / current_price
                    features[f"vwap_{anchor_point}_trend"] = vwap_data.get("trend", "neutral")
            
            # Session VWAP features
            if symbol in self.session_vwaps:
                for session, vwap_data in self.session_vwaps[symbol].items():
                    vwap = vwap_data["vwap"]
                    features[f"vwap_{session}_session"] = vwap
                    features[f"vwap_{session}_deviation"] = abs(vwap - current_price) / current_price
                    features[f"vwap_{session}_trend"] = vwap_data.get("trend", "neutral")
            
            # Entry gate features
            buy_gate = self.generate_vwap_entry_gate(symbol, current_price, "buy")
            sell_gate = self.generate_vwap_entry_gate(symbol, current_price, "sell")
            
            features["vwap_buy_gate_open"] = buy_gate.get("gate_open", False)
            features["vwap_buy_gate_confidence"] = buy_gate.get("confidence", 0.0)
            features["vwap_sell_gate_open"] = sell_gate.get("gate_open", False)
            features["vwap_sell_gate_confidence"] = sell_gate.get("confidence", 0.0)
            
            return features
            
        except Exception as e:
            return {"vwap_error": str(e)}
    
    def update_performance(self, 
                          symbol: str, 
                          signal_data: Dict, 
                          outcome: bool, 
                          pnl: float):
        """Update VWAP performance tracking"""
        try:
            self.vwap_performance[symbol]["total_signals"] += 1
            if outcome:
                self.vwap_performance[symbol]["successful_signals"] += 1
            
            # Update success rate
            perf = self.vwap_performance[symbol]
            perf["success_rate"] = perf["successful_signals"] / perf["total_signals"]
            perf["avg_pnl"] = (perf["avg_pnl"] * (perf["total_signals"] - 1) + pnl) / perf["total_signals"]
            
        except Exception:
            pass
    
    def get_vwap_stats(self, symbol: str = None) -> Dict[str, Any]:
        """Get VWAP engine statistics"""
        try:
            if symbol:
                return {
                    "symbol": symbol,
                    "performance": self.vwap_performance[symbol],
                    "anchored_vwaps": len(self.anchored_vwaps[symbol]),
                    "session_vwaps": len(self.session_vwaps[symbol]),
                    "confluence_zones": len(self.confluence_zones[symbol])
                }
            else:
                return {
                    "total_symbols": len(self.anchored_vwaps),
                    "performance": dict(self.vwap_performance),
                    "last_update": datetime.now().isoformat()
                }
        except Exception as e:
            return {"error": str(e)}