# core/premium_discount_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class PremiumDiscountEngine:
    """
    PREMIUM/DISCOUNT ENGINE - Institutional-Grade Market Structure Analysis
    
    Explicit premium/discount logic tied to active dealing range per timeframe and session.
    This engine identifies where price "should" be relative to institutional levels.
    
    Key Features:
    - Session-specific dealing ranges (Asia/London/NY)
    - Premium/discount identification relative to active range
    - Multi-timeframe range analysis
    - Session bias and volatility patterns
    - Institutional range breakout/breakdown detection
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Session definitions (UTC times)
        self.sessions = {
            'asian': {
                'start': 0, 'end': 8,
                'am_killzone': (0, 2),
                'pm_killzone': (6, 8),
                'volatility_profile': 'low',
                'characteristics': ['range_bound', 'liquidity_building']
            },
            'london': {
                'start': 8, 'end': 16,
                'am_killzone': (8, 10),
                'pm_killzone': (14, 16),
                'volatility_profile': 'high',
                'characteristics': ['breakouts', 'trend_initiation']
            },
            'newyork': {
                'start': 13, 'end': 21,
                'am_killzone': (13, 15),
                'pm_killzone': (17, 19),
                'volatility_profile': 'very_high',
                'characteristics': ['continuation', 'reversals']
            },
            'london_ny_overlap': {
                'start': 13, 'end': 16,
                'volatility_profile': 'extreme',
                'characteristics': ['major_moves', 'institutional_activity']
            }
        }
        
        # Premium/Discount zones
        self.pd_zones = {
            'premium': {'threshold': 0.7, 'action': 'sell_bias'},
            'equilibrium': {'threshold_low': 0.3, 'threshold_high': 0.7, 'action': 'neutral'},
            'discount': {'threshold': 0.3, 'action': 'buy_bias'}
        }
        
        # Dealing ranges by symbol and timeframe
        self.dealing_ranges = defaultdict(lambda: defaultdict(dict))
        self.session_ranges = defaultdict(lambda: defaultdict(dict))
        self.range_history = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.total_analyses = 0
        self.successful_predictions = 0
        self.session_performance = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        # Timeframe hierarchy for multi-TF analysis
        self.timeframe_hierarchy = ['MN1', 'W1', 'D1', 'H4', 'H1', 'M15', 'M5']
        self.timeframe_weights = {
            'MN1': 0.25, 'W1': 0.20, 'D1': 0.20,
            'H4': 0.15, 'H1': 0.10, 'M15': 0.05, 'M5': 0.05
        }
    
    def analyze_premium_discount(self, 
                                symbol: str, 
                                timeframe: str, 
                                candles: List[Dict],
                                current_time: datetime = None) -> Dict[str, Any]:
        """
        Comprehensive premium/discount analysis for given symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe  
            candles: Historical candle data
            current_time: Current analysis time
            
        Returns:
            Complete premium/discount analysis
        """
        try:
            if not candles or len(candles) < 50:
                return self._create_pd_response(False, "Insufficient candle data")
            
            current_time = current_time or datetime.now()
            current_session = self._identify_current_session(current_time)
            
            # 1. Identify dealing ranges for current session
            dealing_range = self._identify_dealing_range(symbol, timeframe, candles, current_session)
            
            # 2. Calculate premium/discount levels
            pd_analysis = self._calculate_premium_discount_levels(dealing_range, candles[-1])
            
            # 3. Multi-timeframe range analysis
            mtf_ranges = self._analyze_multi_timeframe_ranges(symbol, candles)
            
            # 4. Session bias analysis
            session_bias = self._analyze_session_bias(symbol, current_session, candles)
            
            # 5. Range breakout/breakdown probability
            breakout_analysis = self._analyze_breakout_probability(dealing_range, candles)
            
            # 6. Institutional range levels
            institutional_levels = self._identify_institutional_levels(dealing_range, candles)
            
            # 7. Create comprehensive response
            response = self._create_pd_response(
                True,
                dealing_range=dealing_range,
                pd_analysis=pd_analysis,
                mtf_ranges=mtf_ranges,
                session_bias=session_bias,
                breakout_analysis=breakout_analysis,
                institutional_levels=institutional_levels,
                current_session=current_session,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Update performance tracking
            self._update_performance_tracking(response)
            
            return response
            
        except Exception as e:
            return self._create_pd_response(False, f"Premium/discount analysis failed: {str(e)}")
    
    def _identify_current_session(self, current_time: datetime) -> str:
        """Identify current trading session based on UTC time."""
        hour = current_time.hour
        
        # Check for overlap periods first
        if 13 <= hour <= 16:
            return 'london_ny_overlap'
        elif 8 <= hour <= 16:
            return 'london'
        elif 13 <= hour <= 21:
            return 'newyork'
        elif 0 <= hour <= 8:
            return 'asian'
        else:
            return 'off_hours'
    
    def _identify_dealing_range(self, 
                               symbol: str, 
                               timeframe: str, 
                               candles: List[Dict], 
                               session: str) -> Dict[str, Any]:
        """Identify the active dealing range for current session."""
        try:
            # Get session-specific candles (last 24 hours for daily range)
            session_candles = self._get_session_candles(candles, session)
            
            if not session_candles:
                # Fallback to recent range
                session_candles = candles[-20:]
            
            # Calculate range boundaries
            highs = [float(c['high']) for c in session_candles]
            lows = [float(c['low']) for c in session_candles]
            
            range_high = max(highs)
            range_low = min(lows)
            range_mid = (range_high + range_low) / 2
            range_size = range_high - range_low
            
            # Identify key levels within range
            key_levels = self._identify_key_levels_in_range(session_candles)
            
            # Calculate range strength and validity
            range_strength = self._calculate_range_strength(session_candles, range_high, range_low)
            
            # Determine range type
            range_type = self._classify_range_type(session_candles, range_size)
            
            dealing_range = {
                'high': range_high,
                'low': range_low,
                'mid': range_mid,
                'size': range_size,
                'strength': range_strength,
                'type': range_type,
                'key_levels': key_levels,
                'session': session,
                'candle_count': len(session_candles),
                'formation_time': session_candles[0].get('time', 'unknown'),
                'last_update': session_candles[-1].get('time', 'unknown')
            }
            
            # Store in memory
            self.dealing_ranges[symbol][timeframe][session] = dealing_range
            
            return dealing_range
            
        except Exception as e:
            return {
                'high': 0, 'low': 0, 'mid': 0, 'size': 0,
                'strength': 0, 'type': 'unknown', 'key_levels': [],
                'error': str(e)
            }
    
    def _get_session_candles(self, candles: List[Dict], session: str) -> List[Dict]:
        """Get candles for specific session."""
        if session not in self.sessions:
            return candles[-20:]  # Fallback
        
        session_info = self.sessions[session]
        session_candles = []
        
        # Simple approach: get last 24 hours of data
        # In production, you'd filter by actual session times
        return candles[-24:] if len(candles) >= 24 else candles
    
    def _identify_key_levels_in_range(self, candles: List[Dict]) -> List[Dict]:
        """Identify key support/resistance levels within the range."""
        key_levels = []
        
        if len(candles) < 10:
            return key_levels
        
        # Find swing highs and lows
        for i in range(2, len(candles) - 2):
            current = candles[i]
            prev2, prev1 = candles[i-2], candles[i-1]
            next1, next2 = candles[i+1], candles[i+2]
            
            # Swing high
            if (current['high'] > prev2['high'] and current['high'] > prev1['high'] and
                current['high'] > next1['high'] and current['high'] > next2['high']):
                key_levels.append({
                    'level': current['high'],
                    'type': 'resistance',
                    'strength': 0.8,
                    'touches': 1,
                    'time': current.get('time', 'unknown')
                })
            
            # Swing low
            if (current['low'] < prev2['low'] and current['low'] < prev1['low'] and
                current['low'] < next1['low'] and current['low'] < next2['low']):
                key_levels.append({
                    'level': current['low'],
                    'type': 'support',
                    'strength': 0.8,
                    'touches': 1,
                    'time': current.get('time', 'unknown')
                })
        
        return key_levels
    
    def _calculate_range_strength(self, candles: List[Dict], range_high: float, range_low: float) -> float:
        """Calculate the strength/validity of the identified range."""
        if not candles:
            return 0.0
        
        # Count how many times price respected the range
        respects = 0
        breaks = 0
        
        for candle in candles:
            high, low = float(candle['high']), float(candle['low'])
            
            # Check if candle respects range
            if range_low <= low and high <= range_high:
                respects += 1
            elif high > range_high or low < range_low:
                breaks += 1
        
        total_tests = respects + breaks
        strength = respects / total_tests if total_tests > 0 else 0.5
        
        return min(1.0, strength)
    
    def _classify_range_type(self, candles: List[Dict], range_size: float) -> str:
        """Classify the type of range (tight, normal, wide)."""
        if not candles:
            return 'unknown'
        
        # Calculate average candle size
        avg_candle_size = np.mean([
            float(c['high']) - float(c['low']) for c in candles
        ])
        
        if range_size < avg_candle_size * 5:
            return 'tight'
        elif range_size < avg_candle_size * 15:
            return 'normal'
        else:
            return 'wide'
    
    def _calculate_premium_discount_levels(self, dealing_range: Dict, current_candle: Dict) -> Dict[str, Any]:
        """Calculate premium/discount levels relative to dealing range."""
        try:
            range_high = dealing_range['high']
            range_low = dealing_range['low']
            range_size = range_high - range_low
            current_price = float(current_candle['close'])
            
            if range_size == 0:
                return {'status': 'equilibrium', 'percentage': 0.5, 'bias': 'neutral'}
            
            # Calculate position within range (0 = bottom, 1 = top)
            range_position = (current_price - range_low) / range_size
            range_position = max(0, min(1, range_position))  # Clamp to 0-1
            
            # Determine premium/discount status
            if range_position >= self.pd_zones['premium']['threshold']:
                status = 'premium'
                bias = 'sell_bias'
                strength = (range_position - 0.7) / 0.3  # 0-1 strength in premium zone
            elif range_position <= self.pd_zones['discount']['threshold']:
                status = 'discount'
                bias = 'buy_bias'
                strength = (0.3 - range_position) / 0.3  # 0-1 strength in discount zone
            else:
                status = 'equilibrium'
                bias = 'neutral'
                strength = 1 - abs(range_position - 0.5) * 2  # Strongest at exact middle
            
            # Calculate specific levels
            premium_level = range_low + (range_size * 0.8)  # 80% level
            discount_level = range_low + (range_size * 0.2)  # 20% level
            equilibrium_level = range_low + (range_size * 0.5)  # 50% level
            
            return {
                'status': status,
                'bias': bias,
                'percentage': range_position,
                'strength': strength,
                'current_price': current_price,
                'levels': {
                    'premium': premium_level,
                    'equilibrium': equilibrium_level,
                    'discount': discount_level
                },
                'distance_to_premium': abs(current_price - premium_level),
                'distance_to_discount': abs(current_price - discount_level),
                'distance_to_equilibrium': abs(current_price - equilibrium_level)
            }
            
        except Exception as e:
            return {'status': 'unknown', 'error': str(e)}
    
    def _analyze_multi_timeframe_ranges(self, symbol: str, candles: List[Dict]) -> Dict[str, Any]:
        """Analyze ranges across multiple timeframes."""
        mtf_ranges = {}
        
        # Simulate multi-timeframe analysis
        # In production, you'd have actual multi-TF data
        timeframes = ['H4', 'H1', 'M15']
        
        for tf in timeframes:
            # Get appropriate candle subset for timeframe
            if tf == 'H4':
                tf_candles = candles[-48:]  # ~2 days of H4 data
            elif tf == 'H1':
                tf_candles = candles[-24:]  # ~1 day of H1 data
            else:
                tf_candles = candles[-96:]  # ~1 day of M15 data
            
            if tf_candles:
                tf_range = self._identify_dealing_range(symbol, tf, tf_candles, 'current')
                mtf_ranges[tf] = {
                    'range': tf_range,
                    'weight': self.timeframe_weights.get(tf, 0.1)
                }
        
        # Calculate multi-timeframe confluence
        confluence = self._calculate_mtf_confluence(mtf_ranges)
        
        return {
            'ranges': mtf_ranges,
            'confluence': confluence,
            'alignment': 'bullish' if confluence > 0.1 else ('bearish' if confluence < -0.1 else 'neutral')
        }
    
    def _calculate_mtf_confluence(self, mtf_ranges: Dict) -> float:
        """Calculate multi-timeframe confluence score."""
        if not mtf_ranges:
            return 0.0
        
        weighted_positions = []
        
        for tf, data in mtf_ranges.items():
            range_data = data['range']
            weight = data['weight']
            
            if range_data.get('size', 0) > 0:
                # Calculate relative position in range
                mid = range_data.get('mid', 0)
                high = range_data.get('high', 0)
                low = range_data.get('low', 0)
                
                if high > low:
                    position = (mid - low) / (high - low) - 0.5  # -0.5 to +0.5
                    weighted_positions.append(position * weight)
        
        return sum(weighted_positions) if weighted_positions else 0.0
    
    def _analyze_session_bias(self, symbol: str, session: str, candles: List[Dict]) -> Dict[str, Any]:
        """Analyze session-specific bias and characteristics."""
        if session not in self.sessions:
            return {'bias': 'neutral', 'confidence': 0.0}
        
        session_info = self.sessions[session]
        
        # Calculate session performance
        session_perf = self.session_performance[f"{symbol}_{session}"]
        
        # Determine bias based on session characteristics
        characteristics = session_info['characteristics']
        volatility = session_info['volatility_profile']
        
        # Simple bias calculation (would be more sophisticated in production)
        if 'breakouts' in characteristics:
            bias = 'breakout_continuation'
            confidence = 0.7
        elif 'range_bound' in characteristics:
            bias = 'mean_reversion'
            confidence = 0.6
        elif 'trend_initiation' in characteristics:
            bias = 'trend_following'
            confidence = 0.8
        else:
            bias = 'neutral'
            confidence = 0.5
        
        return {
            'bias': bias,
            'confidence': confidence,
            'volatility_profile': volatility,
            'characteristics': characteristics,
            'session_performance': {
                'total_signals': session_perf['total'],
                'success_rate': session_perf['correct'] / max(1, session_perf['total'])
            }
        }
    
    def _analyze_breakout_probability(self, dealing_range: Dict, candles: List[Dict]) -> Dict[str, Any]:
        """Analyze probability of range breakout/breakdown."""
        try:
            if not candles or dealing_range.get('size', 0) == 0:
                return {'breakout_probability': 0.0, 'direction': 'unknown'}
            
            range_high = dealing_range['high']
            range_low = dealing_range['low']
            current_price = float(candles[-1]['close'])
            
            # Calculate proximity to range boundaries
            proximity_to_high = abs(current_price - range_high) / dealing_range['size']
            proximity_to_low = abs(current_price - range_low) / dealing_range['size']
            
            # Calculate momentum towards boundaries
            recent_candles = candles[-5:]
            price_momentum = (float(recent_candles[-1]['close']) - float(recent_candles[0]['close'])) / dealing_range['size']
            
            # Volume analysis (simplified)
            recent_volumes = [float(c.get('tick_volume', 1000)) for c in recent_candles]
            avg_volume = np.mean(recent_volumes)
            current_volume = recent_volumes[-1]
            volume_surge = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate breakout probability
            if proximity_to_high < 0.1 and price_momentum > 0:
                breakout_prob = min(0.9, 0.5 + (volume_surge - 1) * 0.3 + abs(price_momentum) * 2)
                direction = 'upward'
            elif proximity_to_low < 0.1 and price_momentum < 0:
                breakout_prob = min(0.9, 0.5 + (volume_surge - 1) * 0.3 + abs(price_momentum) * 2)
                direction = 'downward'
            else:
                breakout_prob = 0.3  # Base probability
                direction = 'upward' if price_momentum > 0 else 'downward'
            
            return {
                'breakout_probability': breakout_prob,
                'direction': direction,
                'proximity_to_high': proximity_to_high,
                'proximity_to_low': proximity_to_low,
                'price_momentum': price_momentum,
                'volume_surge': volume_surge
            }
            
        except Exception as e:
            return {'breakout_probability': 0.0, 'direction': 'unknown', 'error': str(e)}
    
    def _identify_institutional_levels(self, dealing_range: Dict, candles: List[Dict]) -> Dict[str, Any]:
        """Identify institutional levels within the dealing range."""
        try:
            range_high = dealing_range['high']
            range_low = dealing_range['low']
            range_size = dealing_range['size']
            
            # Standard institutional levels
            levels = {
                'weekly_high': range_high,
                'weekly_low': range_low,
                'daily_high': range_high * 0.95,  # Slightly below weekly high
                'daily_low': range_low * 1.05,   # Slightly above weekly low
                'asian_range_high': range_low + (range_size * 0.6),
                'asian_range_low': range_low + (range_size * 0.4),
                'london_range_high': range_low + (range_size * 0.8),
                'london_range_low': range_low + (range_size * 0.2),
                'ny_range_high': range_high,
                'ny_range_low': range_low + (range_size * 0.3)
            }
            
            # Calculate level strengths based on historical touches
            level_strengths = {}
            for level_name, level_price in levels.items():
                touches = self._count_level_touches(candles, level_price, range_size * 0.001)
                level_strengths[level_name] = min(1.0, touches / 10)  # Normalize to 0-1
            
            return {
                'levels': levels,
                'strengths': level_strengths,
                'key_resistance': max(levels.items(), key=lambda x: level_strengths.get(x[0], 0)),
                'key_support': min(levels.items(), key=lambda x: level_strengths.get(x[0], 0))
            }
            
        except Exception as e:
            return {'levels': {}, 'strengths': {}, 'error': str(e)}
    
    def _count_level_touches(self, candles: List[Dict], level: float, tolerance: float) -> int:
        """Count how many times price touched a specific level."""
        touches = 0
        
        for candle in candles:
            high, low = float(candle['high']), float(candle['low'])
            
            # Check if candle touched the level within tolerance
            if abs(high - level) <= tolerance or abs(low - level) <= tolerance:
                touches += 1
        
        return touches
    
    def _create_pd_response(self, 
                           valid: bool, 
                           dealing_range: Dict = None,
                           pd_analysis: Dict = None,
                           mtf_ranges: Dict = None,
                           session_bias: Dict = None,
                           breakout_analysis: Dict = None,
                           institutional_levels: Dict = None,
                           current_session: str = "",
                           symbol: str = "",
                           timeframe: str = "",
                           error: str = "") -> Dict[str, Any]:
        """Create comprehensive premium/discount response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "current_session": current_session,
            "dealing_range": dealing_range or {},
            "premium_discount": pd_analysis or {},
            "multi_timeframe": mtf_ranges or {},
            "session_bias": session_bias or {},
            "breakout_analysis": breakout_analysis or {},
            "institutional_levels": institutional_levels or {},
            "metadata": {
                "total_analyses": self.total_analyses,
                "engine_version": "1.0.0",
                "analysis_type": "premium_discount"
            }
        }
    
    def _update_performance_tracking(self, response: Dict):
        """Update performance tracking metrics."""
        try:
            self.total_analyses += 1
            
            # Store analysis for future validation
            self.range_history[response.get('symbol', 'unknown')].append({
                'timestamp': datetime.now(),
                'analysis': response,
                'session': response.get('current_session', 'unknown')
            })
            
        except Exception:
            pass  # Silent fail for performance tracking
    
    def is_killzone_active(self, current_time: datetime = None) -> Dict[str, Any]:
        """Check if we're currently in an active killzone."""
        current_time = current_time or datetime.now()
        hour = current_time.hour
        
        active_killzones = []
        
        for session_name, session_info in self.sessions.items():
            if 'am_killzone' in session_info:
                am_start, am_end = session_info['am_killzone']
                if am_start <= hour <= am_end:
                    active_killzones.append(f"{session_name}_am")
            
            if 'pm_killzone' in session_info:
                pm_start, pm_end = session_info['pm_killzone']
                if pm_start <= hour <= pm_end:
                    active_killzones.append(f"{session_name}_pm")
        
        return {
            'active': len(active_killzones) > 0,
            'killzones': active_killzones,
            'current_hour': hour,
            'session_activity': 'high' if active_killzones else 'normal'
        }
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        return {
            "total_analyses": self.total_analyses,
            "successful_predictions": self.successful_predictions,
            "success_rate": self.successful_predictions / max(1, self.total_analyses),
            "session_performance": dict(self.session_performance),
            "dealing_ranges_tracked": len(self.dealing_ranges),
            "range_history_size": sum(len(hist) for hist in self.range_history.values()),
            "engine_version": "1.0.0"
        }