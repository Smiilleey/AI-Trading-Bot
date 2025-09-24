# core/operational_discipline.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, time
from collections import defaultdict, deque
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TradingState(Enum):
    """Trading state definitions."""
    ACTIVE = "active"
    RESTRICTED = "restricted"
    NO_TRADE = "no_trade"
    EMERGENCY_HALT = "emergency_halt"
    MAINTENANCE = "maintenance"

class NoTradeReason(Enum):
    """Reasons for no-trade states."""
    SLIPPAGE_SPIKE = "slippage_spike"
    SPREAD_EXPANSION = "spread_expansion"
    ROLLOVER_PERIOD = "rollover_period"
    NEWS_RELEASE = "news_release"
    CHOPPY_RANGE = "choppy_range"
    LOW_LIQUIDITY = "low_liquidity"
    OVERLAPPING_SESSIONS = "overlapping_sessions_low_quality"
    HOLIDAY_REGIME = "holiday_regime"
    EXTREME_VOLATILITY = "extreme_volatility"
    SYSTEM_MAINTENANCE = "system_maintenance"

class OperationalDiscipline:
    """
    OPERATIONAL DISCIPLINE - Guardrails and No-Trade State Management
    
    Features:
    - Slippage spike and spread expansion detection
    - Rollover and news release avoidance
    - Choppy range and low liquidity detection
    - Session overlap quality assessment
    - Holiday regime management
    - Clear no-trade state definitions
    - Automated system protection
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Slippage and spread thresholds
        self.slippage_thresholds = {
            'normal': 0.5,          # 0.5 pips normal slippage
            'elevated': 1.0,        # 1.0 pips elevated slippage
            'high': 2.0,            # 2.0 pips high slippage
            'extreme': 5.0,         # 5.0 pips extreme slippage
            'halt_threshold': 10.0  # 10.0 pips halt trading
        }
        
        self.spread_thresholds = {
            'forex_majors': 2.0,    # 2 pips max for majors
            'forex_minors': 4.0,    # 4 pips max for minors
            'forex_exotics': 8.0,   # 8 pips max for exotics
            'metals': 5.0,          # 5 pips max for metals
            'crypto': 10.0,         # 10 pips max for crypto
            'emergency_halt': 20.0  # 20 pips emergency halt
        }
        
        # Time-based no-trade periods
        self.no_trade_periods = {
            'rollover': {
                'start_time': time(21, 50),  # 21:50 UTC
                'end_time': time(22, 10),    # 22:10 UTC
                'description': 'Daily rollover period'
            },
            'weekend_close': {
                'day': 4,  # Friday
                'start_time': time(21, 30),  # 21:30 UTC
                'description': 'Weekend close approach'
            },
            'weekend_open': {
                'day': 6,  # Sunday
                'end_time': time(22, 0),     # 22:00 UTC
                'description': 'Weekend open settlement'
            },
            'low_liquidity_asian': {
                'start_time': time(22, 0),   # 22:00 UTC
                'end_time': time(2, 0),      # 02:00 UTC
                'description': 'Low liquidity Asian period'
            }
        }
        
        # Market condition thresholds
        self.market_condition_thresholds = {
            'choppy_range': {
                'min_range_ratio': 0.3,     # Range too small relative to ATR
                'max_false_breaks': 3,      # Too many false breakouts
                'volatility_threshold': 0.5  # Low volatility threshold
            },
            'low_liquidity': {
                'min_volume_ratio': 0.6,    # Volume below 60% of average
                'max_participants': 2,      # Too few active participants
                'session_overlap_min': 0.3  # Minimum session overlap quality
            },
            'extreme_volatility': {
                'atr_multiplier': 3.0,      # 3x normal ATR
                'price_gap_threshold': 0.002, # 20 pips gap
                'volume_surge_ratio': 5.0   # 5x volume surge
            }
        }
        
        # Current trading state
        self.current_state = TradingState.ACTIVE
        self.state_history = deque(maxlen=200)
        self.no_trade_reasons = []
        
        # Monitoring data
        self.slippage_monitoring = deque(maxlen=100)
        self.spread_monitoring = defaultdict(lambda: deque(maxlen=100))
        self.liquidity_monitoring = deque(maxlen=100)
        self.volatility_monitoring = deque(maxlen=100)
        
        # Alert system
        self.alerts = deque(maxlen=50)
        self.emergency_halts = deque(maxlen=20)
        
        # Performance tracking
        self.total_state_assessments = 0
        self.no_trade_periods_detected = 0
        self.emergency_halts_triggered = 0
        self.false_alerts = 0
    
    def assess_trading_state(self, 
                            market_data: Dict,
                            current_positions: List[Dict],
                            symbol: str,
                            current_time: datetime = None) -> Dict[str, Any]:
        """
        Comprehensive trading state assessment with operational discipline.
        
        Args:
            market_data: Real-time market data
            current_positions: Current open positions
            symbol: Trading symbol
            current_time: Current timestamp
            
        Returns:
            Trading state assessment with guardrails and restrictions
        """
        try:
            current_time = current_time or datetime.now()
            self.total_state_assessments += 1
            
            # 1. Check slippage conditions
            slippage_check = self._check_slippage_conditions(market_data, symbol)
            
            # 2. Check spread conditions
            spread_check = self._check_spread_conditions(market_data, symbol)
            
            # 3. Check time-based no-trade periods
            time_check = self._check_time_based_restrictions(current_time)
            
            # 4. Check market condition restrictions
            market_condition_check = self._check_market_conditions(market_data, symbol)
            
            # 5. Check session quality
            session_quality_check = self._check_session_quality(current_time, market_data)
            
            # 6. Check holiday regimes
            holiday_check = self._check_holiday_regimes(current_time)
            
            # 7. Check system health
            system_health_check = self._check_system_health(current_positions, market_data)
            
            # 8. Determine overall trading state
            trading_state_decision = self._determine_trading_state(
                slippage_check, spread_check, time_check, market_condition_check,
                session_quality_check, holiday_check, system_health_check
            )
            
            # Create comprehensive response
            response = self._create_discipline_response(
                True,
                slippage_check=slippage_check,
                spread_check=spread_check,
                time_check=time_check,
                market_condition_check=market_condition_check,
                session_quality_check=session_quality_check,
                holiday_check=holiday_check,
                system_health_check=system_health_check,
                trading_state_decision=trading_state_decision,
                symbol=symbol,
                current_time=current_time
            )
            
            # Update monitoring and tracking
            self._update_monitoring(response, market_data)
            
            return response
            
        except Exception as e:
            return self._create_discipline_response(False, error=f"State assessment failed: {str(e)}")
    
    def _check_slippage_conditions(self, market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Check for slippage spikes that require trading restrictions."""
        try:
            slippage_check = {
                'status': 'normal',
                'current_slippage': 0.0,
                'average_slippage': 0.0,
                'slippage_ratio': 1.0,
                'action_required': 'none',
                'restrictions': []
            }
            
            # Get current slippage (from market data or estimate)
            current_slippage = float(market_data.get('estimated_slippage_pips', 0.5))
            
            # Calculate average slippage
            if self.slippage_monitoring:
                recent_slippage = [s['slippage'] for s in list(self.slippage_monitoring)[-20:]]
                average_slippage = np.mean(recent_slippage)
            else:
                average_slippage = 0.5  # Default
            
            slippage_ratio = current_slippage / average_slippage if average_slippage > 0 else 1.0
            
            slippage_check.update({
                'current_slippage': current_slippage,
                'average_slippage': average_slippage,
                'slippage_ratio': slippage_ratio
            })
            
            # Determine slippage status and required actions
            if current_slippage >= self.slippage_thresholds['halt_threshold']:
                slippage_check['status'] = 'halt'
                slippage_check['action_required'] = 'emergency_halt'
                slippage_check['restrictions'] = ['halt_all_trading', 'close_risky_positions']
                
            elif current_slippage >= self.slippage_thresholds['extreme']:
                slippage_check['status'] = 'extreme'
                slippage_check['action_required'] = 'severe_restrictions'
                slippage_check['restrictions'] = ['no_new_positions', 'reduce_position_sizes']
                
            elif current_slippage >= self.slippage_thresholds['high']:
                slippage_check['status'] = 'high'
                slippage_check['action_required'] = 'moderate_restrictions'
                slippage_check['restrictions'] = ['limit_new_positions', 'smaller_sizes']
                
            elif current_slippage >= self.slippage_thresholds['elevated']:
                slippage_check['status'] = 'elevated'
                slippage_check['action_required'] = 'monitor_closely'
                slippage_check['restrictions'] = ['increased_monitoring']
            
            # Spike detection (sudden increase)
            if slippage_ratio > 3.0:  # 3x normal slippage
                slippage_check['spike_detected'] = True
                slippage_check['restrictions'].append('slippage_spike_detected')
            
            return slippage_check
            
        except Exception as e:
            return {'status': 'unknown', 'error': str(e)}
    
    def _check_spread_conditions(self, market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Check for spread expansion that requires restrictions."""
        try:
            spread_check = {
                'status': 'normal',
                'current_spread': 0.0,
                'average_spread': 0.0,
                'spread_ratio': 1.0,
                'symbol_category': 'unknown',
                'action_required': 'none',
                'restrictions': []
            }
            
            # Get current spread
            current_spread = float(market_data.get('spread_pips', 1.0))
            
            # Determine symbol category
            symbol_category = self._categorize_symbol(symbol)
            spread_check['symbol_category'] = symbol_category
            
            # Get threshold for this symbol category
            threshold = self.spread_thresholds.get(symbol_category, self.spread_thresholds['forex_majors'])
            emergency_threshold = self.spread_thresholds['emergency_halt']
            
            # Calculate average spread for this symbol
            symbol_spreads = list(self.spread_monitoring.get(symbol, deque()))
            if symbol_spreads:
                average_spread = np.mean([s['spread'] for s in symbol_spreads[-20:]])
            else:
                average_spread = threshold * 0.5  # Default to half threshold
            
            spread_ratio = current_spread / average_spread if average_spread > 0 else 1.0
            
            spread_check.update({
                'current_spread': current_spread,
                'average_spread': average_spread,
                'spread_ratio': spread_ratio
            })
            
            # Determine spread status and actions
            if current_spread >= emergency_threshold:
                spread_check['status'] = 'emergency'
                spread_check['action_required'] = 'emergency_halt'
                spread_check['restrictions'] = ['halt_all_trading', 'wait_for_normalization']
                
            elif current_spread >= threshold * 2:
                spread_check['status'] = 'extreme'
                spread_check['action_required'] = 'severe_restrictions'
                spread_check['restrictions'] = ['no_new_positions', 'consider_closing_positions']
                
            elif current_spread >= threshold * 1.5:
                spread_check['status'] = 'elevated'
                spread_check['action_required'] = 'moderate_restrictions'
                spread_check['restrictions'] = ['limit_position_sizes', 'increased_monitoring']
                
            elif current_spread >= threshold:
                spread_check['status'] = 'high'
                spread_check['action_required'] = 'minor_restrictions'
                spread_check['restrictions'] = ['monitor_closely']
            
            # Expansion detection (sudden increase)
            if spread_ratio > 2.5:  # 2.5x normal spread
                spread_check['expansion_detected'] = True
                spread_check['restrictions'].append('spread_expansion_detected')
            
            return spread_check
            
        except Exception as e:
            return {'status': 'unknown', 'error': str(e)}
    
    def _categorize_symbol(self, symbol: str) -> str:
        """Categorize symbol for appropriate spread thresholds."""
        try:
            symbol_upper = symbol.upper()
            
            # Forex majors
            majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
            if any(major in symbol_upper for major in majors):
                return 'forex_majors'
            
            # Forex minors
            minors = ['EURGBP', 'EURJPY', 'EURCHF', 'GBPJPY', 'GBPCHF', 'AUDNZD', 'AUDCAD', 'AUDCHF', 'AUDJPY']
            if any(minor in symbol_upper for minor in minors):
                return 'forex_minors'
            
            # Metals
            if any(metal in symbol_upper for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
                return 'metals'
            
            # Crypto
            if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'XRP', 'ADA']):
                return 'crypto'
            
            # Default to forex exotics
            return 'forex_exotics'
            
        except Exception:
            return 'forex_majors'
    
    def _check_time_based_restrictions(self, current_time: datetime) -> Dict[str, Any]:
        """Check for time-based trading restrictions."""
        try:
            time_check = {
                'restrictions_active': False,
                'active_restrictions': [],
                'next_restriction': None,
                'time_until_next': 0
            }
            
            current_time_only = current_time.time()
            current_weekday = current_time.weekday()
            
            active_restrictions = []
            
            # Check rollover period
            rollover = self.no_trade_periods['rollover']
            if rollover['start_time'] <= current_time_only <= rollover['end_time']:
                active_restrictions.append({
                    'type': 'rollover',
                    'description': rollover['description'],
                    'severity': 'high'
                })
            
            # Check weekend close
            weekend_close = self.no_trade_periods['weekend_close']
            if (current_weekday == weekend_close['day'] and 
                current_time_only >= weekend_close['start_time']):
                active_restrictions.append({
                    'type': 'weekend_close',
                    'description': weekend_close['description'],
                    'severity': 'critical'
                })
            
            # Check weekend open
            weekend_open = self.no_trade_periods['weekend_open']
            if (current_weekday == weekend_open['day'] and 
                current_time_only <= weekend_open['end_time']):
                active_restrictions.append({
                    'type': 'weekend_open',
                    'description': weekend_open['description'],
                    'severity': 'high'
                })
            
            # Check low liquidity Asian period
            low_liquidity = self.no_trade_periods['low_liquidity_asian']
            if (low_liquidity['start_time'] <= current_time_only or
                current_time_only <= low_liquidity['end_time']):
                active_restrictions.append({
                    'type': 'low_liquidity_asian',
                    'description': low_liquidity['description'],
                    'severity': 'medium'
                })
            
            time_check['active_restrictions'] = active_restrictions
            time_check['restrictions_active'] = len(active_restrictions) > 0
            
            # Find next restriction
            next_restriction = self._find_next_time_restriction(current_time)
            time_check['next_restriction'] = next_restriction
            
            return time_check
            
        except Exception as e:
            return {'restrictions_active': False, 'error': str(e)}
    
    def _find_next_time_restriction(self, current_time: datetime) -> Optional[Dict]:
        """Find the next upcoming time-based restriction."""
        try:
            next_restrictions = []
            
            # Calculate time until next rollover
            rollover_time = datetime.combine(current_time.date(), self.no_trade_periods['rollover']['start_time'])
            if rollover_time <= current_time:
                rollover_time += timedelta(days=1)
            
            next_restrictions.append({
                'type': 'rollover',
                'time': rollover_time,
                'minutes_until': (rollover_time - current_time).total_seconds() / 60
            })
            
            # Calculate time until weekend close (Friday)
            days_until_friday = (4 - current_time.weekday()) % 7
            friday_close = datetime.combine(
                current_time.date() + timedelta(days=days_until_friday),
                self.no_trade_periods['weekend_close']['start_time']
            )
            
            if friday_close <= current_time:
                friday_close += timedelta(days=7)
            
            next_restrictions.append({
                'type': 'weekend_close',
                'time': friday_close,
                'minutes_until': (friday_close - current_time).total_seconds() / 60
            })
            
            # Return the soonest restriction
            return min(next_restrictions, key=lambda x: x['minutes_until'])
            
        except Exception:
            return None
    
    def _check_market_conditions(self, market_data: Dict, symbol: str) -> Dict[str, Any]:
        """Check market conditions for trading suitability."""
        try:
            condition_check = {
                'suitable_for_trading': True,
                'detected_conditions': [],
                'severity': 'none',
                'recommendations': []
            }
            
            detected_conditions = []
            
            # 1. Check for choppy range
            choppy_analysis = self._detect_choppy_range(market_data)
            if choppy_analysis['is_choppy']:
                detected_conditions.append({
                    'condition': 'choppy_range',
                    'severity': 'medium',
                    'details': choppy_analysis
                })
            
            # 2. Check for low liquidity
            liquidity_analysis = self._detect_low_liquidity(market_data)
            if liquidity_analysis['is_low_liquidity']:
                detected_conditions.append({
                    'condition': 'low_liquidity',
                    'severity': 'high',
                    'details': liquidity_analysis
                })
            
            # 3. Check for extreme volatility
            volatility_analysis = self._detect_extreme_volatility(market_data)
            if volatility_analysis['is_extreme']:
                detected_conditions.append({
                    'condition': 'extreme_volatility',
                    'severity': 'high',
                    'details': volatility_analysis
                })
            
            condition_check['detected_conditions'] = detected_conditions
            
            # Determine overall suitability
            if detected_conditions:
                max_severity = max(c['severity'] for c in detected_conditions)
                
                if max_severity == 'high' or len(detected_conditions) >= 2:
                    condition_check['suitable_for_trading'] = False
                    condition_check['severity'] = 'high'
                    condition_check['recommendations'] = [
                        'Avoid new positions',
                        'Monitor market conditions closely',
                        'Consider reducing existing exposure'
                    ]
                elif max_severity == 'medium':
                    condition_check['severity'] = 'medium'
                    condition_check['recommendations'] = [
                        'Reduce position sizes',
                        'Use wider stops',
                        'Increase monitoring frequency'
                    ]
            
            return condition_check
            
        except Exception as e:
            return {'suitable_for_trading': True, 'error': str(e)}
    
    def _detect_choppy_range(self, market_data: Dict) -> Dict[str, Any]:
        """Detect choppy, unsuitable range conditions."""
        try:
            choppy_analysis = {
                'is_choppy': False,
                'range_ratio': 0.0,
                'false_breaks': 0,
                'volatility_score': 0.0
            }
            
            candles = market_data.get('candles', [])
            if len(candles) < 20:
                return choppy_analysis
            
            # Calculate range characteristics
            recent_candles = candles[-20:]
            highs = [float(c['high']) for c in recent_candles]
            lows = [float(c['low']) for c in recent_candles]
            closes = [float(c['close']) for c in recent_candles]
            
            current_range = max(highs) - min(lows)
            avg_candle_range = np.mean([highs[i] - lows[i] for i in range(len(highs))])
            
            # Range ratio (current range vs average candle range)
            range_ratio = current_range / (avg_candle_range * len(recent_candles)) if avg_candle_range > 0 else 0
            choppy_analysis['range_ratio'] = range_ratio
            
            # Count false breakouts
            range_high = max(highs[:-1])  # Exclude current candle
            range_low = min(lows[:-1])
            
            false_breaks = 0
            for i in range(len(recent_candles) - 1):
                candle = recent_candles[i]
                next_candle = recent_candles[i + 1] if i + 1 < len(recent_candles) else candle
                
                # False breakout up
                if (float(candle['high']) > range_high and 
                    float(next_candle['close']) < range_high):
                    false_breaks += 1
                
                # False breakout down
                elif (float(candle['low']) < range_low and 
                      float(next_candle['close']) > range_low):
                    false_breaks += 1
            
            choppy_analysis['false_breaks'] = false_breaks
            
            # Calculate volatility score
            price_changes = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]
            volatility_score = np.std(price_changes) if price_changes else 0
            choppy_analysis['volatility_score'] = volatility_score
            
            # Determine if market is choppy
            thresholds = self.market_condition_thresholds['choppy_range']
            
            if (range_ratio < thresholds['min_range_ratio'] and
                false_breaks >= thresholds['max_false_breaks'] and
                volatility_score < thresholds['volatility_threshold']):
                choppy_analysis['is_choppy'] = True
            
            return choppy_analysis
            
        except Exception as e:
            return {'is_choppy': False, 'error': str(e)}
    
    def _detect_low_liquidity(self, market_data: Dict) -> Dict[str, Any]:
        """Detect low liquidity conditions."""
        try:
            liquidity_analysis = {
                'is_low_liquidity': False,
                'volume_ratio': 1.0,
                'participant_count': 0,
                'liquidity_score': 1.0
            }
            
            candles = market_data.get('candles', [])
            if len(candles) < 10:
                return liquidity_analysis
            
            # Volume analysis
            recent_volumes = [float(c.get('tick_volume', 1000)) for c in candles[-10:]]
            current_volume = recent_volumes[-1]
            avg_volume = np.mean(recent_volumes[:-1]) if len(recent_volumes) > 1 else current_volume
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            liquidity_analysis['volume_ratio'] = volume_ratio
            
            # Simple participant count estimation (based on volume distribution)
            volume_std = np.std(recent_volumes)
            volume_mean = np.mean(recent_volumes)
            cv = volume_std / volume_mean if volume_mean > 0 else 1.0
            
            # Higher coefficient of variation suggests fewer participants
            participant_count = max(1, int(5 / (cv + 0.1)))  # Rough estimation
            liquidity_analysis['participant_count'] = participant_count
            
            # Calculate liquidity score
            liquidity_score = (volume_ratio * 0.6) + (participant_count / 5 * 0.4)
            liquidity_analysis['liquidity_score'] = min(1.0, liquidity_score)
            
            # Check thresholds
            thresholds = self.market_condition_thresholds['low_liquidity']
            
            if (volume_ratio < thresholds['min_volume_ratio'] or
                participant_count < thresholds['max_participants'] or
                liquidity_score < 0.4):
                liquidity_analysis['is_low_liquidity'] = True
            
            return liquidity_analysis
            
        except Exception as e:
            return {'is_low_liquidity': False, 'error': str(e)}
    
    def _detect_extreme_volatility(self, market_data: Dict) -> Dict[str, Any]:
        """Detect extreme volatility conditions."""
        try:
            volatility_analysis = {
                'is_extreme': False,
                'atr_ratio': 1.0,
                'price_gaps': [],
                'volume_surge_ratio': 1.0
            }
            
            candles = market_data.get('candles', [])
            if len(candles) < 20:
                return volatility_analysis
            
            # ATR calculation
            recent_candles = candles[-20:]
            true_ranges = []
            
            for i in range(1, len(recent_candles)):
                current = recent_candles[i]
                previous = recent_candles[i-1]
                
                tr1 = float(current['high']) - float(current['low'])
                tr2 = abs(float(current['high']) - float(previous['close']))
                tr3 = abs(float(current['low']) - float(previous['close']))
                
                true_ranges.append(max(tr1, tr2, tr3))
            
            current_atr = np.mean(true_ranges[-5:]) if len(true_ranges) >= 5 else np.mean(true_ranges)
            baseline_atr = np.mean(true_ranges[:-5]) if len(true_ranges) > 5 else current_atr
            
            atr_ratio = current_atr / baseline_atr if baseline_atr > 0 else 1.0
            volatility_analysis['atr_ratio'] = atr_ratio
            
            # Price gap detection
            price_gaps = []
            for i in range(1, len(recent_candles)):
                current = recent_candles[i]
                previous = recent_candles[i-1]
                
                gap_up = float(current['low']) - float(previous['high'])
                gap_down = float(previous['low']) - float(current['high'])
                
                if gap_up > 0:
                    gap_size = gap_up / float(previous['close']) if float(previous['close']) > 0 else 0
                    if gap_size > self.market_condition_thresholds['extreme_volatility']['price_gap_threshold']:
                        price_gaps.append({'type': 'gap_up', 'size': gap_size, 'index': i})
                
                elif gap_down > 0:
                    gap_size = gap_down / float(previous['close']) if float(previous['close']) > 0 else 0
                    if gap_size > self.market_condition_thresholds['extreme_volatility']['price_gap_threshold']:
                        price_gaps.append({'type': 'gap_down', 'size': gap_size, 'index': i})
            
            volatility_analysis['price_gaps'] = price_gaps
            
            # Volume surge analysis
            recent_volumes = [float(c.get('tick_volume', 1000)) for c in recent_candles]
            current_volume = recent_volumes[-1]
            avg_volume = np.mean(recent_volumes[:-1]) if len(recent_volumes) > 1 else current_volume
            
            volume_surge_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volatility_analysis['volume_surge_ratio'] = volume_surge_ratio
            
            # Check for extreme conditions
            thresholds = self.market_condition_thresholds['extreme_volatility']
            
            if (atr_ratio > thresholds['atr_multiplier'] or
                len(price_gaps) > 0 or
                volume_surge_ratio > thresholds['volume_surge_ratio']):
                volatility_analysis['is_extreme'] = True
            
            return volatility_analysis
            
        except Exception as e:
            return {'is_extreme': False, 'error': str(e)}
    
    def _check_session_quality(self, current_time: datetime, market_data: Dict) -> Dict[str, Any]:
        """Check quality of overlapping sessions."""
        try:
            session_check = {
                'quality': 'normal',
                'active_sessions': [],
                'overlap_quality': 1.0,
                'recommendations': []
            }
            
            hour = current_time.hour
            
            # Identify active sessions
            active_sessions = []
            if 0 <= hour <= 8:
                active_sessions.append('asian')
            if 8 <= hour <= 16:
                active_sessions.append('london')
            if 13 <= hour <= 21:
                active_sessions.append('newyork')
            
            session_check['active_sessions'] = active_sessions
            
            # Assess overlap quality
            if len(active_sessions) == 2:
                # London/NY overlap (best quality)
                if 'london' in active_sessions and 'newyork' in active_sessions:
                    session_check['quality'] = 'excellent'
                    session_check['overlap_quality'] = 1.2
                # Asian/London overlap
                elif 'asian' in active_sessions and 'london' in active_sessions:
                    session_check['quality'] = 'good'
                    session_check['overlap_quality'] = 0.9
                # Other overlaps
                else:
                    session_check['quality'] = 'fair'
                    session_check['overlap_quality'] = 0.8
                    
            elif len(active_sessions) == 1:
                # Single session
                session = active_sessions[0]
                if session == 'london':
                    session_check['quality'] = 'good'
                    session_check['overlap_quality'] = 0.9
                elif session == 'newyork':
                    session_check['quality'] = 'good'
                    session_check['overlap_quality'] = 0.8
                else:  # Asian
                    session_check['quality'] = 'low'
                    session_check['overlap_quality'] = 0.6
                    session_check['recommendations'].append('Avoid high-risk strategies during Asian session')
                    
            else:
                # No major session active
                session_check['quality'] = 'poor'
                session_check['overlap_quality'] = 0.4
                session_check['recommendations'].append('Avoid trading during off-hours')
            
            # Check for low-quality overlaps
            min_quality = self.market_condition_thresholds['low_liquidity']['session_overlap_min']
            if session_check['overlap_quality'] < min_quality:
                session_check['recommendations'].append('Low session quality detected - reduce position sizes')
            
            return session_check
            
        except Exception as e:
            return {'quality': 'unknown', 'error': str(e)}
    
    def _check_holiday_regimes(self, current_time: datetime) -> Dict[str, Any]:
        """Check for holiday regimes and special periods."""
        try:
            holiday_check = {
                'is_holiday_regime': False,
                'holiday_type': 'none',
                'impact_level': 'none',
                'restrictions': []
            }
            
            month = current_time.month
            day = current_time.day
            weekday = current_time.weekday()
            
            # Major holiday periods
            holiday_periods = {
                'christmas_new_year': {
                    'condition': (month == 12 and day >= 20) or (month == 1 and day <= 5),
                    'impact': 'extreme',
                    'restrictions': ['halt_all_new_positions', 'close_risky_positions']
                },
                'thanksgiving_week': {
                    'condition': (month == 11 and 22 <= day <= 28),
                    'impact': 'high',
                    'restrictions': ['reduce_position_sizes', 'avoid_risky_strategies']
                },
                'summer_holidays': {
                    'condition': (month in [7, 8]),
                    'impact': 'medium',
                    'restrictions': ['monitor_liquidity_closely']
                },
                'chinese_new_year': {
                    'condition': (month == 2 and 10 <= day <= 17),
                    'impact': 'high',
                    'restrictions': ['avoid_asian_session_trading', 'reduce_commodity_exposure']
                }
            }
            
            # Check each holiday period
            for holiday_name, holiday_info in holiday_periods.items():
                if holiday_info['condition']:
                    holiday_check['is_holiday_regime'] = True
                    holiday_check['holiday_type'] = holiday_name
                    holiday_check['impact_level'] = holiday_info['impact']
                    holiday_check['restrictions'] = holiday_info['restrictions']
                    break
            
            # Special day-of-week considerations
            if weekday == 4:  # Friday
                holiday_check['restrictions'].append('friday_risk_reduction')
            elif weekday == 0:  # Monday
                holiday_check['restrictions'].append('monday_gap_risk')
            
            return holiday_check
            
        except Exception as e:
            return {'is_holiday_regime': False, 'error': str(e)}
    
    def _check_system_health(self, current_positions: List[Dict], market_data: Dict) -> Dict[str, Any]:
        """Check system health and operational status."""
        try:
            health_check = {
                'system_healthy': True,
                'health_score': 1.0,
                'issues': [],
                'recommendations': []
            }
            
            issues = []
            health_factors = []
            
            # 1. Position health
            if current_positions:
                total_unrealized_pnl = sum(float(pos.get('unrealized_pnl', 0)) for pos in current_positions)
                losing_positions = [pos for pos in current_positions if float(pos.get('unrealized_pnl', 0)) < 0]
                
                if len(losing_positions) > len(current_positions) * 0.7:  # 70% losing
                    issues.append('high_losing_position_ratio')
                    health_factors.append(0.6)
                else:
                    health_factors.append(0.9)
            else:
                health_factors.append(1.0)
            
            # 2. Data quality health
            candles = market_data.get('candles', [])
            if len(candles) < 10:
                issues.append('insufficient_market_data')
                health_factors.append(0.4)
            else:
                # Check for missing data
                missing_data = sum(1 for c in candles[-10:] if not all(k in c for k in ['open', 'high', 'low', 'close']))
                if missing_data > 2:
                    issues.append('poor_data_quality')
                    health_factors.append(0.7)
                else:
                    health_factors.append(1.0)
            
            # 3. Performance health (simplified)
            # This would check recent system performance
            health_factors.append(0.8)  # Mock performance factor
            
            health_check['issues'] = issues
            health_check['health_score'] = np.mean(health_factors) if health_factors else 0.5
            
            # Determine system health status
            if health_check['health_score'] < 0.5:
                health_check['system_healthy'] = False
                health_check['recommendations'] = [
                    'System health degraded - consider maintenance mode',
                    'Check data feeds and connectivity',
                    'Review recent performance'
                ]
            elif health_check['health_score'] < 0.7:
                health_check['recommendations'] = [
                    'Monitor system health closely',
                    'Reduce position sizes temporarily'
                ]
            
            return health_check
            
        except Exception as e:
            return {'system_healthy': True, 'error': str(e)}
    
    def _determine_trading_state(self, 
                                slippage_check: Dict,
                                spread_check: Dict,
                                time_check: Dict,
                                market_condition_check: Dict,
                                session_quality_check: Dict,
                                holiday_check: Dict,
                                system_health_check: Dict) -> Dict[str, Any]:
        """Determine overall trading state based on all checks."""
        try:
            decision = {
                'trading_state': TradingState.ACTIVE,
                'restrictions': [],
                'risk_multiplier': 1.0,
                'reasons': [],
                'severity': 'none'
            }
            
            all_restrictions = []
            risk_multipliers = []
            severity_levels = []
            
            # Analyze each check
            checks = [
                ('slippage', slippage_check),
                ('spread', spread_check),
                ('time', time_check),
                ('market_conditions', market_condition_check),
                ('session_quality', session_quality_check),
                ('holiday', holiday_check),
                ('system_health', system_health_check)
            ]
            
            for check_name, check_data in checks:
                if check_data.get('action_required') == 'emergency_halt':
                    decision['trading_state'] = TradingState.EMERGENCY_HALT
                    severity_levels.append('critical')
                    decision['reasons'].append(f'{check_name}_emergency_halt')
                    risk_multipliers.append(0.0)
                    
                elif check_data.get('action_required') == 'severe_restrictions':
                    if decision['trading_state'] not in [TradingState.EMERGENCY_HALT]:
                        decision['trading_state'] = TradingState.NO_TRADE
                    severity_levels.append('high')
                    decision['reasons'].append(f'{check_name}_severe_restrictions')
                    risk_multipliers.append(0.2)
                    
                elif not check_data.get('suitable_for_trading', True):
                    if decision['trading_state'] == TradingState.ACTIVE:
                        decision['trading_state'] = TradingState.RESTRICTED
                    severity_levels.append('medium')
                    decision['reasons'].append(f'{check_name}_restrictions')
                    risk_multipliers.append(0.5)
                
                # Collect restrictions
                all_restrictions.extend(check_data.get('restrictions', []))
            
            # Add time-based restrictions
            if time_check.get('restrictions_active', False):
                for restriction in time_check.get('active_restrictions', []):
                    if restriction['severity'] == 'critical':
                        decision['trading_state'] = TradingState.NO_TRADE
                        severity_levels.append('critical')
                    all_restrictions.append(restriction['type'])
            
            # Add holiday restrictions
            if holiday_check.get('is_holiday_regime', False):
                impact = holiday_check.get('impact_level', 'none')
                if impact in ['extreme', 'high']:
                    decision['trading_state'] = TradingState.NO_TRADE
                    severity_levels.append('high')
                all_restrictions.extend(holiday_check.get('restrictions', []))
            
            decision['restrictions'] = list(set(all_restrictions))  # Remove duplicates
            
            # Calculate final risk multiplier
            if risk_multipliers:
                decision['risk_multiplier'] = min(risk_multipliers)
            
            # Determine severity
            if 'critical' in severity_levels:
                decision['severity'] = 'critical'
            elif 'high' in severity_levels:
                decision['severity'] = 'high'
            elif 'medium' in severity_levels:
                decision['severity'] = 'medium'
            elif severity_levels:
                decision['severity'] = 'low'
            
            # Update current state
            if self.current_state != decision['trading_state']:
                self._record_state_change(decision['trading_state'], decision['reasons'])
            
            return decision
            
        except Exception as e:
            return {'trading_state': TradingState.ACTIVE, 'error': str(e)}
    
    def _record_state_change(self, new_state: TradingState, reasons: List[str]):
        """Record trading state change."""
        try:
            state_change = {
                'timestamp': datetime.now(),
                'from_state': self.current_state.value,
                'to_state': new_state.value,
                'reasons': reasons
            }
            
            self.state_history.append(state_change)
            self.current_state = new_state
            
            # Log emergency halts
            if new_state == TradingState.EMERGENCY_HALT:
                self.emergency_halts_triggered += 1
                self.emergency_halts.append(state_change)
            
            # Log no-trade periods
            if new_state == TradingState.NO_TRADE:
                self.no_trade_periods_detected += 1
                
        except Exception:
            pass  # Silent fail for state recording
    
    def _create_discipline_response(self, 
                                   valid: bool,
                                   slippage_check: Dict = None,
                                   spread_check: Dict = None,
                                   time_check: Dict = None,
                                   market_condition_check: Dict = None,
                                   session_quality_check: Dict = None,
                                   holiday_check: Dict = None,
                                   system_health_check: Dict = None,
                                   trading_state_decision: Dict = None,
                                   symbol: str = "",
                                   current_time: datetime = None,
                                   error: str = "") -> Dict[str, Any]:
        """Create comprehensive operational discipline response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "symbol": symbol,
            "timestamp": (current_time or datetime.now()).isoformat(),
            "current_trading_state": self.current_state.value,
            "slippage_check": slippage_check or {},
            "spread_check": spread_check or {},
            "time_check": time_check or {},
            "market_condition_check": market_condition_check or {},
            "session_quality_check": session_quality_check or {},
            "holiday_check": holiday_check or {},
            "system_health_check": system_health_check or {},
            "trading_state_decision": trading_state_decision or {},
            "summary": {
                "trading_allowed": trading_state_decision.get('trading_state') == TradingState.ACTIVE if trading_state_decision else True,
                "risk_multiplier": trading_state_decision.get('risk_multiplier', 1.0) if trading_state_decision else 1.0,
                "restriction_count": len(trading_state_decision.get('restrictions', [])) if trading_state_decision else 0,
                "severity_level": trading_state_decision.get('severity', 'none') if trading_state_decision else 'none',
                "system_healthy": system_health_check.get('system_healthy', True) if system_health_check else True
            },
            "metadata": {
                "total_assessments": self.total_state_assessments,
                "no_trade_periods_detected": self.no_trade_periods_detected,
                "emergency_halts_triggered": self.emergency_halts_triggered,
                "engine_version": "1.0.0",
                "analysis_type": "operational_discipline"
            }
        }
    
    def _update_monitoring(self, response: Dict, market_data: Dict):
        """Update monitoring data for trend analysis."""
        try:
            timestamp = datetime.now()
            
            # Update slippage monitoring
            slippage = response.get('slippage_check', {}).get('current_slippage', 0.0)
            self.slippage_monitoring.append({
                'timestamp': timestamp,
                'slippage': slippage,
                'symbol': response.get('symbol', '')
            })
            
            # Update spread monitoring
            symbol = response.get('symbol', '')
            spread = response.get('spread_check', {}).get('current_spread', 0.0)
            self.spread_monitoring[symbol].append({
                'timestamp': timestamp,
                'spread': spread
            })
            
            # Update liquidity monitoring
            liquidity_score = response.get('market_condition_check', {}).get('detected_conditions', [])
            liquidity_value = 1.0
            for condition in liquidity_score:
                if condition.get('condition') == 'low_liquidity':
                    liquidity_value = condition.get('details', {}).get('liquidity_score', 1.0)
                    break
            
            self.liquidity_monitoring.append({
                'timestamp': timestamp,
                'liquidity_score': liquidity_value,
                'symbol': symbol
            })
            
            # Update volatility monitoring
            volatility_info = response.get('market_condition_check', {}).get('detected_conditions', [])
            volatility_value = 1.0
            for condition in volatility_info:
                if condition.get('condition') == 'extreme_volatility':
                    volatility_value = condition.get('details', {}).get('atr_ratio', 1.0)
                    break
            
            self.volatility_monitoring.append({
                'timestamp': timestamp,
                'volatility_ratio': volatility_value,
                'symbol': symbol
            })
            
        except Exception:
            pass  # Silent fail for monitoring updates
    
    def get_discipline_stats(self) -> Dict[str, Any]:
        """Get comprehensive operational discipline statistics."""
        return {
            "current_trading_state": self.current_state.value,
            "total_assessments": self.total_state_assessments,
            "no_trade_periods_detected": self.no_trade_periods_detected,
            "emergency_halts_triggered": self.emergency_halts_triggered,
            "false_alerts": self.false_alerts,
            "state_distribution": {
                state.value: sum(1 for entry in self.state_history if entry['to_state'] == state.value)
                for state in TradingState
            },
            "average_slippage": np.mean([s['slippage'] for s in self.slippage_monitoring]) if self.slippage_monitoring else 0.0,
            "average_spread_by_symbol": {
                symbol: np.mean([s['spread'] for s in spreads])
                for symbol, spreads in self.spread_monitoring.items()
                if spreads
            },
            "thresholds": {
                "slippage_thresholds": dict(self.slippage_thresholds),
                "spread_thresholds": dict(self.spread_thresholds),
                "market_condition_thresholds": dict(self.market_condition_thresholds)
            },
            "no_trade_periods": dict(self.no_trade_periods),
            "memory_sizes": {
                "state_history": len(self.state_history),
                "slippage_monitoring": len(self.slippage_monitoring),
                "spread_monitoring": sum(len(spreads) for spreads in self.spread_monitoring.values()),
                "liquidity_monitoring": len(self.liquidity_monitoring),
                "volatility_monitoring": len(self.volatility_monitoring),
                "alerts": len(self.alerts),
                "emergency_halts": len(self.emergency_halts)
            },
            "engine_version": "1.0.0"
<<<<<<< HEAD
        }
=======
        }
>>>>>>> 4323fc9 (upgraded)
