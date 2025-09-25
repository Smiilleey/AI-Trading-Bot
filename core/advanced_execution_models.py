# core/advanced_execution_models.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class EntryStrategy(Enum):
    """Entry execution strategies."""
    FVG_MIDPOINT = "fvg_midpoint"
    OB_EXTREME = "ob_extreme"
    STRUCTURE_BREAK = "structure_break"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    PREMIUM_DISCOUNT = "premium_discount"

class ExitTrigger(Enum):
    """Exit trigger types."""
    OPPOSING_LIQUIDITY = "opposing_liquidity"
    DISPLACEMENT_FADE = "displacement_fade"
    TIME_INVALIDATION = "time_invalidation"
    STRUCTURE_BREAK = "structure_break"
    TARGET_REACHED = "target_reached"
    STOP_LOSS = "stop_loss"

class AdvancedExecutionModels:
    """
    ADVANCED EXECUTION MODELS - Institutional-Grade Entry and Exit Logic
    
    Features:
    - Entry models: partials at FVG mid/OB extremes with refined stops
    - Dynamic TP/BE rules: partial profit-taking at opposing liquidity
    - Breakeven on displacement fade
    - Time-based invalidation criteria
    - Model-specific invalidation rules
    - Institutional-grade position management
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Entry model parameters
        self.entry_params = {
            'fvg_midpoint_allocation': 0.6,    # 60% at FVG midpoint
            'ob_extreme_allocation': 0.4,      # 40% at OB extreme
            'max_entry_levels': 3,             # Maximum entry levels
            'entry_spacing_pips': 5,           # Minimum spacing between entries
            'partial_fill_timeout': 300,      # 5 minutes timeout for partials
        }
        
        # Stop loss parameters
        self.stop_params = {
            'swing_buffer_pips': 2,            # Buffer beyond swing points
            'structure_buffer_pips': 3,        # Buffer for structure breaks
            'atr_multiplier': 1.5,             # ATR-based stop multiplier
            'max_stop_pips': 50,               # Maximum stop loss
            'min_stop_pips': 8,                # Minimum stop loss
        }
        
        # Take profit parameters
        self.tp_params = {
            'partial_tp_percentage': 50,       # 50% partial at first target
            'full_tp_percentage': 100,         # 100% at final target
            'opposing_liquidity_tp': 70,       # 70% at opposing liquidity
            'structure_target_multiplier': 2.0, # 2R at structure targets
            'time_based_tp_hours': 4,          # 4 hours max hold time
        }
        
        # Breakeven parameters
        self.be_params = {
            'displacement_fade_threshold': 0.382,  # Fibonacci retrace level
            'volume_fade_threshold': 0.6,          # Volume decrease threshold
            'time_to_be_minutes': 15,              # Time until BE consideration
            'profit_buffer_pips': 3,               # Buffer above breakeven
        }
        
        # Invalidation criteria
        self.invalidation_criteria = {
            'time_based': {
                'max_hold_hours': 8,               # Maximum position hold time
                'session_cutoff': True,            # Close at session end
                'weekend_cutoff': True             # Close before weekend
            },
            'structure_based': {
                'key_level_break': True,           # Invalidate on key level break
                'htf_structure_change': True,      # Invalidate on HTF structure change
                'pd_zone_exit': True               # Invalidate when leaving PD zone
            },
            'flow_based': {
                'absorption_reversal': True,       # Invalidate on absorption reversal
                'volume_divergence': True,         # Invalidate on volume divergence
                'institutional_exit': True        # Invalidate on institutional exit
            }
        }
        
        # Position tracking
        self.active_positions = {}
        self.position_history = deque(maxlen=1000)
        self.execution_performance = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.partial_fill_rate = 0.0
        self.average_hold_time = 0.0
    
    def calculate_entry_levels(self, 
                              signal_data: Dict,
                              market_structure: Dict,
                              pd_analysis: Dict,
                              symbol: str,
                              total_position_size: float) -> Dict[str, Any]:
        """
        Calculate multiple entry levels for institutional-style partial scaling.
        
        Args:
            signal_data: Validated signal data
            market_structure: Market structure analysis (FVGs, OBs, etc.)
            pd_analysis: Premium/discount analysis
            symbol: Trading symbol
            total_position_size: Total intended position size
            
        Returns:
            Entry execution plan with multiple levels
        """
        try:
            entry_plan = {
                'total_size': total_position_size,
                'entry_levels': [],
                'primary_strategy': None,
                'risk_parameters': {},
                'execution_timeline': {}
            }
            
            signal_direction = signal_data.get('direction', 'neutral')
            if signal_direction == 'neutral':
                return entry_plan
            
            current_price = float(signal_data.get('current_price', 0))
            
            # 1. Identify optimal entry zones
            entry_zones = self._identify_entry_zones(
                signal_direction, market_structure, pd_analysis, current_price
            )
            
            # 2. Calculate entry allocations
            entry_allocations = self._calculate_entry_allocations(
                entry_zones, total_position_size, signal_data
            )
            
            # 3. Set stop loss levels
            stop_levels = self._calculate_refined_stops(
                signal_direction, market_structure, entry_zones, current_price
            )
            
            # 4. Create entry levels
            for i, (zone, allocation) in enumerate(zip(entry_zones, entry_allocations)):
                if allocation > 0:
                    entry_level = {
                        'level': zone['price'],
                        'size': allocation,
                        'strategy': zone['strategy'].value,
                        'priority': i + 1,
                        'stop_loss': stop_levels.get(zone['strategy'].value, stop_levels['default']),
                        'timeout': zone.get('timeout', self.entry_params['partial_fill_timeout']),
                        'zone_info': zone
                    }
                    entry_plan['entry_levels'].append(entry_level)
            
            # Set primary strategy
            if entry_plan['entry_levels']:
                entry_plan['primary_strategy'] = entry_plan['entry_levels'][0]['strategy']
            
            # Calculate risk parameters
            entry_plan['risk_parameters'] = self._calculate_entry_risk_parameters(
                entry_plan['entry_levels'], signal_data
            )
            
            # Set execution timeline
            entry_plan['execution_timeline'] = self._create_execution_timeline(
                entry_plan['entry_levels']
            )
            
            return entry_plan
            
        except Exception as e:
            return {'error': f"Entry calculation failed: {str(e)}"}
    
    def _identify_entry_zones(self, 
                             direction: str,
                             market_structure: Dict,
                             pd_analysis: Dict,
                             current_price: float) -> List[Dict]:
        """Identify optimal entry zones based on market structure."""
        try:
            entry_zones = []
            
            # 1. FVG Midpoint Entry
            fvgs = market_structure.get('fair_value_gaps', {}).get('unfilled_fvgs', [])
            for fvg in fvgs:
                if ((direction == 'bullish' and fvg['type'] == 'bullish') or
                    (direction == 'bearish' and fvg['type'] == 'bearish')):
                    
                    entry_zones.append({
                        'price': fvg['midpoint'],
                        'strategy': EntryStrategy.FVG_MIDPOINT,
                        'strength': fvg['strength'],
                        'timeout': 300,  # 5 minutes
                        'description': f"FVG midpoint at {fvg['midpoint']:.5f}"
                    })
            
            # 2. Order Block Extreme Entry
            order_blocks = market_structure.get('order_blocks', [])
            for ob in order_blocks:
                if ((direction == 'bullish' and ob['type'] == 'bullish') or
                    (direction == 'bearish' and ob['type'] == 'bearish')):
                    
                    # Use body extreme for entry
                    if direction == 'bullish':
                        entry_price = ob['body_bottom']
                    else:
                        entry_price = ob['body_top']
                    
                    entry_zones.append({
                        'price': entry_price,
                        'strategy': EntryStrategy.OB_EXTREME,
                        'strength': ob['strength'],
                        'timeout': 600,  # 10 minutes
                        'description': f"OB extreme at {entry_price:.5f}"
                    })
            
            # 3. Premium/Discount Zone Entry
            pd_info = pd_analysis.get('premium_discount', {})
            if pd_info:
                pd_status = pd_info.get('status', 'equilibrium')
                
                if direction == 'bullish' and pd_status == 'discount':
                    discount_level = pd_info.get('levels', {}).get('discount', current_price)
                    entry_zones.append({
                        'price': discount_level,
                        'strategy': EntryStrategy.PREMIUM_DISCOUNT,
                        'strength': pd_info.get('strength', 0.5),
                        'timeout': 900,  # 15 minutes
                        'description': f"Discount zone entry at {discount_level:.5f}"
                    })
                    
                elif direction == 'bearish' and pd_status == 'premium':
                    premium_level = pd_info.get('levels', {}).get('premium', current_price)
                    entry_zones.append({
                        'price': premium_level,
                        'strategy': EntryStrategy.PREMIUM_DISCOUNT,
                        'strength': pd_info.get('strength', 0.5),
                        'timeout': 900,  # 15 minutes
                        'description': f"Premium zone entry at {premium_level:.5f}"
                    })
            
            # 4. Liquidity Sweep Entry
            liquidity_pools = market_structure.get('liquidity_pools', [])
            for pool in liquidity_pools:
                pool_bias = pool.get('bias', 'neutral')
                
                if ((direction == 'bullish' and 'buy_stops_below' in pool_bias) or
                    (direction == 'bearish' and 'sell_stops_above' in pool_bias)):
                    
                    entry_zones.append({
                        'price': pool['center'],
                        'strategy': EntryStrategy.LIQUIDITY_SWEEP,
                        'strength': pool['strength'],
                        'timeout': 180,  # 3 minutes (quick execution needed)
                        'description': f"Liquidity sweep at {pool['center']:.5f}"
                    })
            
            # Sort by strength and proximity to current price
            entry_zones.sort(key=lambda x: (x['strength'], -abs(x['price'] - current_price)), reverse=True)
            
            # Limit to max entry levels
            return entry_zones[:self.entry_params['max_entry_levels']]
            
        except Exception as e:
            return []
    
    def _calculate_entry_allocations(self, 
                                    entry_zones: List[Dict],
                                    total_size: float,
                                    signal_data: Dict) -> List[float]:
        """Calculate position size allocation for each entry level."""
        try:
            if not entry_zones:
                return []
            
            allocations = []
            remaining_size = total_size
            
            # Primary allocation based on strategy
            for i, zone in enumerate(entry_zones):
                strategy = zone['strategy']
                strength = zone['strength']
                
                if i == 0:  # Primary entry
                    if strategy == EntryStrategy.FVG_MIDPOINT:
                        allocation = total_size * self.entry_params['fvg_midpoint_allocation']
                    elif strategy == EntryStrategy.OB_EXTREME:
                        allocation = total_size * self.entry_params['ob_extreme_allocation']
                    elif strategy == EntryStrategy.LIQUIDITY_SWEEP:
                        allocation = total_size * 0.7  # 70% for liquidity sweeps
                    else:
                        allocation = total_size * 0.5  # 50% default
                        
                elif i == 1:  # Secondary entry
                    allocation = remaining_size * 0.6  # 60% of remaining
                    
                else:  # Tertiary entries
                    allocation = remaining_size / (len(entry_zones) - i)
                
                # Adjust by signal strength
                allocation *= strength
                allocation = min(allocation, remaining_size)
                
                allocations.append(allocation)
                remaining_size -= allocation
                
                if remaining_size <= 0.01:  # Minimum lot size
                    break
            
            return allocations
            
        except Exception as e:
            return [total_size] if entry_zones else []
    
    def _calculate_refined_stops(self, 
                                direction: str,
                                market_structure: Dict,
                                entry_zones: List[Dict],
                                current_price: float) -> Dict[str, float]:
        """Calculate refined stop loss levels beyond swing points."""
        try:
            stop_levels = {}
            
            # Default stop calculation
            swing_buffer = self.stop_params['swing_buffer_pips'] * 0.0001  # Convert to price
            
            # Find recent swing levels
            recent_swing_high = current_price * 1.001  # Placeholder
            recent_swing_low = current_price * 0.999   # Placeholder
            
            if direction == 'bullish':
                default_stop = recent_swing_low - swing_buffer
            else:
                default_stop = recent_swing_high + swing_buffer
            
            stop_levels['default'] = default_stop
            
            # Strategy-specific stops
            for zone in entry_zones:
                strategy = zone['strategy']
                entry_price = zone['price']
                
                if strategy == EntryStrategy.FVG_MIDPOINT:
                    # Stop beyond FVG boundary
                    if direction == 'bullish':
                        stop = entry_price * 0.998  # 20 pips below (approximate)
                    else:
                        stop = entry_price * 1.002  # 20 pips above
                    stop_levels['fvg_midpoint'] = stop
                    
                elif strategy == EntryStrategy.OB_EXTREME:
                    # Stop beyond order block
                    if direction == 'bullish':
                        stop = entry_price * 0.9985  # 15 pips below
                    else:
                        stop = entry_price * 1.0015  # 15 pips above
                    stop_levels['ob_extreme'] = stop
                    
                elif strategy == EntryStrategy.LIQUIDITY_SWEEP:
                    # Tight stop for liquidity sweeps
                    if direction == 'bullish':
                        stop = entry_price * 0.9992  # 8 pips below
                    else:
                        stop = entry_price * 1.0008  # 8 pips above
                    stop_levels['liquidity_sweep'] = stop
                    
                else:
                    stop_levels[strategy.value] = default_stop
            
            # Validate stop levels
            for strategy, stop in stop_levels.items():
                if direction == 'bullish':
                    stop_pips = (current_price - stop) * 10000
                else:
                    stop_pips = (stop - current_price) * 10000
                
                # Enforce min/max stop
                if stop_pips < self.stop_params['min_stop_pips']:
                    if direction == 'bullish':
                        stop_levels[strategy] = current_price - (self.stop_params['min_stop_pips'] * 0.0001)
                    else:
                        stop_levels[strategy] = current_price + (self.stop_params['min_stop_pips'] * 0.0001)
                        
                elif stop_pips > self.stop_params['max_stop_pips']:
                    if direction == 'bullish':
                        stop_levels[strategy] = current_price - (self.stop_params['max_stop_pips'] * 0.0001)
                    else:
                        stop_levels[strategy] = current_price + (self.stop_params['max_stop_pips'] * 0.0001)
            
            return stop_levels
            
        except Exception as e:
            return {'default': current_price * (0.999 if direction == 'bullish' else 1.001)}
    
    def _calculate_entry_risk_parameters(self, entry_levels: List[Dict], signal_data: Dict) -> Dict[str, Any]:
        """Calculate risk parameters for entry execution."""
        try:
            if not entry_levels:
                return {}
            
            # Calculate total risk
            total_risk = 0.0
            for level in entry_levels:
                entry_price = level['level']
                stop_price = level['stop_loss']
                size = level['size']
                
                risk_per_unit = abs(entry_price - stop_price)
                level_risk = risk_per_unit * size * 100000  # Standard lot calculation
                total_risk += level_risk
            
            # Calculate average entry price (weighted by size)
            total_size = sum(level['size'] for level in entry_levels)
            if total_size > 0:
                avg_entry = sum(level['level'] * level['size'] for level in entry_levels) / total_size
            else:
                avg_entry = entry_levels[0]['level']
            
            # Calculate R-multiple targets
            primary_stop = entry_levels[0]['stop_loss']
            r_distance = abs(avg_entry - primary_stop)
            
            return {
                'total_risk': total_risk,
                'average_entry': avg_entry,
                'primary_stop': primary_stop,
                'r_distance': r_distance,
                'r1_target': avg_entry + (r_distance * (1 if signal_data.get('direction') == 'bullish' else -1)),
                'r2_target': avg_entry + (r_distance * 2 * (1 if signal_data.get('direction') == 'bullish' else -1)),
                'r3_target': avg_entry + (r_distance * 3 * (1 if signal_data.get('direction') == 'bullish' else -1)),
                'max_risk_per_trade': total_risk
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_execution_timeline(self, entry_levels: List[Dict]) -> Dict[str, Any]:
        """Create execution timeline for entry levels."""
        try:
            timeline = {
                'immediate_execution': [],
                'conditional_execution': [],
                'total_timeout': 0
            }
            
            for level in entry_levels:
                strategy = level['strategy']
                
                if strategy in ['liquidity_sweep', 'structure_break']:
                    timeline['immediate_execution'].append(level)
                else:
                    timeline['conditional_execution'].append(level)
                
                timeline['total_timeout'] = max(timeline['total_timeout'], level.get('timeout', 300))
            
            return timeline
            
        except Exception:
            return {}
    
    def manage_position_dynamically(self, 
                                   position_id: str,
                                   position_data: Dict,
                                   current_market_data: Dict,
                                   market_structure: Dict) -> Dict[str, Any]:
        """
        Dynamic position management with institutional-style rules.
        
        Args:
            position_id: Unique position identifier
            position_data: Current position information
            current_market_data: Real-time market data
            market_structure: Current market structure analysis
            
        Returns:
            Position management actions (TP, BE, invalidation, etc.)
        """
        try:
            management_actions = {
                'actions': [],
                'risk_updates': [],
                'invalidation_triggers': [],
                'position_status': 'active'
            }
            
            # Extract position info
            entry_time = position_data.get('entry_time', datetime.now())
            entry_price = float(position_data.get('entry_price', 0))
            position_size = float(position_data.get('size', 0))
            direction = position_data.get('direction', 'neutral')
            stop_loss = float(position_data.get('stop_loss', 0))
            
            current_price = float(current_market_data.get('current_price', 0))
            current_time = datetime.now()
            
            # 1. Check for partial profit-taking at opposing liquidity
            opposing_liquidity_action = self._check_opposing_liquidity_tp(
                position_data, current_price, market_structure
            )
            if opposing_liquidity_action:
                management_actions['actions'].append(opposing_liquidity_action)
            
            # 2. Check for breakeven on displacement fade
            be_action = self._check_displacement_fade_be(
                position_data, current_market_data, market_structure
            )
            if be_action:
                management_actions['actions'].append(be_action)
            
            # 3. Check time-based invalidation
            time_invalidation = self._check_time_invalidation(
                position_data, current_time
            )
            if time_invalidation['should_invalidate']:
                management_actions['invalidation_triggers'].append(time_invalidation)
            
            # 4. Check structure-based invalidation
            structure_invalidation = self._check_structure_invalidation(
                position_data, market_structure
            )
            if structure_invalidation['should_invalidate']:
                management_actions['invalidation_triggers'].append(structure_invalidation)
            
            # 5. Check flow-based invalidation
            flow_invalidation = self._check_flow_invalidation(
                position_data, current_market_data
            )
            if flow_invalidation['should_invalidate']:
                management_actions['invalidation_triggers'].append(flow_invalidation)
            
            # 6. Update position status
            if management_actions['invalidation_triggers']:
                management_actions['position_status'] = 'should_close'
            elif management_actions['actions']:
                management_actions['position_status'] = 'managed'
            
            return management_actions
            
        except Exception as e:
            return {'error': f"Position management failed: {str(e)}"}
    
    def _check_opposing_liquidity_tp(self, 
                                    position_data: Dict,
                                    current_price: float,
                                    market_structure: Dict) -> Optional[Dict]:
        """Check for partial profit-taking at opposing liquidity."""
        try:
            direction = position_data.get('direction')
            entry_price = float(position_data.get('entry_price', 0))
            
            # Find opposing liquidity levels
            liquidity_pools = market_structure.get('liquidity_pools', [])
            
            for pool in liquidity_pools:
                pool_center = pool['center']
                pool_bias = pool.get('bias', 'neutral')
                
                # Check if we've reached opposing liquidity
                profit_distance = abs(current_price - entry_price)
                pool_distance = abs(current_price - pool_center)
                
                if pool_distance < profit_distance * 0.1:  # Within 10% of current profit
                    # Check if this is opposing liquidity
                    opposing = False
                    if direction == 'bullish' and 'sell_stops_above' in pool_bias:
                        opposing = True
                    elif direction == 'bearish' and 'buy_stops_below' in pool_bias:
                        opposing = True
                    
                    if opposing:
                        return {
                            'action': 'partial_take_profit',
                            'percentage': self.tp_params['opposing_liquidity_tp'],
                            'target_price': pool_center,
                            'reason': 'opposing_liquidity_reached',
                            'liquidity_info': pool
                        }
            
            return None
            
        except Exception:
            return None
    
    def _check_displacement_fade_be(self, 
                                   position_data: Dict,
                                   current_market_data: Dict,
                                   market_structure: Dict) -> Optional[Dict]:
        """Check for breakeven on displacement fade."""
        try:
            entry_time = position_data.get('entry_time', datetime.now())
            time_elapsed = (datetime.now() - entry_time).total_seconds() / 60  # minutes
            
            # Only consider BE after minimum time
            if time_elapsed < self.be_params['time_to_be_minutes']:
                return None
            
            entry_price = float(position_data.get('entry_price', 0))
            current_price = float(current_market_data.get('current_price', 0))
            direction = position_data.get('direction')
            
            # Calculate current profit
            if direction == 'bullish':
                profit_pips = (current_price - entry_price) * 10000
            else:
                profit_pips = (entry_price - current_price) * 10000
            
            # Only consider BE if in profit
            if profit_pips <= 0:
                return None
            
            # Check for displacement fade indicators
            fade_indicators = []
            
            # 1. Volume fade
            recent_volumes = current_market_data.get('recent_volumes', [])
            if recent_volumes and len(recent_volumes) >= 3:
                avg_recent = np.mean(recent_volumes[-3:])
                avg_previous = np.mean(recent_volumes[:-3]) if len(recent_volumes) > 3 else avg_recent
                
                if avg_recent < avg_previous * self.be_params['volume_fade_threshold']:
                    fade_indicators.append('volume_fade')
            
            # 2. Momentum fade
            momentum = current_market_data.get('momentum', 0)
            if abs(momentum) < 0.0001:  # Very low momentum
                fade_indicators.append('momentum_fade')
            
            # 3. Fibonacci retrace
            move_size = abs(current_price - entry_price)
            retrace_38 = entry_price + (move_size * self.be_params['displacement_fade_threshold'] * 
                                       (-1 if direction == 'bullish' else 1))
            
            if ((direction == 'bullish' and current_price <= retrace_38) or
                (direction == 'bearish' and current_price >= retrace_38)):
                fade_indicators.append('fibonacci_retrace')
            
            # If multiple fade indicators, move to BE
            if len(fade_indicators) >= 2:
                be_price = entry_price + (self.be_params['profit_buffer_pips'] * 0.0001 * 
                                         (1 if direction == 'bullish' else -1))
                
                return {
                    'action': 'move_to_breakeven',
                    'new_stop': be_price,
                    'reason': 'displacement_fade',
                    'fade_indicators': fade_indicators,
                    'current_profit_pips': profit_pips
                }
            
            return None
            
        except Exception:
            return None
    
    def _check_time_invalidation(self, position_data: Dict, current_time: datetime) -> Dict[str, Any]:
        """Check for time-based invalidation criteria."""
        try:
            invalidation = {'should_invalidate': False, 'reasons': []}
            
            entry_time = position_data.get('entry_time', current_time)
            time_elapsed = (current_time - entry_time).total_seconds() / 3600  # hours
            
            # 1. Maximum hold time
            if time_elapsed > self.invalidation_criteria['time_based']['max_hold_hours']:
                invalidation['should_invalidate'] = True
                invalidation['reasons'].append('max_hold_time_exceeded')
            
            # 2. Session cutoff
            if self.invalidation_criteria['time_based']['session_cutoff']:
                if self._is_session_ending(current_time):
                    invalidation['should_invalidate'] = True
                    invalidation['reasons'].append('session_cutoff')
            
            # 3. Weekend cutoff
            if self.invalidation_criteria['time_based']['weekend_cutoff']:
                if current_time.weekday() == 4 and current_time.hour >= 21:  # Friday 9 PM
                    invalidation['should_invalidate'] = True
                    invalidation['reasons'].append('weekend_cutoff')
            
            return invalidation
            
        except Exception:
            return {'should_invalidate': False, 'reasons': []}
    
    def _check_structure_invalidation(self, position_data: Dict, market_structure: Dict) -> Dict[str, Any]:
        """Check for structure-based invalidation."""
        try:
            invalidation = {'should_invalidate': False, 'reasons': []}
            
            direction = position_data.get('direction')
            
            # 1. Key level break
            if self.invalidation_criteria['structure_based']['key_level_break']:
                # Check if key support/resistance has been broken
                key_levels = market_structure.get('institutional_levels', {}).get('levels', {})
                
                if direction == 'bullish':
                    key_support = key_levels.get('key_support', [None, 0])[1]
                    current_price = float(market_structure.get('current_price', 0))
                    if key_support and current_price < key_support:
                        invalidation['should_invalidate'] = True
                        invalidation['reasons'].append('key_support_broken')
                        
                elif direction == 'bearish':
                    key_resistance = key_levels.get('key_resistance', [None, 0])[1]
                    current_price = float(market_structure.get('current_price', 0))
                    if key_resistance and current_price > key_resistance:
                        invalidation['should_invalidate'] = True
                        invalidation['reasons'].append('key_resistance_broken')
            
            # 2. HTF structure change (simplified)
            if self.invalidation_criteria['structure_based']['htf_structure_change']:
                # This would check for HTF trend changes
                # Simplified for now
                pass
            
            return invalidation
            
        except Exception:
            return {'should_invalidate': False, 'reasons': []}
    
    def _check_flow_invalidation(self, position_data: Dict, current_market_data: Dict) -> Dict[str, Any]:
        """Check for flow-based invalidation."""
        try:
            invalidation = {'should_invalidate': False, 'reasons': []}
            
            direction = position_data.get('direction')
            
            # 1. Absorption reversal
            if self.invalidation_criteria['flow_based']['absorption_reversal']:
                absorption = current_market_data.get('absorption_analysis', {})
                if absorption.get('type') == 'full' and absorption.get('direction') != direction:
                    invalidation['should_invalidate'] = True
                    invalidation['reasons'].append('absorption_reversal')
            
            # 2. Volume divergence
            if self.invalidation_criteria['flow_based']['volume_divergence']:
                volume_divergence = current_market_data.get('volume_divergence', False)
                if volume_divergence:
                    invalidation['should_invalidate'] = True
                    invalidation['reasons'].append('volume_divergence')
            
            # 3. Institutional exit
            if self.invalidation_criteria['flow_based']['institutional_exit']:
                institutional_activity = current_market_data.get('institutional_activity', 'low')
                if institutional_activity == 'exit' or institutional_activity == 'distribution':
                    invalidation['should_invalidate'] = True
                    invalidation['reasons'].append('institutional_exit')
            
            return invalidation
            
        except Exception:
            return {'should_invalidate': False, 'reasons': []}
    
    def _is_session_ending(self, current_time: datetime) -> bool:
        """Check if current session is ending."""
        try:
            hour = current_time.hour
            
            # NY session ends at 21:00
            if 20 <= hour <= 21:
                return True
            
            # London session ends at 16:00
            if 15 <= hour <= 16:
                return True
            
            # Asian session ends at 8:00
            if 7 <= hour <= 8:
                return True
            
            return False
            
        except Exception:
            return False
    
    def create_invalidation_plan(self, position_data: Dict) -> Dict[str, Any]:
        """Create model-specific invalidation plan."""
        try:
            plan = {
                'time_based_exit': None,
                'structure_based_exit': None,
                'flow_based_exit': None,
                'emergency_exit': None
            }
            
            entry_time = position_data.get('entry_time', datetime.now())
            
            # Time-based exit
            max_hold = self.invalidation_criteria['time_based']['max_hold_hours']
            exit_time = entry_time + timedelta(hours=max_hold)
            plan['time_based_exit'] = {
                'exit_time': exit_time,
                'reason': 'max_hold_time',
                'action': 'close_full_position'
            }
            
            # Structure-based exit levels
            stop_loss = float(position_data.get('stop_loss', 0))
            plan['structure_based_exit'] = {
                'stop_loss': stop_loss,
                'reason': 'stop_loss_hit',
                'action': 'close_full_position'
            }
            
            # Flow-based exit conditions
            plan['flow_based_exit'] = {
                'condition': 'institutional_exit_detected',
                'action': 'close_75_percent',
                'reason': 'smart_money_exit'
            }
            
            # Emergency exit
            plan['emergency_exit'] = {
                'condition': 'extreme_adverse_move',
                'threshold': '3x_atr_against',
                'action': 'immediate_close',
                'reason': 'risk_management'
            }
            
            return plan
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": self.successful_executions / max(1, self.total_executions),
            "partial_fill_rate": self.partial_fill_rate,
            "average_hold_time": self.average_hold_time,
            "execution_performance": dict(self.execution_performance),
            "active_positions_count": len(self.active_positions),
            "position_history_size": len(self.position_history),
            "entry_parameters": self.entry_params,
            "stop_parameters": self.stop_params,
            "tp_parameters": self.tp_params,
            "be_parameters": self.be_params,
            "invalidation_criteria": self.invalidation_criteria,
            "engine_version": "1.0.0"
        }
<<<<<<< Current (Your changes)
=======
        }
>>>>>>> 4323fc9 (upgraded)
=======
>>>>>>> Incoming (Background Agent changes)
