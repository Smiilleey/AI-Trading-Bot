# core/correlation_aware_risk.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class CorrelationAwareRiskManager:
    """
    CORRELATION-AWARE RISK MANAGER - Institutional-Grade Risk Overlays
    
    Features:
    - Correlation-aware exposure caps per USD leg and risk factor
    - Streak, session, and regime-aware risk multipliers
    - Cooling-off periods after adverse bursts
    - Hard session cutoffs (no fresh risk late NY, avoid low-liquidity Asia)
    - Dynamic correlation matrix updates
    - Real-time exposure monitoring
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Correlation matrix (updated dynamically)
        self.correlation_matrix = {
            'EURUSD': {'GBPUSD': 0.7, 'USDCHF': -0.8, 'USDJPY': -0.3, 'AUDUSD': 0.6, 'NZDUSD': 0.8},
            'GBPUSD': {'EURUSD': 0.7, 'USDCHF': -0.6, 'USDJPY': -0.2, 'AUDUSD': 0.5, 'EURGBP': -0.7},
            'USDCHF': {'EURUSD': -0.8, 'GBPUSD': -0.6, 'USDJPY': 0.4, 'AUDUSD': -0.5},
            'USDJPY': {'EURUSD': -0.3, 'GBPUSD': -0.2, 'USDCHF': 0.4, 'AUDUSD': -0.1},
            'AUDUSD': {'EURUSD': 0.6, 'GBPUSD': 0.5, 'USDCHF': -0.5, 'USDJPY': -0.1, 'NZDUSD': 0.9},
            'NZDUSD': {'EURUSD': 0.8, 'AUDUSD': 0.9, 'GBPUSD': 0.6},
            'XAUUSD': {'EURUSD': 0.3, 'GBPUSD': 0.2, 'USDCHF': -0.4, 'DXY': -0.8},
            'BTCUSD': {'XAUUSD': 0.2, 'SPX500': 0.6, 'NAS100': 0.8}
        }
        
        # USD leg exposure tracking
        self.usd_exposure = {
            'long_usd': 0.0,   # Total long USD exposure
            'short_usd': 0.0,  # Total short USD exposure
            'net_usd': 0.0,    # Net USD exposure
            'gross_usd': 0.0   # Gross USD exposure
        }
        
        # Risk factor exposures
        self.risk_factors = {
            'usd_strength': {'exposure': 0.0, 'limit': 0.3},
            'eur_strength': {'exposure': 0.0, 'limit': 0.25},
            'commodity_currencies': {'exposure': 0.0, 'limit': 0.2},  # AUD, NZD, CAD
            'safe_havens': {'exposure': 0.0, 'limit': 0.15},          # CHF, JPY
            'risk_on': {'exposure': 0.0, 'limit': 0.25},              # High beta currencies
            'precious_metals': {'exposure': 0.0, 'limit': 0.1},       # Gold, Silver
            'crypto': {'exposure': 0.0, 'limit': 0.05}                # Crypto exposure
        }
        
        # Exposure limits
        self.exposure_limits = {
            'max_usd_exposure': 0.4,           # 40% max USD exposure
            'max_single_currency': 0.3,        # 30% max single currency
            'max_correlated_exposure': 0.5,    # 50% max correlated pairs
            'max_total_exposure': 1.0,         # 100% max total exposure
            'max_risk_factor_exposure': 0.3    # 30% max per risk factor
        }
        
        # Session-based risk parameters
        self.session_risk_params = {
            'london_am': {'multiplier': 1.2, 'max_new_risk': 0.6, 'description': 'High activity period'},
            'london_pm': {'multiplier': 1.0, 'max_new_risk': 0.4, 'description': 'Moderate activity'},
            'ny_am': {'multiplier': 1.3, 'max_new_risk': 0.8, 'description': 'Peak activity period'},
            'ny_pm': {'multiplier': 0.8, 'max_new_risk': 0.3, 'description': 'Reduced activity'},
            'ny_late': {'multiplier': 0.3, 'max_new_risk': 0.1, 'description': 'Minimal new risk'},
            'asian': {'multiplier': 0.6, 'max_new_risk': 0.2, 'description': 'Low liquidity period'},
            'weekend_approach': {'multiplier': 0.2, 'max_new_risk': 0.05, 'description': 'Weekend approach'}
        }
        
        # Streak-based risk adjustments
        self.streak_adjustments = {
            'win_streak': {
                3: 1.1, 4: 1.15, 5: 1.2, 6: 1.25, 7: 1.3,  # Max 30% increase
                'max_multiplier': 1.3
            },
            'loss_streak': {
                2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 6: 0.5,     # Max 50% decrease
                'max_reduction': 0.5
            },
            'cooling_off': {
                'trigger_losses': 3,      # 3 consecutive losses triggers cooling off
                'duration_hours': 2,      # 2 hours cooling off
                'risk_reduction': 0.3     # 70% risk reduction during cooling off
            }
        }
        
        # Regime-based risk adjustments
        self.regime_adjustments = {
            'trending': {'multiplier': 1.2, 'description': 'Increase risk in trending markets'},
            'volatile': {'multiplier': 0.6, 'description': 'Reduce risk in volatile markets'},
            'quiet': {'multiplier': 0.8, 'description': 'Reduce risk in quiet markets'},
            'crisis': {'multiplier': 0.3, 'description': 'Minimal risk during crisis'},
            'normal': {'multiplier': 1.0, 'description': 'Normal risk levels'}
        }
        
        # Current positions and exposure tracking
        self.current_positions = {}
        self.exposure_history = deque(maxlen=500)
        self.risk_events = deque(maxlen=200)
        
        # Performance tracking
        self.total_risk_assessments = 0
        self.risk_violations_prevented = 0
        self.cooling_off_periods = 0
        self.current_cooling_off = None
        
        # Streak tracking
        self.current_streak = {'type': 'none', 'count': 0, 'last_update': datetime.now()}
        self.streak_history = deque(maxlen=100)
    
    def assess_position_risk(self, 
                            proposed_position: Dict,
                            current_positions: List[Dict],
                            market_context: Dict,
                            symbol: str) -> Dict[str, Any]:
        """
        Comprehensive position risk assessment with correlation awareness.
        
        Args:
            proposed_position: New position being considered
            current_positions: All current open positions
            market_context: Current market regime and conditions
            symbol: Trading symbol for new position
            
        Returns:
            Risk assessment with approval/rejection and sizing adjustments
        """
        try:
            self.total_risk_assessments += 1
            
            # 1. Update current exposure tracking
            self._update_exposure_tracking(current_positions)
            
            # 2. Calculate correlation impact
            correlation_impact = self._calculate_correlation_impact(
                proposed_position, current_positions, symbol
            )
            
            # 3. Check USD exposure limits
            usd_exposure_check = self._check_usd_exposure_limits(
                proposed_position, symbol
            )
            
            # 4. Check risk factor exposure
            risk_factor_check = self._check_risk_factor_exposure(
                proposed_position, symbol
            )
            
            # 5. Apply session-based risk limits
            session_risk_check = self._apply_session_risk_limits(
                proposed_position, market_context
            )
            
            # 6. Apply streak-based adjustments
            streak_adjustment = self._apply_streak_adjustments(
                proposed_position, market_context
            )
            
            # 7. Apply regime-based adjustments
            regime_adjustment = self._apply_regime_adjustments(
                proposed_position, market_context
            )
            
            # 8. Check cooling-off period
            cooling_off_check = self._check_cooling_off_period()
            
            # 9. Make final risk decision
            risk_decision = self._make_risk_decision(
                correlation_impact, usd_exposure_check, risk_factor_check,
                session_risk_check, streak_adjustment, regime_adjustment,
                cooling_off_check, proposed_position
            )
            
            # Create comprehensive response
            response = self._create_risk_response(
                True,
                proposed_position=proposed_position,
                correlation_impact=correlation_impact,
                usd_exposure_check=usd_exposure_check,
                risk_factor_check=risk_factor_check,
                session_risk_check=session_risk_check,
                streak_adjustment=streak_adjustment,
                regime_adjustment=regime_adjustment,
                cooling_off_check=cooling_off_check,
                risk_decision=risk_decision,
                symbol=symbol
            )
            
            # Update tracking
            self._update_risk_tracking(response)
            
            return response
            
        except Exception as e:
            return self._create_risk_response(False, error=f"Risk assessment failed: {str(e)}")
    
    def _update_exposure_tracking(self, current_positions: List[Dict]):
        """Update real-time exposure tracking."""
        try:
            # Reset exposures
            self.usd_exposure = {'long_usd': 0.0, 'short_usd': 0.0, 'net_usd': 0.0, 'gross_usd': 0.0}
            for factor in self.risk_factors:
                self.risk_factors[factor]['exposure'] = 0.0
            
            # Calculate exposures from current positions
            for position in current_positions:
                symbol = position.get('symbol', '')
                size = float(position.get('size', 0))
                direction = position.get('direction', 'neutral')
                
                # USD exposure calculation
                if 'USD' in symbol:
                    if symbol.endswith('USD'):  # USD is quote currency
                        if direction == 'bullish':
                            self.usd_exposure['short_usd'] += size
                        else:
                            self.usd_exposure['long_usd'] += size
                    elif symbol.startswith('USD'):  # USD is base currency
                        if direction == 'bullish':
                            self.usd_exposure['long_usd'] += size
                        else:
                            self.usd_exposure['short_usd'] += size
                
                # Risk factor exposures
                self._update_risk_factor_exposure(symbol, direction, size)
            
            # Calculate net and gross USD exposure
            self.usd_exposure['net_usd'] = self.usd_exposure['long_usd'] - self.usd_exposure['short_usd']
            self.usd_exposure['gross_usd'] = self.usd_exposure['long_usd'] + self.usd_exposure['short_usd']
            
        except Exception:
            pass  # Silent fail for exposure tracking
    
    def _update_risk_factor_exposure(self, symbol: str, direction: str, size: float):
        """Update risk factor exposures."""
        try:
            # Commodity currencies
            if any(curr in symbol for curr in ['AUD', 'NZD', 'CAD']):
                if direction == 'bullish':
                    self.risk_factors['commodity_currencies']['exposure'] += size
                else:
                    self.risk_factors['commodity_currencies']['exposure'] -= size
            
            # Safe haven currencies
            if any(curr in symbol for curr in ['CHF', 'JPY']):
                if direction == 'bullish':
                    self.risk_factors['safe_havens']['exposure'] += size
                else:
                    self.risk_factors['safe_havens']['exposure'] -= size
            
            # EUR strength
            if 'EUR' in symbol:
                if (symbol.startswith('EUR') and direction == 'bullish') or (symbol.endswith('EUR') and direction == 'bearish'):
                    self.risk_factors['eur_strength']['exposure'] += size
                else:
                    self.risk_factors['eur_strength']['exposure'] -= size
            
            # Precious metals
            if 'XAU' in symbol or 'XAG' in symbol:
                if direction == 'bullish':
                    self.risk_factors['precious_metals']['exposure'] += size
                else:
                    self.risk_factors['precious_metals']['exposure'] -= size
            
            # Crypto
            if 'BTC' in symbol or 'ETH' in symbol:
                if direction == 'bullish':
                    self.risk_factors['crypto']['exposure'] += size
                else:
                    self.risk_factors['crypto']['exposure'] -= size
                    
        except Exception:
            pass  # Silent fail for risk factor updates
    
    def _calculate_correlation_impact(self, 
                                     proposed_position: Dict,
                                     current_positions: List[Dict],
                                     symbol: str) -> Dict[str, Any]:
        """Calculate correlation impact of proposed position."""
        try:
            correlation_impact = {
                'correlated_positions': [],
                'total_correlation_exposure': 0.0,
                'max_correlation': 0.0,
                'risk_adjustment': 1.0,
                'approval': True
            }
            
            proposed_direction = proposed_position.get('direction', 'neutral')
            proposed_size = float(proposed_position.get('size', 0))
            
            # Check correlations with existing positions
            correlated_exposure = 0.0
            max_correlation = 0.0
            
            for position in current_positions:
                pos_symbol = position.get('symbol', '')
                pos_direction = position.get('direction', 'neutral')
                pos_size = float(position.get('size', 0))
                
                # Get correlation coefficient
                correlation = self._get_correlation(symbol, pos_symbol)
                
                if abs(correlation) > 0.3:  # Significant correlation
                    # Calculate correlation exposure
                    same_direction = proposed_direction == pos_direction
                    
                    if same_direction and correlation > 0:
                        # Positive correlation, same direction = increased exposure
                        corr_exposure = abs(correlation) * pos_size
                    elif not same_direction and correlation < 0:
                        # Negative correlation, opposite direction = increased exposure
                        corr_exposure = abs(correlation) * pos_size
                    else:
                        # Hedging effect
                        corr_exposure = -abs(correlation) * pos_size * 0.5
                    
                    correlated_exposure += corr_exposure
                    max_correlation = max(max_correlation, abs(correlation))
                    
                    correlation_impact['correlated_positions'].append({
                        'symbol': pos_symbol,
                        'correlation': correlation,
                        'direction': pos_direction,
                        'size': pos_size,
                        'exposure_impact': corr_exposure
                    })
            
            correlation_impact['total_correlation_exposure'] = correlated_exposure
            correlation_impact['max_correlation'] = max_correlation
            
            # Calculate risk adjustment
            if correlated_exposure > self.exposure_limits['max_correlated_exposure']:
                correlation_impact['approval'] = False
                correlation_impact['risk_adjustment'] = 0.0
            elif correlated_exposure > 0.3:
                # Reduce position size based on correlation
                reduction = min(0.7, correlated_exposure / self.exposure_limits['max_correlated_exposure'])
                correlation_impact['risk_adjustment'] = 1.0 - reduction
            
            return correlation_impact
            
        except Exception as e:
            return {'approval': True, 'risk_adjustment': 1.0, 'error': str(e)}
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation coefficient between two symbols."""
        try:
            # Direct lookup
            if symbol1 in self.correlation_matrix and symbol2 in self.correlation_matrix[symbol1]:
                return self.correlation_matrix[symbol1][symbol2]
            elif symbol2 in self.correlation_matrix and symbol1 in self.correlation_matrix[symbol2]:
                return self.correlation_matrix[symbol2][symbol1]
            
            # Calculate implied correlation for USD pairs
            if 'USD' in symbol1 and 'USD' in symbol2:
                # Both have USD component
                if symbol1.endswith('USD') and symbol2.endswith('USD'):
                    # Both are XXX/USD pairs - usually positively correlated
                    return 0.5
                elif symbol1.startswith('USD') and symbol2.startswith('USD'):
                    # Both are USD/XXX pairs - usually positively correlated
                    return 0.5
                else:
                    # One is XXX/USD, other is USD/XXX - usually negatively correlated
                    return -0.3
            
            # Default correlation for unrelated pairs
            return 0.1
            
        except Exception:
            return 0.0
    
    def _check_usd_exposure_limits(self, proposed_position: Dict, symbol: str) -> Dict[str, Any]:
        """Check USD exposure limits."""
        try:
            exposure_check = {
                'within_limits': True,
                'current_exposure': dict(self.usd_exposure),
                'projected_exposure': {},
                'risk_adjustment': 1.0,
                'warnings': []
            }
            
            if 'USD' not in symbol:
                return exposure_check  # No USD impact
            
            proposed_size = float(proposed_position.get('size', 0))
            proposed_direction = proposed_position.get('direction', 'neutral')
            
            # Calculate projected exposure
            projected_long = self.usd_exposure['long_usd']
            projected_short = self.usd_exposure['short_usd']
            
            if symbol.endswith('USD'):  # USD is quote currency
                if proposed_direction == 'bullish':
                    projected_short += proposed_size
                else:
                    projected_long += proposed_size
            elif symbol.startswith('USD'):  # USD is base currency
                if proposed_direction == 'bullish':
                    projected_long += proposed_size
                else:
                    projected_short += proposed_size
            
            projected_net = projected_long - projected_short
            projected_gross = projected_long + projected_short
            
            exposure_check['projected_exposure'] = {
                'long_usd': projected_long,
                'short_usd': projected_short,
                'net_usd': projected_net,
                'gross_usd': projected_gross
            }
            
            # Check limits
            warnings = []
            
            if projected_gross > self.exposure_limits['max_usd_exposure']:
                exposure_check['within_limits'] = False
                warnings.append(f"USD exposure limit exceeded: {projected_gross:.2f} > {self.exposure_limits['max_usd_exposure']}")
                exposure_check['risk_adjustment'] = 0.0
            elif projected_gross > self.exposure_limits['max_usd_exposure'] * 0.8:
                warnings.append(f"Approaching USD exposure limit: {projected_gross:.2f}")
                exposure_check['risk_adjustment'] = 0.7
            
            exposure_check['warnings'] = warnings
            
            return exposure_check
            
        except Exception as e:
            return {'within_limits': True, 'error': str(e)}
    
    def _check_risk_factor_exposure(self, proposed_position: Dict, symbol: str) -> Dict[str, Any]:
        """Check risk factor exposure limits."""
        try:
            factor_check = {
                'within_limits': True,
                'factor_exposures': dict(self.risk_factors),
                'risk_adjustment': 1.0,
                'violations': []
            }
            
            proposed_size = float(proposed_position.get('size', 0))
            proposed_direction = proposed_position.get('direction', 'neutral')
            
            # Calculate impact on each risk factor
            violations = []
            min_adjustment = 1.0
            
            for factor_name, factor_data in self.risk_factors.items():
                current_exposure = factor_data['exposure']
                limit = factor_data['limit']
                
                # Calculate new exposure for this factor
                factor_impact = self._calculate_factor_impact(symbol, proposed_direction, proposed_size, factor_name)
                new_exposure = current_exposure + factor_impact
                
                # Check if limit would be exceeded
                if abs(new_exposure) > limit:
                    violations.append({
                        'factor': factor_name,
                        'current_exposure': current_exposure,
                        'projected_exposure': new_exposure,
                        'limit': limit,
                        'excess': abs(new_exposure) - limit
                    })
                    
                    # Calculate required adjustment
                    if abs(current_exposure) < limit:
                        max_additional = limit - abs(current_exposure)
                        adjustment = max_additional / abs(factor_impact) if factor_impact != 0 else 0
                        min_adjustment = min(min_adjustment, adjustment)
                    else:
                        min_adjustment = 0.0  # No room for this factor
                elif abs(new_exposure) > limit * 0.8:
                    # Approaching limit
                    adjustment = 0.8  # Reduce size by 20%
                    min_adjustment = min(min_adjustment, adjustment)
            
            factor_check['violations'] = violations
            factor_check['within_limits'] = len(violations) == 0
            factor_check['risk_adjustment'] = min_adjustment
            
            return factor_check
            
        except Exception as e:
            return {'within_limits': True, 'error': str(e)}
    
    def _calculate_factor_impact(self, symbol: str, direction: str, size: float, factor_name: str) -> float:
        """Calculate impact on specific risk factor."""
        try:
            impact = 0.0
            
            if factor_name == 'commodity_currencies':
                if any(curr in symbol for curr in ['AUD', 'NZD', 'CAD']):
                    impact = size if direction == 'bullish' else -size
            
            elif factor_name == 'safe_havens':
                if any(curr in symbol for curr in ['CHF', 'JPY']):
                    impact = size if direction == 'bullish' else -size
            
            elif factor_name == 'eur_strength':
                if 'EUR' in symbol:
                    if (symbol.startswith('EUR') and direction == 'bullish') or (symbol.endswith('EUR') and direction == 'bearish'):
                        impact = size
                    else:
                        impact = -size
            
            elif factor_name == 'precious_metals':
                if 'XAU' in symbol or 'XAG' in symbol:
                    impact = size if direction == 'bullish' else -size
            
            elif factor_name == 'crypto':
                if 'BTC' in symbol or 'ETH' in symbol:
                    impact = size if direction == 'bullish' else -size
            
            return impact
            
        except Exception:
            return 0.0
    
    def _apply_session_risk_limits(self, proposed_position: Dict, market_context: Dict) -> Dict[str, Any]:
        """Apply session-based risk limits."""
        try:
            session_check = {
                'session_allowed': True,
                'session_multiplier': 1.0,
                'session': 'unknown',
                'warnings': []
            }
            
            current_time = datetime.now()
            hour = current_time.hour
            
            # Determine current session
            if 8 <= hour <= 10:
                session = 'london_am'
            elif 10 <= hour <= 14:
                session = 'london_pm'
            elif 13 <= hour <= 15:
                session = 'ny_am'
            elif 15 <= hour <= 19:
                session = 'ny_pm'
            elif 19 <= hour <= 24:
                session = 'ny_late'
            elif 0 <= hour <= 8:
                session = 'asian'
            else:
                session = 'unknown'
            
            # Check for weekend approach
            if current_time.weekday() == 4 and hour >= 20:  # Friday after 8 PM
                session = 'weekend_approach'
            
            session_check['session'] = session
            
            # Apply session parameters
            if session in self.session_risk_params:
                params = self.session_risk_params[session]
                session_check['session_multiplier'] = params['multiplier']
                
                # Check if new risk is allowed
                max_new_risk = params['max_new_risk']
                current_total_exposure = self.usd_exposure['gross_usd']
                
                if current_total_exposure + float(proposed_position.get('size', 0)) > max_new_risk:
                    if params['multiplier'] < 0.5:  # Very restrictive sessions
                        session_check['session_allowed'] = False
                        session_check['warnings'].append(f"Session {session}: No new risk allowed")
                    else:
                        # Reduce position size
                        max_additional = max(0, max_new_risk - current_total_exposure)
                        if max_additional > 0:
                            session_check['session_multiplier'] *= (max_additional / float(proposed_position.get('size', 1)))
                        else:
                            session_check['session_allowed'] = False
                        session_check['warnings'].append(f"Session {session}: Risk reduced due to limits")
            
            return session_check
            
        except Exception as e:
            return {'session_allowed': True, 'session_multiplier': 1.0, 'error': str(e)}
    
    def _apply_streak_adjustments(self, proposed_position: Dict, market_context: Dict) -> Dict[str, Any]:
        """Apply streak-based risk adjustments."""
        try:
            streak_adjustment = {
                'streak_multiplier': 1.0,
                'current_streak': dict(self.current_streak),
                'cooling_off_active': False,
                'adjustment_reason': 'none'
            }
            
            streak_type = self.current_streak['type']
            streak_count = self.current_streak['count']
            
            # Win streak adjustments
            if streak_type == 'win' and streak_count >= 3:
                multiplier = self.streak_adjustments['win_streak'].get(
                    streak_count, 
                    self.streak_adjustments['win_streak']['max_multiplier']
                )
                streak_adjustment['streak_multiplier'] = multiplier
                streak_adjustment['adjustment_reason'] = f'win_streak_{streak_count}'
            
            # Loss streak adjustments
            elif streak_type == 'loss':
                if streak_count >= self.streak_adjustments['cooling_off']['trigger_losses']:
                    # Trigger cooling-off period
                    cooling_duration = self.streak_adjustments['cooling_off']['duration_hours']
                    if (self.current_cooling_off is None or 
                        datetime.now() - self.current_cooling_off > timedelta(hours=cooling_duration)):
                        self.current_cooling_off = datetime.now()
                        self.cooling_off_periods += 1
                    
                    streak_adjustment['cooling_off_active'] = True
                    streak_adjustment['streak_multiplier'] = self.streak_adjustments['cooling_off']['risk_reduction']
                    streak_adjustment['adjustment_reason'] = f'cooling_off_loss_streak_{streak_count}'
                
                else:
                    # Regular loss streak adjustment
                    multiplier = self.streak_adjustments['loss_streak'].get(
                        streak_count,
                        self.streak_adjustments['loss_streak']['max_reduction']
                    )
                    streak_adjustment['streak_multiplier'] = multiplier
                    streak_adjustment['adjustment_reason'] = f'loss_streak_{streak_count}'
            
            return streak_adjustment
            
        except Exception as e:
            return {'streak_multiplier': 1.0, 'error': str(e)}
    
    def _apply_regime_adjustments(self, proposed_position: Dict, market_context: Dict) -> Dict[str, Any]:
        """Apply regime-based risk adjustments."""
        try:
            regime_adjustment = {
                'regime_multiplier': 1.0,
                'current_regime': 'normal',
                'adjustment_reason': 'none'
            }
            
            # Get current market regime
            regime = market_context.get('regime', 'normal')
            volatility = market_context.get('volatility', 'normal')
            
            regime_adjustment['current_regime'] = regime
            
            # Apply regime-based multiplier
            if regime in self.regime_adjustments:
                multiplier = self.regime_adjustments[regime]['multiplier']
                regime_adjustment['regime_multiplier'] = multiplier
                regime_adjustment['adjustment_reason'] = f'regime_{regime}'
            
            # Additional volatility adjustments
            if volatility == 'extreme':
                regime_adjustment['regime_multiplier'] *= 0.5
                regime_adjustment['adjustment_reason'] += '_extreme_volatility'
            elif volatility == 'very_high':
                regime_adjustment['regime_multiplier'] *= 0.7
                regime_adjustment['adjustment_reason'] += '_high_volatility'
            
            return regime_adjustment
            
        except Exception as e:
            return {'regime_multiplier': 1.0, 'error': str(e)}
    
    def _check_cooling_off_period(self) -> Dict[str, Any]:
        """Check if we're in a cooling-off period."""
        try:
            cooling_check = {
                'in_cooling_off': False,
                'remaining_time': 0,
                'triggered_by': 'none'
            }
            
            if self.current_cooling_off:
                elapsed = (datetime.now() - self.current_cooling_off).total_seconds() / 3600  # hours
                duration = self.streak_adjustments['cooling_off']['duration_hours']
                
                if elapsed < duration:
                    cooling_check['in_cooling_off'] = True
                    cooling_check['remaining_time'] = duration - elapsed
                    cooling_check['triggered_by'] = 'loss_streak'
                else:
                    # Cooling off period expired
                    self.current_cooling_off = None
            
            return cooling_check
            
        except Exception:
            return {'in_cooling_off': False}
    
    def _make_risk_decision(self, 
                           correlation_impact: Dict,
                           usd_exposure_check: Dict,
                           risk_factor_check: Dict,
                           session_risk_check: Dict,
                           streak_adjustment: Dict,
                           regime_adjustment: Dict,
                           cooling_off_check: Dict,
                           proposed_position: Dict) -> Dict[str, Any]:
        """Make final risk decision considering all factors."""
        try:
            decision = {
                'approved': True,
                'final_size_multiplier': 1.0,
                'rejection_reasons': [],
                'warnings': [],
                'risk_level': 'normal'
            }
            
            # Collect all multipliers
            multipliers = []
            rejection_reasons = []
            warnings = []
            
            # 1. Correlation impact
            if not correlation_impact.get('approval', True):
                rejection_reasons.append('correlation_exposure_exceeded')
            else:
                multipliers.append(correlation_impact.get('risk_adjustment', 1.0))
            
            # 2. USD exposure
            if not usd_exposure_check.get('within_limits', True):
                rejection_reasons.append('usd_exposure_limit_exceeded')
            else:
                multipliers.append(usd_exposure_check.get('risk_adjustment', 1.0))
                warnings.extend(usd_exposure_check.get('warnings', []))
            
            # 3. Risk factor exposure
            if not risk_factor_check.get('within_limits', True):
                rejection_reasons.append('risk_factor_exposure_exceeded')
            else:
                multipliers.append(risk_factor_check.get('risk_adjustment', 1.0))
            
            # 4. Session limits
            if not session_risk_check.get('session_allowed', True):
                rejection_reasons.append('session_risk_limits')
            else:
                multipliers.append(session_risk_check.get('session_multiplier', 1.0))
                warnings.extend(session_risk_check.get('warnings', []))
            
            # 5. Streak adjustments
            multipliers.append(streak_adjustment.get('streak_multiplier', 1.0))
            if streak_adjustment.get('cooling_off_active', False):
                warnings.append('Position sizing reduced due to cooling-off period')
            
            # 6. Regime adjustments
            multipliers.append(regime_adjustment.get('regime_multiplier', 1.0))
            
            # 7. Cooling-off impact
            if cooling_off_check.get('in_cooling_off', False):
                multipliers.append(self.streak_adjustments['cooling_off']['risk_reduction'])
                warnings.append(f"Cooling-off period active: {cooling_off_check['remaining_time']:.1f} hours remaining")
            
            # Make final decision
            if rejection_reasons:
                decision['approved'] = False
                decision['rejection_reasons'] = rejection_reasons
                decision['final_size_multiplier'] = 0.0
                self.risk_violations_prevented += 1
            else:
                # Calculate final multiplier (take minimum to be conservative)
                final_multiplier = min(multipliers) if multipliers else 1.0
                
                # Ensure minimum viable position size
                if final_multiplier < 0.01:  # Less than 1% of original size
                    decision['approved'] = False
                    decision['rejection_reasons'] = ['position_too_small_after_adjustments']
                    decision['final_size_multiplier'] = 0.0
                else:
                    decision['final_size_multiplier'] = final_multiplier
                    
                    # Determine risk level
                    if final_multiplier < 0.3:
                        decision['risk_level'] = 'very_high'
                    elif final_multiplier < 0.6:
                        decision['risk_level'] = 'high'
                    elif final_multiplier < 0.8:
                        decision['risk_level'] = 'elevated'
                    else:
                        decision['risk_level'] = 'normal'
            
            decision['warnings'] = warnings
            decision['applied_multipliers'] = multipliers
            
            return decision
            
        except Exception as e:
            return {'approved': False, 'error': str(e)}
    
    def _create_risk_response(self, 
                             valid: bool,
                             proposed_position: Dict = None,
                             correlation_impact: Dict = None,
                             usd_exposure_check: Dict = None,
                             risk_factor_check: Dict = None,
                             session_risk_check: Dict = None,
                             streak_adjustment: Dict = None,
                             regime_adjustment: Dict = None,
                             cooling_off_check: Dict = None,
                             risk_decision: Dict = None,
                             symbol: str = "",
                             error: str = "") -> Dict[str, Any]:
        """Create comprehensive risk assessment response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "proposed_position": proposed_position or {},
            "correlation_impact": correlation_impact or {},
            "usd_exposure_check": usd_exposure_check or {},
            "risk_factor_check": risk_factor_check or {},
            "session_risk_check": session_risk_check or {},
            "streak_adjustment": streak_adjustment or {},
            "regime_adjustment": regime_adjustment or {},
            "cooling_off_check": cooling_off_check or {},
            "risk_decision": risk_decision or {},
            "current_exposures": {
                "usd_exposure": dict(self.usd_exposure),
                "risk_factors": {k: v['exposure'] for k, v in self.risk_factors.items()}
            },
            "summary": {
                "position_approved": risk_decision.get('approved', False) if risk_decision else False,
                "final_size_multiplier": risk_decision.get('final_size_multiplier', 0.0) if risk_decision else 0.0,
                "risk_level": risk_decision.get('risk_level', 'unknown') if risk_decision else 'unknown',
                "total_warnings": len(risk_decision.get('warnings', [])) if risk_decision else 0,
                "in_cooling_off": cooling_off_check.get('in_cooling_off', False) if cooling_off_check else False
            },
            "metadata": {
                "total_assessments": self.total_risk_assessments,
                "violations_prevented": self.risk_violations_prevented,
                "cooling_off_periods": self.cooling_off_periods,
                "engine_version": "1.0.0",
                "analysis_type": "correlation_aware_risk"
            }
        }
    
    def _update_risk_tracking(self, response: Dict):
        """Update risk tracking and performance metrics."""
        try:
            # Store risk assessment
            self.risk_events.append({
                'timestamp': datetime.now(),
                'response': response,
                'approved': response.get('summary', {}).get('position_approved', False)
            })
            
            # Update exposure history
            self.exposure_history.append({
                'timestamp': datetime.now(),
                'usd_exposure': dict(self.usd_exposure),
                'risk_factors': {k: v['exposure'] for k, v in self.risk_factors.items()}
            })
            
        except Exception:
            pass  # Silent fail for tracking updates
    
    def update_streak(self, outcome: str, pnl: float = 0.0):
        """Update streak tracking."""
        try:
            current_type = self.current_streak['type']
            current_count = self.current_streak['count']
            
            if outcome == 'win':
                if current_type == 'win':
                    self.current_streak['count'] += 1
                else:
                    self.current_streak = {'type': 'win', 'count': 1, 'last_update': datetime.now()}
            elif outcome == 'loss':
                if current_type == 'loss':
                    self.current_streak['count'] += 1
                else:
                    self.current_streak = {'type': 'loss', 'count': 1, 'last_update': datetime.now()}
            else:
                # Neutral outcome resets streak
                self.current_streak = {'type': 'none', 'count': 0, 'last_update': datetime.now()}
            
            # Store in history
            self.streak_history.append({
                'timestamp': datetime.now(),
                'outcome': outcome,
                'pnl': pnl,
                'streak_after': dict(self.current_streak)
            })
            
        except Exception:
            pass  # Silent fail for streak updates
    
    def update_correlation_matrix(self, symbol_data: Dict[str, List[float]]):
        """Update correlation matrix with recent price data."""
        try:
            if len(symbol_data) < 2:
                return
            
            # Calculate correlations between all symbol pairs
            symbols = list(symbol_data.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i != j and len(symbol_data[symbol1]) >= 20 and len(symbol_data[symbol2]) >= 20:
                        # Take recent 50 data points for correlation
                        data1 = symbol_data[symbol1][-50:]
                        data2 = symbol_data[symbol2][-50:]
                        
                        min_length = min(len(data1), len(data2))
                        if min_length >= 20:
                            correlation = np.corrcoef(data1[:min_length], data2[:min_length])[0, 1]
                            
                            if not np.isnan(correlation):
                                # Update correlation matrix
                                if symbol1 not in self.correlation_matrix:
                                    self.correlation_matrix[symbol1] = {}
                                self.correlation_matrix[symbol1][symbol2] = correlation
                                
        except Exception:
            pass  # Silent fail for correlation updates
    
    def get_risk_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive risk manager statistics."""
        return {
            "total_assessments": self.total_risk_assessments,
            "violations_prevented": self.risk_violations_prevented,
            "prevention_rate": self.risk_violations_prevented / max(1, self.total_risk_assessments),
            "cooling_off_periods": self.cooling_off_periods,
            "current_streak": dict(self.current_streak),
            "current_exposures": {
                "usd_exposure": dict(self.usd_exposure),
                "risk_factor_exposures": {k: v['exposure'] for k, v in self.risk_factors.items()}
            },
            "exposure_limits": dict(self.exposure_limits),
            "correlation_matrix_size": sum(len(correlations) for correlations in self.correlation_matrix.values()),
            "session_risk_params": dict(self.session_risk_params),
            "streak_adjustments": dict(self.streak_adjustments),
            "regime_adjustments": dict(self.regime_adjustments),
            "memory_sizes": {
                "exposure_history": len(self.exposure_history),
                "risk_events": len(self.risk_events),
                "streak_history": len(self.streak_history)
            },
            "engine_version": "1.0.0"
        }
<<<<<<< Current (Your changes)
=======
        }
>>>>>>> 4323fc9 (upgraded)
=======
>>>>>>> Incoming (Background Agent changes)
