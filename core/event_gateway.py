# core/event_gateway.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class EventGateway:
    """
    EVENT GATEWAY - Calendar Ingestion and Event Filtering
    
    Handles event gating with news, central bank announcements, and data surprises.
    Includes volatility regime adaptation and no-trade state management.
    
    Features:
    - Economic calendar integration
    - News event proximity detection
    - Central bank announcement filtering
    - Data surprise impact assessment
    - Volatility regime adaptation
    - Holiday and low-liquidity detection
    - Event impact scoring
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Event types and their impact levels
        self.event_types = {
            'central_bank': {
                'impact': 'high',
                'pre_event_buffer': 60,    # 60 minutes before
                'post_event_buffer': 30,   # 30 minutes after
                'volatility_multiplier': 2.5
            },
            'nfp': {
                'impact': 'very_high',
                'pre_event_buffer': 30,
                'post_event_buffer': 60,
                'volatility_multiplier': 3.0
            },
            'cpi': {
                'impact': 'high',
                'pre_event_buffer': 15,
                'post_event_buffer': 45,
                'volatility_multiplier': 2.0
            },
            'gdp': {
                'impact': 'medium',
                'pre_event_buffer': 15,
                'post_event_buffer': 30,
                'volatility_multiplier': 1.5
            },
            'manufacturing_pmi': {
                'impact': 'medium',
                'pre_event_buffer': 10,
                'post_event_buffer': 20,
                'volatility_multiplier': 1.3
            },
            'retail_sales': {
                'impact': 'medium',
                'pre_event_buffer': 10,
                'post_event_buffer': 20,
                'volatility_multiplier': 1.4
            },
            'unemployment': {
                'impact': 'high',
                'pre_event_buffer': 15,
                'post_event_buffer': 30,
                'volatility_multiplier': 1.8
            },
            'interest_rate': {
                'impact': 'very_high',
                'pre_event_buffer': 120,   # 2 hours before
                'post_event_buffer': 120, # 2 hours after
                'volatility_multiplier': 4.0
            }
        }
        
        # Currency-specific event impacts
        self.currency_events = {
            'USD': ['nfp', 'cpi', 'fomc', 'fed_speech', 'unemployment', 'retail_sales'],
            'EUR': ['ecb_rate', 'cpi_eu', 'manufacturing_pmi_eu', 'draghi_speech'],
            'GBP': ['boe_rate', 'cpi_uk', 'manufacturing_pmi_uk', 'brexit_news'],
            'JPY': ['boj_rate', 'tankan_survey', 'core_cpi_jp'],
            'AUD': ['rba_rate', 'employment_au', 'retail_sales_au'],
            'CAD': ['boc_rate', 'employment_ca', 'cpi_ca'],
            'CHF': ['snb_rate', 'cpi_ch'],
            'NZD': ['rbnz_rate', 'employment_nz']
        }
        
        # Holiday calendars
        self.holidays = {
            'us': ['new_year', 'mlk_day', 'presidents_day', 'memorial_day', 'independence_day', 
                   'labor_day', 'columbus_day', 'veterans_day', 'thanksgiving', 'christmas'],
            'uk': ['new_year', 'good_friday', 'easter_monday', 'may_day', 'spring_bank', 
                   'summer_bank', 'christmas', 'boxing_day'],
            'eu': ['new_year', 'good_friday', 'easter_monday', 'may_day', 'christmas'],
            'jp': ['new_year', 'coming_age', 'national_foundation', 'golden_week', 'respect_aged', 'culture_day']
        }
        
        # Low liquidity periods
        self.low_liquidity_periods = {
            'christmas_week': {'start': '12-25', 'end': '01-02', 'impact': 'very_high'},
            'summer_holidays': {'start': '07-15', 'end': '08-15', 'impact': 'medium'},
            'chinese_new_year': {'start': '02-10', 'end': '02-17', 'impact': 'high'},
            'week_between_christmas': {'start': '12-25', 'end': '01-01', 'impact': 'extreme'}
        }
        
        # Volatility regimes
        self.volatility_regimes = {
            'extreme': {'multiplier': 0.2, 'description': 'Avoid all trading'},
            'very_high': {'multiplier': 0.4, 'description': 'Severely reduced position sizing'},
            'high': {'multiplier': 0.6, 'description': 'Reduced position sizing'},
            'elevated': {'multiplier': 0.8, 'description': 'Slightly reduced sizing'},
            'normal': {'multiplier': 1.0, 'description': 'Normal operations'},
            'low': {'multiplier': 1.2, 'description': 'Increased sizing opportunity'}
        }
        
        # Event memory and tracking
        self.event_memory = deque(maxlen=1000)
        self.impact_tracking = defaultdict(lambda: {'total': 0, 'accurate': 0})
        self.volatility_regime_history = deque(maxlen=100)
        
        # Mock event calendar (in production, this would be live data)
        self.mock_calendar = self._initialize_mock_calendar()
        
        # Performance tracking
        self.total_event_assessments = 0
        self.correct_impact_predictions = 0
    
    def _initialize_mock_calendar(self) -> List[Dict]:
        """Initialize mock economic calendar for testing."""
        # This would be replaced with real calendar data integration
        return [
            {
                'event': 'nfp',
                'currency': 'USD',
                'impact': 'very_high',
                'time': '13:30',
                'description': 'Non-Farm Payrolls',
                'frequency': 'monthly'
            },
            {
                'event': 'cpi',
                'currency': 'USD',
                'impact': 'high',
                'time': '13:30',
                'description': 'Consumer Price Index',
                'frequency': 'monthly'
            },
            {
                'event': 'fomc',
                'currency': 'USD',
                'impact': 'very_high',
                'time': '19:00',
                'description': 'FOMC Rate Decision',
                'frequency': 'quarterly'
            }
        ]
    
    def assess_event_environment(self, 
                                symbol: str, 
                                current_time: datetime = None) -> Dict[str, Any]:
        """
        Comprehensive event environment assessment.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            current_time: Current time for assessment
            
        Returns:
            Complete event environment analysis
        """
        try:
            current_time = current_time or datetime.now()
            self.total_event_assessments += 1
            
            # Extract currencies from symbol
            currencies = self._extract_currencies_from_symbol(symbol)
            
            # 1. Check for upcoming events
            upcoming_events = self._check_upcoming_events(currencies, current_time)
            
            # 2. Check current event proximity
            event_proximity = self._check_event_proximity(currencies, current_time)
            
            # 3. Assess holiday and low liquidity periods
            liquidity_assessment = self._assess_liquidity_conditions(current_time)
            
            # 4. Calculate volatility regime
            volatility_regime = self._calculate_volatility_regime(upcoming_events, event_proximity)
            
            # 5. Determine trading restrictions
            trading_restrictions = self._determine_trading_restrictions(
                event_proximity, liquidity_assessment, volatility_regime
            )
            
            # 6. Calculate event impact scores
            impact_scores = self._calculate_event_impact_scores(upcoming_events, currencies)
            
            # 7. Generate trading recommendations
            recommendations = self._generate_trading_recommendations(
                trading_restrictions, impact_scores, volatility_regime
            )
            
            # Create comprehensive response
            response = self._create_event_response(
                True,
                upcoming_events=upcoming_events,
                event_proximity=event_proximity,
                liquidity_assessment=liquidity_assessment,
                volatility_regime=volatility_regime,
                trading_restrictions=trading_restrictions,
                impact_scores=impact_scores,
                recommendations=recommendations,
                symbol=symbol,
                currencies=currencies
            )
            
            # Update tracking
            self._update_event_tracking(response)
            
            return response
            
        except Exception as e:
            return self._create_event_response(False, error=f"Event assessment failed: {str(e)}")
    
    def _extract_currencies_from_symbol(self, symbol: str) -> List[str]:
        """Extract currency codes from trading symbol."""
        try:
            # Handle different symbol formats
            if len(symbol) >= 6:
                base = symbol[:3].upper()
                quote = symbol[3:6].upper()
                return [base, quote]
            else:
                # Handle other formats like XAUUSD, BTCUSD
                if 'USD' in symbol:
                    return ['USD']
                return []
        except Exception:
            return []
    
    def _check_upcoming_events(self, currencies: List[str], current_time: datetime) -> List[Dict]:
        """Check for upcoming high-impact events."""
        try:
            upcoming_events = []
            
            # Look ahead 24 hours
            check_until = current_time + timedelta(hours=24)
            
            for currency in currencies:
                currency_events = self.currency_events.get(currency, [])
                
                for event_type in currency_events:
                    if event_type in self.event_types:
                        event_info = self.event_types[event_type]
                        
                        # Mock event scheduling (in production, use real calendar)
                        # For demo, assume some events are scheduled
                        if self._is_event_scheduled(event_type, current_time):
                            event_time = self._get_next_event_time(event_type, current_time)
                            
                            if event_time <= check_until:
                                upcoming_events.append({
                                    'event_type': event_type,
                                    'currency': currency,
                                    'event_time': event_time,
                                    'impact_level': event_info['impact'],
                                    'pre_buffer': event_info['pre_event_buffer'],
                                    'post_buffer': event_info['post_event_buffer'],
                                    'volatility_multiplier': event_info['volatility_multiplier'],
                                    'time_until_event': (event_time - current_time).total_seconds() / 60  # minutes
                                })
            
            return sorted(upcoming_events, key=lambda x: x['time_until_event'])
            
        except Exception as e:
            return []
    
    def _is_event_scheduled(self, event_type: str, current_time: datetime) -> bool:
        """Check if event is scheduled (mock implementation)."""
        # Mock scheduling logic - in production, this would check real calendar
        hour = current_time.hour
        day = current_time.weekday()
        
        # NFP is first Friday of month at 13:30
        if event_type == 'nfp':
            return day == 4 and hour >= 13 and current_time.day <= 7
        
        # CPI is around mid-month
        elif event_type == 'cpi':
            return 10 <= current_time.day <= 20 and hour >= 13
        
        # FOMC roughly every 6-8 weeks
        elif event_type == 'fomc':
            return current_time.day in [15, 16, 17] and hour >= 19
        
        # Default: random chance for demo
        return np.random.random() < 0.1  # 10% chance for demo purposes
    
    def _get_next_event_time(self, event_type: str, current_time: datetime) -> datetime:
        """Get next scheduled time for event (mock implementation)."""
        # Mock implementation - in production, use real calendar
        if event_type == 'nfp':
            # Next Friday at 13:30
            days_ahead = 4 - current_time.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return current_time.replace(hour=13, minute=30, second=0) + timedelta(days=days_ahead)
        
        elif event_type == 'cpi':
            # Next month 15th at 13:30
            next_month = current_time.replace(day=15, hour=13, minute=30, second=0)
            if next_month <= current_time:
                if next_month.month == 12:
                    next_month = next_month.replace(year=next_month.year + 1, month=1)
                else:
                    next_month = next_month.replace(month=next_month.month + 1)
            return next_month
        
        else:
            # Default: within next 24 hours
            return current_time + timedelta(hours=np.random.randint(1, 24))
    
    def _check_event_proximity(self, currencies: List[str], current_time: datetime) -> Dict[str, Any]:
        """Check proximity to current or recent events."""
        try:
            proximity_info = {
                'in_event_window': False,
                'active_events': [],
                'recent_events': [],
                'proximity_score': 0.0
            }
            
            # Check for events in the last 4 hours and next 2 hours
            check_window_start = current_time - timedelta(hours=4)
            check_window_end = current_time + timedelta(hours=2)
            
            for currency in currencies:
                currency_events = self.currency_events.get(currency, [])
                
                for event_type in currency_events:
                    if event_type in self.event_types:
                        event_info = self.event_types[event_type]
                        
                        # Mock recent event check
                        if self._was_event_recent(event_type, current_time):
                            event_time = current_time - timedelta(minutes=np.random.randint(5, 240))
                            
                            time_since_event = (current_time - event_time).total_seconds() / 60
                            
                            if time_since_event <= event_info['post_event_buffer']:
                                proximity_info['in_event_window'] = True
                                proximity_info['active_events'].append({
                                    'event_type': event_type,
                                    'currency': currency,
                                    'event_time': event_time,
                                    'time_since': time_since_event,
                                    'impact': event_info['impact'],
                                    'still_active': True
                                })
                            else:
                                proximity_info['recent_events'].append({
                                    'event_type': event_type,
                                    'currency': currency,
                                    'event_time': event_time,
                                    'time_since': time_since_event,
                                    'impact': event_info['impact']
                                })
            
            # Calculate proximity score
            proximity_info['proximity_score'] = self._calculate_proximity_score(
                proximity_info['active_events'], proximity_info['recent_events']
            )
            
            return proximity_info
            
        except Exception as e:
            return {'in_event_window': False, 'error': str(e)}
    
    def _was_event_recent(self, event_type: str, current_time: datetime) -> bool:
        """Check if event was recent (mock implementation)."""
        # Mock implementation for demo
        return np.random.random() < 0.2  # 20% chance for demo
    
    def _calculate_proximity_score(self, active_events: List[Dict], recent_events: List[Dict]) -> float:
        """Calculate event proximity impact score."""
        try:
            score = 0.0
            
            # Active events have higher impact
            for event in active_events:
                impact_multiplier = {
                    'very_high': 1.0,
                    'high': 0.8,
                    'medium': 0.6,
                    'low': 0.4
                }.get(event['impact'], 0.5)
                
                # Closer to event = higher score
                time_factor = max(0.1, 1.0 - event['time_since'] / 120)  # Decay over 2 hours
                score += impact_multiplier * time_factor
            
            # Recent events have lower impact
            for event in recent_events:
                impact_multiplier = {
                    'very_high': 0.4,
                    'high': 0.3,
                    'medium': 0.2,
                    'low': 0.1
                }.get(event['impact'], 0.1)
                
                time_factor = max(0.05, 1.0 - event['time_since'] / 240)  # Decay over 4 hours
                score += impact_multiplier * time_factor
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _assess_liquidity_conditions(self, current_time: datetime) -> Dict[str, Any]:
        """Assess holiday and low liquidity conditions."""
        try:
            liquidity_assessment = {
                'is_holiday': False,
                'is_low_liquidity': False,
                'holiday_info': {},
                'liquidity_score': 1.0,
                'trading_recommended': True
            }
            
            # Check for holidays
            holiday_check = self._check_holidays(current_time)
            if holiday_check['is_holiday']:
                liquidity_assessment.update(holiday_check)
                liquidity_assessment['liquidity_score'] *= 0.3
                liquidity_assessment['trading_recommended'] = False
            
            # Check for low liquidity periods
            low_liquidity_check = self._check_low_liquidity_periods(current_time)
            if low_liquidity_check['is_low_liquidity']:
                liquidity_assessment.update(low_liquidity_check)
                impact_multiplier = {
                    'extreme': 0.1,
                    'very_high': 0.2,
                    'high': 0.4,
                    'medium': 0.6
                }.get(low_liquidity_check['impact'], 0.8)
                
                liquidity_assessment['liquidity_score'] *= impact_multiplier
                if impact_multiplier <= 0.2:
                    liquidity_assessment['trading_recommended'] = False
            
            # Check session overlap for liquidity
            session_liquidity = self._assess_session_liquidity(current_time)
            liquidity_assessment['session_info'] = session_liquidity
            liquidity_assessment['liquidity_score'] *= session_liquidity['multiplier']
            
            return liquidity_assessment
            
        except Exception as e:
            return {'is_holiday': False, 'is_low_liquidity': False, 'error': str(e)}
    
    def _check_holidays(self, current_time: datetime) -> Dict[str, Any]:
        """Check for major holidays."""
        try:
            month_day = f"{current_time.month:02d}-{current_time.day:02d}"
            
            # Major holidays that affect multiple markets
            major_holidays = [
                ('01-01', 'New Year'),
                ('12-25', 'Christmas'),
                ('12-24', 'Christmas Eve'),
                ('07-04', 'US Independence Day'),
                ('11-11', 'Veterans Day'),
                ('12-31', 'New Year Eve')
            ]
            
            for holiday_date, holiday_name in major_holidays:
                if month_day == holiday_date:
                    return {
                        'is_holiday': True,
                        'holiday_name': holiday_name,
                        'impact': 'very_high',
                        'affected_markets': ['US', 'EU', 'UK', 'AU']
                    }
            
            # Thanksgiving (4th Thursday of November)
            if (current_time.month == 11 and 
                current_time.weekday() == 3 and  # Thursday
                22 <= current_time.day <= 28):
                return {
                    'is_holiday': True,
                    'holiday_name': 'Thanksgiving',
                    'impact': 'high',
                    'affected_markets': ['US']
                }
            
            return {'is_holiday': False}
            
        except Exception:
            return {'is_holiday': False}
    
    def _check_low_liquidity_periods(self, current_time: datetime) -> Dict[str, Any]:
        """Check for known low liquidity periods."""
        try:
            month_day = f"{current_time.month:02d}-{current_time.day:02d}"
            
            # Christmas/New Year period
            if '12-20' <= month_day <= '12-31' or '01-01' <= month_day <= '01-05':
                return {
                    'is_low_liquidity': True,
                    'period': 'christmas_new_year',
                    'impact': 'extreme',
                    'description': 'Christmas and New Year period - extremely low liquidity'
                }
            
            # Summer holidays (July-August)
            if current_time.month in [7, 8]:
                return {
                    'is_low_liquidity': True,
                    'period': 'summer_holidays',
                    'impact': 'medium',
                    'description': 'European summer holidays - reduced liquidity'
                }
            
            # Chinese New Year period (varies by year, approximate)
            if current_time.month == 2 and 10 <= current_time.day <= 20:
                return {
                    'is_low_liquidity': True,
                    'period': 'chinese_new_year',
                    'impact': 'high',
                    'description': 'Chinese New Year - reduced Asian liquidity'
                }
            
            return {'is_low_liquidity': False}
            
        except Exception:
            return {'is_low_liquidity': False}
    
    def _assess_session_liquidity(self, current_time: datetime) -> Dict[str, Any]:
        """Assess liquidity based on trading session overlaps."""
        try:
            hour = current_time.hour
            
            # Define session overlaps and their liquidity multipliers
            if 13 <= hour <= 16:  # London/NY overlap
                return {
                    'session': 'london_ny_overlap',
                    'liquidity': 'very_high',
                    'multiplier': 1.5,
                    'description': 'Peak liquidity - London/NY overlap'
                }
            elif 8 <= hour <= 12:  # London session
                return {
                    'session': 'london',
                    'liquidity': 'high',
                    'multiplier': 1.2,
                    'description': 'High liquidity - London session'
                }
            elif 17 <= hour <= 21:  # NY session
                return {
                    'session': 'new_york',
                    'liquidity': 'high',
                    'multiplier': 1.1,
                    'description': 'Good liquidity - NY session'
                }
            elif 0 <= hour <= 7:  # Asian session
                return {
                    'session': 'asian',
                    'liquidity': 'medium',
                    'multiplier': 0.8,
                    'description': 'Moderate liquidity - Asian session'
                }
            else:  # Off hours
                return {
                    'session': 'off_hours',
                    'liquidity': 'low',
                    'multiplier': 0.6,
                    'description': 'Low liquidity - off market hours'
                }
                
        except Exception:
            return {'session': 'unknown', 'liquidity': 'normal', 'multiplier': 1.0}
    
    def _calculate_volatility_regime(self, upcoming_events: List[Dict], event_proximity: Dict) -> Dict[str, Any]:
        """Calculate current volatility regime."""
        try:
            base_volatility = 'normal'
            volatility_factors = []
            
            # Factor in upcoming high-impact events
            for event in upcoming_events:
                if event['time_until_event'] <= 60:  # Within 1 hour
                    if event['impact_level'] == 'very_high':
                        volatility_factors.append(3.0)
                    elif event['impact_level'] == 'high':
                        volatility_factors.append(2.0)
                elif event['time_until_event'] <= 240:  # Within 4 hours
                    if event['impact_level'] == 'very_high':
                        volatility_factors.append(1.5)
                    elif event['impact_level'] == 'high':
                        volatility_factors.append(1.2)
            
            # Factor in event proximity
            proximity_score = event_proximity.get('proximity_score', 0)
            if proximity_score > 0.7:
                volatility_factors.append(2.0)
            elif proximity_score > 0.4:
                volatility_factors.append(1.5)
            
            # Calculate regime
            if volatility_factors:
                max_factor = max(volatility_factors)
                avg_factor = np.mean(volatility_factors)
                
                if max_factor >= 3.0 or avg_factor >= 2.0:
                    regime = 'extreme'
                elif max_factor >= 2.0 or avg_factor >= 1.5:
                    regime = 'very_high'
                elif max_factor >= 1.5 or avg_factor >= 1.2:
                    regime = 'high'
                elif max_factor >= 1.2:
                    regime = 'elevated'
                else:
                    regime = 'normal'
            else:
                regime = 'normal'
            
            regime_info = self.volatility_regimes[regime]
            
            return {
                'regime': regime,
                'multiplier': regime_info['multiplier'],
                'description': regime_info['description'],
                'factors': volatility_factors,
                'max_factor': max(volatility_factors) if volatility_factors else 1.0,
                'avg_factor': np.mean(volatility_factors) if volatility_factors else 1.0
            }
            
        except Exception as e:
            return {'regime': 'normal', 'multiplier': 1.0, 'error': str(e)}
    
    def _determine_trading_restrictions(self, 
                                       event_proximity: Dict,
                                       liquidity_assessment: Dict,
                                       volatility_regime: Dict) -> Dict[str, Any]:
        """Determine trading restrictions based on event environment."""
        try:
            restrictions = {
                'trading_allowed': True,
                'position_size_multiplier': 1.0,
                'restrictions': [],
                'severity': 'none'
            }
            
            # Event proximity restrictions
            if event_proximity.get('in_event_window', False):
                restrictions['restrictions'].append('in_event_window')
                restrictions['position_size_multiplier'] *= 0.5
            
            # Holiday restrictions
            if liquidity_assessment.get('is_holiday', False):
                restrictions['trading_allowed'] = False
                restrictions['restrictions'].append('holiday')
                restrictions['severity'] = 'critical'
            
            # Low liquidity restrictions
            if liquidity_assessment.get('is_low_liquidity', False):
                impact = liquidity_assessment.get('impact', 'medium')
                if impact in ['extreme', 'very_high']:
                    restrictions['trading_allowed'] = False
                    restrictions['restrictions'].append('extremely_low_liquidity')
                    restrictions['severity'] = 'critical'
                else:
                    restrictions['position_size_multiplier'] *= 0.3
                    restrictions['restrictions'].append('low_liquidity')
            
            # Volatility regime restrictions
            volatility_multiplier = volatility_regime.get('multiplier', 1.0)
            if volatility_multiplier <= 0.4:
                restrictions['trading_allowed'] = False
                restrictions['restrictions'].append('extreme_volatility')
                restrictions['severity'] = 'critical'
            else:
                restrictions['position_size_multiplier'] *= volatility_multiplier
            
            # Liquidity score impact
            liquidity_score = liquidity_assessment.get('liquidity_score', 1.0)
            restrictions['position_size_multiplier'] *= liquidity_score
            
            # Final position size check
            if restrictions['position_size_multiplier'] < 0.1:
                restrictions['trading_allowed'] = False
                restrictions['restrictions'].append('position_size_too_small')
                restrictions['severity'] = 'high'
            
            # Set severity if not already critical
            if restrictions['severity'] == 'none':
                if len(restrictions['restrictions']) >= 3:
                    restrictions['severity'] = 'high'
                elif len(restrictions['restrictions']) >= 2:
                    restrictions['severity'] = 'medium'
                elif len(restrictions['restrictions']) >= 1:
                    restrictions['severity'] = 'low'
            
            return restrictions
            
        except Exception as e:
            return {'trading_allowed': True, 'position_size_multiplier': 1.0, 'error': str(e)}
    
    def _calculate_event_impact_scores(self, upcoming_events: List[Dict], currencies: List[str]) -> Dict[str, Any]:
        """Calculate event impact scores for currencies."""
        try:
            impact_scores = {}
            
            for currency in currencies:
                currency_impact = 0.0
                relevant_events = []
                
                for event in upcoming_events:
                    if event['currency'] == currency:
                        # Weight by time until event and impact level
                        time_weight = max(0.1, 1.0 - event['time_until_event'] / 1440)  # Decay over 24 hours
                        
                        impact_weight = {
                            'very_high': 1.0,
                            'high': 0.8,
                            'medium': 0.6,
                            'low': 0.4
                        }.get(event['impact_level'], 0.5)
                        
                        event_impact = time_weight * impact_weight
                        currency_impact += event_impact
                        
                        relevant_events.append({
                            'event_type': event['event_type'],
                            'impact_score': event_impact,
                            'time_until': event['time_until_event']
                        })
                
                impact_scores[currency] = {
                    'total_impact': min(1.0, currency_impact),
                    'relevant_events': relevant_events,
                    'event_count': len(relevant_events)
                }
            
            return impact_scores
            
        except Exception as e:
            return {}
    
    def _generate_trading_recommendations(self, 
                                        trading_restrictions: Dict,
                                        impact_scores: Dict,
                                        volatility_regime: Dict) -> Dict[str, Any]:
        """Generate trading recommendations based on event environment."""
        try:
            recommendations = {
                'overall': 'proceed_with_caution',
                'position_sizing': 'normal',
                'specific_recommendations': [],
                'risk_level': 'medium'
            }
            
            # Check if trading is allowed
            if not trading_restrictions.get('trading_allowed', True):
                recommendations['overall'] = 'avoid_trading'
                recommendations['position_sizing'] = 'none'
                recommendations['risk_level'] = 'extreme'
                recommendations['specific_recommendations'] = [
                    'Avoid all new positions',
                    'Consider closing existing positions',
                    'Wait for event/holiday to pass'
                ]
                return recommendations
            
            # Position sizing recommendations
            multiplier = trading_restrictions.get('position_size_multiplier', 1.0)
            if multiplier < 0.3:
                recommendations['position_sizing'] = 'very_small'
                recommendations['risk_level'] = 'high'
            elif multiplier < 0.6:
                recommendations['position_sizing'] = 'small'
                recommendations['risk_level'] = 'medium_high'
            elif multiplier < 0.8:
                recommendations['position_sizing'] = 'reduced'
                recommendations['risk_level'] = 'medium'
            else:
                recommendations['position_sizing'] = 'normal'
                recommendations['risk_level'] = 'low'
            
            # Volatility-based recommendations
            regime = volatility_regime.get('regime', 'normal')
            if regime in ['extreme', 'very_high']:
                recommendations['specific_recommendations'].extend([
                    'Use wider stops due to increased volatility',
                    'Consider shorter time horizons',
                    'Monitor positions more closely'
                ])
            elif regime == 'high':
                recommendations['specific_recommendations'].extend([
                    'Slightly wider stops recommended',
                    'Be prepared for quick movements'
                ])
            
            # Event-specific recommendations
            restrictions = trading_restrictions.get('restrictions', [])
            if 'in_event_window' in restrictions:
                recommendations['specific_recommendations'].append('Currently in event impact window - exercise caution')
            if 'low_liquidity' in restrictions:
                recommendations['specific_recommendations'].append('Low liquidity detected - expect wider spreads')
            
            return recommendations
            
        except Exception as e:
            return {'overall': 'proceed_with_caution', 'error': str(e)}
    
    def _create_event_response(self, 
                              valid: bool,
                              upcoming_events: List[Dict] = None,
                              event_proximity: Dict = None,
                              liquidity_assessment: Dict = None,
                              volatility_regime: Dict = None,
                              trading_restrictions: Dict = None,
                              impact_scores: Dict = None,
                              recommendations: Dict = None,
                              symbol: str = "",
                              currencies: List[str] = None,
                              error: str = "") -> Dict[str, Any]:
        """Create comprehensive event environment response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "symbol": symbol,
            "currencies": currencies or [],
            "timestamp": datetime.now().isoformat(),
            "upcoming_events": upcoming_events or [],
            "event_proximity": event_proximity or {},
            "liquidity_assessment": liquidity_assessment or {},
            "volatility_regime": volatility_regime or {},
            "trading_restrictions": trading_restrictions or {},
            "impact_scores": impact_scores or {},
            "recommendations": recommendations or {},
            "summary": {
                "trading_allowed": trading_restrictions.get('trading_allowed', True) if trading_restrictions else True,
                "position_size_multiplier": trading_restrictions.get('position_size_multiplier', 1.0) if trading_restrictions else 1.0,
                "risk_level": recommendations.get('risk_level', 'medium') if recommendations else 'medium',
                "upcoming_events_count": len(upcoming_events or []),
                "in_event_window": event_proximity.get('in_event_window', False) if event_proximity else False
            },
            "metadata": {
                "total_assessments": self.total_event_assessments,
                "engine_version": "1.0.0",
                "analysis_type": "event_gateway"
            }
        }
    
    def _update_event_tracking(self, response: Dict):
        """Update event tracking and performance metrics."""
        try:
            # Store event assessment for future analysis
            self.event_memory.append({
                'timestamp': datetime.now(),
                'response': response,
                'symbol': response.get('symbol'),
                'volatility_regime': response.get('volatility_regime', {}).get('regime', 'normal')
            })
            
            # Track volatility regime history
            regime = response.get('volatility_regime', {}).get('regime', 'normal')
            self.volatility_regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime
            })
            
        except Exception:
            pass  # Silent fail for tracking updates
    
    def update_impact_accuracy(self, event_type: str, predicted_impact: str, actual_impact: str):
        """Update impact prediction accuracy."""
        try:
            self.impact_tracking[event_type]['total'] += 1
            if predicted_impact == actual_impact:
                self.impact_tracking[event_type]['accurate'] += 1
                self.correct_impact_predictions += 1
                
        except Exception:
            pass  # Silent fail for accuracy updates
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get comprehensive gateway statistics."""
        return {
            "total_assessments": self.total_event_assessments,
            "correct_impact_predictions": self.correct_impact_predictions,
            "overall_accuracy": self.correct_impact_predictions / max(1, self.total_event_assessments),
            "impact_tracking_accuracy": {
                event: tracking['accurate'] / max(1, tracking['total'])
                for event, tracking in self.impact_tracking.items()
            },
            "volatility_regime_distribution": {
                regime: sum(1 for entry in self.volatility_regime_history if entry['regime'] == regime)
                for regime in self.volatility_regimes.keys()
            },
            "event_types_configured": list(self.event_types.keys()),
            "currencies_supported": list(self.currency_events.keys()),
            "memory_sizes": {
                "event_memory": len(self.event_memory),
                "volatility_regime_history": len(self.volatility_regime_history)
            },
            "engine_version": "1.0.0"
<<<<<<< HEAD
        }
=======
        }
>>>>>>> 4323fc9 (upgraded)
