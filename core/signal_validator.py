# core/signal_validator.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    HTF_OVERRIDE = "htf_override"           # Higher timeframe wins
    CONFLUENCE_WEIGHTED = "confluence_weighted"  # Weight by confluence
    RISK_ADJUSTED = "risk_adjusted"         # Adjust risk based on conflict
    REJECT_SIGNAL = "reject_signal"         # Reject conflicting signals

class SignalValidator:
    """
    SIGNAL VALIDATOR - Top-Down Bias Enforcement and Conflict Resolution
    
    Implements comprehensive signal validation with:
    - Top-down bias enforcement across timeframes
    - Explicit conflict handlers for HTF vs LTF signals
    - Premium/discount boundary respect
    - Unified confluence scoring (IPDA + OrderFlow + Fourier + ML)
    - Multi-timeframe alignment checking
    - Signal strength and consistency validation
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Timeframe hierarchy (higher index = higher timeframe)
        self.timeframe_hierarchy = {
            'M1': 0, 'M5': 1, 'M15': 2, 'M30': 3,
            'H1': 4, 'H4': 5, 'D1': 6, 'W1': 7, 'MN1': 8
        }
        
        # Timeframe weights for conflict resolution
        self.timeframe_weights = {
            'MN1': 1.0, 'W1': 0.9, 'D1': 0.8,
            'H4': 0.7, 'H1': 0.6, 'M30': 0.4,
            'M15': 0.3, 'M5': 0.2, 'M1': 0.1
        }
        
        # Confluence component weights (calibrated)
        self.confluence_weights = {
            'ipda_phase': 0.25,        # IPDA accumulation/distribution
            'orderflow': 0.25,         # Order flow analysis
            'fourier': 0.20,           # Fourier wave analysis
            'ml_prediction': 0.15,     # ML confidence
            'structure': 0.10,         # Market structure
            'pd_array': 0.05           # PD array alignment
        }
        
        # Validation thresholds
        self.validation_thresholds = {
            'min_confluence_score': 0.6,
            'htf_bias_override_threshold': 0.7,
            'pd_boundary_respect_threshold': 0.8,
            'signal_consistency_threshold': 0.65,
            'multi_tf_alignment_threshold': 0.5
        }
        
        # Conflict resolution preferences
        self.conflict_resolution_strategy = ConflictResolution.CONFLUENCE_WEIGHTED
        
        # Performance tracking
        self.validation_history = deque(maxlen=1000)
        self.conflict_resolution_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        self.htf_bias_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Signal memory
        self.validated_signals = deque(maxlen=500)
        self.rejected_signals = deque(maxlen=200)
        self.conflict_cases = deque(maxlen=200)
        
        # Statistics
        self.total_validations = 0
        self.signals_passed = 0
        self.signals_rejected = 0
        self.conflicts_resolved = 0
    
    def validate_signal(self, 
                       signal_data: Dict,
                       multi_timeframe_data: Dict,
                       pd_analysis: Dict,
                       market_context: Dict,
                       symbol: str = "",
                       primary_timeframe: str = "M15") -> Dict[str, Any]:
        """
        Comprehensive signal validation with conflict resolution.
        
        Args:
            signal_data: Signal from various sources (ML, rules, etc.)
            multi_timeframe_data: Data from multiple timeframes
            pd_analysis: Premium/discount analysis
            market_context: Current market conditions
            symbol: Trading symbol
            primary_timeframe: Primary trading timeframe
            
        Returns:
            Validation result with conflicts resolved
        """
        try:
            self.total_validations += 1
            
            # 1. Extract signal components
            signal_components = self._extract_signal_components(signal_data)
            
            # 2. Check top-down bias alignment
            htf_bias_check = self._check_htf_bias_alignment(
                signal_components, multi_timeframe_data, primary_timeframe
            )
            
            # 3. Validate premium/discount boundary respect
            pd_boundary_check = self._validate_pd_boundary_respect(
                signal_components, pd_analysis
            )
            
            # 4. Calculate unified confluence score
            confluence_score = self._calculate_unified_confluence_score(
                signal_data, market_context
            )
            
            # 5. Check multi-timeframe consistency
            mtf_consistency = self._check_multi_timeframe_consistency(
                signal_components, multi_timeframe_data
            )
            
            # 6. Detect and resolve conflicts
            conflict_analysis = self._detect_and_resolve_conflicts(
                htf_bias_check, pd_boundary_check, confluence_score, 
                mtf_consistency, signal_components
            )
            
            # 7. Make final validation decision
            validation_decision = self._make_validation_decision(
                htf_bias_check, pd_boundary_check, confluence_score,
                mtf_consistency, conflict_analysis
            )
            
            # 8. Create comprehensive response
            response = self._create_validation_response(
                True,
                signal_components=signal_components,
                htf_bias_check=htf_bias_check,
                pd_boundary_check=pd_boundary_check,
                confluence_score=confluence_score,
                mtf_consistency=mtf_consistency,
                conflict_analysis=conflict_analysis,
                validation_decision=validation_decision,
                symbol=symbol,
                primary_timeframe=primary_timeframe
            )
            
            # Update tracking
            self._update_validation_tracking(response)
            
            return response
            
        except Exception as e:
            return self._create_validation_response(False, error=f"Signal validation failed: {str(e)}")
    
    def _extract_signal_components(self, signal_data: Dict) -> Dict[str, Any]:
        """Extract and normalize signal components from various sources."""
        try:
            components = {
                'ml_signals': {},
                'rule_signals': {},
                'orderflow_signals': {},
                'structure_signals': {},
                'fourier_signals': {},
                'primary_direction': 'neutral',
                'primary_strength': 0.0
            }
            
            # Extract ML signals
            ml_data = signal_data.get('ml_signals', {})
            components['ml_signals'] = {
                'confidence': float(ml_data.get('confidence', 0.0)),
                'probability': float(ml_data.get('probability', 0.0)),
                'direction': ml_data.get('direction', 'neutral'),
                'uncertainty': float(ml_data.get('uncertainty', 0.5))
            }
            
            # Extract rule-based signals
            rule_data = signal_data.get('rule_signals', {})
            components['rule_signals'] = {
                'cisd_score': float(rule_data.get('cisd_score', 0.0)),
                'structure_score': float(rule_data.get('structure_score', 0.0)),
                'fourier_score': float(rule_data.get('fourier_score', 0.0)),
                'regime_score': float(rule_data.get('regime_score', 0.0))
            }
            
            # Extract order flow signals
            of_data = signal_data.get('order_flow', {})
            components['orderflow_signals'] = {
                'delta_momentum': float(of_data.get('delta_momentum', 0.0)),
                'absorption_strength': float(of_data.get('absorption_strength', 0.0)),
                'institutional_pressure': float(of_data.get('institutional_pressure', 0.0)),
                'dominant_side': of_data.get('dominant_side', 'neutral')
            }
            
            # Extract structure signals
            struct_data = signal_data.get('structure', {})
            components['structure_signals'] = {
                'choch': bool(struct_data.get('choch', False)),
                'bos': bool(struct_data.get('bos', False)),
                'false_breakout': bool(struct_data.get('false_breakout', False)),
                'stop_run': bool(struct_data.get('stop_run', False))
            }
            
            # Extract Fourier signals
            fourier_data = signal_data.get('fourier', {})
            components['fourier_signals'] = {
                'wave_phase': fourier_data.get('current_phase', 'unknown'),
                'absorption_type': fourier_data.get('absorption_type', 'none'),
                'confidence': float(fourier_data.get('confidence', 0.0))
            }
            
            # Determine primary direction and strength
            components['primary_direction'] = self._determine_primary_direction(components)
            components['primary_strength'] = self._calculate_primary_strength(components)
            
            return components
            
        except Exception as e:
            return {'error': str(e)}
    
    def _determine_primary_direction(self, components: Dict) -> str:
        """Determine primary signal direction from all components."""
        try:
            direction_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            # ML vote
            ml_conf = components['ml_signals']['confidence']
            ml_dir = components['ml_signals']['direction']
            if ml_conf > 0.5:
                direction_votes[ml_dir] += ml_conf
            
            # Order flow vote
            delta = components['orderflow_signals']['delta_momentum']
            if abs(delta) > 0.1:
                direction_votes['bullish' if delta > 0 else 'bearish'] += abs(delta)
            
            # Structure vote
            if components['structure_signals']['choch']:
                direction_votes['bullish'] += 0.3
            elif components['structure_signals']['bos']:
                direction_votes['bearish'] += 0.3
            
            # CISD vote
            cisd_score = components['rule_signals']['cisd_score']
            if cisd_score > 0.5:
                direction_votes['bullish'] += cisd_score * 0.5
            elif cisd_score < -0.5:
                direction_votes['bearish'] += abs(cisd_score) * 0.5
            
            # Return direction with most votes
            return max(direction_votes, key=direction_votes.get)
            
        except Exception:
            return 'neutral'
    
    def _calculate_primary_strength(self, components: Dict) -> float:
        """Calculate primary signal strength."""
        try:
            strengths = []
            
            # ML strength
            ml_conf = components['ml_signals']['confidence']
            if ml_conf > 0:
                strengths.append(ml_conf)
            
            # Order flow strength
            delta = abs(components['orderflow_signals']['delta_momentum'])
            absorption = components['orderflow_signals']['absorption_strength']
            strengths.append((delta + absorption) / 2)
            
            # Structure strength
            struct_strength = 0
            if components['structure_signals']['choch']:
                struct_strength += 0.8
            if components['structure_signals']['bos']:
                struct_strength += 0.6
            strengths.append(min(1.0, struct_strength))
            
            # Rule strength
            rule_strength = max(
                components['rule_signals']['cisd_score'],
                components['rule_signals']['structure_score'],
                components['rule_signals']['fourier_score']
            )
            strengths.append(abs(rule_strength))
            
            return np.mean(strengths) if strengths else 0.0
            
        except Exception:
            return 0.0
    
    def _check_htf_bias_alignment(self, 
                                 signal_components: Dict,
                                 multi_timeframe_data: Dict,
                                 primary_timeframe: str) -> Dict[str, Any]:
        """Check higher timeframe bias alignment."""
        try:
            htf_check = {
                'aligned': True,
                'conflicts': [],
                'htf_bias': 'neutral',
                'htf_strength': 0.0,
                'override_required': False,
                'confidence': 0.0
            }
            
            primary_direction = signal_components['primary_direction']
            primary_tf_rank = self.timeframe_hierarchy.get(primary_timeframe, 2)
            
            conflicts = []
            htf_biases = []
            
            # Check each higher timeframe
            for tf, tf_data in multi_timeframe_data.items():
                tf_rank = self.timeframe_hierarchy.get(tf, 0)
                
                if tf_rank > primary_tf_rank:  # Higher timeframe
                    tf_trend = tf_data.get('trend_direction', 0)
                    tf_strength = tf_data.get('trend_strength', 0.0)
                    
                    # Convert trend direction to bias
                    if tf_trend > 0.3:
                        tf_bias = 'bullish'
                    elif tf_trend < -0.3:
                        tf_bias = 'bearish'
                    else:
                        tf_bias = 'neutral'
                    
                    htf_biases.append({
                        'timeframe': tf,
                        'bias': tf_bias,
                        'strength': tf_strength,
                        'weight': self.timeframe_weights.get(tf, 0.5)
                    })
                    
                    # Check for conflicts
                    if (tf_bias in ['bullish', 'bearish'] and 
                        primary_direction in ['bullish', 'bearish'] and
                        tf_bias != primary_direction):
                        
                        conflicts.append({
                            'timeframe': tf,
                            'htf_bias': tf_bias,
                            'ltf_signal': primary_direction,
                            'htf_strength': tf_strength,
                            'severity': 'high' if tf_strength > 0.7 else 'medium'
                        })
            
            # Determine dominant HTF bias
            if htf_biases:
                # Weight by timeframe importance
                weighted_bullish = sum(b['weight'] * b['strength'] for b in htf_biases if b['bias'] == 'bullish')
                weighted_bearish = sum(b['weight'] * b['strength'] for b in htf_biases if b['bias'] == 'bearish')
                
                if weighted_bullish > weighted_bearish * 1.2:
                    htf_check['htf_bias'] = 'bullish'
                    htf_check['htf_strength'] = weighted_bullish
                elif weighted_bearish > weighted_bullish * 1.2:
                    htf_check['htf_bias'] = 'bearish'
                    htf_check['htf_strength'] = weighted_bearish
                else:
                    htf_check['htf_bias'] = 'neutral'
                    htf_check['htf_strength'] = max(weighted_bullish, weighted_bearish)
            
            # Set conflict information
            htf_check['conflicts'] = conflicts
            htf_check['aligned'] = len(conflicts) == 0
            
            # Determine if override is required
            if conflicts:
                max_htf_strength = max(c['htf_strength'] for c in conflicts)
                if max_htf_strength > self.validation_thresholds['htf_bias_override_threshold']:
                    htf_check['override_required'] = True
            
            # Calculate confidence
            if htf_check['aligned']:
                htf_check['confidence'] = 0.9
            elif not htf_check['override_required']:
                htf_check['confidence'] = 0.6
            else:
                htf_check['confidence'] = 0.3
            
            return htf_check
            
        except Exception as e:
            return {'aligned': False, 'error': str(e)}
    
    def _validate_pd_boundary_respect(self, 
                                     signal_components: Dict,
                                     pd_analysis: Dict) -> Dict[str, Any]:
        """Validate that signals respect premium/discount boundaries."""
        try:
            pd_check = {
                'respects_boundaries': True,
                'violations': [],
                'pd_status': 'unknown',
                'signal_appropriateness': 'appropriate',
                'confidence': 0.0
            }
            
            # Extract PD information
            pd_info = pd_analysis.get('premium_discount', {})
            if not pd_info:
                return {'respects_boundaries': True, 'confidence': 0.5, 'note': 'No PD data available'}
            
            pd_status = pd_info.get('status', 'unknown')
            pd_strength = pd_info.get('strength', 0.0)
            primary_direction = signal_components['primary_direction']
            
            pd_check['pd_status'] = pd_status
            
            # Check for PD boundary violations
            violations = []
            
            # Premium zone - should favor sell signals
            if pd_status == 'premium' and pd_strength > 0.7:
                if primary_direction == 'bullish':
                    violations.append({
                        'type': 'premium_buy_signal',
                        'severity': 'high' if pd_strength > 0.8 else 'medium',
                        'description': 'Buy signal in strong premium zone'
                    })
            
            # Discount zone - should favor buy signals
            elif pd_status == 'discount' and pd_strength > 0.7:
                if primary_direction == 'bearish':
                    violations.append({
                        'type': 'discount_sell_signal',
                        'severity': 'high' if pd_strength > 0.8 else 'medium',
                        'description': 'Sell signal in strong discount zone'
                    })
            
            # Equilibrium - both directions acceptable
            elif pd_status == 'equilibrium':
                pd_check['signal_appropriateness'] = 'appropriate'
            
            pd_check['violations'] = violations
            pd_check['respects_boundaries'] = len(violations) == 0
            
            # Determine signal appropriateness
            if violations:
                max_severity = max(v['severity'] for v in violations)
                if max_severity == 'high':
                    pd_check['signal_appropriateness'] = 'inappropriate'
                else:
                    pd_check['signal_appropriateness'] = 'questionable'
            
            # Calculate confidence
            if pd_check['respects_boundaries']:
                pd_check['confidence'] = 0.9
            elif pd_check['signal_appropriateness'] == 'questionable':
                pd_check['confidence'] = 0.6
            else:
                pd_check['confidence'] = 0.3
            
            return pd_check
            
        except Exception as e:
            return {'respects_boundaries': False, 'error': str(e)}
    
    def _calculate_unified_confluence_score(self, 
                                           signal_data: Dict,
                                           market_context: Dict) -> Dict[str, Any]:
        """Calculate unified confluence score from all components."""
        try:
            confluence = {
                'total_score': 0.0,
                'component_scores': {},
                'weighted_score': 0.0,
                'confidence': 0.0,
                'breakdown': {}
            }
            
            component_scores = {}
            
            # 1. IPDA Phase Score
            ipda_data = market_context.get('ipda_phase', {})
            ipda_score = 0.0
            if ipda_data.get('phase') in ['accumulation', 'distribution']:
                ipda_score = ipda_data.get('confidence', 0.0)
            elif ipda_data.get('phase') in ['markup', 'markdown']:
                ipda_score = ipda_data.get('confidence', 0.0) * 0.8  # Slightly lower weight
            component_scores['ipda_phase'] = ipda_score
            
            # 2. Order Flow Score
            of_data = signal_data.get('order_flow', {})
            of_score = (
                abs(of_data.get('delta_momentum', 0.0)) * 0.4 +
                of_data.get('absorption_strength', 0.0) * 0.3 +
                of_data.get('institutional_pressure', 0.0) * 0.3
            )
            component_scores['orderflow'] = min(1.0, of_score)
            
            # 3. Fourier Wave Score
            fourier_data = signal_data.get('fourier', {})
            fourier_score = fourier_data.get('confidence', 0.0)
            if fourier_data.get('absorption_type') in ['full', 'strong']:
                fourier_score += 0.2  # Bonus for strong absorption
            component_scores['fourier'] = min(1.0, fourier_score)
            
            # 4. ML Prediction Score
            ml_data = signal_data.get('ml_signals', {})
            ml_score = ml_data.get('confidence', 0.0)
            # Reduce score by uncertainty
            uncertainty = ml_data.get('uncertainty', 0.5)
            ml_score *= (1.0 - uncertainty)
            component_scores['ml_prediction'] = ml_score
            
            # 5. Structure Score
            struct_data = signal_data.get('structure', {})
            struct_score = 0.0
            if struct_data.get('choch'):
                struct_score += 0.8
            if struct_data.get('bos'):
                struct_score += 0.6
            if struct_data.get('false_breakout'):
                struct_score -= 0.4  # Negative for false signals
            component_scores['structure'] = max(0.0, min(1.0, struct_score))
            
            # 6. PD Array Score
            rule_data = signal_data.get('rule_signals', {})
            pd_array_score = max(
                rule_data.get('cisd_score', 0.0),
                rule_data.get('structure_score', 0.0)
            )
            component_scores['pd_array'] = abs(pd_array_score)
            
            confluence['component_scores'] = component_scores
            
            # Calculate weighted score
            weighted_score = 0.0
            total_weight = 0.0
            
            for component, score in component_scores.items():
                weight = self.confluence_weights.get(component, 0.1)
                weighted_score += score * weight
                total_weight += weight
                
                confluence['breakdown'][component] = {
                    'score': score,
                    'weight': weight,
                    'weighted_contribution': score * weight
                }
            
            confluence['weighted_score'] = weighted_score / total_weight if total_weight > 0 else 0.0
            confluence['total_score'] = sum(component_scores.values()) / len(component_scores)
            
            # Calculate confidence based on score distribution
            score_variance = np.var(list(component_scores.values()))
            confluence['confidence'] = max(0.3, 1.0 - score_variance)  # Lower variance = higher confidence
            
            return confluence
            
        except Exception as e:
            return {'total_score': 0.0, 'weighted_score': 0.0, 'error': str(e)}
    
    def _check_multi_timeframe_consistency(self, 
                                          signal_components: Dict,
                                          multi_timeframe_data: Dict) -> Dict[str, Any]:
        """Check consistency across multiple timeframes."""
        try:
            consistency = {
                'is_consistent': True,
                'alignment_score': 0.0,
                'timeframe_analysis': {},
                'conflicts': [],
                'confidence': 0.0
            }
            
            primary_direction = signal_components['primary_direction']
            timeframe_directions = {}
            
            # Analyze each timeframe
            for tf, tf_data in multi_timeframe_data.items():
                trend_direction = tf_data.get('trend_direction', 0)
                trend_strength = tf_data.get('trend_strength', 0.0)
                
                # Convert to direction
                if trend_direction > 0.3:
                    tf_direction = 'bullish'
                elif trend_direction < -0.3:
                    tf_direction = 'bearish'
                else:
                    tf_direction = 'neutral'
                
                timeframe_directions[tf] = {
                    'direction': tf_direction,
                    'strength': trend_strength,
                    'weight': self.timeframe_weights.get(tf, 0.5)
                }
                
                consistency['timeframe_analysis'][tf] = {
                    'direction': tf_direction,
                    'strength': trend_strength,
                    'aligned_with_signal': tf_direction == primary_direction or tf_direction == 'neutral'
                }
            
            # Calculate alignment score
            aligned_weight = 0.0
            total_weight = 0.0
            conflicts = []
            
            for tf, tf_info in timeframe_directions.items():
                weight = tf_info['weight']
                total_weight += weight
                
                if (tf_info['direction'] == primary_direction or 
                    tf_info['direction'] == 'neutral' or
                    primary_direction == 'neutral'):
                    aligned_weight += weight
                else:
                    conflicts.append({
                        'timeframe': tf,
                        'tf_direction': tf_info['direction'],
                        'signal_direction': primary_direction,
                        'strength': tf_info['strength']
                    })
            
            consistency['alignment_score'] = aligned_weight / total_weight if total_weight > 0 else 0.0
            consistency['conflicts'] = conflicts
            consistency['is_consistent'] = len(conflicts) == 0
            
            # Calculate confidence
            if consistency['is_consistent']:
                consistency['confidence'] = min(1.0, consistency['alignment_score'] + 0.2)
            else:
                consistency['confidence'] = max(0.1, consistency['alignment_score'] - 0.2)
            
            return consistency
            
        except Exception as e:
            return {'is_consistent': False, 'alignment_score': 0.0, 'error': str(e)}
    
    def _detect_and_resolve_conflicts(self, 
                                     htf_bias_check: Dict,
                                     pd_boundary_check: Dict,
                                     confluence_score: Dict,
                                     mtf_consistency: Dict,
                                     signal_components: Dict) -> Dict[str, Any]:
        """Detect and resolve signal conflicts."""
        try:
            conflict_analysis = {
                'conflicts_detected': [],
                'resolution_strategy': self.conflict_resolution_strategy.value,
                'resolved_signal': None,
                'confidence_adjustment': 1.0,
                'risk_adjustment': 1.0
            }
            
            conflicts = []
            
            # 1. HTF vs LTF conflict
            if not htf_bias_check.get('aligned', True):
                conflicts.append({
                    'type': 'htf_ltf_conflict',
                    'severity': 'high' if htf_bias_check.get('override_required', False) else 'medium',
                    'details': htf_bias_check.get('conflicts', [])
                })
            
            # 2. PD boundary violation
            if not pd_boundary_check.get('respects_boundaries', True):
                conflicts.append({
                    'type': 'pd_boundary_violation',
                    'severity': 'high' if pd_boundary_check.get('signal_appropriateness') == 'inappropriate' else 'medium',
                    'details': pd_boundary_check.get('violations', [])
                })
            
            # 3. Multi-timeframe inconsistency
            if not mtf_consistency.get('is_consistent', True):
                conflicts.append({
                    'type': 'mtf_inconsistency',
                    'severity': 'medium',
                    'details': mtf_consistency.get('conflicts', [])
                })
            
            # 4. Low confluence score
            if confluence_score.get('weighted_score', 0.0) < self.validation_thresholds['min_confluence_score']:
                conflicts.append({
                    'type': 'low_confluence',
                    'severity': 'medium',
                    'details': {'score': confluence_score.get('weighted_score', 0.0)}
                })
            
            conflict_analysis['conflicts_detected'] = conflicts
            
            # Resolve conflicts based on strategy
            if conflicts:
                # Note: conflicts_resolved is tracked in the parent method
                resolved_signal = self._resolve_conflicts(
                    conflicts, signal_components, htf_bias_check, 
                    pd_boundary_check, confluence_score
                )
                conflict_analysis['resolved_signal'] = resolved_signal
                
                # Adjust confidence and risk based on conflicts
                conflict_severity_weights = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
                total_severity = sum(conflict_severity_weights.get(c['severity'], 0.1) for c in conflicts)
                
                conflict_analysis['confidence_adjustment'] = max(0.3, 1.0 - total_severity)
                conflict_analysis['risk_adjustment'] = max(0.5, 1.0 - total_severity * 0.5)
            else:
                conflict_analysis['resolved_signal'] = signal_components
            
            return conflict_analysis
            
        except Exception as e:
            return {'conflicts_detected': [], 'error': str(e)}
    
    def _resolve_conflicts(self, 
                          conflicts: List[Dict],
                          signal_components: Dict,
                          htf_bias_check: Dict,
                          pd_boundary_check: Dict,
                          confluence_score: Dict) -> Dict[str, Any]:
        """Resolve conflicts using the configured strategy."""
        try:
            strategy = self.conflict_resolution_strategy
            resolved_signal = signal_components.copy()
            
            if strategy == ConflictResolution.HTF_OVERRIDE:
                # Higher timeframe bias overrides
                if htf_bias_check.get('override_required', False):
                    htf_bias = htf_bias_check.get('htf_bias', 'neutral')
                    if htf_bias != 'neutral':
                        resolved_signal['primary_direction'] = htf_bias
                        resolved_signal['primary_strength'] *= 0.8  # Reduce strength due to conflict
            
            elif strategy == ConflictResolution.CONFLUENCE_WEIGHTED:
                # Weight signal by confluence score
                confluence_weight = confluence_score.get('weighted_score', 0.5)
                resolved_signal['primary_strength'] *= confluence_weight
                
                # If confluence is very low, switch to neutral
                if confluence_weight < 0.3:
                    resolved_signal['primary_direction'] = 'neutral'
            
            elif strategy == ConflictResolution.RISK_ADJUSTED:
                # Keep signal but adjust risk dramatically
                conflict_count = len(conflicts)
                risk_reduction = min(0.9, conflict_count * 0.3)
                resolved_signal['risk_adjustment'] = 1.0 - risk_reduction
            
            elif strategy == ConflictResolution.REJECT_SIGNAL:
                # Reject signal entirely
                resolved_signal['primary_direction'] = 'neutral'
                resolved_signal['primary_strength'] = 0.0
                resolved_signal['rejected'] = True
            
            resolved_signal['conflict_resolution_applied'] = strategy.value
            return resolved_signal
            
        except Exception as e:
            return signal_components
    
    def _make_validation_decision(self, 
                                 htf_bias_check: Dict,
                                 pd_boundary_check: Dict,
                                 confluence_score: Dict,
                                 mtf_consistency: Dict,
                                 conflict_analysis: Dict) -> Dict[str, Any]:
        """Make final validation decision."""
        try:
            decision = {
                'signal_approved': False,
                'rejection_reasons': [],
                'approval_confidence': 0.0,
                'final_signal': None,
                'risk_multiplier': 1.0
            }
            
            rejection_reasons = []
            confidence_factors = []
            
            # Check each validation component
            
            # 1. HTF bias alignment
            if htf_bias_check.get('aligned', True):
                confidence_factors.append(htf_bias_check.get('confidence', 0.5))
            else:
                if htf_bias_check.get('override_required', False):
                    rejection_reasons.append('Strong HTF bias conflict')
                else:
                    confidence_factors.append(0.4)  # Weak approval
            
            # 2. PD boundary respect
            if pd_boundary_check.get('respects_boundaries', True):
                confidence_factors.append(pd_boundary_check.get('confidence', 0.5))
            else:
                if pd_boundary_check.get('signal_appropriateness') == 'inappropriate':
                    rejection_reasons.append('PD boundary violation')
                else:
                    confidence_factors.append(0.3)  # Very weak approval
            
            # 3. Confluence score
            confluence_weighted = confluence_score.get('weighted_score', 0.0)
            if confluence_weighted >= self.validation_thresholds['min_confluence_score']:
                confidence_factors.append(confluence_weighted)
            else:
                rejection_reasons.append('Insufficient confluence score')
            
            # 4. Multi-timeframe consistency
            if mtf_consistency.get('is_consistent', True):
                confidence_factors.append(mtf_consistency.get('confidence', 0.5))
            else:
                if mtf_consistency.get('alignment_score', 0.0) < self.validation_thresholds['multi_tf_alignment_threshold']:
                    rejection_reasons.append('Multi-timeframe inconsistency')
                else:
                    confidence_factors.append(0.4)
            
            # Make decision
            if not rejection_reasons:
                decision['signal_approved'] = True
                decision['approval_confidence'] = np.mean(confidence_factors) if confidence_factors else 0.5
                decision['final_signal'] = conflict_analysis.get('resolved_signal')
                decision['risk_multiplier'] = conflict_analysis.get('risk_adjustment', 1.0)
                self.signals_passed += 1
            else:
                decision['signal_approved'] = False
                decision['rejection_reasons'] = rejection_reasons
                decision['approval_confidence'] = 0.0
                self.signals_rejected += 1
            
            # Apply conflict adjustments if approved
            if decision['signal_approved']:
                decision['approval_confidence'] *= conflict_analysis.get('confidence_adjustment', 1.0)
            
            return decision
            
        except Exception as e:
            return {'signal_approved': False, 'error': str(e)}
    
    def _create_validation_response(self, 
                                   valid: bool,
                                   signal_components: Dict = None,
                                   htf_bias_check: Dict = None,
                                   pd_boundary_check: Dict = None,
                                   confluence_score: Dict = None,
                                   mtf_consistency: Dict = None,
                                   conflict_analysis: Dict = None,
                                   validation_decision: Dict = None,
                                   symbol: str = "",
                                   primary_timeframe: str = "",
                                   error: str = "") -> Dict[str, Any]:
        """Create comprehensive validation response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "symbol": symbol,
            "primary_timeframe": primary_timeframe,
            "timestamp": datetime.now().isoformat(),
            "signal_components": signal_components or {},
            "htf_bias_check": htf_bias_check or {},
            "pd_boundary_check": pd_boundary_check or {},
            "confluence_score": confluence_score or {},
            "mtf_consistency": mtf_consistency or {},
            "conflict_analysis": conflict_analysis or {},
            "validation_decision": validation_decision or {},
            "summary": {
                "signal_approved": validation_decision.get('signal_approved', False) if validation_decision else False,
                "approval_confidence": validation_decision.get('approval_confidence', 0.0) if validation_decision else 0.0,
                "conflicts_count": len(conflict_analysis.get('conflicts_detected', [])) if conflict_analysis else 0,
                "htf_aligned": htf_bias_check.get('aligned', True) if htf_bias_check else True,
                "pd_compliant": pd_boundary_check.get('respects_boundaries', True) if pd_boundary_check else True,
                "confluence_score": confluence_score.get('weighted_score', 0.0) if confluence_score else 0.0
            },
            "metadata": {
                "total_validations": self.total_validations,
                "signals_passed": self.signals_passed,
                "signals_rejected": self.signals_rejected,
                "conflicts_resolved": self.conflicts_resolved,
                "pass_rate": self.signals_passed / max(1, self.total_validations),
                "engine_version": "1.0.0",
                "analysis_type": "signal_validation"
            }
        }
    
    def _update_validation_tracking(self, response: Dict):
        """Update validation tracking and performance metrics."""
        try:
            # Store validation result
            self.validation_history.append({
                'timestamp': datetime.now(),
                'response': response,
                'approved': response.get('summary', {}).get('signal_approved', False)
            })
            
            # Track by signal type
            if response.get('validation_decision', {}).get('signal_approved', False):
                self.validated_signals.append(response)
            else:
                self.rejected_signals.append(response)
            
            # Track conflicts
            conflicts = response.get('conflict_analysis', {}).get('conflicts_detected', [])
            if conflicts:
                self.conflict_cases.append({
                    'timestamp': datetime.now(),
                    'conflicts': conflicts,
                    'resolution': response.get('conflict_analysis', {}).get('resolution_strategy')
                })
                
        except Exception:
            pass  # Silent fail for tracking updates
    
    def update_htf_bias_accuracy(self, htf_bias: str, actual_outcome: str, symbol: str = "default"):
        """Update HTF bias prediction accuracy."""
        try:
            self.htf_bias_accuracy[symbol]['total'] += 1
            if htf_bias == actual_outcome:
                self.htf_bias_accuracy[symbol]['correct'] += 1
                
        except Exception:
            pass  # Silent fail for accuracy updates
    
    def calibrate_confluence_weights(self, performance_data: List[Dict]):
        """Calibrate confluence component weights based on performance."""
        try:
            if not performance_data or len(performance_data) < 20:
                return
            
            # Simple calibration based on component performance
            component_performance = defaultdict(list)
            
            for trade_data in performance_data:
                outcome = trade_data.get('successful', False)
                confluence_breakdown = trade_data.get('confluence_breakdown', {})
                
                for component, data in confluence_breakdown.items():
                    score = data.get('score', 0.0)
                    component_performance[component].append((score, outcome))
            
            # Adjust weights based on predictive power
            for component in self.confluence_weights:
                if component in component_performance:
                    data = component_performance[component]
                    # Simple correlation with outcomes
                    scores = [d[0] for d in data]
                    outcomes = [1.0 if d[1] else 0.0 for d in data]
                    
                    if len(scores) > 5:
                        correlation = np.corrcoef(scores, outcomes)[0, 1]
                        if not np.isnan(correlation):
                            # Adjust weight based on correlation
                            current_weight = self.confluence_weights[component]
                            adjustment = correlation * 0.1  # Small adjustments
                            self.confluence_weights[component] = max(0.05, min(0.4, current_weight + adjustment))
            
            # Renormalize weights
            total_weight = sum(self.confluence_weights.values())
            for component in self.confluence_weights:
                self.confluence_weights[component] /= total_weight
                
        except Exception:
            pass  # Silent fail for calibration
    
    def get_validator_stats(self) -> Dict[str, Any]:
        """Get comprehensive validator statistics."""
        return {
            "total_validations": self.total_validations,
            "signals_passed": self.signals_passed,
            "signals_rejected": self.signals_rejected,
            "conflicts_resolved": self.conflicts_resolved,
            "pass_rate": self.signals_passed / max(1, self.total_validations),
            "rejection_rate": self.signals_rejected / max(1, self.total_validations),
            "conflict_resolution_stats": dict(self.conflict_resolution_stats),
            "htf_bias_accuracy": {
                symbol: accuracy['correct'] / max(1, accuracy['total'])
                for symbol, accuracy in self.htf_bias_accuracy.items()
            },
            "confluence_weights": dict(self.confluence_weights),
            "validation_thresholds": dict(self.validation_thresholds),
            "memory_sizes": {
                "validation_history": len(self.validation_history),
                "validated_signals": len(self.validated_signals),
                "rejected_signals": len(self.rejected_signals),
                "conflict_cases": len(self.conflict_cases)
            },
            "current_strategy": self.conflict_resolution_strategy.value,
            "engine_version": "1.0.0"
        }