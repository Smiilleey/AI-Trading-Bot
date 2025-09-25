# core/microstructure_state_machine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class MicrostructureState(Enum):
    """Microstructure states for the state machine."""
    NEUTRAL = "neutral"
    SWEEP = "sweep"
    RECLAIM = "reclaim"
    DISPLACEMENT = "displacement"
    RETRACE = "retrace"
    ABSORPTION = "absorption"
    EXHAUSTION = "exhaustion"

class MicrostructureStateMachine:
    """
    MICROSTRUCTURE STATE MACHINE - True Volume/Imbalance Proxies
    
    Implements the complete microstructure state progression:
    sweep → reclaim → displacement → retrace → absorption → exhaustion
    
    Features:
    - State transition logic with volume confirmation
    - Participant inference from execution footprints
    - Absorption/divergence detection with statistics
    - Continuation vs failed breaks tracking
    - Volume imbalance analysis
    - Institutional activity detection
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Current state
        self.current_state = MicrostructureState.NEUTRAL
        self.state_history = deque(maxlen=1000)
        self.state_durations = defaultdict(list)
        
        # State transition parameters
        self.transition_params = {
            'sweep_volume_threshold': 1.5,      # Volume spike for sweep detection
            'displacement_momentum_threshold': 0.0002,  # Price momentum for displacement
            'reclaim_confirmation_candles': 2,   # Candles needed to confirm reclaim
            'retrace_percentage': 0.382,        # Fibonacci retrace percentage
            'absorption_volume_ratio': 0.7,     # Volume decrease for absorption
            'exhaustion_momentum_decay': 0.5    # Momentum decay for exhaustion
        }
        
        # Performance tracking
        self.state_transition_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.continuation_failure_stats = defaultdict(lambda: {'continuation': 0, 'failure': 0})
        
        # Volume and momentum tracking
        self.volume_profile = deque(maxlen=100)
        self.momentum_profile = deque(maxlen=100)
        self.imbalance_history = deque(maxlen=100)
        
        # Participant inference data
        self.execution_footprints = defaultdict(list)
        self.institutional_activity = defaultdict(list)
        
        # Statistics
        self.total_state_changes = 0
        self.correct_predictions = 0
    
    def process_market_data(self, 
                           market_data: Dict, 
                           symbol: str = "",
                           timeframe: str = "") -> Dict[str, Any]:
        """
        Process market data and determine current microstructure state.
        
        Args:
            market_data: Market data including candles, volume, etc.
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Complete microstructure analysis with state transitions
        """
        try:
            candles = market_data.get('candles', [])
            if not candles or len(candles) < 10:
                return self._create_state_response(False, "Insufficient market data")
            
            # Extract current market information
            current_candle = candles[-1]
            previous_candles = candles[-10:]
            
            # Analyze current market conditions
            volume_analysis = self._analyze_volume_conditions(previous_candles)
            momentum_analysis = self._analyze_momentum_conditions(previous_candles)
            imbalance_analysis = self._analyze_volume_imbalances(previous_candles)
            
            # Determine new state based on conditions
            new_state = self._determine_new_state(
                current_candle, previous_candles, 
                volume_analysis, momentum_analysis, imbalance_analysis
            )
            
            # Process state transition
            transition_info = self._process_state_transition(new_state, current_candle)
            
            # Analyze execution footprints
            execution_footprint = self._analyze_execution_footprint(
                current_candle, previous_candles, volume_analysis
            )
            
            # Participant inference
            participant_analysis = self._infer_participants(
                volume_analysis, momentum_analysis, execution_footprint
            )
            
            # Continuation vs failure prediction
            continuation_prediction = self._predict_continuation_failure(
                new_state, volume_analysis, momentum_analysis
            )
            
            # Create comprehensive response
            response = self._create_state_response(
                True,
                current_state=new_state,
                transition_info=transition_info,
                volume_analysis=volume_analysis,
                momentum_analysis=momentum_analysis,
                imbalance_analysis=imbalance_analysis,
                execution_footprint=execution_footprint,
                participant_analysis=participant_analysis,
                continuation_prediction=continuation_prediction,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Update tracking
            self._update_tracking(response)
            
            return response
            
        except Exception as e:
            return self._create_state_response(False, f"Microstructure analysis failed: {str(e)}")
    
    def _analyze_volume_conditions(self, candles: List[Dict]) -> Dict[str, Any]:
        """Analyze volume conditions for state determination."""
        try:
            volumes = [float(c.get('tick_volume', 1000)) for c in candles]
            
            if len(volumes) < 5:
                return {'analysis': 'insufficient_data'}
            
            # Calculate volume metrics
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1])
            volume_std = np.std(volumes[:-1])
            
            # Volume ratios and trends
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            
            # Volume classifications
            if volume_ratio > self.transition_params['sweep_volume_threshold']:
                volume_classification = 'surge'
            elif volume_ratio < self.transition_params['absorption_volume_ratio']:
                volume_classification = 'absorption'
            elif volume_ratio > 1.2:
                volume_classification = 'elevated'
            elif volume_ratio < 0.8:
                volume_classification = 'low'
            else:
                volume_classification = 'normal'
            
            # Volume distribution analysis
            volume_percentiles = {
                'p10': np.percentile(volumes, 10),
                'p25': np.percentile(volumes, 25),
                'p50': np.percentile(volumes, 50),
                'p75': np.percentile(volumes, 75),
                'p90': np.percentile(volumes, 90)
            }
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'classification': volume_classification,
                'percentiles': volume_percentiles,
                'volume_std': volume_std,
                'is_surge': volume_ratio > self.transition_params['sweep_volume_threshold'],
                'is_absorption': volume_ratio < self.transition_params['absorption_volume_ratio']
            }
            
        except Exception as e:
            return {'analysis': 'error', 'error': str(e)}
    
    def _analyze_momentum_conditions(self, candles: List[Dict]) -> Dict[str, Any]:
        """Analyze momentum conditions for state determination."""
        try:
            if len(candles) < 5:
                return {'analysis': 'insufficient_data'}
            
            # Price momentum calculation
            closes = [float(c['close']) for c in candles]
            highs = [float(c['high']) for c in candles]
            lows = [float(c['low']) for c in candles]
            
            # Current momentum
            current_momentum = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
            
            # Short-term momentum (3 candles)
            short_momentum = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 and closes[-4] > 0 else 0
            
            # Medium-term momentum (5 candles)
            medium_momentum = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0
            
            # Rate of change analysis
            roc = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]
            avg_roc = np.mean(roc) if roc else 0
            roc_std = np.std(roc) if len(roc) > 1 else 0
            
            # Momentum classification
            if abs(current_momentum) > self.transition_params['displacement_momentum_threshold']:
                momentum_classification = 'displacement'
            elif abs(current_momentum) < self.transition_params['displacement_momentum_threshold'] * 0.3:
                momentum_classification = 'stagnant'
            elif current_momentum > 0:
                momentum_classification = 'bullish'
            else:
                momentum_classification = 'bearish'
            
            # Momentum decay detection (for exhaustion)
            momentum_decay = 0
            if len(roc) >= 3:
                recent_roc = roc[-3:]
                if all(abs(recent_roc[i]) < abs(recent_roc[i-1]) for i in range(1, len(recent_roc))):
                    momentum_decay = 1 - (abs(recent_roc[-1]) / abs(recent_roc[0])) if recent_roc[0] != 0 else 0
            
            return {
                'current_momentum': current_momentum,
                'short_momentum': short_momentum,
                'medium_momentum': medium_momentum,
                'average_roc': avg_roc,
                'roc_std': roc_std,
                'classification': momentum_classification,
                'momentum_decay': momentum_decay,
                'is_displacement': abs(current_momentum) > self.transition_params['displacement_momentum_threshold'],
                'is_exhaustion': momentum_decay > self.transition_params['exhaustion_momentum_decay']
            }
            
        except Exception as e:
            return {'analysis': 'error', 'error': str(e)}
    
    def _analyze_volume_imbalances(self, candles: List[Dict]) -> Dict[str, Any]:
        """Analyze volume imbalances and delta."""
        try:
            if len(candles) < 5:
                return {'analysis': 'insufficient_data'}
            
            # Calculate delta (simplified - in production you'd have bid/ask volume)
            deltas = []
            for i in range(1, len(candles)):
                current = candles[i]
                previous = candles[i-1]
                
                price_change = float(current['close']) - float(previous['close'])
                volume = float(current.get('tick_volume', 1000))
                
                # Simplified delta calculation
                if price_change > 0:
                    delta = volume  # Buying pressure
                elif price_change < 0:
                    delta = -volume  # Selling pressure
                else:
                    delta = 0
                
                deltas.append(delta)
            
            # Cumulative delta
            cumulative_delta = sum(deltas)
            
            # Recent delta trend
            recent_deltas = deltas[-5:] if len(deltas) >= 5 else deltas
            delta_trend = np.polyfit(range(len(recent_deltas)), recent_deltas, 1)[0] if len(recent_deltas) > 1 else 0
            
            # Delta divergence detection
            prices = [float(c['close']) for c in candles[-len(deltas):]]
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0] if len(prices) > 1 else 0
            
            # Divergence occurs when price and delta trends oppose
            divergence = False
            if price_trend > 0 and delta_trend < 0:
                divergence = True
                divergence_type = 'bearish'
            elif price_trend < 0 and delta_trend > 0:
                divergence = True
                divergence_type = 'bullish'
            else:
                divergence_type = 'none'
            
            # Imbalance strength
            total_volume = sum(abs(d) for d in deltas)
            imbalance_strength = abs(cumulative_delta) / total_volume if total_volume > 0 else 0
            
            return {
                'cumulative_delta': cumulative_delta,
                'delta_trend': delta_trend,
                'imbalance_strength': imbalance_strength,
                'divergence': divergence,
                'divergence_type': divergence_type,
                'recent_deltas': recent_deltas,
                'dominant_side': 'buying' if cumulative_delta > 0 else 'selling' if cumulative_delta < 0 else 'neutral'
            }
            
        except Exception as e:
            return {'analysis': 'error', 'error': str(e)}
    
    def _determine_new_state(self, 
                            current_candle: Dict, 
                            previous_candles: List[Dict],
                            volume_analysis: Dict,
                            momentum_analysis: Dict,
                            imbalance_analysis: Dict) -> MicrostructureState:
        """Determine new microstructure state based on analysis."""
        try:
            current_price = float(current_candle['close'])
            
            # State transition logic
            
            # 1. SWEEP: High volume spike with price breaking key levels
            if (volume_analysis.get('is_surge', False) and
                momentum_analysis.get('is_displacement', False)):
                return MicrostructureState.SWEEP
            
            # 2. RECLAIM: Price returns to previously broken level
            if self.current_state == MicrostructureState.SWEEP:
                # Check if price is reclaiming the swept level
                if self._check_reclaim_conditions(current_candle, previous_candles):
                    return MicrostructureState.RECLAIM
            
            # 3. DISPLACEMENT: Strong directional move with momentum
            if (momentum_analysis.get('is_displacement', False) and
                not volume_analysis.get('is_absorption', False)):
                return MicrostructureState.DISPLACEMENT
            
            # 4. RETRACE: Pullback after displacement
            if self.current_state == MicrostructureState.DISPLACEMENT:
                if self._check_retrace_conditions(current_candle, previous_candles, momentum_analysis):
                    return MicrostructureState.RETRACE
            
            # 5. ABSORPTION: Volume decreasing while price stable
            if volume_analysis.get('is_absorption', False):
                return MicrostructureState.ABSORPTION
            
            # 6. EXHAUSTION: Momentum decay with divergence
            if (momentum_analysis.get('is_exhaustion', False) or
                imbalance_analysis.get('divergence', False)):
                return MicrostructureState.EXHAUSTION
            
            # Default: stay in current state or neutral
            return self.current_state if self.current_state != MicrostructureState.NEUTRAL else MicrostructureState.NEUTRAL
            
        except Exception:
            return MicrostructureState.NEUTRAL
    
    def _check_reclaim_conditions(self, current_candle: Dict, previous_candles: List[Dict]) -> bool:
        """Check if conditions are met for reclaim state."""
        try:
            # Simple reclaim check - price returning to previous level
            # In production, this would check against actual swept levels
            
            if len(previous_candles) < 5:
                return False
            
            current_close = float(current_candle['close'])
            recent_highs = [float(c['high']) for c in previous_candles[-5:]]
            recent_lows = [float(c['low']) for c in previous_candles[-5:]]
            
            # Check if price is back in previous range
            range_high = max(recent_highs)
            range_low = min(recent_lows)
            
            return range_low <= current_close <= range_high
            
        except Exception:
            return False
    
    def _check_retrace_conditions(self, 
                                 current_candle: Dict, 
                                 previous_candles: List[Dict],
                                 momentum_analysis: Dict) -> bool:
        """Check if conditions are met for retrace state."""
        try:
            # Check if momentum is decreasing (retrace after displacement)
            current_momentum = momentum_analysis.get('current_momentum', 0)
            momentum_decay = momentum_analysis.get('momentum_decay', 0)
            
            # Simple retrace detection
            return (abs(current_momentum) < self.transition_params['displacement_momentum_threshold'] * 0.5 or
                    momentum_decay > 0.3)
            
        except Exception:
            return False
    
    def _process_state_transition(self, new_state: MicrostructureState, current_candle: Dict) -> Dict[str, Any]:
        """Process state transition and update history."""
        try:
            old_state = self.current_state
            state_changed = new_state != old_state
            
            if state_changed:
                self.total_state_changes += 1
                
                # Record transition
                transition = {
                    'from_state': old_state.value,
                    'to_state': new_state.value,
                    'timestamp': current_candle.get('time', 'unknown'),
                    'price': float(current_candle['close']),
                    'volume': float(current_candle.get('tick_volume', 1000))
                }
                
                self.state_history.append(transition)
                
                # Update current state
                self.current_state = new_state
                
                # Calculate state duration
                if len(self.state_history) >= 2:
                    # This is simplified - in production you'd track actual time
                    duration = 1  # 1 candle duration
                    self.state_durations[old_state.value].append(duration)
            
            return {
                'state_changed': state_changed,
                'old_state': old_state.value,
                'new_state': new_state.value,
                'transition_count': self.total_state_changes,
                'state_duration': len(self.state_durations.get(old_state.value, [])),
                'confidence': self._calculate_transition_confidence(new_state, current_candle)
            }
            
        except Exception as e:
            return {'state_changed': False, 'error': str(e)}
    
    def _calculate_transition_confidence(self, state: MicrostructureState, candle: Dict) -> float:
        """Calculate confidence in state transition."""
        try:
            # Simple confidence calculation based on state characteristics
            confidence_map = {
                MicrostructureState.SWEEP: 0.8,
                MicrostructureState.DISPLACEMENT: 0.9,
                MicrostructureState.RECLAIM: 0.7,
                MicrostructureState.RETRACE: 0.6,
                MicrostructureState.ABSORPTION: 0.7,
                MicrostructureState.EXHAUSTION: 0.8,
                MicrostructureState.NEUTRAL: 0.5
            }
            
            base_confidence = confidence_map.get(state, 0.5)
            
            # Adjust based on volume (higher volume = higher confidence)
            volume = float(candle.get('tick_volume', 1000))
            if volume > 1500:  # High volume
                base_confidence += 0.1
            elif volume < 500:  # Low volume
                base_confidence -= 0.1
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception:
            return 0.5
    
    def _analyze_execution_footprint(self, 
                                   current_candle: Dict, 
                                   previous_candles: List[Dict],
                                   volume_analysis: Dict) -> Dict[str, Any]:
        """Analyze execution footprint for participant inference."""
        try:
            # Execution footprint analysis
            current_volume = float(current_candle.get('tick_volume', 1000))
            avg_volume = volume_analysis.get('average_volume', 1000)
            
            # Calculate execution characteristics
            price_range = float(current_candle['high']) - float(current_candle['low'])
            body_size = abs(float(current_candle['close']) - float(current_candle['open']))
            
            # Volume per pip analysis
            volume_per_pip = current_volume / (price_range * 10000) if price_range > 0 else 0
            
            # Execution speed (simplified)
            execution_speed = 'fast' if current_volume > avg_volume * 1.5 else 'normal'
            
            # Order size inference
            if current_volume > avg_volume * 2:
                order_size_class = 'institutional'
            elif current_volume > avg_volume * 1.3:
                order_size_class = 'large_retail'
            else:
                order_size_class = 'retail'
            
            # Absorption vs continuation footprint
            if body_size / price_range < 0.3:  # Small body, large wicks
                footprint_type = 'absorption'
            elif body_size / price_range > 0.8:  # Large body, small wicks
                footprint_type = 'continuation'
            else:
                footprint_type = 'indecision'
            
            return {
                'volume_per_pip': volume_per_pip,
                'execution_speed': execution_speed,
                'order_size_class': order_size_class,
                'footprint_type': footprint_type,
                'body_to_range_ratio': body_size / price_range if price_range > 0 else 0,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
            }
            
        except Exception as e:
            return {'analysis': 'error', 'error': str(e)}
    
    def _infer_participants(self, 
                           volume_analysis: Dict,
                           momentum_analysis: Dict, 
                           execution_footprint: Dict) -> Dict[str, Any]:
        """Infer market participants from execution characteristics."""
        try:
            participants = []
            confidence_scores = {}
            
            # Institutional participant detection
            if (execution_footprint.get('order_size_class') == 'institutional' and
                volume_analysis.get('classification') in ['surge', 'elevated']):
                participants.append('institutional')
                confidence_scores['institutional'] = 0.8
            
            # Retail participant detection
            if execution_footprint.get('order_size_class') == 'retail':
                participants.append('retail')
                confidence_scores['retail'] = 0.6
            
            # Algorithmic trading detection
            if (execution_footprint.get('execution_speed') == 'fast' and
                momentum_analysis.get('classification') == 'displacement'):
                participants.append('algorithmic')
                confidence_scores['algorithmic'] = 0.7
            
            # Market maker detection
            if (execution_footprint.get('footprint_type') == 'absorption' and
                volume_analysis.get('classification') == 'absorption'):
                participants.append('market_maker')
                confidence_scores['market_maker'] = 0.7
            
            # Determine dominant participant
            if confidence_scores:
                dominant_participant = max(confidence_scores, key=confidence_scores.get)
                dominant_confidence = confidence_scores[dominant_participant]
            else:
                dominant_participant = 'unknown'
                dominant_confidence = 0.0
            
            return {
                'participants': participants,
                'confidence_scores': confidence_scores,
                'dominant_participant': dominant_participant,
                'dominant_confidence': dominant_confidence,
                'participant_count': len(participants)
            }
            
        except Exception as e:
            return {'participants': [], 'error': str(e)}
    
    def _predict_continuation_failure(self, 
                                     state: MicrostructureState,
                                     volume_analysis: Dict,
                                     momentum_analysis: Dict) -> Dict[str, Any]:
        """Predict whether current move will continue or fail."""
        try:
            # Continuation probability based on state and conditions
            continuation_factors = []
            
            # State-based factors
            if state == MicrostructureState.DISPLACEMENT:
                continuation_factors.append(0.3)  # Displacement suggests continuation
            elif state == MicrostructureState.EXHAUSTION:
                continuation_factors.append(-0.4)  # Exhaustion suggests failure
            elif state == MicrostructureState.ABSORPTION:
                continuation_factors.append(-0.2)  # Absorption suggests potential reversal
            
            # Volume factors
            if volume_analysis.get('classification') == 'surge':
                continuation_factors.append(0.2)  # High volume supports continuation
            elif volume_analysis.get('classification') == 'absorption':
                continuation_factors.append(-0.3)  # Volume absorption suggests failure
            
            # Momentum factors
            if momentum_analysis.get('is_displacement'):
                continuation_factors.append(0.25)  # Strong momentum supports continuation
            elif momentum_analysis.get('is_exhaustion'):
                continuation_factors.append(-0.35)  # Momentum exhaustion suggests failure
            
            # Calculate continuation probability
            base_probability = 0.5  # Neutral starting point
            adjustment = sum(continuation_factors)
            continuation_probability = max(0.1, min(0.9, base_probability + adjustment))
            
            # Determine prediction
            if continuation_probability > 0.6:
                prediction = 'continuation'
            elif continuation_probability < 0.4:
                prediction = 'failure'
            else:
                prediction = 'uncertain'
            
            # Update statistics (simplified)
            symbol_key = 'default'  # Would use actual symbol
            self.continuation_failure_stats[symbol_key]['total'] = (
                self.continuation_failure_stats[symbol_key].get('total', 0) + 1
            )
            
            return {
                'prediction': prediction,
                'continuation_probability': continuation_probability,
                'failure_probability': 1 - continuation_probability,
                'confidence': abs(continuation_probability - 0.5) * 2,  # Distance from neutral
                'factors': continuation_factors,
                'statistics': dict(self.continuation_failure_stats[symbol_key])
            }
            
        except Exception as e:
            return {'prediction': 'uncertain', 'error': str(e)}
    
    def _create_state_response(self, 
                              valid: bool,
                              current_state: MicrostructureState = None,
                              transition_info: Dict = None,
                              volume_analysis: Dict = None,
                              momentum_analysis: Dict = None,
                              imbalance_analysis: Dict = None,
                              execution_footprint: Dict = None,
                              participant_analysis: Dict = None,
                              continuation_prediction: Dict = None,
                              symbol: str = "",
                              timeframe: str = "",
                              error: str = "") -> Dict[str, Any]:
        """Create comprehensive microstructure state response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "current_state": current_state.value if current_state else "unknown",
            "transition_info": transition_info or {},
            "volume_analysis": volume_analysis or {},
            "momentum_analysis": momentum_analysis or {},
            "imbalance_analysis": imbalance_analysis or {},
            "execution_footprint": execution_footprint or {},
            "participant_analysis": participant_analysis or {},
            "continuation_prediction": continuation_prediction or {},
            "state_history": [t for t in list(self.state_history)[-5:]] if self.state_history else [],
            "statistics": {
                "total_state_changes": self.total_state_changes,
                "correct_predictions": self.correct_predictions,
                "accuracy": self.correct_predictions / max(1, self.total_state_changes),
                "state_durations": {k: np.mean(v) if v else 0 for k, v in self.state_durations.items()}
            },
            "metadata": {
                "engine_version": "1.0.0",
                "analysis_type": "microstructure_state_machine"
            }
        }
    
    def _update_tracking(self, response: Dict):
        """Update performance tracking."""
        try:
            # Update volume and momentum profiles
            if response.get('volume_analysis'):
                self.volume_profile.append(response['volume_analysis'])
            
            if response.get('momentum_analysis'):
                self.momentum_profile.append(response['momentum_analysis'])
            
            if response.get('imbalance_analysis'):
                self.imbalance_history.append(response['imbalance_analysis'])
            
            # Store execution footprints for future analysis
            symbol = response.get('symbol', 'default')
            if response.get('execution_footprint'):
                self.execution_footprints[symbol].append({
                    'timestamp': datetime.now(),
                    'footprint': response['execution_footprint'],
                    'state': response.get('current_state')
                })
                
                # Keep memory manageable
                if len(self.execution_footprints[symbol]) > 200:
                    self.execution_footprints[symbol] = self.execution_footprints[symbol][-100:]
                    
        except Exception:
            pass  # Silent fail for tracking updates
    
    def update_prediction_accuracy(self, prediction: str, actual_outcome: str):
        """Update prediction accuracy tracking."""
        try:
            if prediction == actual_outcome:
                self.correct_predictions += 1
            
            # Update state-specific accuracy
            state_key = f"{self.current_state.value}_{prediction}"
            self.state_transition_accuracy[state_key]['total'] += 1
            if prediction == actual_outcome:
                self.state_transition_accuracy[state_key]['correct'] += 1
                
        except Exception:
            pass  # Silent fail for accuracy updates
    
    def get_machine_stats(self) -> Dict[str, Any]:
        """Get comprehensive state machine statistics."""
        return {
            "current_state": self.current_state.value,
            "total_state_changes": self.total_state_changes,
            "correct_predictions": self.correct_predictions,
            "overall_accuracy": self.correct_predictions / max(1, self.total_state_changes),
            "state_transition_accuracy": {
                k: v['correct'] / max(1, v['total']) 
                for k, v in self.state_transition_accuracy.items()
            },
            "state_durations": {
                k: {
                    'avg_duration': np.mean(v) if v else 0,
                    'total_occurrences': len(v)
                } for k, v in self.state_durations.items()
            },
            "continuation_failure_stats": dict(self.continuation_failure_stats),
            "memory_sizes": {
                "state_history": len(self.state_history),
                "volume_profile": len(self.volume_profile),
                "momentum_profile": len(self.momentum_profile),
                "imbalance_history": len(self.imbalance_history),
                "execution_footprints": sum(len(fp) for fp in self.execution_footprints.values())
            },
            "transition_parameters": self.transition_params,
            "engine_version": "1.0.0"
        }
