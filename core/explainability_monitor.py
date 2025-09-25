# core/explainability_monitor.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class ExplainabilityMonitor:
    """
    EXPLAINABILITY AND MONITORING SYSTEM - Complete Trade Transparency
    
    Features:
    - Per-trade narrative generation (targeted liquidity, PD array, participant inference)
    - Calibration curves for ML confidence monitoring
    - Edge decay detection and alerting
    - Model drift monitoring per session/pair
    - Performance attribution analysis
    - Real-time system health monitoring
    - Comprehensive audit trail
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Narrative generation parameters
        self.narrative_components = {
            'liquidity_targeting': {
                'weight': 0.25,
                'description': 'Which liquidity levels were targeted'
            },
            'pd_array_analysis': {
                'weight': 0.25,
                'description': 'PD array positioning and confluence'
            },
            'participant_inference': {
                'weight': 0.20,
                'description': 'Institutional vs retail participant activity'
            },
            'market_structure': {
                'weight': 0.15,
                'description': 'Market structure context and timing'
            },
            'risk_context': {
                'weight': 0.10,
                'description': 'Risk management and position sizing rationale'
            },
            'session_context': {
                'weight': 0.05,
                'description': 'Session-specific factors and timing'
            }
        }
        
        # Calibration monitoring
        self.calibration_bins = np.linspace(0, 1, 11)  # 10 bins for calibration curves
        self.calibration_data = defaultdict(lambda: {
            'predictions': [],
            'outcomes': [],
            'timestamps': [],
            'calibration_curve': None,
            'last_update': None
        })
        
        # Edge decay monitoring
        self.edge_monitoring = defaultdict(lambda: {
            'historical_performance': deque(maxlen=100),
            'rolling_metrics': {},
            'decay_alerts': [],
            'last_performance_check': datetime.now()
        })
        
        # Model drift monitoring
        self.drift_monitoring = defaultdict(lambda: {
            'feature_distributions': defaultdict(list),
            'prediction_distributions': [],
            'drift_scores': {},
            'drift_alerts': [],
            'baseline_established': False
        })
        
        # Trade narratives storage
        self.trade_narratives = deque(maxlen=1000)
        self.narrative_templates = self._initialize_narrative_templates()
        
        # Performance attribution
        self.performance_attribution = defaultdict(lambda: {
            'by_liquidity_type': defaultdict(lambda: {'count': 0, 'pnl': 0.0}),
            'by_pd_array': defaultdict(lambda: {'count': 0, 'pnl': 0.0}),
            'by_participant': defaultdict(lambda: {'count': 0, 'pnl': 0.0}),
            'by_session': defaultdict(lambda: {'count': 0, 'pnl': 0.0}),
            'by_structure': defaultdict(lambda: {'count': 0, 'pnl': 0.0})
        })
        
        # Alert system
        self.alerts = deque(maxlen=100)
        self.critical_alerts = deque(maxlen=50)
        
        # Statistics
        self.total_narratives_generated = 0
        self.calibration_updates = 0
        self.drift_detections = 0
        self.edge_decay_alerts = 0
    
    def generate_trade_narrative(self, 
                                trade_data: Dict,
                                market_analysis: Dict,
                                execution_details: Dict,
                                outcome_data: Dict = None) -> Dict[str, Any]:
        """
        Generate comprehensive per-trade narrative explaining the decision process.
        
        Args:
            trade_data: Basic trade information
            market_analysis: Complete market analysis that led to trade
            execution_details: Execution specifics
            outcome_data: Trade outcome (if available)
            
        Returns:
            Complete trade narrative with explanations
        """
        try:
            self.total_narratives_generated += 1
            
            # 1. Analyze liquidity targeting
            liquidity_narrative = self._generate_liquidity_narrative(
                trade_data, market_analysis
            )
            
            # 2. Analyze PD array positioning
            pd_array_narrative = self._generate_pd_array_narrative(
                trade_data, market_analysis
            )
            
            # 3. Infer participant behavior
            participant_narrative = self._generate_participant_narrative(
                trade_data, market_analysis
            )
            
            # 4. Explain market structure context
            structure_narrative = self._generate_structure_narrative(
                trade_data, market_analysis
            )
            
            # 5. Explain risk management rationale
            risk_narrative = self._generate_risk_narrative(
                trade_data, execution_details
            )
            
            # 6. Add session context
            session_narrative = self._generate_session_narrative(
                trade_data, market_analysis
            )
            
            # 7. Compile complete narrative
            complete_narrative = self._compile_complete_narrative(
                liquidity_narrative, pd_array_narrative, participant_narrative,
                structure_narrative, risk_narrative, session_narrative,
                trade_data, outcome_data
            )
            
            # 8. Create response
            response = self._create_narrative_response(
                True,
                trade_id=trade_data.get('trade_id', 'unknown'),
                complete_narrative=complete_narrative,
                component_narratives={
                    'liquidity': liquidity_narrative,
                    'pd_array': pd_array_narrative,
                    'participant': participant_narrative,
                    'structure': structure_narrative,
                    'risk': risk_narrative,
                    'session': session_narrative
                },
                trade_data=trade_data,
                market_analysis=market_analysis
            )
            
            # Store narrative
            self._store_narrative(response)
            
            return response
            
        except Exception as e:
            return self._create_narrative_response(False, error=f"Narrative generation failed: {str(e)}")
    
    def _initialize_narrative_templates(self) -> Dict[str, str]:
        """Initialize narrative templates for different scenarios."""
        return {
            'liquidity_sweep': "Targeted {liquidity_type} liquidity at {level:.5f}, expecting {direction} continuation post-sweep",
            'pd_array_entry': "Entered at {pd_status} zone ({percentage:.1f}% of range), expecting {bias} from this level",
            'fvg_fill': "Anticipated FVG fill at {fvg_level:.5f}, {fvg_type} gap with {strength} strength",
            'order_block_retest': "Order block retest at {ob_level:.5f}, {ob_type} OB with {test_count} previous tests",
            'institutional_flow': "Detected {participant_type} activity with {confidence:.1f}% confidence, {flow_direction} bias",
            'structure_break': "Structure break: {structure_type} at {break_level:.5f}, confirmation via {confirmation_method}",
            'session_play': "Session-specific {session} play, {bias} bias expected during {killzone}",
            'risk_management': "Position sized at {position_size:.2f} lots ({risk_percentage:.1f}% risk) due to {risk_factors}"
        }
    
    def _generate_liquidity_narrative(self, trade_data: Dict, market_analysis: Dict) -> str:
        """Generate narrative about liquidity targeting."""
        try:
            liquidity_pools = market_analysis.get('liquidity_pools', [])
            entry_level = float(trade_data.get('entry_price', 0))
            direction = trade_data.get('direction', 'unknown')
            
            if liquidity_pools:
                # Find closest liquidity pool
                closest_pool = min(liquidity_pools, key=lambda p: abs(p['center'] - entry_level))
                
                template = self.narrative_templates['liquidity_sweep']
                return template.format(
                    liquidity_type=closest_pool.get('type', 'unknown'),
                    level=closest_pool['center'],
                    direction=direction
                )
            else:
                return f"No specific liquidity targeting identified for {direction} entry at {entry_level:.5f}"
                
        except Exception:
            return "Liquidity analysis unavailable"
    
    def _generate_pd_array_narrative(self, trade_data: Dict, market_analysis: Dict) -> str:
        """Generate narrative about PD array positioning."""
        try:
            pd_analysis = market_analysis.get('premium_discount', {})
            
            if pd_analysis:
                pd_status = pd_analysis.get('status', 'unknown')
                percentage = pd_analysis.get('percentage', 0.5) * 100
                bias = pd_analysis.get('bias', 'neutral')
                
                template = self.narrative_templates['pd_array_entry']
                return template.format(
                    pd_status=pd_status,
                    percentage=percentage,
                    bias=bias
                )
            else:
                return "PD array analysis not available for this trade"
                
        except Exception:
            return "PD array analysis unavailable"
    
    def _generate_participant_narrative(self, trade_data: Dict, market_analysis: Dict) -> str:
        """Generate narrative about participant inference."""
        try:
            participant_analysis = market_analysis.get('participant_analysis', {})
            
            if participant_analysis:
                dominant_participant = participant_analysis.get('dominant_participant', 'unknown')
                confidence = participant_analysis.get('dominant_confidence', 0.0) * 100
                
                orderflow_data = market_analysis.get('order_flow', {})
                flow_direction = orderflow_data.get('dominant_side', 'neutral')
                
                template = self.narrative_templates['institutional_flow']
                return template.format(
                    participant_type=dominant_participant,
                    confidence=confidence,
                    flow_direction=flow_direction
                )
            else:
                return "Participant inference not available for this trade"
                
        except Exception:
            return "Participant analysis unavailable"
    
    def _generate_structure_narrative(self, trade_data: Dict, market_analysis: Dict) -> str:
        """Generate narrative about market structure context."""
        try:
            structure_data = market_analysis.get('structure', {})
            
            if structure_data.get('choch'):
                return "Change of Character (CHoCH) detected, indicating potential trend reversal"
            elif structure_data.get('bos'):
                return "Break of Structure (BOS) confirmed, suggesting trend continuation"
            elif structure_data.get('false_breakout'):
                return "False breakout detected, expecting reversal back into range"
            else:
                structure_type = market_analysis.get('structure_type', 'ranging')
                return f"Market structure: {structure_type}, no significant structural shifts detected"
                
        except Exception:
            return "Structure analysis unavailable"
    
    def _generate_risk_narrative(self, trade_data: Dict, execution_details: Dict) -> str:
        """Generate narrative about risk management decisions."""
        try:
            position_size = float(trade_data.get('position_size', 0))
            risk_percentage = float(execution_details.get('risk_percentage', 0))
            risk_factors = execution_details.get('risk_factors', [])
            
            template = self.narrative_templates['risk_management']
            return template.format(
                position_size=position_size,
                risk_percentage=risk_percentage,
                risk_factors=', '.join(risk_factors) if risk_factors else 'standard risk assessment'
            )
            
        except Exception:
            return "Risk analysis unavailable"
    
    def _generate_session_narrative(self, trade_data: Dict, market_analysis: Dict) -> str:
        """Generate narrative about session-specific context."""
        try:
            current_time = datetime.now()
            hour = current_time.hour
            
            # Determine session and characteristics
            if 8 <= hour <= 12:
                session = "London AM"
                characteristics = "High volatility, trend initiation period"
            elif 12 <= hour <= 16:
                session = "London PM"
                characteristics = "Consolidation and continuation patterns"
            elif 13 <= hour <= 17:
                session = "NY AM"
                characteristics = "Peak activity, major moves expected"
            elif 17 <= hour <= 21:
                session = "NY PM"
                characteristics = "Reversal and profit-taking period"
            elif 0 <= hour <= 8:
                session = "Asian"
                characteristics = "Range-bound, liquidity building"
            else:
                session = "Off-hours"
                characteristics = "Low liquidity, avoid major positions"
            
            return f"Trade executed during {session} session: {characteristics}"
            
        except Exception:
            return "Session analysis unavailable"
    
    def _compile_complete_narrative(self, 
                                   liquidity_narrative: str,
                                   pd_array_narrative: str,
                                   participant_narrative: str,
                                   structure_narrative: str,
                                   risk_narrative: str,
                                   session_narrative: str,
                                   trade_data: Dict,
                                   outcome_data: Dict = None) -> Dict[str, Any]:
        """Compile complete trade narrative."""
        try:
            # Create main narrative
            main_narrative = f"""
TRADE NARRATIVE - {trade_data.get('symbol', 'UNKNOWN')} {trade_data.get('direction', 'UNKNOWN').upper()}

📍 LIQUIDITY TARGETING:
{liquidity_narrative}

📊 PD ARRAY POSITIONING:
{pd_array_narrative}

🏦 PARTICIPANT INFERENCE:
{participant_narrative}

🏗️ MARKET STRUCTURE:
{structure_narrative}

⚖️ RISK MANAGEMENT:
{risk_narrative}

🕒 SESSION CONTEXT:
{session_narrative}
"""
            
            # Add outcome narrative if available
            if outcome_data:
                outcome_narrative = self._generate_outcome_narrative(outcome_data, trade_data)
                main_narrative += f"\n\n📈 TRADE OUTCOME:\n{outcome_narrative}"
            
            # Create structured narrative data
            structured_narrative = {
                'main_narrative': main_narrative.strip(),
                'components': {
                    'liquidity_targeting': liquidity_narrative,
                    'pd_array_analysis': pd_array_narrative,
                    'participant_inference': participant_narrative,
                    'market_structure': structure_narrative,
                    'risk_management': risk_narrative,
                    'session_context': session_narrative
                },
                'trade_summary': {
                    'symbol': trade_data.get('symbol', 'UNKNOWN'),
                    'direction': trade_data.get('direction', 'UNKNOWN'),
                    'entry_price': float(trade_data.get('entry_price', 0)),
                    'position_size': float(trade_data.get('position_size', 0)),
                    'timestamp': trade_data.get('timestamp', datetime.now().isoformat())
                },
                'confidence_scores': {
                    'overall_confidence': float(trade_data.get('confidence', 0.5)),
                    'ml_confidence': float(trade_data.get('ml_confidence', 0.5)),
                    'structure_confidence': float(trade_data.get('structure_confidence', 0.5)),
                    'flow_confidence': float(trade_data.get('flow_confidence', 0.5))
                }
            }
            
            if outcome_data:
                structured_narrative['outcome'] = {
                    'result': outcome_data.get('result', 'unknown'),
                    'pnl': float(outcome_data.get('pnl', 0)),
                    'rr_achieved': float(outcome_data.get('rr', 0)),
                    'hold_time_minutes': float(outcome_data.get('hold_time_minutes', 0))
                }
            
            return structured_narrative
            
        except Exception as e:
            return {'main_narrative': f"Narrative generation failed: {str(e)}"}
    
    def _generate_outcome_narrative(self, outcome_data: Dict, trade_data: Dict) -> str:
        """Generate narrative about trade outcome."""
        try:
            result = outcome_data.get('result', 'unknown')
            pnl = float(outcome_data.get('pnl', 0))
            rr = float(outcome_data.get('rr', 0))
            exit_reason = outcome_data.get('exit_reason', 'unknown')
            
            if result == 'win':
                return f"✅ SUCCESSFUL TRADE: +{pnl:.2f} PnL ({rr:.2f}R) - Exit: {exit_reason}"
            elif result == 'loss':
                return f"❌ LOSING TRADE: {pnl:.2f} PnL ({rr:.2f}R) - Exit: {exit_reason}"
            else:
                return f"⏸️ NEUTRAL TRADE: {pnl:.2f} PnL - Exit: {exit_reason}"
                
        except Exception:
            return "Outcome analysis unavailable"
    
    def update_calibration_data(self, 
                               symbol: str,
                               session: str,
                               predicted_confidence: float,
                               actual_outcome: bool,
                               model_name: str = "default") -> Dict[str, Any]:
        """
        Update calibration data for ML confidence monitoring.
        
        Args:
            symbol: Trading symbol
            session: Trading session
            predicted_confidence: Model's predicted confidence (0-1)
            actual_outcome: Actual trade outcome (True/False)
            model_name: Name of the model making prediction
            
        Returns:
            Calibration update analysis
        """
        try:
            calibration_key = f"{symbol}_{session}_{model_name}"
            calib_data = self.calibration_data[calibration_key]
            
            # Add new data point
            calib_data['predictions'].append(predicted_confidence)
            calib_data['outcomes'].append(1.0 if actual_outcome else 0.0)
            calib_data['timestamps'].append(datetime.now())
            
            # Keep memory manageable
            if len(calib_data['predictions']) > 500:
                calib_data['predictions'] = calib_data['predictions'][-250:]
                calib_data['outcomes'] = calib_data['outcomes'][-250:]
                calib_data['timestamps'] = calib_data['timestamps'][-250:]
            
            # Update calibration curve if enough samples
            update_result = {'curve_updated': False, 'samples_count': len(calib_data['predictions'])}
            
            if len(calib_data['predictions']) >= 50:  # Minimum samples for reliable calibration
                calibration_curve = self._calculate_calibration_curve(
                    calib_data['predictions'], 
                    calib_data['outcomes']
                )
                
                calib_data['calibration_curve'] = calibration_curve
                calib_data['last_update'] = datetime.now()
                update_result['curve_updated'] = True
                update_result['calibration_error'] = calibration_curve.get('calibration_error', 0.0)
                
                # Check for calibration degradation
                if calibration_curve.get('calibration_error', 0.0) > 0.15:  # 15% error threshold
                    self._generate_calibration_alert(calibration_key, calibration_curve)
                
                self.calibration_updates += 1
            
            return update_result
            
        except Exception as e:
            return {'curve_updated': False, 'error': str(e)}
    
    def _calculate_calibration_curve(self, predictions: List[float], outcomes: List[float]) -> Dict[str, Any]:
        """Calculate calibration curve and reliability metrics."""
        try:
            from sklearn.calibration import calibration_curve
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                outcomes, predictions, n_bins=min(10, len(predictions) // 5)
            )
            
            # Calculate Expected Calibration Error (ECE)
            ece = 0.0
            for i in range(len(fraction_of_positives)):
                if len(predictions) > 0:
                    bin_size = len(predictions) / len(fraction_of_positives)
                    ece += abs(fraction_of_positives[i] - mean_predicted_value[i]) * (bin_size / len(predictions))
            
            # Calculate Brier Score
            brier_score = np.mean([(pred - outcome) ** 2 for pred, outcome in zip(predictions, outcomes)])
            
            # Calculate reliability
            reliability = 1.0 - ece
            
            return {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'calibration_error': ece,
                'brier_score': brier_score,
                'reliability': reliability,
                'sample_count': len(predictions),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Fallback calculation
            if predictions and outcomes:
                predicted_mean = np.mean(predictions)
                actual_mean = np.mean(outcomes)
                calibration_error = abs(predicted_mean - actual_mean)
                
                return {
                    'calibration_error': calibration_error,
                    'reliability': 1.0 - calibration_error,
                    'sample_count': len(predictions),
                    'error': str(e)
                }
            return {'error': str(e)}
    
    def _generate_calibration_alert(self, calibration_key: str, calibration_curve: Dict):
        """Generate alert for calibration degradation."""
        try:
            alert = {
                'timestamp': datetime.now(),
                'type': 'calibration_degradation',
                'severity': 'high',
                'model_key': calibration_key,
                'calibration_error': calibration_curve.get('calibration_error', 0.0),
                'sample_count': calibration_curve.get('sample_count', 0),
                'message': f"Model {calibration_key} shows calibration degradation: {calibration_curve.get('calibration_error', 0.0):.3f} error"
            }
            
            self.critical_alerts.append(alert)
            
        except Exception:
            pass  # Silent fail for alert generation
    
    def monitor_edge_decay(self, 
                          symbol: str,
                          session: str,
                          recent_performance_data: List[Dict]) -> Dict[str, Any]:
        """
        Monitor for edge decay in trading performance.
        
        Args:
            symbol: Trading symbol
            session: Trading session
            recent_performance_data: Recent trade performance data
            
        Returns:
            Edge decay analysis and alerts
        """
        try:
            edge_key = f"{symbol}_{session}"
            edge_data = self.edge_monitoring[edge_key]
            
            # Update performance history
            for perf_data in recent_performance_data:
                edge_data['historical_performance'].append({
                    'timestamp': datetime.now(),
                    'win_rate': float(perf_data.get('win_rate', 0.0)),
                    'avg_pnl': float(perf_data.get('avg_pnl', 0.0)),
                    'sharpe_ratio': float(perf_data.get('sharpe_ratio', 0.0)),
                    'total_trades': int(perf_data.get('total_trades', 0))
                })
            
            # Calculate rolling metrics
            if len(edge_data['historical_performance']) >= 10:
                recent_performance = list(edge_data['historical_performance'])[-20:]
                
                # Calculate rolling metrics
                rolling_metrics = {
                    'rolling_win_rate': np.mean([p['win_rate'] for p in recent_performance]),
                    'rolling_avg_pnl': np.mean([p['avg_pnl'] for p in recent_performance]),
                    'rolling_sharpe': np.mean([p['sharpe_ratio'] for p in recent_performance]),
                    'win_rate_trend': self._calculate_trend([p['win_rate'] for p in recent_performance]),
                    'pnl_trend': self._calculate_trend([p['avg_pnl'] for p in recent_performance]),
                    'sharpe_trend': self._calculate_trend([p['sharpe_ratio'] for p in recent_performance])
                }
                
                edge_data['rolling_metrics'] = rolling_metrics
                
                # Check for edge decay
                decay_detected = self._detect_edge_decay(rolling_metrics, edge_key)
                
                edge_data['last_performance_check'] = datetime.now()
                
                return {
                    'edge_key': edge_key,
                    'rolling_metrics': rolling_metrics,
                    'decay_detected': decay_detected,
                    'sample_count': len(recent_performance),
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {
                    'edge_key': edge_key,
                    'sample_count': len(edge_data['historical_performance']),
                    'status': 'insufficient_data'
                }
                
        except Exception as e:
            return {'error': f"Edge monitoring failed: {str(e)}"}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction for performance metrics."""
        try:
            if len(values) < 3:
                return 0.0
            
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope
            
        except Exception:
            return 0.0
    
    def _detect_edge_decay(self, rolling_metrics: Dict, edge_key: str) -> bool:
        """Detect edge decay in performance metrics."""
        try:
            decay_indicators = []
            
            # Check win rate decline
            if rolling_metrics['win_rate_trend'] < -0.01:  # 1% decline per period
                decay_indicators.append('win_rate_decline')
            
            # Check PnL decline
            if rolling_metrics['pnl_trend'] < -1.0:  # $1 decline per period
                decay_indicators.append('pnl_decline')
            
            # Check Sharpe decline
            if rolling_metrics['sharpe_trend'] < -0.05:  # 0.05 decline per period
                decay_indicators.append('sharpe_decline')
            
            # Check absolute performance levels
            if rolling_metrics['rolling_win_rate'] < 0.45:  # Below 45% win rate
                decay_indicators.append('low_win_rate')
            
            if rolling_metrics['rolling_sharpe'] < 0.5:  # Below 0.5 Sharpe
                decay_indicators.append('low_sharpe')
            
            # Edge decay detected if multiple indicators
            decay_detected = len(decay_indicators) >= 2
            
            if decay_detected:
                self.edge_decay_alerts += 1
                self._generate_edge_decay_alert(edge_key, decay_indicators, rolling_metrics)
            
            return decay_detected
            
        except Exception:
            return False
    
    def _generate_edge_decay_alert(self, edge_key: str, indicators: List[str], metrics: Dict):
        """Generate edge decay alert."""
        try:
            alert = {
                'timestamp': datetime.now(),
                'type': 'edge_decay',
                'severity': 'high',
                'edge_key': edge_key,
                'decay_indicators': indicators,
                'current_metrics': metrics,
                'message': f"Edge decay detected for {edge_key}: {', '.join(indicators)}",
                'recommendations': [
                    'Review and retrain models',
                    'Check for market regime changes',
                    'Consider reducing position sizes',
                    'Analyze recent trade patterns'
                ]
            }
            
            self.critical_alerts.append(alert)
            
        except Exception:
            pass  # Silent fail for alert generation
    
    def monitor_model_drift(self, 
                           symbol: str,
                           session: str,
                           features: Dict[str, float],
                           prediction: float,
                           model_name: str = "default") -> Dict[str, Any]:
        """
        Monitor model drift per session/pair.
        
        Args:
            symbol: Trading symbol
            session: Trading session
            features: Input features used for prediction
            prediction: Model prediction
            model_name: Name of the model
            
        Returns:
            Drift monitoring analysis
        """
        try:
            drift_key = f"{symbol}_{session}_{model_name}"
            drift_data = self.drift_monitoring[drift_key]
            
            # Add new feature distributions
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    drift_data['feature_distributions'][feature_name].append(value)
                    
                    # Keep memory manageable
                    if len(drift_data['feature_distributions'][feature_name]) > 200:
                        drift_data['feature_distributions'][feature_name] = drift_data['feature_distributions'][feature_name][-100:]
            
            # Add prediction to distribution
            drift_data['prediction_distributions'].append(prediction)
            if len(drift_data['prediction_distributions']) > 200:
                drift_data['prediction_distributions'] = drift_data['prediction_distributions'][-100:]
            
            # Calculate drift scores if baseline established
            drift_analysis = {'drift_detected': False, 'drifted_features': []}
            
            if len(drift_data['prediction_distributions']) >= 50:
                if not drift_data['baseline_established']:
                    # Establish baseline
                    self._establish_drift_baseline(drift_data)
                    drift_data['baseline_established'] = True
                else:
                    # Calculate drift scores
                    drift_analysis = self._calculate_drift_scores(drift_data, features, prediction)
            
            return {
                'drift_key': drift_key,
                'drift_analysis': drift_analysis,
                'sample_count': len(drift_data['prediction_distributions']),
                'baseline_established': drift_data['baseline_established'],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f"Drift monitoring failed: {str(e)}"}
    
    def _establish_drift_baseline(self, drift_data: Dict):
        """Establish baseline distributions for drift detection."""
        try:
            baselines = {}
            
            # Feature baselines
            for feature_name, values in drift_data['feature_distributions'].items():
                if len(values) >= 20:
                    baselines[feature_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'percentiles': {
                            'p25': np.percentile(values, 25),
                            'p50': np.percentile(values, 50),
                            'p75': np.percentile(values, 75)
                        }
                    }
            
            # Prediction baseline
            predictions = drift_data['prediction_distributions']
            baselines['predictions'] = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'percentiles': {
                    'p25': np.percentile(predictions, 25),
                    'p50': np.percentile(predictions, 50),
                    'p75': np.percentile(predictions, 75)
                }
            }
            
            drift_data['baselines'] = baselines
            
        except Exception:
            pass  # Silent fail for baseline establishment
    
    def _calculate_drift_scores(self, drift_data: Dict, current_features: Dict, current_prediction: float) -> Dict[str, Any]:
        """Calculate drift scores for features and predictions."""
        try:
            drift_analysis = {
                'drift_detected': False,
                'drifted_features': [],
                'drift_scores': {},
                'prediction_drift': 0.0
            }
            
            baselines = drift_data.get('baselines', {})
            if not baselines:
                return drift_analysis
            
            drifted_features = []
            drift_scores = {}
            
            # Check feature drift
            for feature_name, current_value in current_features.items():
                if feature_name in baselines and isinstance(current_value, (int, float)):
                    baseline = baselines[feature_name]
                    
                    # Calculate standardized drift score
                    if baseline['std'] > 0:
                        drift_score = abs(current_value - baseline['mean']) / baseline['std']
                    else:
                        drift_score = 0.0
                    
                    drift_scores[feature_name] = drift_score
                    
                    # Check if drift is significant (>3 standard deviations)
                    if drift_score > 3.0:
                        drifted_features.append({
                            'feature': feature_name,
                            'drift_score': drift_score,
                            'current_value': current_value,
                            'baseline_mean': baseline['mean'],
                            'baseline_std': baseline['std']
                        })
            
            # Check prediction drift
            pred_baseline = baselines.get('predictions', {})
            if pred_baseline and pred_baseline.get('std', 0) > 0:
                pred_drift_score = abs(current_prediction - pred_baseline['mean']) / pred_baseline['std']
                drift_analysis['prediction_drift'] = pred_drift_score
                
                if pred_drift_score > 2.0:  # Prediction drift threshold
                    drifted_features.append({
                        'feature': 'model_prediction',
                        'drift_score': pred_drift_score,
                        'current_value': current_prediction,
                        'baseline_mean': pred_baseline['mean']
                    })
            
            drift_analysis['drifted_features'] = drifted_features
            drift_analysis['drift_scores'] = drift_scores
            drift_analysis['drift_detected'] = len(drifted_features) > 0
            
            # Generate drift alert if detected
            if drift_analysis['drift_detected']:
                self.drift_detections += 1
                self._generate_drift_alert(drift_data, drift_analysis)
            
            return drift_analysis
            
        except Exception as e:
            return {'drift_detected': False, 'error': str(e)}
    
    def _generate_drift_alert(self, drift_data: Dict, drift_analysis: Dict):
        """Generate model drift alert."""
        try:
            alert = {
                'timestamp': datetime.now(),
                'type': 'model_drift',
                'severity': 'medium',
                'drifted_features': [f['feature'] for f in drift_analysis['drifted_features']],
                'max_drift_score': max(f['drift_score'] for f in drift_analysis['drifted_features']),
                'message': f"Model drift detected in {len(drift_analysis['drifted_features'])} features",
                'recommendations': [
                    'Review feature engineering pipeline',
                    'Check for market regime changes',
                    'Consider model retraining',
                    'Update feature baselines'
                ]
            }
            
            self.alerts.append(alert)
            
        except Exception:
            pass  # Silent fail for alert generation
    
    def _create_narrative_response(self, 
                                  valid: bool,
                                  trade_id: str = "",
                                  complete_narrative: Dict = None,
                                  component_narratives: Dict = None,
                                  trade_data: Dict = None,
                                  market_analysis: Dict = None,
                                  error: str = "") -> Dict[str, Any]:
        """Create comprehensive narrative response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat(),
            "complete_narrative": complete_narrative or {},
            "component_narratives": component_narratives or {},
            "trade_summary": {
                "symbol": trade_data.get('symbol') if trade_data else "",
                "direction": trade_data.get('direction') if trade_data else "",
                "confidence": float(trade_data.get('confidence', 0)) if trade_data else 0.0
            },
            "analysis_components": {
                "market_structure_analyzed": bool(market_analysis.get('structure')) if market_analysis else False,
                "liquidity_analyzed": bool(market_analysis.get('liquidity_pools')) if market_analysis else False,
                "pd_array_analyzed": bool(market_analysis.get('premium_discount')) if market_analysis else False,
                "participant_analyzed": bool(market_analysis.get('participant_analysis')) if market_analysis else False
            },
            "metadata": {
                "total_narratives": self.total_narratives_generated,
                "engine_version": "1.0.0",
                "analysis_type": "trade_explainability"
            }
        }
    
    def _store_narrative(self, response: Dict):
        """Store trade narrative for future reference."""
        try:
            self.trade_narratives.append({
                'timestamp': datetime.now(),
                'trade_id': response.get('trade_id'),
                'narrative': response.get('complete_narrative', {}),
                'trade_summary': response.get('trade_summary', {})
            })
            
        except Exception:
            pass  # Silent fail for narrative storage
    
    def get_explainability_stats(self) -> Dict[str, Any]:
        """Get comprehensive explainability and monitoring statistics."""
        return {
            "total_narratives_generated": self.total_narratives_generated,
            "calibration_updates": self.calibration_updates,
            "drift_detections": self.drift_detections,
            "edge_decay_alerts": self.edge_decay_alerts,
            "active_calibration_models": len(self.calibration_data),
            "monitored_edges": len(self.edge_monitoring),
            "monitored_drift_models": len(self.drift_monitoring),
            "calibration_status": {
                model_key: {
                    'sample_count': len(data['predictions']),
                    'last_update': data['last_update'].isoformat() if data['last_update'] else 'never',
                    'calibration_error': data.get('calibration_curve', {}).get('calibration_error', 'unknown')
                }
                for model_key, data in self.calibration_data.items()
                if len(data['predictions']) >= 10
            },
            "recent_alerts": [
                {
                    'timestamp': alert['timestamp'].isoformat(),
                    'type': alert['type'],
                    'severity': alert['severity'],
                    'message': alert['message']
                }
                for alert in list(self.alerts)[-10:]  # Last 10 alerts
            ],
            "critical_alerts": [
                {
                    'timestamp': alert['timestamp'].isoformat(),
                    'type': alert['type'],
                    'severity': alert['severity'],
                    'message': alert['message']
                }
                for alert in list(self.critical_alerts)[-5:]  # Last 5 critical alerts
            ],
            "memory_sizes": {
                "trade_narratives": len(self.trade_narratives),
                "alerts": len(self.alerts),
                "critical_alerts": len(self.critical_alerts)
            },
            "narrative_components": self.narrative_components,
            "engine_version": "1.0.0"
        }
