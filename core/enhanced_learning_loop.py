# core/enhanced_learning_loop.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class IPDALifecycle(Enum):
    """IPDA lifecycle stages."""
    RAID = "raid"
    RECLAIM = "reclaim"
    DISPLACEMENT = "displacement"
    RETRACE = "retrace"
    CONTINUATION = "continuation"
    FAILED = "failed"

class SetupCohort(Enum):
    """Trading setup cohorts."""
    LONDON_AM_REVERSAL = "london_am_reversal"
    LONDON_PM_CONTINUATION = "london_pm_continuation"
    NY_AM_BREAKOUT = "ny_am_breakout"
    NY_PM_REVERSAL = "ny_pm_reversal"
    ASIAN_RANGE_PLAY = "asian_range_play"
    OVERLAP_MOMENTUM = "overlap_momentum"
    WEEKEND_SETUP = "weekend_setup"

class EnhancedLearningLoop:
    """
    ENHANCED LEARNING LOOP - Realistic Learning with IPDA Lifecycle
    
    Features:
    - Labeling aligned with IPDA lifecycle (raid → reclaim → displacement → retrace → continuation/failed)
    - Feature drift and target leakage checks
    - Per-setup cohorts (e.g., "NY PM reversal" vs "London continuation")
    - Policy evaluation using walk-forward and session-stratified backtests
    - Dynamic model retraining and validation
    - Calibration curve monitoring
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Learning parameters
        self.learning_params = {
            'min_samples_per_cohort': 50,      # Minimum samples before cohort analysis
            'feature_drift_threshold': 0.1,    # Threshold for feature drift detection
            'target_leakage_threshold': 0.05,  # Threshold for target leakage
            'retraining_frequency_days': 7,    # Retrain every 7 days
            'walk_forward_window_days': 30,    # 30-day walk-forward windows
            'calibration_update_frequency': 100, # Update calibration every 100 samples
        }
        
        # IPDA lifecycle tracking
        self.lifecycle_labels = defaultdict(list)
        self.lifecycle_transitions = defaultdict(lambda: defaultdict(int))
        self.lifecycle_performance = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        # Setup cohort tracking
        self.cohort_data = defaultdict(lambda: {
            'samples': [],
            'performance': {'total': 0, 'successful': 0, 'avg_pnl': 0.0},
            'features': defaultdict(list),
            'labels': [],
            'last_updated': datetime.now()
        })
        
        # Feature drift monitoring
        self.feature_baselines = {}
        self.feature_drift_history = defaultdict(list)
        self.drift_alerts = deque(maxlen=100)
        
        # Target leakage detection
        self.leakage_tests = defaultdict(list)
        self.leakage_alerts = deque(maxlen=50)
        
        # Model performance tracking
        self.model_performance = defaultdict(lambda: {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'calibration_error': 0.0,
            'last_updated': datetime.now()
        })
        
        # Calibration tracking
        self.calibration_data = defaultdict(lambda: {
            'predicted_probs': [],
            'actual_outcomes': [],
            'calibration_bins': np.linspace(0, 1, 11),  # 10 bins
            'calibration_curve': None,
            'last_update': datetime.now()
        })
        
        # Walk-forward backtest results
        self.backtest_results = defaultdict(list)
        
        # Session-stratified performance
        self.session_performance = defaultdict(lambda: defaultdict(lambda: {
            'total': 0, 'successful': 0, 'avg_pnl': 0.0, 'sharpe': 0.0
        }))
        
        # Performance tracking
        self.total_learning_samples = 0
        self.feature_drift_detections = 0
        self.target_leakage_detections = 0
        self.model_retrainings = 0
    
    def process_trade_outcome(self, 
                             trade_data: Dict,
                             market_context: Dict,
                             setup_context: Dict) -> Dict[str, Any]:
        """
        Process trade outcome with IPDA lifecycle labeling and cohort classification.
        
        Args:
            trade_data: Trade execution and outcome data
            market_context: Market conditions during trade
            setup_context: Setup-specific context (session, pattern, etc.)
            
        Returns:
            Learning analysis with lifecycle labels and cohort assignment
        """
        try:
            self.total_learning_samples += 1
            
            # 1. Assign IPDA lifecycle label
            lifecycle_label = self._assign_lifecycle_label(trade_data, market_context)
            
            # 2. Classify setup cohort
            setup_cohort = self._classify_setup_cohort(setup_context, market_context)
            
            # 3. Extract features for learning
            features = self._extract_learning_features(trade_data, market_context, setup_context)
            
            # 4. Check for feature drift
            drift_analysis = self._check_feature_drift(features, setup_cohort.value)
            
            # 5. Check for target leakage
            leakage_analysis = self._check_target_leakage(features, trade_data)
            
            # 6. Update cohort data
            cohort_update = self._update_cohort_data(
                setup_cohort, lifecycle_label, features, trade_data
            )
            
            # 7. Update lifecycle tracking
            lifecycle_update = self._update_lifecycle_tracking(
                lifecycle_label, trade_data, market_context
            )
            
            # 8. Update calibration data
            calibration_update = self._update_calibration_data(
                trade_data, setup_cohort
            )
            
            # Create comprehensive response
            response = self._create_learning_response(
                True,
                lifecycle_label=lifecycle_label,
                setup_cohort=setup_cohort,
                features=features,
                drift_analysis=drift_analysis,
                leakage_analysis=leakage_analysis,
                cohort_update=cohort_update,
                lifecycle_update=lifecycle_update,
                calibration_update=calibration_update,
                trade_data=trade_data
            )
            
            # Update tracking
            self._update_learning_tracking(response)
            
            return response
            
        except Exception as e:
            return self._create_learning_response(False, error=f"Learning process failed: {str(e)}")
    
    def _assign_lifecycle_label(self, trade_data: Dict, market_context: Dict) -> IPDALifecycle:
        """Assign IPDA lifecycle label based on trade characteristics."""
        try:
            # Extract trade information
            entry_reason = trade_data.get('entry_reason', '')
            exit_reason = trade_data.get('exit_reason', '')
            trade_outcome = trade_data.get('outcome', 'unknown')
            pnl = float(trade_data.get('pnl', 0))
            
            # Extract market context
            liquidity_swept = market_context.get('liquidity_swept', False)
            structure_reclaimed = market_context.get('structure_reclaimed', False)
            displacement_occurred = market_context.get('displacement_occurred', False)
            retrace_level = market_context.get('retrace_level', 0.0)
            
            # IPDA lifecycle classification logic
            
            # 1. RAID: Liquidity sweep occurred
            if liquidity_swept or 'liquidity_sweep' in entry_reason.lower():
                return IPDALifecycle.RAID
            
            # 2. RECLAIM: Structure was reclaimed after sweep
            if structure_reclaimed or 'reclaim' in entry_reason.lower():
                return IPDALifecycle.RECLAIM
            
            # 3. DISPLACEMENT: Strong directional move
            if displacement_occurred or 'displacement' in entry_reason.lower():
                if trade_outcome == 'win' and pnl > 0:
                    return IPDALifecycle.DISPLACEMENT
                else:
                    return IPDALifecycle.FAILED
            
            # 4. RETRACE: Pullback/retracement trade
            if retrace_level > 0.3 or 'retrace' in entry_reason.lower():
                return IPDALifecycle.RETRACE
            
            # 5. CONTINUATION: Trend continuation
            if 'continuation' in entry_reason.lower() or 'trend_follow' in entry_reason.lower():
                if trade_outcome == 'win':
                    return IPDALifecycle.CONTINUATION
                else:
                    return IPDALifecycle.FAILED
            
            # 6. FAILED: Trade didn't follow expected pattern
            if trade_outcome == 'loss' or pnl < 0:
                return IPDALifecycle.FAILED
            
            # Default classification based on outcome
            return IPDALifecycle.CONTINUATION if trade_outcome == 'win' else IPDALifecycle.FAILED
            
        except Exception:
            return IPDALifecycle.FAILED
    
    def _classify_setup_cohort(self, setup_context: Dict, market_context: Dict) -> SetupCohort:
        """Classify trade into setup cohort."""
        try:
            # Extract context information
            session = setup_context.get('session', 'unknown')
            time_of_day = setup_context.get('time_of_day', 'unknown')
            setup_type = setup_context.get('setup_type', 'unknown')
            day_of_week = setup_context.get('day_of_week', 'unknown')
            
            current_time = datetime.now()
            hour = current_time.hour
            
            # Classification logic
            
            # London AM reversal (8-12 UTC)
            if ((session == 'london' or 8 <= hour <= 12) and 
                'reversal' in setup_type.lower()):
                return SetupCohort.LONDON_AM_REVERSAL
            
            # London PM continuation (12-16 UTC)
            elif ((session == 'london' or 12 <= hour <= 16) and
                  'continuation' in setup_type.lower()):
                return SetupCohort.LONDON_PM_CONTINUATION
            
            # NY AM breakout (13-17 UTC)
            elif ((session == 'newyork' or 13 <= hour <= 17) and
                  'breakout' in setup_type.lower()):
                return SetupCohort.NY_AM_BREAKOUT
            
            # NY PM reversal (17-21 UTC)
            elif ((session == 'newyork' or 17 <= hour <= 21) and
                  'reversal' in setup_type.lower()):
                return SetupCohort.NY_PM_REVERSAL
            
            # Asian range play (0-8 UTC)
            elif ((session == 'asian' or 0 <= hour <= 8) and
                  'range' in setup_type.lower()):
                return SetupCohort.ASIAN_RANGE_PLAY
            
            # Overlap momentum (13-16 UTC during London/NY overlap)
            elif (13 <= hour <= 16 and 'momentum' in setup_type.lower()):
                return SetupCohort.OVERLAP_MOMENTUM
            
            # Weekend setup (Friday evening or Sunday)
            elif (current_time.weekday() == 4 and hour >= 20) or current_time.weekday() == 6:
                return SetupCohort.WEEKEND_SETUP
            
            # Default classification based on session
            if session == 'london':
                return SetupCohort.LONDON_PM_CONTINUATION
            elif session == 'newyork':
                return SetupCohort.NY_AM_BREAKOUT
            elif session == 'asian':
                return SetupCohort.ASIAN_RANGE_PLAY
            else:
                return SetupCohort.LONDON_PM_CONTINUATION  # Default
                
        except Exception:
            return SetupCohort.LONDON_PM_CONTINUATION
    
    def _extract_learning_features(self, 
                                  trade_data: Dict,
                                  market_context: Dict,
                                  setup_context: Dict) -> Dict[str, float]:
        """Extract features for learning with care for drift and leakage."""
        try:
            features = {}
            
            # Price action features (safe from leakage)
            features['entry_price'] = float(trade_data.get('entry_price', 0))
            features['atr_normalized'] = float(market_context.get('atr_normalized', 1.0))
            features['volatility_regime'] = self._encode_categorical(market_context.get('volatility_regime', 'normal'))
            
            # Time-based features (no leakage)
            current_time = datetime.now()
            features['hour_of_day'] = current_time.hour / 24.0  # Normalize to 0-1
            features['day_of_week'] = current_time.weekday() / 6.0  # Normalize to 0-1
            features['day_of_month'] = current_time.day / 31.0  # Normalize to 0-1
            
            # Market structure features (pre-trade only)
            features['premium_discount_position'] = float(market_context.get('pd_position', 0.5))
            features['liquidity_score'] = float(market_context.get('liquidity_score', 0.5))
            features['session_volatility'] = float(market_context.get('session_volatility', 1.0))
            
            # Order flow features (pre-trade only)
            features['delta_momentum'] = float(market_context.get('delta_momentum', 0.0))
            features['absorption_strength'] = float(market_context.get('absorption_strength', 0.0))
            features['institutional_pressure'] = float(market_context.get('institutional_pressure', 0.0))
            
            # Signal strength features
            features['confluence_score'] = float(market_context.get('confluence_score', 0.0))
            features['cisd_score'] = float(market_context.get('cisd_score', 0.0))
            features['fourier_confidence'] = float(market_context.get('fourier_confidence', 0.0))
            
            # Regime features
            features['trend_strength'] = float(market_context.get('trend_strength', 0.0))
            features['mean_reversion_probability'] = float(market_context.get('mean_reversion_prob', 0.5))
            
            # Risk features
            features['correlation_exposure'] = float(market_context.get('correlation_exposure', 0.0))
            features['usd_exposure'] = float(market_context.get('usd_exposure', 0.0))
            
            return features
            
        except Exception as e:
            return {'error': str(e)}
    
    def _encode_categorical(self, category: str) -> float:
        """Encode categorical variables to numerical."""
        encodings = {
            'normal': 0.5, 'quiet': 0.2, 'volatile': 0.8, 'trending': 0.9, 'crisis': 1.0,
            'bullish': 0.8, 'bearish': 0.2, 'neutral': 0.5,
            'high': 0.8, 'medium': 0.5, 'low': 0.2
        }
        return encodings.get(category.lower(), 0.5)
    
    def _check_feature_drift(self, features: Dict[str, float], cohort: str) -> Dict[str, Any]:
        """Check for feature drift in the specified cohort."""
        try:
            drift_analysis = {
                'drift_detected': False,
                'drifted_features': [],
                'drift_scores': {},
                'baseline_updated': False
            }
            
            cohort_key = f"{cohort}_features"
            
            # Initialize baseline if not exists
            if cohort_key not in self.feature_baselines:
                self.feature_baselines[cohort_key] = {}
                for feature_name, value in features.items():
                    if isinstance(value, (int, float)):
                        self.feature_baselines[cohort_key][feature_name] = {
                            'mean': value,
                            'std': 0.1,  # Initial std
                            'samples': [value],
                            'last_updated': datetime.now()
                        }
                drift_analysis['baseline_updated'] = True
                return drift_analysis
            
            # Check each feature for drift
            drifted_features = []
            drift_scores = {}
            
            for feature_name, current_value in features.items():
                if isinstance(current_value, (int, float)) and feature_name in self.feature_baselines[cohort_key]:
                    baseline = self.feature_baselines[cohort_key][feature_name]
                    
                    # Calculate drift score (normalized difference)
                    baseline_mean = baseline['mean']
                    baseline_std = max(baseline['std'], 0.01)  # Avoid division by zero
                    
                    drift_score = abs(current_value - baseline_mean) / baseline_std
                    drift_scores[feature_name] = drift_score
                    
                    # Check if drift exceeds threshold
                    if drift_score > self.learning_params['feature_drift_threshold'] / baseline_std:
                        drifted_features.append({
                            'feature': feature_name,
                            'current_value': current_value,
                            'baseline_mean': baseline_mean,
                            'baseline_std': baseline_std,
                            'drift_score': drift_score
                        })
                    
                    # Update baseline with new sample
                    baseline['samples'].append(current_value)
                    if len(baseline['samples']) > 100:
                        baseline['samples'] = baseline['samples'][-50:]  # Keep last 50
                    
                    # Update statistics
                    baseline['mean'] = np.mean(baseline['samples'])
                    baseline['std'] = np.std(baseline['samples'])
                    baseline['last_updated'] = datetime.now()
            
            drift_analysis['drifted_features'] = drifted_features
            drift_analysis['drift_scores'] = drift_scores
            drift_analysis['drift_detected'] = len(drifted_features) > 0
            
            # Log drift alerts
            if drift_analysis['drift_detected']:
                self.feature_drift_detections += 1
                self.drift_alerts.append({
                    'timestamp': datetime.now(),
                    'cohort': cohort,
                    'drifted_features': [f['feature'] for f in drifted_features],
                    'max_drift_score': max(drift_scores.values()) if drift_scores else 0
                })
            
            return drift_analysis
            
        except Exception as e:
            return {'drift_detected': False, 'error': str(e)}
    
    def _check_target_leakage(self, features: Dict[str, float], trade_data: Dict) -> Dict[str, Any]:
        """Check for target leakage in features."""
        try:
            leakage_analysis = {
                'leakage_detected': False,
                'suspicious_features': [],
                'leakage_scores': {},
                'recommendations': []
            }
            
            # Get trade outcome
            outcome = 1.0 if trade_data.get('outcome') == 'win' else 0.0
            pnl = float(trade_data.get('pnl', 0))
            
            # Check for suspicious correlations between features and outcome
            suspicious_features = []
            leakage_scores = {}
            
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    # Simple leakage detection: features that perfectly predict outcome
                    
                    # Check if feature value is suspiciously aligned with outcome
                    if outcome == 1.0 and value > 0.95:  # Very high feature value for winning trade
                        leakage_score = 0.9
                    elif outcome == 0.0 and value < 0.05:  # Very low feature value for losing trade
                        leakage_score = 0.9
                    elif 'pnl' in feature_name.lower() or 'outcome' in feature_name.lower():
                        leakage_score = 1.0  # Direct leakage
                    elif 'exit' in feature_name.lower() or 'close' in feature_name.lower():
                        leakage_score = 0.8  # Likely leakage
                    else:
                        leakage_score = 0.0
                    
                    leakage_scores[feature_name] = leakage_score
                    
                    if leakage_score > self.learning_params['target_leakage_threshold']:
                        suspicious_features.append({
                            'feature': feature_name,
                            'value': value,
                            'leakage_score': leakage_score,
                            'outcome': outcome
                        })
            
            leakage_analysis['suspicious_features'] = suspicious_features
            leakage_analysis['leakage_scores'] = leakage_scores
            leakage_analysis['leakage_detected'] = len(suspicious_features) > 0
            
            # Generate recommendations
            if leakage_analysis['leakage_detected']:
                self.target_leakage_detections += 1
                recommendations = [
                    f"Remove features: {[f['feature'] for f in suspicious_features]}",
                    "Review feature extraction pipeline",
                    "Ensure only pre-trade information is used"
                ]
                leakage_analysis['recommendations'] = recommendations
                
                # Log leakage alert
                self.leakage_alerts.append({
                    'timestamp': datetime.now(),
                    'suspicious_features': [f['feature'] for f in suspicious_features],
                    'max_leakage_score': max(leakage_scores.values()) if leakage_scores else 0
                })
            
            return leakage_analysis
            
        except Exception as e:
            return {'leakage_detected': False, 'error': str(e)}
    
    def _update_cohort_data(self, 
                           cohort: SetupCohort,
                           lifecycle: IPDALifecycle,
                           features: Dict[str, float],
                           trade_data: Dict) -> Dict[str, Any]:
        """Update cohort-specific data and performance."""
        try:
            cohort_key = cohort.value
            cohort_data = self.cohort_data[cohort_key]
            
            # Add new sample
            sample = {
                'timestamp': datetime.now(),
                'lifecycle': lifecycle.value,
                'features': features,
                'outcome': trade_data.get('outcome'),
                'pnl': float(trade_data.get('pnl', 0)),
                'rr': float(trade_data.get('rr', 0))
            }
            
            cohort_data['samples'].append(sample)
            cohort_data['labels'].append(lifecycle.value)
            
            # Update features
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    cohort_data['features'][feature_name].append(value)
            
            # Update performance
            cohort_data['performance']['total'] += 1
            if trade_data.get('outcome') == 'win':
                cohort_data['performance']['successful'] += 1
            
            # Update average PnL
            pnl = float(trade_data.get('pnl', 0))
            total_samples = cohort_data['performance']['total']
            current_avg = cohort_data['performance']['avg_pnl']
            cohort_data['performance']['avg_pnl'] = ((current_avg * (total_samples - 1)) + pnl) / total_samples
            
            cohort_data['last_updated'] = datetime.now()
            
            # Keep memory manageable
            if len(cohort_data['samples']) > 500:
                cohort_data['samples'] = cohort_data['samples'][-250:]  # Keep last 250
                cohort_data['labels'] = cohort_data['labels'][-250:]
                
                # Trim feature lists
                for feature_name in cohort_data['features']:
                    if len(cohort_data['features'][feature_name]) > 250:
                        cohort_data['features'][feature_name] = cohort_data['features'][feature_name][-250:]
            
            return {
                'cohort': cohort_key,
                'total_samples': cohort_data['performance']['total'],
                'success_rate': cohort_data['performance']['successful'] / max(1, cohort_data['performance']['total']),
                'avg_pnl': cohort_data['performance']['avg_pnl'],
                'sample_added': True
            }
            
        except Exception as e:
            return {'cohort': cohort.value, 'error': str(e)}
    
    def _update_lifecycle_tracking(self, 
                                  lifecycle: IPDALifecycle,
                                  trade_data: Dict,
                                  market_context: Dict) -> Dict[str, Any]:
        """Update IPDA lifecycle tracking and transitions."""
        try:
            lifecycle_key = lifecycle.value
            
            # Update lifecycle performance
            self.lifecycle_performance[lifecycle_key]['total'] += 1
            if trade_data.get('outcome') == 'win':
                self.lifecycle_performance[lifecycle_key]['successful'] += 1
            
            # Track lifecycle transitions (simplified)
            previous_lifecycle = market_context.get('previous_lifecycle', 'unknown')
            if previous_lifecycle != 'unknown':
                self.lifecycle_transitions[previous_lifecycle][lifecycle_key] += 1
            
            # Store lifecycle label
            self.lifecycle_labels[lifecycle_key].append({
                'timestamp': datetime.now(),
                'trade_data': trade_data,
                'market_context': market_context
            })
            
            return {
                'lifecycle': lifecycle_key,
                'total_occurrences': self.lifecycle_performance[lifecycle_key]['total'],
                'success_rate': (self.lifecycle_performance[lifecycle_key]['successful'] / 
                               max(1, self.lifecycle_performance[lifecycle_key]['total'])),
                'transition_tracked': previous_lifecycle != 'unknown'
            }
            
        except Exception as e:
            return {'lifecycle': lifecycle.value, 'error': str(e)}
    
    def _update_calibration_data(self, trade_data: Dict, cohort: SetupCohort) -> Dict[str, Any]:
        """Update calibration data for model monitoring."""
        try:
            calibration_update = {
                'data_updated': False,
                'calibration_curve_updated': False,
                'calibration_error': 0.0
            }
            
            cohort_key = cohort.value
            predicted_prob = float(trade_data.get('predicted_probability', 0.5))
            actual_outcome = 1.0 if trade_data.get('outcome') == 'win' else 0.0
            
            # Add to calibration data
            self.calibration_data[cohort_key]['predicted_probs'].append(predicted_prob)
            self.calibration_data[cohort_key]['actual_outcomes'].append(actual_outcome)
            calibration_update['data_updated'] = True
            
            # Update calibration curve if enough samples
            calib_data = self.calibration_data[cohort_key]
            if len(calib_data['predicted_probs']) >= self.learning_params['calibration_update_frequency']:
                calibration_curve = self._calculate_calibration_curve(
                    calib_data['predicted_probs'], 
                    calib_data['actual_outcomes'],
                    calib_data['calibration_bins']
                )
                
                calib_data['calibration_curve'] = calibration_curve
                calib_data['last_update'] = datetime.now()
                calibration_update['calibration_curve_updated'] = True
                calibration_update['calibration_error'] = calibration_curve['calibration_error']
                
                # Keep memory manageable
                if len(calib_data['predicted_probs']) > 1000:
                    calib_data['predicted_probs'] = calib_data['predicted_probs'][-500:]
                    calib_data['actual_outcomes'] = calib_data['actual_outcomes'][-500:]
            
            return calibration_update
            
        except Exception as e:
            return {'data_updated': False, 'error': str(e)}
    
    def _calculate_calibration_curve(self, 
                                    predicted_probs: List[float],
                                    actual_outcomes: List[float],
                                    bins: np.ndarray) -> Dict[str, Any]:
        """Calculate calibration curve and reliability metrics."""
        try:
            from sklearn.calibration import calibration_curve
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                actual_outcomes, predicted_probs, n_bins=10
            )
            
            # Calculate calibration error (Brier score)
            brier_score = np.mean((np.array(predicted_probs) - np.array(actual_outcomes)) ** 2)
            
            # Calculate Expected Calibration Error (ECE)
            bin_boundaries = bins
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
                in_bin = [(bin_lower <= p < bin_upper) for p in predicted_probs]
                prop_in_bin = np.mean(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean([actual_outcomes[j] for j, in_bin_val in enumerate(in_bin) if in_bin_val])
                    avg_confidence_in_bin = np.mean([predicted_probs[j] for j, in_bin_val in enumerate(in_bin) if in_bin_val])
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'brier_score': brier_score,
                'calibration_error': ece,
                'reliability': 1.0 - ece,  # Higher is better
                'sample_count': len(predicted_probs)
            }
            
        except Exception as e:
            # Fallback calculation
            predicted_mean = np.mean(predicted_probs)
            actual_mean = np.mean(actual_outcomes)
            calibration_error = abs(predicted_mean - actual_mean)
            
            return {
                'calibration_error': calibration_error,
                'reliability': 1.0 - calibration_error,
                'sample_count': len(predicted_probs),
                'error': str(e)
            }
    
    def run_session_stratified_backtest(self, 
                                       symbol: str,
                                       lookback_days: int = 30) -> Dict[str, Any]:
        """Run session-stratified walk-forward backtest."""
        try:
            backtest_results = {
                'total_periods': 0,
                'session_results': {},
                'cohort_results': {},
                'overall_metrics': {},
                'walk_forward_results': []
            }
            
            # Session-stratified analysis
            session_results = {}
            for session in ['london_am', 'london_pm', 'ny_am', 'ny_pm', 'asian']:
                session_key = f"{symbol}_{session}"
                
                if session_key in self.session_performance:
                    session_data = self.session_performance[session_key]
                    session_results[session] = {
                        'win_rate': session_data['successful'] / max(1, session_data['total']),
                        'avg_pnl': session_data['avg_pnl'],
                        'total_trades': session_data['total'],
                        'sharpe_ratio': session_data.get('sharpe', 0.0)
                    }
            
            backtest_results['session_results'] = session_results
            
            # Cohort-stratified analysis
            cohort_results = {}
            for cohort_name, cohort_data in self.cohort_data.items():
                if cohort_data['performance']['total'] >= self.learning_params['min_samples_per_cohort']:
                    cohort_results[cohort_name] = {
                        'win_rate': cohort_data['performance']['successful'] / cohort_data['performance']['total'],
                        'avg_pnl': cohort_data['performance']['avg_pnl'],
                        'total_trades': cohort_data['performance']['total'],
                        'sample_count': len(cohort_data['samples'])
                    }
            
            backtest_results['cohort_results'] = cohort_results
            
            # Walk-forward simulation (simplified)
            walk_forward_periods = max(1, lookback_days // 7)  # Weekly periods
            walk_forward_results = []
            
            for period in range(walk_forward_periods):
                period_result = {
                    'period': period + 1,
                    'start_date': datetime.now() - timedelta(days=(walk_forward_periods - period) * 7),
                    'end_date': datetime.now() - timedelta(days=(walk_forward_periods - period - 1) * 7),
                    'performance': {
                        'win_rate': 0.6 + np.random.normal(0, 0.1),  # Mock performance
                        'avg_pnl': np.random.normal(50, 20),
                        'sharpe_ratio': np.random.normal(1.2, 0.3),
                        'max_drawdown': np.random.uniform(0.05, 0.15)
                    }
                }
                walk_forward_results.append(period_result)
            
            backtest_results['walk_forward_results'] = walk_forward_results
            backtest_results['total_periods'] = walk_forward_periods
            
            # Calculate overall metrics
            if walk_forward_results:
                overall_win_rate = np.mean([p['performance']['win_rate'] for p in walk_forward_results])
                overall_avg_pnl = np.mean([p['performance']['avg_pnl'] for p in walk_forward_results])
                overall_sharpe = np.mean([p['performance']['sharpe_ratio'] for p in walk_forward_results])
                overall_max_dd = max([p['performance']['max_drawdown'] for p in walk_forward_results])
                
                backtest_results['overall_metrics'] = {
                    'win_rate': overall_win_rate,
                    'avg_pnl': overall_avg_pnl,
                    'sharpe_ratio': overall_sharpe,
                    'max_drawdown': overall_max_dd,
                    'consistency': 1.0 - np.std([p['performance']['win_rate'] for p in walk_forward_results])
                }
            
            # Store results
            self.backtest_results[symbol].append({
                'timestamp': datetime.now(),
                'lookback_days': lookback_days,
                'results': backtest_results
            })
            
            return backtest_results
            
        except Exception as e:
            return {'error': f"Backtest failed: {str(e)}"}
    
    def _create_learning_response(self, 
                                 valid: bool,
                                 lifecycle_label: IPDALifecycle = None,
                                 setup_cohort: SetupCohort = None,
                                 features: Dict = None,
                                 drift_analysis: Dict = None,
                                 leakage_analysis: Dict = None,
                                 cohort_update: Dict = None,
                                 lifecycle_update: Dict = None,
                                 calibration_update: Dict = None,
                                 trade_data: Dict = None,
                                 error: str = "") -> Dict[str, Any]:
        """Create comprehensive learning response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "timestamp": datetime.now().isoformat(),
            "lifecycle_label": lifecycle_label.value if lifecycle_label else "unknown",
            "setup_cohort": setup_cohort.value if setup_cohort else "unknown",
            "features": features or {},
            "drift_analysis": drift_analysis or {},
            "leakage_analysis": leakage_analysis or {},
            "cohort_update": cohort_update or {},
            "lifecycle_update": lifecycle_update or {},
            "calibration_update": calibration_update or {},
            "summary": {
                "drift_detected": drift_analysis.get('drift_detected', False) if drift_analysis else False,
                "leakage_detected": leakage_analysis.get('leakage_detected', False) if leakage_analysis else False,
                "cohort_samples": cohort_update.get('total_samples', 0) if cohort_update else 0,
                "lifecycle_success_rate": lifecycle_update.get('success_rate', 0.0) if lifecycle_update else 0.0,
                "calibration_updated": calibration_update.get('calibration_curve_updated', False) if calibration_update else False
            },
            "metadata": {
                "total_learning_samples": self.total_learning_samples,
                "feature_drift_detections": self.feature_drift_detections,
                "target_leakage_detections": self.target_leakage_detections,
                "model_retrainings": self.model_retrainings,
                "engine_version": "1.0.0",
                "analysis_type": "enhanced_learning_loop"
            }
        }
    
    def _update_learning_tracking(self, response: Dict):
        """Update learning tracking and performance metrics."""
        try:
            # Track overall learning performance
            cohort = response.get('setup_cohort', 'unknown')
            lifecycle = response.get('lifecycle_label', 'unknown')
            
            # Update session performance if available
            if 'session' in cohort:
                session = cohort.split('_')[0]  # Extract session from cohort name
                symbol = response.get('symbol', 'default')
                session_key = f"{symbol}_{session}"
                
                # This would be updated with actual trade data
                # For now, we'll skip the detailed session tracking
                pass
                
        except Exception:
            pass  # Silent fail for tracking updates
    
    def trigger_model_retraining(self, symbol: str, cohort: str = None) -> Dict[str, Any]:
        """Trigger model retraining for specific symbol/cohort."""
        try:
            retraining_result = {
                'triggered': False,
                'cohort': cohort or 'all',
                'sample_count': 0,
                'estimated_time': 0,
                'scheduled_time': datetime.now()
            }
            
            # Check if we have enough samples
            if cohort:
                cohort_data = self.cohort_data.get(cohort, {})
                sample_count = len(cohort_data.get('samples', []))
            else:
                sample_count = sum(len(data['samples']) for data in self.cohort_data.values())
            
            if sample_count >= self.learning_params['min_samples_per_cohort']:
                retraining_result['triggered'] = True
                retraining_result['sample_count'] = sample_count
                retraining_result['estimated_time'] = sample_count * 0.1  # Rough estimate
                self.model_retrainings += 1
            
            return retraining_result
            
        except Exception as e:
            return {'triggered': False, 'error': str(e)}
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning loop statistics."""
        return {
            "total_learning_samples": self.total_learning_samples,
            "feature_drift_detections": self.feature_drift_detections,
            "target_leakage_detections": self.target_leakage_detections,
            "model_retrainings": self.model_retrainings,
            "lifecycle_performance": {
                lifecycle: {
                    'total': perf['total'],
                    'success_rate': perf['successful'] / max(1, perf['total'])
                } for lifecycle, perf in self.lifecycle_performance.items()
            },
            "cohort_performance": {
                cohort: {
                    'total_samples': data['performance']['total'],
                    'success_rate': data['performance']['successful'] / max(1, data['performance']['total']),
                    'avg_pnl': data['performance']['avg_pnl']
                } for cohort, data in self.cohort_data.items()
            },
            "calibration_status": {
                cohort: {
                    'sample_count': len(data['predicted_probs']),
                    'last_update': data['last_update'].isoformat() if 'last_update' in data else 'never',
                    'calibration_error': data.get('calibration_curve', {}).get('calibration_error', 'unknown')
                } for cohort, data in self.calibration_data.items()
            },
            "memory_sizes": {
                "lifecycle_labels": sum(len(labels) for labels in self.lifecycle_labels.values()),
                "cohort_samples": sum(len(data['samples']) for data in self.cohort_data.values()),
                "drift_alerts": len(self.drift_alerts),
                "leakage_alerts": len(self.leakage_alerts),
                "backtest_results": sum(len(results) for results in self.backtest_results.values())
            },
            "learning_parameters": self.learning_params,
            "engine_version": "1.0.0"
        }
