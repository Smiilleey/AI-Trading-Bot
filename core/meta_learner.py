import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import joblib
import os

class MetaLearner:
    """
    Meta-learner that combines and ranks signals from multiple sources with uncertainty quantification.
    
    Sources:
    - ML predictions (confidence scores)
    - Rule-based signals (CISD, structure, etc.)
    - Order flow analysis (delta, absorption, institutional activity)
    - Market regime context
    
    Features IPDA and top-down analysis integration:
    - Higher timeframe context for signal validation
    - Institutional accumulation/distribution phases
    - Liquidity hunting and stop run detection
    - Multi-timeframe alignment scoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.ensemble_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # IPDA and top-down analysis parameters
        self.timeframe_weights = {
            'D': 0.4,    # Daily - highest weight for overall structure
            '4H': 0.3,   # 4H - trend confirmation
            '1H': 0.2,   # 1H - entry timing
            '15M': 0.1   # 15M - precise execution
        }
        
        self.ipda_phases = {
            'accumulation': 0.8,    # Strong buy signal
            'manipulation': 0.3,    # Weak signal (false moves)
            'distribution': -0.8,   # Strong sell signal
            'markup': 0.6,         # Trend continuation
            'markdown': -0.6       # Trend continuation
        }
        
        # Model persistence
        self.model_dir = "memory/meta_learner"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self, features_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the meta-learner ensemble on historical signal data.
        
        Args:
            features_df: Features from all signal sources
            outcomes_df: Trade outcomes (PnL, RR, etc.)
            
        Returns:
            Dictionary with model performance metrics
        """
        try:
            self.logger.info("Training meta-learner ensemble...")
            
            # Prepare training data
            X = features_df.drop(['timestamp', 'symbol'], axis=1, errors='ignore')
            y = outcomes_df['pnl'].values if 'pnl' in outcomes_df.columns else outcomes_df['outcome'].values
            
            # Handle missing values
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train ensemble models
            model_scores = {}
            for name, model in self.ensemble_models.items():
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                model_scores[name] = score
                self.logger.info(f"{name} model R² score: {score:.4f}")
            
            # Save models
            self._save_models()
            self.is_trained = True
            
            return {
                'ensemble_scores': model_scores,
                'overall_score': np.mean(list(model_scores.values())),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            self.logger.error(f"Error training meta-learner: {e}")
            return {'error': str(e)}
    
    def predict_signal_ranking(self, 
                              signal_data: Dict[str, Any],
                              market_context: Dict[str, Any],
                              multi_timeframe_data: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Rank and combine signals from multiple sources with uncertainty quantification.
        
        Args:
            signal_data: Dictionary with signals from different sources
            market_context: Current market regime and conditions
            multi_timeframe_data: Data from multiple timeframes for top-down analysis
            
        Returns:
            Dictionary with ranked signals, confidence scores, and uncertainty
        """
        try:
            if not self.is_trained:
                return self._fallback_ranking(signal_data, market_context)
            
            # Extract features from all signal sources
            features = self._extract_meta_features(signal_data, market_context, multi_timeframe_data)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get predictions from all models
            predictions = {}
            uncertainties = {}
            
            for name, model in self.ensemble_models.items():
                pred = model.predict(features_scaled)[0]
                predictions[name] = pred
                
                # Calculate uncertainty (for tree-based models, use prediction variance)
                if hasattr(model, 'estimators_'):
                    # Random Forest: use std of individual tree predictions
                    tree_preds = [tree.predict(features_scaled)[0] for tree in model.estimators_]
                    uncertainties[name] = np.std(tree_preds)
                else:
                    # For other models, use a simple heuristic
                    uncertainties[name] = abs(pred) * 0.1
            
            # Ensemble prediction with uncertainty weighting
            weights = self._calculate_model_weights(predictions, uncertainties)
            final_prediction = sum(pred * weights[name] for name, pred in predictions.items())
            final_uncertainty = sum(unc * weights[name] for name, unc in uncertainties.items())
            
            # Apply IPDA and top-down analysis adjustments
            adjusted_prediction = self._apply_ipda_adjustments(
                final_prediction, signal_data, market_context, multi_timeframe_data
            )
            
            # Generate signal ranking
            signal_ranking = self._rank_signals(signal_data, adjusted_prediction, final_uncertainty)
            
            return {
                'final_prediction': adjusted_prediction,
                'uncertainty': final_uncertainty,
                'confidence': max(0, 1 - final_uncertainty),
                'signal_ranking': signal_ranking,
                'model_predictions': predictions,
                'model_uncertainties': uncertainties,
                'weights': weights,
                'ipda_phase': self._detect_ipda_phase(signal_data, market_context),
                'timeframe_alignment': self._calculate_timeframe_alignment(multi_timeframe_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error in meta-learner prediction: {e}")
            return self._fallback_ranking(signal_data, market_context)
    
    def _extract_meta_features(self, 
                              signal_data: Dict[str, Any],
                              market_context: Dict[str, Any],
                              multi_timeframe_data: Dict[str, Dict] = None) -> List[float]:
        """Extract comprehensive features for meta-learning."""
        features = []
        
        # 1. ML signal features
        ml_features = signal_data.get('ml_signals', {})
        features.extend([
            ml_features.get('confidence', 0.0),
            ml_features.get('probability', 0.0),
            ml_features.get('uncertainty', 0.0)
        ])
        
        # 2. Rule-based signal features
        rule_features = signal_data.get('rule_signals', {})
        features.extend([
            rule_features.get('cisd_score', 0.0),
            rule_features.get('structure_score', 0.0),
            rule_features.get('fourier_score', 0.0),
            rule_features.get('regime_score', 0.0)
        ])
        
        # 3. Order flow features
        of_features = signal_data.get('order_flow', {})
        features.extend([
            of_features.get('delta_momentum', 0.0),
            of_features.get('absorption_strength', 0.0),
            of_features.get('institutional_pressure', 0.0),
            of_features.get('liquidity_imbalance', 0.0)
        ])
        
        # 4. Market context features
        features.extend([
            market_context.get('volatility', 0.0),
            market_context.get('trend_strength', 0.0),
            market_context.get('regime', 0.0),
            market_context.get('session', 0.0)
        ])
        
        # 5. Multi-timeframe alignment features (top-down analysis)
        if multi_timeframe_data:
            tf_alignment = self._calculate_timeframe_alignment(multi_timeframe_data)
            features.extend([
                tf_alignment.get('overall_alignment', 0.0),
                tf_alignment.get('trend_consistency', 0.0),
                tf_alignment.get('level_agreement', 0.0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 6. IPDA phase features
        ipda_phase = self._detect_ipda_phase(signal_data, market_context)
        features.extend([
            ipda_phase.get('accumulation_score', 0.0),
            ipda_phase.get('distribution_score', 0.0),
            ipda_phase.get('manipulation_score', 0.0)
        ])
        
        return features
    
    def _apply_ipda_adjustments(self, 
                               prediction: float,
                               signal_data: Dict[str, Any],
                               market_context: Dict[str, Any],
                               multi_timeframe_data: Dict[str, Dict] = None) -> float:
        """Apply IPDA and top-down analysis adjustments to the prediction."""
        
        # 1. IPDA Phase Adjustment
        ipda_phase = self._detect_ipda_phase(signal_data, market_context)
        phase_multiplier = ipda_phase.get('phase_multiplier', 1.0)
        
        # 2. Top-Down Analysis Adjustment
        if multi_timeframe_data:
            tf_alignment = self._calculate_timeframe_alignment(multi_timeframe_data)
            alignment_multiplier = tf_alignment.get('overall_alignment', 1.0)
        else:
            alignment_multiplier = 1.0
        
        # 3. Institutional Activity Adjustment
        inst_activity = signal_data.get('order_flow', {}).get('institutional_pressure', 0.0)
        inst_multiplier = 1.0 + (inst_activity * 0.2)  # ±20% based on institutional pressure
        
        # 4. Market Regime Adjustment
        regime = market_context.get('regime', 'normal')
        regime_multipliers = {
            'quiet': 0.7,      # Reduce signals in quiet markets
            'normal': 1.0,     # Normal multiplier
            'trending': 1.3,   # Boost signals in trending markets
            'volatile': 0.8    # Reduce signals in volatile markets
        }
        regime_multiplier = regime_multipliers.get(regime, 1.0)
        
        # Apply all adjustments
        adjusted_prediction = prediction * phase_multiplier * alignment_multiplier * inst_multiplier * regime_multiplier
        
        return adjusted_prediction
    
    def _detect_ipda_phase(self, signal_data: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, float]:
        """Detect current IPDA phase based on order flow and market structure."""
        
        of_data = signal_data.get('order_flow', {})
        structure_data = signal_data.get('structure', {})
        
        # Analyze accumulation/distribution patterns
        delta_momentum = of_data.get('delta_momentum', 0.0)
        volume_profile = of_data.get('volume_profile', [])
        absorption = of_data.get('absorption_strength', 0.0)
        
        # Detect accumulation (smart money buying)
        accumulation_score = 0.0
        if delta_momentum > 0.3 and absorption > 0.5:
            accumulation_score = min(1.0, delta_momentum + absorption)
        
        # Detect distribution (smart money selling)
        distribution_score = 0.0
        if delta_momentum < -0.3 and absorption > 0.5:
            distribution_score = min(1.0, abs(delta_momentum) + absorption)
        
        # Detect manipulation (false moves, stop runs)
        manipulation_score = 0.0
        if structure_data.get('false_breakout', False) or structure_data.get('stop_run', False):
            manipulation_score = 0.8
        
        # Determine dominant phase
        phase_scores = {
            'accumulation': accumulation_score,
            'distribution': distribution_score,
            'manipulation': manipulation_score
        }
        
        dominant_phase = max(phase_scores, key=phase_scores.get)
        phase_multiplier = self.ipda_phases.get(dominant_phase, 1.0)
        
        return {
            'phase': dominant_phase,
            'phase_multiplier': phase_multiplier,
            'accumulation_score': accumulation_score,
            'distribution_score': distribution_score,
            'manipulation_score': manipulation_score
        }
    
    def _calculate_timeframe_alignment(self, multi_timeframe_data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate top-down analysis alignment across timeframes."""
        
        if not multi_timeframe_data:
            return {'overall_alignment': 0.0, 'trend_consistency': 0.0, 'level_agreement': 0.0}
        
        # Extract trend directions from each timeframe
        trend_directions = {}
        for tf, data in multi_timeframe_data.items():
            trend = data.get('trend_direction', 0)  # -1, 0, 1
            trend_directions[tf] = trend
        
        # Calculate trend consistency
        trend_values = list(trend_directions.values())
        if len(trend_values) > 1:
            trend_consistency = 1.0 - (np.std(trend_values) / 2.0)  # Normalize to 0-1
        else:
            trend_consistency = 0.5
        
        # Calculate level agreement (support/resistance alignment)
        level_agreement = 0.0
        if len(multi_timeframe_data) >= 2:
            # Simplified: check if key levels align across timeframes
            levels = []
            for tf, data in multi_timeframe_data.items():
                key_levels = data.get('key_levels', [])
                levels.extend(key_levels)
            
            if levels:
                # Calculate how clustered the levels are
                level_std = np.std(levels) if len(levels) > 1 else 0
                level_agreement = max(0, 1.0 - level_std / 100)  # Normalize
        
        # Overall alignment (weighted combination)
        overall_alignment = (trend_consistency * 0.6 + level_agreement * 0.4)
        
        return {
            'overall_alignment': overall_alignment,
            'trend_consistency': trend_consistency,
            'level_agreement': level_agreement,
            'trend_directions': trend_directions
        }
    
    def _rank_signals(self, signal_data: Dict[str, Any], prediction: float, uncertainty: float) -> List[Dict[str, Any]]:
        """Rank individual signals based on meta-learner output."""
        
        signals = []
        
        # ML signals
        ml_signals = signal_data.get('ml_signals', {})
        if ml_signals.get('confidence', 0) > 0:
            signals.append({
                'source': 'ML',
                'score': ml_signals.get('confidence', 0),
                'signal': ml_signals.get('signal', 'HOLD'),
                'weight': 0.3
            })
        
        # CISD signals
        cisd_score = signal_data.get('rule_signals', {}).get('cisd_score', 0)
        if cisd_score > 0:
            signals.append({
                'source': 'CISD',
                'score': cisd_score,
                'signal': 'BUY' if cisd_score > 0.5 else 'SELL',
                'weight': 0.25
            })
        
        # Order flow signals
        of_data = signal_data.get('order_flow', {})
        if of_data.get('delta_momentum', 0) != 0:
            of_score = abs(of_data.get('delta_momentum', 0))
            signals.append({
                'source': 'OrderFlow',
                'score': of_score,
                'signal': 'BUY' if of_data.get('delta_momentum', 0) > 0 else 'SELL',
                'weight': 0.25
            })
        
        # Structure signals
        structure_score = signal_data.get('rule_signals', {}).get('structure_score', 0)
        if structure_score > 0:
            signals.append({
                'source': 'Structure',
                'score': structure_score,
                'signal': 'BUY' if structure_score > 0.5 else 'SELL',
                'weight': 0.2
            })
        
        # Sort by weighted score
        for signal in signals:
            signal['weighted_score'] = signal['score'] * signal['weight']
        
        signals.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return signals
    
    def _calculate_model_weights(self, predictions: Dict[str, float], uncertainties: Dict[str, float]) -> Dict[str, float]:
        """Calculate ensemble weights based on prediction confidence."""
        
        # Inverse uncertainty weighting (lower uncertainty = higher weight)
        weights = {}
        total_weight = 0
        
        for name in predictions.keys():
            uncertainty = uncertainties.get(name, 0.1)
            weight = 1.0 / (uncertainty + 0.01)  # Add small epsilon to avoid division by zero
            weights[name] = weight
            total_weight += weight
        
        # Normalize weights
        for name in weights:
            weights[name] /= total_weight
        
        return weights
    
    def _fallback_ranking(self, signal_data: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback ranking when models are not trained."""
        
        # Simple weighted average of available signals
        total_score = 0.0
        total_weight = 0.0
        
        # ML signals
        ml_confidence = signal_data.get('ml_signals', {}).get('confidence', 0.0)
        if ml_confidence > 0:
            total_score += ml_confidence * 0.3
            total_weight += 0.3
        
        # CISD signals
        cisd_score = signal_data.get('rule_signals', {}).get('cisd_score', 0.0)
        if cisd_score > 0:
            total_score += cisd_score * 0.25
            total_weight += 0.25
        
        # Order flow signals
        of_momentum = signal_data.get('order_flow', {}).get('delta_momentum', 0.0)
        if of_momentum != 0:
            of_score = abs(of_momentum)
            total_score += of_score * 0.25
            total_weight += 0.25
        
        # Structure signals
        structure_score = signal_data.get('rule_signals', {}).get('structure_score', 0.0)
        if structure_score > 0:
            total_score += structure_score * 0.2
            total_weight += 0.2
        
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        
        return {
            'final_prediction': final_score,
            'uncertainty': 0.3,  # Default uncertainty
            'confidence': max(0, 1 - 0.3),
            'signal_ranking': self._rank_signals(signal_data, final_score, 0.3),
            'model_predictions': {},
            'model_uncertainties': {},
            'weights': {},
            'ipda_phase': {'phase': 'unknown', 'phase_multiplier': 1.0},
            'timeframe_alignment': {'overall_alignment': 0.5}
        }
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            for name, model in self.ensemble_models.items():
                model_path = os.path.join(self.model_dir, f"{name}_model.joblib")
                joblib.dump(model, model_path)
            
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            self.logger.info("Meta-learner models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk."""
        try:
            for name in self.ensemble_models.keys():
                model_path = os.path.join(self.model_dir, f"{name}_model.joblib")
                if os.path.exists(model_path):
                    self.ensemble_models[name] = joblib.load(model_path)
            
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                self.logger.info("Meta-learner models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
