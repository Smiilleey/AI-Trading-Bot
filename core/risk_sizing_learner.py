import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import joblib
import os
from collections import defaultdict, deque

class RiskSizingLearner:
    """
    Advanced risk sizing system that learns optimal position sizing and stop-loss parameters.
    
    Key Features:
    - Dynamic lot size calculation based on regime, volatility, and account equity
    - Adaptive stop-loss multipliers based on market conditions
    - Risk-reward optimization
    - Drawdown-aware position sizing
    - Regime-specific risk parameters
    - Kelly Criterion integration
    - Volatility-adjusted position sizing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk sizing models
        self.lot_size_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.stop_multiplier_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.take_profit_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Base risk parameters
        self.base_risk_per_trade = config.get('base_risk_per_trade', 0.02)  # 2% per trade
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.05)    # 5% max
        self.min_risk_per_trade = config.get('min_risk_per_trade', 0.005)   # 0.5% min
        
        # Account management
        self.account_equity = config.get('initial_equity', 10000.0)
        self.max_drawdown = config.get('max_drawdown', 0.20)  # 20% max drawdown
        self.current_drawdown = 0.0
        
        # Regime-specific risk parameters
        self.regime_risk_params = {
            'quiet': {
                'base_lot_multiplier': 0.8,
                'stop_multiplier': 1.2,
                'tp_multiplier': 1.5,
                'max_risk_multiplier': 0.8
            },
            'normal': {
                'base_lot_multiplier': 1.0,
                'stop_multiplier': 1.0,
                'tp_multiplier': 2.0,
                'max_risk_multiplier': 1.0
            },
            'trending': {
                'base_lot_multiplier': 1.2,
                'stop_multiplier': 0.8,
                'tp_multiplier': 3.0,
                'max_risk_multiplier': 1.2
            },
            'volatile': {
                'base_lot_multiplier': 0.6,
                'stop_multiplier': 1.5,
                'tp_multiplier': 1.2,
                'max_risk_multiplier': 0.6
            }
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.regime_performance = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.0,
            'avg_risk_reward': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        })
        
        # Learning parameters
        self.learning_rate = config.get('risk_learning_rate', 0.01)
        self.min_samples_for_learning = config.get('min_samples_for_learning', 50)
        self.update_frequency = config.get('risk_update_frequency', 25)
        
        # Kelly Criterion parameters
        self.kelly_enabled = config.get('kelly_enabled', True)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Use 25% of Kelly recommendation
        
        # Volatility adjustment
        self.volatility_lookback = config.get('volatility_lookback', 20)
        self.volatility_threshold = config.get('volatility_threshold', 0.01)
        
        # Model persistence
        self.model_dir = "memory/risk_sizing_learner"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.is_trained = False
        
    def calculate_position_size(self, 
                              signal_strength: float,
                              regime: str,
                              volatility: float,
                              account_equity: float,
                              market_context: Dict = None) -> Dict[str, float]:
        """
        Calculate optimal position size based on multiple factors.
        
        Args:
            signal_strength: Strength of the trading signal (0-1)
            regime: Current market regime
            volatility: Current market volatility
            account_equity: Current account equity
            market_context: Additional market context
            
        Returns:
            Dictionary with position sizing parameters
        """
        try:
            # Update account equity
            self.account_equity = account_equity
            
            # Calculate base risk amount
            base_risk = self._calculate_base_risk(regime, volatility, market_context)
            
            # Apply signal strength adjustment
            signal_adjusted_risk = base_risk * signal_strength
            
            # Apply Kelly Criterion if enabled
            if self.kelly_enabled:
                kelly_risk = self._calculate_kelly_risk(regime, signal_strength)
                signal_adjusted_risk = min(signal_adjusted_risk, kelly_risk)
            
            # Apply drawdown protection
            drawdown_adjusted_risk = self._apply_drawdown_protection(signal_adjusted_risk)
            
            # Calculate lot size
            lot_size = self._calculate_lot_size(drawdown_adjusted_risk, volatility, market_context)
            
            # Calculate stop and take profit levels
            stop_multiplier = self._calculate_stop_multiplier(regime, volatility, market_context)
            tp_multiplier = self._calculate_tp_multiplier(regime, volatility, market_context)
            
            return {
                'lot_size': lot_size,
                'risk_amount': drawdown_adjusted_risk,
                'risk_percentage': (drawdown_adjusted_risk / account_equity) * 100,
                'stop_multiplier': stop_multiplier,
                'take_profit_multiplier': tp_multiplier,
                'kelly_risk': kelly_risk if self.kelly_enabled else 0.0,
                'regime': regime,
                'volatility': volatility,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self._get_default_position_size(account_equity)
    
    def update_with_trade_outcome(self, 
                                trade_outcome: Dict,
                                position_params: Dict,
                                market_context: Dict = None):
        """
        Update risk sizing models with trade outcome.
        
        Args:
            trade_outcome: Dictionary with trade results
            position_params: Position sizing parameters used
            market_context: Market context at time of trade
        """
        try:
            # Extract trade information
            pnl = trade_outcome.get('pnl', 0.0)
            duration = trade_outcome.get('duration', 0)
            regime = position_params.get('regime', 'normal')
            volatility = position_params.get('volatility', 0.0)
            signal_strength = position_params.get('signal_strength', 0.5)
            
            # Calculate risk-reward ratio
            risk_amount = position_params.get('risk_amount', 0.0)
            risk_reward = pnl / risk_amount if risk_amount > 0 else 0.0
            
            # Store performance data
            performance_data = {
                'pnl': pnl,
                'duration': duration,
                'regime': regime,
                'volatility': volatility,
                'signal_strength': signal_strength,
                'risk_amount': risk_amount,
                'risk_reward': risk_reward,
                'lot_size': position_params.get('lot_size', 0.0),
                'stop_multiplier': position_params.get('stop_multiplier', 1.0),
                'tp_multiplier': position_params.get('take_profit_multiplier', 2.0),
                'timestamp': pd.Timestamp.now()
            }
            
            self.performance_history.append(performance_data)
            
            # Update regime performance
            self._update_regime_performance(regime, performance_data)
            
            # Update drawdown
            self._update_drawdown(pnl)
            
            # Trigger model update if enough samples
            if len(self.performance_history) % self.update_frequency == 0:
                self._update_risk_models()
            
        except Exception as e:
            self.logger.error(f"Error updating risk sizing with outcome: {e}")
    
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train risk sizing models on historical data.
        
        Args:
            training_data: DataFrame with trade outcomes and features
            
        Returns:
            Dictionary with training performance metrics
        """
        try:
            self.logger.info("Training risk sizing models...")
            
            # Prepare features and targets
            feature_cols = [col for col in training_data.columns 
                          if col not in ['lot_size', 'stop_multiplier', 'tp_multiplier', 
                                       'pnl', 'timestamp', 'symbol']]
            
            X = training_data[feature_cols].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_lot_train, y_lot_test = train_test_split(
                X_scaled, training_data['lot_size'], test_size=0.2, random_state=42
            )
            
            # Train lot size model
            self.lot_size_model.fit(X_train, y_lot_train)
            lot_score = self.lot_size_model.score(X_test, y_lot_test)
            
            # Train stop multiplier model
            _, _, y_stop_train, y_stop_test = train_test_split(
                X_scaled, training_data['stop_multiplier'], test_size=0.2, random_state=42
            )
            self.stop_multiplier_model.fit(X_train, y_stop_train)
            stop_score = self.stop_multiplier_model.score(X_test, y_stop_test)
            
            # Train take profit model
            _, _, y_tp_train, y_tp_test = train_test_split(
                X_scaled, training_data['take_profit_multiplier'], test_size=0.2, random_state=42
            )
            self.take_profit_model.fit(X_train, y_tp_train)
            tp_score = self.take_profit_model.score(X_test, y_tp_test)
            
            # Save models
            self._save_models()
            self.is_trained = True
            
            return {
                'lot_size_score': lot_score,
                'stop_multiplier_score': stop_score,
                'take_profit_score': tp_score,
                'overall_score': np.mean([lot_score, stop_score, tp_score]),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            self.logger.error(f"Error training risk sizing models: {e}")
            return {'error': str(e)}
    
    def _calculate_base_risk(self, regime: str, volatility: float, market_context: Dict = None) -> float:
        """Calculate base risk amount based on regime and volatility."""
        try:
            # Get regime-specific parameters
            regime_params = self.regime_risk_params.get(regime, self.regime_risk_params['normal'])
            base_multiplier = regime_params['max_risk_multiplier']
            
            # Calculate volatility-adjusted risk
            volatility_factor = min(2.0, max(0.5, volatility / self.volatility_threshold))
            volatility_adjusted_risk = self.base_risk_per_trade * volatility_factor
            
            # Apply regime multiplier
            regime_adjusted_risk = volatility_adjusted_risk * base_multiplier
            
            # Ensure within bounds
            final_risk = max(self.min_risk_per_trade, 
                           min(self.max_risk_per_trade, regime_adjusted_risk))
            
            return final_risk * self.account_equity
            
        except Exception as e:
            self.logger.error(f"Error calculating base risk: {e}")
            return self.base_risk_per_trade * self.account_equity
    
    def _calculate_kelly_risk(self, regime: str, signal_strength: float) -> float:
        """Calculate Kelly Criterion risk amount."""
        try:
            if not self.kelly_enabled:
                return self.max_risk_per_trade * self.account_equity
            
            # Get regime performance
            regime_perf = self.regime_performance[regime]
            
            if regime_perf['total_trades'] < 10:
                return self.base_risk_per_trade * self.account_equity
            
            # Calculate Kelly fraction
            win_rate = regime_perf['win_rate']
            avg_risk_reward = regime_perf['avg_risk_reward']
            
            if avg_risk_reward <= 0:
                return self.base_risk_per_trade * self.account_equity
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_risk_reward, p = win_rate, q = 1 - win_rate
            kelly_fraction = (avg_risk_reward * win_rate - (1 - win_rate)) / avg_risk_reward
            
            # Apply Kelly fraction and signal strength
            kelly_risk = max(0, kelly_fraction * self.kelly_fraction * signal_strength)
            
            # Cap at maximum risk
            return min(kelly_risk, self.max_risk_per_trade) * self.account_equity
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly risk: {e}")
            return self.base_risk_per_trade * self.account_equity
    
    def _apply_drawdown_protection(self, risk_amount: float) -> float:
        """Apply drawdown protection to risk amount."""
        try:
            if self.current_drawdown > self.max_drawdown * 0.5:  # If drawdown > 10%
                # Reduce risk by drawdown factor
                drawdown_factor = 1.0 - (self.current_drawdown / self.max_drawdown)
                risk_amount *= max(0.1, drawdown_factor)  # Minimum 10% of original risk
            
            return risk_amount
            
        except Exception as e:
            self.logger.error(f"Error applying drawdown protection: {e}")
            return risk_amount
    
    def _calculate_lot_size(self, risk_amount: float, volatility: float, market_context: Dict = None) -> float:
        """Calculate lot size based on risk amount and volatility."""
        try:
            if self.is_trained and market_context:
                # Use ML model for lot size prediction
                features = self._extract_lot_size_features(risk_amount, volatility, market_context)
                if features:
                    features_scaled = self.scaler.transform([features])
                    predicted_lot_size = self.lot_size_model.predict(features_scaled)[0]
                    return max(0.01, min(10.0, predicted_lot_size))  # Reasonable bounds
            
            # Fallback to rule-based calculation
            # Assume 1 pip = $10 for standard lot (simplified)
            pip_value = 10.0
            stop_distance_pips = max(10, volatility * 10000)  # Convert volatility to pips
            lot_size = risk_amount / (stop_distance_pips * pip_value)
            
            return max(0.01, min(10.0, lot_size))  # Reasonable bounds
            
        except Exception as e:
            self.logger.error(f"Error calculating lot size: {e}")
            return 0.01
    
    def _calculate_stop_multiplier(self, regime: str, volatility: float, market_context: Dict = None) -> float:
        """Calculate stop-loss multiplier based on regime and volatility."""
        try:
            if self.is_trained and market_context:
                # Use ML model for stop multiplier prediction
                features = self._extract_stop_features(regime, volatility, market_context)
                if features:
                    features_scaled = self.scaler.transform([features])
                    predicted_multiplier = self.stop_multiplier_model.predict(features_scaled)[0]
                    return max(0.5, min(3.0, predicted_multiplier))  # Reasonable bounds
            
            # Fallback to rule-based calculation
            regime_params = self.regime_risk_params.get(regime, self.regime_risk_params['normal'])
            base_multiplier = regime_params['stop_multiplier']
            
            # Adjust for volatility
            volatility_factor = min(2.0, max(0.5, volatility / self.volatility_threshold))
            
            return max(0.5, min(3.0, base_multiplier * volatility_factor))
            
        except Exception as e:
            self.logger.error(f"Error calculating stop multiplier: {e}")
            return 1.0
    
    def _calculate_tp_multiplier(self, regime: str, volatility: float, market_context: Dict = None) -> float:
        """Calculate take-profit multiplier based on regime and volatility."""
        try:
            if self.is_trained and market_context:
                # Use ML model for TP multiplier prediction
                features = self._extract_tp_features(regime, volatility, market_context)
                if features:
                    features_scaled = self.scaler.transform([features])
                    predicted_multiplier = self.take_profit_model.predict(features_scaled)[0]
                    return max(0.5, min(5.0, predicted_multiplier))  # Reasonable bounds
            
            # Fallback to rule-based calculation
            regime_params = self.regime_risk_params.get(regime, self.regime_risk_params['normal'])
            base_multiplier = regime_params['tp_multiplier']
            
            # Adjust for volatility
            volatility_factor = min(2.0, max(0.5, volatility / self.volatility_threshold))
            
            return max(0.5, min(5.0, base_multiplier * volatility_factor)
            
        except Exception as e:
            self.logger.error(f"Error calculating TP multiplier: {e}")
            return 2.0
    
    def _extract_lot_size_features(self, risk_amount: float, volatility: float, market_context: Dict) -> List[float]:
        """Extract features for lot size prediction."""
        features = []
        
        # Basic features
        features.extend([
            risk_amount / self.account_equity,  # Risk percentage
            volatility,
            market_context.get('trend_strength', 0.0),
            market_context.get('liquidity', 0.0),
            market_context.get('depth', 0.0)
        ])
        
        # Regime encoding
        regime = market_context.get('regime', 'normal')
        regime_encoding = {
            'quiet': [1.0, 0.0, 0.0, 0.0],
            'normal': [0.0, 1.0, 0.0, 0.0],
            'trending': [0.0, 0.0, 1.0, 0.0],
            'volatile': [0.0, 0.0, 0.0, 1.0]
        }
        features.extend(regime_encoding.get(regime, [0.0, 0.0, 0.0, 0.0]))
        
        # Session encoding
        session = market_context.get('session', 'unknown')
        session_encoding = {
            'london': [1.0, 0.0, 0.0, 0.0],
            'newyork': [0.0, 1.0, 0.0, 0.0],
            'tokyo': [0.0, 0.0, 1.0, 0.0],
            'overlap': [0.0, 0.0, 0.0, 1.0]
        }
        features.extend(session_encoding.get(session, [0.0, 0.0, 0.0, 0.0]))
        
        return features
    
    def _extract_stop_features(self, regime: str, volatility: float, market_context: Dict) -> List[float]:
        """Extract features for stop multiplier prediction."""
        features = []
        
        # Basic features
        features.extend([
            volatility,
            market_context.get('trend_strength', 0.0),
            market_context.get('liquidity', 0.0),
            market_context.get('stress', 0.0),
            self.current_drawdown
        ])
        
        # Regime encoding
        regime_encoding = {
            'quiet': [1.0, 0.0, 0.0, 0.0],
            'normal': [0.0, 1.0, 0.0, 0.0],
            'trending': [0.0, 0.0, 1.0, 0.0],
            'volatile': [0.0, 0.0, 0.0, 1.0]
        }
        features.extend(regime_encoding.get(regime, [0.0, 0.0, 0.0, 0.0]))
        
        return features
    
    def _extract_tp_features(self, regime: str, volatility: float, market_context: Dict) -> List[float]:
        """Extract features for take-profit multiplier prediction."""
        features = []
        
        # Basic features
        features.extend([
            volatility,
            market_context.get('trend_strength', 0.0),
            market_context.get('liquidity', 0.0),
            market_context.get('depth', 0.0)
        ])
        
        # Regime encoding
        regime_encoding = {
            'quiet': [1.0, 0.0, 0.0, 0.0],
            'normal': [0.0, 1.0, 0.0, 0.0],
            'trending': [0.0, 0.0, 1.0, 0.0],
            'volatile': [0.0, 0.0, 0.0, 1.0]
        }
        features.extend(regime_encoding.get(regime, [0.0, 0.0, 0.0, 0.0]))
        
        return features
    
    def _update_regime_performance(self, regime: str, performance_data: Dict):
        """Update performance metrics for a regime."""
        perf = self.regime_performance[regime]
        
        pnl = performance_data['pnl']
        risk_reward = performance_data['risk_reward']
        
        perf['total_trades'] += 1
        perf['total_pnl'] += pnl
        
        if pnl > 0:
            perf['winning_trades'] += 1
        
        perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
        perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
        
        # Update average risk-reward
        if risk_reward > 0:
            if perf['avg_risk_reward'] == 0:
                perf['avg_risk_reward'] = risk_reward
            else:
                perf['avg_risk_reward'] = (perf['avg_risk_reward'] + risk_reward) / 2
    
    def _update_drawdown(self, pnl: float):
        """Update current drawdown."""
        if pnl < 0:
            self.current_drawdown += abs(pnl) / self.account_equity
        else:
            # Reduce drawdown by profit
            self.current_drawdown = max(0, self.current_drawdown - pnl / self.account_equity)
    
    def _update_risk_models(self):
        """Update risk sizing models with recent performance data."""
        try:
            if len(self.performance_history) < self.min_samples_for_learning:
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(list(self.performance_history))
            
            # Prepare features and targets
            feature_cols = ['regime', 'volatility', 'signal_strength', 'risk_amount']
            X = df[feature_cols].copy()
            
            # Encode regime
            X['regime_quiet'] = (X['regime'] == 'quiet').astype(int)
            X['regime_normal'] = (X['regime'] == 'normal').astype(int)
            X['regime_trending'] = (X['regime'] == 'trending').astype(int)
            X['regime_volatile'] = (X['regime'] == 'volatile').astype(int)
            X = X.drop('regime', axis=1)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Update models
            self.lot_size_model.fit(X_scaled, df['lot_size'])
            self.stop_multiplier_model.fit(X_scaled, df['stop_multiplier'])
            self.take_profit_model.fit(X_scaled, df['tp_multiplier'])
            
            self.logger.info("Risk sizing models updated with recent performance")
            
        except Exception as e:
            self.logger.error(f"Error updating risk models: {e}")
    
    def _get_default_position_size(self, account_equity: float) -> Dict[str, float]:
        """Get default position size parameters."""
        return {
            'lot_size': 0.01,
            'risk_amount': self.base_risk_per_trade * account_equity,
            'risk_percentage': self.base_risk_per_trade * 100,
            'stop_multiplier': 1.0,
            'take_profit_multiplier': 2.0,
            'kelly_risk': 0.0,
            'regime': 'normal',
            'volatility': 0.0,
            'signal_strength': 0.5
        }
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            # Save models
            joblib.dump(self.lot_size_model, os.path.join(self.model_dir, "lot_size_model.joblib"))
            joblib.dump(self.stop_multiplier_model, os.path.join(self.model_dir, "stop_multiplier_model.joblib"))
            joblib.dump(self.take_profit_model, os.path.join(self.model_dir, "take_profit_model.joblib"))
            joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
            
            # Save parameters
            import json
            params = {
                'base_risk_per_trade': self.base_risk_per_trade,
                'max_risk_per_trade': self.max_risk_per_trade,
                'min_risk_per_trade': self.min_risk_per_trade,
                'regime_risk_params': self.regime_risk_params,
                'kelly_enabled': self.kelly_enabled,
                'kelly_fraction': self.kelly_fraction
            }
            
            with open(os.path.join(self.model_dir, "risk_params.json"), 'w') as f:
                json.dump(params, f, indent=2)
            
            self.logger.info("Risk sizing models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving risk models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk."""
        try:
            # Load models
            self.lot_size_model = joblib.load(os.path.join(self.model_dir, "lot_size_model.joblib"))
            self.stop_multiplier_model = joblib.load(os.path.join(self.model_dir, "stop_multiplier_model.joblib"))
            self.take_profit_model = joblib.load(os.path.join(self.model_dir, "take_profit_model.joblib"))
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))
            
            # Load parameters
            import json
            with open(os.path.join(self.model_dir, "risk_params.json"), 'r') as f:
                params = json.load(f)
                self.base_risk_per_trade = params.get('base_risk_per_trade', 0.02)
                self.max_risk_per_trade = params.get('max_risk_per_trade', 0.05)
                self.min_risk_per_trade = params.get('min_risk_per_trade', 0.005)
                self.regime_risk_params = params.get('regime_risk_params', self.regime_risk_params)
                self.kelly_enabled = params.get('kelly_enabled', True)
                self.kelly_fraction = params.get('kelly_fraction', 0.25)
            
            self.is_trained = True
            self.logger.info("Risk sizing models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading risk models: {e}")
