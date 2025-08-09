# core/advanced_risk_engine.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
from datetime import datetime

class AdvancedRiskEngine:
    """
    Institutional-grade risk management:
    - Portfolio VaR calculation
    - Dynamic position sizing
    - Correlation-aware risk
    - Multi-timeframe analysis
    - Adaptive stop-loss
    """
    def __init__(self, base_risk: float = 0.01):
        self.base_risk = base_risk
        self.portfolio_positions = {}
        self.correlation_matrix = None
        self.volatility_metrics = {}
        self.var_history = []
        
    def calculate_position_size(
        self,
        pair: str,
        entry_price: float,
        stop_loss: float,
        account_size: float,
        confidence: float = 0.5,
        market_context: Dict = None
    ) -> Dict:
        """
        Calculate optimal position size considering:
        - Portfolio correlation
        - Market volatility
        - Current exposure
        - Risk metrics
        """
        # Base position size
        pip_value = self._get_pip_value(pair)
        stop_distance = abs(entry_price - stop_loss)
        base_size = (account_size * self.base_risk) / (stop_distance * pip_value)
        
        # Apply adjustments
        adjusted_size = base_size
        adjustments = []
        
        # Volatility adjustment
        vol_factor = self._calculate_volatility_factor(pair, market_context)
        adjusted_size *= vol_factor
        adjustments.append(f"Volatility: {vol_factor:.2f}x")
        
        # Correlation adjustment
        corr_factor = self._calculate_correlation_factor(pair)
        adjusted_size *= corr_factor
        adjustments.append(f"Correlation: {corr_factor:.2f}x")
        
        # Confidence adjustment
        conf_factor = self._calculate_confidence_factor(confidence)
        adjusted_size *= conf_factor
        adjustments.append(f"Confidence: {conf_factor:.2f}x")
        
        # Portfolio exposure check
        exposure_factor = self._check_portfolio_exposure(pair)
        adjusted_size *= exposure_factor
        adjustments.append(f"Exposure: {exposure_factor:.2f}x")
        
        return {
            "size": adjusted_size,
            "original_size": base_size,
            "adjustments": adjustments,
            "risk_metrics": self._get_risk_metrics(pair, adjusted_size)
        }
        
    def calculate_adaptive_stops(
        self,
        pair: str,
        entry_price: float,
        direction: str,
        market_context: Dict
    ) -> Dict:
        """
        Calculate adaptive stop-loss levels based on:
        - Market volatility
        - Support/Resistance levels
        - Volume profile
        - Market regime
        """
        # Get volatility metrics
        atr = self._calculate_atr(pair)
        volatility = self._calculate_volatility(pair)
        
        # Base stop distance
        base_stop = atr * 1.5
        
        # Adjust for market regime
        regime = market_context.get("market_regime", "normal")
        regime_multipliers = {
            "high_volatility": 2.0,
            "normal": 1.5,
            "low_volatility": 1.0
        }
        stop_distance = base_stop * regime_multipliers.get(regime, 1.5)
        
        # Calculate stop levels
        if direction == "buy":
            initial_stop = entry_price - stop_distance
            trailing_stop = entry_price - (stop_distance * 0.8)
        else:
            initial_stop = entry_price + stop_distance
            trailing_stop = entry_price + (stop_distance * 0.8)
            
        return {
            "initial_stop": initial_stop,
            "trailing_stop": trailing_stop,
            "stop_distance": stop_distance,
            "atr": atr,
            "volatility": volatility
        }
        
    def calculate_portfolio_var(
        self,
        positions: Dict,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate portfolio Value at Risk using:
        - Monte Carlo simulation
        - Historical VaR
        - Parametric VaR
        """
        # Calculate returns
        returns = self._calculate_portfolio_returns(positions)
        
        # Historical VaR
        hist_var = self._calculate_historical_var(returns, confidence_level)
        
        # Parametric VaR
        param_var = self._calculate_parametric_var(returns, confidence_level)
        
        # Monte Carlo VaR
        mc_var = self._calculate_monte_carlo_var(returns, confidence_level)
        
        # Combine results
        var_metrics = {
            "historical_var": hist_var,
            "parametric_var": param_var,
            "monte_carlo_var": mc_var,
            "combined_var": (hist_var + param_var + mc_var) / 3,
            "confidence_level": confidence_level,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.var_history.append(var_metrics)
        return var_metrics
        
    def _calculate_volatility_factor(
        self,
        pair: str,
        market_context: Dict = None
    ) -> float:
        """Calculate volatility-based position adjustment"""
        if not market_context:
            return 1.0
            
        regime = market_context.get("volatility_regime", "normal")
        multipliers = {
            "high_volatility": 0.5,
            "normal": 1.0,
            "low_volatility": 1.5
        }
        
        return multipliers.get(regime, 1.0)
        
    def _calculate_correlation_factor(self, pair: str) -> float:
        """Calculate correlation-based position adjustment"""
        if not self.correlation_matrix or pair not in self.correlation_matrix:
            return 1.0
            
        # Get correlations with existing positions
        correlations = []
        for pos_pair in self.portfolio_positions:
            if pos_pair in self.correlation_matrix[pair]:
                correlations.append(abs(self.correlation_matrix[pair][pos_pair]))
                
        if not correlations:
            return 1.0
            
        # Reduce position size for highly correlated portfolio
        avg_correlation = np.mean(correlations)
        return 1.0 - (avg_correlation * 0.5)  # Max 50% reduction
        
    def _calculate_confidence_factor(self, confidence: float) -> float:
        """Calculate confidence-based position adjustment"""
        # Scale from 0.5 to 1.5 based on confidence
        return 0.5 + confidence
        
    def _check_portfolio_exposure(self, pair: str) -> float:
        """Check current portfolio exposure"""
        if not self.portfolio_positions:
            return 1.0
            
        # Calculate current exposure per currency
        exposures = self._calculate_currency_exposures()
        base, quote = pair[:3], pair[3:]
        
        # Reduce size if currencies already heavily exposed
        base_factor = 1.0 - (abs(exposures.get(base, 0)) * 0.2)  # Max 20% reduction
        quote_factor = 1.0 - (abs(exposures.get(quote, 0)) * 0.2)
        
        return min(base_factor, quote_factor)
        
    def _get_risk_metrics(
        self,
        pair: str,
        position_size: float
    ) -> Dict:
        """Calculate comprehensive risk metrics"""
        return {
            "position_size": position_size,
            "currency_exposure": self._calculate_currency_exposures(),
            "portfolio_correlation": self._get_portfolio_correlation(pair),
            "var_contribution": self._calculate_var_contribution(pair, position_size),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _calculate_currency_exposures(self) -> Dict:
        """Calculate exposure per currency"""
        exposures = {}
        for pair, position in self.portfolio_positions.items():
            base, quote = pair[:3], pair[3:]
            size = position["size"]
            exposures[base] = exposures.get(base, 0) + size
            exposures[quote] = exposures.get(quote, 0) - size
        return exposures
        
    def _get_portfolio_correlation(self, pair: str) -> float:
        """Get correlation with current portfolio"""
        if not self.correlation_matrix or not self.portfolio_positions:
            return 0.0
            
        correlations = []
        for pos_pair in self.portfolio_positions:
            if pos_pair in self.correlation_matrix[pair]:
                correlations.append(self.correlation_matrix[pair][pos_pair])
                
        return np.mean(correlations) if correlations else 0.0
        
    def _calculate_var_contribution(
        self,
        pair: str,
        position_size: float
    ) -> float:
        """Calculate position's contribution to portfolio VaR"""
        if not self.var_history:
            return 0.0
            
        latest_var = self.var_history[-1]["combined_var"]
        return latest_var * position_size  # Simplified calculation
