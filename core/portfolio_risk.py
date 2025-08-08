# core/portfolio_risk.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from collections import defaultdict

@dataclass
class PortfolioState:
    total_exposure: float
    correlation_risk: float
    drawdown: float
    var_95: float  # 95% Value at Risk
    sharpe_ratio: float
    positions: Dict[str, Dict]
    timestamp: datetime

class AdaptivePortfolioManager:
    """
    Advanced portfolio risk management:
    - Dynamic position sizing
    - Correlation-aware allocation
    - Adaptive risk limits
    - Drawdown protection
    - Market regime integration
    """
    def __init__(
        self,
        max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
        target_sharpe: float = 1.5,
        drawdown_limit: float = 0.1,  # 10% max drawdown
        correlation_threshold: float = 0.7
    ):
        self.max_portfolio_risk = max_portfolio_risk
        self.target_sharpe = target_sharpe
        self.drawdown_limit = drawdown_limit
        self.correlation_threshold = correlation_threshold
        
        # State tracking
        self.portfolio_state = None
        self.position_history = defaultdict(list)
        self.risk_metrics = defaultdict(list)
        
        # Adaptive parameters
        self.risk_multipliers = {
            "trending": 1.2,
            "ranging": 0.8,
            "volatile": 0.6
        }
        self.correlation_penalties = defaultdict(float)
        
        # Performance tracking
        self.realized_sharpe = None
        self.max_drawdown = 0.0
        self.var_history = []
        
    def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        market_state: Dict,
        portfolio_context: Dict,
        correlation_data: Dict
    ) -> Dict:
        """
        Calculate optimal position size considering:
        - Portfolio correlation
        - Market regime
        - Current exposure
        - Risk limits
        """
        # Get current portfolio state
        current_exposure = portfolio_context.get("total_exposure", 0.0)
        current_positions = portfolio_context.get("positions", {})
        
        # Calculate base position size
        base_size = self._calculate_base_size(
            signal_confidence,
            market_state
        )
        
        # Apply correlation adjustments
        adjusted_size = self._adjust_for_correlations(
            base_size,
            symbol,
            correlation_data,
            current_positions
        )
        
        # Apply portfolio limits
        final_size = self._apply_portfolio_limits(
            adjusted_size,
            current_exposure,
            market_state
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            final_size,
            symbol,
            market_state,
            portfolio_context
        )
        
        return {
            "position_size": final_size,
            "risk_metrics": risk_metrics,
            "adjustments": self._get_size_adjustments(
                base_size,
                adjusted_size,
                final_size
            )
        }
        
    def _calculate_base_size(
        self,
        confidence: float,
        market_state: Dict
    ) -> float:
        """Calculate base position size"""
        # Start with confidence-based size
        base_size = confidence * self.max_portfolio_risk
        
        # Apply market regime multiplier
        regime = market_state.get("regime", "ranging")
        regime_multiplier = self.risk_multipliers.get(regime, 1.0)
        base_size *= regime_multiplier
        
        # Adjust for volatility
        vol_regime = market_state.get("volatility_regime", "normal")
        vol_multiplier = {
            "low": 1.2,
            "normal": 1.0,
            "high": 0.8
        }.get(vol_regime, 1.0)
        base_size *= vol_multiplier
        
        return base_size
        
    def _adjust_for_correlations(
        self,
        base_size: float,
        symbol: str,
        correlation_data: Dict,
        current_positions: Dict
    ) -> float:
        """
        Adjust position size based on correlations
        with existing positions
        """
        if not current_positions:
            return base_size
            
        # Calculate correlation penalty
        total_penalty = 0.0
        for pos_symbol, pos_data in current_positions.items():
            correlation = correlation_data.get(
                f"{symbol}_{pos_symbol}",
                {"correlation": 0}
            ).get("correlation", 0)
            
            if abs(correlation) > self.correlation_threshold:
                penalty = (
                    abs(correlation) - self.correlation_threshold
                ) / (1 - self.correlation_threshold)
                total_penalty += penalty * pos_data["size"]
                
        # Apply penalty
        adjusted_size = base_size * (1 - min(total_penalty, 0.8))
        
        return adjusted_size
        
    def _apply_portfolio_limits(
        self,
        size: float,
        current_exposure: float,
        market_state: Dict
    ) -> float:
        """
        Apply portfolio-wide risk limits
        """
        # Check total exposure
        total_exposure = current_exposure + size
        if total_exposure > self.max_portfolio_risk:
            size = max(0, self.max_portfolio_risk - current_exposure)
            
        # Apply market regime limits
        regime = market_state.get("regime", "ranging")
        if regime == "volatile":
            size = min(size, self.max_portfolio_risk * 0.5)
            
        # Apply drawdown protection
        if self.max_drawdown > self.drawdown_limit * 0.8:
            size *= 0.5  # Reduce size when approaching drawdown limit
            
        return size
        
    def _calculate_risk_metrics(
        self,
        size: float,
        symbol: str,
        market_state: Dict,
        portfolio_context: Dict
    ) -> Dict:
        """
        Calculate comprehensive risk metrics
        """
        metrics = {
            "position_var": self._calculate_var(
                size,
                symbol,
                market_state
            ),
            "portfolio_correlation": self._calculate_portfolio_correlation(
                symbol,
                portfolio_context
            ),
            "risk_contribution": size / (
                portfolio_context.get("total_exposure", size) + size
            ),
            "regime_risk": self.risk_multipliers.get(
                market_state.get("regime", "ranging"),
                1.0
            )
        }
        
        # Calculate portfolio Sharpe
        if self.realized_sharpe:
            metrics["expected_sharpe"] = self.realized_sharpe * (
                1 - metrics["portfolio_correlation"]
            )
            
        return metrics
        
    def _calculate_var(
        self,
        size: float,
        symbol: str,
        market_state: Dict
    ) -> float:
        """
        Calculate 95% Value at Risk for position
        """
        # Get historical volatility
        volatility = market_state.get("volatility", 0.01)
        
        # Calculate VaR
        var_95 = size * volatility * 1.96  # Assuming normal distribution
        
        # Adjust for market regime
        regime = market_state.get("regime", "ranging")
        if regime == "volatile":
            var_95 *= 1.5  # Increase VaR in volatile markets
            
        return var_95
        
    def _calculate_portfolio_correlation(
        self,
        symbol: str,
        portfolio_context: Dict
    ) -> float:
        """
        Calculate average correlation with portfolio
        """
        if not portfolio_context.get("positions"):
            return 0.0
            
        correlations = []
        for pos_symbol in portfolio_context["positions"]:
            corr = portfolio_context.get("correlations", {}).get(
                f"{symbol}_{pos_symbol}",
                {"correlation": 0}
            ).get("correlation", 0)
            correlations.append(abs(corr))
            
        return np.mean(correlations) if correlations else 0.0
        
    def _get_size_adjustments(
        self,
        base: float,
        adjusted: float,
        final: float
    ) -> List[Dict]:
        """
        Generate explanation of size adjustments
        """
        adjustments = []
        
        if adjusted < base:
            adjustments.append({
                "type": "correlation",
                "impact": (adjusted - base) / base,
                "description": "Reduced due to portfolio correlation"
            })
            
        if final < adjusted:
            adjustments.append({
                "type": "portfolio_limit",
                "impact": (final - adjusted) / adjusted,
                "description": "Reduced due to portfolio limits"
            })
            
        return adjustments
        
    def update_portfolio_state(
        self,
        positions: Dict[str, Dict],
        market_state: Dict,
        performance_metrics: Dict
    ):
        """
        Update portfolio state and learn from performance
        """
        # Calculate current state
        total_exposure = sum(p["size"] for p in positions.values())
        correlation_risk = self._calculate_correlation_risk(positions)
        
        # Update drawdown
        if performance_metrics.get("equity"):
            drawdown = (
                performance_metrics["peak_equity"] -
                performance_metrics["equity"]
            ) / performance_metrics["peak_equity"]
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
        # Calculate VaR
        portfolio_var = self._calculate_portfolio_var(
            positions,
            market_state
        )
        self.var_history.append(portfolio_var)
        
        # Update Sharpe ratio
        if performance_metrics.get("returns"):
            self.realized_sharpe = (
                np.mean(performance_metrics["returns"]) /
                np.std(performance_metrics["returns"])
            )
            
        # Create new state
        self.portfolio_state = PortfolioState(
            total_exposure=total_exposure,
            correlation_risk=correlation_risk,
            drawdown=self.max_drawdown,
            var_95=portfolio_var,
            sharpe_ratio=self.realized_sharpe or 0.0,
            positions=positions,
            timestamp=datetime.utcnow()
        )
        
        # Learn from performance
        self._update_risk_parameters(
            market_state,
            performance_metrics
        )
        
    def _calculate_correlation_risk(
        self,
        positions: Dict[str, Dict]
    ) -> float:
        """
        Calculate portfolio correlation risk score
        """
        if len(positions) < 2:
            return 0.0
            
        # Build correlation matrix
        symbols = list(positions.keys())
        corr_matrix = np.zeros((len(symbols), len(symbols)))
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i == j:
                    corr_matrix[i,j] = 1.0
                else:
                    corr_matrix[i,j] = abs(
                        positions[sym1].get("correlations", {}).get(
                            sym2,
                            {"correlation": 0}
                        ).get("correlation", 0)
                    )
                    
        # Calculate risk score
        weights = np.array([
            positions[sym]["size"] for sym in symbols
        ])
        weights /= np.sum(weights)
        
        risk_score = np.dot(
            np.dot(weights.T, corr_matrix),
            weights
        )
        
        return risk_score
        
    def _calculate_portfolio_var(
        self,
        positions: Dict[str, Dict],
        market_state: Dict
    ) -> float:
        """
        Calculate portfolio Value at Risk
        """
        if not positions:
            return 0.0
            
        # Calculate weighted VaR
        total_var = 0.0
        for pos in positions.values():
            var = self._calculate_var(
                pos["size"],
                pos["symbol"],
                market_state
            )
            total_var += var
            
        # Add correlation adjustment
        correlation_risk = self._calculate_correlation_risk(positions)
        total_var *= (1 + correlation_risk)
        
        return total_var
        
    def _update_risk_parameters(
        self,
        market_state: Dict,
        performance: Dict
    ):
        """
        Learn from performance and update risk parameters
        """
        if not performance.get("returns"):
            return
            
        # Get recent performance
        recent_returns = performance["returns"][-20:]
        recent_sharpe = (
            np.mean(recent_returns) /
            np.std(recent_returns)
            if len(recent_returns) > 1 else 0
        )
        
        # Update regime multipliers
        regime = market_state.get("regime", "ranging")
        if recent_sharpe > self.target_sharpe:
            self.risk_multipliers[regime] *= 1.01  # Increase risk slightly
        elif recent_sharpe < 0:
            self.risk_multipliers[regime] *= 0.95  # Reduce risk
            
        # Update correlation penalties
        if self.portfolio_state:
            if recent_sharpe < 0 and self.portfolio_state.correlation_risk > 0.5:
                self.correlation_threshold *= 0.99  # Make correlation limits stricter
            elif recent_sharpe > self.target_sharpe:
                self.correlation_threshold = min(
                    self.correlation_threshold * 1.01,
                    0.8
                )
                
        # Ensure parameters stay in reasonable bounds
        for regime in self.risk_multipliers:
            self.risk_multipliers[regime] = max(
                0.3,
                min(self.risk_multipliers[regime], 2.0)
            )
            
    def get_portfolio_analytics(self) -> Dict:
        """
        Get comprehensive portfolio analytics
        """
        if not self.portfolio_state:
            return {}
            
        analytics = {
            "exposure": {
                "total": self.portfolio_state.total_exposure,
                "by_regime": self._get_exposure_by_regime(),
                "correlation_risk": self.portfolio_state.correlation_risk
            },
            "risk_metrics": {
                "var_95": self.portfolio_state.var_95,
                "max_drawdown": self.max_drawdown,
                "sharpe_ratio": self.portfolio_state.sharpe_ratio
            },
            "parameters": {
                "risk_multipliers": self.risk_multipliers,
                "correlation_threshold": self.correlation_threshold
            },
            "positions": self._get_position_analytics(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return analytics
        
    def _get_exposure_by_regime(self) -> Dict:
        """
        Calculate exposure breakdown by market regime
        """
        exposure = defaultdict(float)
        
        if self.portfolio_state and self.portfolio_state.positions:
            for pos in self.portfolio_state.positions.values():
                regime = pos.get("market_state", {}).get("regime", "unknown")
                exposure[regime] += pos["size"]
                
        return dict(exposure)
        
    def _get_position_analytics(self) -> List[Dict]:
        """
        Get detailed analytics for each position
        """
        analytics = []
        
        if self.portfolio_state and self.portfolio_state.positions:
            for symbol, pos in self.portfolio_state.positions.items():
                analytics.append({
                    "symbol": symbol,
                    "size": pos["size"],
                    "var_contribution": (
                        pos["size"] * pos.get("risk_metrics", {}).get("var", 0)
                    ),
                    "correlation_risk": pos.get("risk_metrics", {}).get(
                        "portfolio_correlation",
                        0
                    ),
                    "regime": pos.get("market_state", {}).get("regime", "unknown")
                })
                
        return analytics
