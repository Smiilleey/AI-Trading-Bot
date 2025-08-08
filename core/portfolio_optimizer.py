# core/portfolio_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import cvxopt as cv
from cvxopt import matrix, solvers

@dataclass
class PortfolioAllocation:
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_score: float
    rebalance_needed: bool
    timestamp: datetime

class InstitutionalPortfolioOptimizer:
    """
    Institutional-grade portfolio optimizer:
    - Risk parity allocation
    - Dynamic rebalancing
    - Cross-asset correlation
    - Risk-adjusted sizing
    """
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        target_volatility: float = 0.10,
        rebalance_threshold: float = 0.05,
        min_weight: float = 0.05,
        max_weight: float = 0.30
    ):
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.rebalance_threshold = rebalance_threshold
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # State tracking
        self.current_allocation = None
        self.asset_correlations = None
        self.volatility_targets = {}
        self.position_limits = {}
        
    def optimize(
        self,
        returns: pd.DataFrame,
        market_data: Dict,
        constraints: Optional[Dict] = None
    ) -> PortfolioAllocation:
        """
        Run portfolio optimization:
        - Calculate optimal weights
        - Check rebalancing needs
        - Apply risk constraints
        """
        # Calculate return characteristics
        expected_returns = self._calculate_expected_returns(returns)
        covariance = self._calculate_covariance(returns)
        
        # Get risk parity weights
        risk_parity = self._calculate_risk_parity(covariance)
        
        # Get minimum variance weights
        min_var = self._calculate_minimum_variance(covariance)
        
        # Get maximum Sharpe weights
        max_sharpe = self._calculate_maximum_sharpe(
            expected_returns,
            covariance
        )
        
        # Blend allocations
        weights = self._blend_allocations(
            risk_parity,
            min_var,
            max_sharpe,
            market_data
        )
        
        # Apply constraints
        weights = self._apply_constraints(
            weights,
            constraints or {}
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            weights,
            expected_returns,
            covariance
        )
        
        # Check rebalancing
        rebalance = self._check_rebalance_needed(
            weights,
            self.current_allocation.weights if self.current_allocation else None
        )
        
        # Create allocation
        allocation = PortfolioAllocation(
            weights=dict(weights),
            expected_return=metrics["return"],
            expected_risk=metrics["risk"],
            sharpe_ratio=metrics["sharpe"],
            diversification_score=metrics["diversification"],
            rebalance_needed=rebalance,
            timestamp=datetime.now()
        )
        
        self.current_allocation = allocation
        return allocation
        
    def _calculate_risk_parity(
        self,
        covariance: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate risk parity weights:
        - Equal risk contribution
        - Volatility scaling
        - Cross-asset correlation
        """
        n_assets = len(covariance)
        
        def risk_parity_objective(x):
            # Calculate risk contributions
            portfolio_risk = np.sqrt(x.T @ covariance @ x)
            risk_contrib = x * (covariance @ x) / portfolio_risk
            
            # Calculate risk parity score
            risk_target = portfolio_risk / n_assets
            parity_score = sum((risk_contrib - risk_target)**2)
            
            return parity_score
            
        # Optimization constraints
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x) - 1},  # Weights sum to 1
            {"type": "ineq", "fun": lambda x: x - self.min_weight},  # Minimum weight
            {"type": "ineq", "fun": lambda x: self.max_weight - x}  # Maximum weight
        ]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Run optimization
        result = minimize(
            risk_parity_objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            bounds=[(self.min_weight, self.max_weight)] * n_assets
        )
        
        return result.x
        
    def _calculate_minimum_variance(
        self,
        covariance: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate minimum variance weights:
        - Lowest portfolio volatility
        - Correlation-aware
        - Constrained optimization
        """
        n_assets = len(covariance)
        
        # Convert to cvxopt format
        S = matrix(covariance.values)
        pbar = matrix(np.zeros(n_assets))
        
        # Constraints
        G = matrix(np.vstack((
            -np.eye(n_assets),  # Minimum weight
            np.eye(n_assets)   # Maximum weight
        )))
        h = matrix(np.hstack((
            -np.ones(n_assets) * self.min_weight,
            np.ones(n_assets) * self.max_weight
        )))
        A = matrix(np.ones((1, n_assets)))
        b = matrix(np.ones(1))
        
        # Solve quadratic program
        solvers.options["show_progress"] = False
        sol = solvers.qp(S, pbar, G, h, A, b)
        
        return np.array(sol["x"]).flatten()
        
    def _calculate_maximum_sharpe(
        self,
        returns: pd.Series,
        covariance: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate maximum Sharpe ratio weights:
        - Highest risk-adjusted return
        - Return forecasting
        - Risk adjustment
        """
        n_assets = len(returns)
        
        def sharpe_objective(x):
            portfolio_return = np.sum(returns * x)
            portfolio_risk = np.sqrt(x.T @ covariance @ x)
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
            return -sharpe  # Minimize negative Sharpe
            
        # Optimization constraints
        constraints = [
            {"type": "eq", "fun": lambda x: sum(x) - 1},  # Weights sum to 1
            {"type": "ineq", "fun": lambda x: x - self.min_weight},  # Minimum weight
            {"type": "ineq", "fun": lambda x: self.max_weight - x}  # Maximum weight
        ]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Run optimization
        result = minimize(
            sharpe_objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            bounds=[(self.min_weight, self.max_weight)] * n_assets
        )
        
        return result.x
        
    def _blend_allocations(
        self,
        risk_parity: np.ndarray,
        min_var: np.ndarray,
        max_sharpe: np.ndarray,
        market_data: Dict
    ) -> np.ndarray:
        """
        Blend different allocation approaches:
        - Dynamic weighting
        - Market regime adaptation
        - Risk targeting
        """
        # Get market regime
        volatility = market_data.get("volatility", "normal")
        trend = market_data.get("trend", "neutral")
        liquidity = market_data.get("liquidity", "normal")
        
        # Base weights for each strategy
        weights = {
            "risk_parity": 0.4,
            "min_var": 0.3,
            "max_sharpe": 0.3
        }
        
        # Adjust for volatility regime
        if volatility == "high":
            weights["min_var"] += 0.1
            weights["max_sharpe"] -= 0.1
        elif volatility == "low":
            weights["risk_parity"] += 0.1
            weights["min_var"] -= 0.1
            
        # Adjust for trend regime
        if trend in ["strong_up", "strong_down"]:
            weights["max_sharpe"] += 0.1
            weights["risk_parity"] -= 0.1
            
        # Adjust for liquidity
        if liquidity == "low":
            weights["min_var"] += 0.1
            weights["max_sharpe"] -= 0.1
            
        # Blend allocations
        final_weights = (
            risk_parity * weights["risk_parity"] +
            min_var * weights["min_var"] +
            max_sharpe * weights["max_sharpe"]
        )
        
        # Normalize weights
        final_weights = final_weights / sum(final_weights)
        
        return final_weights
        
    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: Dict
    ) -> np.ndarray:
        """
        Apply portfolio constraints:
        - Position limits
        - Sector exposure
        - Risk limits
        """
        adjusted_weights = weights.copy()
        
        # Apply position limits
        if "position_limits" in constraints:
            for asset, limit in constraints["position_limits"].items():
                idx = self.asset_index[asset]
                adjusted_weights[idx] = min(
                    adjusted_weights[idx],
                    limit
                )
                
        # Apply sector constraints
        if "sector_limits" in constraints:
            for sector, limit in constraints["sector_limits"].items():
                sector_exposure = sum(
                    adjusted_weights[i]
                    for i, asset in enumerate(self.assets)
                    if self.asset_sectors[asset] == sector
                )
                if sector_exposure > limit:
                    scale = limit / sector_exposure
                    for i, asset in enumerate(self.assets):
                        if self.asset_sectors[asset] == sector:
                            adjusted_weights[i] *= scale
                            
        # Apply risk limits
        if "risk_limits" in constraints:
            portfolio_risk = self._calculate_portfolio_risk(
                adjusted_weights
            )
            if portfolio_risk > constraints["risk_limits"]["max_risk"]:
                scale = constraints["risk_limits"]["max_risk"] / portfolio_risk
                adjusted_weights *= scale
                
        # Renormalize weights
        adjusted_weights = adjusted_weights / sum(adjusted_weights)
        
        return adjusted_weights
        
    def _calculate_metrics(
        self,
        weights: np.ndarray,
        returns: pd.Series,
        covariance: pd.DataFrame
    ) -> Dict:
        """
        Calculate portfolio metrics:
        - Expected return
        - Expected risk
        - Sharpe ratio
        - Diversification
        """
        # Calculate return and risk
        exp_return = np.sum(returns * weights)
        exp_risk = np.sqrt(weights.T @ covariance @ weights)
        
        # Calculate Sharpe ratio
        sharpe = (exp_return - self.risk_free_rate) / exp_risk
        
        # Calculate diversification score
        n_assets = len(weights)
        effective_n = 1 / sum(weights**2)
        diversification = effective_n / n_assets
        
        return {
            "return": exp_return,
            "risk": exp_risk,
            "sharpe": sharpe,
            "diversification": diversification
        }
        
    def _check_rebalance_needed(
        self,
        new_weights: np.ndarray,
        current_weights: Optional[Dict]
    ) -> bool:
        """
        Check if rebalancing is needed:
        - Weight drift
        - Risk change
        - Return change
        """
        if current_weights is None:
            return True
            
        # Convert current weights to array
        current_array = np.array([
            current_weights.get(asset, 0)
            for asset in self.assets
        ])
        
        # Calculate weight drift
        drift = np.abs(new_weights - current_array)
        max_drift = np.max(drift)
        
        # Calculate risk change
        new_risk = self._calculate_portfolio_risk(new_weights)
        current_risk = self._calculate_portfolio_risk(current_array)
        risk_change = abs(new_risk - current_risk) / current_risk
        
        # Check thresholds
        return (
            max_drift > self.rebalance_threshold or
            risk_change > self.rebalance_threshold
        )
        
    def _calculate_portfolio_risk(
        self,
        weights: np.ndarray
    ) -> float:
        """Calculate portfolio risk"""
        return np.sqrt(
            weights.T @ self.current_covariance @ weights
        )
        
    def _calculate_expected_returns(
        self,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate expected returns:
        - Historical average
        - Trend adjustment
        - Volatility adjustment
        """
        # Calculate historical mean
        hist_mean = returns.mean()
        
        # Calculate trend factor
        trend = returns.rolling(window=60).mean().iloc[-1]
        
        # Calculate volatility adjustment
        vol = returns.rolling(window=20).std().iloc[-1]
        vol_adj = 1 / (1 + vol)
        
        # Combine factors
        expected_returns = (
            hist_mean * 0.4 +
            trend * 0.4 +
            vol_adj * hist_mean * 0.2
        )
        
        return expected_returns
        
    def _calculate_covariance(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix:
        - Exponential weighting
        - Correlation stability
        - Minimum variance
        """
        # Calculate exponential weights
        span = 252  # One year
        alpha = 2 / (span + 1)
        weights = np.exp(-alpha * np.arange(len(returns)))
        weights /= weights.sum()
        
        # Calculate weighted covariance
        weighted_returns = returns - returns.mean()
        weighted_returns = weighted_returns * np.sqrt(weights[:, np.newaxis])
        covariance = weighted_returns.T @ weighted_returns
        
        # Apply minimum variance
        min_variance = np.diag(covariance).min() * 0.001
        covariance = covariance + np.eye(len(covariance)) * min_variance
        
        return covariance
        
    def update_market_data(
        self,
        market_data: Dict
    ):
        """Update market regime data"""
        self.current_market_data = market_data
        
    def update_constraints(
        self,
        constraints: Dict
    ):
        """Update portfolio constraints"""
        self.current_constraints = constraints
        
    def get_rebalance_trades(
        self,
        current_positions: Dict,
        new_allocation: PortfolioAllocation
    ) -> List[Dict]:
        """
        Get required rebalancing trades:
        - Minimize turnover
        - Consider transaction costs
        - Maintain risk balance
        """
        trades = []
        
        for asset, target_weight in new_allocation.weights.items():
            current_weight = current_positions.get(asset, 0)
            if abs(target_weight - current_weight) > self.rebalance_threshold:
                trades.append({
                    "asset": asset,
                    "direction": "buy" if target_weight > current_weight else "sell",
                    "weight_change": abs(target_weight - current_weight),
                    "priority": self._calculate_trade_priority(
                        asset,
                        target_weight,
                        current_weight
                    )
                })
                
        # Sort trades by priority
        trades.sort(key=lambda x: x["priority"], reverse=True)
        
        return trades
        
    def _calculate_trade_priority(
        self,
        asset: str,
        target_weight: float,
        current_weight: float
    ) -> float:
        """
        Calculate trade priority:
        - Weight difference
        - Risk contribution
        - Market impact
        """
        # Weight difference component
        weight_diff = abs(target_weight - current_weight)
        
        # Risk contribution
        risk_contrib = self._calculate_risk_contribution(
            asset,
            current_weight
        )
        
        # Market impact estimate
        impact = self._estimate_market_impact(
            asset,
            weight_diff
        )
        
        # Combined priority score
        priority = (
            weight_diff * 0.4 +
            risk_contrib * 0.4 +
            (1 - impact) * 0.2
        )
        
        return priority
        
    def _calculate_risk_contribution(
        self,
        asset: str,
        weight: float
    ) -> float:
        """Calculate asset's risk contribution"""
        # Implementation details...
        pass
        
    def _estimate_market_impact(
        self,
        asset: str,
        weight_change: float
    ) -> float:
        """Estimate market impact of trade"""
        # Implementation details...
        pass
