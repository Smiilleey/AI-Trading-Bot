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
        
        # Instrument-specific parameters
        self.instrument_params = {
            "BTCUSD": {
                "min_weight": 0.02,  # Lower minimum due to higher volatility
                "max_weight": 0.15,  # Lower maximum for risk control
                "volatility_scalar": 2.0,  # Higher volatility adjustment
                "correlation_threshold": 0.6,  # Correlation significance threshold
                "key_correlations": ["XAUUSD", "EURUSD", "USDJPY"],  # Key pairs to monitor
                "regime_adjustments": {
                    "high_volatility": 0.5,  # Reduce allocation in high vol
                    "low_volatility": 1.2,   # Increase in low vol
                    "bull_trend": 1.3,       # Increase in bull market
                    "bear_trend": 0.7        # Reduce in bear market
                }
            },
            "XAUUSD": {
                "min_weight": 0.03,  # Moderate minimum weight
                "max_weight": 0.20,  # Moderate maximum weight
                "volatility_scalar": 1.5,  # Moderate volatility adjustment
                "correlation_threshold": 0.7,  # Higher correlation significance
                "key_correlations": ["USDX", "EURUSD", "USDJPY"],  # Key pairs to monitor
                "regime_adjustments": {
                    "high_volatility": 0.7,  # Moderate reduction in high vol
                    "low_volatility": 1.1,   # Slight increase in low vol
                    "usd_strength": 0.8,     # Reduce when USD strong
                    "usd_weakness": 1.2      # Increase when USD weak
                }
            }
        }
        
        # Correlation tracking
        self.correlation_state = {
            "BTCUSD": {
                "current_correlations": {},
                "correlation_regime": "normal",
                "hedging_opportunities": []
            },
            "XAUUSD": {
                "current_correlations": {},
                "correlation_regime": "normal",
                "hedging_opportunities": []
            }
        }
        
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
        Blend different allocation approaches with crypto and gold specifics:
        - Dynamic weighting
        - Market regime adaptation
        - Risk targeting
        - Crypto/Gold regime adjustments
        """
        # Get market regime
        volatility = market_data.get("volatility", "normal")
        trend = market_data.get("trend", "neutral")
        liquidity = market_data.get("liquidity", "normal")
        
        # Get crypto and gold specific data
        crypto_data = market_data.get("crypto_data", {})
        gold_data = market_data.get("gold_data", {})
        
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
            
        # Crypto-specific adjustments
        if "BTCUSD" in market_data.get("symbols", []):
            crypto_regime = crypto_data.get("market_regime", "normal")
            crypto_params = self.instrument_params["BTCUSD"]
            
            # Apply crypto regime adjustments
            if crypto_regime == "high_volatility":
                weights["min_var"] *= crypto_params["regime_adjustments"]["high_volatility"]
            elif crypto_regime == "low_volatility":
                weights["max_sharpe"] *= crypto_params["regime_adjustments"]["low_volatility"]
            elif crypto_regime == "bull_trend":
                weights["max_sharpe"] *= crypto_params["regime_adjustments"]["bull_trend"]
            elif crypto_regime == "bear_trend":
                weights["risk_parity"] *= crypto_params["regime_adjustments"]["bear_trend"]
            
            # Check correlation opportunities
            for pair in crypto_params["key_correlations"]:
                corr = crypto_data.get("correlations", {}).get(pair, 0)
                if abs(corr) > crypto_params["correlation_threshold"]:
                    self.correlation_state["BTCUSD"]["current_correlations"][pair] = corr
                    if corr < -0.7:  # Strong negative correlation
                        self.correlation_state["BTCUSD"]["hedging_opportunities"].append(pair)
        
        # Gold-specific adjustments
        if "XAUUSD" in market_data.get("symbols", []):
            gold_regime = gold_data.get("market_regime", "normal")
            gold_params = self.instrument_params["XAUUSD"]
            
            # Apply gold regime adjustments
            if gold_regime == "high_volatility":
                weights["min_var"] *= gold_params["regime_adjustments"]["high_volatility"]
            elif gold_regime == "low_volatility":
                weights["risk_parity"] *= gold_params["regime_adjustments"]["low_volatility"]
            
            # USD impact on gold
            usd_strength = gold_data.get("usd_strength", "neutral")
            if usd_strength == "strong":
                weights["max_sharpe"] *= gold_params["regime_adjustments"]["usd_strength"]
            elif usd_strength == "weak":
                weights["max_sharpe"] *= gold_params["regime_adjustments"]["usd_weakness"]
            
            # Check correlation opportunities
            for pair in gold_params["key_correlations"]:
                corr = gold_data.get("correlations", {}).get(pair, 0)
                if abs(corr) > gold_params["correlation_threshold"]:
                    self.correlation_state["XAUUSD"]["current_correlations"][pair] = corr
                    if corr < -0.7:  # Strong negative correlation
                        self.correlation_state["XAUUSD"]["hedging_opportunities"].append(pair)
            
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
        Apply portfolio constraints with crypto and gold specifics:
        - Position limits
        - Sector exposure
        - Risk limits
        - Crypto/Gold specific limits
        """
        adjusted_weights = weights.copy()
        
        # Apply position limits
        if "position_limits" in constraints:
            for asset, limit in constraints["position_limits"].items():
                # Apply instrument-specific limits
                if asset in self.instrument_params:
                    params = self.instrument_params[asset]
                    # Use the more conservative limit
                    limit = min(limit, params["max_weight"])
                    # Ensure minimum weight
                    if adjusted_weights[self.asset_index[asset]] > 0:
                        limit = max(limit, params["min_weight"])
                
                idx = self.asset_index[asset]
                adjusted_weights[idx] = min(
                    adjusted_weights[idx],
                    limit
                )
                
                # Apply volatility scaling for crypto and gold
                if asset in ["BTCUSD", "XAUUSD"]:
                    vol_scalar = self.instrument_params[asset]["volatility_scalar"]
                    adjusted_weights[idx] *= vol_scalar
                
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
        Get required rebalancing trades with crypto and gold specifics:
        - Minimize turnover
        - Consider transaction costs
        - Maintain risk balance
        - Handle crypto/gold specific considerations
        """
        trades = []
        
        for asset, target_weight in new_allocation.weights.items():
            current_weight = current_positions.get(asset, 0)
            
            # Get instrument-specific threshold
            threshold = self.rebalance_threshold
            if asset in self.instrument_params:
                # Use tighter thresholds for crypto and gold
                if asset == "BTCUSD":
                    threshold *= 0.8  # More frequent rebalancing for crypto
                elif asset == "XAUUSD":
                    threshold *= 0.9  # Slightly more frequent for gold
            
            if abs(target_weight - current_weight) > threshold:
                # Get correlation-based hedging opportunities
                hedging_pairs = []
                if asset in ["BTCUSD", "XAUUSD"]:
                    hedging_pairs = self.correlation_state[asset]["hedging_opportunities"]
                
                trades.append({
                    "asset": asset,
                    "direction": "buy" if target_weight > current_weight else "sell",
                    "weight_change": abs(target_weight - current_weight),
                    "priority": self._calculate_trade_priority(
                        asset,
                        target_weight,
                        current_weight
                    ),
                    "hedging_opportunities": hedging_pairs,
                    "market_impact_estimate": self._estimate_market_impact(
                        asset,
                        abs(target_weight - current_weight)
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
        """
        Estimate market impact of trade with crypto and gold specifics:
        - Volume-based impact
        - Time of day considerations
        - Market regime impact
        - Instrument-specific factors
        """
        base_impact = weight_change * 0.1  # Base 10% impact per weight change
        
        # Apply instrument-specific impact factors
        if asset == "BTCUSD":
            # Higher impact for crypto due to market structure
            base_impact *= 1.5
            
            # Consider market regime
            if self.correlation_state["BTCUSD"]["correlation_regime"] == "high_correlation":
                base_impact *= 1.2  # Higher impact in correlated markets
            
            # Consider exchange fragmentation
            base_impact *= 1.1  # Additional impact due to exchange fragmentation
            
        elif asset == "XAUUSD":
            # Moderate impact for gold
            base_impact *= 1.2
            
            # Consider market hours
            current_hour = datetime.now().hour
            if 8 <= current_hour <= 16:  # London hours
                base_impact *= 0.9  # Lower impact during main trading hours
            
            # Consider USD correlation
            if self.correlation_state["XAUUSD"]["correlation_regime"] == "high_correlation":
                base_impact *= 1.1  # Higher impact during strong USD correlation
        
        # Cap the impact
        return min(base_impact, 0.5)  # Maximum 50% impact
