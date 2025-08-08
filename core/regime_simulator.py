# core/regime_simulator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm, t
from collections import defaultdict

@dataclass
class MarketScenario:
    regime: str
    price_data: pd.DataFrame
    volume_data: pd.DataFrame
    order_flow: pd.DataFrame
    market_impact: Dict
    liquidity_profile: Dict
    timestamp: datetime

class MarketRegimeSimulator:
    """
    Advanced market regime simulator:
    - Realistic price generation
    - Order flow simulation
    - Liquidity dynamics
    - Market impact modeling
    - Regime transitions
    """
    def __init__(
        self,
        base_volatility: float = 0.01,
        liquidity_scale: float = 1.0,
        regime_params: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        self.base_volatility = base_volatility
        self.liquidity_scale = liquidity_scale
        self.regime_params = regime_params or self._default_regime_params()
        
        # State tracking
        self.current_regime = None
        self.regime_history = []
        self.price_history = []
        
        # Random state
        self.rng = np.random.RandomState(seed)
        
        # Regime transition matrix
        self.transition_matrix = self._build_transition_matrix()
        
    def generate_scenario(
        self,
        initial_price: float,
        duration_days: int,
        timeframe: str = "5min",
        regime: Optional[str] = None
    ) -> MarketScenario:
        """
        Generate complete market scenario:
        - Price paths
        - Volume profiles
        - Order flow
        - Market impact
        - Liquidity conditions
        """
        # Set initial regime
        self.current_regime = regime or self._random_regime()
        
        # Generate time index
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=int(duration_days * 24 * 60 / self._timeframe_minutes(timeframe)),
            freq=timeframe
        )
        
        # Generate price path
        prices = self._generate_price_path(
            initial_price,
            len(timestamps),
            timeframe
        )
        
        # Generate volume profile
        volumes = self._generate_volume_profile(
            len(timestamps),
            timeframe
        )
        
        # Generate order flow
        order_flow = self._generate_order_flow(
            prices,
            volumes,
            timeframe
        )
        
        # Generate market impact model
        impact = self._generate_market_impact(
            prices,
            volumes,
            order_flow
        )
        
        # Generate liquidity profile
        liquidity = self._generate_liquidity_profile(
            timestamps,
            volumes,
            order_flow
        )
        
        return MarketScenario(
            regime=self.current_regime,
            price_data=self._create_price_df(prices, timestamps),
            volume_data=self._create_volume_df(volumes, timestamps),
            order_flow=self._create_order_flow_df(order_flow, timestamps),
            market_impact=impact,
            liquidity_profile=liquidity,
            timestamp=datetime.now()
        )
        
    def _generate_price_path(
        self,
        initial_price: float,
        n_steps: int,
        timeframe: str
    ) -> np.ndarray:
        """
        Generate realistic price path:
        - Regime-specific dynamics
        - Volatility clustering
        - Jump processes
        - Mean reversion
        """
        prices = np.zeros(n_steps)
        prices[0] = initial_price
        
        # Get regime parameters
        regime_params = self.regime_params[self.current_regime]
        
        # Calculate timestep
        dt = self._timeframe_minutes(timeframe) / (24 * 60)
        
        # Generate path
        for i in range(1, n_steps):
            # Check for regime change
            if self.rng.random() < regime_params["transition_prob"]:
                self.current_regime = self._next_regime()
                regime_params = self.regime_params[self.current_regime]
                
            # Calculate drift and volatility
            drift = regime_params["drift"] * dt
            vol = regime_params["volatility"] * np.sqrt(dt)
            
            # Add jump component
            jump = 0
            if self.rng.random() < regime_params["jump_prob"]:
                jump = self.rng.normal(
                    regime_params["jump_mean"],
                    regime_params["jump_std"]
                )
                
            # Generate return
            if regime_params.get("mean_reverting", False):
                # Mean reverting process
                mean_level = regime_params["mean_level"]
                speed = regime_params["reversion_speed"]
                mean_rev = speed * (mean_level - prices[i-1]) * dt
                noise = vol * self.rng.normal()
                returns = mean_rev + noise + jump
            else:
                # Geometric Brownian Motion
                returns = drift + vol * self.rng.normal() + jump
                
            # Update price
            prices[i] = prices[i-1] * np.exp(returns)
            
        return prices
        
    def _generate_volume_profile(
        self,
        n_steps: int,
        timeframe: str
    ) -> Dict:
        """
        Generate realistic volume profile:
        - Time-of-day patterns
        - Regime-specific characteristics
        - Clustering effects
        """
        volumes = {
            "total": np.zeros(n_steps),
            "buy": np.zeros(n_steps),
            "sell": np.zeros(n_steps)
        }
        
        # Get regime parameters
        regime_params = self.regime_params[self.current_regime]
        
        # Generate base volume profile
        base_volume = self._generate_base_volume(
            n_steps,
            timeframe
        )
        
        for i in range(n_steps):
            # Add regime-specific volume adjustment
            volume_mult = regime_params["volume_multiplier"]
            if regime_params.get("volume_clustering", False):
                # Add volume clustering
                if i > 0 and volumes["total"][i-1] > base_volume[i-1]:
                    volume_mult *= 1.2
                    
            # Generate buy/sell volumes
            total_vol = base_volume[i] * volume_mult
            buy_ratio = self.rng.beta(
                regime_params["buy_alpha"],
                regime_params["buy_beta"]
            )
            
            volumes["total"][i] = total_vol
            volumes["buy"][i] = total_vol * buy_ratio
            volumes["sell"][i] = total_vol * (1 - buy_ratio)
            
        return volumes
        
    def _generate_order_flow(
        self,
        prices: np.ndarray,
        volumes: Dict,
        timeframe: str
    ) -> Dict:
        """
        Generate realistic order flow:
        - Institutional order patterns
        - Retail flow
        - Hidden liquidity
        """
        n_steps = len(prices)
        order_flow = {
            "institutional": {
                "buy": np.zeros(n_steps),
                "sell": np.zeros(n_steps)
            },
            "retail": {
                "buy": np.zeros(n_steps),
                "sell": np.zeros(n_steps)
            },
            "hidden": np.zeros(n_steps)
        }
        
        # Get regime parameters
        regime_params = self.regime_params[self.current_regime]
        
        for i in range(n_steps):
            # Generate institutional flow
            inst_ratio = regime_params["institutional_ratio"]
            inst_volume = volumes["total"][i] * inst_ratio
            
            order_flow["institutional"]["buy"][i] = (
                inst_volume * volumes["buy"][i] / volumes["total"][i]
            )
            order_flow["institutional"]["sell"][i] = (
                inst_volume * volumes["sell"][i] / volumes["total"][i]
            )
            
            # Generate retail flow
            retail_volume = volumes["total"][i] * (1 - inst_ratio)
            order_flow["retail"]["buy"][i] = (
                retail_volume * volumes["buy"][i] / volumes["total"][i]
            )
            order_flow["retail"]["sell"][i] = (
                retail_volume * volumes["sell"][i] / volumes["total"][i]
            )
            
            # Generate hidden liquidity
            if regime_params.get("hidden_liquidity", False):
                order_flow["hidden"][i] = (
                    volumes["total"][i] *
                    regime_params["hidden_ratio"] *
                    self.rng.random()
                )
                
        return order_flow
        
    def _generate_market_impact(
        self,
        prices: np.ndarray,
        volumes: Dict,
        order_flow: Dict
    ) -> Dict:
        """
        Generate market impact model:
        - Price impact function
        - Liquidity cost curves
        - Impact decay
        """
        impact = {
            "temporary": self._generate_temporary_impact(
                prices,
                volumes,
                order_flow
            ),
            "permanent": self._generate_permanent_impact(
                prices,
                volumes,
                order_flow
            ),
            "decay": self._generate_impact_decay()
        }
        
        return impact
        
    def _generate_liquidity_profile(
        self,
        timestamps: pd.DatetimeIndex,
        volumes: Dict,
        order_flow: Dict
    ) -> Dict:
        """
        Generate liquidity profile:
        - Time-varying liquidity
        - Depth distribution
        - Spread dynamics
        """
        n_steps = len(timestamps)
        profile = {
            "depth": np.zeros(n_steps),
            "spread": np.zeros(n_steps),
            "resilience": np.zeros(n_steps)
        }
        
        # Get regime parameters
        regime_params = self.regime_params[self.current_regime]
        
        for i in range(n_steps):
            # Calculate market depth
            base_depth = volumes["total"][i] * self.liquidity_scale
            inst_ratio = order_flow["institutional"]["buy"][i] / volumes["total"][i]
            
            profile["depth"][i] = (
                base_depth *
                (1 + inst_ratio) *
                regime_params["depth_multiplier"]
            )
            
            # Calculate spread
            base_spread = regime_params["base_spread"]
            vol_ratio = volumes["total"][i] / np.mean(volumes["total"])
            
            profile["spread"][i] = (
                base_spread *
                (2 - vol_ratio) *
                regime_params["spread_multiplier"]
            )
            
            # Calculate resilience
            profile["resilience"][i] = regime_params["resilience"]
            
        return profile
        
    def _generate_temporary_impact(
        self,
        prices: np.ndarray,
        volumes: Dict,
        order_flow: Dict
    ) -> Dict:
        """Generate temporary impact function"""
        regime_params = self.regime_params[self.current_regime]
        
        # Square root impact model
        def impact_func(volume, price):
            return (
                regime_params["impact_factor"] *
                price *
                np.sqrt(volume / volumes["total"].mean())
            )
            
        return {
            "function": impact_func,
            "parameters": {
                "factor": regime_params["impact_factor"],
                "power": 0.5
            }
        }
        
    def _generate_permanent_impact(
        self,
        prices: np.ndarray,
        volumes: Dict,
        order_flow: Dict
    ) -> Dict:
        """Generate permanent impact function"""
        regime_params = self.regime_params[self.current_regime]
        
        # Linear impact model
        def impact_func(volume, price):
            return (
                regime_params["permanent_impact"] *
                price *
                (volume / volumes["total"].mean())
            )
            
        return {
            "function": impact_func,
            "parameters": {
                "factor": regime_params["permanent_impact"],
                "power": 1.0
            }
        }
        
    def _generate_impact_decay(self) -> Dict:
        """Generate impact decay function"""
        regime_params = self.regime_params[self.current_regime]
        
        # Exponential decay
        def decay_func(impact, time):
            return impact * np.exp(-regime_params["decay_rate"] * time)
            
        return {
            "function": decay_func,
            "parameters": {
                "rate": regime_params["decay_rate"]
            }
        }
        
    def _default_regime_params(self) -> Dict:
        """Default regime parameters"""
        return {
            "trending": {
                "drift": 0.1,
                "volatility": 0.15,
                "jump_prob": 0.05,
                "jump_mean": 0.0,
                "jump_std": 0.02,
                "transition_prob": 0.01,
                "volume_multiplier": 1.2,
                "buy_alpha": 2.0,
                "buy_beta": 1.0,
                "institutional_ratio": 0.7,
                "impact_factor": 0.2,
                "permanent_impact": 0.1,
                "decay_rate": 0.1,
                "base_spread": 0.0001,
                "spread_multiplier": 1.0,
                "depth_multiplier": 1.2,
                "resilience": 0.8
            },
            "ranging": {
                "drift": 0.0,
                "volatility": 0.1,
                "jump_prob": 0.02,
                "jump_mean": 0.0,
                "jump_std": 0.01,
                "transition_prob": 0.02,
                "mean_reverting": True,
                "mean_level": 0.0,
                "reversion_speed": 0.1,
                "volume_multiplier": 1.0,
                "buy_alpha": 1.0,
                "buy_beta": 1.0,
                "institutional_ratio": 0.5,
                "impact_factor": 0.15,
                "permanent_impact": 0.05,
                "decay_rate": 0.2,
                "base_spread": 0.0002,
                "spread_multiplier": 1.2,
                "depth_multiplier": 1.0,
                "resilience": 0.6
            },
            "volatile": {
                "drift": 0.0,
                "volatility": 0.25,
                "jump_prob": 0.1,
                "jump_mean": 0.0,
                "jump_std": 0.03,
                "transition_prob": 0.03,
                "volume_multiplier": 1.5,
                "volume_clustering": True,
                "buy_alpha": 1.5,
                "buy_beta": 1.5,
                "institutional_ratio": 0.3,
                "hidden_liquidity": True,
                "hidden_ratio": 0.2,
                "impact_factor": 0.3,
                "permanent_impact": 0.15,
                "decay_rate": 0.05,
                "base_spread": 0.0003,
                "spread_multiplier": 1.5,
                "depth_multiplier": 0.8,
                "resilience": 0.4
            }
        }
        
    def _build_transition_matrix(self) -> np.ndarray:
        """Build regime transition probability matrix"""
        regimes = list(self.regime_params.keys())
        n_regimes = len(regimes)
        
        # Initialize matrix
        matrix = np.zeros((n_regimes, n_regimes))
        
        for i, regime in enumerate(regimes):
            # Get transition probability
            p_transition = self.regime_params[regime]["transition_prob"]
            
            # Set diagonal (probability of staying in regime)
            matrix[i,i] = 1 - p_transition
            
            # Distribute transition probability to other regimes
            p_other = p_transition / (n_regimes - 1)
            for j in range(n_regimes):
                if i != j:
                    matrix[i,j] = p_other
                    
        return matrix
        
    def _random_regime(self) -> str:
        """Select random initial regime"""
        regimes = list(self.regime_params.keys())
        return self.rng.choice(regimes)
        
    def _next_regime(self) -> str:
        """Select next regime based on transition matrix"""
        current_idx = list(self.regime_params.keys()).index(
            self.current_regime
        )
        probs = self.transition_matrix[current_idx]
        next_idx = self.rng.choice(
            len(probs),
            p=probs
        )
        return list(self.regime_params.keys())[next_idx]
        
    def _timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        if timeframe.endswith('min'):
            return int(timeframe[:-3])
        elif timeframe.endswith('H'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('D'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
            
    def _create_price_df(
        self,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Create price DataFrame"""
        return pd.DataFrame({
            'close': prices,
            'high': prices * (1 + self.rng.random(len(prices)) * 0.001),
            'low': prices * (1 - self.rng.random(len(prices)) * 0.001),
            'open': prices
        }, index=timestamps)
        
    def _create_volume_df(
        self,
        volumes: Dict,
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Create volume DataFrame"""
        return pd.DataFrame({
            'total_volume': volumes["total"],
            'buy_volume': volumes["buy"],
            'sell_volume': volumes["sell"]
        }, index=timestamps)
        
    def _create_order_flow_df(
        self,
        order_flow: Dict,
        timestamps: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Create order flow DataFrame"""
        return pd.DataFrame({
            'inst_buy': order_flow["institutional"]["buy"],
            'inst_sell': order_flow["institutional"]["sell"],
            'retail_buy': order_flow["retail"]["buy"],
            'retail_sell': order_flow["retail"]["sell"],
            'hidden': order_flow["hidden"]
        }, index=timestamps)
        
    def _generate_base_volume(
        self,
        n_steps: int,
        timeframe: str
    ) -> np.ndarray:
        """Generate base volume profile with time-of-day pattern"""
        # Create time points
        minutes = self._timeframe_minutes(timeframe)
        times = np.arange(n_steps) * minutes % (24 * 60)
        
        # Define volume patterns
        patterns = {
            "asia": {
                "center": 3 * 60,  # 3:00
                "width": 240,
                "scale": 0.7
            },
            "london": {
                "center": 10 * 60,  # 10:00
                "width": 180,
                "scale": 1.0
            },
            "ny": {
                "center": 15 * 60,  # 15:00
                "width": 180,
                "scale": 1.0
            }
        }
        
        # Generate volume profile
        base = np.ones(n_steps) * 0.3  # Base volume
        
        for session in patterns.values():
            profile = np.exp(
                -(times - session["center"])**2 /
                (2 * session["width"]**2)
            ) * session["scale"]
            base += profile
            
        return base * self.liquidity_scale
