# core/backtesting_engine.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm
import json
import logging
from pathlib import Path

@dataclass
class BacktestResult:
    performance_metrics: Dict
    trade_history: List[Dict]
    equity_curve: List[float]
    drawdown_curve: List[float]
    regime_performance: Dict
    ml_metrics: Dict
    optimization_results: Optional[Dict]
    timestamp: datetime

class AdvancedBacktester:
    """
    Institutional-grade backtesting engine:
    - Walk-forward optimization
    - Monte Carlo simulation
    - Market regime testing
    - ML model validation
    - Strategy robustness analysis
    """
    def __init__(
        self,
        data_loader: callable,
        signal_generator: callable,
        risk_manager: callable,
        execution_simulator: callable,
        ml_validator: callable,
        config: Dict
    ):
        self.data_loader = data_loader
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.execution_simulator = execution_simulator
        self.ml_validator = ml_validator
        self.config = config
        
        # Performance tracking
        self.results_history = []
        self.optimization_history = []
        self.validation_metrics = {}
        
        # Market regime detection
        self.regime_classifier = None
        self.regime_transitions = []
        
        # ML components
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize logging
        self._setup_logging()
        
    def run_full_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        **kwargs
    ) -> BacktestResult:
        """
        Run comprehensive backtest with:
        - Multi-timeframe analysis
        - Market regime detection
        - ML model validation
        - Monte Carlo simulation
        """
        self.logger.info("Starting comprehensive backtest...")
        
        # Load and prepare data
        data = self._prepare_data(symbols, start_date, end_date)
        
        # Detect market regimes
        regimes = self._detect_market_regimes(data)
        
        # Run main backtest
        base_results = self._run_base_backtest(
            data,
            regimes,
            initial_capital
        )
        
        # Validate ML models
        ml_validation = self._validate_ml_models(
            data,
            base_results["trade_history"]
        )
        
        # Run Monte Carlo simulation
        monte_carlo = self._run_monte_carlo(
            base_results["trade_history"],
            initial_capital,
            iterations=1000
        )
        
        # Optimize parameters (if requested)
        optimization = None
        if kwargs.get("optimize", False):
            optimization = self._optimize_parameters(
                data,
                base_results,
                regimes
            )
            
        # Combine results
        final_results = self._combine_results(
            base_results,
            ml_validation,
            monte_carlo,
            optimization
        )
        
        self.logger.info("Backtest completed successfully")
        return final_results
        
    def _prepare_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Prepare data for backtesting:
        - Load multi-timeframe data
        - Calculate technical features
        - Prepare market microstructure
        """
        data = {}
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for symbol in symbols:
                for tf in ["M5", "M15", "H1", "H4", "D1"]:
                    futures.append(
                        executor.submit(
                            self.data_loader,
                            symbol,
                            tf,
                            start_date,
                            end_date
                        )
                    )
                    
            # Collect results
            for future in futures:
                result = future.result()
                if result:
                    key = f"{result['symbol']}_{result['timeframe']}"
                    data[key] = result
                    
        return data
        
    def _detect_market_regimes(self, data: Dict) -> Dict:
        """
        Detect market regimes using:
        - Volatility clustering
        - Trend strength
        - Market microstructure
        - Volume analysis
        """
        regimes = {}
        
        for key, market_data in data.items():
            # Extract features
            features = self._extract_regime_features(market_data)
            
            # Classify regimes
            if self.regime_classifier is None:
                self._train_regime_classifier(features)
                
            regimes[key] = self._classify_regimes(features)
            
            # Detect regime transitions
            transitions = self._detect_regime_transitions(regimes[key])
            self.regime_transitions.extend(transitions)
            
        return regimes
        
    def _run_base_backtest(
        self,
        data: Dict,
        regimes: Dict,
        initial_capital: float
    ) -> Dict:
        """
        Run main backtest with:
        - Multi-timeframe signals
        - Portfolio management
        - Risk adjustment
        - Execution simulation
        """
        results = {
            "trade_history": [],
            "equity_curve": [initial_capital],
            "drawdown_curve": [0.0],
            "regime_performance": defaultdict(list)
        }
        
        current_capital = initial_capital
        max_capital = initial_capital
        
        # Iterate through time
        timestamps = self._get_aligned_timestamps(data)
        
        for timestamp in timestamps:
            # Get current market state
            market_state = self._get_market_state(
                data,
                regimes,
                timestamp
            )
            
            # Generate signals
            signals = self.signal_generator(
                data,
                market_state,
                timestamp
            )
            
            # Apply risk management
            positions = self.risk_manager(
                signals,
                current_capital,
                market_state
            )
            
            # Simulate execution
            execution_results = self.execution_simulator(
                positions,
                data,
                timestamp
            )
            
            # Update results
            current_capital += execution_results["pnl"]
            max_capital = max(max_capital, current_capital)
            drawdown = (max_capital - current_capital) / max_capital
            
            results["equity_curve"].append(current_capital)
            results["drawdown_curve"].append(drawdown)
            
            # Record trades
            for trade in execution_results["trades"]:
                trade["regime"] = market_state["regime"]
                results["trade_history"].append(trade)
                results["regime_performance"][market_state["regime"]].append(
                    trade["pnl"]
                )
                
        return results
        
    def _validate_ml_models(
        self,
        data: Dict,
        trade_history: List[Dict]
    ) -> Dict:
        """
        Validate ML models with:
        - Out-of-sample testing
        - Feature importance
        - Prediction accuracy
        - Model stability
        """
        validation = {
            "model_performance": {},
            "feature_importance": {},
            "prediction_accuracy": {},
            "stability_metrics": {}
        }
        
        # Prepare validation data
        train_data, test_data = self._prepare_validation_data(
            data,
            trade_history
        )
        
        # Validate each model
        for model_name, model in self.ml_validator.models.items():
            # Out-of-sample testing
            oos_performance = self._test_model_oos(
                model,
                test_data
            )
            validation["model_performance"][model_name] = oos_performance
            
            # Feature importance
            importance = self._analyze_feature_importance(
                model,
                train_data
            )
            validation["feature_importance"][model_name] = importance
            
            # Prediction accuracy
            accuracy = self._analyze_prediction_accuracy(
                model,
                test_data
            )
            validation["prediction_accuracy"][model_name] = accuracy
            
            # Model stability
            stability = self._analyze_model_stability(
                model,
                train_data,
                test_data
            )
            validation["stability_metrics"][model_name] = stability
            
        return validation
        
    def _run_monte_carlo(
        self,
        trade_history: List[Dict],
        initial_capital: float,
        iterations: int = 1000
    ) -> Dict:
        """
        Run Monte Carlo simulation:
        - Trade sequence randomization
        - Win/loss distribution analysis
        - Risk metrics calculation
        - Confidence intervals
        """
        results = {
            "metrics": {},
            "distributions": {},
            "confidence_intervals": {},
            "risk_metrics": {}
        }
        
        # Extract trade characteristics
        returns = [t["pnl"] / t["capital"] for t in trade_history]
        
        # Run simulations
        equity_curves = []
        max_drawdowns = []
        final_capitals = []
        
        for _ in range(iterations):
            # Randomize trade sequence
            sim_returns = np.random.choice(
                returns,
                size=len(returns),
                replace=True
            )
            
            # Calculate equity curve
            equity = initial_capital
            equity_curve = [initial_capital]
            max_equity = initial_capital
            max_drawdown = 0
            
            for ret in sim_returns:
                equity *= (1 + ret)
                equity_curve.append(equity)
                max_equity = max(max_equity, equity)
                drawdown = (max_equity - equity) / max_equity
                max_drawdown = max(max_drawdown, drawdown)
                
            equity_curves.append(equity_curve)
            max_drawdowns.append(max_drawdown)
            final_capitals.append(equity)
            
        # Calculate metrics
        results["metrics"] = {
            "mean_final_capital": np.mean(final_capitals),
            "std_final_capital": np.std(final_capitals),
            "mean_max_drawdown": np.mean(max_drawdowns),
            "worst_drawdown": np.max(max_drawdowns)
        }
        
        # Calculate distributions
        results["distributions"] = {
            "final_capital": self._calculate_distribution(final_capitals),
            "max_drawdown": self._calculate_distribution(max_drawdowns)
        }
        
        # Calculate confidence intervals
        results["confidence_intervals"] = {
            "final_capital_95": np.percentile(final_capitals, [2.5, 97.5]),
            "max_drawdown_95": np.percentile(max_drawdowns, [2.5, 97.5])
        }
        
        # Calculate risk metrics
        results["risk_metrics"] = {
            "var_95": np.percentile(final_capitals, 5),
            "cvar_95": np.mean(
                [x for x in final_capitals if x <= np.percentile(final_capitals, 5)]
            ),
            "probability_profit": np.mean([x > initial_capital for x in final_capitals])
        }
        
        return results
        
    def _optimize_parameters(
        self,
        data: Dict,
        base_results: Dict,
        regimes: Dict
    ) -> Dict:
        """
        Optimize strategy parameters:
        - Walk-forward optimization
        - Regime-specific optimization
        - Parameter sensitivity analysis
        - Cross-validation
        """
        optimization = {
            "optimal_params": {},
            "regime_params": {},
            "sensitivity": {},
            "cross_validation": {}
        }
        
        # Define parameter space
        param_space = self._define_parameter_space()
        
        # Walk-forward optimization
        wfo_results = self._walk_forward_optimization(
            data,
            param_space,
            regimes
        )
        optimization["optimal_params"] = wfo_results["optimal_params"]
        
        # Regime-specific optimization
        regime_optimization = self._optimize_by_regime(
            data,
            param_space,
            regimes
        )
        optimization["regime_params"] = regime_optimization
        
        # Parameter sensitivity
        sensitivity = self._analyze_parameter_sensitivity(
            data,
            optimization["optimal_params"]
        )
        optimization["sensitivity"] = sensitivity
        
        # Cross-validation
        cv_results = self._cross_validate_parameters(
            data,
            optimization["optimal_params"],
            regimes
        )
        optimization["cross_validation"] = cv_results
        
        return optimization
        
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("Backtester")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler("backtest.log")
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        
    def _calculate_distribution(self, data: List[float]) -> Dict:
        """Calculate distribution statistics"""
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "skew": stats.skew(data),
            "kurtosis": stats.kurtosis(data),
            "percentiles": np.percentile(data, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        }
        
    def save_results(self, results: BacktestResult, path: str):
        """Save backtest results to file"""
        output = {
            "performance_metrics": results.performance_metrics,
            "trade_history": results.trade_history,
            "equity_curve": results.equity_curve,
            "drawdown_curve": results.drawdown_curve,
            "regime_performance": results.regime_performance,
            "ml_metrics": results.ml_metrics,
            "optimization_results": results.optimization_results,
            "timestamp": results.timestamp.isoformat()
        }
        
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
            
    def load_results(self, path: str) -> BacktestResult:
        """Load backtest results from file"""
        with open(path, "r") as f:
            data = json.load(f)
            
        return BacktestResult(
            performance_metrics=data["performance_metrics"],
            trade_history=data["trade_history"],
            equity_curve=data["equity_curve"],
            drawdown_curve=data["drawdown_curve"],
            regime_performance=data["regime_performance"],
            ml_metrics=data["ml_metrics"],
            optimization_results=data["optimization_results"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
