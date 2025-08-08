# core/strategy_optimizer.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ProcessPoolExecutor
import logging
from collections import defaultdict

@dataclass
class OptimizationResult:
    optimal_params: Dict
    performance_metrics: Dict
    optimization_path: List[Dict]
    cross_validation: Dict
    regime_specific: Dict
    robustness_score: float
    timestamp: datetime

class StrategyOptimizer:
    """
    Advanced strategy optimization:
    - Multi-objective optimization
    - Walk-forward analysis
    - Regime-specific optimization
    - Parameter sensitivity
    - Robustness testing
    """
    def __init__(
        self,
        strategy_evaluator: Callable,
        parameter_space: Dict,
        objectives: List[str],
        constraints: Optional[List[Dict]] = None,
        config: Optional[Dict] = None
    ):
        self.strategy_evaluator = strategy_evaluator
        self.parameter_space = parameter_space
        self.objectives = objectives
        self.constraints = constraints or []
        self.config = config or {}
        
        # Optimization tracking
        self.optimization_history = []
        self.best_solutions = []
        self.sensitivity_analysis = {}
        
        # Performance tracking
        self.cross_validation_results = {}
        self.regime_performance = defaultdict(list)
        
        # Setup logging
        self._setup_logging()
        
    def optimize(
        self,
        data: Dict,
        initial_params: Optional[Dict] = None,
        method: str = "differential_evolution"
    ) -> OptimizationResult:
        """
        Run comprehensive strategy optimization:
        - Multi-objective optimization
        - Walk-forward validation
        - Regime-specific optimization
        - Robustness analysis
        """
        self.logger.info("Starting strategy optimization...")
        
        # Initial optimization
        base_results = self._run_base_optimization(
            data,
            initial_params,
            method
        )
        
        # Walk-forward validation
        wf_results = self._walk_forward_validation(
            data,
            base_results["optimal_params"]
        )
        
        # Regime-specific optimization
        regime_results = self._optimize_by_regime(
            data,
            base_results["optimal_params"]
        )
        
        # Robustness analysis
        robustness = self._analyze_robustness(
            data,
            base_results["optimal_params"]
        )
        
        # Combine results
        final_results = OptimizationResult(
            optimal_params=base_results["optimal_params"],
            performance_metrics=base_results["metrics"],
            optimization_path=base_results["path"],
            cross_validation=wf_results,
            regime_specific=regime_results,
            robustness_score=robustness["score"],
            timestamp=datetime.utcnow()
        )
        
        self.logger.info("Optimization completed successfully")
        return final_results
        
    def _run_base_optimization(
        self,
        data: Dict,
        initial_params: Optional[Dict],
        method: str
    ) -> Dict:
        """
        Run main optimization process:
        - Multi-objective optimization
        - Parameter constraints
        - Convergence tracking
        """
        if method == "differential_evolution":
            results = self._run_differential_evolution(data)
        else:
            results = self._run_gradient_descent(
                data,
                initial_params
            )
            
        return results
        
    def _run_differential_evolution(
        self,
        data: Dict
    ) -> Dict:
        """
        Run differential evolution optimization:
        - Population-based search
        - Global optimization
        - Constraint handling
        """
        # Prepare bounds
        bounds = [
            (space["min"], space["max"])
            for space in self.parameter_space.values()
        ]
        
        # Optimization function
        def objective(x):
            params = dict(zip(
                self.parameter_space.keys(),
                x
            ))
            return -self._evaluate_strategy(
                params,
                data
            )["fitness"]
            
        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating='immediate',
            workers=-1
        )
        
        # Extract results
        optimal_params = dict(zip(
            self.parameter_space.keys(),
            result.x
        ))
        
        return {
            "optimal_params": optimal_params,
            "metrics": self._evaluate_strategy(
                optimal_params,
                data
            ),
            "path": self._extract_optimization_path(result)
        }
        
    def _run_gradient_descent(
        self,
        data: Dict,
        initial_params: Dict
    ) -> Dict:
        """
        Run gradient-based optimization:
        - Local optimization
        - Fast convergence
        - Constraint handling
        """
        # Prepare initial guess
        x0 = [
            initial_params.get(
                param,
                space["default"]
            )
            for param, space in self.parameter_space.items()
        ]
        
        # Optimization function
        def objective(x):
            params = dict(zip(
                self.parameter_space.keys(),
                x
            ))
            return -self._evaluate_strategy(
                params,
                data
            )["fitness"]
            
        # Run optimization
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[
                (space["min"], space["max"])
                for space in self.parameter_space.values()
            ],
            constraints=self.constraints
        )
        
        # Extract results
        optimal_params = dict(zip(
            self.parameter_space.keys(),
            result.x
        ))
        
        return {
            "optimal_params": optimal_params,
            "metrics": self._evaluate_strategy(
                optimal_params,
                data
            ),
            "path": self._extract_optimization_path(result)
        }
        
    def _walk_forward_validation(
        self,
        data: Dict,
        params: Dict
    ) -> Dict:
        """
        Perform walk-forward validation:
        - Time series cross-validation
        - Performance stability
        - Parameter stability
        """
        results = {
            "performance": [],
            "parameter_stability": {},
            "regime_performance": defaultdict(list)
        }
        
        # Create time series splits
        tscv = TimeSeriesSplit(
            n_splits=5,
            test_size=int(len(data) * 0.2)
        )
        
        # Run walk-forward optimization
        for train_idx, test_idx in tscv.split(data):
            # Split data
            train_data = self._split_data(data, train_idx)
            test_data = self._split_data(data, test_idx)
            
            # Optimize on train data
            train_results = self._run_base_optimization(
                train_data,
                params,
                "differential_evolution"
            )
            
            # Evaluate on test data
            test_metrics = self._evaluate_strategy(
                train_results["optimal_params"],
                test_data
            )
            
            # Record results
            results["performance"].append({
                "train_metrics": train_results["metrics"],
                "test_metrics": test_metrics,
                "params": train_results["optimal_params"]
            })
            
            # Record regime performance
            for regime, perf in test_metrics["regime_performance"].items():
                results["regime_performance"][regime].append(perf)
                
        # Calculate parameter stability
        results["parameter_stability"] = self._analyze_parameter_stability(
            results["performance"]
        )
        
        return results
        
    def _optimize_by_regime(
        self,
        data: Dict,
        base_params: Dict
    ) -> Dict:
        """
        Optimize parameters for each market regime:
        - Regime-specific optimization
        - Performance comparison
        - Transition analysis
        """
        results = {
            "regime_params": {},
            "performance_comparison": {},
            "transition_analysis": {}
        }
        
        # Split data by regime
        regime_data = self._split_by_regime(data)
        
        # Optimize for each regime
        for regime, regime_data in regime_data.items():
            # Run optimization
            regime_results = self._run_base_optimization(
                regime_data,
                base_params,
                "differential_evolution"
            )
            
            results["regime_params"][regime] = regime_results["optimal_params"]
            results["performance_comparison"][regime] = {
                "regime_specific": regime_results["metrics"],
                "base_performance": self._evaluate_strategy(
                    base_params,
                    regime_data
                )
            }
            
        # Analyze regime transitions
        results["transition_analysis"] = self._analyze_regime_transitions(
            data,
            results["regime_params"]
        )
        
        return results
        
    def _analyze_robustness(
        self,
        data: Dict,
        params: Dict
    ) -> Dict:
        """
        Analyze strategy robustness:
        - Parameter sensitivity
        - Monte Carlo simulation
        - Stress testing
        """
        results = {
            "sensitivity": self._analyze_sensitivity(
                data,
                params
            ),
            "monte_carlo": self._run_monte_carlo(
                data,
                params
            ),
            "stress_test": self._run_stress_tests(
                data,
                params
            )
        }
        
        # Calculate overall robustness score
        results["score"] = self._calculate_robustness_score(results)
        
        return results
        
    def _analyze_sensitivity(
        self,
        data: Dict,
        params: Dict
    ) -> Dict:
        """
        Analyze parameter sensitivity:
        - Local sensitivity
        - Global sensitivity
        - Interaction effects
        """
        sensitivity = {}
        
        # Analyze each parameter
        for param, value in params.items():
            # Local sensitivity
            local_sensitivity = self._analyze_local_sensitivity(
                data,
                params,
                param
            )
            
            # Global sensitivity
            global_sensitivity = self._analyze_global_sensitivity(
                data,
                params,
                param
            )
            
            sensitivity[param] = {
                "local": local_sensitivity,
                "global": global_sensitivity,
                "importance": (
                    local_sensitivity["score"] +
                    global_sensitivity["score"]
                ) / 2
            }
            
        return sensitivity
        
    def _run_monte_carlo(
        self,
        data: Dict,
        params: Dict
    ) -> Dict:
        """
        Run Monte Carlo simulation:
        - Parameter perturbation
        - Market scenario generation
        - Distribution analysis
        """
        results = {
            "parameter_stability": [],
            "performance_distribution": [],
            "risk_metrics": {}
        }
        
        # Run simulations
        n_sims = 1000
        with ProcessPoolExecutor() as executor:
            futures = []
            for _ in range(n_sims):
                # Perturb parameters
                perturbed_params = self._perturb_parameters(params)
                
                # Generate market scenario
                scenario = self._generate_market_scenario(data)
                
                # Submit simulation
                futures.append(
                    executor.submit(
                        self._evaluate_strategy,
                        perturbed_params,
                        scenario
                    )
                )
                
        # Collect results
        for future in futures:
            results["performance_distribution"].append(
                future.result()
            )
            
        # Calculate risk metrics
        results["risk_metrics"] = self._calculate_risk_metrics(
            results["performance_distribution"]
        )
        
        return results
        
    def _run_stress_tests(
        self,
        data: Dict,
        params: Dict
    ) -> Dict:
        """
        Run stress tests:
        - Extreme market scenarios
        - Parameter stress
        - System stress
        """
        results = {
            "market_stress": self._test_market_stress(
                data,
                params
            ),
            "parameter_stress": self._test_parameter_stress(
                data,
                params
            ),
            "system_stress": self._test_system_stress(
                data,
                params
            )
        }
        
        return results
        
    def _evaluate_strategy(
        self,
        params: Dict,
        data: Dict
    ) -> Dict:
        """
        Evaluate strategy with given parameters:
        - Calculate performance metrics
        - Apply constraints
        - Compute fitness
        """
        # Run strategy evaluation
        metrics = self.strategy_evaluator(params, data)
        
        # Calculate fitness
        fitness = self._calculate_fitness(metrics)
        
        # Apply penalties for constraint violations
        if self.constraints:
            penalties = self._calculate_penalties(params, metrics)
            fitness -= sum(penalties)
            
        return {
            "metrics": metrics,
            "fitness": fitness,
            "constraint_violations": bool(penalties) if self.constraints else False
        }
        
    def _calculate_fitness(self, metrics: Dict) -> float:
        """
        Calculate multi-objective fitness:
        - Weighted combination of objectives
        - Constraint satisfaction
        - Risk adjustment
        """
        fitness = 0.0
        
        # Add objective components
        for objective in self.objectives:
            if objective == "sharpe":
                fitness += metrics["sharpe_ratio"] * 0.4
            elif objective == "returns":
                fitness += metrics["total_return"] * 0.3
            elif objective == "drawdown":
                fitness += (1 / (1 + metrics["max_drawdown"])) * 0.2
            elif objective == "consistency":
                fitness += metrics["consistency_score"] * 0.1
                
        return fitness
        
    def _calculate_penalties(
        self,
        params: Dict,
        metrics: Dict
    ) -> List[float]:
        """Calculate constraint violation penalties"""
        penalties = []
        
        for constraint in self.constraints:
            if constraint["type"] == "param_range":
                param = constraint["param"]
                if params[param] < constraint["min"]:
                    penalties.append(
                        abs(params[param] - constraint["min"]) *
                        constraint["penalty"]
                    )
                elif params[param] > constraint["max"]:
                    penalties.append(
                        abs(params[param] - constraint["max"]) *
                        constraint["penalty"]
                    )
            elif constraint["type"] == "metric_threshold":
                metric = constraint["metric"]
                if metrics[metric] < constraint["min"]:
                    penalties.append(
                        abs(metrics[metric] - constraint["min"]) *
                        constraint["penalty"]
                    )
                    
        return penalties
        
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("StrategyOptimizer")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler("optimization.log")
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        
    def save_results(
        self,
        results: OptimizationResult,
        path: str
    ):
        """Save optimization results"""
        import json
        
        output = {
            "optimal_params": results.optimal_params,
            "performance_metrics": results.performance_metrics,
            "optimization_path": results.optimization_path,
            "cross_validation": results.cross_validation,
            "regime_specific": results.regime_specific,
            "robustness_score": results.robustness_score,
            "timestamp": results.timestamp.isoformat()
        }
        
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
            
    def load_results(self, path: str) -> OptimizationResult:
        """Load optimization results"""
        import json
        
        with open(path, "r") as f:
            data = json.load(f)
            
        return OptimizationResult(
            optimal_params=data["optimal_params"],
            performance_metrics=data["performance_metrics"],
            optimization_path=data["optimization_path"],
            cross_validation=data["cross_validation"],
            regime_specific=data["regime_specific"],
            robustness_score=data["robustness_score"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
