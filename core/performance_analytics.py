# core/performance_analytics.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    recovery_factor: float
    calmar_ratio: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    time_in_market: float
    trades_per_day: float
    daily_std: float
    monthly_returns: Dict[str, float]
    regime_performance: Dict[str, Dict]
    alpha: float
    beta: float
    timestamp: datetime

class PerformanceAnalyzer:
    """
    Advanced performance analytics:
    - Comprehensive metrics
    - Real-time monitoring
    - Regime analysis
    - Risk decomposition
    - Visual analytics
    """
    def __init__(self, benchmark_data: Optional[Dict] = None):
        self.benchmark_data = benchmark_data
        self.trade_history = []
        self.equity_curve = []
        self.regime_history = []
        self.risk_metrics = defaultdict(list)
        
    def calculate_metrics(
        self,
        trade_history: List[Dict],
        equity_curve: List[float],
        regime_history: List[Dict],
        risk_free_rate: float = 0.02
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        """
        # Basic metrics
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Risk-adjusted metrics
        excess_returns = returns - (risk_free_rate / 252)
        daily_std = np.std(returns) * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        sharpe = (np.mean(excess_returns) * 252) / daily_std if daily_std > 0 else 0
        sortino = (np.mean(excess_returns) * 252) / downside_std if downside_std > 0 else 0
        
        # Drawdown analysis
        drawdowns = self._calculate_drawdowns(equity_curve)
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Trade metrics
        wins = [t for t in trade_history if t["pnl"] > 0]
        losses = [t for t in trade_history if t["pnl"] <= 0]
        
        win_rate = len(wins) / len(trade_history) if trade_history else 0
        profit_factor = (
            abs(sum(t["pnl"] for t in wins)) /
            abs(sum(t["pnl"] for t in losses))
            if losses else float("inf")
        )
        
        # Streak analysis
        consecutive = self._analyze_streaks(trade_history)
        
        # Time analysis
        time_metrics = self._analyze_time_metrics(trade_history)
        
        # Regime analysis
        regime_perf = self._analyze_regime_performance(
            trade_history,
            regime_history
        )
        
        # Risk decomposition
        risk_metrics = self._decompose_risk(returns)
        
        # Market metrics
        market_metrics = self._calculate_market_metrics(
            returns,
            self.benchmark_data
        )
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recovery_factor=total_return / max_drawdown if max_drawdown > 0 else float("inf"),
            calmar_ratio=total_return / max_drawdown if max_drawdown > 0 else float("inf"),
            avg_trade=np.mean([t["pnl"] for t in trade_history]) if trade_history else 0,
            avg_win=np.mean([t["pnl"] for t in wins]) if wins else 0,
            avg_loss=np.mean([t["pnl"] for t in losses]) if losses else 0,
            largest_win=max([t["pnl"] for t in wins]) if wins else 0,
            largest_loss=min([t["pnl"] for t in losses]) if losses else 0,
            consecutive_wins=consecutive["max_wins"],
            consecutive_losses=consecutive["max_losses"],
            time_in_market=time_metrics["time_in_market"],
            trades_per_day=time_metrics["trades_per_day"],
            daily_std=daily_std,
            monthly_returns=self._calculate_monthly_returns(equity_curve),
            regime_performance=regime_perf,
            alpha=market_metrics["alpha"],
            beta=market_metrics["beta"],
            timestamp=datetime.utcnow()
        )
        
    def create_dashboard(
        self,
        metrics: PerformanceMetrics,
        trade_history: List[Dict],
        equity_curve: List[float]
    ) -> go.Figure:
        """
        Create interactive performance dashboard
        """
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Equity Curve",
                "Drawdown",
                "Monthly Returns",
                "Win/Loss Distribution",
                "Regime Performance",
                "Risk Decomposition"
            )
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                y=equity_curve,
                name="Equity",
                line=dict(color="blue")
            ),
            row=1,
            col=1
        )
        
        # Drawdown
        drawdowns = self._calculate_drawdowns(equity_curve)
        fig.add_trace(
            go.Scatter(
                y=drawdowns,
                name="Drawdown",
                line=dict(color="red")
            ),
            row=1,
            col=2
        )
        
        # Monthly returns
        months = list(metrics.monthly_returns.keys())
        returns = list(metrics.monthly_returns.values())
        fig.add_trace(
            go.Bar(
                x=months,
                y=returns,
                name="Monthly Returns"
            ),
            row=2,
            col=1
        )
        
        # Win/Loss distribution
        wins = [t["pnl"] for t in trade_history if t["pnl"] > 0]
        losses = [t["pnl"] for t in trade_history if t["pnl"] <= 0]
        fig.add_trace(
            go.Histogram(
                x=wins,
                name="Wins",
                nbinsx=20,
                marker_color="green"
            ),
            row=2,
            col=2
        )
        fig.add_trace(
            go.Histogram(
                x=losses,
                name="Losses",
                nbinsx=20,
                marker_color="red"
            ),
            row=2,
            col=2
        )
        
        # Regime performance
        regimes = list(metrics.regime_performance.keys())
        regime_returns = [
            metrics.regime_performance[r]["return"]
            for r in regimes
        ]
        fig.add_trace(
            go.Bar(
                x=regimes,
                y=regime_returns,
                name="Regime Returns"
            ),
            row=3,
            col=1
        )
        
        # Risk decomposition
        risk_sources = ["Market", "Specific", "Factor"]
        risk_contrib = [0.4, 0.3, 0.3]  # Example values
        fig.add_trace(
            go.Pie(
                labels=risk_sources,
                values=risk_contrib,
                name="Risk Sources"
            ),
            row=3,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Performance Analytics Dashboard"
        )
        
        return fig
        
    def _calculate_drawdowns(
        self,
        equity_curve: List[float]
    ) -> List[float]:
        """Calculate drawdown series"""
        peaks = pd.Series(equity_curve).expanding(min_periods=1).max()
        drawdowns = (pd.Series(equity_curve) - peaks) / peaks
        return drawdowns.tolist()
        
    def _analyze_streaks(
        self,
        trade_history: List[Dict]
    ) -> Dict:
        """Analyze winning and losing streaks"""
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in trade_history:
            if trade["pnl"] > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
                
        return {
            "max_wins": max_win_streak,
            "max_losses": max_loss_streak
        }
        
    def _analyze_time_metrics(
        self,
        trade_history: List[Dict]
    ) -> Dict:
        """Calculate time-based metrics"""
        if not trade_history:
            return {"time_in_market": 0, "trades_per_day": 0}
            
        # Calculate total trading duration
        start_time = min(t["entry_time"] for t in trade_history)
        end_time = max(t["exit_time"] for t in trade_history)
        total_days = (end_time - start_time).days + 1
        
        # Calculate time in market
        in_market_time = sum(
            (t["exit_time"] - t["entry_time"]).total_seconds()
            for t in trade_history
        )
        time_in_market = in_market_time / (total_days * 24 * 3600)
        
        # Calculate trades per day
        trades_per_day = len(trade_history) / total_days
        
        return {
            "time_in_market": time_in_market,
            "trades_per_day": trades_per_day
        }
        
    def _analyze_regime_performance(
        self,
        trade_history: List[Dict],
        regime_history: List[Dict]
    ) -> Dict:
        """Analyze performance by market regime"""
        regime_performance = defaultdict(lambda: {
            "trades": 0,
            "wins": 0,
            "pnl": 0,
            "return": 0,
            "sharpe": 0
        })
        
        for trade, regime in zip(trade_history, regime_history):
            r = regime["regime"]
            regime_performance[r]["trades"] += 1
            regime_performance[r]["wins"] += 1 if trade["pnl"] > 0 else 0
            regime_performance[r]["pnl"] += trade["pnl"]
            
        # Calculate regime-specific metrics
        for regime in regime_performance:
            trades = regime_performance[regime]["trades"]
            if trades > 0:
                regime_performance[regime]["return"] = (
                    regime_performance[regime]["pnl"] /
                    trades
                )
                
        return dict(regime_performance)
        
    def _decompose_risk(
        self,
        returns: np.ndarray
    ) -> Dict:
        """
        Decompose risk into:
        - Market risk
        - Specific risk
        - Factor risk
        """
        # Calculate components (example implementation)
        total_risk = np.std(returns)
        market_risk = total_risk * 0.4  # Example allocation
        specific_risk = total_risk * 0.3
        factor_risk = total_risk * 0.3
        
        return {
            "total_risk": total_risk,
            "market_risk": market_risk,
            "specific_risk": specific_risk,
            "factor_risk": factor_risk
        }
        
    def _calculate_monthly_returns(
        self,
        equity_curve: List[float]
    ) -> Dict[str, float]:
        """Calculate monthly returns"""
        # Convert to pandas series
        equity = pd.Series(equity_curve)
        
        # Resample to monthly returns
        monthly_returns = (
            equity.pct_change()
            .resample("M")
            .agg(lambda x: (1 + x).prod() - 1)
        )
        
        return monthly_returns.to_dict()
        
    def _calculate_market_metrics(
        self,
        returns: np.ndarray,
        benchmark_data: Optional[Dict]
    ) -> Dict:
        """Calculate market-relative metrics"""
        if benchmark_data is None:
            return {"alpha": 0, "beta": 1}
            
        # Calculate beta
        benchmark_returns = np.array(benchmark_data["returns"])
        covariance = np.cov(returns, benchmark_returns)[0,1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance if variance > 0 else 1
        
        # Calculate alpha
        alpha = (
            np.mean(returns) -
            beta * np.mean(benchmark_returns)
        ) * 252
        
        return {
            "alpha": alpha,
            "beta": beta
        }
        
    def generate_report(
        self,
        metrics: PerformanceMetrics,
        include_plots: bool = True
    ) -> Dict:
        """
        Generate comprehensive performance report
        """
        report = {
            "summary": {
                "total_return": f"{metrics.total_return:.2%}",
                "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                "max_drawdown": f"{metrics.max_drawdown:.2%}",
                "win_rate": f"{metrics.win_rate:.2%}"
            },
            "risk_metrics": {
                "sortino_ratio": f"{metrics.sortino_ratio:.2f}",
                "calmar_ratio": f"{metrics.calmar_ratio:.2f}",
                "daily_std": f"{metrics.daily_std:.2%}",
                "beta": f"{metrics.beta:.2f}"
            },
            "trade_metrics": {
                "avg_trade": f"${metrics.avg_trade:.2f}",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "trades_per_day": f"{metrics.trades_per_day:.1f}",
                "time_in_market": f"{metrics.time_in_market:.2%}"
            },
            "regime_analysis": metrics.regime_performance,
            "monthly_returns": metrics.monthly_returns
        }
        
        if include_plots:
            report["plots"] = {
                "equity_curve": self._plot_equity_curve(metrics),
                "drawdown": self._plot_drawdown(metrics),
                "monthly_returns": self._plot_monthly_returns(metrics),
                "regime_performance": self._plot_regime_performance(metrics)
            }
            
        return report
        
    def _plot_equity_curve(
        self,
        metrics: PerformanceMetrics
    ) -> go.Figure:
        """Create equity curve plot"""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=self.equity_curve,
                name="Equity",
                line=dict(color="blue")
            )
        )
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Trade",
            yaxis_title="Equity"
        )
        return fig
        
    def _plot_drawdown(
        self,
        metrics: PerformanceMetrics
    ) -> go.Figure:
        """Create drawdown plot"""
        drawdowns = self._calculate_drawdowns(self.equity_curve)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=drawdowns,
                name="Drawdown",
                line=dict(color="red")
            )
        )
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Trade",
            yaxis_title="Drawdown"
        )
        return fig
        
    def _plot_monthly_returns(
        self,
        metrics: PerformanceMetrics
    ) -> go.Figure:
        """Create monthly returns plot"""
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=list(metrics.monthly_returns.keys()),
                y=list(metrics.monthly_returns.values()),
                name="Monthly Returns"
            )
        )
        fig.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="Return"
        )
        return fig
        
    def _plot_regime_performance(
        self,
        metrics: PerformanceMetrics
    ) -> go.Figure:
        """Create regime performance plot"""
        fig = go.Figure()
        regimes = list(metrics.regime_performance.keys())
        returns = [
            metrics.regime_performance[r]["return"]
            for r in regimes
        ]
        fig.add_trace(
            go.Bar(
                x=regimes,
                y=returns,
                name="Regime Returns"
            )
        )
        fig.update_layout(
            title="Performance by Regime",
            xaxis_title="Regime",
            yaxis_title="Return"
        )
        return fig
