# main.py - Unified Advanced Trading System
# Combines the best of both main.py and main_new.py

import time
import os
import MetaTrader5 as mt5
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

# Core imports
from core.signal_engine import AdvancedSignalEngine
from core.structure_engine import StructureEngine
from core.zone_engine import ZoneEngine
from core.liquidity_filter import LiquidityFilter
from core.order_flow_engine import OrderFlowEngine
from core.dashboard_logger import DashboardLogger
from core.visual_playbook import VisualPlaybook
from core.order_flow_visualizer import OrderFlowVisualizer
from core.situational_analysis import SituationalAnalyzer
from core.multi_timeframe import MultiTimeframeAnalyzer
from core.trade_executor import execute_trade
from core.risk_manager import AdaptiveRiskManager
from core.smart_exit import AdaptiveExitManager
from core.risk_overlay import GlobalRiskOverlay
from memory.learning import AdvancedLearningEngine
from core.prophetic_layer import AdvancedPropheticEngine
from core.prophetic_ml import PropheticMLEngine
from core.telegram_notifier import TelegramNotifier
from core.intelligence import IntelligenceCore
from core.cisd_engine import CISDEngine
from core.policy_service import PolicyService
from core.ml_tracker import ml_tracker
from core.drift_monitor import drift_monitor
from core.online_learner import online_learner
from core.model_manager import model_manager
from core.theory_features import compute_theory_features

# New connector system
from execution.connectors.base import BaseConnector
from execution.connectors.paper import PaperConnector
try:
    from execution.connectors.mt5 import MT5Connector
except Exception:
    MT5Connector = None
try:
    from execution.connectors.binance import BinanceConnector
except Exception:
    BinanceConnector = None

# Legacy MT5 utilities (for backward compatibility)
from utils.mt5_connector import initialize, get_candles, fetch_latest_data, shutdown
from utils.session_timer import is_in_liquidity_window
from utils.helpers import calculate_rr
from utils.config import (
    SYMBOL, TIMEFRAME, DATA_COUNT, BASE_RISK, START_BALANCE,
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    ENABLE_ML_LEARNING, ML_CONFIDENCE_THRESHOLD,
    DISCORD_WEBHOOK, DISCORD_USERNAME, DISCORD_AVATAR,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    ENABLE_GLOBAL_OVERLAY, OVERLAY_MAX_DRAWDOWN, OVERLAY_WARN_DRAWDOWN,
    OVERLAY_VOL_THROTTLE_HIGH, OVERLAY_BASE_THROTTLE
)
from utils.pair_config import TRADING_PAIRS
from monitor.dashboard import update_dashboard
from brokers.mt5_adapter import MT5Adapter
from core.exec_engine import ExecEngine
from utils.perf_logger import snapshot as perf_snapshot
from utils.feature_store import FeatureStore
from utils.feature_store import FeatureStore

# New system imports
from utils.config import cfg
from utils.logging_setup import setup_logger
from utils.execution_filters import within_spread_limit, within_slippage_limit
from risk.rules import RiskRules
from core.risk_model import RiskModel
from utils.perf_logger import on_trade_close as perf_on_close, set_equity as perf_set_eq

def load_connector(exec_cfg):
    """Load the appropriate connector based on configuration"""
    driver = exec_cfg.get("driver", "paper").lower()
    if driver == "paper": 
        return PaperConnector()
    if driver == "mt5" and MT5Connector: 
        return MT5Connector()
    if driver == "binance" and BinanceConnector: 
        return BinanceConnector()
    raise RuntimeError(f"Unknown or unavailable driver: {driver}")

def direction_from_score(score): 
    return "buy" if score >= 0.0 else "sell"

def startup_self_check():
    """Comprehensive startup validation"""
    print("🔍 Performing comprehensive startup validation...")
    
    try:
        from core.system_validator import run_system_validation
        
        # Run full system validation
        validation_passed = run_system_validation()
        
        if not validation_passed:
            print("❌ System validation failed - check logs/validation/ for details")
            return False
        
        # Quick runtime checks for initialized components
        try:
            # Check signal engine exists and has methods
            if 'signal_engine' in globals() and hasattr(signal_engine, 'generate_signal'):
                print("✅ Signal engine validated")
            
            # Check learning engine
            if 'learning_engine' in globals() and hasattr(learning_engine, 'suggest_confidence'):
                print("✅ Learning engine validated")
            
            # Check ML components
            from core.policy_service import PolicyService
            from core.ml_tracker import ml_tracker
            from core.online_learner import online_learner
            from core.model_manager import model_manager
            print("✅ ML components loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Runtime component check warning: {e}")
            # Don't fail for runtime checks - components may not be initialized yet
        
        print("✅ Comprehensive startup validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Startup validation failed: {e}")
        return False

def main():
    """Main trading loop with unified architecture"""
    print("🚀 Initializing Unified Advanced Trading System...")
    
    # Load configuration
    try:
        config = cfg()
        print("✅ Configuration loaded successfully")
except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        # Fallback to legacy config
        config = {
            "execution": {"driver": "mt5", "symbols": [SYMBOL]},
            "mode": {"autonomous": True, "require_all_confirm": False},
            "hybrid": {"entry_threshold_base": 0.62},
            "risk": {"daily_loss_cap": 0.015, "weekly_dd_brake": 0.04},
            "filters": {"max_spread_pips": 5.0, "max_slippage_pips": 2.0}
        }
        print("⚠️ Using fallback configuration")
    
    # Setup logging
    logger = setup_logger("main")
    
    # Initialize connector system
    try:
        conn = load_connector(config["execution"])
        print(f"✅ Connected to {conn.name}")
    except Exception as e:
        print(f"❌ Failed to load connector: {e}")
        print("🔄 Falling back to legacy MT5 system...")
        
        # Legacy MT5 initialization
        try:
            initialize(SYMBOL, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
            print(f"✅ MT5 initialized successfully")
            conn = None  # Use legacy system
        except Exception as mt5_e:
            print(f"❌ Failed to initialize MT5: {mt5_e}")
            exit(1)

    # Initialize core components
    broker = conn if conn else MT5Adapter()
    risk_model = RiskModel(broker)
    
    # Initialize signal engine with new constructor
    signal_engine = AdvancedSignalEngine(config)
    
    # Initialize all other engines
structure_engine = StructureEngine()
zone_engine = ZoneEngine()
order_flow_engine = OrderFlowEngine()
liquidity_filter = LiquidityFilter()
dashboard_logger = DashboardLogger(
    discord_webhook_url=DISCORD_WEBHOOK,
    username=DISCORD_USERNAME,
    avatar_url=DISCORD_AVATAR
)
visual_playbook = VisualPlaybook()
visualizer = OrderFlowVisualizer()
situational_analyzer = SituationalAnalyzer()
learning_engine = AdvancedLearningEngine()
exit_manager = AdaptiveExitManager()
mtf_analyzer = MultiTimeframeAnalyzer(timeframes=["M5", "M15", "H1", "H4", "D1", "W1", "MN1"])
    
    # Initialize new system components
    intel = IntelligenceCore(
        logger=logger,
        base_threshold=config["hybrid"]["entry_threshold_base"],
        require_all_confirm=config["mode"]["require_all_confirm"]
    )
    
    # Initialize Advanced CISD Engine
    cisd_engine = CISDEngine(config)
    
    # Central policy + feature logging (in-process, upgradable to service)
    policy = PolicyService(
        champion=intel,
        challenger=None,
        mode=config.get("policy", {}).get("mode", "shadow"),
        challenger_pct=config.get("policy", {}).get("challenger_pct", 0.0),
        logger=logger
    )
    feature_store = FeatureStore()
    
    # Initialize ML tracking and monitoring
    ml_tracker.start_run(
        run_name=f"trading_session_{int(time.time())}",
        tags={
            "environment": "production" if config["mode"]["autonomous"] else "simulation",
            "symbols": ",".join(config["execution"]["symbols"] if conn else list(TRADING_PAIRS.keys())),
            "version": "1.0.0"
        }
    )
    
    # Setup online learning for each symbol
    for sym in (config["execution"]["symbols"] if conn else list(TRADING_PAIRS.keys())):
        # Setup parameter bandits for key trading parameters
        online_learner.setup_parameter_bandit(
            sym, "confidence_threshold", 
            [0.5, 0.6, 0.7, 0.8, 0.9], 
            "ucb"
        )
        online_learner.setup_parameter_bandit(
            sym, "risk_multiplier", 
            [0.5, 1.0, 1.5, 2.0], 
            "epsilon_greedy"
        )
        
        # Setup contextual bandit for trade actions
        online_learner.create_contextual_bandit(
            sym, 
            ["buy", "sell", "hold"],
            ["ml_confidence", "trend_align", "structure_score", "volatility"]
        )
        
        # Set baseline performance for drift monitoring
        drift_monitor.set_baseline_performance(sym, {
            "win_rate": 0.6,
            "avg_return": 0.02,
            "sharpe_ratio": 1.5
        })
        
        # Initialize model management
        model_manager.set_performance_baseline(sym, {
            "win_rate": 0.6,
            "avg_return": 0.02,
            "sharpe_ratio": 1.5,
            "model_mae": 0.1
        })
        
        # Register initial champion model
        model_manager.register_model(
            sym, "intelligence_core", "1.0", "hybrid",
            {"win_rate": 0.6, "confidence": 0.7}
        )
        model_manager.set_champion(sym, "intelligence_core", "1.0")
    
    # Initialize overlay
overlay = GlobalRiskOverlay(max_drawdown=OVERLAY_MAX_DRAWDOWN, config={
    "max_exposure": 2.0,
    "volatility_limit": OVERLAY_VOL_THROTTLE_HIGH,
    "factor_limit": 0.3,
    "correlation_limit": 0.7,
})

    # Get symbols
    if conn:
        symbols = config["execution"]["symbols"]
        poll = int(config["execution"].get("poll_ms", 500)) / 1000.0
    else:
        # Legacy symbol handling
        symbols = list(TRADING_PAIRS.keys())
        poll = 60.0
        
        # Select all symbols in MT5
        for sym in symbols:
            try:
                mt5.symbol_select(sym, True)
            except Exception:
                print(f"⚠️ Could not select symbol {sym} in MT5")
    
    # Per-symbol components
    risk_managers = {sym: AdaptiveRiskManager(BASE_RISK) for sym in symbols}
    prophetic_engines = {sym: AdvancedPropheticEngine() for sym in symbols}
    prophetic_ml_map = {
        sym: PropheticMLEngine(
            config={"model_path": f"models/prophetic/{sym}"},
            model_path=f"models/prophetic/{sym}"
        )
        for sym in symbols
    }
    
    # Create required directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/prophetic", exist_ok=True)
    for sym in symbols:
        os.makedirs(f"models/prophetic/{sym}", exist_ok=True)
    
    # Simple equity tracker
equity = START_BALANCE
peak_equity = START_BALANCE

# Optional Telegram notifier
telegram_notifier = None
try:
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        telegram_notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
except Exception:
    telegram_notifier = None

# Perform startup self-check
if not startup_self_check():
    print("❌ Critical system check failed. Exiting...")
    exit(1)

    print(f"🤖 Unified Advanced Trading Bot running on {len(symbols)} symbols...")
print(f"📊 ML Learning: {'Enabled' if ENABLE_ML_LEARNING else 'Disabled'}")
print(f"🎯 ML Confidence Threshold: {ML_CONFIDENCE_THRESHOLD}")
    print(f"🔗 Connector: {conn.name if conn else 'Legacy MT5'}")

    # Timeframe mapping for legacy system
timeframe_map = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}
timeframe_const = timeframe_map.get(TIMEFRAME, mt5.TIMEFRAME_M15)

# --- Main Loop ---
while True:
    try:
            # Update equity
            if conn:
                equity = conn.equity()
            else:
                # Legacy equity tracking
                pass  # Keep existing equity logic
            
            RiskRules.on_equity_update(equity)
            perf_set_eq(equity)
            
            # Kill switches
            if RiskRules.hit_weekly_brake(equity) or RiskRules.hit_daily_loss_cap(equity):
                update_dashboard({"status": "HALTED", "equity": equity})
                logger.warning("Trading halted by risk rules.")
                time.sleep(poll)
                continue
            
            for sym in symbols:
            try:
                # --- Get Market Data ---
                    if conn:
                        # New connector system
                        q = conn.get_quote(sym)
                        spread_ok = within_spread_limit(q["spread_pips"], config["filters"]["max_spread_pips"])
                        if not spread_ok:
                            continue
                        
                        # Build features for new system
                        features = {
                            "atr_pips_14": 10.0,
                            "swing_stop_pips": 8.0,
                            "trend_align": True,
                            "vwap_dist": 0.8,
                            "pullback_score": 0.6,
                            "structure_score": 0.6,
                            "liquidity_sweep_score": 0.1,
                            "impulse_exhaustion": 0.4,
                            "trend_slope": 0.4,
                            "atr_norm": 1.0,
                            "ml_confidence": 0.58,
                            "direction": "long",
                            "spread_pips": q["spread_pips"],
                            "est_slippage_pips": 0.5,
                            "intended_price": q["ask"]
                        }
                        
                        # Enhanced CISD Analysis for new system
                        cisd_analysis = None
                        try:
                            mock_candles = [
                                {"open": q["bid"] - 0.001, "high": q["ask"] + 0.001, "low": q["bid"] - 0.002, "close": q["ask"], "tick_volume": 1000},
                                {"open": q["bid"] - 0.002, "high": q["ask"] - 0.001, "low": q["bid"] - 0.003, "close": q["bid"] - 0.001, "tick_volume": 800},
                                {"open": q["bid"] - 0.001, "high": q["ask"], "low": q["bid"] - 0.002, "close": q["ask"] - 0.001, "tick_volume": 1200}
                            ]
                            
                            mock_structure = {"event": "FLIP", "symbol": sym}
                            mock_order_flow = {"volume_total": 5000, "delta": 1000, "absorption": True}
                            mock_market_context = {"regime": "normal", "volatility": "normal", "trend_strength": 0.6}
                            mock_time_context = {"hour": 8}
                            
                            cisd_analysis = cisd_engine.detect_cisd(
                                candles=mock_candles,
                                structure_data=mock_structure,
                                order_flow_data=mock_order_flow,
                                market_context=mock_market_context,
                                time_context=mock_time_context
                            )
                            
                            if cisd_analysis and cisd_analysis["cisd_valid"]:
                                logger.info(f"Advanced CISD Validated for {sym}: Score={cisd_analysis['cisd_score']:.3f}")
                            
                        except Exception as e:
                            logger.warning(f"CISD analysis failed for {sym}: {e}")
                            cisd_analysis = None
                        
                        # Theory-based features (raid prob, extrema, session gate)
                        try:
                            theory_ctx = {
                                "now": time.time(),
                                "depleted_side": "high",  # heuristic default; improve with orderflow
                                "prev_session_completed": True,
                                "opposition_ok": True
                            }
                            theory_feats = compute_theory_features(mock_candles, theory_ctx)
                            if theory_feats:
                                features.update(theory_feats)
                        except Exception:
                            pass

                        # Online learning with error handling
                        optimized_threshold = None
                        risk_multiplier = None
                        bandit_action = None
                        
                        try:
                            # Optimize parameters using bandits
                            optimized_threshold = online_learner.select_parameter(sym, "confidence_threshold")
                            risk_multiplier = online_learner.select_parameter(sym, "risk_multiplier")
                            
                            if optimized_threshold:
                                features["optimized_confidence_threshold"] = optimized_threshold
                            if risk_multiplier:
                                features["risk_multiplier"] = risk_multiplier
                            
                            # Use contextual bandit for action selection
                            bandit_action = online_learner.select_action(sym, {
                                "ml_confidence": features["ml_confidence"],
                                "trend_align": float(features["trend_align"]),
                                "structure_score": features["structure_score"],
                                "volatility": features.get("atr_norm", 1.0)
                            })
                        except Exception as e:
                            logger.warning(f"Online learning failed for {sym}: {e}")
                            # Continue without ML optimization
                        
                        # Use central policy service with fallback
                        try:
                            decision = policy.decide(sym, features)
                            idea = decision.get("idea")
                        except Exception as e:
                            logger.error(f"Policy service failed for {sym}: {e}")
                            # Fallback to direct intelligence core
                            try:
                                idea = intel.decide(sym, features)
                                decision = {"idea": idea, "meta": {"trace_id": "fallback", "variant": "direct"}}
                            except Exception as e2:
                                logger.error(f"Fallback decision failed for {sym}: {e2}")
                                continue
                        
                        if not idea:
                            continue
                        
                        # Override action if bandit suggests different action (with safety check)
                        if (bandit_action in ["buy", "sell"] and 
                            bandit_action != idea["side"] and
                            features.get("ml_confidence", 0) > 0.6):  # Only override if confident
                            logger.info(f"🤖 Bandit override: {idea['side']} → {bandit_action}")
                            idea["side"] = bandit_action
                        
                        # Risk gates
                        if conn.open_positions_count() >= RiskRules.max_open_trades():
                            continue
                        
                        stop_pips = risk_model.stop_pips(sym, features)
                        size = risk_model.size_from_risk(sym, equity, stop_pips, RiskRules.per_trade_risk())
                        
                        # Slippage guard
                        slip_ok = within_slippage_limit(features["est_slippage_pips"], config["filters"]["max_slippage_pips"])
                        if not slip_ok or size <= 0:
                            continue
                        
                        meta = {
                            "score": idea["score"], "ml": idea["ml"], "rule": idea["rule"],
                            "prophetic": idea["prophetic"], "threshold": idea["threshold"], "regime": idea["regime"]
                        }
                        meta["trace_id"] = decision["meta"]["trace_id"]
                        meta["variant"] = decision["meta"]["variant"]
                        meta["challenger_shadow_score"] = decision["meta"]["shadow"].get("challenger_score")
                        meta["trace_id"] = decision["meta"]["trace_id"]
                        meta["variant"] = decision["meta"]["variant"]
                        meta["challenger_shadow_score"] = decision["meta"]["shadow"].get("challenger_score")
                        
                        # Add CISD analysis to meta
                        if cisd_analysis:
                            meta["cisd_score"] = cisd_analysis["cisd_score"]
                            meta["cisd_valid"] = cisd_analysis["cisd_valid"]
                            meta["cisd_confidence"] = cisd_analysis["confidence"]
                            meta["cisd_components"] = cisd_analysis.get("summary", {})
                        
                        # Write feature + decision row to lightweight store
                        try:
                            if config.get("policy", {}).get("enable_feature_logging", True):
                                feature_store.write_row(sym, TIMEFRAME, features, meta=meta)
                        except Exception:
                            pass
                        
                        # Write feature + decision row to lightweight store
                        try:
                            if config.get("policy", {}).get("enable_feature_logging", True):
                                feature_store.write_row(sym, TIMEFRAME, features, meta=meta)
                        except Exception:
                            pass
                        
                        # Execute trade
                        if config["mode"]["autonomous"]:
                            trade_id = conn.place_market(sym, idea["side"], size)
                            if trade_id:
                                conn.attach_stop_loss(trade_id, stop_pips)
                                conn.attach_take_profit(trade_id, rr=2.0)
                                logger.info(f"EXEC {sym} {idea['side']} lots={size} id={trade_id} meta={meta}")
                        else:
                            logger.info(f"[MANUAL MODE] Proposed {sym} {idea['side']} lots={size} meta={meta}")
                        
                        # Handle trade closure for paper connector
                        if hasattr(conn, "maybe_close_random"):
                            closed = conn.maybe_close_random(sym)
                            closed = conn.maybe_close_random(sym)
                            if closed:
                                led_by = 'ml' if idea["ml"] >= idea["rule"] else 'rules'
                                perf_on_close(sym, closed["pnl"], led_by, meta)
                                
                                # Advanced ML learning and monitoring
                                try:
                                    # Update online learner with outcome
                                    online_learner.update_model(
                                        sym, features, closed["pnl"], "adaptive"
                                    )
                                    
                                    # Update parameter bandits with rewards
                                    reward = 1.0 if closed["pnl"] > 0 else -1.0
                                    if optimized_threshold:
                                        online_learner.update_parameter_reward(
                                            sym, "confidence_threshold", optimized_threshold, reward
                                        )
                                    if risk_multiplier:
                                        online_learner.update_parameter_reward(
                                            sym, "risk_multiplier", risk_multiplier, reward
                                        )
                                    
                                    # Update contextual bandit
                                    online_learner.update_action_reward(
                                        sym, idea["side"], {
                                            "ml_confidence": features["ml_confidence"],
                                            "trend_align": float(features["trend_align"]),
                                            "structure_score": features["structure_score"],
                                            "volatility": features.get("atr_norm", 1.0)
                                        }, reward
                                    )
                                    
                                    # Drift monitoring
                                    feature_df = pd.DataFrame([features])
                                    target_series = pd.Series([closed["pnl"]])
                                    performance_metrics = {
                                        "win_rate": 1.0 if closed["pnl"] > 0 else 0.0,
                                        "return": closed["pnl"] / max(abs(features.get("intended_price", 1.0)), 1e-8)
                                    }
                                    
                                    drift_monitor.add_current_data(
                                        sym, feature_df, target_series, 
                                        performance_metrics=performance_metrics
                                    )
                                    
                                    # MLflow logging
                                    ml_tracker.log_metrics({
                                        f"{sym}_pnl": closed["pnl"],
                                        f"{sym}_win": 1.0 if closed["pnl"] > 0 else 0.0,
                                        f"{sym}_ml_confidence": features["ml_confidence"],
                                        f"{sym}_structure_score": features["structure_score"]
                                    })
                                    
                                    # Log prediction batch for monitoring
                                    ml_tracker.log_prediction_batch(
                                        predictions=np.array([idea["score"]]),
                                        actuals=np.array([reward]),
                                        metadata={
                                            "symbol": sym,
                                            "side": idea["side"],
                                            "trace_id": meta.get("trace_id")
                                        }
                                    )
                                    
                                except Exception as e:
                                    logger.warning(f"ML learning update failed: {e}")
                                
                                # Persist outcome row with trace id for learning
                                try:
                                    feature_store.write_outcome(
                                        sym,
                                        TIMEFRAME,
                                        idea["side"],
                                        closed["pnl"],
                                        rr=0.0,
                                        led_by=led_by,
                                        extra={
                                            "trace_id": meta.get("trace_id"),
                                            "trade_id": trade_id if 'trade_id' in locals() else None
                                        }
                                    )
                                except Exception:
                                    pass
                                
                                # Update CISD performance if CISD was involved
                                if cisd_analysis and cisd_analysis["cisd_valid"]:
                                    mock_signal_data = {
                                        "signal": idea["side"],
                                        "timestamp": time.time(),
                                        "cisd_analysis": cisd_analysis
                                    }
                                    cisd_engine.update_performance(f"trade_{trade_id}", closed["pnl"] > 0, closed["pnl"])
                    
                    else:
                        # Legacy MT5 system - this is the full advanced system from original main.py
                candles = get_candles(sym, timeframe_const, DATA_COUNT)

                # Multi-timeframe candles
                mtf_timeframes = ["M5", "M15", "H1", "H4", "D1", "W1", "MN1"]
                candles_by_tf = {}
                for tf in mtf_timeframes:
                    try:
                        candles_by_tf[tf] = get_candles(sym, timeframe_map[tf], max(50, DATA_COUNT // (2 if tf in ["M5", "M15"] else 1)))
                    except Exception:
                                candles_by_tf[tf] = candles

                # Prepare market data context
                market_data = {
                    "symbol": sym,
                    "candles": candles,
                    "timestamp": candles[-1]["time"] if candles else None,
                    "timeframe": TIMEFRAME
                }

                # --- Liquidity Filter ---
                now = candles[-1]["time"] if candles else None
                liquidity_context = liquidity_filter.get_liquidity_context(now, symbol=sym)
                sessions = liquidity_context.get("active_sessions", [])
                in_window = liquidity_filter.is_liquid_time(now, symbol=sym)
                
                if not in_window:
                    dashboard_logger.log_none(sym)
                    continue

                # --- Enhanced Market Analysis ---
                structure_data = structure_engine.analyze(candles)
                zone_data = zone_engine.identify_zones(candles, structure_data)
                order_flow_data = order_flow_engine.process(candles)

                # Enhanced situational analysis
                situational_context = situational_analyzer.analyze(candles)
                situational_context["sessions"] = sessions
                situational_context["symbol"] = sym

                # --- Multi-Timeframe Analysis ---
                try:
                    mtf = mtf_analyzer.analyze(candles_by_tf)
                    situational_context["mtf_bias"] = mtf.get("bias")
                    situational_context["mtf_confidence"] = mtf.get("confidence")
                    situational_context["mtf_levels"] = mtf.get("confluences", {}).get("levels", [])
                    situational_context["mtf_fourier"] = mtf.get("fourier", {})
                    situational_context["mtf_three_wave"] = mtf.get("three_wave", {})
                    situational_context["mtf_participants"] = mtf.get("participants", {})
                    
                    # Check confluence near current price with higher+lower TFs
                    last_close = candles[-1]["close"]
                    mtf_entry_ok = False
                    mtf_strength = 0.0
                    for lvl in situational_context["mtf_levels"]:
                        price = lvl.get("price")
                                if price and abs(price - last_close) / max(last_close, 1e-8) < 0.001:
                            tfs = set(lvl.get("timeframes", []))
                            if ("H1" in tfs or "H4" in tfs or "D1" in tfs or "W1" in tfs or "MN1" in tfs) and ("M5" in tfs or "M15" in tfs):
                                mtf_entry_ok = True
                                mtf_strength = max(mtf_strength, lvl.get("strength", 0.0))
                    situational_context["mtf_entry_ok"] = mtf_entry_ok
                    situational_context["mtf_confluence_strength"] = mtf_strength
                except Exception:
                    situational_context["mtf_entry_ok"] = False

                # Prophetic components per symbol
                p_engine = prophetic_engines[sym]
                p_ml = prophetic_ml_map[sym]

                # Prophetic analysis
                prophetic_context = p_engine.analyze(
                    timestamp=candles[-1]["time"],
                    market_data={
                        "prices": [c["close"] for c in candles],
                        "volumes": [c["tick_volume"] for c in candles],
                        "high": [c["high"] for c in candles],
                        "low": [c["low"] for c in candles]
                    },
                    context=situational_context
                )

                # ML-enhanced cycle prediction
                cycle_prediction = p_ml.predict_cycle(
                    market_data={
                        "prices": [c["close"] for c in candles],
                        "volumes": [c["tick_volume"] for c in candles]
                    },
                    context=situational_context
                )

                # --- Fourier Wave Analysis Integration ---
                try:
                    from core.fourier_wave_engine import FourierWaveEngine
                    fourier_engine = FourierWaveEngine()
                    
                    # Extract price data for wave analysis
                    prices = [float(c["close"]) for c in candles]
                    volumes = [float(c.get("tick_volume", 1000)) for c in candles]
                    
                    # Perform Fourier wave cycle analysis
                    wave_analysis = fourier_engine.analyze_wave_cycle(
                        price_data=prices,
                        volume_data=volumes,
                        symbol=sym,
                        timeframe=TIMEFRAME
                    )
                    
                    if wave_analysis["valid"]:
                        situational_context["fourier_wave"] = {
                            "absorption_type": wave_analysis["summary"]["absorption_type"],
                            "current_phase": wave_analysis["summary"]["current_phase"],
                            "phase_completion": wave_analysis["summary"]["phase_completion"],
                            "pattern": wave_analysis["summary"]["pattern"],
                            "confidence": wave_analysis["summary"]["confidence"],
                            "fft_quality": wave_analysis["fft_quality"]
                        }
                        
                        # Boost CISD score if full absorption detected
                        if wave_analysis["summary"]["absorption_type"] == "full":
                            if "cisd_score" in situational_context:
                                situational_context["cisd_score"] = min(1.0, situational_context["cisd_score"] + 0.1)
                        
                        print(f"🌊 {sym} Wave: {wave_analysis['summary']['pattern']} | Phase: {wave_analysis['summary']['current_phase']} | Absorption: {wave_analysis['summary']['absorption_type']}")
                    else:
                        situational_context["fourier_wave"] = {"error": wave_analysis.get("error", "Analysis failed")}
                        
                except Exception as e:
                    situational_context["fourier_wave"] = {"error": f"Fourier analysis failed: {str(e)}"}
                    print(f"⚠️ Fourier analysis failed for {sym}: {str(e)}")

                # Integrate prophetic insights
                situational_context["prophetic_window"] = prophetic_context["window"]
                situational_context["cycle_phase"] = cycle_prediction["current_phase"]
                situational_context["cycle_confidence"] = cycle_prediction["confidence"]

                # Add volatility regime to market context
                if situational_context.get("volatility_regime"):
                    market_data["volatility_regime"] = situational_context["volatility_regime"]
                    risk_managers[sym].update_market_conditions(
                        situational_context["volatility_regime"],
                        {**situational_context, "active_sessions": sessions}
                    )

                # --- Advanced Signal Generation ---
                signal = signal_engine.generate_signal(
                    market_data,
                    structure_data,
                    zone_data,
                    order_flow_data,
                    situational_context,
                    liquidity_context,
                    prophetic_context
                )

                # --- Enhanced Filtering and Logging ---
                if signal:
                    dashboard_logger.log_signal(sym, signal)
                    visualizer.add_frame(
                        [c.get("bid_volume", 0) for c in candles],
                        [c.get("ask_volume", 0) for c in candles],
                        tags=[signal.get("signal", "")] + signal.get("reasons", [])
                    )

                    # --- Advanced Risk Management ---
                    entry_price = candles[-1]["close"]
                    stop_loss = None
                    target = None
                    
                    # Adaptive exit planning (reflexive exits)
                    try:
                        position_type = "long" if signal["signal"] == "bullish" else "short"
                        market_state = {
                            "volatility_regime": situational_context.get("volatility_regime", "normal"),
                            "momentum_state": "accelerating" if situational_context.get("momentum_shift") else "neutral",
                        }
                        of = order_flow_data or {}
                        if "delta" in of:
                            of["institutional_activity"] = of.get("institutional_activity", abs(of.get("delta", 0)) > 1000)
                        risk_params = {"risk_percent": risk_managers[sym].base_risk}
                        exit_plan = exit_manager.calculate_exits(entry_price, position_type, market_state, of, risk_params)
                        
                        # Use adaptive stop and main TP if available
                        if exit_plan.get("stop_loss"):
                            stop_loss = exit_plan["stop_loss"]["price"]
                        if exit_plan.get("take_profits"):
                            target = exit_plan["take_profits"][0]["price"]
                    except Exception:
                        exit_plan = None

                    # Fallback zone-based stops if adaptive not available
                    if stop_loss is None and zone_data.get("zones"):
                        if signal["signal"] == "bullish":
                            stop_loss = zone_data["zones"][0]["low"] - (zone_data["zones"][0]["high"] - zone_data["zones"][0]["low"]) * 0.1
                            if stop_loss and stop_loss < entry_price and target is None:
                                target = entry_price + 2.5 * (entry_price - stop_loss)
                        else:
                            stop_loss = zone_data["zones"][0]["high"] + (zone_data["zones"][0]["high"] - zone_data["zones"][0]["low"]) * 0.1
                            if stop_loss and stop_loss > entry_price and target is None:
                                target = entry_price - 2.5 * (stop_loss - entry_price)

                    rr = calculate_rr(entry_price, stop_loss, target) if stop_loss and target else 0

                    # Global overlay: compute portfolio state and risk throttle
                    risk_throttle = OVERLAY_BASE_THROTTLE
                    if 'ENABLE_GLOBAL_OVERLAY' in globals() and ENABLE_GLOBAL_OVERLAY:
                        portfolio_state = {
                            "equity": equity,
                            "peak_equity": peak_equity,
                            "drawdown": (peak_equity - equity) / max(peak_equity, 1e-9),
                            "positions": {},
                        }
                        market_state = {"volatility": situational_context.get("volatility_regime", "normal")}
                        try:
                            risk_state = overlay.analyze_risk(portfolio_state, market_state)
                            if risk_state.stress_level in ("high", "extreme"):
                                risk_throttle = 0.5 if risk_state.stress_level == "high" else 0.25
                        except Exception:
                            risk_throttle = OVERLAY_BASE_THROTTLE

                    # Advanced position sizing with ML confidence + overlay throttle
                    pip_value = TRADING_PAIRS.get(sym, {}).get("pip_value", 0.0001 if "JPY" not in sym else 0.01)
                    sl_pips = abs(entry_price - stop_loss) / pip_value if stop_loss else 0
                    lot, lot_reasons = risk_managers[sym].calculate_position_size(
                        START_BALANCE,
                        sl_pips,
                        confidence_level=signal.get("confidence", "medium"),
                        streak=risk_managers[sym].streak,
                        tags=signal.get("reasons", []),
                        market_context={**situational_context, "active_sessions": sessions},
                        ml_confidence=signal.get("ml_confidence"),
                        symbol=sym
                    )
                    lot *= risk_throttle

                    # --- Execute Trade with Enhanced Monitoring ---
                    result = execute_trade(signal, entry_price, lot=lot, sl=stop_loss, tp=target)

                    # Calculate actual PnL
                    if result.get("status") == "success":
                        entry = result.get("price", entry_price)
                        current_price = fetch_latest_data(sym)["price"]
                        pnl = (current_price - entry) * lot * 100000 if signal["signal"] == "bullish" else (entry - current_price) * lot * 100000
                    else:
                        pnl = 0

                    # --- Enhanced Trade Recording & Learning ---
                    trade_record = {
                        "timestamp": candles[-1]["time"],
                        "pair": sym,
                        "signal": signal["signal"],
                        "confidence": signal["confidence"],
                        "ml_confidence": signal.get("ml_confidence"),
                        "cisd": signal.get("cisd", False),
                        "pnl": pnl,
                        "rr": rr,
                        "lot": lot,
                        "reasons": signal.get("reasons", []) + lot_reasons,
                        "pattern": signal.get("pattern", {}),
                        "sessions": sessions,
                        "market_context": signal.get("market_context", {})
                    }

                    dashboard_logger.log_trade(trade_record)
                    
                    # Update equity tracking
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    
                    # Update signal engine with new equity
                    signal_engine.on_account_update(equity)

                    # Telegram alert (optional)
                    if telegram_notifier:
                        try:
                            tmsg = f"{sym} {signal['signal'].upper()} | conf={signal['confidence']} | lot={lot}\nPnL: {pnl:.2f} | RR: {rr:.2f}"
                            telegram_notifier.send(tmsg)
                        except Exception:
                            pass

                    # Record outcome for ML learning
                    signal_engine.record_signal_outcome(
                        signal,
                        "win" if pnl > 0 else "loss",
                        pnl,
                        rr
                    )

                    # Update risk manager with outcome
                    risk_managers[sym].update_streak("win" if pnl > 0 else "loss", symbol=sym)

                    # Update prophetic accuracy
                    p_engine.update_accuracy(
                        prophetic_context["window"],
                        {
                            "outcome": "win" if pnl > 0 else "loss",
                            "pnl": pnl,
                            "rr": rr,
                            "market_data": market_data
                        }
                    )

                    # Update cycle prediction model
                    p_ml.update(
                        market_data={
                            "prices": [c["close"] for c in candles],
                            "volumes": [c["tick_volume"] for c in candles]
                        },
                        context=situational_context,
                        actual_phase=cycle_prediction["current_phase"]
                    )

                    # Visualization
                    visualizer.plot_last_frame()

                    # Print performance metrics
                    stats = signal_engine.get_signal_stats(sym)
                    if stats:
                        print(f"\n📊 Performance Metrics for {sym}:")
                        print(f"Win Rate: {stats['win_rate']:.2%}")
                        print(f"Average RR: {stats['average_rr']:.2f}")
                        print(f"ML Model Accuracy: {stats['ml_performance']['accuracy']:.2%}")
                        print(f"Current Streak: {risk_managers[sym].streak}")
                        print(f"Prophetic Window: {prophetic_context['window'].market_bias} ({prophetic_context['window'].confidence:.2%})")
                        print(f"Cycle Phase: {cycle_prediction['current_phase']} ({cycle_prediction['confidence']:.2%})")

                else:
                    dashboard_logger.log_none(sym)
                    
            except Exception as inner_e:
                print(f"❌ Error on {sym}: {inner_e}")
                continue

        # Periodic ML management checks (every 10 minutes)
        current_time = time.time()
        if not hasattr(main, 'last_ml_check'):
            main.last_ml_check = current_time
        
        if current_time - main.last_ml_check > 600:  # 10 minutes
            try:
                ml_check_results = model_manager.run_periodic_checks()
                logger.info(f"🤖 ML Management Check: {len(ml_check_results['actions_taken'])} actions taken")
                
                # Log important ML events
                for action in ml_check_results['actions_taken']:
                    if action['action'] in ['retrain_triggered', 'champion_promoted']:
                        print(f"🔄 ML Action: {action['action']} for {action['symbol']} - {action.get('reason', '')}")
                
                main.last_ml_check = current_time
            except Exception as e:
                logger.warning(f"ML management check failed: {e}")

        # Update dashboard with current system state
        try:
                if conn:
                    # New system dashboard update with ML metrics
                    cisd_stats = cisd_engine.get_cisd_stats()
                    
                    # Gather ML status for all symbols
                    ml_status = {}
                    for sym in symbols:
                        ml_status[sym] = model_manager.get_model_status(sym)
                    
                    update_dashboard({
                        "mode": "AUTO" if config["mode"]["autonomous"] else "MANUAL",
                        "weights": intel.tuner.get_weights(),
                        "entry_threshold_base": config["hybrid"]["entry_threshold_base"],
                        "risk": {
                            "per_trade_risk": RiskRules.per_trade_risk(),
                            "daily_loss_cap": config["risk"]["daily_loss_cap"],
                            "weekly_dd_brake": config["risk"]["weekly_dd_brake"],
                            "max_open_trades": RiskRules.max_open_trades()
                        },
                        "equity": equity,
                        "cisd_stats": cisd_stats,
                        "ml_status": ml_status,
                        "online_learning": {
                            sym: online_learner.get_learning_summary(sym) 
                            for sym in symbols
                        },
                        "drift_monitoring": {
                            sym: drift_monitor.get_drift_summary(sym, days=1)
                            for sym in symbols
                        }
                    })
                else:
                    # Legacy dashboard update
            perf_data = perf_snapshot()
            update_dashboard({
                        "mode": "AUTO",
                "weights": signal_engine.weight_tuner.get_weights(),
                        "entry_threshold_base": 0.62,
                "risk": {
                    "per_trade_risk": RiskRules.per_trade_risk(),
                    "daily_loss_cap": 0.015,
                    "weekly_dd_brake": 0.04,
                    "max_open_trades": RiskRules.max_open_trades()
                },
                "last_10_trades": perf_data.get("last_10", []),
                "equity": equity,
                "peak_equity": peak_equity,
                "performance": {
                    "wins": perf_data.get("wins", 0),
                    "losses": perf_data.get("losses", 0),
                    "win_rate": perf_data.get("wins", 0) / max(1, perf_data.get("wins", 0) + perf_data.get("losses", 0))
                }
            })
        except Exception as e:
            print(f"⚠️ Dashboard update failed: {e}")

        # Sleep after processing all symbols
            time.sleep(poll)

        except KeyboardInterrupt:
            logger.info("Shutdown requested.")
            break
    except Exception as e:
        print(f"❌ Error occurred: {e}")
            time.sleep(poll)

# Cleanup on exit
    try:
learning_engine.force_save()
        
        # Save ML models and close MLflow run
        try:
            ml_tracker.end_run()
            print("✅ MLflow run completed")
        except Exception:
            pass
        
for sym, p_ml in prophetic_ml_map.items():
    try:
        p_ml._save_models()
    except Exception:
        pass
        
        # Final ML summary
        try:
            print("\n🤖 Final ML Summary:")
            for sym in symbols:
                status = model_manager.get_model_status(sym)
                print(f"  {sym}: Champion={status['champion_model']}, Retrain={status['should_retrain']}")
        except Exception:
            pass
        
        if not conn:
shutdown()
        print("✅ Cleanup completed successfully")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

if __name__ == "__main__":
    main()
