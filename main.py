# main.py

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

import time
import MetaTrader5 as mt5
from datetime import datetime
import os

# --- Initialization ---
print("🚀 Initializing Advanced Trading System...")

try:
    # Initialize MT5 on primary symbol (or without symbol selection)
    initialize(SYMBOL, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    print(f"✅ MT5 initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize MT5: {e}")
    exit(1)

# Build symbol universe (all pairs configured)
SYMBOLS = list(TRADING_PAIRS.keys())

# Select all symbols in MT5
for sym in SYMBOLS:
    try:
        mt5.symbol_select(sym, True)
    except Exception:
        print(f"⚠️ Could not select symbol {sym} in MT5")

# Initialize core shared components
signal_engine = AdvancedSignalEngine()
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
overlay = GlobalRiskOverlay(max_drawdown=OVERLAY_MAX_DRAWDOWN, config={
    "max_exposure": 2.0,
    "volatility_limit": OVERLAY_VOL_THROTTLE_HIGH,
    "factor_limit": 0.3,
    "correlation_limit": 0.7,
})

# Simple equity tracker (optional)
equity = START_BALANCE
peak_equity = START_BALANCE

# Optional Telegram notifier
telegram_notifier = None
try:
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        telegram_notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
except Exception:
    telegram_notifier = None

# Per-symbol components
risk_managers = {sym: AdaptiveRiskManager(BASE_RISK) for sym in SYMBOLS}
prophetic_engines = {sym: AdvancedPropheticEngine() for sym in SYMBOLS}
prophetic_ml_map = {
    sym: PropheticMLEngine(
        config={"model_path": f"models/prophetic/{sym}"},
        model_path=f"models/prophetic/{sym}"
    )
    for sym in SYMBOLS
}

# Create required directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models/prophetic", exist_ok=True)
for sym in SYMBOLS:
    os.makedirs(f"models/prophetic/{sym}", exist_ok=True)

print(f"🤖 Advanced Trading Bot running on {len(SYMBOLS)} symbols...")
print(f"📊 ML Learning: {'Enabled' if ENABLE_ML_LEARNING else 'Disabled'}")
print(f"🎯 ML Confidence Threshold: {ML_CONFIDENCE_THRESHOLD}")

# Timeframe mapping
# Timeframe mapping
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
        for sym in SYMBOLS:
            try:
                # --- Get Market Data ---
                candles = get_candles(sym, timeframe_const, DATA_COUNT)

                # Multi-timeframe candles
                mtf_timeframes = ["M5", "M15", "H1", "H4", "D1", "W1", "MN1"]
                candles_by_tf = {}
                for tf in mtf_timeframes:
                    try:
                        candles_by_tf[tf] = get_candles(sym, timeframe_map[tf], max(50, DATA_COUNT // (2 if tf in ["M5", "M15"] else 1)))
                    except Exception:
                        candles_by_tf[tf] = candles  # fallback to current

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
                        if price and abs(price - last_close) / max(last_close, 1e-8) < 0.001:  # within 0.1%
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

        # Sleep after processing all symbols
        time.sleep(60)

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        time.sleep(60)

# Cleanup on exit
learning_engine.force_save()
for sym, p_ml in prophetic_ml_map.items():
    try:
        p_ml._save_models()
    except Exception:
        pass
shutdown()
