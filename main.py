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
from core.trade_executor import execute_trade
from core.risk_manager import AdaptiveRiskManager
from memory.learning import AdvancedLearningEngine
from core.prophetic_layer import AdvancedPropheticEngine
from core.prophetic_ml import PropheticMLEngine

from utils.mt5_connector import initialize, get_candles, fetch_latest_data, shutdown
from utils.session_timer import is_in_liquidity_window
from utils.helpers import calculate_rr
from utils.config import (
    SYMBOL, TIMEFRAME, DATA_COUNT, BASE_RISK, START_BALANCE,
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    ENABLE_ML_LEARNING, ML_CONFIDENCE_THRESHOLD
)

import time
import MetaTrader5 as mt5
from datetime import datetime
import os

# --- Initialization ---
print("🚀 Initializing Advanced Trading System...")

try:
    initialize(SYMBOL, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    print(f"✅ MT5 initialized successfully for {SYMBOL}")
except Exception as e:
    print(f"❌ Failed to initialize MT5: {e}")
    exit(1)

# Initialize core components
signal_engine = AdvancedSignalEngine()
structure_engine = StructureEngine()
zone_engine = ZoneEngine()
order_flow_engine = OrderFlowEngine()
liquidity_filter = LiquidityFilter()
dashboard_logger = DashboardLogger()
visual_playbook = VisualPlaybook()
visualizer = OrderFlowVisualizer()
situational_analyzer = SituationalAnalyzer()
risk_manager = AdaptiveRiskManager(BASE_RISK)
learning_engine = AdvancedLearningEngine()

# Initialize prophetic components
prophetic_engine = AdvancedPropheticEngine()
prophetic_ml = PropheticMLEngine(
    config={"model_path": "models/prophetic"},
    model_path="models/prophetic"
)

# Create required directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models/prophetic", exist_ok=True)

print(f"🤖 Advanced Trading Bot running on {SYMBOL}...")
print(f"📊 ML Learning: {'Enabled' if ENABLE_ML_LEARNING else 'Disabled'}")
print(f"🎯 ML Confidence Threshold: {ML_CONFIDENCE_THRESHOLD}")

# --- Main Loop ---
while True:
    try:
        # --- Get Market Data ---
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
        candles = get_candles(
            SYMBOL,
            timeframe_const,
            DATA_COUNT,
        )

        # Prepare market data context
        market_data = {
            "symbol": SYMBOL,
            "candles": candles,
            "timestamp": candles[-1]["time"] if candles else None,
            "timeframe": TIMEFRAME
        }

        # --- Liquidity Filter ---
        now = candles[-1]["time"] if candles else None
        in_window, sessions = is_in_liquidity_window(now)
        liquidity_context = {
            "in_window": in_window,
            "active_sessions": sessions
        }
        
        if not in_window:
            print(f"⏰ Outside liquidity window ({sessions}). Waiting...")
            time.sleep(60)
            continue

        # --- Enhanced Market Analysis ---
        structure_data = structure_engine.analyze(candles)
        zone_data = zone_engine.identify_zones(candles, structure_data)
        order_flow_data = order_flow_engine.process(candles)
        
        # Enhanced situational analysis
        situational_context = situational_analyzer.analyze(candles)
        situational_context["sessions"] = sessions
        
        # Prophetic analysis
        prophetic_context = prophetic_engine.analyze(
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
        cycle_prediction = prophetic_ml.predict_cycle(
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
            risk_manager.update_market_conditions(
                situational_context["volatility_regime"],
                situational_context
            )

        # --- Advanced Signal Generation ---
        signal = signal_engine.generate_signal(
            market_data,
            structure_data,
            zone_data,
            order_flow_data,
            situational_context,
            liquidity_context
        )

        # --- Enhanced Filtering and Logging ---
        if signal:
            dashboard_logger.log_signal(SYMBOL, signal)
            visualizer.add_frame(
                [c.get("bid_volume", 0) for c in candles],
                [c.get("ask_volume", 0) for c in candles],
                tags=[signal.get("signal", "")] + signal.get("reasons", [])
            )

            # --- Advanced Risk Management ---
            entry_price = candles[-1]["close"]
            stop_loss = None
            target = None
            
            # Enhanced stop loss calculation
            if signal["signal"] == "bullish" and zone_data.get("zones"):
                stop_loss = zone_data["zones"][0]["low"] - (zone_data["zones"][0]["high"] - zone_data["zones"][0]["low"]) * 0.1
                if stop_loss and stop_loss < entry_price:
                    target = entry_price + 2.5 * (entry_price - stop_loss)  # Improved RR ratio
                else:
                    stop_loss = None
            elif signal["signal"] == "bearish" and zone_data.get("zones"):
                stop_loss = zone_data["zones"][0]["high"] + (zone_data["zones"][0]["high"] - zone_data["zones"][0]["low"]) * 0.1
                if stop_loss and stop_loss > entry_price:
                    target = entry_price - 2.5 * (stop_loss - entry_price)  # Improved RR ratio
                else:
                    stop_loss = None

            rr = calculate_rr(entry_price, stop_loss, target) if stop_loss and target else 0

            # Advanced position sizing with ML confidence
            lot, lot_reasons = risk_manager.calculate_position_size(
                START_BALANCE,
                abs(entry_price - stop_loss) * 10000 if stop_loss else 0,
                confidence_level=signal.get("confidence", "medium"),
                streak=risk_manager.streak,
                tags=signal.get("reasons", []),
                market_context=situational_context,
                ml_confidence=signal.get("ml_confidence")
            )

            # --- Execute Trade with Enhanced Monitoring ---
            result = execute_trade(signal, entry_price, lot=lot, sl=stop_loss, tp=target)
            
            # Calculate actual PnL
            if result.get("status") == "success":
                entry = result.get("price", entry_price)
                current_price = fetch_latest_data(SYMBOL)["price"]
                pnl = (current_price - entry) * lot * 100000 if signal["signal"] == "bullish" else (entry - current_price) * lot * 100000
            else:
                pnl = 0

            # --- Enhanced Trade Recording & Learning ---
            trade_record = {
                "timestamp": candles[-1]["time"],
                "pair": SYMBOL,
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
            
            # Record outcome for ML learning
            signal_engine.record_signal_outcome(
                signal,
                "win" if pnl > 0 else "loss",
                pnl,
                rr
            )
            
            # Update risk manager with outcome
            risk_manager.update_streak("win" if pnl > 0 else "loss")
            
            # Update prophetic accuracy
            prophetic_engine.update_accuracy(
                prophetic_context["window"],
                {
                    "outcome": "win" if pnl > 0 else "loss",
                    "pnl": pnl,
                    "rr": rr,
                    "market_data": market_data
                }
            )
            
            # Update cycle prediction model
            prophetic_ml.update(
                market_data={
                    "prices": [c["close"] for c in candles],
                    "volumes": [c["tick_volume"] for c in candles]
                },
                context=situational_context,
                actual_phase=cycle_prediction["current_phase"]
            )

            # --- Enhanced Visualization ---
            visualizer.plot_last_frame()

            # Print performance metrics
            if signal_engine.get_signal_stats(SYMBOL):
                stats = signal_engine.get_signal_stats(SYMBOL)
                print(f"\n📊 Performance Metrics for {SYMBOL}:")
                print(f"Win Rate: {stats['win_rate']:.2%}")
                print(f"Average RR: {stats['average_rr']:.2f}")
                print(f"ML Model Accuracy: {stats['ml_performance']['accuracy']:.2%}")
                print(f"Current Streak: {risk_manager.streak}")
                print(f"Prophetic Window: {prophetic_context['window'].market_bias} ({prophetic_context['window'].confidence:.2%})")
                print(f"Cycle Phase: {cycle_prediction['current_phase']} ({cycle_prediction['confidence']:.2%})")

        else:
            dashboard_logger.log_none(SYMBOL)

        time.sleep(60)

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        time.sleep(60)

# Cleanup on exit
learning_engine.force_save()
prophetic_ml._save_models()  # Save prophetic ML models
shutdown()
