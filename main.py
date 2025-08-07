# main.py

from core.signal_engine import SignalEngine
from core.structure_engine import StructureEngine
from core.zone_engine import ZoneEngine
from core.liquidity_filter import LiquidityFilter
from core.order_flow_engine import OrderFlowEngine
from core.dashboard_logger import DashboardLogger
from core.visual_playbook import VisualPlaybook
from core.order_flow_visualizer import OrderFlowVisualizer
from core.situational_analysis import SituationalAnalyzer
from core.trade_executor import execute_trade
from core.risk_manager import RiskManager
from memory.learning import LearningEngine

from utils.mt5_connector import initialize, get_candles, fetch_latest_data, shutdown
from utils.session_timer import is_in_liquidity_window
from utils.helpers import calculate_rr

import time
import MetaTrader5 as mt5

# --- Initialization ---
symbol = "EURUSD"
timeframe = "M15"  # Or mt5.TIMEFRAME_M15, depending on your setup
data_count = 100
base_risk = 0.01
balance = 10000  # Placeholder, fetch from broker for live

try:
    initialize(symbol)
    print(f"✅ MT5 initialized successfully for {symbol}")
except Exception as e:
    print(f"❌ Failed to initialize MT5: {e}")
    exit(1)
signal_engine = SignalEngine()
structure_engine = StructureEngine()
zone_engine = ZoneEngine()
order_flow_engine = OrderFlowEngine()
liquidity_filter = LiquidityFilter()
dashboard_logger = DashboardLogger()
visual_playbook = VisualPlaybook()
visualizer = OrderFlowVisualizer()
situational_analyzer = SituationalAnalyzer()
risk_manager = RiskManager(base_risk)
learning_engine = LearningEngine()

print(f"Bot running on {symbol}...")

# --- Main Loop ---
while True:
    try:
        # --- Get Candles ---
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        timeframe_const = timeframe_map.get(timeframe, mt5.TIMEFRAME_M15)
        candles = get_candles(
            symbol,
            timeframe_const,
            data_count,
        )

        # --- Liquidity Filter ---
        now = candles[-1]["time"] if candles else None
        in_window, sessions = is_in_liquidity_window(now)
        if not in_window:
            print(f"Outside liquidity window ({sessions}). Skipping...")
            time.sleep(60)
            continue

        # --- Structure & Situational Context ---
        structure_data = structure_engine.analyze(candles)
        zone_data = zone_engine.identify_zones(candles, structure_data)
        order_flow_data = order_flow_engine.process(candles)
        situational_context = situational_analyzer.analyze(candles)
        # Add session names to context for full logging
        situational_context["sessions"] = sessions

        # --- Signal Generation ---
        signal = signal_engine.generate_signal(
            candles, structure_data, zone_data, order_flow_data, situational_context
        )

        # --- Filtering and Logging ---
        if signal:
            dashboard_logger.log_signal(symbol, signal)
            visualizer.add_frame(
                [c.get("bid_volume", 0) for c in candles],
                [c.get("ask_volume", 0) for c in candles],
                tags=[signal.get("signal", "")] + signal.get("reasons", [])
            )

            # --- Risk Management ---
            entry_price = candles[-1]["close"]
            stop_loss = None
            target = None
            
            if signal["signal"] == "bullish" and zone_data.get("zones"):
                stop_loss = zone_data["zones"][0]["low"]
                # For bullish trades, stop loss must be below entry price
                if stop_loss and stop_loss < entry_price:
                    risk_distance = entry_price - stop_loss
                    target = entry_price + 2 * risk_distance
                else:
                    stop_loss = None  # Invalid stop loss
                    print(f"Warning: Invalid stop loss for bullish trade - SL: {stop_loss}, Entry: {entry_price}")
            elif signal["signal"] == "bearish" and zone_data.get("zones"):
                stop_loss = zone_data["zones"][0]["high"]
                # For bearish trades, stop loss must be above entry price
                if stop_loss and stop_loss > entry_price:
                    risk_distance = stop_loss - entry_price
                    target = entry_price - 2 * risk_distance
                else:
                    stop_loss = None  # Invalid stop loss
                    print(f"Warning: Invalid stop loss for bearish trade - SL: {stop_loss}, Entry: {entry_price}")

            rr = calculate_rr(entry_price, stop_loss, target) if stop_loss and target else 0

            # Use confidence for position sizing
            lot, lot_reasons = risk_manager.calculate_position_size(
                balance, abs(entry_price - stop_loss) * 10000,  # Assuming pips
                confidence_level=signal.get("confidence", "medium"),
                streak=risk_manager.streak,
                tags=signal.get("reasons", [])
            )

            # --- Execute Trade (live) ---
            result = execute_trade(signal, entry_price, lot=lot, sl=stop_loss, tp=target)
            pnl = result.get("pnl", 0)

            # --- Log trade & update learning/memory ---
            trade_record = {
                "timestamp": candles[-1]["time"],
                "pair": symbol,
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "cisd": signal.get("cisd", False),
                "pnl": pnl,
                "rr": rr,
                "lot": lot,
                "reasons": signal.get("reasons", []) + lot_reasons,
                "pattern": signal.get("pattern", {}),
                "sessions": sessions,
            }
            dashboard_logger.log_trade(trade_record)
            learning_engine.record_result(symbol, situational_context, signal["signal"], "win" if pnl > 0 else "loss", rr, candles[-1]["time"],
                                         tags=signal.get("reasons", []))

            # --- Visuals (optional) ---
            visualizer.plot_last_frame()

        else:
            dashboard_logger.log_none(symbol)

        time.sleep(60)

    except Exception as e:
        import traceback
        print(f"Critical error in main loop: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Log to dashboard if available
        try:
            dashboard_logger.log_error(f"Main loop error: {e}")
        except:
            pass
        time.sleep(60)

# Optionally, call shutdown() when exiting script (not needed in infinite loop)
