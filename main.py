from core.signal_engine import SignalEngine
from core.structure_engine import StructureEngine
from core.zone_engine import ZoneEngine
from core.liquidity_filter import LiquidityFilter
from core.order_flow_engine import OrderFlowEngine
from core.dashboard_logger import DashboardLogger
from core.visual_playbook import VisualPlaybook
from core.order_flow_visualizer import OrderFlowVisualizer
from utils.mt5_connector import initialize, get_candles
from utils.session_timer import is_in_liquidity_window

import time

# --- Initialization ---
symbol = "EURUSD"
timeframe = "M15"
data_count = 100
risk = 0.01

initialize(symbol)

signal_engine = SignalEngine()
structure_engine = StructureEngine()
zone_engine = ZoneEngine()
order_flow_engine = OrderFlowEngine()
liquidity_filter = LiquidityFilter()
dashboard_logger = DashboardLogger()
visual_playbook = VisualPlaybook()
visualizer = OrderFlowVisualizer()

print(f"Bot running on {symbol}...")

# --- Main Loop ---
while True:
    try:
        candles = get_candles(
            symbol,
            getattr(__import__("MetaTrader5"), f"TIMEFRAME_{timeframe}"),
            data_count,
        )

        if not is_in_liquidity_window():
            print("Outside liquidity window. Skipping...")
            time.sleep(60)
            continue

        structure_data = structure_engine.analyze(candles)
        zone_data = zone_engine.identify_zones(candles, structure_data)
        order_flow_data = order_flow_engine.process(candles)
        situational_context = liquidity_filter.generate_context(candles)

        signal = signal_engine.generate_signal(
            candles, structure_data, zone_data, order_flow_data, situational_context
        )

        if signal:
            # Log and visualize
            dashboard_logger.log_signal(symbol, signal)
            visualizer.render(
                candles, order_flow_data, signal, structure_data, zone_data
            )

        else:
            print("No valid signal generated at this time.")

        time.sleep(60)

    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(60)
