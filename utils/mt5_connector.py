# utils/mt5_connector.py

import MetaTrader5 as mt5
import time
from datetime import datetime, timezone

def initialize(symbol, login=None, password=None, server=None, max_retries=3):
    """
    Initializes MetaTrader 5 connection with retry logic and fallback handling.
    Optionally supports login/password/server for VPS/cloud use.
    Returns True on success, raises Exception on failure.
    """
    for attempt in range(max_retries):
        try:
            if not mt5.initialize(login=login, password=password, server=server):
                error_msg = f"MT5 initialization failed (attempt {attempt + 1}/{max_retries}): {mt5.last_error()}"
                print(error_msg)
                if attempt == max_retries - 1:
                    raise ConnectionError(error_msg)
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            
            if symbol:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    error_msg = f"Symbol {symbol} not found on MT5 terminal (attempt {attempt + 1}/{max_retries})."
                    print(error_msg)
                    if attempt == max_retries - 1:
                        raise ValueError(error_msg)
                    time.sleep(2 ** attempt)
                    continue
                
                if not mt5.symbol_select(symbol, True):
                    error_msg = f"Failed to select symbol {symbol} (attempt {attempt + 1}/{max_retries})"
                    print(error_msg)
                    if attempt == max_retries - 1:
                        raise RuntimeError(error_msg)
                    time.sleep(2 ** attempt)
                    continue
            
            print(f"✅ MT5 initialized successfully on attempt {attempt + 1}")
            return True
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"⚠️ MT5 initialization attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)
    
    return False

def get_candles(symbol, timeframe, count=100, max_retries=2):
    """
    Returns a list of dicts: symbol, time (ISO), open, high, low, close, tick_volume.
    Includes retry logic and fallback handling.
    """
    for attempt in range(max_retries):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                if attempt == max_retries - 1:
                    raise Exception(f"No candle data for {symbol} on {timeframe}")
                print(f"⚠️ No candle data for {symbol} on {timeframe} (attempt {attempt + 1}). Retrying...")
                time.sleep(1)
                continue
            
            candles = []
            for r in rates:
                candles.append({
                    "symbol": symbol,
                    "time": datetime.fromtimestamp(r['time'], tz=timezone.utc).isoformat(),
                    "open": r["open"],
                    "high": r["high"],
                    "low": r["low"],
                    "close": r["close"],
                    "tick_volume": r["tick_volume"]
                })
            return candles
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"⚠️ Error getting candles for {symbol} (attempt {attempt + 1}): {e}. Retrying...")
            time.sleep(1)
    
    return []

def fetch_latest_data(symbol):
    """
    Gets the latest price and timestamp for a symbol (for signal logic).
    """
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        raise Exception(f"No tick data for {symbol}")
    return {
        "price": tick.last,
        "timestamp": datetime.utcfromtimestamp(tick.time).replace(tzinfo=timezone.utc).isoformat()
    }

def place_order(symbol, direction, lot, sl=None, tp=None, comment="AutoTrade"):
    """
    Places a BUY or SELL order.
    direction: "BUY" or "SELL"
    lot: trade size (float)
    sl: stop-loss price (optional)
    tp: take-profit price (optional)
    Returns MT5 order result.
    """
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        raise Exception(f"No tick data for {symbol}")
    price = tick.ask if direction.upper() == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl or 0.0,
        "tp": tp or 0.0,
        "deviation": 20,
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment}, code: {result.retcode}")
    else:
        print(f"Order placed successfully: Ticket {result.order}")
    return result

def shutdown():
    """Closes the MetaTrader 5 connection."""
    mt5.shutdown()
