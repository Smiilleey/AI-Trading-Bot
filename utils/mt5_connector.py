import MetaTrader5 as mt5
from datetime import datetime, timezone
import time

def initialize(symbol=None, login=None, password=None, server=None, max_retries=3):
    """
    Initialize MT5 connection with retry logic.
    Returns True on success, raises Exception on failure.
    """
    for attempt in range(max_retries):
        try:
            # Convert login to integer if provided
            mt5_login = int(login) if login else None
            
            if not mt5.initialize(login=mt5_login, password=password, server=server):
                error_msg = f"MT5 initialization failed (attempt {attempt + 1}/{max_retries}): {mt5.last_error()}"
                print(error_msg)
                if attempt == max_retries - 1:
                    raise ConnectionError(error_msg)
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
    
            if symbol:
                # First, let's see what symbols are available
                symbols = mt5.symbols_get()
                if symbols:
                    print(f"📊 Available symbols: {[s.name for s in symbols[:10]]}...")
                
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
    comment: order comment (optional)
    """
    try:
        # Get current market price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            raise Exception(f"Could not get tick data for {symbol}")
        
        # Determine order type and price
        if direction.upper() == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction.upper() == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'BUY' or 'SELL'")
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add stop loss if specified
        if sl is not None:
            request["sl"] = sl
        
        # Add take profit if specified
        if tp is not None:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise Exception(f"Order failed: {result.comment}, code: {result.retcode}")
        
        return result.order
        
    except Exception as e:
        print(f"Order failed: {e}")
        return None

def close_position(ticket):
    """
    Closes a position by ticket number.
    """
    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise Exception(f"Position {ticket} not found")
        
        position = position[0]
        
        # Determine close order type
        if position.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise Exception(f"Close order failed: {result.comment}, code: {result.retcode}")
        
        return True
        
    except Exception as e:
        print(f"Close position failed: {e}")
        return False

def get_positions():
    """
    Returns all open positions.
    """
    try:
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        return [
            {
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "profit": pos.profit,
                "swap": pos.swap,
                "time": datetime.fromtimestamp(pos.time, tz=timezone.utc).isoformat()
            }
            for pos in positions
        ]
    except Exception as e:
        print(f"Failed to get positions: {e}")
        return []

def shutdown():
    """
    Shutdown MT5 connection.
    """
    try:
        mt5.shutdown()
        print("✅ MT5 connection closed")
    except Exception as e:
        print(f"⚠️ Error closing MT5 connection: {e}")



