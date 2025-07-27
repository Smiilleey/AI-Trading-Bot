import MetaTrader5 as mt5

def initialize(symbol):
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    if not mt5.symbol_select(symbol, True):
        print(f"Symbol {symbol} not found")
        return False
    return True

def get_candles(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    candles = []
    for r in rates:
        candles.append({
            "time": r["time"],
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["tick_volume"]
        })
    return candles

def place_order(symbol, direction, lot, sl, tp):
    type_ = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if direction == "BUY" else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": type_,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "AutoTrade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    return result
