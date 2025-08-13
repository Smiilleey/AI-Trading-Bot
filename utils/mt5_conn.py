def ensure_mt5_connection():
    """
    Initialize MetaTrader5 reliably with retries.
    Raise explicit errors if not available.
    """
    try:
        import MetaTrader5 as mt5
    except Exception as e:
        raise ImportError("MetaTrader5 module is required but not installed") from e

    if mt5.initialize():
        return True

    for _ in range(3):
        if mt5.initialize():
            return True

    raise ConnectionError("MetaTrader5 connection could not be established after retries.")
