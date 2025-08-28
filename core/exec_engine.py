import time

class ExecEngine:
    """
    Adapter around your MT5 API calls. Replace stubs with actual broker ops.
    """
    def __init__(self, mt5_adapter, logger):
        self.mt5 = mt5_adapter
        self.logger = logger

    def estimate_slippage_pips(self, symbol, intended_price, side):
        # Replace with real quote/last/DOM compare if available
        best = self.mt5.best_price(symbol, side)  # must return a price
        pip = self.mt5.pip(symbol)
        return abs(best - intended_price) / pip

    def submit_bracket(self, symbol, side, size, stop_pips, meta=None):
        price = self.mt5.best_price(symbol, side)
        trade_id = self.mt5.place_market(symbol, side, size)
        if not trade_id:
            self.logger.error(f"Order failed: {symbol} {side} {size}")
            return None
        # place SL/TP (example: TP at +2R)
        sl = self.mt5.attach_stop_loss(trade_id, stop_pips)
        tp = self.mt5.attach_take_profit(trade_id, rr=2.0)
        self.logger.info(f"EXEC {symbol} side={side} lots={size} id={trade_id} meta={meta}")
        return {"id": trade_id, "price": price, "sl": sl, "tp": tp, "meta": meta}
