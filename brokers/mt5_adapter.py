class MT5Adapter:
    def pip(self, symbol):          # pip size in price units
        return 0.0001 if "JPY" not in symbol else 0.01
    def pip_value(self, symbol):    # $ per pip per 1.0 lot (approx; replace with broker calc)
        return 10.0 if "JPY" not in symbol else 9.0
    def best_price(self, symbol, side):
        # query current bid/ask. If side=='buy' -> ask; 'sell' -> bid
        q = self.quote(symbol)
        return q["ask"] if side == "buy" else q["bid"]
    def quote(self, symbol):
        # TODO: connect to MT5 to fetch current quote
        return {"bid": 1.00000, "ask": 1.00020, "spread": 2.0}
    def place_market(self, symbol, side, lots):
        # TODO: send market order via MT5
        return f"T{symbol}-{side}-lot{lots}"
    def attach_stop_loss(self, trade_id, stop_pips):
        # TODO: attach SL to the order
        return True
    def attach_take_profit(self, trade_id, rr=2.0):
        # TODO: attach TP based on RR
        return True
