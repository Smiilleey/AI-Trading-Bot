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
        try:
            import MetaTrader5 as mt5
            q = mt5.symbol_info_tick(symbol)
            if not q:
                raise RuntimeError("No tick")
            pip = self.pip(symbol)
            return {"bid": float(q.bid), "ask": float(q.ask), "spread": (q.ask - q.bid) / pip}
        except Exception:
            # Fallback mock for offline scenarios
            return {"bid": 1.00000, "ask": 1.00020, "spread": 2.0}
    def place_market(self, symbol, side, lots):
        try:
            import MetaTrader5 as mt5
            q = mt5.symbol_info_tick(symbol)
            order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
            price = q.ask if side == "buy" else q.bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lots),
                "type": order_type,
                "price": price,
                "deviation": 20,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req)
            if getattr(res, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                return str(res.order)
        except Exception:
            pass
        # Fallback mock id
        return f"T{symbol}-{side}-lot{lots}"
    def attach_stop_loss(self, trade_id, stop_pips):
        try:
            # Placeholder: depends on broker; return True for now
            return True
        except Exception:
            return False
    def attach_take_profit(self, trade_id, rr=2.0):
        try:
            return True
        except Exception:
            return False
