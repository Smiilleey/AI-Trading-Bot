from .base import BaseConnector
class MT5Connector(BaseConnector):
    name="mt5"
    def __init__(self):
        import MetaTrader5 as mt5
        if not mt5.initialize(): raise RuntimeError("MT5 init failed")
        self.mt5=mt5
    def get_quote(self, symbol)->dict:
        q=self.mt5.symbol_info_tick(symbol)
        pip=self.pip(symbol); spread_pips=(q.ask-q.bid)/pip
        return {"bid":q.bid,"ask":q.ask,"spread_pips":spread_pips,"time":q.time}
    def equity(self)->float:
        acc=self.mt5.account_info(); return float(acc.equity)
    def open_positions_count(self)->int:
        poss=self.mt5.positions_get(); return 0 if poss is None else len(poss)
    def place_market(self, symbol, side, lots)->str|None:
        # implement mt5.order_send(...) here
        return "MT5-ORDER-ID"
    def attach_stop_loss(self, trade_id, stop_pips)->bool: return True
    def attach_take_profit(self, trade_id, rr=2.0)->bool: return True
