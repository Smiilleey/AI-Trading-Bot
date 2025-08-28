import time, random
from .base import BaseConnector

class PaperConnector(BaseConnector):
    name="paper"
    def __init__(self, starting_equity=10000.0):
        self._equity=starting_equity; self._open=0

    def get_quote(self, symbol)->dict:
        # toy random walk + spread
        price=1.1000 + random.uniform(-0.002,0.002) if "JPY" not in symbol else 150 + random.uniform(-0.5,0.5)
        pip=self.pip(symbol); spread_pips=2.0
        bid=price; ask=price+spread_pips*pip
        return {"bid":bid,"ask":ask,"spread_pips":spread_pips,"time":time.time()}

    def equity(self)->float: return float(self._equity)
    def open_positions_count(self)->int: return int(self._open)

    def place_market(self, symbol, side, lots)->str|None:
        self._open += 1
        return f"PAPER-{int(time.time()*1000)}"

    def attach_stop_loss(self, trade_id, stop_pips)->bool: return True
    def attach_take_profit(self, trade_id, rr=2.0)->bool: return True

    # Simple PnL simulator to close trades randomly (for demo/testing loops)
    def maybe_close_random(self, symbol)->dict|None:
        import random
        if random.random()<0.15 and self._open>0:
            pnl = random.uniform(-30, 60)  # dollars
            self._equity += pnl; self._open -= 1
            return {"symbol":symbol,"pnl":pnl,"confidence":0.0,"rule_score":0.0,"prophetic_signal":0.0}
        return None
