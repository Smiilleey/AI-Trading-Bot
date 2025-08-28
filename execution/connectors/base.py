class BaseConnector:
    name="base"
    lot_step=0.01; min_lot=0.01

    # --- Market data ---
    def get_quote(self, symbol)->dict:
        """Return dict: {'bid':float,'ask':float,'spread_pips':float,'time':ts}"""
        raise NotImplementedError

    def pip(self, symbol)->float:
        return 0.01 if "JPY" in symbol else 0.0001

    def pip_value(self, symbol)->float:
        """$ per pip per 1 lot (approx or broker precise)"""
        return 10.0

    # --- Account state ---
    def equity(self)->float:
        raise NotImplementedError
    def open_positions_count(self)->int:
        raise NotImplementedError

    # --- Trading ---
    def estimate_slippage_pips(self, symbol, side, intended_price)->float:
        q=self.get_quote(symbol); pip=self.pip(symbol)
        best = q["ask"] if side=="buy" else q["bid"]
        return abs(best-intended_price)/pip

    def place_market(self, symbol, side, lots)->str|None:
        raise NotImplementedError
    def attach_stop_loss(self, trade_id, stop_pips)->bool:
        raise NotImplementedError
    def attach_take_profit(self, trade_id, rr=2.0)->bool:
        raise NotImplementedError
