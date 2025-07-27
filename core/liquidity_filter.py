from utils.session_timer import is_in_liquidity_window

class LiquidityFilter:
    def is_liquid_time(self, timestamp):
        return is_in_liquidity_window(timestamp)

    def filter_signal(self, signal_data, timestamp):
        if not self.is_liquid_time(timestamp):
            signal_data['reasons'].append("Outside Liquidity Window ❌")
        else:
            signal_data['reasons'].append("Inside Liquidity Window ✅")
        return signal_data
