def within_spread_limit(current_spread_pips: float, max_spread_pips: float) -> bool:
    return current_spread_pips <= max_spread_pips

def within_slippage_limit(estimated_slippage_pips: float, max_slippage_pips: float) -> bool:
    return estimated_slippage_pips <= max_slippage_pips
