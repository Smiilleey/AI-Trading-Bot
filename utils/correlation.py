_CORR_GROUPS = [
    {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
    {"USDJPY", "USDCHF"},
    {"USDCAD", "WTICOUSD", "UKOILUSD"},  # crude sensitivity
    {"XAUUSD", "XAGUSD"}
]

def is_correlated(sym_a: str, sym_b: str) -> bool:
    sa, sb = sym_a.upper(), sym_b.upper()
    for g in _CORR_GROUPS:
        if sa in g and sb in g:
            return True
    return False
