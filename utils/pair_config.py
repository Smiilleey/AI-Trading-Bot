# utils/pair_config.py

TRADING_PAIRS = {
    # Major Pairs (Exness format with 'z' suffix)
    "GBPUSDz": {
        "name": "GU",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["London", "New York"],
        "correlations": ["GBPJPYz", "EURUSDz", "EURGBPz", "GBPNZDz"],
        "is_main": True
    },
    "EURUSDz": {
        "name": "EU",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["London", "New York"],
        "correlations": ["EURGBPz", "EURJPYz", "EURNZDz"],
        "is_main": True
    },
    "USDJPYz": {
        "name": "UJ",
        "pip_value": 0.01,
        "min_lot": 0.01,
        "sessions": ["Tokyo", "New York"],
        "correlations": ["EURJPYz", "GBPJPYz", "AUDJPYz", "NZDJPYz", "CADJPYz", "CHFJPYz"],
        "is_main": True
    },

    # GBP Pairs
    "GBPJPY": {
        "name": "GJ",
        "pip_value": 0.01,
        "min_lot": 0.01,
        "sessions": ["London", "Tokyo"],
        "correlations": ["GBPUSD", "USDJPY", "EURJPY"],
        "is_main": True
    },
    "GBPNZD": {
        "name": "GN",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["London", "Sydney"],
        "correlations": ["GBPUSD", "NZDUSD"],
        "is_main": True
    },
    "EURGBP": {
        "name": "EG",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["London"],
        "correlations": ["EURUSD", "GBPUSD"],
        "is_main": True
    },

    # EUR Pairs
    "EURJPY": {
        "name": "EJ",
        "pip_value": 0.01,
        "min_lot": 0.01,
        "sessions": ["London", "Tokyo"],
        "correlations": ["EURUSD", "USDJPY", "GBPJPY"],
        "is_main": True
    },
    "EURNZD": {
        "name": "EN",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["London", "Sydney"],
        "correlations": ["EURUSD", "NZDUSD"],
        "is_main": True
    },

    # JPY Pairs
    "AUDJPY": {
        "name": "AJ",
        "pip_value": 0.01,
        "min_lot": 0.01,
        "sessions": ["Sydney", "Tokyo"],
        "correlations": ["AUDUSD", "USDJPY"],
        "is_main": True
    },
    "NZDJPY": {
        "name": "NJ",
        "pip_value": 0.01,
        "min_lot": 0.01,
        "sessions": ["Sydney", "Tokyo"],
        "correlations": ["NZDUSD", "USDJPY"],
        "is_main": True
    },
    "CADJPY": {
        "name": "CJ",
        "pip_value": 0.01,
        "min_lot": 0.01,
        "sessions": ["Tokyo", "New York"],
        "correlations": ["USDCAD", "USDJPY"],
        "is_main": True
    },
    "CHFJPY": {
        "name": "CHJ",
        "pip_value": 0.01,
        "min_lot": 0.01,
        "sessions": ["Tokyo", "London"],
        "correlations": ["USDCHF", "USDJPY"],
        "is_main": True
    },

    # USD Pairs
    "AUDUSD": {
        "name": "AU",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["Sydney", "Asia"],
        "correlations": ["AUDJPY", "USDJPY"],
        "is_main": True
    },
    "USDCAD": {
        "name": "UCad",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["New York"],
        "correlations": ["CADJPY", "USDJPY"],
        "is_main": True
    },
    "USDCHF": {
        "name": "UChf",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["London"],
        "correlations": ["CHFJPY", "USDJPY"],
        "is_main": True
    },
    "NZDUSD": {
        "name": "NU",
        "pip_value": 0.0001,
        "min_lot": 0.01,
        "sessions": ["Sydney", "Asia"],
        "correlations": ["EURNZD", "GBPNZD", "NZDJPY"],
        "is_main": True
    }
    ,
    # Metals & Crypto
    "XAUUSD": {
        "name": "Gold",
        "pip_value": 0.01,  # treat 1 cent as pip unit
        "min_lot": 0.01,
        "sessions": ["London AM Fix", "London PM Fix", "COMEX Open"],
        "correlations": ["USDX", "US10Y", "USDJPY", "BTCUSD"],
        "is_main": True
    },
    "BTCUSD": {
        "name": "BTC",
        "pip_value": 1.0,  # treat $1 as pip unit for sizing conversion
        "min_lot": 0.01,
        "sessions": ["24/7", "US Peak", "Asia Peak", "Europe Peak"],
        "correlations": ["XAUUSD", "EURUSD", "USDJPY"],
        "is_main": True
    }
}