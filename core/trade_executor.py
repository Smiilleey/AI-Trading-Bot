# core/trade_executor.py

from utils.mt5_connector import place_order

def execute_trade(signal_info, price, lot=0.1, sl=None, tp=None, dry_run=False):
    """
    Executes a trade using signal info and price.
    - signal_info: dict from SignalEngine (must have 'signal', 'confidence', etc.)
    - price: float, current execution price
    - lot: trade size (default 0.1)
    - sl/tp: stop-loss / take-profit prices (optional)
    - dry_run: if True, does not place real order, just simulates for dashboard/dev
    Returns result dict for dashboard/logger/memory.
    """
    if not signal_info:
        return {"status": "no_signal", "msg": "No signal to execute."}

    side = "BUY" if signal_info.get("signal") == "bullish" else "SELL"
    conf = signal_info.get("confidence", "unknown")
    reasons = signal_info.get("reasons", [])
    cisd = signal_info.get("cisd", False)

    # Display execution info
    print(f"[EXECUTING TRADE] {side} @ {price} | Lot: {lot} | SL: {sl} | TP: {tp} | Confidence: {conf} | CISD: {cisd}")
    for reason in reasons:
        print(f"  → {reason}")

    if dry_run:
        return {
            "status": "simulated",
            "side": side,
            "price": price,
            "lot": lot,
            "confidence": conf,
            "cisd": cisd,
            "msg": "Simulated trade (dry run)."
        }

    # Real order placement
    try:
        result = place_order(
            symbol=signal_info.get("pair", "UNKNOWN"),
            direction=side,
            lot=lot,
            sl=sl,
            tp=tp,
            comment="AutoTrade"
        )
        status = "success" if result.retcode == 10009 else "fail"  # 10009 = TRADE_RETCODE_DONE
        return {
            "status": status,
            "side": side,
            "price": price,
            "lot": lot,
            "confidence": conf,
            "cisd": cisd,
            "result": str(result),
            "msg": "Trade executed." if status == "success" else result.comment
        }
    except Exception as e:
        return {
            "status": "error",
            "side": side,
            "msg": f"Execution failed: {e}"
        }
