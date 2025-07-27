def analyze_situations(candles):
    results = {
        "day_bias": None,
        "notes": []
    }

    if len(candles) < 5:
        results["notes"].append("Not enough historical data")
        return results

    monday = candles[-5]
    wednesday = candles[-3]
    thursday = candles[-2]
    friday = candles[-1]

    if friday["high"] < thursday["high"]:
        results["day_bias"] = "expect_monday_to_revisit_friday_low"
        results["notes"].append("Friday's high < Thursday's high")

    if wednesday["high"] < monday["high"]:
        results["day_bias"] = "expect_thursday_to_revisit_wed_low"
        results["notes"].append("Wednesday's high < Monday's high")

    return results
