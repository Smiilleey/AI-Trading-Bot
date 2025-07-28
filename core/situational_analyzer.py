# core/situational_analyzer.py

class SituationalAnalyzer:
    """
    Institutional situational logic layer:
    - Detects special weekly patterns (Friday/Monday, Wednesday/Thursday, etc.)
    - Outputs symbolic context tags and dashboard notes
    - Easily extendable for more advanced pattern memory
    """
    def __init__(self):
        pass

    def analyze(self, candles):
        results = {
            "day_bias": None,
            "notes": [],
            "situational_tags": []
        }

        if len(candles) < 5:
            results["notes"].append("Not enough historical data (need 5+ daily candles)")
            return results

        # Assume candles are daily and ordered oldest to newest
        monday = candles[-5]
        wednesday = candles[-3]
        thursday = candles[-2]
        friday = candles[-1]

        # Pattern 1: Friday's high < Thursday's high → Monday likely to revisit Friday's low
        if friday["high"] < thursday["high"]:
            results["day_bias"] = "expect_monday_to_revisit_friday_low"
            results["notes"].append("Friday's high < Thursday's high: Monday may sweep Friday's low")
            results["situational_tags"].append("friday_thursday_reversal")

        # Pattern 2: Wednesday's high < Monday's high → Thursday likely to revisit Wednesday's low
        if wednesday["high"] < monday["high"]:
            results["day_bias"] = "expect_thursday_to_revisit_wed_low"
            results["notes"].append("Wednesday's high < Monday's high: Thursday may sweep Wednesday's low")
            results["situational_tags"].append("wednesday_monday_pullback")

        # Add more pattern rules here as your system evolves!

        if not results["day_bias"]:
            results["notes"].append("No situational weekly pattern detected")

        return results
