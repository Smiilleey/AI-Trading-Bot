from typing import TypedDict, Optional, Literal


class DecisionIdea(TypedDict, total=False):
    """Standardized decision payload exchanged between policy/intelligence/risk.

    Fields are optional to allow progressive adoption across modules.
    """
    side: Literal["buy", "sell", "hold"]
    score: float
    ml: float
    rule: float
    prophetic: float
    threshold: float
    regime: Optional[str]
    # Additional context fields can be added over time


