import uuid
import random
from typing import Any, Dict, Optional
from .types import DecisionIdea


class PolicyService:
    """
    Central policy orchestrator with optional A/B or shadow evaluation.

    - champion: uses the system's primary intelligence (e.g., IntelligenceCore)
    - challenger: optional alternative scorer/policy (stubbed by default)
    - mode: "shadow" (evaluate challenger but do not act) or "active" (route a
      percentage of decisions to challenger)
    - challenger_pct: in active mode, fraction of decisions to route to challenger
    """

    def __init__(self,
                 champion: Any,
                 challenger: Optional[Any] = None,
                 mode: str = "shadow",
                 challenger_pct: float = 0.0,
                 logger: Optional[Any] = None) -> None:
        self.champion = champion
        self.challenger = challenger
        self.mode = (mode or "shadow").lower()
        self.challenger_pct = float(max(0.0, min(1.0, challenger_pct or 0.0)))
        self.logger = logger

    def new_trace_id(self) -> str:
        return str(uuid.uuid4())

    def _challenger_score(self, symbol: str, features: Dict[str, Any]) -> float:
        # Minimal stub if no challenger is provided
        try:
            if self.challenger is None:
                # Fallback to the champion's ML confidence path where possible
                return float(features.get("ml_confidence", 0.5))
            if hasattr(self.challenger, "predict_confidence"):
                return float(self.challenger.predict_confidence(symbol, features))
            if hasattr(self.challenger, "decide"):
                idea = self.challenger.decide(symbol, features)
                return float(idea.get("score", 0.0)) if idea else 0.0
        except Exception:
            pass
        return 0.5

    def decide(self, symbol: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a dict with keys:
          - idea: the action proposal from the acting policy (champion or challenger)
          - meta: {trace_id, variant, shadow: {challenger_score?}}
        """
        trace_id = self.new_trace_id()
        
        # Shadow/challenger evaluation (always compute for observability)
        challenger_score = None
        try:
            challenger_score = self._challenger_score(symbol, features)
        except Exception:
            challenger_score = None

        # Decide acting variant
        acting_variant = "champion"
        route_to_challenger = False
        if self.mode == "active" and self.challenger is not None:
            try:
                route_to_challenger = random.random() < self.challenger_pct
            except Exception:
                route_to_challenger = False
        if route_to_challenger:
            acting_variant = "challenger"

        # Get decision from acting policy with robust fallback
        idea: Optional[DecisionIdea] = None
        if acting_variant == "challenger" and self.challenger is not None:
            try:
                if hasattr(self.challenger, "decide"):
                    idea = self.challenger.decide(symbol, features)
                elif hasattr(self.challenger, "predict_confidence"):
                    score = float(self.challenger.predict_confidence(symbol, features))
                    idea = {"side": "buy" if score >= 0 else "sell", "score": score, "ml": score, "rule": 0.0, "prophetic": 0.0, "threshold": 0.0, "regime": None}
            except Exception:
                idea = None

        # Fallback to champion if challenger failed or not routed
        if idea is None:
            try:
                idea = self.champion.decide(symbol, features)
                acting_variant = "champion"
            except Exception:
                idea = None

        meta = {
            "trace_id": trace_id,
            "variant": acting_variant,
            "shadow": {"challenger_score": challenger_score,
                        "mode": self.mode,
                        "challenger_pct": self.challenger_pct}
        }

        return {"idea": idea, "meta": meta}


