import random
from core.policy_service import PolicyService


class DummyChampion:
    def decide(self, symbol, features):
        return {"side": "buy", "score": 0.7, "ml": 0.7, "rule": 0.5, "prophetic": 0.1, "threshold": 0.6, "regime": "normal"}


class DummyChallenger:
    def decide(self, symbol, features):
        return {"side": "sell", "score": -0.2, "ml": -0.2, "rule": 0.2, "prophetic": 0.0, "threshold": 0.6, "regime": "normal"}


def test_shadow_mode_returns_champion_and_logs_challenger_score():
    svc = PolicyService(champion=DummyChampion(), challenger=DummyChallenger(), mode="shadow", challenger_pct=0.5)
    out = svc.decide("EURUSD", {"ml_confidence": 0.55})
    assert out["meta"]["variant"] == "champion"
    assert out["idea"]["side"] == "buy"
    assert "challenger_score" in out["meta"]["shadow"]


def test_active_mode_can_route_to_challenger(monkeypatch):
    svc = PolicyService(champion=DummyChampion(), challenger=DummyChallenger(), mode="active", challenger_pct=1.0)
    # Force RNG to always route
    monkeypatch.setattr(random, "random", lambda: 0.0)
    out = svc.decide("EURUSD", {"ml_confidence": 0.55})
    assert out["meta"]["variant"] == "challenger"
    assert out["idea"]["side"] == "sell"


def test_active_mode_fallback_to_champion_on_error(monkeypatch):
    class BadChallenger:
        def decide(self, s, f):
            raise RuntimeError("boom")

    svc = PolicyService(champion=DummyChampion(), challenger=BadChallenger(), mode="active", challenger_pct=1.0)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    out = svc.decide("EURUSD", {"ml_confidence": 0.55})
    assert out["meta"]["variant"] == "champion"
    assert out["idea"]["side"] == "buy"


