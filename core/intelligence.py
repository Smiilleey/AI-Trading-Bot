from utils.auto_weight_tuner import AutoWeightTuner
from core.regime_classifier import RegimeClassifier
from core.rulebook import RuleBook
from core.prophetic_engine import PropheticEngine

class IntelligenceCore:
    """
    Platform-agnostic decision engine.
    Plug in your own ML model by overriding predict_confidence(...).
    """
    def __init__(self, logger, base_threshold:float=0.62, require_all_confirm=True):
        self.logger=logger; self.tuner=AutoWeightTuner()
        self.regime=RegimeClassifier(); self.rulebook=RuleBook(); self.prophet=PropheticEngine()
        self.base_th=base_threshold; self.require_all=require_all_confirm

    # --- Override this with your ML predictor ---
    def predict_confidence(self, symbol, features)->float:
        # Placeholder: replace with your trained model's probability [0..1]
        return float(features.get("ml_confidence", 0.55))

    def _hybrid(self, ml, rule, pro):
        w=self.tuner.get_weights()
        return ml*w['ml_weight'] + rule*w['rule_weight'] + pro*w['prophetic_weight']

    def decide(self, symbol, features:dict):
        ml = float(self.predict_confidence(symbol, features))
        rule = float(self.rulebook.score(symbol, features))
        pro = float(self.prophet.timing(symbol, features))  # [-1..+1]
        regime = self.regime.classify(features)
        th = self.regime.dynamic_entry_threshold(self.base_th, regime)

        if self.require_all:
            aligned=(ml>=th and rule>=th and abs(pro)>=0.1)
            if not aligned: return None

        hybrid=self._hybrid(ml,rule,pro)
        if hybrid < th: return None

        side = "buy" if features.get("direction","long") in ("long","buy") else "sell"
        return {"symbol":symbol,"score":hybrid,"ml":ml,"rule":rule,"prophetic":pro,"threshold":th,"regime":regime,"side":side}
