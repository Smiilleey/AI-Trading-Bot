import json, os
from collections import deque

class AutoWeightTuner:
    def __init__(self, state_path=os.path.join('memory', 'weights_state.json')):
        self.state_path = state_path
        self.history = deque(maxlen=100)
        self.weights = {'ml_weight': 0.5, 'rule_weight': 0.4, 'prophetic_weight': 0.1}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                self.weights.update(data.get('weights', {}))
        except Exception: pass

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w') as f: json.dump({'weights': self.weights}, f)
        except Exception: pass

    def update(self, outcome, led_by='ml'):
        try:
            self.history.append({'outcome': float(outcome), 'led_by': led_by})
            if len(self.history) < 20: return
            perf = {'ml': [], 'rules': [], 'prophetic': []}
            for h in self.history: perf[h['led_by']].append(h['outcome'])
            avg = lambda xs: sum(xs)/len(xs) if xs else 0.0
            ml, ru, pr = avg(perf['ml']), avg(perf['rules']), avg(perf['prophetic'])

            clamp = lambda x, lo, hi: max(lo, min(hi, x))
            if ml > ru + 0.05:
                self.weights['ml_weight'] = clamp(self.weights['ml_weight'] + 0.02, 0.2, 0.8)
                self.weights['rule_weight'] = clamp(self.weights['rule_weight'] - 0.02, 0.2, 0.8)
            elif ru > ml + 0.05:
                self.weights['rule_weight'] = clamp(self.weights['rule_weight'] + 0.02, 0.2, 0.8)
                self.weights['ml_weight'] = clamp(self.weights['ml_weight'] - 0.02, 0.2, 0.8)

            if pr > 0.55:
                self.weights['prophetic_weight'] = clamp(self.weights['prophetic_weight'] + 0.01, 0.0, 0.25)
            else:
                self.weights['prophetic_weight'] = clamp(self.weights['prophetic_weight'] - 0.005, 0.0, 0.25)

            total = sum(self.weights.values())
            if total > 1.1:
                for k in self.weights: self.weights[k] /= total
            self._save()
        except Exception: pass

    def get_weights(self):
        return dict(self.weights)
