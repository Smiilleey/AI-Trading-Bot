import json, os
from collections import deque

class AutoWeightTuner:
    """
    Adaptive weight tuner that adjusts trust in ML vs rules vs prophetic signals
    based on recent performance. Designed to be slow & stable.
    """
    def __init__(self, state_path=os.path.join('memory', 'weights_state.json')):
        self.state_path = state_path
        self.history = deque(maxlen=100)
        # Start weights (editable). Bounds enforced in get_weights().
        self.weights = {'ml_weight': 0.5, 'rule_weight': 0.4, 'prophetic_weight': 0.1}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                if 'weights' in data:
                    self.weights.update(data['weights'])
        except Exception:
            pass

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump({'weights': self.weights}, f)
        except Exception:
            pass

    def update(self, outcome: float, led_by: str = 'ml'):
        """
        outcome: 1.0 for win, 0.0 for loss
        led_by: 'ml' | 'rules' | 'prophetic'
        """
        try:
            self.history.append({'outcome': float(outcome), 'led_by': led_by})
            if len(self.history) < 20:
                return  # not enough data yet

            perf = {'ml': [], 'rules': [], 'prophetic': []}
            for h in self.history:
                perf[h['led_by']].append(h['outcome'])

            def avg(xs): return sum(xs)/len(xs) if xs else 0.0
            ml_perf = avg(perf['ml'])
            rule_perf = avg(perf['rules'])
            pro_perf = avg(perf['prophetic'])

            def clamp(x, lo=0.05, hi=0.85): return max(lo, min(hi, x))

            # Gentle nudges
            if ml_perf > rule_perf + 0.05:
                self.weights['ml_weight'] = clamp(self.weights['ml_weight'] + 0.02)
                self.weights['rule_weight'] = clamp(self.weights['rule_weight'] - 0.02)
            elif rule_perf > ml_perf + 0.05:
                self.weights['rule_weight'] = clamp(self.weights['rule_weight'] + 0.02)
                self.weights['ml_weight'] = clamp(self.weights['ml_weight'] - 0.02)

            # Prophetic gets small, bounded influence
            if pro_perf > 0.55:
                self.weights['prophetic_weight'] = clamp(self.weights['prophetic_weight'] + 0.01, 0.0, 0.25)
            else:
                self.weights['prophetic_weight'] = clamp(self.weights['prophetic_weight'] - 0.005, 0.0, 0.25)

            # Normalize softly if sum drifts too high
            total = sum(self.weights.values())
            if total > 1.1:
                for k in self.weights:
                    self.weights[k] /= total

            self._save()
        except Exception:
            pass

    def get_weights(self):
        # Hard floors so the system stays hybrid (never 100% one thing)
        w = dict(self.weights)
        w['ml_weight']       = max(0.20, min(0.80, w['ml_weight']))
        w['rule_weight']     = max(0.20, min(0.80, w['rule_weight']))
        w['prophetic_weight']= max(0.00, min(0.25, w['prophetic_weight']))
        return w
