# memory/learning.py

import json
import os
from datetime import datetime

MEMORY_FILE = "memory/pattern_memory.json"

class LearningEngine:
    """
    Institutional pattern memory & learning engine.
    - Stores every pattern, outcome, RR, tags, context, entry/exit time
    - Suggests symbolic confidence for dashboard and engine
    - Provides stats: winrate, avg RR, streaks, etc.
    """
    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = memory_file
        self.memory = self._load_memory()

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return {}
        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                # Validate the loaded data structure
                if not isinstance(data, dict):
                    print(f"Warning: Invalid memory file format, resetting to empty dict")
                    return {}
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load memory file ({e}), starting with empty memory")
            return {}
        except Exception as e:
            print(f"Warning: Unexpected error loading memory file ({e}), starting with empty memory")
            return {}

    def _save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)

    def record_result(self, pair, context, signal, outcome, rr, entry_time, tags=None, exit_time=None, pnl=None):
        """
        Records a new trade outcome to memory.
        """
        record = {
            "pair": pair,
            "context": context,
            "signal": signal,
            "outcome": outcome,
            "rr": rr,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "tags": tags or [],
            "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat()
        }
        if pair not in self.memory:
            self.memory[pair] = []
        self.memory[pair].append(record)
        self._save_memory()

    def suggest_confidence(self, pair, signal_type):
        """
        Returns symbolic confidence ("high"/"medium"/"low"/"unknown") for a pair/signal.
        """
        if not pair or not signal_type:
            return "unknown"
            
        records = self.memory.get(pair, [])
        if not records:
            return "unknown"
            
        relevant = [r for r in records if r.get("signal") == signal_type]
        if not relevant:
            return "low"
            
        wins = [r for r in relevant if r.get("outcome") == "win"]
        win_rate = len(wins) / len(relevant) if relevant else 0
        
        # Ensure win_rate is within valid range
        win_rate = max(0.0, min(1.0, win_rate))
        
        if win_rate >= 0.7:
            return "high"
        elif win_rate >= 0.5:
            return "medium"
        else:
            return "low"

    def get_pattern_stats(self, pair, signal_type=None):
        """
        Returns stats for all or given signal type: total, wins, win_rate, avg RR, tags, streak.
        """
        records = self.memory.get(pair, [])
        if signal_type:
            records = [r for r in records if r["signal"] == signal_type]
        total = len(records)
        if total == 0:
            return {"total": 0, "wins": 0, "win_rate": 0, "average_rr": 0, "tags": [], "streak": 0}
        wins = [r for r in records if r["outcome"] == "win"]
        avg_rr = sum(r.get("rr", 0) for r in records) / total
        tag_list = [tag for r in records for tag in r.get("tags", [])]
        last_results = [r["outcome"] for r in records[-10:]]
        streak = 0
        for outcome in reversed(last_results):
            if outcome == "win":
                streak = streak + 1 if streak >= 0 else 1
            elif outcome == "loss":
                streak = streak - 1 if streak <= 0 else -1
            else:
                break
        return {
            "total": total,
            "wins": len(wins),
            "win_rate": len(wins) / total,
            "average_rr": avg_rr,
            "tags": list(set(tag_list)),
            "streak": streak
        }

    def load_memory(self):
        """Manual memory reload (for hot-reload in runtime)."""
        self.memory = self._load_memory()

    def save_memory(self):
        """Manual save (not usually needed, for advanced use)."""
        self._save_memory()
