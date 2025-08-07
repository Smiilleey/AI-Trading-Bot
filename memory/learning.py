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
        self.unsaved_changes = 0
        self.save_threshold = 10  # Save every 10 changes

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return {}
        try:
            with open(self.memory_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=2)
            self.unsaved_changes = 0
        except Exception as e:
            print(f"Failed to save memory: {e}")

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
        self.unsaved_changes += 1
        
        # Only save periodically to improve performance
        if self.unsaved_changes >= self.save_threshold:
            self._save_memory()

    def suggest_confidence(self, pair, signal_type):
        """
        Returns symbolic confidence ("high"/"medium"/"low"/"unknown") for a pair/signal.
        """
        records = self.memory.get(pair, [])
        if not records:
            return "unknown"
        relevant = [r for r in records if r["signal"] == signal_type]
        if not relevant:
            return "low"
        wins = [r for r in relevant if r["outcome"] == "win"]
        win_rate = len(wins) / len(relevant)
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
    
    def force_save(self):
        """Force save all unsaved changes immediately."""
        if self.unsaved_changes > 0:
            self._save_memory()
    
    def __del__(self):
        """Ensure data is saved when object is destroyed."""
        try:
            self.force_save()
        except:
            pass  # Avoid exceptions during cleanup
