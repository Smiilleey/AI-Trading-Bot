import json
import os

MEMORY_FILE = "memory/pattern_memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def log_pattern(pair, signal, outcome, rr, confidence, tags=None):
    memory = load_memory()
    if pair not in memory:
        memory[pair] = []

    memory[pair].append({
        "signal": signal,
        "outcome": outcome,
        "rr": rr,
        "confidence": confidence,
        "tags": tags or []
    })

    save_memory(memory)

def get_pattern_stats(pair, signal_type=None):
    memory = load_memory()
    patterns = memory.get(pair, [])
    if signal_type:
        patterns = [p for p in patterns if p["signal"] == signal_type]

    total = len(patterns)
    if total == 0:
        return {"win_rate": 0, "average_rr": 0}

    wins = sum(1 for p in patterns if p["outcome"] == "win")
    avg_rr = sum(p["rr"] for p in patterns) / total

    return {
        "total": total,
        "wins": wins,
        "win_rate": wins / total,
        "average_rr": avg_rr
    }


# memory/learning.py

import json
from datetime import datetime

class LearningEngine:
    def __init__(self, memory_file="memory/pattern_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()

    def _load_memory(self):
        try:
            with open(self.memory_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def suggest_confidence(self, pair, signal_type):
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

    def record_result(self, pair, context, signal, outcome, rr, entry_time):
        record = {
            "pair": pair,
            "context": context,
            "signal": signal,
            "outcome": outcome,
            "rr": rr,
            "entry_time": entry_time,
            "timestamp": datetime.utcnow().isoformat()
        }

        if pair not in self.memory:
            self.memory[pair] = []
        self.memory[pair].append(record)

        self._save_memory()

    def _save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
