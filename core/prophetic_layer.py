# core/prophetic_engine.py

from datetime import datetime

class PropheticEngine:
    """
    Symbolic timing and alignment engine.
    - Blends moon phase, astro, emotional/market cycle, and time window logic
    - Ready for dashboard, memory, and trading context overlays
    - Extensible with additional symbolic alignment (numerology, day of week, etc.)
    """
    def __init__(self):
        # All alignments can be expanded/adjusted as needed
        self.alignment_rules = {
            "moon_phase": ["new_moon", "full_moon", "waxing_gibbous", "waning_crescent"],
            "astro_angle": ["trine", "sextile", "conjunction", "opposition"],
            "emotional_rhythm": ["tension", "euphoria", "capitulation"],
            "special_days": ["Monday", "Friday"]  # e.g., certain days for reversals
        }

    def analyze(self, astro_data, context=None, date=None):
        """
        Evaluates symbolic alignment for a specific date/context.
        Returns tags, score, and reasons for dashboard/logging.
        """
        date = date or datetime.utcnow().date().isoformat()
        alignment = []
        reasons = []

        # --- Moon Phase Alignment ---
        moon_phase = astro_data.get("moon_phase")
        if moon_phase in self.alignment_rules["moon_phase"]:
            alignment.append(f"Moon Phase ✅ ({moon_phase})")
            reasons.append(f"Moon phase aligns: {moon_phase}")

        # --- Astro Angle Alignment ---
        astro_angle = astro_data.get("astro_angle")
        if astro_angle in self.alignment_rules["astro_angle"]:
            alignment.append(f"Astro Angle ✅ ({astro_angle})")
            reasons.append(f"Astro angle aligns: {astro_angle}")

        # --- Emotional Rhythm (optional, if provided) ---
        if astro_data.get("emotional_rhythm") in self.alignment_rules["emotional_rhythm"]:
            tag = astro_data["emotional_rhythm"]
            alignment.append(f"Emotional Rhythm ✅ ({tag})")
            reasons.append(f"Emotional rhythm aligns: {tag}")

        # --- Special Day Alignment (optional) ---
        today = context.get("day_of_week") if context and "day_of_week" in context else datetime.utcnow().strftime("%A")
        if today in self.alignment_rules["special_days"]:
            alignment.append(f"Special Day ✅ ({today})")
            reasons.append(f"Day aligns: {today}")

        # --- Score is # of alignment factors matched ---
        prophetic_score = len(alignment)

        return {
            "alignment": alignment,
            "score": prophetic_score,
            "date": date,
            "reasons": reasons,
            "window_open": prophetic_score > 0
        }

    def is_window_open(self, context):
        """
        Returns True if current day/time aligns with special symbolic windows.
        Used for gating signal logic in main.py.
        """
        # You can plug in logic based on moon phase, day, or other cycle here
        now = datetime.utcnow()
        day = context.get("day_of_week") if context and "day_of_week" in context else now.strftime("%A")
        # Example: only open on Monday/Friday or special moon phase (expand as needed)
        return day in self.alignment_rules["special_days"]
