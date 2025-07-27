from datetime import datetime

class PropheticLayer:
    def __init__(self):
        self.alignment_rules = {
            "moon_phase": ["new_moon", "full_moon"],
            "astro_angle": ["trine", "sextile"]
        }

    def analyze(self, astro_data, date=None):
        date = date or datetime.utcnow().date().isoformat()
        alignment = []

        if astro_data.get("moon_phase") in self.alignment_rules["moon_phase"]:
            alignment.append(f"Moon Phase ✅ ({astro_data['moon_phase']})")

        if astro_data.get("astro_angle") in self.alignment_rules["astro_angle"]:
            alignment.append(f"Astro Angle ✅ ({astro_data['astro_angle']})")

        prophetic_score = len(alignment)
        return {"alignment": alignment, "score": prophetic_score}
