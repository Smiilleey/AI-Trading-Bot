import json, os, time

DASH_PATH = os.path.join("memory", "dashboard_state.json")

def update_dashboard(state: dict):
    state = dict(state)
    state["ts"] = time.time()
    try:
        os.makedirs(os.path.dirname(DASH_PATH), exist_ok=True)
        with open(DASH_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass
