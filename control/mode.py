import json, os

class Mode:
    """
    Dual-mode control:
    - autonomous=True: Bot executes trades automatically
    - autonomous=False: Bot enqueues trade proposals for manual approval
    """
    def __init__(self, cfg_path="config.json"):
        self.cfg_path = cfg_path
        self._cfg = {}
        self.reload()

    def reload(self):
        try:
            with open(self.cfg_path, "r") as f:
                self._cfg = json.load(f)
        except Exception:
            self._cfg = {}
        self._mode = self._cfg.get("mode", {})
        self._risk = self._cfg.get("risk", {})
        self._hyb  = self._cfg.get("hybrid", {})
        self._flt  = self._cfg.get("filters", {})

    @property
    def autonomous(self) -> bool:
        return bool(self._mode.get("autonomous", True))

    @property
    def require_all_confirm(self) -> bool:
        return bool(self._mode.get("require_all_confirm", True))

    @property
    def risk(self): return dict(self._risk)
    @property
    def hybrid(self): return dict(self._hyb)
    @property
    def filters(self): return dict(self._flt)
