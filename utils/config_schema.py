from typing import Any, Dict


CONFIG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "execution": {
            "type": "object",
            "properties": {
                "driver": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "poll_ms": {"type": "integer", "minimum": 100}
            },
            "required": ["driver", "symbols"]
        },
        "mode": {
            "type": "object",
            "properties": {
                "autonomous": {"type": "boolean"},
                "require_all_confirm": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
                "simulation": {"type": "boolean"}
            },
            "required": ["autonomous"]
        },
        "hybrid": {
            "type": "object",
            "properties": {
                "entry_threshold_base": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["entry_threshold_base"]
        },
        "filters": {
            "type": "object",
            "properties": {
                "max_spread_pips": {"type": "number", "minimum": 0},
                "max_slippage_pips": {"type": "number", "minimum": 0}
            },
            "required": ["max_spread_pips", "max_slippage_pips"]
        },
        "risk": {
            "type": "object",
            "properties": {
                "daily_loss_cap": {"type": "number", "minimum": 0.0, "maximum": 0.2},
                "weekly_dd_brake": {"type": "number", "minimum": 0.0, "maximum": 0.5}
            },
            "required": ["daily_loss_cap", "weekly_dd_brake"]
        },
        "policy": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["shadow", "active"]},
                "challenger_pct": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "enable_feature_logging": {"type": "boolean"}
            }
        }
    },
    "required": ["execution", "mode", "hybrid", "filters", "risk"],
    "additionalProperties": True
}

def validate_config(cfg: Dict[str, Any]) -> None:
    # Lightweight validation without external deps: basic presence and types
    def _ensure(path: str, cond: bool):
        if not cond:
            raise ValueError(f"Invalid config: {path}")

    ex = cfg.get("execution", {})
    _ensure("execution.driver", isinstance(ex.get("driver"), str) and len(ex.get("driver")) > 0)
    _ensure("execution.symbols", isinstance(ex.get("symbols"), list) and len(ex.get("symbols")) > 0)

    mode = cfg.get("mode", {})
    _ensure("mode.autonomous", isinstance(mode.get("autonomous"), bool))

    hyb = cfg.get("hybrid", {})
    _ensure("hybrid.entry_threshold_base", isinstance(hyb.get("entry_threshold_base"), (int, float)))

    fil = cfg.get("filters", {})
    _ensure("filters.max_spread_pips", isinstance(fil.get("max_spread_pips"), (int, float)))
    _ensure("filters.max_slippage_pips", isinstance(fil.get("max_slippage_pips"), (int, float)))

    risk = cfg.get("risk", {})
    _ensure("risk.daily_loss_cap", isinstance(risk.get("daily_loss_cap"), (int, float)))
    _ensure("risk.weekly_dd_brake", isinstance(risk.get("weekly_dd_brake"), (int, float)))


