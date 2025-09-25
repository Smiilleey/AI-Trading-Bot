from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

# Enhanced news filter with real calendar integration and post-news playbooks
class NewsCalendar:
    """Real-time news calendar with post-event playbook support."""
    
    def __init__(self, calendar_file: str = None):
        self.calendar_file = calendar_file or "data/economic_calendar.json"
        self.events = []
        self.post_news_states = {}  # Track post-event states
        self.load_calendar()
    
    def load_calendar(self):
        """Load economic calendar from file or create default."""
        try:
            if os.path.exists(self.calendar_file):
                with open(self.calendar_file, 'r') as f:
                    self.events = json.load(f)
            else:
                # Default high-impact events
                self.events = [
                    {
                        "event": "NFP",
                        "symbols": ["USD", "EURUSD", "GBPUSD", "USDJPY"],
                        "day_of_week": "friday",
                        "time_utc": "13:30",
                        "impact": "high",
                        "duration_minutes": 120,
                        "playbook": "fade_initial_move"
                    },
                    {
                        "event": "FOMC",
                        "symbols": ["USD", "EURUSD", "GBPUSD", "USDJPY"],
                        "day_of_week": "variable",
                        "time_utc": "19:00",
                        "impact": "high",
                        "duration_minutes": 180,
                        "playbook": "continuation_after_volatility"
                    },
                    {
                        "event": "CPI",
                        "symbols": ["USD", "EURUSD", "GBPUSD", "USDJPY"],
                        "day_of_week": "variable",
                        "time_utc": "13:30",
                        "impact": "high",
                        "duration_minutes": 90,
                        "playbook": "fade_initial_move"
                    }
                ]
        except Exception as e:
            print(f"Error loading calendar: {e}")
            self.events = []
    
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict]:
        """Get events in the next N hours."""
        now = datetime.utcnow()
        upcoming = []
        
        for event in self.events:
            # This is simplified - in production you'd parse actual dates
            if event.get("impact") == "high":
                upcoming.append(event)
        
        return upcoming
    
    def is_news_window(self, symbol: str, now: datetime) -> Tuple[bool, Optional[Dict]]:
        """Check if symbol is in news window."""
        for event in self.events:
            if symbol in event.get("symbols", []):
                # Simplified time check - in production you'd parse actual times
                if event.get("impact") == "high":
                    return True, event
        return False, None
    
    def get_post_news_state(self, symbol: str, event: str) -> Dict:
        """Get post-news state for playbook execution."""
        key = f"{symbol}_{event}"
        return self.post_news_states.get(key, {
            "phase": "pre_news",
            "time_since_event": 0,
            "volatility_regime": "normal",
            "playbook_active": False
        })
    
    def update_post_news_state(self, symbol: str, event: str, state: Dict):
        """Update post-news state."""
        key = f"{symbol}_{event}"
        self.post_news_states[key] = state

# Global calendar instance
news_calendar = NewsCalendar()

def in_news_window(symbol: str, now: datetime) -> bool:
    """Check if symbol is in news window."""
    in_window, event = news_calendar.is_news_window(symbol, now)
    return in_window

def get_news_context(symbol: str, now: datetime) -> Dict:
    """Get comprehensive news context including post-event state."""
    in_window, event = news_calendar.is_news_window(symbol, now)
    
    context = {
        "in_news_window": in_window,
        "event": event,
        "post_news_state": None,
        "playbook_recommendation": None
    }
    
    if event:
        post_state = news_calendar.get_post_news_state(symbol, event["event"])
        context["post_news_state"] = post_state
        
        # Determine playbook recommendation
        if post_state["phase"] == "pre_news":
            context["playbook_recommendation"] = "wait_for_event"
        elif post_state["phase"] == "immediate_post":
            if event.get("playbook") == "fade_initial_move":
                context["playbook_recommendation"] = "fade_initial_move"
            elif event.get("playbook") == "continuation_after_volatility":
                context["playbook_recommendation"] = "wait_for_volatility_subsidence"
        elif post_state["phase"] == "post_volatility":
            context["playbook_recommendation"] = "resume_normal_trading"
    
    return context

def update_post_news_state(symbol: str, event: str, phase: str, 
                          volatility_regime: str = "normal", 
                          time_since_event: int = 0):
    """Update post-news state for playbook tracking."""
    state = {
        "phase": phase,  # pre_news, immediate_post, post_volatility
        "time_since_event": time_since_event,
        "volatility_regime": volatility_regime,
        "playbook_active": phase in ["immediate_post", "post_volatility"]
    }
    news_calendar.update_post_news_state(symbol, event, state)

def get_playbook_signals(symbol: str, market_data: Dict, news_context: Dict) -> Dict:
    """Generate playbook-specific trading signals."""
    if not news_context.get("playbook_recommendation"):
        return {"signal": "normal", "confidence": 0.0}
    
    playbook = news_context["playbook_recommendation"]
    post_state = news_context.get("post_news_state", {})
    
    signals = {
        "signal": "normal",
        "confidence": 0.0,
        "playbook": playbook,
        "entry_gate": False,
        "risk_adjustment": 1.0
    }
    
    if playbook == "fade_initial_move":
        # Look for initial move to fade
        if post_state.get("phase") == "immediate_post":
            signals.update({
                "signal": "fade_initial",
                "confidence": 0.7,
                "entry_gate": True,
                "risk_adjustment": 0.5  # Reduce risk for news trades
            })
    
    elif playbook == "wait_for_volatility_subsidence":
        # Wait for volatility to subside before trading
        if post_state.get("volatility_regime") == "high":
            signals.update({
                "signal": "wait",
                "confidence": 0.8,
                "entry_gate": False
            })
        else:
            signals.update({
                "signal": "resume",
                "confidence": 0.6,
                "entry_gate": True
            })
    
    elif playbook == "resume_normal_trading":
        signals.update({
            "signal": "normal",
            "confidence": 0.5,
            "entry_gate": True
        })
    
    return signals
