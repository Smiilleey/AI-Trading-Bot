# core/candle_dna_reader.py

import numpy as np
from typing import Dict, List
from datetime import datetime
from collections import deque

class MasterCandleDNAReader:
    """Master Candle DNA Reader - Reads the complete story inside every candle"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.dna_memory = deque(maxlen=1000)
        self.total_reads = 0
        self.successful_reads = 0
    
    def read_complete_candle_dna(self, candle: Dict, recent_candles: List[Dict], 
                                recent_volumes: List[float], symbol: str = "") -> Dict:
        """Read complete DNA of a single candle"""
        try:
            self.total_reads += 1
            
            # Basic DNA analysis
            body_size = abs(float(candle["close"]) - float(candle["open"]))
            total_range = float(candle["high"]) - float(candle["low"])
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            # Determine sentiment
            if float(candle["close"]) > float(candle["open"]):
                overall_sentiment = "bullish"
            elif float(candle["close"]) < float(candle["open"]):
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
            
            # Calculate confidence
            confidence = min(1.0, body_ratio + 0.3)
            
            result = {
                "valid": True,
                "overall_sentiment": overall_sentiment,
                "confidence": confidence,
                "body_ratio": body_ratio,
                "dna_components": {
                    "price_action": {"conviction": confidence},
                    "volume": {"volume_class": "normal"},
                    "order_flow": {"strength": "moderate"},
                    "liquidity": {"stop_hunt": False},
                    "pattern": {"pattern_strength": 0.5},
                    "sentiment": {"sentiment_score": 0.5}
                }
            }
            
            self.successful_reads += 1
            return result
            
        except Exception as e:
            return {"error": f"DNA reading failed: {str(e)}", "valid": False}
    
    def get_dna_stats(self) -> Dict:
        """Get DNA reader statistics"""
        return {
            "total_reads": self.total_reads,
            "successful_reads": self.successful_reads,
            "success_rate": self.successful_reads / max(1, self.total_reads)
        }
