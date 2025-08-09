# core/triangular_analyzer.py

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class TriangularAnalyzer:
    """
    Advanced triangular arbitrage and currency strength analyzer:
    - Triangular relationship validation (e.g., EG * GU = EU)
    - Currency strength calculation
    - Cross-rate opportunities
    - Arbitrage detection
    """
    def __init__(self):
        # Define triangular relationships
        self.triangles = {
            # EUR triangles
            "EUR/GBP/USD": {
                "pairs": ["EURGBP", "GBPUSD", "EURUSD"],
                "formula": "EG * GU = EU",
                "relationships": [
                    {"result": "EURUSD", "calc": ["EURGBP", "GBPUSD", "multiply"]},
                    {"result": "GBPUSD", "calc": ["EURUSD", "EURGBP", "divide"]},
                    {"result": "EURGBP", "calc": ["EURUSD", "GBPUSD", "divide"]}
                ]
            },
            "EUR/JPY/USD": {
                "pairs": ["EURJPY", "USDJPY", "EURUSD"],
                "formula": "EJ = EU * UJ",
                "relationships": [
                    {"result": "EURJPY", "calc": ["EURUSD", "USDJPY", "multiply"]},
                    {"result": "USDJPY", "calc": ["EURJPY", "EURUSD", "divide"]},
                    {"result": "EURUSD", "calc": ["EURJPY", "USDJPY", "divide"]}
                ]
            },
            "EUR/GBP/JPY": {
                "pairs": ["EURGBP", "GBPJPY", "EURJPY"],
                "formula": "EG * GJ = EJ",
                "relationships": [
                    {"result": "EURJPY", "calc": ["EURGBP", "GBPJPY", "multiply"]},
                    {"result": "GBPJPY", "calc": ["EURJPY", "EURGBP", "divide"]},
                    {"result": "EURGBP", "calc": ["EURJPY", "GBPJPY", "divide"]}
                ]
            },

            # GBP triangles
            "GBP/JPY/USD": {
                "pairs": ["GBPJPY", "USDJPY", "GBPUSD"],
                "formula": "GJ = GU * UJ",
                "relationships": [
                    {"result": "GBPJPY", "calc": ["GBPUSD", "USDJPY", "multiply"]},
                    {"result": "USDJPY", "calc": ["GBPJPY", "GBPUSD", "divide"]},
                    {"result": "GBPUSD", "calc": ["GBPJPY", "USDJPY", "divide"]}
                ]
            },
            "GBP/NZD/USD": {
                "pairs": ["GBPNZD", "NZDUSD", "GBPUSD"],
                "formula": "GN * NU = GU",
                "relationships": [
                    {"result": "GBPUSD", "calc": ["GBPNZD", "NZDUSD", "multiply"]},
                    {"result": "NZDUSD", "calc": ["GBPUSD", "GBPNZD", "divide"]},
                    {"result": "GBPNZD", "calc": ["GBPUSD", "NZDUSD", "divide"]}
                ]
            },

            # USD triangles
            "USD/CAD/JPY": {
                "pairs": ["USDCAD", "CADJPY", "USDJPY"],
                "formula": "UC * CJ = UJ",
                "relationships": [
                    {"result": "USDJPY", "calc": ["USDCAD", "CADJPY", "multiply"]},
                    {"result": "CADJPY", "calc": ["USDJPY", "USDCAD", "divide"]},
                    {"result": "USDCAD", "calc": ["USDJPY", "CADJPY", "divide"]}
                ]
            },
            "USD/CHF/JPY": {
                "pairs": ["USDCHF", "CHFJPY", "USDJPY"],
                "formula": "UCH * CHJ = UJ",
                "relationships": [
                    {"result": "USDJPY", "calc": ["USDCHF", "CHFJPY", "multiply"]},
                    {"result": "CHFJPY", "calc": ["USDJPY", "USDCHF", "divide"]},
                    {"result": "USDCHF", "calc": ["USDJPY", "CHFJPY", "divide"]}
                ]
            },

            # AUD triangles
            "AUD/USD/JPY": {
                "pairs": ["AUDUSD", "USDJPY", "AUDJPY"],
                "formula": "AU * UJ = AJ",
                "relationships": [
                    {"result": "AUDJPY", "calc": ["AUDUSD", "USDJPY", "multiply"]},
                    {"result": "USDJPY", "calc": ["AUDJPY", "AUDUSD", "divide"]},
                    {"result": "AUDUSD", "calc": ["AUDJPY", "USDJPY", "divide"]}
                ]
            },

            # NZD triangles
            "NZD/USD/JPY": {
                "pairs": ["NZDUSD", "USDJPY", "NZDJPY"],
                "formula": "NU * UJ = NJ",
                "relationships": [
                    {"result": "NZDJPY", "calc": ["NZDUSD", "USDJPY", "multiply"]},
                    {"result": "USDJPY", "calc": ["NZDJPY", "NZDUSD", "divide"]},
                    {"result": "NZDUSD", "calc": ["NZDJPY", "USDJPY", "divide"]}
                ]
            },

            # EUR/NZD relationships
            "EUR/NZD/USD": {
                "pairs": ["EURNZD", "NZDUSD", "EURUSD"],
                "formula": "EN * NU = EU",
                "relationships": [
                    {"result": "EURUSD", "calc": ["EURNZD", "NZDUSD", "multiply"]},
                    {"result": "NZDUSD", "calc": ["EURUSD", "EURNZD", "divide"]},
                    {"result": "EURNZD", "calc": ["EURUSD", "NZDUSD", "divide"]}
                ]
            }
        }
        
        self.currency_strength = {}
        self.last_update = None

    def analyze(self, prices: Dict[str, float]) -> Dict:
        """
        Analyze triangular relationships and currency strength
        
        Args:
            prices: Dict of current prices for each pair
            
        Returns:
            Dict containing:
            - Currency strength scores
            - Triangular opportunities
            - Arbitrage signals
            - Trading recommendations
        """
        now = datetime.utcnow()
        analysis = {
            "currency_strength": {},
            "triangular_opportunities": [],
            "arbitrage_signals": [],
            "recommendations": [],
            "timestamp": now.isoformat()
        }
        
        # Analyze each triangle
        for triangle_name, triangle in self.triangles.items():
            # Check if we have all required prices
            if not all(pair in prices for pair in triangle["pairs"]):
                continue
                
            # Calculate theoretical cross rates
            opportunities = self._analyze_triangle(
                triangle_name,
                triangle,
                prices
            )
            
            if opportunities:
                analysis["triangular_opportunities"].extend(opportunities)
                
        # Calculate currency strength
        self._update_currency_strength(prices)
        analysis["currency_strength"] = self.currency_strength
        
        # Generate trading recommendations
        analysis["recommendations"] = self._generate_recommendations(
            analysis["triangular_opportunities"]
        )
        
        return analysis
        
    def _analyze_triangle(
        self,
        triangle_name: str,
        triangle: Dict,
        prices: Dict[str, float]
    ) -> List[Dict]:
        """Analyze a specific triangular relationship"""
        opportunities = []
        
        for relationship in triangle["relationships"]:
            result_pair = relationship["result"]
            calc_pairs = relationship["calc"][:2]
            operation = relationship["calc"][2]
            
            # Get actual price
            actual_price = prices[result_pair]
            
            # Calculate theoretical price
            if operation == "multiply":
                theo_price = prices[calc_pairs[0]] * prices[calc_pairs[1]]
            else:  # divide
                theo_price = prices[calc_pairs[0]] / prices[calc_pairs[1]]
                
            # Calculate difference
            diff_pips = (actual_price - theo_price) * 10000
            diff_percent = (actual_price - theo_price) / theo_price * 100
            
            # If significant difference found
            if abs(diff_pips) > 3:  # More than 3 pips difference
                opportunities.append({
                    "triangle": triangle_name,
                    "relationship": relationship["formula"],
                    "actual_price": actual_price,
                    "theoretical_price": theo_price,
                    "difference_pips": diff_pips,
                    "difference_percent": diff_percent,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
        return opportunities
        
    def _update_currency_strength(self, prices: Dict[str, float]):
        """Calculate individual currency strength"""
        strength_scores = {
            "EUR": 0.0,
            "GBP": 0.0,
            "USD": 0.0,
            "JPY": 0.0,
            "AUD": 0.0,
            "NZD": 0.0,
            "CAD": 0.0,
            "CHF": 0.0
        }
        
        # Calculate relative strength
        for pair, price in prices.items():
            base = pair[:3]
            quote = pair[3:]
            
            # Simple strength calculation
            if price > 1:
                strength_scores[base] += 1
                strength_scores[quote] -= 1
            else:
                strength_scores[base] -= 1
                strength_scores[quote] += 1
                
        # Normalize scores
        min_score = min(strength_scores.values())
        max_score = max(strength_scores.values())
        range_score = max_score - min_score
        
        if range_score > 0:
            for currency in strength_scores:
                normalized = (strength_scores[currency] - min_score) / range_score
                self.currency_strength[currency] = {
                    "strength": normalized,
                    "raw_score": strength_scores[currency],
                    "timestamp": datetime.utcnow().isoformat()
                }
                
    def _generate_recommendations(
        self,
        opportunities: List[Dict]
    ) -> List[Dict]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        for opp in opportunities:
            # Strong signals only
            if abs(opp["difference_pips"]) > 5:
                recommendations.append({
                    "pair": opp["relationship"].split("=")[0].strip(),
                    "action": "buy" if opp["difference_pips"] > 0 else "sell",
                    "strength": min(abs(opp["difference_pips"]) / 10, 1.0),
                    "reason": f"Triangular arbitrage opportunity: {opp['difference_pips']:.1f} pips",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
        return recommendations
        
    def get_currency_strength(self, currency: str) -> float:
        """Get normalized strength score for a currency"""
        if currency in self.currency_strength:
            return self.currency_strength[currency]["strength"]
        return 0.0
        
    def get_pair_bias(self, pair: str) -> Dict:
        """Get trading bias for a pair based on currency strength"""
        base = pair[:3]
        quote = pair[3:]
        
        base_strength = self.get_currency_strength(base)
        quote_strength = self.get_currency_strength(quote)
        
        diff = base_strength - quote_strength
        
        return {
            "bias": "buy" if diff > 0 else "sell",
            "strength": abs(diff),
            "base_currency": {
                "currency": base,
                "strength": base_strength
            },
            "quote_currency": {
                "currency": quote,
                "strength": quote_strength
            },
            "timestamp": datetime.utcnow().isoformat()
        }
