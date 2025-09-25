# core/pd_array_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class PDArrayEngine:
    """
    PD ARRAY ENGINE - Complete Institutional PD Array Model
    
    Implements the complete PD array model:
    - Equal highs/lows detection with tolerance
    - External vs internal range logic
    - Liquidity pools identification
    - Fair Value Gaps (FVG) with sync detection
    - Breaker blocks (BPR) - broken structure becomes new S/R
    - Mitigation blocks - institutional order management
    - Order blocks (OB) with enhanced institutional context
    
    This engine identifies all the key institutional levels that smart money uses.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Detection parameters
        self.detection_params = {
            'equal_level_tolerance': 0.0001,  # 1 pip tolerance for equal levels
            'fvg_min_size': 0.00005,          # Minimum FVG size
            'ob_min_size': 0.0001,            # Minimum order block size
            'liquidity_cluster_distance': 0.0002,  # Max distance for liquidity clustering
            'breaker_confirmation_candles': 3,      # Candles needed to confirm breaker
            'mitigation_test_tolerance': 0.00005   # Tolerance for mitigation tests
        }
        
        # PD array memory
        self.equal_highs = defaultdict(list)
        self.equal_lows = defaultdict(list)
        self.fair_value_gaps = defaultdict(list)
        self.order_blocks = defaultdict(list)
        self.breaker_blocks = defaultdict(list)
        self.mitigation_blocks = defaultdict(list)
        self.liquidity_pools = defaultdict(list)
        
        # Performance tracking
        self.total_detections = 0
        self.successful_trades_from_arrays = 0
        self.array_performance = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        # Range classification
        self.range_types = {
            'external': 'price_outside_previous_range',
            'internal': 'price_inside_previous_range',
            'expansion': 'range_expanding',
            'contraction': 'range_contracting'
        }
    
    def detect_all_pd_arrays(self, 
                            symbol: str, 
                            candles: List[Dict], 
                            timeframe: str = "M15") -> Dict[str, Any]:
        """
        Comprehensive PD array detection for all institutional levels.
        
        Args:
            symbol: Trading symbol
            candles: Historical candle data
            timeframe: Analysis timeframe
            
        Returns:
            Complete PD array analysis with all detected levels
        """
        try:
            if not candles or len(candles) < 20:
                return self._create_array_response(False, "Insufficient candle data")
            
            self.total_detections += 1
            
            # 1. Detect equal highs and lows
            equal_levels = self._detect_equal_highs_lows(candles)
            
            # 2. Identify Fair Value Gaps
            fvg_analysis = self._detect_fair_value_gaps(candles)
            
            # 3. Detect order blocks
            order_blocks = self._detect_order_blocks(candles)
            
            # 4. Identify breaker blocks
            breaker_blocks = self._detect_breaker_blocks(candles)
            
            # 5. Find mitigation blocks
            mitigation_blocks = self._detect_mitigation_blocks(candles)
            
            # 6. Identify liquidity pools
            liquidity_pools = self._identify_liquidity_pools(candles, equal_levels)
            
            # 7. Classify range type (external vs internal)
            range_classification = self._classify_range_type(candles)
            
            # 8. Calculate array confluence
            confluence_analysis = self._calculate_array_confluence(
                equal_levels, fvg_analysis, order_blocks, breaker_blocks
            )
            
            # 9. Create comprehensive response
            response = self._create_array_response(
                True,
                equal_levels=equal_levels,
                fvg_analysis=fvg_analysis,
                order_blocks=order_blocks,
                breaker_blocks=breaker_blocks,
                mitigation_blocks=mitigation_blocks,
                liquidity_pools=liquidity_pools,
                range_classification=range_classification,
                confluence_analysis=confluence_analysis,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Update memory
            self._update_array_memory(symbol, response)
            
            return response
            
        except Exception as e:
            return self._create_array_response(False, f"PD array detection failed: {str(e)}")
    
    def _detect_equal_highs_lows(self, candles: List[Dict]) -> Dict[str, Any]:
        """Detect equal highs and equal lows with institutional tolerance."""
        try:
            equal_highs = []
            equal_lows = []
            tolerance = self.detection_params['equal_level_tolerance']
            
            # Find all swing highs and lows first
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(candles) - 2):
                current = candles[i]
                
                # Check for swing high
                if (current['high'] > candles[i-1]['high'] and 
                    current['high'] > candles[i-2]['high'] and
                    current['high'] > candles[i+1]['high'] and 
                    current['high'] > candles[i+2]['high']):
                    swing_highs.append({
                        'price': float(current['high']),
                        'index': i,
                        'time': current.get('time', 'unknown'),
                        'strength': self._calculate_swing_strength(candles, i, 'high')
                    })
                
                # Check for swing low
                if (current['low'] < candles[i-1]['low'] and 
                    current['low'] < candles[i-2]['low'] and
                    current['low'] < candles[i+1]['low'] and 
                    current['low'] < candles[i+2]['low']):
                    swing_lows.append({
                        'price': float(current['low']),
                        'index': i,
                        'time': current.get('time', 'unknown'),
                        'strength': self._calculate_swing_strength(candles, i, 'low')
                    })
            
            # Find equal highs
            for i in range(len(swing_highs)):
                for j in range(i + 1, len(swing_highs)):
                    price1, price2 = swing_highs[i]['price'], swing_highs[j]['price']
                    
                    if abs(price1 - price2) <= tolerance:
                        equal_highs.append({
                            'level': (price1 + price2) / 2,
                            'touches': 2,
                            'first_touch': swing_highs[i],
                            'second_touch': swing_highs[j],
                            'strength': (swing_highs[i]['strength'] + swing_highs[j]['strength']) / 2,
                            'type': 'equal_highs',
                            'last_respect': swing_highs[j]['time']
                        })
            
            # Find equal lows
            for i in range(len(swing_lows)):
                for j in range(i + 1, len(swing_lows)):
                    price1, price2 = swing_lows[i]['price'], swing_lows[j]['price']
                    
                    if abs(price1 - price2) <= tolerance:
                        equal_lows.append({
                            'level': (price1 + price2) / 2,
                            'touches': 2,
                            'first_touch': swing_lows[i],
                            'second_touch': swing_lows[j],
                            'strength': (swing_lows[i]['strength'] + swing_lows[j]['strength']) / 2,
                            'type': 'equal_lows',
                            'last_respect': swing_lows[j]['time']
                        })
            
            return {
                'equal_highs': equal_highs,
                'equal_lows': equal_lows,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'total_equal_levels': len(equal_highs) + len(equal_lows)
            }
            
        except Exception as e:
            return {'equal_highs': [], 'equal_lows': [], 'error': str(e)}
    
    def _calculate_swing_strength(self, candles: List[Dict], index: int, swing_type: str) -> float:
        """Calculate the strength of a swing high or low."""
        try:
            if swing_type == 'high':
                swing_price = float(candles[index]['high'])
                # Check how much higher than surrounding candles
                surrounding = [float(candles[i]['high']) for i in range(max(0, index-3), min(len(candles), index+4)) if i != index]
            else:
                swing_price = float(candles[index]['low'])
                # Check how much lower than surrounding candles
                surrounding = [float(candles[i]['low']) for i in range(max(0, index-3), min(len(candles), index+4)) if i != index]
            
            if not surrounding:
                return 0.5
            
            if swing_type == 'high':
                avg_surrounding = np.mean(surrounding)
                strength = (swing_price - avg_surrounding) / avg_surrounding if avg_surrounding > 0 else 0.5
            else:
                avg_surrounding = np.mean(surrounding)
                strength = (avg_surrounding - swing_price) / avg_surrounding if avg_surrounding > 0 else 0.5
            
            return min(1.0, max(0.0, strength * 10))  # Scale to 0-1
            
        except Exception:
            return 0.5
    
    def _detect_fair_value_gaps(self, candles: List[Dict]) -> Dict[str, Any]:
        """Enhanced Fair Value Gap detection with institutional context."""
        try:
            fvgs = []
            min_size = self.detection_params['fvg_min_size']
            
            for i in range(1, len(candles) - 1):
                prev_candle = candles[i-1]
                current_candle = candles[i]
                next_candle = candles[i+1]
                
                # Bullish FVG: current low > previous high
                if float(current_candle['low']) > float(prev_candle['high']):
                    gap_size = float(current_candle['low']) - float(prev_candle['high'])
                    
                    if gap_size >= min_size:
                        fvg = {
                            'type': 'bullish',
                            'top': float(current_candle['low']),
                            'bottom': float(prev_candle['high']),
                            'size': gap_size,
                            'midpoint': (float(current_candle['low']) + float(prev_candle['high'])) / 2,
                            'formation_index': i,
                            'formation_time': current_candle.get('time', 'unknown'),
                            'status': 'unfilled',
                            'strength': self._calculate_fvg_strength(candles, i, 'bullish'),
                            'volume_context': self._analyze_fvg_volume(candles, i)
                        }
                        
                        # Check if FVG has been filled
                        fvg['filled'] = self._check_fvg_filled(candles, fvg, i)
                        fvgs.append(fvg)
                
                # Bearish FVG: current high < previous low
                elif float(current_candle['high']) < float(prev_candle['low']):
                    gap_size = float(prev_candle['low']) - float(current_candle['high'])
                    
                    if gap_size >= min_size:
                        fvg = {
                            'type': 'bearish',
                            'top': float(prev_candle['low']),
                            'bottom': float(current_candle['high']),
                            'size': gap_size,
                            'midpoint': (float(prev_candle['low']) + float(current_candle['high'])) / 2,
                            'formation_index': i,
                            'formation_time': current_candle.get('time', 'unknown'),
                            'status': 'unfilled',
                            'strength': self._calculate_fvg_strength(candles, i, 'bearish'),
                            'volume_context': self._analyze_fvg_volume(candles, i)
                        }
                        
                        # Check if FVG has been filled
                        fvg['filled'] = self._check_fvg_filled(candles, fvg, i)
                        fvgs.append(fvg)
            
            # Analyze FVG clusters and confluence
            fvg_clusters = self._analyze_fvg_clusters(fvgs)
            
            return {
                'fvgs': fvgs,
                'bullish_fvgs': [fvg for fvg in fvgs if fvg['type'] == 'bullish'],
                'bearish_fvgs': [fvg for fvg in fvgs if fvg['type'] == 'bearish'],
                'unfilled_fvgs': [fvg for fvg in fvgs if not fvg.get('filled', False)],
                'fvg_clusters': fvg_clusters,
                'total_fvgs': len(fvgs)
            }
            
        except Exception as e:
            return {'fvgs': [], 'error': str(e)}
    
    def _calculate_fvg_strength(self, candles: List[Dict], index: int, fvg_type: str) -> float:
        """Calculate FVG strength based on context."""
        try:
            # Volume context
            volumes = [float(c.get('tick_volume', 1000)) for c in candles[max(0, index-5):index+5]]
            avg_volume = np.mean(volumes) if volumes else 1000
            current_volume = float(candles[index].get('tick_volume', 1000))
            volume_strength = min(2.0, current_volume / avg_volume) / 2.0
            
            # Price movement context
            prev_candle = candles[index-1]
            current_candle = candles[index]
            
            if fvg_type == 'bullish':
                price_movement = (float(current_candle['close']) - float(prev_candle['close'])) / float(prev_candle['close'])
            else:
                price_movement = (float(prev_candle['close']) - float(current_candle['close'])) / float(prev_candle['close'])
            
            movement_strength = min(1.0, abs(price_movement) * 1000)  # Scale for forex
            
            # Combine factors
            strength = (volume_strength * 0.6) + (movement_strength * 0.4)
            return min(1.0, max(0.0, strength))
            
        except Exception:
            return 0.5
    
    def _analyze_fvg_volume(self, candles: List[Dict], index: int) -> Dict[str, Any]:
        """Analyze volume context around FVG formation."""
        try:
            if index < 2 or index >= len(candles) - 2:
                return {'analysis': 'insufficient_data'}
            
            # Get volume data around FVG
            volumes = []
            for i in range(max(0, index-2), min(len(candles), index+3)):
                volumes.append(float(candles[i].get('tick_volume', 1000)))
            
            avg_volume = np.mean(volumes)
            formation_volume = volumes[2] if len(volumes) >= 3 else avg_volume  # Index 2 is the formation candle
            
            volume_ratio = formation_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > 1.5:
                volume_context = 'high_volume'
            elif volume_ratio < 0.7:
                volume_context = 'low_volume'
            else:
                volume_context = 'normal_volume'
            
            return {
                'volume_ratio': volume_ratio,
                'context': volume_context,
                'formation_volume': formation_volume,
                'average_volume': avg_volume
            }
            
        except Exception:
            return {'analysis': 'error'}
    
    def _check_fvg_filled(self, candles: List[Dict], fvg: Dict, formation_index: int) -> bool:
        """Check if FVG has been filled by subsequent price action."""
        try:
            fvg_top = fvg['top']
            fvg_bottom = fvg['bottom']
            
            # Check subsequent candles
            for i in range(formation_index + 1, len(candles)):
                candle = candles[i]
                high, low = float(candle['high']), float(candle['low'])
                
                # For bullish FVG, check if price went back down into the gap
                if fvg['type'] == 'bullish':
                    if low <= fvg_bottom:
                        return True
                # For bearish FVG, check if price went back up into the gap
                else:
                    if high >= fvg_top:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _analyze_fvg_clusters(self, fvgs: List[Dict]) -> List[Dict]:
        """Analyze FVG clusters for confluence zones."""
        if not fvgs:
            return []
        
        clusters = []
        cluster_distance = self.detection_params['liquidity_cluster_distance']
        
        # Group FVGs by proximity
        for i, fvg1 in enumerate(fvgs):
            cluster = [fvg1]
            
            for j, fvg2 in enumerate(fvgs):
                if i != j:
                    # Check if FVGs are close to each other
                    distance = abs(fvg1['midpoint'] - fvg2['midpoint'])
                    if distance <= cluster_distance:
                        cluster.append(fvg2)
            
            if len(cluster) > 1:
                clusters.append({
                    'fvgs': cluster,
                    'center': np.mean([fvg['midpoint'] for fvg in cluster]),
                    'strength': np.mean([fvg['strength'] for fvg in cluster]),
                    'count': len(cluster)
                })
        
        return clusters
    
    def _detect_order_blocks(self, candles: List[Dict]) -> List[Dict]:
        """Enhanced order block detection with institutional context."""
        try:
            order_blocks = []
            min_size = self.detection_params['ob_min_size']
            
            for i in range(1, len(candles) - 3):
                current = candles[i]
                next_candles = candles[i+1:i+4]  # Next 3 candles
                
                # Check for bullish order block
                # Large green candle followed by continuation higher
                if (float(current['close']) > float(current['open']) and  # Green candle
                    float(current['high']) - float(current['low']) >= min_size):  # Minimum size
                    
                    # Check if next candles continue higher
                    continuation = all(float(c['close']) > float(current['high']) * 0.999 for c in next_candles[:2])
                    
                    if continuation:
                        ob = {
                            'type': 'bullish',
                            'top': float(current['high']),
                            'bottom': float(current['low']),
                            'body_top': max(float(current['open']), float(current['close'])),
                            'body_bottom': min(float(current['open']), float(current['close'])),
                            'formation_index': i,
                            'formation_time': current.get('time', 'unknown'),
                            'strength': self._calculate_ob_strength(candles, i, 'bullish'),
                            'tested': False,
                            'broken': False,
                            'volume': float(current.get('tick_volume', 1000))
                        }
                        
                        # Check if OB has been tested or broken
                        ob.update(self._check_ob_status(candles, ob, i))
                        order_blocks.append(ob)
                
                # Check for bearish order block
                elif (float(current['close']) < float(current['open']) and  # Red candle
                      float(current['high']) - float(current['low']) >= min_size):  # Minimum size
                    
                    # Check if next candles continue lower
                    continuation = all(float(c['close']) < float(current['low']) * 1.001 for c in next_candles[:2])
                    
                    if continuation:
                        ob = {
                            'type': 'bearish',
                            'top': float(current['high']),
                            'bottom': float(current['low']),
                            'body_top': max(float(current['open']), float(current['close'])),
                            'body_bottom': min(float(current['open']), float(current['close'])),
                            'formation_index': i,
                            'formation_time': current.get('time', 'unknown'),
                            'strength': self._calculate_ob_strength(candles, i, 'bearish'),
                            'tested': False,
                            'broken': False,
                            'volume': float(current.get('tick_volume', 1000))
                        }
                        
                        # Check if OB has been tested or broken
                        ob.update(self._check_ob_status(candles, ob, i))
                        order_blocks.append(ob)
            
            return order_blocks
            
        except Exception as e:
            return []
    
    def _calculate_ob_strength(self, candles: List[Dict], index: int, ob_type: str) -> float:
        """Calculate order block strength."""
        try:
            current = candles[index]
            
            # Volume strength
            volumes = [float(c.get('tick_volume', 1000)) for c in candles[max(0, index-5):index+5]]
            avg_volume = np.mean(volumes) if volumes else 1000
            volume_strength = min(2.0, float(current.get('tick_volume', 1000)) / avg_volume) / 2.0
            
            # Size strength
            body_size = abs(float(current['close']) - float(current['open']))
            total_size = float(current['high']) - float(current['low'])
            body_ratio = body_size / total_size if total_size > 0 else 0.5
            
            # Momentum strength
            if index > 0:
                prev = candles[index-1]
                momentum = abs(float(current['close']) - float(prev['close'])) / float(prev['close'])
                momentum_strength = min(1.0, momentum * 1000)  # Scale for forex
            else:
                momentum_strength = 0.5
            
            # Combine factors
            strength = (volume_strength * 0.4) + (body_ratio * 0.3) + (momentum_strength * 0.3)
            return min(1.0, max(0.0, strength))
            
        except Exception:
            return 0.5
    
    def _check_ob_status(self, candles: List[Dict], ob: Dict, formation_index: int) -> Dict[str, Any]:
        """Check if order block has been tested or broken."""
        try:
            tested = False
            broken = False
            test_count = 0
            
            # Check subsequent candles
            for i in range(formation_index + 1, len(candles)):
                candle = candles[i]
                high, low = float(candle['high']), float(candle['low'])
                
                if ob['type'] == 'bullish':
                    # Test: price comes back to OB level
                    if low <= ob['top'] and high >= ob['bottom']:
                        tested = True
                        test_count += 1
                    
                    # Broken: price closes below OB
                    if float(candle['close']) < ob['bottom']:
                        broken = True
                        break
                        
                else:  # bearish OB
                    # Test: price comes back to OB level
                    if high >= ob['bottom'] and low <= ob['top']:
                        tested = True
                        test_count += 1
                    
                    # Broken: price closes above OB
                    if float(candle['close']) > ob['top']:
                        broken = True
                        break
            
            return {
                'tested': tested,
                'broken': broken,
                'test_count': test_count,
                'status': 'broken' if broken else ('tested' if tested else 'untested')
            }
            
        except Exception:
            return {'tested': False, 'broken': False, 'test_count': 0, 'status': 'unknown'}
    
    def _detect_breaker_blocks(self, candles: List[Dict]) -> List[Dict]:
        """Detect breaker blocks - broken structure that becomes new support/resistance."""
        try:
            breaker_blocks = []
            confirmation_candles = self.detection_params['breaker_confirmation_candles']
            
            # First, find significant structure breaks
            for i in range(10, len(candles) - confirmation_candles):
                # Look for significant support/resistance levels
                lookback_candles = candles[i-10:i]
                resistance_level = max(float(c['high']) for c in lookback_candles)
                support_level = min(float(c['low']) for c in lookback_candles)
                
                current = candles[i]
                confirmation_candles_data = candles[i+1:i+1+confirmation_candles]
                
                # Check for resistance break (now becomes support)
                if (float(current['close']) > resistance_level and
                    all(float(c['low']) > resistance_level * 0.999 for c in confirmation_candles_data)):
                    
                    breaker = {
                        'type': 'bullish_breaker',
                        'level': resistance_level,
                        'break_candle_index': i,
                        'break_time': current.get('time', 'unknown'),
                        'strength': self._calculate_breaker_strength(lookback_candles, resistance_level, 'resistance'),
                        'confirmed': True,
                        'status': 'active',
                        'test_count': 0
                    }
                    
                    # Check if breaker has been tested
                    breaker.update(self._check_breaker_tests(candles, breaker, i))
                    breaker_blocks.append(breaker)
                
                # Check for support break (now becomes resistance)
                elif (float(current['close']) < support_level and
                      all(float(c['high']) < support_level * 1.001 for c in confirmation_candles_data)):
                    
                    breaker = {
                        'type': 'bearish_breaker',
                        'level': support_level,
                        'break_candle_index': i,
                        'break_time': current.get('time', 'unknown'),
                        'strength': self._calculate_breaker_strength(lookback_candles, support_level, 'support'),
                        'confirmed': True,
                        'status': 'active',
                        'test_count': 0
                    }
                    
                    # Check if breaker has been tested
                    breaker.update(self._check_breaker_tests(candles, breaker, i))
                    breaker_blocks.append(breaker)
            
            return breaker_blocks
            
        except Exception as e:
            return []
    
    def _calculate_breaker_strength(self, candles: List[Dict], level: float, level_type: str) -> float:
        """Calculate breaker block strength based on how well level was respected."""
        try:
            touches = 0
            total_candles = len(candles)
            tolerance = self.detection_params['equal_level_tolerance']
            
            for candle in candles:
                high, low = float(candle['high']), float(candle['low'])
                
                if level_type == 'resistance':
                    if abs(high - level) <= tolerance:
                        touches += 1
                else:  # support
                    if abs(low - level) <= tolerance:
                        touches += 1
            
            # More touches = stronger level
            strength = min(1.0, touches / (total_candles * 0.3))
            return strength
            
        except Exception:
            return 0.5
    
    def _check_breaker_tests(self, candles: List[Dict], breaker: Dict, break_index: int) -> Dict[str, Any]:
        """Check if breaker has been tested as new support/resistance."""
        try:
            test_count = 0
            still_active = True
            tolerance = self.detection_params['mitigation_test_tolerance']
            
            # Check subsequent candles
            for i in range(break_index + 1, len(candles)):
                candle = candles[i]
                high, low = float(candle['high']), float(candle['low'])
                close = float(candle['close'])
                
                if breaker['type'] == 'bullish_breaker':
                    # Test as support
                    if abs(low - breaker['level']) <= tolerance:
                        test_count += 1
                    
                    # Check if breaker failed (closed below)
                    if close < breaker['level']:
                        still_active = False
                        break
                        
                else:  # bearish_breaker
                    # Test as resistance
                    if abs(high - breaker['level']) <= tolerance:
                        test_count += 1
                    
                    # Check if breaker failed (closed above)
                    if close > breaker['level']:
                        still_active = False
                        break
            
            return {
                'test_count': test_count,
                'status': 'active' if still_active else 'failed',
                'strength_updated': min(1.0, breaker.get('strength', 0.5) + (test_count * 0.1))
            }
            
        except Exception:
            return {'test_count': 0, 'status': 'unknown', 'strength_updated': 0.5}
    
    def _detect_mitigation_blocks(self, candles: List[Dict]) -> List[Dict]:
        """Detect mitigation blocks - institutional order management areas."""
        try:
            mitigation_blocks = []
            
            # Look for areas where price returns to "mitigate" imbalances
            for i in range(5, len(candles) - 5):
                current_area = candles[i-2:i+3]  # 5-candle area
                
                # Calculate area boundaries
                area_high = max(float(c['high']) for c in current_area)
                area_low = min(float(c['low']) for c in current_area)
                area_mid = (area_high + area_low) / 2
                
                # Look for return to this area (mitigation)
                future_candles = candles[i+3:i+15]  # Look ahead 12 candles
                
                mitigation_tests = 0
                for future_candle in future_candles:
                    high, low = float(future_candle['high']), float(future_candle['low'])
                    
                    # Check if price returned to this area
                    if (low <= area_high and high >= area_low):
                        mitigation_tests += 1
                
                # If area was tested multiple times, it's a mitigation block
                if mitigation_tests >= 2:
                    mitigation = {
                        'top': area_high,
                        'bottom': area_low,
                        'midpoint': area_mid,
                        'formation_index': i,
                        'formation_time': candles[i].get('time', 'unknown'),
                        'mitigation_count': mitigation_tests,
                        'strength': min(1.0, mitigation_tests / 5),  # Max strength at 5 tests
                        'type': 'mitigation_block',
                        'status': 'active'
                    }
                    
                    mitigation_blocks.append(mitigation)
            
            return mitigation_blocks
            
        except Exception as e:
            return []
    
    def _identify_liquidity_pools(self, candles: List[Dict], equal_levels: Dict) -> List[Dict]:
        """Identify liquidity pools where stops are likely clustered."""
        try:
            liquidity_pools = []
            
            # Combine equal highs and lows
            all_levels = []
            all_levels.extend(equal_levels.get('equal_highs', []))
            all_levels.extend(equal_levels.get('equal_lows', []))
            
            # Add swing highs and lows
            all_levels.extend(equal_levels.get('swing_highs', []))
            all_levels.extend(equal_levels.get('swing_lows', []))
            
            if not all_levels:
                return liquidity_pools
            
            # Group levels by proximity to find clusters
            cluster_distance = self.detection_params['liquidity_cluster_distance']
            
            for level_data in all_levels:
                level_price = level_data.get('level', level_data.get('price', 0))
                
                # Count nearby levels
                nearby_levels = []
                for other_level in all_levels:
                    other_price = other_level.get('level', other_level.get('price', 0))
                    
                    if abs(level_price - other_price) <= cluster_distance:
                        nearby_levels.append(other_level)
                
                # If multiple levels clustered, it's a liquidity pool
                if len(nearby_levels) >= 3:
                    pool_center = np.mean([l.get('level', l.get('price', 0)) for l in nearby_levels])
                    pool_strength = len(nearby_levels) / 10  # Normalize
                    
                    # Calculate pool type
                    highs = [l for l in nearby_levels if l.get('type') in ['equal_highs', 'resistance'] or 'high' in str(l.get('type', ''))]
                    lows = [l for l in nearby_levels if l.get('type') in ['equal_lows', 'support'] or 'low' in str(l.get('type', ''))]
                    
                    if len(highs) > len(lows):
                        pool_type = 'resistance_pool'
                        bias = 'sell_stops_above'
                    elif len(lows) > len(highs):
                        pool_type = 'support_pool'
                        bias = 'buy_stops_below'
                    else:
                        pool_type = 'mixed_pool'
                        bias = 'stops_both_sides'
                    
                    pool = {
                        'center': pool_center,
                        'type': pool_type,
                        'bias': bias,
                        'strength': min(1.0, pool_strength),
                        'level_count': len(nearby_levels),
                        'levels': nearby_levels,
                        'radius': cluster_distance
                    }
                    
                    # Avoid duplicates
                    if not any(abs(pool['center'] - existing['center']) < cluster_distance/2 for existing in liquidity_pools):
                        liquidity_pools.append(pool)
            
            return liquidity_pools
            
        except Exception as e:
            return []
    
    def _classify_range_type(self, candles: List[Dict]) -> Dict[str, Any]:
        """Classify range type (external vs internal)."""
        try:
            if len(candles) < 20:
                return {'type': 'insufficient_data'}
            
            # Compare recent range to previous range
            recent_candles = candles[-10:]
            previous_candles = candles[-20:-10]
            
            recent_high = max(float(c['high']) for c in recent_candles)
            recent_low = min(float(c['low']) for c in recent_candles)
            
            previous_high = max(float(c['high']) for c in previous_candles)
            previous_low = min(float(c['low']) for c in previous_candles)
            
            # Determine range relationship
            if recent_high > previous_high and recent_low > previous_low:
                range_type = 'external_higher'
                description = 'Current range is above previous range'
            elif recent_high < previous_high and recent_low < previous_low:
                range_type = 'external_lower'
                description = 'Current range is below previous range'
            elif recent_high <= previous_high and recent_low >= previous_low:
                range_type = 'internal'
                description = 'Current range is inside previous range'
            elif recent_high >= previous_high and recent_low <= previous_low:
                range_type = 'expansion'
                description = 'Current range has expanded beyond previous range'
            else:
                range_type = 'overlap'
                description = 'Current range partially overlaps previous range'
            
            # Calculate expansion/contraction ratio
            recent_size = recent_high - recent_low
            previous_size = previous_high - previous_low
            size_ratio = recent_size / previous_size if previous_size > 0 else 1.0
            
            return {
                'type': range_type,
                'description': description,
                'recent_range': {'high': recent_high, 'low': recent_low, 'size': recent_size},
                'previous_range': {'high': previous_high, 'low': previous_low, 'size': previous_size},
                'size_ratio': size_ratio,
                'expansion': size_ratio > 1.2,
                'contraction': size_ratio < 0.8
            }
            
        except Exception as e:
            return {'type': 'error', 'error': str(e)}
    
    def _calculate_array_confluence(self, 
                                   equal_levels: Dict, 
                                   fvg_analysis: Dict, 
                                   order_blocks: List[Dict], 
                                   breaker_blocks: List[Dict]) -> Dict[str, Any]:
        """Calculate confluence between different PD arrays."""
        try:
            confluence_zones = []
            tolerance = self.detection_params['liquidity_cluster_distance']
            
            # Collect all significant levels
            all_levels = []
            
            # Add equal levels
            for level in equal_levels.get('equal_highs', []):
                all_levels.append({'price': level['level'], 'type': 'equal_high', 'strength': level['strength']})
            for level in equal_levels.get('equal_lows', []):
                all_levels.append({'price': level['level'], 'type': 'equal_low', 'strength': level['strength']})
            
            # Add FVG midpoints
            for fvg in fvg_analysis.get('fvgs', []):
                all_levels.append({'price': fvg['midpoint'], 'type': f"fvg_{fvg['type']}", 'strength': fvg['strength']})
            
            # Add order block levels
            for ob in order_blocks:
                all_levels.append({'price': (ob['top'] + ob['bottom']) / 2, 'type': f"ob_{ob['type']}", 'strength': ob['strength']})
            
            # Add breaker levels
            for breaker in breaker_blocks:
                all_levels.append({'price': breaker['level'], 'type': f"breaker_{breaker['type']}", 'strength': breaker['strength']})
            
            # Find confluence zones
            for i, level1 in enumerate(all_levels):
                confluent_levels = [level1]
                
                for j, level2 in enumerate(all_levels):
                    if i != j and abs(level1['price'] - level2['price']) <= tolerance:
                        confluent_levels.append(level2)
                
                if len(confluent_levels) >= 2:
                    zone_center = np.mean([l['price'] for l in confluent_levels])
                    zone_strength = np.mean([l['strength'] for l in confluent_levels])
                    
                    confluence_zone = {
                        'center': zone_center,
                        'strength': zone_strength,
                        'level_count': len(confluent_levels),
                        'contributing_levels': confluent_levels,
                        'types': list(set(l['type'] for l in confluent_levels))
                    }
                    
                    # Avoid duplicates
                    if not any(abs(zone_center - existing['center']) < tolerance/2 for existing in confluence_zones):
                        confluence_zones.append(confluence_zone)
            
            # Sort by strength
            confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'zones': confluence_zones,
                'total_zones': len(confluence_zones),
                'strongest_zone': confluence_zones[0] if confluence_zones else None,
                'high_confluence_zones': [z for z in confluence_zones if z['level_count'] >= 3]
            }
            
        except Exception as e:
            return {'zones': [], 'error': str(e)}
    
    def _create_array_response(self, 
                              valid: bool,
                              equal_levels: Dict = None,
                              fvg_analysis: Dict = None,
                              order_blocks: List[Dict] = None,
                              breaker_blocks: List[Dict] = None,
                              mitigation_blocks: List[Dict] = None,
                              liquidity_pools: List[Dict] = None,
                              range_classification: Dict = None,
                              confluence_analysis: Dict = None,
                              symbol: str = "",
                              timeframe: str = "",
                              error: str = "") -> Dict[str, Any]:
        """Create comprehensive PD array response."""
        
        if not valid:
            return {"valid": False, "error": error}
        
        return {
            "valid": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "equal_levels": equal_levels or {},
            "fair_value_gaps": fvg_analysis or {},
            "order_blocks": order_blocks or [],
            "breaker_blocks": breaker_blocks or [],
            "mitigation_blocks": mitigation_blocks or [],
            "liquidity_pools": liquidity_pools or [],
            "range_classification": range_classification or {},
            "confluence_analysis": confluence_analysis or {},
            "summary": {
                "total_equal_levels": len((equal_levels or {}).get('equal_highs', [])) + len((equal_levels or {}).get('equal_lows', [])),
                "total_fvgs": len((fvg_analysis or {}).get('fvgs', [])),
                "total_order_blocks": len(order_blocks or []),
                "total_breaker_blocks": len(breaker_blocks or []),
                "total_mitigation_blocks": len(mitigation_blocks or []),
                "total_liquidity_pools": len(liquidity_pools or []),
                "total_confluence_zones": len((confluence_analysis or {}).get('zones', []))
            },
            "metadata": {
                "total_detections": self.total_detections,
                "engine_version": "1.0.0",
                "analysis_type": "pd_array_detection"
            }
        }
    
    def _update_array_memory(self, symbol: str, response: Dict):
        """Update PD array memory for performance tracking."""
        try:
            # Store arrays in memory for future reference
            if response.get('equal_levels'):
                self.equal_highs[symbol].extend(response['equal_levels'].get('equal_highs', []))
                self.equal_lows[symbol].extend(response['equal_levels'].get('equal_lows', []))
            
            if response.get('fair_value_gaps'):
                self.fair_value_gaps[symbol].extend(response['fair_value_gaps'].get('fvgs', []))
            
            if response.get('order_blocks'):
                self.order_blocks[symbol].extend(response['order_blocks'])
            
            if response.get('breaker_blocks'):
                self.breaker_blocks[symbol].extend(response['breaker_blocks'])
            
            if response.get('mitigation_blocks'):
                self.mitigation_blocks[symbol].extend(response['mitigation_blocks'])
            
            if response.get('liquidity_pools'):
                self.liquidity_pools[symbol].extend(response['liquidity_pools'])
            
            # Keep memory size manageable
            for memory_dict in [self.equal_highs, self.equal_lows, self.fair_value_gaps, 
                               self.order_blocks, self.breaker_blocks, self.mitigation_blocks, 
                               self.liquidity_pools]:
                if len(memory_dict[symbol]) > 100:
                    memory_dict[symbol] = memory_dict[symbol][-50:]  # Keep last 50
                    
        except Exception:
            pass  # Silent fail for memory updates
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        return {
            "total_detections": self.total_detections,
            "successful_trades_from_arrays": self.successful_trades_from_arrays,
            "success_rate": self.successful_trades_from_arrays / max(1, self.total_detections),
            "array_performance": dict(self.array_performance),
            "memory_sizes": {
                "equal_highs": sum(len(levels) for levels in self.equal_highs.values()),
                "equal_lows": sum(len(levels) for levels in self.equal_lows.values()),
                "fair_value_gaps": sum(len(fvgs) for fvgs in self.fair_value_gaps.values()),
                "order_blocks": sum(len(obs) for obs in self.order_blocks.values()),
                "breaker_blocks": sum(len(bbs) for bbs in self.breaker_blocks.values()),
                "mitigation_blocks": sum(len(mbs) for mbs in self.mitigation_blocks.values()),
                "liquidity_pools": sum(len(pools) for pools in self.liquidity_pools.values())
            },
            "detection_parameters": self.detection_params,
            "engine_version": "1.0.0"
        }
<<<<<<< Current (Your changes)
<<<<<<< HEAD
=======
<<<<<<< Current (Your changes)
=======
        }
>>>>>>> 4323fc9 (upgraded)
=======
>>>>>>> Incoming (Background Agent changes)
>>>>>>> 9af9454 (merge)
=======
>>>>>>> Incoming (Background Agent changes)
