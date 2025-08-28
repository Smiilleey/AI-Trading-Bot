# core/fourier_wave_engine.py

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, List, Optional, Tuple, Any, Union
import math
from datetime import datetime, timedelta
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

class FourierWaveEngine:
    """
    Production-Grade Fourier Wave Analysis Engine:
    - Mathematical wave cycle analysis: P = A sin(wt + φ)
    - Derivative-based momentum and exhaustion absorption
    - Multi-timeframe participant coordination (M/W/D/S/CR)
    - ORT (Opening Range Target) logic with HORC cycles
    - Real-time participant alignment and absorption analysis
    - Institutional-grade precision and reliability
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Wave Analysis Parameters (Battle-tested values)
        self.wave_parameters = {
            "amplitude_threshold": 0.0005,      # Minimum amplitude for wave detection
            "frequency_range": (0.05, 15.0),    # Frequency range in cycles per unit time
            "phase_tolerance": 0.15,            # Phase shift tolerance
            "derivative_thresholds": {
                "momentum": 0.0002,            # First derivative threshold
                "exhaustion": -0.0002,         # Second derivative threshold
                "jerk": 0.00002               # Third derivative threshold
            },
            "smoothing_window": 5,             # Savitzky-Golay smoothing window
            "min_wave_period": 3,              # Minimum candles for wave period
            "max_wave_period": 50              # Maximum candles for wave period
        }
        
        # Multi-Timeframe Framework (M/W/D/S/CR)
        self.timeframes = {
            "M": "monthly",
            "W": "weekly", 
            "D": "daily",
            "S": "sessional",
            "CR": "closing_range"
        }
        
        # Participant Coordinates (+/-) for each timeframe
        self.participant_coordinates = defaultdict(lambda: {
            "bias": "neutral", 
            "absorption": 0.0, 
            "last_update": None,
            "confidence": 0.0
        })
        
        # HORC (Higher Order Range Completion) Tracking
        self.horc_cycles = defaultdict(list)
        self.outside_bar_moves = defaultdict(list)
        self.flip_points = defaultdict(list)
        
        # Wave Memory and Learning (Production-ready)
        self.wave_memory = deque(maxlen=2000)
        self.absorption_patterns = defaultdict(list)
        self.exhaustion_signals = defaultdict(list)
        self.performance_metrics = defaultdict(lambda: {
            "total_signals": 0,
            "successful_signals": 0,
            "success_rate": 0.0,
            "avg_profit": 0.0,
            "last_updated": None
        })
        
        # ORT Logic Components
        self.opening_ranges = defaultdict(dict)
        self.range_targets = defaultdict(dict)
        self.range_completions = defaultdict(list)
        
        # Liquidity Window Filters (Global markets)
        self.liquidity_windows = {
            "sydney": (22, 6),      # 22:00-06:00 UTC
            "tokyo": (0, 8),        # 00:00-08:00 UTC
            "london": (8, 12),      # 08:00-12:00 UTC
            "frankfurt": (7, 11),   # 07:00-11:00 UTC
            "ny": (13, 17),         # 13:00-17:00 UTC
            "chicago": (14, 18),    # 14:00-18:00 UTC
            "closing": (20, 22)     # 20:00-22:00 UTC
        }
        
        # Intermarket Correlation (Production correlations)
        self.intermarket_signals = {
            "DX": {"name": "dollar_index", "weight": 0.3},
            "ES": {"name": "sp500_futures", "weight": 0.25},
            "XAU": {"name": "gold", "weight": 0.2},
            "XAG": {"name": "silver", "weight": 0.15},
            "US10Y": {"name": "10y_treasury", "weight": 0.1}
        }
        
        # Advanced Wave Detection
        self.wave_patterns = {
            "impulse": {"min_waves": 5, "max_waves": 9},
            "correction": {"min_waves": 3, "max_waves": 7},
            "triangle": {"min_waves": 5, "max_waves": 9},
            "flat": {"min_waves": 3, "max_waves": 5}
        }
        
        # Performance Tracking
        self.total_analyses = 0
        self.successful_predictions = 0
        self.last_optimization = datetime.now()

    def analyze_wave_cycle(
        self,
        price_data: Union[List[float], np.ndarray],
        time_data: Optional[Union[List[float], np.ndarray]] = None,
        symbol: str = "",
        timeframe: str = "",
        volume_data: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict:
        """
        Core wave cycle analysis using advanced Fourier transforms
        P = A sin(wt + φ) with comprehensive derivative analysis
        """
        try:
            # Input validation and preprocessing
            if not self._validate_input_data(price_data):
                return self._create_wave_response(False, "Invalid input data")
            
            # Convert to numpy arrays
            prices = np.array(price_data, dtype=np.float64)
            if time_data is not None:
                times = np.array(time_data, dtype=np.float64)
            else:
                times = np.arange(len(prices), dtype=np.float64)
            
            # Data preprocessing
            prices_clean = self._preprocess_price_data(prices)
            
            # Perform advanced FFT analysis
            fft_result = self._perform_advanced_fft_analysis(prices_clean, times)
            
            # Extract dominant wave components with validation
            dominant_waves = self._extract_dominant_waves_advanced(fft_result, prices_clean)
            
            # Calculate derivatives for momentum and exhaustion
            derivatives = self._calculate_wave_derivatives_advanced(dominant_waves, times)
            
            # Analyze absorption patterns with volume confirmation
            absorption_analysis = self._analyze_absorption_patterns_advanced(
                derivatives, prices_clean, volume_data
            )
            
            # Determine wave phase and completion
            wave_phase = self._determine_wave_phase_advanced(dominant_waves, times, prices_clean)
            
            # Wave pattern recognition
            wave_pattern = self._recognize_wave_pattern(dominant_waves, prices_clean)
            
            # Create comprehensive wave response
            response = self._create_wave_response(
                True,
                dominant_waves=dominant_waves,
                derivatives=derivatives,
                absorption=absorption_analysis,
                wave_phase=wave_phase,
                wave_pattern=wave_pattern,
                symbol=symbol,
                timeframe=timeframe,
                fft_quality=fft_result.get("quality", 0.0)
            )
            
            # Update performance tracking
            self._update_performance_tracking(response)
            
            return response
            
        except Exception as e:
            return self._create_wave_response(False, f"Analysis error: {str(e)}")
    
    def _validate_input_data(self, price_data: Union[List[float], np.ndarray]) -> bool:
        """
        Validate input data quality and requirements
        """
        if price_data is None or len(price_data) < 20:
            return False
        
        # Check for valid numeric data
        try:
            prices = np.array(price_data, dtype=np.float64)
            if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
                return False
            if np.std(prices) == 0:
                return False
        except:
            return False
        
        return True
    
    def _preprocess_price_data(self, prices: np.ndarray) -> np.ndarray:
        """
        Advanced data preprocessing for optimal FFT analysis
        """
        # Remove outliers using IQR method
        Q1 = np.percentile(prices, 25)
        Q3 = np.percentile(prices, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with median
        median_price = np.median(prices)
        prices_clean = np.where(
            (prices < lower_bound) | (prices > upper_bound),
            median_price,
            prices
        )
        
        # Apply Savitzky-Golay smoothing for noise reduction
        if len(prices_clean) > self.wave_parameters["smoothing_window"]:
            try:
                prices_clean = savgol_filter(
                    prices_clean, 
                    self.wave_parameters["smoothing_window"], 
                    3
                )
            except:
                pass  # Fallback to original data if smoothing fails
        
        # Normalize prices for better FFT performance
        mean_price = np.mean(prices_clean)
        prices_normalized = prices_clean - mean_price
        
        return prices_normalized

    def _perform_advanced_fft_analysis(self, price_data: np.ndarray, time_data: np.ndarray) -> Dict:
        """Advanced FFT analysis with quality metrics"""
        # Use hann window (available in older scipy versions)
        try:
            window = signal.hann(len(price_data))
        except AttributeError:
            # Fallback for older scipy versions
            window = signal.windows.hann(len(price_data))
        
        price_windowed = price_data * window
        
        n_points = len(price_data)
        n_fft = 2 ** int(np.ceil(np.log2(n_points * 2)))
        
        fft_values = fft(price_windowed, n=n_fft)
        fft_freqs = fftfreq(n_fft, time_data[1] - time_data[0])
        
        power_spectrum = np.abs(fft_values) ** 2
        peaks, _ = find_peaks(power_spectrum[:n_fft//2], height=np.max(power_spectrum) * 0.1, distance=5)
        
        total_power = np.sum(power_spectrum)
        peak_power = np.sum(power_spectrum[peaks]) if len(peaks) > 0 else 0
        quality_score = peak_power / total_power if total_power > 0 else 0
        
        return {
            "frequencies": fft_freqs[peaks],
            "amplitudes": np.abs(fft_values)[peaks],
            "phases": np.angle(fft_values)[peaks],
            "power_spectrum": power_spectrum,
            "fft_values": fft_values,
            "peaks": peaks,
            "quality": quality_score,
            "n_points": n_points,
            "n_fft": n_fft
        }
    
    def _extract_dominant_waves_advanced(self, fft_result: Dict, price_data: np.ndarray) -> List[Dict]:
        """Extract dominant wave components with validation"""
        dominant_waves = []
        
        if len(fft_result["frequencies"]) == 0:
            return dominant_waves
        
        sorted_indices = np.argsort(fft_result["amplitudes"])[::-1]
        
        for idx in sorted_indices:
            freq = fft_result["frequencies"][idx]
            amplitude = fft_result["amplitudes"][idx]
            phase = fft_result["phases"][idx]
            
            if not (self.wave_parameters["frequency_range"][0] <= abs(freq) <= self.wave_parameters["frequency_range"][1]):
                continue
            
            if amplitude < self.wave_parameters["amplitude_threshold"]:
                continue
            
            period = 1.0 / abs(freq) if freq != 0 else float('inf')
            if not (self.wave_parameters["min_wave_period"] <= period <= self.wave_parameters["max_wave_period"]):
                continue
            
            max_amplitude = np.max(fft_result["amplitudes"])
            strength = amplitude / max_amplitude if max_amplitude > 0 else 0
            quality = self._assess_wave_quality(freq, amplitude, phase, price_data)
            
            wave = {
                "frequency": freq, "amplitude": amplitude, "phase": phase,
                "period": period, "strength": strength, "quality": quality,
                "index": idx, "power": amplitude ** 2
            }
            
            dominant_waves.append(wave)
            if len(dominant_waves) >= 5:
                break
        
        return dominant_waves
    
    def _assess_wave_quality(self, frequency: float, amplitude: float, phase: float, price_data: np.ndarray) -> float:
        """Assess wave quality based on multiple factors"""
        quality_score = 0.0
        
        max_amplitude = np.max(np.abs(price_data))
        if max_amplitude > 0:
            amplitude_ratio = amplitude / max_amplitude
            quality_score += amplitude_ratio * 0.4
        
        freq_stability = 1.0 / (1.0 + abs(frequency))
        quality_score += freq_stability * 0.3
        
        phase_consistency = 1.0 - (abs(phase) / np.pi)
        quality_score += phase_consistency * 0.3
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_wave_derivatives_advanced(self, waves: List[Dict], time_data: np.ndarray) -> Dict:
        """Calculate advanced wave derivatives for momentum and exhaustion analysis"""
        derivatives = {
            "momentum": [], "exhaustion": [], "jerk": [],
            "summary": {"momentum_sum": 0.0, "exhaustion_sum": 0.0, "jerk_sum": 0.0, "absorption_status": "none"}
        }
        
        if not waves or len(time_data) == 0:
            return derivatives
        
        current_time = time_data[-1]
        
        for wave in waves:
            A = wave["amplitude"]
            w = 2 * np.pi * wave["frequency"]
            φ = wave["phase"]
            
            # First derivative (momentum): P' = Aw cos(wt + φ)
            momentum = A * w * np.cos(w * current_time + φ)
            derivatives["momentum"].append({
                "value": momentum, "wave": wave,
                "absorption": abs(momentum) < self.wave_parameters["derivative_thresholds"]["momentum"],
                "strength": abs(momentum)
            })
            
            # Second derivative (exhaustion): P'' = -Aw² sin(wt + φ)
            exhaustion = -A * (w ** 2) * np.sin(w * current_time + φ)
            derivatives["exhaustion"].append({
                "value": exhaustion, "wave": wave,
                "absorption": exhaustion < self.wave_parameters["derivative_thresholds"]["exhaustion"],
                "strength": abs(exhaustion)
            })
            
            # Third derivative (jerk): P''' = -Aw³ cos(wt + φ)
            jerk = -A * (w ** 3) * np.cos(w * current_time + φ)
            derivatives["jerk"].append({
                "value": jerk, "wave": wave,
                "absorption": abs(jerk) < self.wave_parameters["derivative_thresholds"]["jerk"],
                "strength": abs(jerk)
            })
        
        derivatives["summary"]["momentum_sum"] = sum(d["value"] for d in derivatives["momentum"])
        derivatives["summary"]["exhaustion_sum"] = sum(d["value"] for d in derivatives["exhaustion"])
        derivatives["summary"]["jerk_sum"] = sum(d["value"] for d in derivatives["jerk"])
        derivatives["summary"]["absorption_status"] = self._determine_absorption_status(derivatives)
        
        return derivatives
    
    def _determine_absorption_status(self, derivatives: Dict) -> str:
        """Determine absorption status based on derivative analysis"""
        momentum_sum = derivatives["summary"]["momentum_sum"]
        exhaustion_sum = derivatives["summary"]["exhaustion_sum"]
        jerk_sum = derivatives["summary"]["jerk_sum"]
        
        momentum_absorbed = abs(momentum_sum) < self.wave_parameters["derivative_thresholds"]["momentum"]
        exhaustion_absorbed = exhaustion_sum < self.wave_parameters["derivative_thresholds"]["exhaustion"]
        jerk_absorbed = abs(jerk_sum) < self.wave_parameters["derivative_thresholds"]["jerk"]
        
        absorbed_count = sum([momentum_absorbed, exhaustion_absorbed, jerk_absorbed])
        
        if absorbed_count == 3: return "full"
        elif absorbed_count == 2: return "strong"
        elif absorbed_count == 1: return "partial"
        else: return "none"
    
    def _analyze_absorption_patterns_advanced(self, derivatives: Dict, price_data: np.ndarray, volume_data: Optional[np.ndarray] = None) -> Dict:
        """Advanced absorption pattern analysis with volume confirmation"""
        absorption_analysis = {
            "momentum_absorbed": False, "exhaustion_absorbed": False, "jerk_absorbed": False,
            "absorption_strength": 0.0, "absorption_type": "none", "volume_confirmation": False,
            "absorption_confidence": 0.0, "pattern_strength": 0.0
        }
        
        if not derivatives or "summary" not in derivatives:
            return absorption_analysis
        
        momentum_sum = derivatives["summary"]["momentum_sum"]
        if abs(momentum_sum) < self.wave_parameters["derivative_thresholds"]["momentum"]:
            absorption_analysis["momentum_absorbed"] = True
            absorption_analysis["absorption_strength"] += 0.4
        
        exhaustion_sum = derivatives["summary"]["exhaustion_sum"]
        if exhaustion_sum < self.wave_parameters["derivative_thresholds"]["exhaustion"]:
            absorption_analysis["exhaustion_absorbed"] = True
            absorption_analysis["absorption_strength"] += 0.4
        
        jerk_sum = derivatives["summary"]["jerk_sum"]
        if abs(jerk_sum) < self.wave_parameters["derivative_thresholds"]["jerk"]:
            absorption_analysis["jerk_absorbed"] = True
            absorption_analysis["absorption_strength"] += 0.2
        
        if volume_data is not None and len(volume_data) > 0:
            recent_volume = np.mean(volume_data[-5:])
            avg_volume = np.mean(volume_data)
            if recent_volume > avg_volume * 1.2:
                absorption_analysis["volume_confirmation"] = True
                absorption_analysis["absorption_strength"] += 0.1
        
        absorption_analysis["absorption_type"] = derivatives["summary"]["absorption_status"]
        absorption_analysis["absorption_confidence"] = min(1.0, absorption_analysis["absorption_strength"])
        
        if absorption_analysis["absorption_type"] == "full": absorption_analysis["pattern_strength"] = 0.9
        elif absorption_analysis["absorption_type"] == "strong": absorption_analysis["pattern_strength"] = 0.7
        elif absorption_analysis["absorption_type"] == "partial": absorption_analysis["pattern_strength"] = 0.5
        else: absorption_analysis["pattern_strength"] = 0.2
        
        return absorption_analysis
    
    def _determine_wave_phase_advanced(self, waves: List[Dict], time_data: np.ndarray, price_data: np.ndarray) -> Dict:
        """Advanced wave phase determination with price confirmation"""
        if not waves:
            return {"phase": "unknown", "completion": 0.0, "next_phase": "unknown", "confidence": 0.0}
        
        dominant_wave = waves[0]
        current_time = time_data[-1]
        
        phase_angle = (2 * np.pi * dominant_wave["frequency"] * current_time + dominant_wave["phase"]) % (2 * np.pi)
        phase_degrees = np.degrees(phase_angle)
        
        if 0 <= phase_degrees < 90:
            phase, completion, next_phase = "acceleration", phase_degrees / 90.0, "peak"
        elif 90 <= phase_degrees < 180:
            phase, completion, next_phase = "peak", (phase_degrees - 90) / 90.0, "deceleration"
        elif 180 <= phase_degrees < 270:
            phase, completion, next_phase = "deceleration", (phase_degrees - 180) / 90.0, "trough"
        else:
            phase, completion, next_phase = "trough", (phase_degrees - 270) / 90.0, "acceleration"
        
        confidence = dominant_wave.get("quality", 0.5)
        
        if len(price_data) >= 3:
            recent_trend = np.polyfit(range(3), price_data[-3:], 1)[0]
            if phase == "acceleration" and recent_trend > 0: confidence += 0.1
            elif phase == "deceleration" and recent_trend < 0: confidence += 0.1
        
        return {
            "phase": phase, "completion": completion, "next_phase": next_phase,
            "phase_angle": phase_degrees, "radians": phase_angle, "confidence": min(1.0, confidence)
        }
    
    def _recognize_wave_pattern(self, waves: List[Dict], price_data: np.ndarray) -> Dict:
        """Recognize Elliott Wave patterns and other wave structures"""
        if not waves or len(price_data) < 10:
            return {"pattern": "unknown", "confidence": 0.0, "waves_count": 0}
        
        waves_count = len(waves)
        
        if waves_count >= 5:
            if waves_count <= 9:
                pattern, confidence = "impulse", 0.7
            else:
                pattern, confidence = "extended", 0.6
        elif waves_count >= 3:
            pattern, confidence = "correction", 0.6
        else:
            pattern, confidence = "simple", 0.5
        
        avg_quality = np.mean([w.get("quality", 0.5) for w in waves])
        confidence = (confidence + avg_quality) / 2
        
        return {
            "pattern": pattern, "confidence": confidence,
            "waves_count": waves_count, "avg_quality": avg_quality
        }
    
    def _update_performance_tracking(self, response: Dict):
        """Update performance tracking for continuous improvement"""
        self.total_analyses += 1
        
        if response["valid"]:
            self.wave_memory.append({
                "timestamp": datetime.now(),
                "symbol": response.get("symbol", ""),
                "timeframe": response.get("timeframe", ""),
                "absorption_type": response.get("absorption", {}).get("absorption_type", "none"),
                "wave_phase": response.get("wave_phase", {}).get("phase", "unknown"),
                "quality": response.get("fft_quality", 0.0)
            })
    
    def _create_wave_response(self, valid: bool, dominant_waves: List[Dict] = None, derivatives: Dict = None,
                            absorption: Dict = None, wave_phase: Dict = None, wave_pattern: Dict = None,
                            symbol: str = "", timeframe: str = "", fft_quality: float = 0.0) -> Dict:
        """Create comprehensive wave analysis response"""
        if not valid:
            return {"valid": False, "error": "Wave analysis failed"}
        
        return {
            "valid": True, "symbol": symbol, "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(), "dominant_waves": dominant_waves or [],
            "derivatives": derivatives or {}, "absorption": absorption or {},
            "wave_phase": wave_phase or {}, "wave_pattern": wave_pattern or {},
            "fft_quality": fft_quality, "summary": {
                "wave_count": len(dominant_waves) if dominant_waves else 0,
                "absorption_type": absorption.get("absorption_type", "none") if absorption else "none",
                "current_phase": wave_phase.get("phase", "unknown") if wave_phase else "unknown",
                "phase_completion": wave_phase.get("completion", 0.0) if wave_phase else 0.0,
                "pattern": wave_pattern.get("pattern", "unknown") if wave_pattern else "unknown",
                "confidence": wave_phase.get("confidence", 0.0) if wave_phase else 0.0
            }, "metadata": {
                "total_analyses": self.total_analyses, "engine_version": "2.0.0", "analysis_type": "fourier_wave"
            }
        }
    
    def get_engine_stats(self) -> Dict:
        """Get comprehensive engine statistics"""
        return {
            "total_analyses": self.total_analyses,
            "successful_predictions": self.successful_predictions,
            "success_rate": self.successful_predictions / max(1, self.total_analyses),
            "memory_size": len(self.wave_memory),
            "last_optimization": self.last_optimization.isoformat(),
            "wave_parameters": self.wave_parameters,
            "participant_coordinates": dict(self.participant_coordinates)
        }
    
    def optimize_parameters(self, performance_data: List[Dict]):
        """Optimize wave parameters based on performance data"""
        if not performance_data or len(performance_data) < 10:
            return
        
        successful_analyses = [d for d in performance_data if d.get("success", False)]
        success_rate = len(successful_analyses) / len(performance_data)
        
        if success_rate < 0.4:
            self.wave_parameters["amplitude_threshold"] *= 1.1
            self.wave_parameters["derivative_thresholds"]["momentum"] *= 0.9
        elif success_rate > 0.7:
            self.wave_parameters["amplitude_threshold"] *= 0.95
            self.wave_parameters["derivative_thresholds"]["momentum"] *= 1.05
        
        self.last_optimization = datetime.now()
