# core/system_validator.py

import os
import sys
import importlib
import warnings
from typing import Dict, List, Tuple, Any, Optional
import time
import json

warnings.filterwarnings('ignore')


class SystemValidator:
    """
    Comprehensive system validation for production readiness.
    
    Validates:
    - Dependencies and imports
    - Configuration integrity  
    - ML component functionality
    - Data pipeline health
    - Performance benchmarks
    """
    
    def __init__(self):
        self.validation_results = {}
        self.critical_failures = []
        self.warnings = []
        self.performance_benchmarks = {}
        
    def validate_all(self) -> Dict[str, Any]:
        """Run complete system validation"""
        
        print("🔍 Starting comprehensive system validation...")
        start_time = time.time()
        
        # Core validation steps
        self._validate_dependencies()
        self._validate_imports() 
        self._validate_configuration()
        self._validate_ml_components()
        self._validate_data_pipeline()
        self._run_performance_benchmarks()
        self._validate_error_handling()
        
        validation_time = time.time() - start_time
        
        # Generate final report
        report = {
            "validation_time": validation_time,
            "total_checks": len(self.validation_results),
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "overall_status": "PASS" if len(self.critical_failures) == 0 else "FAIL",
            "results": self.validation_results,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "performance_benchmarks": self.performance_benchmarks,
            "recommendations": self._generate_recommendations()
        }
        
        self._print_validation_report(report)
        return report
    
    def _validate_dependencies(self):
        """Validate required dependencies are available"""
        print("📦 Validating dependencies...")
        
        required_packages = [
            ('numpy', '1.26.0'),
            ('pandas', '2.2.0'),
            ('scikit-learn', '1.6.0'),
            ('joblib', '1.3.0')
        ]
        
        optional_packages = [
            ('mlflow', '2.8.0'),
            ('evidently', '0.4.0'), 
            ('whylogs', '1.3.0'),
            ('river', '0.21.0'),
            ('torch', '2.1.0'),
            ('tensorflow', '2.15.0')
        ]
        
        # Check required packages
        missing_required = []
        for package, min_version in required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.validation_results[f"dependency_{package}"] = {
                    "status": "PASS",
                    "version": version,
                    "required": min_version
                }
            except ImportError:
                missing_required.append(package)
                self.validation_results[f"dependency_{package}"] = {
                    "status": "FAIL",
                    "error": "Package not found"
                }
                self.critical_failures.append(f"Required package missing: {package}")
        
        # Check optional packages
        missing_optional = []
        for package, min_version in optional_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.validation_results[f"optional_{package}"] = {
                    "status": "PASS",
                    "version": version
                }
            except ImportError:
                missing_optional.append(package)
                self.validation_results[f"optional_{package}"] = {
                    "status": "WARN",
                    "message": "Optional package not available - using fallback"
                }
                self.warnings.append(f"Optional package missing: {package}")
        
        if missing_required:
            print(f"❌ Missing required packages: {missing_required}")
        if missing_optional:
            print(f"⚠️ Missing optional packages: {missing_optional}")
        else:
            print("✅ All dependencies available")
    
    def _validate_imports(self):
        """Validate all custom imports work correctly"""
        print("📥 Validating imports...")
        
        core_modules = [
            'core.policy_service',
            'core.ml_tracker', 
            'core.drift_monitor',
            'core.online_learner',
            'core.model_manager',
            'utils.feature_store',
            'utils.config',
            'memory.learning'
        ]
        
        for module_name in core_modules:
            try:
                module = importlib.import_module(module_name)
                self.validation_results[f"import_{module_name}"] = {
                    "status": "PASS",
                    "module": str(module)
                }
            except ImportError as e:
                self.validation_results[f"import_{module_name}"] = {
                    "status": "FAIL", 
                    "error": str(e)
                }
                self.critical_failures.append(f"Import failed: {module_name} - {e}")
            except Exception as e:
                self.validation_results[f"import_{module_name}"] = {
                    "status": "WARN",
                    "error": str(e)
                }
                self.warnings.append(f"Import warning: {module_name} - {e}")
        
        print(f"✅ Import validation complete")
    
    def _validate_configuration(self):
        """Validate configuration integrity"""
        print("⚙️ Validating configuration...")
        
        try:
            from utils.config import cfg
            config = cfg()
            
            required_sections = ['mode', 'risk', 'hybrid', 'filters', 'policy']
            for section in required_sections:
                if section in config:
                    self.validation_results[f"config_{section}"] = {
                        "status": "PASS",
                        "keys": list(config[section].keys())
                    }
                else:
                    self.validation_results[f"config_{section}"] = {
                        "status": "FAIL",
                        "error": f"Missing config section: {section}"
                    }
                    self.critical_failures.append(f"Missing config section: {section}")
            
            # Validate specific config values
            if config.get("risk", {}).get("per_trade_risk", 0) <= 0:
                self.critical_failures.append("Invalid per_trade_risk: must be > 0")
            
            if config.get("hybrid", {}).get("entry_threshold_base", 0) <= 0:
                self.critical_failures.append("Invalid entry_threshold_base: must be > 0")
                
            print("✅ Configuration validation complete")
            
        except Exception as e:
            self.validation_results["config_load"] = {
                "status": "FAIL",
                "error": str(e)
            }
            self.critical_failures.append(f"Configuration load failed: {e}")
    
    def _validate_ml_components(self):
        """Validate ML components functionality"""
        print("🧠 Validating ML components...")
        
        # Test PolicyService
        try:
            from core.policy_service import PolicyService
            from core.intelligence import IntelligenceCore
            from utils.logging_setup import setup_logger
            
            logger = setup_logger("test")
            intel = IntelligenceCore(logger=logger, base_threshold=0.6)
            policy = PolicyService(champion=intel, logger=logger)
            
            # Test decision making
            test_features = {
                "ml_confidence": 0.7,
                "trend_align": True,
                "structure_score": 0.6,
                "direction": "long"
            }
            
            decision = policy.decide("EURUSD", test_features)
            
            if "idea" in decision and "meta" in decision:
                self.validation_results["ml_policy_service"] = {
                    "status": "PASS",
                    "test_decision": "Generated successfully"
                }
            else:
                self.validation_results["ml_policy_service"] = {
                    "status": "FAIL",
                    "error": "Invalid decision format"
                }
                self.critical_failures.append("PolicyService decision format invalid")
                
        except Exception as e:
            self.validation_results["ml_policy_service"] = {
                "status": "FAIL",
                "error": str(e)
            }
            self.critical_failures.append(f"PolicyService validation failed: {e}")
        
        # Test MLTracker
        try:
            from core.ml_tracker import ml_tracker
            
            # Test basic functionality
            run_id = ml_tracker.start_run("validation_test")
            ml_tracker.log_params({"test_param": 1.0})
            ml_tracker.log_metrics({"test_metric": 0.8})
            ml_tracker.end_run()
            
            self.validation_results["ml_tracker"] = {
                "status": "PASS",
                "run_id": run_id
            }
            
        except Exception as e:
            self.validation_results["ml_tracker"] = {
                "status": "WARN",
                "error": str(e)
            }
            self.warnings.append(f"MLTracker validation failed: {e}")
        
        # Test OnlineLearner
        try:
            from core.online_learner import online_learner
            
            # Test model creation and update
            features = {"feature1": 1.0, "feature2": 0.5}
            target = 1.0
            
            result = online_learner.update_model("EURUSD", features, target, "linear")
            prediction = online_learner.predict("EURUSD", features, "linear")
            
            if isinstance(prediction, (int, float)):
                self.validation_results["ml_online_learner"] = {
                    "status": "PASS",
                    "prediction": prediction
                }
            else:
                self.validation_results["ml_online_learner"] = {
                    "status": "FAIL",
                    "error": "Invalid prediction type"
                }
                self.critical_failures.append("OnlineLearner prediction invalid")
                
        except Exception as e:
            self.validation_results["ml_online_learner"] = {
                "status": "WARN",
                "error": str(e)
            }
            self.warnings.append(f"OnlineLearner validation failed: {e}")
        
        print("✅ ML components validation complete")
    
    def _validate_data_pipeline(self):
        """Validate data pipeline functionality"""
        print("📊 Validating data pipeline...")
        
        try:
            from utils.feature_store import FeatureStore
            import tempfile
            import shutil
            
            # Create temporary feature store
            temp_dir = tempfile.mkdtemp()
            fs = FeatureStore(temp_dir)
            
            # Test writing features
            test_features = {
                "price": 1.1234,
                "volume": 1000,
                "trend": 0.6
            }
            
            test_meta = {
                "trace_id": "test_123",
                "variant": "champion"
            }
            
            fs.write_row("EURUSD", "M15", test_features, meta=test_meta)
            
            # Test writing outcome
            fs.write_outcome("EURUSD", "M15", "buy", 10.0, 2.0, "ml")
            
            # Check files were created
            expected_file = os.path.join(temp_dir, "EURUSD", "M15.ndjson")
            if os.path.exists(expected_file):
                self.validation_results["data_pipeline"] = {
                    "status": "PASS",
                    "file_created": expected_file
                }
            else:
                self.validation_results["data_pipeline"] = {
                    "status": "FAIL",
                    "error": "Feature store file not created"
                }
                self.critical_failures.append("FeatureStore not writing files")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            self.validation_results["data_pipeline"] = {
                "status": "FAIL",
                "error": str(e)
            }
            self.critical_failures.append(f"Data pipeline validation failed: {e}")
        
        print("✅ Data pipeline validation complete")
    
    def _run_performance_benchmarks(self):
        """Run performance benchmarks"""
        print("⚡ Running performance benchmarks...")
        
        import numpy as np
        
        # Benchmark feature processing
        start_time = time.time()
        for _ in range(1000):
            features = {
                "ml_confidence": np.random.random(),
                "trend_align": np.random.choice([True, False]),
                "structure_score": np.random.random(),
                "volatility": np.random.random()
            }
        feature_processing_time = time.time() - start_time
        
        # Benchmark decision making
        try:
            from core.policy_service import PolicyService
            from core.intelligence import IntelligenceCore
            from utils.logging_setup import setup_logger
            
            logger = setup_logger("benchmark")
            intel = IntelligenceCore(logger=logger, base_threshold=0.6)
            policy = PolicyService(champion=intel, logger=logger)
            
            start_time = time.time()
            for _ in range(100):
                features = {
                    "ml_confidence": np.random.random(),
                    "trend_align": np.random.choice([True, False]),
                    "structure_score": np.random.random(),
                    "direction": np.random.choice(["long", "short"])
                }
                decision = policy.decide("EURUSD", features)
            decision_time = time.time() - start_time
            
        except Exception:
            decision_time = float('inf')
        
        self.performance_benchmarks = {
            "feature_processing_1000_ops": f"{feature_processing_time:.4f}s",
            "decision_making_100_ops": f"{decision_time:.4f}s",
            "avg_decision_time": f"{decision_time/100:.6f}s"
        }
        
        # Performance thresholds
        if decision_time/100 > 0.01:  # 10ms per decision
            self.warnings.append(f"Decision making slow: {decision_time/100:.6f}s per decision")
        
        print("✅ Performance benchmarks complete")
    
    def _validate_error_handling(self):
        """Validate error handling and fallbacks"""
        print("🛡️ Validating error handling...")
        
        try:
            from core.policy_service import PolicyService
            
            # Test with invalid champion
            policy = PolicyService(champion=None)
            decision = policy.decide("EURUSD", {})
            
            if decision and "meta" in decision:
                self.validation_results["error_handling"] = {
                    "status": "PASS",
                    "message": "Graceful handling of invalid champion"
                }
            else:
                self.validation_results["error_handling"] = {
                    "status": "FAIL",
                    "error": "No graceful fallback for invalid champion"
                }
                self.critical_failures.append("Error handling insufficient")
                
        except Exception as e:
            self.validation_results["error_handling"] = {
                "status": "WARN",
                "error": str(e)
            }
            self.warnings.append(f"Error handling test failed: {e}")
        
        print("✅ Error handling validation complete")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if len(self.critical_failures) > 0:
            recommendations.append("🚨 CRITICAL: Fix all critical failures before production")
        
        if len(self.warnings) > 5:
            recommendations.append("⚠️ Consider addressing warnings for better reliability")
        
        # Performance recommendations
        avg_decision_time = float(self.performance_benchmarks.get("avg_decision_time", "0s").replace("s", ""))
        if avg_decision_time > 0.005:  # 5ms
            recommendations.append("⚡ Consider optimizing decision making performance")
        
        # Missing optional packages
        missing_ml_packages = [
            result for key, result in self.validation_results.items()
            if key.startswith("optional_") and result["status"] == "WARN"
        ]
        
        if len(missing_ml_packages) > 2:
            recommendations.append("📦 Install optional ML packages for full functionality")
        
        if not recommendations:
            recommendations.append("✅ System appears ready for production")
        
        return recommendations
    
    def _print_validation_report(self, report: Dict[str, Any]):
        """Print formatted validation report"""
        print("\n" + "="*60)
        print("🔍 SYSTEM VALIDATION REPORT")
        print("="*60)
        print(f"Overall Status: {'✅ PASS' if report['overall_status'] == 'PASS' else '❌ FAIL'}")
        print(f"Validation Time: {report['validation_time']:.2f}s")
        print(f"Total Checks: {report['total_checks']}")
        print(f"Critical Failures: {report['critical_failures']}")
        print(f"Warnings: {report['warnings']}")
        
        if report['critical_failures'] > 0:
            print("\n🚨 CRITICAL FAILURES:")
            for failure in report['critical_failures']:
                print(f"  ❌ {failure}")
        
        if report['warnings'] > 0:
            print("\n⚠️ WARNINGS:")
            for warning in report['warnings'][:5]:  # Show first 5
                print(f"  ⚠️ {warning}")
            if len(report['warnings']) > 5:
                print(f"  ... and {len(report['warnings']) - 5} more")
        
        print("\n⚡ PERFORMANCE BENCHMARKS:")
        for metric, value in report['performance_benchmarks'].items():
            print(f"  {metric}: {value}")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print("="*60)


def run_system_validation() -> bool:
    """Run system validation and return True if ready for production"""
    validator = SystemValidator()
    report = validator.validate_all()
    
    # Save report
    os.makedirs("logs/validation", exist_ok=True)
    with open(f"logs/validation/validation_report_{int(time.time())}.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report["overall_status"] == "PASS"


if __name__ == "__main__":
    success = run_system_validation()
    sys.exit(0 if success else 1)
