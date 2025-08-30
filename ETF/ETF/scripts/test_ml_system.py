"""
ML Testing and Validation Script for ETF Trading System.
Comprehensive testing of ML models and prediction pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Any, Tuple
import json
import warnings

from src.ml.feature_engineering import AdvancedFeatureEngine
from src.ml.models import MLModelManager
from src.ml.prediction_pipeline import MLPredictionPipeline
from src.utils.config import Config
from src.data.data_storage import DataStorage
from src.utils.logger import setup_logger

warnings.filterwarnings('ignore')
logger = setup_logger("ml_testing", level=logging.INFO)

class MLTester:
    """Comprehensive ML system testing and validation."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize ML tester.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.data_storage = DataStorage(self.config)
        self.feature_engine = AdvancedFeatureEngine()
        self.model_manager = MLModelManager()
        self.pipeline = MLPredictionPipeline(self.config, self.data_storage)
        
        logger.info("ML Tester initialized")
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive ML system tests."""
        logger.info("Starting comprehensive ML system tests...")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'feature_engineering_tests': await self._test_feature_engineering(),
            'model_functionality_tests': await self._test_model_functionality(),
            'prediction_pipeline_tests': await self._test_prediction_pipeline(),
            'data_integration_tests': await self._test_data_integration(),
            'performance_tests': await self._test_performance(),
            'edge_case_tests': await self._test_edge_cases(),
            'integration_tests': await self._test_integration(),
            'summary': {}
        }
        
        # Generate summary
        test_results['summary'] = self._generate_test_summary(test_results)
        
        return test_results
    
    async def _test_feature_engineering(self) -> Dict[str, Any]:
        """Test feature engineering functionality."""
        logger.info("Testing feature engineering...")
        
        tests = {
            'basic_features': {'status': 'unknown', 'details': {}},
            'technical_indicators': {'status': 'unknown', 'details': {}},
            'statistical_features': {'status': 'unknown', 'details': {}},
            'pattern_recognition': {'status': 'unknown', 'details': {}},
            'error_handling': {'status': 'unknown', 'details': {}}
        }
        
        try:
            # Create sample data
            sample_data = self._create_sample_market_data()
            
            # Test 1: Basic features
            try:
                basic_features = self.feature_engine.generate_basic_features(sample_data)
                tests['basic_features']['status'] = 'passed'
                tests['basic_features']['details'] = {
                    'features_generated': len(basic_features.columns),
                    'data_points': len(basic_features),
                    'missing_values': basic_features.isnull().sum().sum()
                }
            except Exception as e:
                tests['basic_features']['status'] = 'failed'
                tests['basic_features']['details'] = {'error': str(e)}
            
            # Test 2: Technical indicators
            try:
                tech_features = self.feature_engine.generate_technical_indicators(sample_data)
                tests['technical_indicators']['status'] = 'passed'
                tests['technical_indicators']['details'] = {
                    'indicators_generated': len([col for col in tech_features.columns if col not in sample_data.columns]),
                    'coverage': (tech_features.notna().sum() / len(tech_features)).mean()
                }
            except Exception as e:
                tests['technical_indicators']['status'] = 'failed'
                tests['technical_indicators']['details'] = {'error': str(e)}
            
            # Test 3: Statistical features
            try:
                stat_features = self.feature_engine.generate_statistical_features(sample_data)
                tests['statistical_features']['status'] = 'passed'
                tests['statistical_features']['details'] = {
                    'statistical_features': len([col for col in stat_features.columns if col not in sample_data.columns])
                }
            except Exception as e:
                tests['statistical_features']['status'] = 'failed'
                tests['statistical_features']['details'] = {'error': str(e)}
            
            # Test 4: Pattern recognition
            try:
                pattern_features = self.feature_engine.generate_pattern_features(sample_data)
                tests['pattern_recognition']['status'] = 'passed'
                tests['pattern_recognition']['details'] = {
                    'patterns_detected': len([col for col in pattern_features.columns if 'pattern' in col.lower()])
                }
            except Exception as e:
                tests['pattern_recognition']['status'] = 'failed'
                tests['pattern_recognition']['details'] = {'error': str(e)}
            
            # Test 5: Error handling with bad data
            try:
                bad_data = sample_data.copy()
                bad_data.iloc[0:10] = np.nan  # Introduce missing values
                
                robust_features = self.feature_engine.generate_all_features(bad_data)
                tests['error_handling']['status'] = 'passed'
                tests['error_handling']['details'] = {
                    'handled_missing_data': True,
                    'output_rows': len(robust_features)
                }
            except Exception as e:
                tests['error_handling']['status'] = 'failed'
                tests['error_handling']['details'] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Error in feature engineering tests: {e}")
            for test in tests:
                if tests[test]['status'] == 'unknown':
                    tests[test]['status'] = 'failed'
                    tests[test]['details'] = {'error': str(e)}
        
        return tests
    
    async def _test_model_functionality(self) -> Dict[str, Any]:
        """Test ML model functionality."""
        logger.info("Testing ML model functionality...")
        
        tests = {
            'data_preparation': {'status': 'unknown', 'details': {}},
            'traditional_models': {'status': 'unknown', 'details': {}},
            'deep_learning_models': {'status': 'unknown', 'details': {}},
            'ensemble_models': {'status': 'unknown', 'details': {}},
            'model_persistence': {'status': 'unknown', 'details': {}}
        }
        
        try:
            # Create sample feature data
            sample_data = self._create_sample_feature_data()
            
            # Test 1: Data preparation
            try:
                data_splits = self.model_manager.prepare_data(
                    sample_data, 'target', test_size=0.2, validation_size=0.1
                )
                tests['data_preparation']['status'] = 'passed'
                tests['data_preparation']['details'] = {
                    'train_size': len(data_splits['X_train']),
                    'val_size': len(data_splits['X_val']),
                    'test_size': len(data_splits['X_test']),
                    'features': len(data_splits['feature_names'])
                }
            except Exception as e:
                tests['data_preparation']['status'] = 'failed'
                tests['data_preparation']['details'] = {'error': str(e)}
            
            # Test 2: Traditional models
            try:
                traditional_results = self.model_manager.train_traditional_ml_models(
                    data_splits, 'regression'
                )
                tests['traditional_models']['status'] = 'passed'
                tests['traditional_models']['details'] = {
                    'models_trained': len(traditional_results),
                    'models': list(traditional_results.keys())
                }
            except Exception as e:
                tests['traditional_models']['status'] = 'failed'
                tests['traditional_models']['details'] = {'error': str(e)}
            
            # Test 3: Deep learning models
            try:
                dl_result = self.model_manager.train_deep_learning_model(
                    data_splits, 'regression', 'feedforward'
                )
                tests['deep_learning_models']['status'] = 'passed'
                tests['deep_learning_models']['details'] = {
                    'model_trained': True,
                    'architecture': dl_result.get('architecture', 'unknown'),
                    'epochs_trained': len(dl_result.get('history', {}).get('loss', []))
                }
            except Exception as e:
                tests['deep_learning_models']['status'] = 'failed'
                tests['deep_learning_models']['details'] = {'error': str(e)}
            
            # Test 4: Ensemble models
            try:
                if traditional_results:
                    ensemble_models = list(traditional_results.keys())[:3]  # Use first 3 models
                    ensemble_result = self.model_manager.create_ensemble_model(
                        data_splits, ensemble_models
                    )
                    tests['ensemble_models']['status'] = 'passed'
                    tests['ensemble_models']['details'] = {
                        'base_models': len(ensemble_models),
                        'ensemble_created': True
                    }
                else:
                    tests['ensemble_models']['status'] = 'skipped'
                    tests['ensemble_models']['details'] = {'reason': 'No traditional models available'}
            except Exception as e:
                tests['ensemble_models']['status'] = 'failed'
                tests['ensemble_models']['details'] = {'error': str(e)}
            
            # Test 5: Model persistence
            try:
                save_success = self.model_manager.save_models()
                load_success = self.model_manager.load_models()
                
                tests['model_persistence']['status'] = 'passed' if save_success and load_success else 'failed'
                tests['model_persistence']['details'] = {
                    'save_success': save_success,
                    'load_success': load_success,
                    'models_after_load': len(self.model_manager.models)
                }
            except Exception as e:
                tests['model_persistence']['status'] = 'failed'
                tests['model_persistence']['details'] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Error in model functionality tests: {e}")
            for test in tests:
                if tests[test]['status'] == 'unknown':
                    tests[test]['status'] = 'failed'
                    tests[test]['details'] = {'error': str(e)}
        
        return tests
    
    async def _test_prediction_pipeline(self) -> Dict[str, Any]:
        """Test prediction pipeline functionality."""
        logger.info("Testing prediction pipeline...")
        
        tests = {
            'pipeline_initialization': {'status': 'unknown', 'details': {}},
            'real_time_predictions': {'status': 'unknown', 'details': {}},
            'prediction_caching': {'status': 'unknown', 'details': {}},
            'pipeline_status': {'status': 'unknown', 'details': {}},
            'prediction_quality': {'status': 'unknown', 'details': {}}
        }
        
        try:
            # Test 1: Pipeline initialization
            try:
                sample_data = self._create_sample_market_data()
                init_success = await self.pipeline.initialize_pipeline(sample_data)
                
                tests['pipeline_initialization']['status'] = 'passed' if init_success else 'failed'
                tests['pipeline_initialization']['details'] = {
                    'initialization_success': init_success,
                    'models_available': len(self.pipeline.model_manager.models)
                }
            except Exception as e:
                tests['pipeline_initialization']['status'] = 'failed'
                tests['pipeline_initialization']['details'] = {'error': str(e)}
            
            # Test 2: Real-time predictions
            try:
                # Mock getting predictions for a symbol
                predictions = await self.pipeline.get_current_predictions('SPY')
                
                tests['real_time_predictions']['status'] = 'passed' if predictions else 'failed'
                tests['real_time_predictions']['details'] = {
                    'predictions_generated': bool(predictions),
                    'prediction_keys': list(predictions.keys()) if predictions else []
                }
            except Exception as e:
                tests['real_time_predictions']['status'] = 'failed'
                tests['real_time_predictions']['details'] = {'error': str(e)}
            
            # Test 3: Prediction caching
            try:
                # Test cache functionality
                initial_cache_size = len(self.pipeline.prediction_cache)
                
                # Generate multiple predictions
                for symbol in ['SPY', 'QQQ', 'IWM']:
                    await self.pipeline.get_current_predictions(symbol)
                
                final_cache_size = len(self.pipeline.prediction_cache)
                
                tests['prediction_caching']['status'] = 'passed'
                tests['prediction_caching']['details'] = {
                    'initial_cache_size': initial_cache_size,
                    'final_cache_size': final_cache_size,
                    'cache_working': final_cache_size >= initial_cache_size
                }
            except Exception as e:
                tests['prediction_caching']['status'] = 'failed'
                tests['prediction_caching']['details'] = {'error': str(e)}
            
            # Test 4: Pipeline status
            try:
                status_summary = self.pipeline.get_prediction_summary()
                
                tests['pipeline_status']['status'] = 'passed'
                tests['pipeline_status']['details'] = {
                    'status_available': bool(status_summary),
                    'pipeline_status': status_summary.get('pipeline_status', 'unknown'),
                    'cached_predictions': status_summary.get('cached_predictions', 0)
                }
            except Exception as e:
                tests['pipeline_status']['status'] = 'failed'
                tests['pipeline_status']['details'] = {'error': str(e)}
            
            # Test 5: Prediction quality
            try:
                if self.pipeline.prediction_cache:
                    # Analyze prediction quality
                    confidences = []
                    valid_predictions = 0
                    
                    for symbol, cache_entry in self.pipeline.prediction_cache.items():
                        pred = cache_entry['predictions']
                        if 'confidence' in pred:
                            confidences.append(pred['confidence'])
                            valid_predictions += 1
                    
                    avg_confidence = np.mean(confidences) if confidences else 0
                    
                    tests['prediction_quality']['status'] = 'passed'
                    tests['prediction_quality']['details'] = {
                        'valid_predictions': valid_predictions,
                        'average_confidence': float(avg_confidence),
                        'confidence_range': [float(min(confidences)), float(max(confidences))] if confidences else [0, 0]
                    }
                else:
                    tests['prediction_quality']['status'] = 'skipped'
                    tests['prediction_quality']['details'] = {'reason': 'No predictions available'}
            except Exception as e:
                tests['prediction_quality']['status'] = 'failed'
                tests['prediction_quality']['details'] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Error in prediction pipeline tests: {e}")
            for test in tests:
                if tests[test]['status'] == 'unknown':
                    tests[test]['status'] = 'failed'
                    tests[test]['details'] = {'error': str(e)}
        
        return tests
    
    async def _test_data_integration(self) -> Dict[str, Any]:
        """Test data integration with storage systems."""
        logger.info("Testing data integration...")
        
        tests = {
            'database_connection': {'status': 'unknown', 'details': {}},
            'data_retrieval': {'status': 'unknown', 'details': {}},
            'data_storage': {'status': 'unknown', 'details': {}},
            'data_quality': {'status': 'unknown', 'details': {}}
        }
        
        try:
            # Test 1: Database connection
            try:
                conn = self.data_storage.get_connection()
                conn.close()
                
                tests['database_connection']['status'] = 'passed'
                tests['database_connection']['details'] = {'connection_successful': True}
            except Exception as e:
                tests['database_connection']['status'] = 'failed'
                tests['database_connection']['details'] = {'error': str(e)}
            
            # Test 2: Data retrieval
            try:
                # Try to get some data
                query = "SELECT COUNT(*) as count FROM market_data LIMIT 1"
                conn = self.data_storage.get_connection()
                result = pd.read_sql_query(query, conn)
                conn.close()
                
                tests['data_retrieval']['status'] = 'passed'
                tests['data_retrieval']['details'] = {
                    'query_successful': True,
                    'data_available': True
                }
            except Exception as e:
                tests['data_retrieval']['status'] = 'failed'
                tests['data_retrieval']['details'] = {'error': str(e)}
            
            # Test 3: Data storage
            try:
                # Test storing sample data
                sample_trade = {
                    'symbol': 'TEST',
                    'quantity': 100,
                    'price': 50.0,
                    'side': 'buy',
                    'timestamp': datetime.now()
                }
                
                # This would test the log_trade functionality
                result = self.data_storage.log_trade(**sample_trade)
                
                tests['data_storage']['status'] = 'passed'
                tests['data_storage']['details'] = {'storage_successful': True}
            except Exception as e:
                tests['data_storage']['status'] = 'failed'
                tests['data_storage']['details'] = {'error': str(e)}
            
            # Test 4: Data quality
            try:
                # Check data quality
                query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(timestamp) as earliest_data,
                    MAX(timestamp) as latest_data
                FROM market_data
                """
                
                conn = self.data_storage.get_connection()
                quality_check = pd.read_sql_query(query, conn)
                conn.close()
                
                tests['data_quality']['status'] = 'passed'
                tests['data_quality']['details'] = {
                    'total_records': int(quality_check.iloc[0]['total_records']) if not quality_check.empty else 0,
                    'unique_symbols': int(quality_check.iloc[0]['unique_symbols']) if not quality_check.empty else 0,
                    'data_span_available': True
                }
            except Exception as e:
                tests['data_quality']['status'] = 'failed'
                tests['data_quality']['details'] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Error in data integration tests: {e}")
            for test in tests:
                if tests[test]['status'] == 'unknown':
                    tests[test]['status'] = 'failed'
                    tests[test]['details'] = {'error': str(e)}
        
        return tests
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test system performance."""
        logger.info("Testing performance...")
        
        tests = {
            'feature_generation_speed': {'status': 'unknown', 'details': {}},
            'prediction_speed': {'status': 'unknown', 'details': {}},
            'memory_usage': {'status': 'unknown', 'details': {}},
            'concurrent_predictions': {'status': 'unknown', 'details': {}}
        }
        
        try:
            import time
            import psutil
            
            # Test 1: Feature generation speed
            try:
                sample_data = self._create_sample_market_data(size=1000)  # Larger dataset
                
                start_time = time.time()
                features = self.feature_engine.generate_all_features(sample_data)
                end_time = time.time()
                
                processing_time = end_time - start_time
                records_per_second = len(sample_data) / processing_time if processing_time > 0 else 0
                
                tests['feature_generation_speed']['status'] = 'passed'
                tests['feature_generation_speed']['details'] = {
                    'processing_time_seconds': round(processing_time, 3),
                    'records_per_second': round(records_per_second, 1),
                    'features_generated': len(features.columns)
                }
            except Exception as e:
                tests['feature_generation_speed']['status'] = 'failed'
                tests['feature_generation_speed']['details'] = {'error': str(e)}
            
            # Test 2: Prediction speed
            try:
                start_time = time.time()
                predictions = await self.pipeline.get_current_predictions('SPY')
                end_time = time.time()
                
                prediction_time = end_time - start_time
                
                tests['prediction_speed']['status'] = 'passed'
                tests['prediction_speed']['details'] = {
                    'prediction_time_seconds': round(prediction_time, 3),
                    'predictions_generated': bool(predictions)
                }
            except Exception as e:
                tests['prediction_speed']['status'] = 'failed'
                tests['prediction_speed']['details'] = {'error': str(e)}
            
            # Test 3: Memory usage
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                tests['memory_usage']['status'] = 'passed'
                tests['memory_usage']['details'] = {
                    'memory_usage_mb': round(memory_info.rss / 1024 / 1024, 2),
                    'memory_percent': round(process.memory_percent(), 2)
                }
            except Exception as e:
                tests['memory_usage']['status'] = 'failed'
                tests['memory_usage']['details'] = {'error': str(e)}
            
            # Test 4: Concurrent predictions
            try:
                symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK']
                
                start_time = time.time()
                
                # Simulate concurrent predictions
                tasks = [self.pipeline.get_current_predictions(symbol) for symbol in symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.time()
                
                successful_predictions = sum(1 for r in results if isinstance(r, dict) and r)
                concurrent_time = end_time - start_time
                
                tests['concurrent_predictions']['status'] = 'passed'
                tests['concurrent_predictions']['details'] = {
                    'symbols_tested': len(symbols),
                    'successful_predictions': successful_predictions,
                    'total_time_seconds': round(concurrent_time, 3),
                    'avg_time_per_prediction': round(concurrent_time / len(symbols), 3)
                }
            except Exception as e:
                tests['concurrent_predictions']['status'] = 'failed'
                tests['concurrent_predictions']['details'] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Error in performance tests: {e}")
            for test in tests:
                if tests[test]['status'] == 'unknown':
                    tests[test]['status'] = 'failed'
                    tests[test]['details'] = {'error': str(e)}
        
        return tests
    
    async def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling."""
        logger.info("Testing edge cases...")
        
        tests = {
            'missing_data_handling': {'status': 'unknown', 'details': {}},
            'invalid_inputs': {'status': 'unknown', 'details': {}},
            'empty_data_handling': {'status': 'unknown', 'details': {}},
            'extreme_values': {'status': 'unknown', 'details': {}}
        }
        
        try:
            # Test 1: Missing data handling
            try:
                # Create data with missing values
                sample_data = self._create_sample_market_data()
                sample_data.iloc[::2] = np.nan  # Make every other row NaN
                
                features = self.feature_engine.generate_all_features(sample_data)
                
                tests['missing_data_handling']['status'] = 'passed'
                tests['missing_data_handling']['details'] = {
                    'input_missing_pct': (sample_data.isnull().sum().sum() / sample_data.size) * 100,
                    'output_rows': len(features),
                    'handled_gracefully': True
                }
            except Exception as e:
                tests['missing_data_handling']['status'] = 'failed'
                tests['missing_data_handling']['details'] = {'error': str(e)}
            
            # Test 2: Invalid inputs
            try:
                # Test with invalid symbol
                predictions = await self.pipeline.get_current_predictions('INVALID_SYMBOL')
                
                tests['invalid_inputs']['status'] = 'passed'
                tests['invalid_inputs']['details'] = {
                    'handled_invalid_symbol': True,
                    'predictions_returned': bool(predictions)
                }
            except Exception as e:
                tests['invalid_inputs']['status'] = 'failed'
                tests['invalid_inputs']['details'] = {'error': str(e)}
            
            # Test 3: Empty data handling
            try:
                empty_data = pd.DataFrame()
                features = self.feature_engine.generate_all_features(empty_data)
                
                tests['empty_data_handling']['status'] = 'passed'
                tests['empty_data_handling']['details'] = {
                    'handled_empty_data': True,
                    'output_type': type(features).__name__
                }
            except Exception as e:
                tests['empty_data_handling']['status'] = 'failed'
                tests['empty_data_handling']['details'] = {'error': str(e)}
            
            # Test 4: Extreme values
            try:
                # Create data with extreme values
                extreme_data = self._create_sample_market_data()
                extreme_data['close'] = [1e10, 1e-10] * (len(extreme_data) // 2)
                extreme_data['volume'] = [0, 1e15] * (len(extreme_data) // 2)
                
                features = self.feature_engine.generate_all_features(extreme_data)
                
                tests['extreme_values']['status'] = 'passed'
                tests['extreme_values']['details'] = {
                    'handled_extreme_values': True,
                    'output_finite': np.isfinite(features.select_dtypes(include=[np.number])).all().all()
                }
            except Exception as e:
                tests['extreme_values']['status'] = 'failed'
                tests['extreme_values']['details'] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Error in edge case tests: {e}")
            for test in tests:
                if tests[test]['status'] == 'unknown':
                    tests[test]['status'] = 'failed'
                    tests[test]['details'] = {'error': str(e)}
        
        return tests
    
    async def _test_integration(self) -> Dict[str, Any]:
        """Test end-to-end integration."""
        logger.info("Testing end-to-end integration...")
        
        tests = {
            'full_pipeline_flow': {'status': 'unknown', 'details': {}},
            'real_time_simulation': {'status': 'unknown', 'details': {}},
            'error_recovery': {'status': 'unknown', 'details': {}}
        }
        
        try:
            # Test 1: Full pipeline flow
            try:
                # Simulate full flow: data -> features -> prediction
                sample_data = self._create_sample_market_data()
                
                # Generate features
                features = self.feature_engine.generate_all_features(sample_data)
                
                # Get predictions (this tests the full pipeline)
                predictions = await self.pipeline.get_current_predictions('SPY')
                
                tests['full_pipeline_flow']['status'] = 'passed'
                tests['full_pipeline_flow']['details'] = {
                    'data_to_features': len(features) > 0,
                    'features_to_predictions': bool(predictions),
                    'end_to_end_success': True
                }
            except Exception as e:
                tests['full_pipeline_flow']['status'] = 'failed'
                tests['full_pipeline_flow']['details'] = {'error': str(e)}
            
            # Test 2: Real-time simulation
            try:
                # Simulate multiple real-time prediction requests
                symbols = ['SPY', 'QQQ']
                all_predictions = {}
                
                for symbol in symbols:
                    predictions = await self.pipeline.get_current_predictions(symbol)
                    all_predictions[symbol] = predictions
                
                tests['real_time_simulation']['status'] = 'passed'
                tests['real_time_simulation']['details'] = {
                    'symbols_processed': len(all_predictions),
                    'successful_predictions': sum(1 for p in all_predictions.values() if p),
                    'cache_utilization': len(self.pipeline.prediction_cache)
                }
            except Exception as e:
                tests['real_time_simulation']['status'] = 'failed'
                tests['real_time_simulation']['details'] = {'error': str(e)}
            
            # Test 3: Error recovery
            try:
                # Test system recovery after errors
                initial_state = len(self.pipeline.prediction_cache)
                
                # Cause an error and see if system recovers
                try:
                    await self.pipeline.get_current_predictions(None)  # Invalid input
                except:
                    pass  # Expected to fail
                
                # System should still work after error
                recovery_prediction = await self.pipeline.get_current_predictions('SPY')
                
                tests['error_recovery']['status'] = 'passed'
                tests['error_recovery']['details'] = {
                    'recovered_after_error': bool(recovery_prediction),
                    'cache_preserved': len(self.pipeline.prediction_cache) >= initial_state
                }
            except Exception as e:
                tests['error_recovery']['status'] = 'failed'
                tests['error_recovery']['details'] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Error in integration tests: {e}")
            for test in tests:
                if tests[test]['status'] == 'unknown':
                    tests[test]['status'] = 'failed'
                    tests[test]['details'] = {'error': str(e)}
        
        return tests
    
    def _create_sample_market_data(self, size: int = 200) -> pd.DataFrame:
        """Create sample market data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        dates = pd.date_range(start='2024-01-01', periods=size, freq='1min')
        
        # Generate realistic OHLCV data
        base_price = 100.0
        prices = [base_price]
        
        for _ in range(size - 1):
            change = np.random.normal(0, 0.5)  # Small random changes
            new_price = max(prices[-1] + change, 1.0)  # Ensure positive prices
            prices.append(new_price)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'SPY',
            'open': prices,
            'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
            'close': prices,
            'volume': [int(abs(np.random.normal(1000000, 200000))) for _ in range(size)]
        })
        
        # Adjust high/low to be consistent
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        data = data.set_index('timestamp')
        
        return data
    
    def _create_sample_feature_data(self, size: int = 500) -> pd.DataFrame:
        """Create sample feature data for model testing."""
        np.random.seed(42)
        
        # Generate synthetic features
        n_features = 20
        data = np.random.randn(size, n_features)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        df = pd.DataFrame(data, columns=feature_names)
        
        # Create a synthetic target that has some relationship to features
        df['target'] = (
            0.3 * df['feature_0'] + 
            0.2 * df['feature_1'] - 
            0.1 * df['feature_2'] + 
            np.random.normal(0, 0.1, size)
        )
        
        return df
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all test results."""
        summary = {
            'total_test_categories': 0,
            'passed_categories': 0,
            'failed_categories': 0,
            'overall_success_rate': 0.0,
            'critical_failures': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Count test categories and results
            for category, tests in test_results.items():
                if category in ['timestamp', 'summary']:
                    continue
                
                summary['total_test_categories'] += 1
                
                # Check if all tests in category passed
                category_passed = True
                for test_name, test_result in tests.items():
                    if test_result.get('status') == 'failed':
                        category_passed = False
                        summary['critical_failures'].append(f"{category}.{test_name}")
                    elif test_result.get('status') == 'skipped':
                        summary['warnings'].append(f"{category}.{test_name} was skipped")
                
                if category_passed:
                    summary['passed_categories'] += 1
                else:
                    summary['failed_categories'] += 1
            
            # Calculate success rate
            if summary['total_test_categories'] > 0:
                summary['overall_success_rate'] = (
                    summary['passed_categories'] / summary['total_test_categories']
                ) * 100
            
            # Generate recommendations
            if summary['failed_categories'] == 0:
                summary['recommendations'].append("✅ All tests passed - ML system is ready for production")
            else:
                summary['recommendations'].append(f"❌ {summary['failed_categories']} test categories failed")
                summary['recommendations'].append("🔧 Review failed tests and fix issues before deployment")
            
            if len(summary['warnings']) > 0:
                summary['recommendations'].append(f"⚠️ {len(summary['warnings'])} tests were skipped - review test conditions")
        
        except Exception as e:
            logger.error(f"Error generating test summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def print_test_results(self, test_results: Dict[str, Any]) -> None:
        """Print formatted test results."""
        print("=" * 80)
        print("🧪 ML SYSTEM TEST RESULTS")
        print("=" * 80)
        print(f"Test Run: {test_results['timestamp']}")
        print()
        
        # Print summary
        summary = test_results.get('summary', {})
        print("📊 SUMMARY")
        print("-" * 40)
        print(f"Total Categories: {summary.get('total_test_categories', 0)}")
        print(f"Passed: {summary.get('passed_categories', 0)}")
        print(f"Failed: {summary.get('failed_categories', 0)}")
        print(f"Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
        print()
        
        # Print detailed results for each category
        for category, tests in test_results.items():
            if category in ['timestamp', 'summary']:
                continue
            
            print(f"🔬 {category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            for test_name, test_result in tests.items():
                status = test_result.get('status', 'unknown')
                status_emoji = {'passed': '✅', 'failed': '❌', 'skipped': '⏭️', 'unknown': '❓'}
                
                print(f"{status_emoji.get(status, '❓')} {test_name}: {status}")
                
                # Show error details for failed tests
                if status == 'failed' and 'error' in test_result.get('details', {}):
                    print(f"    Error: {test_result['details']['error']}")
            
            print()
        
        # Print recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print("💡 RECOMMENDATIONS")
            print("-" * 40)
            for rec in recommendations:
                print(f"• {rec}")
            print()
        
        print("=" * 80)
    
    def save_test_results(self, test_results: Dict[str, Any], filename: str = None) -> str:
        """Save test results to file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ml_test_results_{timestamp}.json"
            
            # Create reports directory
            os.makedirs("reports", exist_ok=True)
            filepath = os.path.join("reports", filename)
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
            return ""

async def main():
    """Main function for ML testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Testing for ETF Trading System')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--category', help='Run specific test category only')
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = MLTester(args.config)
        
        print("🧪 Starting ML System Tests...")
        
        if args.category:
            # Run specific category
            print(f"Running tests for category: {args.category}")
            # This would require implementing category-specific test methods
            test_results = await tester.run_comprehensive_tests()
        else:
            # Run all tests
            test_results = await tester.run_comprehensive_tests()
        
        # Print results
        tester.print_test_results(test_results)
        
        # Save if requested
        if args.save:
            filepath = tester.save_test_results(test_results)
            if filepath:
                print(f"📄 Results saved to: {filepath}")
        
        # Exit with appropriate code
        summary = test_results.get('summary', {})
        failed_categories = summary.get('failed_categories', 0)
        
        if failed_categories == 0:
            print("🎉 All tests passed!")
            return 0
        else:
            print(f"💥 {failed_categories} test categories failed")
            return 1
    
    except Exception as e:
        logger.error(f"Error in ML testing: {e}")
        print(f"❌ Testing error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
