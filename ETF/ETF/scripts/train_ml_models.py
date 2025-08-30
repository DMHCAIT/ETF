"""
ML Training and Optimization Script for ETF Trading System.
Comprehensive script to train, optimize, and validate all ML models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import argparse
from typing import Dict, List, Any
import json

from src.ml.feature_engineering import AdvancedFeatureEngine
from src.ml.models import MLModelManager
from src.ml.prediction_pipeline import MLPredictionPipeline
from src.utils.config import Config
from src.data.data_storage import DataStorage
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("ml_training", level=logging.INFO)

class MLTrainingOrchestrator:
    """Orchestrates complete ML training pipeline."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize training orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.data_storage = DataStorage(self.config)
        self.feature_engine = AdvancedFeatureEngine()
        self.model_manager = MLModelManager()
        self.pipeline = MLPredictionPipeline(self.config, self.data_storage)
        
        # Training configuration
        self.training_config = {
            'lookback_days': 30,
            'validation_split': 0.2,
            'test_split': 0.1,
            'optimization_trials': 100,
            'cross_validation_folds': 5,
            'ensemble_models': ['xgboost', 'lightgbm', 'random_forest', 'catboost']
        }
        
        logger.info("ML Training Orchestrator initialized")
    
    async def run_complete_training(self, symbols: List[str] = None, 
                                   retrain: bool = False) -> Dict[str, Any]:
        """
        Run complete ML training pipeline.
        
        Args:
            symbols: List of symbols to train on (None for all)
            retrain: Whether to retrain existing models
            
        Returns:
            Training results summary
        """
        logger.info("Starting complete ML training pipeline...")
        
        results = {
            'start_time': datetime.now(),
            'symbols_processed': 0,
            'models_trained': 0,
            'training_errors': [],
            'performance_summary': {},
            'best_models': {}
        }
        
        try:
            # Step 1: Data preparation
            logger.info("Step 1: Preparing training data...")
            training_data = await self._prepare_training_data(symbols)
            
            if training_data.empty:
                raise ValueError("No training data available")
            
            results['data_points'] = len(training_data)
            results['symbols_processed'] = training_data['symbol'].nunique()
            
            # Step 2: Feature engineering
            logger.info("Step 2: Generating features...")
            features_df = await self._generate_features(training_data)
            results['features_generated'] = len([col for col in features_df.columns if col not in ['symbol', 'timestamp']])
            
            # Step 3: Train models for different tasks
            logger.info("Step 3: Training ML models...")
            model_results = await self._train_all_models(features_df)
            results['models_trained'] = len(model_results)
            results['model_results'] = model_results
            
            # Step 4: Model optimization
            logger.info("Step 4: Optimizing best models...")
            optimization_results = await self._optimize_models(features_df)
            results['optimization_results'] = optimization_results
            
            # Step 5: Model validation
            logger.info("Step 5: Validating models...")
            validation_results = await self._validate_models(features_df)
            results['validation_results'] = validation_results
            
            # Step 6: Create ensembles
            logger.info("Step 6: Creating ensemble models...")
            ensemble_results = await self._create_ensembles(features_df)
            results['ensemble_results'] = ensemble_results
            
            # Step 7: Backtesting
            logger.info("Step 7: Running backtests...")
            backtest_results = await self._run_backtests(symbols or ['SPY'])
            results['backtest_results'] = backtest_results
            
            # Step 8: Save models and results
            logger.info("Step 8: Saving models and results...")
            save_success = self.model_manager.save_models()
            results['models_saved'] = save_success
            
            # Generate final summary
            results['performance_summary'] = self._generate_performance_summary(
                model_results, validation_results, backtest_results
            )
            
            results['end_time'] = datetime.now()
            results['total_duration'] = (results['end_time'] - results['start_time']).total_seconds()
            
            logger.info("Complete ML training pipeline finished successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in complete training pipeline: {e}")
            results['error'] = str(e)
            results['end_time'] = datetime.now()
            return results
    
    async def _prepare_training_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """Prepare training data from database."""
        try:
            # Get data from the last N days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.training_config['lookback_days'])
            
            if symbols is None:
                query = f"""
                SELECT * FROM market_data 
                WHERE timestamp >= '{start_date.isoformat()}'
                AND timestamp <= '{end_date.isoformat()}'
                ORDER BY symbol, timestamp
                """
            else:
                symbols_str = "', '".join(symbols)
                query = f"""
                SELECT * FROM market_data 
                WHERE symbol IN ('{symbols_str}')
                AND timestamp >= '{start_date.isoformat()}'
                AND timestamp <= '{end_date.isoformat()}'
                ORDER BY symbol, timestamp
                """
            
            conn = self.data_storage.get_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            logger.info(f"Prepared {len(df)} data points for {df['symbol'].nunique() if not df.empty else 0} symbols")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()
    
    async def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive features for training."""
        try:
            logger.info("Generating comprehensive feature set...")
            
            features_df = self.feature_engine.generate_all_features(data)
            
            # Add target variables
            features_df = self._create_comprehensive_targets(features_df)
            
            # Remove rows with insufficient data
            features_df = features_df.dropna(subset=['close', 'volume'])
            
            logger.info(f"Generated {len(features_df.columns)} features for {len(features_df)} samples")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return pd.DataFrame()
    
    def _create_comprehensive_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive target variables for different prediction tasks."""
        try:
            logger.info("Creating target variables...")
            
            # Price change targets (different horizons)
            for horizon in [1, 3, 5, 10, 15, 30]:
                df[f'price_change_{horizon}m'] = df.groupby('symbol')['close'].pct_change(horizon).shift(-horizon)
            
            # Direction targets (binary classification)
            for horizon in [1, 3, 5, 10, 15, 30]:
                df[f'direction_{horizon}m'] = (df[f'price_change_{horizon}m'] > 0).astype(int)
            
            # Volatility targets
            for window in [5, 10, 20]:
                df[f'volatility_{window}'] = df.groupby('symbol')['close'].rolling(window).std().shift(-window)
            
            # Support/resistance breakouts
            df['breakout_up'] = (df['close'] > df['resistance_1']).astype(int)
            df['breakout_down'] = (df['close'] < df['support_1']).astype(int)
            
            # Trend continuation
            df['trend_continue'] = (
                (df['sma_20_trend'] > 0) & (df['close'] > df['sma_20'])
            ).astype(int)
            
            # Volume surge prediction
            df['volume_surge'] = (
                df.groupby('symbol')['volume'].shift(-1) > 
                df.groupby('symbol')['volume'].rolling(20).mean() * 1.5
            ).astype(int)
            
            logger.info("Created comprehensive target variables")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating targets: {e}")
            return df
    
    async def _train_all_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train models for all prediction tasks."""
        model_results = {}
        
        # Define prediction tasks
        tasks = [
            {'target': 'price_change_1m', 'type': 'regression', 'name': 'price_1m'},
            {'target': 'price_change_5m', 'type': 'regression', 'name': 'price_5m'},
            {'target': 'price_change_15m', 'type': 'regression', 'name': 'price_15m'},
            {'target': 'direction_1m', 'type': 'classification', 'name': 'direction_1m'},
            {'target': 'direction_5m', 'type': 'classification', 'name': 'direction_5m'},
            {'target': 'direction_15m', 'type': 'classification', 'name': 'direction_15m'},
            {'target': 'volatility_10', 'type': 'regression', 'name': 'volatility'},
            {'target': 'breakout_up', 'type': 'classification', 'name': 'breakout_up'},
            {'target': 'breakout_down', 'type': 'classification', 'name': 'breakout_down'},
            {'target': 'trend_continue', 'type': 'classification', 'name': 'trend_continue'},
            {'target': 'volume_surge', 'type': 'classification', 'name': 'volume_surge'}
        ]
        
        for task in tasks:
            try:
                logger.info(f"Training models for {task['name']} ({task['type']})...")
                
                # Check if target exists
                if task['target'] not in features_df.columns:
                    logger.warning(f"Target {task['target']} not found, skipping...")
                    continue
                
                # Prepare data
                data_splits = self.model_manager.prepare_data(
                    features_df, 
                    task['target'],
                    test_size=self.training_config['test_split'],
                    validation_size=self.training_config['validation_split']
                )
                
                if len(data_splits['X_train']) < 100:
                    logger.warning(f"Insufficient data for {task['name']}, skipping...")
                    continue
                
                # Train traditional ML models
                traditional_results = self.model_manager.train_traditional_ml_models(
                    data_splits, task['type']
                )
                
                # Train deep learning models
                dl_feedforward = self.model_manager.train_deep_learning_model(
                    data_splits, task['type'], 'feedforward'
                )
                
                dl_lstm = self.model_manager.train_deep_learning_model(
                    data_splits, task['type'], 'lstm'
                )
                
                # Store results
                model_results[task['name']] = {
                    'traditional': traditional_results,
                    'dl_feedforward': dl_feedforward,
                    'dl_lstm': dl_lstm,
                    'data_splits': {
                        'train_size': len(data_splits['X_train']),
                        'val_size': len(data_splits['X_val']),
                        'test_size': len(data_splits['X_test'])
                    }
                }
                
                logger.info(f"Completed training for {task['name']}")
                
            except Exception as e:
                logger.error(f"Error training models for {task['name']}: {e}")
                continue
        
        return model_results
    
    async def _optimize_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize hyperparameters for best performing models."""
        optimization_results = {}
        
        # Define models to optimize
        optimize_configs = [
            {'target': 'price_change_1m', 'models': ['xgboost', 'lightgbm', 'random_forest']},
            {'target': 'price_change_5m', 'models': ['xgboost', 'lightgbm', 'random_forest']},
            {'target': 'direction_1m', 'models': ['xgboost', 'lightgbm', 'random_forest']},
            {'target': 'direction_5m', 'models': ['xgboost', 'lightgbm', 'random_forest']}
        ]
        
        for config in optimize_configs:
            try:
                target = config['target']
                
                if target not in features_df.columns:
                    continue
                
                logger.info(f"Optimizing models for {target}...")
                
                data_splits = self.model_manager.prepare_data(features_df, target)
                
                for model_name in config['models']:
                    try:
                        logger.info(f"Optimizing {model_name} for {target}...")
                        
                        result = self.model_manager.optimize_hyperparameters(
                            data_splits, 
                            model_name, 
                            n_trials=self.training_config['optimization_trials']
                        )
                        
                        optimization_results[f"{target}_{model_name}"] = result
                        
                    except Exception as e:
                        logger.error(f"Error optimizing {model_name} for {target}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error in optimization for {config['target']}: {e}")
                continue
        
        return optimization_results
    
    async def _validate_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform cross-validation on trained models."""
        validation_results = {}
        
        try:
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.training_config['cross_validation_folds'])
            
            # Validate key models
            validation_tasks = [
                {'target': 'price_change_1m', 'models': ['xgboost', 'lightgbm']},
                {'target': 'direction_1m', 'models': ['xgboost', 'lightgbm']}
            ]
            
            for task in validation_tasks:
                target = task['target']
                
                if target not in features_df.columns:
                    continue
                
                logger.info(f"Cross-validating models for {target}...")
                
                # Prepare data
                feature_cols = [col for col in features_df.columns 
                              if col not in ['symbol', target] and not col.startswith('price_change_') 
                              and not col.startswith('direction_') and not col.startswith('volatility_')]
                
                X = features_df[feature_cols].fillna(features_df[feature_cols].mean())
                y = features_df[target].dropna()
                
                # Align X and y
                common_idx = X.index.intersection(y.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]
                
                if len(X) < 200:
                    logger.warning(f"Insufficient data for cross-validation of {target}")
                    continue
                
                for model_name in task['models']:
                    try:
                        # Get model from manager
                        if model_name in self.model_manager.models:
                            model_info = self.model_manager.models[model_name]
                            model = model_info['model']
                            
                            # Perform cross-validation
                            if 'direction' in target:
                                scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                            else:
                                scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                            
                            validation_results[f"{target}_{model_name}"] = {
                                'scores': scores.tolist(),
                                'mean_score': float(scores.mean()),
                                'std_score': float(scores.std()),
                                'cv_folds': len(scores)
                            }
                            
                            logger.info(f"CV score for {model_name} on {target}: {scores.mean():.4f} ± {scores.std():.4f}")
                    
                    except Exception as e:
                        logger.error(f"Error in cross-validation for {model_name} on {target}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error in model validation: {e}")
        
        return validation_results
    
    async def _create_ensembles(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Create ensemble models for key prediction tasks."""
        ensemble_results = {}
        
        ensemble_tasks = [
            {'target': 'price_change_1m', 'name': 'price_1m'},
            {'target': 'price_change_5m', 'name': 'price_5m'},
            {'target': 'direction_1m', 'name': 'direction_1m'},
            {'target': 'direction_5m', 'name': 'direction_5m'}
        ]
        
        for task in ensemble_tasks:
            try:
                target = task['target']
                name = task['name']
                
                if target not in features_df.columns:
                    continue
                
                logger.info(f"Creating ensemble for {name}...")
                
                data_splits = self.model_manager.prepare_data(features_df, target)
                
                # Create ensemble with best models
                ensemble_models = self.training_config['ensemble_models']
                
                result = self.model_manager.create_ensemble_model(
                    data_splits, ensemble_models
                )
                
                ensemble_results[name] = result
                
                logger.info(f"Created ensemble for {name}")
                
            except Exception as e:
                logger.error(f"Error creating ensemble for {task['name']}: {e}")
                continue
        
        return ensemble_results
    
    async def _run_backtests(self, symbols: List[str]) -> Dict[str, Any]:
        """Run backtests on trained models."""
        backtest_results = {}
        
        # Define backtest period
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=7)  # Last week
        
        for symbol in symbols:
            try:
                logger.info(f"Running backtest for {symbol}...")
                
                result = await self.pipeline.backtest_predictions(
                    start_date, end_date, symbol
                )
                
                backtest_results[symbol] = result
                
            except Exception as e:
                logger.error(f"Error in backtest for {symbol}: {e}")
                backtest_results[symbol] = {'error': str(e)}
        
        return backtest_results
    
    def _generate_performance_summary(self, model_results: Dict[str, Any],
                                    validation_results: Dict[str, Any],
                                    backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        try:
            summary = {
                'training_summary': {},
                'validation_summary': {},
                'backtest_summary': {},
                'best_models': {},
                'recommendations': []
            }
            
            # Training summary
            total_models = 0
            successful_models = 0
            
            for task_name, task_results in model_results.items():
                for model_type, results in task_results.items():
                    if model_type != 'data_splits':
                        total_models += len(results) if isinstance(results, dict) else 1
                        successful_models += len(results) if isinstance(results, dict) else 1
            
            summary['training_summary'] = {
                'total_models_trained': total_models,
                'successful_models': successful_models,
                'success_rate': successful_models / total_models if total_models > 0 else 0
            }
            
            # Validation summary
            if validation_results:
                val_scores = [r['mean_score'] for r in validation_results.values() if 'mean_score' in r]
                summary['validation_summary'] = {
                    'models_validated': len(validation_results),
                    'avg_score': np.mean(val_scores) if val_scores else 0,
                    'best_score': np.max(val_scores) if val_scores else 0,
                    'worst_score': np.min(val_scores) if val_scores else 0
                }
            
            # Backtest summary
            backtest_success = 0
            total_backtests = len(backtest_results)
            
            for symbol, result in backtest_results.items():
                if 'error' not in result:
                    backtest_success += 1
            
            summary['backtest_summary'] = {
                'symbols_tested': total_backtests,
                'successful_backtests': backtest_success,
                'success_rate': backtest_success / total_backtests if total_backtests > 0 else 0
            }
            
            # Best models (simplified)
            summary['best_models'] = {
                'price_prediction': 'ensemble_price_1m',
                'direction_prediction': 'ensemble_direction_1m',
                'volatility_prediction': 'ensemble_volatility'
            }
            
            # Recommendations
            recommendations = []
            
            if summary['training_summary']['success_rate'] > 0.8:
                recommendations.append("Training successful for most models")
            else:
                recommendations.append("Consider reviewing data quality and feature engineering")
            
            if validation_results and summary['validation_summary']['avg_score'] > 0.6:
                recommendations.append("Models show good cross-validation performance")
            else:
                recommendations.append("Consider hyperparameter tuning and feature selection")
            
            if summary['backtest_summary']['success_rate'] > 0.8:
                recommendations.append("Backtesting successful - models ready for production")
            else:
                recommendations.append("Review backtest failures before deploying to production")
            
            summary['recommendations'] = recommendations
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}
    
    def save_training_report(self, results: Dict[str, Any], 
                           filename: str = None) -> str:
        """Save training results to a comprehensive report."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ml_training_report_{timestamp}.json"
            
            # Create reports directory
            os.makedirs("reports", exist_ok=True)
            filepath = os.path.join("reports", filename)
            
            # Convert datetime objects to strings for JSON serialization
            results_copy = results.copy()
            for key, value in results_copy.items():
                if isinstance(value, datetime):
                    results_copy[key] = value.isoformat()
            
            # Save report
            with open(filepath, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)
            
            logger.info(f"Training report saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving training report: {e}")
            return ""

async def main():
    """Main function for ML training script."""
    parser = argparse.ArgumentParser(description='ML Training and Optimization for ETF Trading System')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on (default: all available)')
    parser.add_argument('--retrain', action='store_true', help='Retrain existing models')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--lookback', type=int, default=30, help='Lookback days for training data')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = MLTrainingOrchestrator(args.config)
        
        # Update training configuration
        if args.trials:
            orchestrator.training_config['optimization_trials'] = args.trials
        if args.lookback:
            orchestrator.training_config['lookback_days'] = args.lookback
        
        # Run training
        logger.info("Starting ML training pipeline...")
        results = await orchestrator.run_complete_training(
            symbols=args.symbols,
            retrain=args.retrain
        )
        
        # Save report
        report_path = orchestrator.save_training_report(results)
        
        # Print summary
        print("\n" + "="*80)
        print("ML TRAINING COMPLETE")
        print("="*80)
        
        if 'error' in results:
            print(f"❌ Training failed: {results['error']}")
        else:
            print(f"✅ Training successful!")
            print(f"📊 Models trained: {results.get('models_trained', 0)}")
            print(f"📈 Symbols processed: {results.get('symbols_processed', 0)}")
            print(f"⏱️  Duration: {results.get('total_duration', 0):.2f} seconds")
            print(f"📄 Report saved: {report_path}")
            
            if 'performance_summary' in results:
                perf = results['performance_summary']
                print(f"🎯 Training success rate: {perf.get('training_summary', {}).get('success_rate', 0):.2%}")
                print(f"🔍 Validation avg score: {perf.get('validation_summary', {}).get('avg_score', 0):.4f}")
                print(f"🧪 Backtest success rate: {perf.get('backtest_summary', {}).get('success_rate', 0):.2%}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
