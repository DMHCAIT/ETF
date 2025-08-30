"""
ML Prediction Pipeline for ETF Trading System.
Integrates machine learning models with trading strategy for automated predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import asyncio
import threading
import time

from .feature_engineering import AdvancedFeatureEngine
from .models import MLModelManager
from ..utils.config import Config
from ..data.data_storage import DataStorage

logger = logging.getLogger(__name__)

class MLPredictionPipeline:
    """Automated ML prediction pipeline for trading decisions."""
    
    def __init__(self, config: Config, data_storage: DataStorage):
        """
        Initialize ML prediction pipeline.
        
        Args:
            config: Configuration instance
            data_storage: Data storage instance
        """
        self.config = config
        self.data_storage = data_storage
        
        # Initialize components
        self.feature_engine = AdvancedFeatureEngine()
        self.model_manager = MLModelManager()
        
        # Pipeline state
        self.is_running = False
        self.last_prediction_time = None
        self.prediction_cache = {}
        self.model_performance = {}
        
        # Settings
        self.prediction_interval = 60  # seconds
        self.lookback_periods = 100  # number of periods for feature calculation
        self.ensemble_models = ['xgboost', 'lightgbm', 'random_forest']
        
        logger.info("ML Prediction Pipeline initialized")
    
    async def initialize_pipeline(self, initial_data: pd.DataFrame = None) -> bool:
        """
        Initialize the ML pipeline with data and train initial models.
        
        Args:
            initial_data: Initial market data for training
            
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing ML prediction pipeline...")
            
            # Get initial data if not provided
            if initial_data is None:
                initial_data = await self._get_training_data()
            
            if initial_data.empty:
                logger.error("No data available for ML pipeline initialization")
                return False
            
            # Generate features
            logger.info("Generating features for initial training...")
            features_df = self.feature_engine.generate_all_features(initial_data)
            
            # Create target variables
            features_df = self._create_target_variables(features_df)
            
            # Train initial models
            await self._train_initial_models(features_df)
            
            logger.info("ML prediction pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ML pipeline: {e}")
            return False
    
    async def start_real_time_predictions(self) -> None:
        """Start real-time prediction loop."""
        if self.is_running:
            logger.warning("Prediction pipeline already running")
            return
        
        self.is_running = True
        logger.info("Starting real-time ML predictions...")
        
        while self.is_running:
            try:
                await self._generate_real_time_predictions()
                await asyncio.sleep(self.prediction_interval)
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(10)  # Short pause before retrying
    
    def stop_predictions(self) -> None:
        """Stop real-time prediction loop."""
        self.is_running = False
        logger.info("Stopped real-time ML predictions")
    
    async def get_current_predictions(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ML predictions for a symbol.
        
        Args:
            symbol: ETF symbol
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        try:
            # Check cache first
            if symbol in self.prediction_cache:
                cache_entry = self.prediction_cache[symbol]
                if (datetime.now() - cache_entry['timestamp']).seconds < self.prediction_interval:
                    return cache_entry['predictions']
            
            # Generate fresh predictions
            predictions = await self._generate_symbol_predictions(symbol)
            
            # Update cache
            self.prediction_cache[symbol] = {
                'timestamp': datetime.now(),
                'predictions': predictions
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions for {symbol}: {e}")
            return {}
    
    async def retrain_models(self, symbols: List[str] = None) -> bool:
        """
        Retrain models with latest data.
        
        Args:
            symbols: List of symbols to retrain on (None for all)
            
        Returns:
            True if retraining successful
        """
        try:
            logger.info("Starting model retraining...")
            
            # Get fresh training data
            training_data = await self._get_training_data(symbols)
            
            if training_data.empty:
                logger.error("No data available for retraining")
                return False
            
            # Generate features
            features_df = self.feature_engine.generate_all_features(training_data)
            features_df = self._create_target_variables(features_df)
            
            # Retrain models
            await self._train_initial_models(features_df)
            
            # Update performance metrics
            await self._update_model_performance()
            
            logger.info("Model retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return False
    
    async def _get_training_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """Get training data from data storage."""
        try:
            # Get data from storage
            if symbols is None:
                # Get data for all symbols
                query = """
                SELECT * FROM market_data 
                WHERE timestamp >= datetime('now', '-30 days')
                ORDER BY timestamp DESC
                """
            else:
                symbols_str = "', '".join(symbols)
                query = f"""
                SELECT * FROM market_data 
                WHERE symbol IN ('{symbols_str}')
                AND timestamp >= datetime('now', '-30 days')
                ORDER BY timestamp DESC
                """
            
            # Execute query using data storage
            conn = self.data_storage.get_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML training."""
        try:
            # Price prediction targets
            df['price_change_1m'] = df.groupby('symbol')['close'].pct_change(1).shift(-1)
            df['price_change_5m'] = df.groupby('symbol')['close'].pct_change(5).shift(-5)
            df['price_change_15m'] = df.groupby('symbol')['close'].pct_change(15).shift(-15)
            
            # Direction prediction targets (binary)
            df['direction_1m'] = (df['price_change_1m'] > 0).astype(int)
            df['direction_5m'] = (df['price_change_5m'] > 0).astype(int)
            df['direction_15m'] = (df['price_change_15m'] > 0).astype(int)
            
            # Volatility prediction
            df['volatility_next'] = df.groupby('symbol')['close'].rolling(10).std().shift(-10)
            
            # Support/resistance breakout
            df['breakout_up'] = (df['close'] > df['resistance_1']).astype(int)
            df['breakout_down'] = (df['close'] < df['support_1']).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            return df
    
    async def _train_initial_models(self, features_df: pd.DataFrame) -> None:
        """Train initial set of ML models."""
        try:
            # Prepare datasets for different prediction tasks
            target_configs = [
                {'target': 'price_change_1m', 'type': 'regression', 'name': 'price_1m'},
                {'target': 'price_change_5m', 'type': 'regression', 'name': 'price_5m'},
                {'target': 'direction_1m', 'type': 'classification', 'name': 'direction_1m'},
                {'target': 'direction_5m', 'type': 'classification', 'name': 'direction_5m'},
                {'target': 'volatility_next', 'type': 'regression', 'name': 'volatility'}
            ]
            
            for config in target_configs:
                target_col = config['target']
                model_type = config['type']
                model_name = config['name']
                
                logger.info(f"Training models for {model_name} ({model_type})...")
                
                # Prepare data
                data_splits = self.model_manager.prepare_data(
                    features_df, target_col, test_size=0.2, validation_size=0.1
                )
                
                # Train traditional ML models
                traditional_models = self.model_manager.train_traditional_ml_models(
                    data_splits, model_type
                )
                
                # Train deep learning model
                dl_model = self.model_manager.train_deep_learning_model(
                    data_splits, model_type, 'feedforward'
                )
                
                # Train LSTM for time series
                lstm_model = self.model_manager.train_deep_learning_model(
                    data_splits, model_type, 'lstm'
                )
                
                # Optimize best performing traditional model
                best_traditional = self._get_best_traditional_model(traditional_models)
                if best_traditional:
                    optimized_model = self.model_manager.optimize_hyperparameters(
                        data_splits, best_traditional, n_trials=50
                    )
                
                # Create ensemble
                ensemble_model = self.model_manager.create_ensemble_model(
                    data_splits, self.ensemble_models
                )
                
                logger.info(f"Completed training for {model_name}")
            
            # Save all models
            self.model_manager.save_models()
            
        except Exception as e:
            logger.error(f"Error training initial models: {e}")
    
    def _get_best_traditional_model(self, models: Dict[str, Any]) -> Optional[str]:
        """Get the best performing traditional model."""
        best_score = float('-inf')
        best_model = None
        
        for name, model_info in models.items():
            if 'val_metrics' in model_info:
                # Use R2 for regression, accuracy for classification
                if 'r2' in model_info['val_metrics']:
                    score = model_info['val_metrics']['r2']
                elif 'accuracy' in model_info['val_metrics']:
                    score = model_info['val_metrics']['accuracy']
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_model = name
        
        return best_model
    
    async def _generate_real_time_predictions(self) -> None:
        """Generate predictions for all active symbols."""
        try:
            # Get active symbols
            active_symbols = await self._get_active_symbols()
            
            for symbol in active_symbols:
                try:
                    predictions = await self._generate_symbol_predictions(symbol)
                    
                    # Update cache
                    self.prediction_cache[symbol] = {
                        'timestamp': datetime.now(),
                        'predictions': predictions
                    }
                    
                except Exception as e:
                    logger.error(f"Error generating predictions for {symbol}: {e}")
            
            self.last_prediction_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in real-time prediction generation: {e}")
    
    async def _generate_symbol_predictions(self, symbol: str) -> Dict[str, Any]:
        """Generate ML predictions for a specific symbol."""
        try:
            # Get recent data for the symbol
            recent_data = await self._get_recent_symbol_data(symbol)
            
            if recent_data.empty:
                return {}
            
            # Generate features
            features_df = self.feature_engine.generate_all_features(recent_data)
            
            # Get the latest feature row
            latest_features = features_df.iloc[-1:].drop(columns=['symbol'], errors='ignore')
            
            # Generate predictions using different models
            predictions = {}
            
            # Price predictions
            for horizon in ['1m', '5m']:
                model_name = f'price_{horizon}'
                if f'ensemble_{model_name}' in self.model_manager.models:
                    pred = self.model_manager.predict(f'ensemble_{model_name}', latest_features)
                    predictions[f'price_change_{horizon}'] = float(pred[0])
            
            # Direction predictions
            for horizon in ['1m', '5m']:
                model_name = f'direction_{horizon}'
                if f'ensemble_{model_name}' in self.model_manager.models:
                    pred = self.model_manager.predict(f'ensemble_{model_name}', latest_features)
                    predictions[f'direction_{horizon}'] = float(pred[0])
                    predictions[f'direction_{horizon}_prob'] = float(pred[0])  # Probability
            
            # Volatility prediction
            if 'ensemble_volatility' in self.model_manager.models:
                pred = self.model_manager.predict('ensemble_volatility', latest_features)
                predictions['volatility_forecast'] = float(pred[0])
            
            # Calculate confidence scores
            predictions['confidence'] = self._calculate_prediction_confidence(predictions)
            
            # Add metadata
            predictions['timestamp'] = datetime.now().isoformat()
            predictions['symbol'] = symbol
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions for {symbol}: {e}")
            return {}
    
    async def _get_recent_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Get recent market data for a symbol."""
        try:
            query = f"""
            SELECT * FROM market_data 
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC 
            LIMIT {self.lookback_periods}
            """
            
            conn = self.data_storage.get_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting recent data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols."""
        try:
            query = """
            SELECT DISTINCT symbol FROM market_data 
            WHERE timestamp >= datetime('now', '-1 hour')
            """
            
            conn = self.data_storage.get_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df['symbol'].tolist()
            
        except Exception as e:
            logger.error(f"Error getting active symbols: {e}")
            return []
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall confidence score for predictions."""
        try:
            confidence_factors = []
            
            # Direction consistency across timeframes
            if 'direction_1m_prob' in predictions and 'direction_5m_prob' in predictions:
                prob_1m = predictions['direction_1m_prob']
                prob_5m = predictions['direction_5m_prob']
                
                # Higher confidence when both timeframes agree
                agreement = 1 - abs(prob_1m - prob_5m)
                confidence_factors.append(agreement)
                
                # Higher confidence when probabilities are extreme (close to 0 or 1)
                extremeness = max(abs(prob_1m - 0.5), abs(prob_5m - 0.5)) * 2
                confidence_factors.append(extremeness)
            
            # Price change magnitude consistency
            if 'price_change_1m' in predictions and 'price_change_5m' in predictions:
                change_1m = abs(predictions['price_change_1m'])
                change_5m = abs(predictions['price_change_5m'])
                
                # Higher confidence when changes are significant
                magnitude_factor = min(change_1m * 100, 1.0)  # Cap at 100%
                confidence_factors.append(magnitude_factor)
            
            # Overall confidence
            if confidence_factors:
                confidence = np.mean(confidence_factors)
            else:
                confidence = 0.5  # Neutral confidence
            
            return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def _update_model_performance(self) -> None:
        """Update model performance metrics."""
        try:
            # Get recent predictions and actual outcomes
            # This would compare predictions made earlier with actual market movements
            # Implementation depends on how you want to track performance
            logger.info("Model performance tracking not yet implemented")
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of recent predictions and pipeline status."""
        try:
            summary = {
                'pipeline_status': 'running' if self.is_running else 'stopped',
                'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                'cached_predictions': len(self.prediction_cache),
                'active_models': len(self.model_manager.models),
                'recent_predictions': {}
            }
            
            # Add recent predictions for each symbol
            for symbol, cache_entry in self.prediction_cache.items():
                if (datetime.now() - cache_entry['timestamp']).seconds < 300:  # Last 5 minutes
                    summary['recent_predictions'][symbol] = {
                        'timestamp': cache_entry['timestamp'].isoformat(),
                        'confidence': cache_entry['predictions'].get('confidence', 0.0),
                        'direction_1m': cache_entry['predictions'].get('direction_1m', None),
                        'price_change_1m': cache_entry['predictions'].get('price_change_1m', None)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting prediction summary: {e}")
            return {}
    
    async def backtest_predictions(self, start_date: datetime, end_date: datetime, 
                                 symbol: str) -> Dict[str, Any]:
        """
        Backtest ML predictions against historical data.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            symbol: Symbol to backtest
            
        Returns:
            Backtest results with performance metrics
        """
        try:
            logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
            
            # Get historical data
            query = f"""
            SELECT * FROM market_data 
            WHERE symbol = '{symbol}'
            AND timestamp BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
            ORDER BY timestamp
            """
            
            conn = self.data_storage.get_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return {'error': 'No historical data available'}
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Generate features and targets
            features_df = self.feature_engine.generate_all_features(df)
            features_df = self._create_target_variables(features_df)
            
            # Run predictions
            predictions = []
            actuals = []
            
            # Use sliding window approach
            window_size = self.lookback_periods
            
            for i in range(window_size, len(features_df)):
                # Use data up to current point for prediction
                train_data = features_df.iloc[:i]
                current_features = features_df.iloc[i:i+1]
                
                # Get actual future values
                if i + 1 < len(features_df):
                    actual_change = features_df.iloc[i+1]['price_change_1m']
                    actual_direction = features_df.iloc[i+1]['direction_1m']
                    
                    # Make prediction (simplified - using pre-trained models)
                    if 'ensemble_price_1m' in self.model_manager.models:
                        pred_change = self.model_manager.predict(
                            'ensemble_price_1m', 
                            current_features.drop(columns=['symbol'], errors='ignore')
                        )[0]
                        
                        predictions.append(pred_change)
                        actuals.append(actual_change)
            
            # Calculate backtest metrics
            if predictions and actuals:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                # Regression metrics
                mse = np.mean((predictions - actuals) ** 2)
                mae = np.mean(np.abs(predictions - actuals))
                correlation = np.corrcoef(predictions, actuals)[0, 1]
                
                # Direction accuracy
                pred_directions = (predictions > 0).astype(int)
                actual_directions = (actuals > 0).astype(int)
                direction_accuracy = np.mean(pred_directions == actual_directions)
                
                backtest_results = {
                    'symbol': symbol,
                    'period': f"{start_date.date()} to {end_date.date()}",
                    'total_predictions': len(predictions),
                    'mse': float(mse),
                    'mae': float(mae),
                    'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                    'direction_accuracy': float(direction_accuracy),
                    'predictions': predictions.tolist(),
                    'actuals': actuals.tolist()
                }
                
                return backtest_results
            
            else:
                return {'error': 'Insufficient data for backtesting'}
                
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {'error': str(e)}
