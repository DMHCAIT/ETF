"""
Advanced ML Models for ETF Trading System.
Implements multiple machine learning algorithms for price prediction and trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from datetime import datetime, timedelta
import joblib
import pickle

# Core ML libraries with fallback handling
try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# Advanced ML libraries with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logging.warning("XGBoost not available - advanced boosting models disabled")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logging.warning("LightGBM not available - fast boosting models disabled")
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    logging.warning("CatBoost not available - categorical boosting models disabled")
    CATBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Deep Learning - with fallback for missing DLLs
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorFlow not available: {e}")
    logger.warning("Deep learning models will be disabled. Install Visual C++ Redistributable to enable.")
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class keras:
        class Model: pass
        @staticmethod
        def Sequential(*args, **kwargs): return None
        class utils:
            @staticmethod
            def to_categorical(*args, **kwargs): return None
    
    class tf:
        @staticmethod
        def constant(*args, **kwargs): return None
        class keras:
            class Model: pass
            @staticmethod
            def Sequential(*args, **kwargs): return None
    
    class layers:
        @staticmethod
        def Dense(*args, **kwargs): pass
        @staticmethod
        def LSTM(*args, **kwargs): pass
        @staticmethod
        def Dropout(*args, **kwargs): pass
        @staticmethod
        def Conv1D(*args, **kwargs): pass
        @staticmethod
        def MaxPooling1D(*args, **kwargs): pass
        @staticmethod
        def GlobalMaxPooling1D(*args, **kwargs): pass
        @staticmethod
        def Flatten(*args, **kwargs): pass
    
    class EarlyStopping: pass
    class ReduceLROnPlateau: pass

# Time Series specific - with fallback handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logging.warning("Prophet not available - time series forecasting disabled")
    PROPHET_AVAILABLE = False
    class Prophet:
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): return self
        def predict(self, *args, **kwargs): return pd.DataFrame()

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    logging.warning("Optuna not available - hyperparameter optimization disabled")
    OPTUNA_AVAILABLE = False

class MLModelManager:
    """Manages multiple ML models for trading predictions."""
    
    def __init__(self, model_save_path: str = "models/"):
        """
        Initialize ML model manager.
        
        Args:
            model_save_path: Path to save trained models
        """
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
        # Create models directory
        import os
        os.makedirs(model_save_path, exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    test_size: float = 0.2, validation_size: float = 0.1) -> Dict[str, Any]:
        """
        Prepare data for ML training with proper time series split.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Dictionary with train/validation/test splits
        """
        logger.info("Preparing data for ML training...")
        
        # Remove rows with missing target values
        df_clean = df.dropna(subset=[target_col])
        
        # Separate features and target
        feature_cols = [col for col in df_clean.columns if col != target_col]
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Time series split (preserve temporal order)
        total_samples = len(X)
        train_end = int(total_samples * (1 - test_size - validation_size))
        val_end = int(total_samples * (1 - test_size))
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
        
        # Handle missing values in features
        X_train = X_train.fillna(X_train.mean())
        X_val = X_val.fillna(X_train.mean())  # Use train means
        X_test = X_test.fillna(X_train.mean())  # Use train means
        
        data_splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': feature_cols
        }
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return data_splits
    
    def train_traditional_ml_models(self, data_splits: Dict[str, Any], 
                                  model_type: str = 'regression') -> Dict[str, Any]:
        """
        Train traditional ML models (Random Forest, XGBoost, etc.).
        
        Args:
            data_splits: Data splits from prepare_data
            model_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with trained models and performance metrics
        """
        logger.info(f"Training traditional ML models for {model_type}...")
        
        X_train, y_train = data_splits['X_train'], data_splits['y_train']
        X_val, y_val = data_splits['X_val'], data_splits['y_val']
        
        models = {}
        
        if model_type == 'regression':
            # Define regression models
            model_configs = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42),
                'catboost': CatBoostRegressor(iterations=100, random_state=42, verbose=False),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=1.0),
                'elastic_net': ElasticNet(alpha=1.0),
                'svr': SVR(kernel='rbf'),
                'knn': KNeighborsRegressor(n_neighbors=5)
            }
        else:
            # Define classification models
            model_configs = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42),
                'catboost': CatBoostClassifier(iterations=100, random_state=42, verbose=False),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic': LogisticRegression(random_state=42),
                'svc': SVC(kernel='rbf', random_state=42),
                'knn': KNeighborsClassifier(n_neighbors=5)
            }
        
        # Train each model
        for name, model in model_configs.items():
            try:
                logger.info(f"Training {name}...")
                
                # Scale data for algorithms that need it
                if name in ['ridge', 'lasso', 'elastic_net', 'svr', 'svc', 'knn', 'logistic']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train_scaled)
                val_pred = model.predict(X_val_scaled)
                
                # Calculate metrics
                if model_type == 'regression':
                    train_metrics = self._calculate_regression_metrics(y_train, train_pred)
                    val_metrics = self._calculate_regression_metrics(y_val, val_pred)
                else:
                    train_metrics = self._calculate_classification_metrics(y_train, train_pred)
                    val_metrics = self._calculate_classification_metrics(y_val, val_pred)
                
                models[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'feature_importance': self._get_feature_importance(model, data_splits['feature_names'])
                }
                
                logger.info(f"{name} trained successfully")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        self.models.update(models)
        return models
    
    def train_deep_learning_model(self, data_splits: Dict[str, Any], 
                                 model_type: str = 'regression',
                                 architecture: str = 'feedforward') -> Dict[str, Any]:
        """
        Train deep learning models using TensorFlow/Keras.
        
        Args:
            data_splits: Data splits from prepare_data
            model_type: 'regression' or 'classification'
            architecture: 'feedforward', 'lstm', or 'cnn'
            
        Returns:
            Dictionary with trained model and metrics
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping deep learning model training.")
            return {
                'model': None,
                'history': {},
                'train_metrics': {'error': 'TensorFlow not available'},
                'val_metrics': {'error': 'TensorFlow not available'},
                'architecture': architecture
            }
        
        logger.info(f"Training {architecture} deep learning model for {model_type}...")
        
        X_train, y_train = data_splits['X_train'], data_splits['y_train']
        X_val, y_val = data_splits['X_val'], data_splits['y_val']
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers[f'dl_{architecture}'] = scaler
        
        # Build model based on architecture
        if architecture == 'feedforward':
            model = self._build_feedforward_model(X_train_scaled.shape[1], model_type)
        elif architecture == 'lstm':
            # Reshape for LSTM (samples, timesteps, features)
            X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
            model = self._build_lstm_model(X_train_scaled.shape[1:], model_type)
        elif architecture == 'cnn':
            # Reshape for CNN
            X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
            model = self._build_cnn_model(X_train_scaled.shape[1:], model_type)
        
        # Compile model
        if model_type == 'regression':
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Make predictions
        train_pred = model.predict(X_train_scaled).flatten()
        val_pred = model.predict(X_val_scaled).flatten()
        
        # Calculate metrics
        if model_type == 'regression':
            train_metrics = self._calculate_regression_metrics(y_train, train_pred)
            val_metrics = self._calculate_regression_metrics(y_val, val_pred)
        else:
            train_pred_binary = (train_pred > 0.5).astype(int)
            val_pred_binary = (val_pred > 0.5).astype(int)
            train_metrics = self._calculate_classification_metrics(y_train, train_pred_binary)
            val_metrics = self._calculate_classification_metrics(y_val, val_pred_binary)
        
        dl_model = {
            'model': model,
            'history': history.history,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'architecture': architecture
        }
        
        self.models[f'dl_{architecture}'] = dl_model
        
        return dl_model
    
    def _build_feedforward_model(self, input_dim: int, model_type: str):
        """Build feedforward neural network."""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid' if model_type == 'classification' else 'linear')
        ])
        return model
    
    def _build_lstm_model(self, input_shape: Tuple[int, int], model_type: str):
        """Build LSTM neural network."""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            layers.Dense(1, activation='sigmoid' if model_type == 'classification' else 'linear')
        ])
        return model
    
    def _build_cnn_model(self, input_shape: Tuple[int, int], model_type: str):
        """Build CNN for time series."""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.Dropout(0.5),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(1, activation='sigmoid' if model_type == 'classification' else 'linear')
        ])
        return model
    
    def train_prophet_model(self, df: pd.DataFrame, target_col: str, 
                           date_col: str = 'timestamp') -> Dict[str, Any]:
        """
        Train Facebook Prophet model for time series forecasting.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            date_col: Date column name
            
        Returns:
            Dictionary with trained Prophet model and metrics
        """
        logger.info("Training Prophet model...")
        
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df = prophet_df.dropna()
        prophet_df.columns = ['ds', 'y']
        
        # Split data (80% train, 20% test)
        split_point = int(len(prophet_df) * 0.8)
        train_df = prophet_df[:split_point]
        test_df = prophet_df[split_point:]
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(train_df)
        
        # Make predictions
        forecast = model.predict(test_df[['ds']])
        
        # Calculate metrics
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        metrics = self._calculate_regression_metrics(y_true, y_pred)
        
        prophet_model = {
            'model': model,
            'forecast': forecast,
            'test_metrics': metrics,
            'train_data': train_df,
            'test_data': test_df
        }
        
        self.models['prophet'] = prophet_model
        
        return prophet_model
    
    def optimize_hyperparameters(self, data_splits: Dict[str, Any], 
                                model_name: str, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            data_splits: Data splits from prepare_data
            model_name: Name of model to optimize
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimized model and best parameters
        """
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            # Define hyperparameter search space based on model
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'random_state': 42
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
            
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {model_name}")
            
            # Train and evaluate model
            X_train, y_train = data_splits['X_train'], data_splits['y_train']
            X_val, y_val = data_splits['X_val'], data_splits['y_val']
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Return negative MSE (Optuna minimizes)
            return -mean_squared_error(y_val, y_pred)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train best model
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        
        if model_name == 'xgboost':
            best_model = xgb.XGBRegressor(**best_params)
        elif model_name == 'lightgbm':
            best_model = lgb.LGBMRegressor(**best_params)
        elif model_name == 'random_forest':
            best_model = RandomForestRegressor(**best_params)
        
        # Train with best parameters
        X_train, y_train = data_splits['X_train'], data_splits['y_train']
        X_val, y_val = data_splits['X_val'], data_splits['y_val']
        
        best_model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        
        train_metrics = self._calculate_regression_metrics(y_train, train_pred)
        val_metrics = self._calculate_regression_metrics(y_val, val_pred)
        
        optimized_model = {
            'model': best_model,
            'best_params': best_params,
            'best_score': study.best_value,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'study': study
        }
        
        self.models[f'{model_name}_optimized'] = optimized_model
        
        return optimized_model
    
    def create_ensemble_model(self, data_splits: Dict[str, Any], 
                             model_names: List[str]) -> Dict[str, Any]:
        """
        Create ensemble model combining multiple base models.
        
        Args:
            data_splits: Data splits from prepare_data
            model_names: List of model names to ensemble
            
        Returns:
            Dictionary with ensemble model and metrics
        """
        logger.info(f"Creating ensemble from models: {model_names}")
        
        X_train, y_train = data_splits['X_train'], data_splits['y_train']
        X_val, y_val = data_splits['X_val'], data_splits['y_val']
        
        # Collect predictions from base models
        train_predictions = []
        val_predictions = []
        
        for model_name in model_names:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found, skipping...")
                continue
            
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Handle scaling if needed
            if model_name in self.scalers:
                scaler = self.scalers[model_name]
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Get predictions
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            # Handle deep learning models
            if hasattr(train_pred, 'flatten'):
                train_pred = train_pred.flatten()
                val_pred = val_pred.flatten()
            
            train_predictions.append(train_pred)
            val_predictions.append(val_pred)
        
        # Average predictions (simple ensemble)
        ensemble_train_pred = np.mean(train_predictions, axis=0)
        ensemble_val_pred = np.mean(val_predictions, axis=0)
        
        # Calculate metrics
        train_metrics = self._calculate_regression_metrics(y_train, ensemble_train_pred)
        val_metrics = self._calculate_regression_metrics(y_val, ensemble_val_pred)
        
        ensemble_model = {
            'base_models': model_names,
            'train_predictions': train_predictions,
            'val_predictions': val_predictions,
            'ensemble_train_pred': ensemble_train_pred,
            'ensemble_val_pred': ensemble_val_pred,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        self.models['ensemble'] = ensemble_model
        
        return ensemble_model
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                return None
            
            return dict(zip(feature_names, importances))
        except:
            return None
    
    def save_models(self) -> bool:
        """Save all trained models to disk."""
        try:
            for name, model_info in self.models.items():
                if 'dl_' in name:
                    # Save Keras model
                    model_info['model'].save(f"{self.model_save_path}/{name}.h5")
                elif name == 'prophet':
                    # Save Prophet model
                    with open(f"{self.model_save_path}/{name}.pkl", 'wb') as f:
                        pickle.dump(model_info, f)
                else:
                    # Save scikit-learn compatible models
                    joblib.dump(model_info, f"{self.model_save_path}/{name}.joblib")
            
            # Save scalers
            joblib.dump(self.scalers, f"{self.model_save_path}/scalers.joblib")
            
            logger.info(f"Saved {len(self.models)} models to {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            import os
            import glob
            
            # Load scalers
            scaler_path = f"{self.model_save_path}/scalers.joblib"
            if os.path.exists(scaler_path):
                self.scalers = joblib.load(scaler_path)
            
            # Load models
            for file_path in glob.glob(f"{self.model_save_path}/*"):
                if file_path.endswith('.joblib') and 'scalers' not in file_path:
                    name = os.path.basename(file_path).replace('.joblib', '')
                    self.models[name] = joblib.load(file_path)
                elif file_path.endswith('.h5'):
                    name = os.path.basename(file_path).replace('.h5', '')
                    model = keras.models.load_model(file_path)
                    self.models[name] = {'model': model}
                elif file_path.endswith('.pkl'):
                    name = os.path.basename(file_path).replace('.pkl', '')
                    with open(file_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            logger.info(f"Loaded {len(self.models)} models from {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Compare performance of all trained models."""
        comparison_data = []
        
        for name, model_info in self.models.items():
            if name == 'ensemble':
                continue
                
            row = {'model': name}
            
            # Add validation metrics
            if 'val_metrics' in model_info:
                for metric, value in model_info['val_metrics'].items():
                    row[f'val_{metric}'] = value
            
            # Add training metrics
            if 'train_metrics' in model_info:
                for metric, value in model_info['train_metrics'].items():
                    row[f'train_{metric}'] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Apply scaling if needed
        if model_name in self.scalers:
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Handle different model types
        if 'dl_' in model_name:
            if 'lstm' in model_name:
                X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            elif 'cnn' in model_name:
                X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            
            predictions = model.predict(X_scaled).flatten()
        else:
            predictions = model.predict(X_scaled)
        
        return predictions
