"""
Otimizador de Hiperparâmetros para Modelos de Trading
Usa Grid Search, Random Search e Optuna para otimização
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config
from ml_model import TradingMLModel
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Classe para otimização de hiperparâmetros"""
    
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.best_params = None
        self.best_score = 0.0
        self.optimization_history = []
        
    def get_param_space(self, search_type: str = "grid") -> Dict[str, Any]:
        """
        Define o espaço de busca de hiperparâmetros
        
        Args:
            search_type: 'grid', 'random', ou 'optuna'
        """
        if self.model_type == "lightgbm":
            if search_type == "grid":
                return {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'min_child_samples': [20, 30, 50],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            elif search_type == "random":
                return {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 10, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                    'num_leaves': [20, 31, 50, 100, 150],
                    'min_child_samples': [10, 20, 30, 50, 100],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0]
                }
        
        return {}
    
    def grid_search_optimization(self, X: np.ndarray, y: np.ndarray, 
                               cv_folds: int = 5) -> Dict[str, Any]:
        """
        Otimização usando Grid Search
        """
        logger.info("Iniciando otimização Grid Search...")
        
        # Time Series Split para validação cruzada
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Modelo base
        if self.model_type == "lightgbm":
            model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        # Parâmetros para busca
        param_grid = self.get_param_space("grid")
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        logger.info(f"Melhor score Grid Search: {self.best_score:.4f}")
        logger.info(f"Melhores parâmetros: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': grid_search.cv_results_
        }
    
    def random_search_optimization(self, X: np.ndarray, y: np.ndarray,
                                 n_iter: int = 100, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Otimização usando Random Search
        """
        logger.info("Iniciando otimização Random Search...")
        
        # Time Series Split para validação cruzada
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Modelo base
        if self.model_type == "lightgbm":
            model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        # Parâmetros para busca
        param_distributions = self.get_param_space("random")
        
        # Random Search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        logger.info(f"Melhor score Random Search: {self.best_score:.4f}")
        logger.info(f"Melhores parâmetros: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': random_search.cv_results_
        }
    
    def optuna_optimization(self, X: np.ndarray, y: np.ndarray,
                          n_trials: int = 100, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Otimização usando Optuna (Bayesian Optimization)
        """
        logger.info("Iniciando otimização Optuna...")
        
        def objective(trial):
            # Sugerir hiperparâmetros
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            
            # Validação cruzada
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Criar estudo Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Melhor score Optuna: {self.best_score:.4f}")
        logger.info(f"Melhores parâmetros: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study
        }
    
    def optimize_model(self, df: pd.DataFrame, method: str = "optuna",
                      **kwargs) -> TradingMLModel:
        """
        Otimiza e treina o modelo com os melhores parâmetros
        
        Args:
            df: DataFrame com dados de treino
            method: 'grid', 'random', ou 'optuna'
            **kwargs: Argumentos específicos do método
        """
        logger.info(f"Iniciando otimização usando método: {method}")
        
        # Preparar dados
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(df)
        X, y = feature_engineer.prepare_features(features_df)
        
        # Otimização
        if method == "grid":
            results = self.grid_search_optimization(X, y, **kwargs)
        elif method == "random":
            results = self.random_search_optimization(X, y, **kwargs)
        elif method == "optuna":
            results = self.optuna_optimization(X, y, **kwargs)
        else:
            raise ValueError(f"Método não suportado: {method}")
        
        # Treinar modelo final com melhores parâmetros
        model = TradingMLModel(self.model_type)
        model.feature_engineer = feature_engineer
        
        if self.model_type == "lightgbm":
            model.model = lgb.LGBMClassifier(**self.best_params)
            model.model.fit(X, y)
        
        # Salvar histórico
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'method': method,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'data_shape': X.shape
        })
        
        logger.info("Otimização concluída!")
        return model
    
    def save_optimization_results(self, filepath: str = None):
        """Salva resultados da otimização"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"models/optimization_results_{timestamp}.joblib"
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'model_type': self.model_type
        }
        
        joblib.dump(results, filepath)
        logger.info(f"Resultados salvos em: {filepath}")
    
    def load_optimization_results(self, filepath: str):
        """Carrega resultados da otimização"""
        results = joblib.load(filepath)
        self.best_params = results['best_params']
        self.best_score = results['best_score']
        self.optimization_history = results.get('optimization_history', [])
        self.model_type = results.get('model_type', 'lightgbm')
        
        logger.info(f"Resultados carregados de: {filepath}")

def optimize_trading_model(df: pd.DataFrame, method: str = "optuna",
                         model_type: str = "lightgbm", **kwargs) -> TradingMLModel:
    """
    Função utilitária para otimizar modelo de trading
    
    Args:
        df: DataFrame com dados OHLCV
        method: Método de otimização ('grid', 'random', 'optuna')
        model_type: Tipo do modelo ('lightgbm')
        **kwargs: Argumentos específicos do método
    
    Returns:
        Modelo otimizado
    """
    optimizer = ModelOptimizer(model_type)
    return optimizer.optimize_model(df, method, **kwargs)