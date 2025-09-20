"""
Modelo de Machine Learning para Previsão de Direção de Preços
Implementa LightGBM com validação e métricas de performance
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config import config
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class TradingMLModel:
    """Modelo de ML para trading"""
    
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.training_metrics = {}
        self.feature_importance = pd.DataFrame()
        
        # Parâmetros do modelo
        self.model_params = config.ml.lgb_params.copy()
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para treinamento"""
        logger.info("Preparando dados para treinamento...")
        
        # Criar features
        features_df = self.feature_engineer.create_features(df)
        
        if features_df.empty:
            raise ValueError("Nenhuma feature foi criada")
        
        # Preparar arrays X, y
        X, y = self.feature_engineer.prepare_features(features_df, fit_scaler=True)
        
        if len(X) < config.ml.min_training_samples:
            raise ValueError(f"Dados insuficientes: {len(X)} < {config.ml.min_training_samples}")
        
        logger.info(f"Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
        return X, y
    
    def train(self, df: pd.DataFrame, validation_split: float = None) -> Dict[str, float]:
        """
        Treina o modelo
        
        Args:
            df: DataFrame com dados OHLCV
            validation_split: Proporção para validação
            
        Returns:
            Métricas de treinamento
        """
        logger.info("Iniciando treinamento do modelo...")
        
        # Preparar dados
        X, y = self.prepare_data(df)
        
        # Split temporal para séries temporais
        if validation_split is None:
            validation_split = config.ml.validation_split
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Split: {len(X_train)} treino, {len(X_val)} validação")
        
        # Treinar modelo
        if self.model_type == "lightgbm":
            self.model = self._train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
        
        # Avaliar modelo
        metrics = self._evaluate_model(X_val, y_val)
        self.training_metrics = metrics
        self.is_trained = True
        
        # Feature importance
        self.feature_importance = self.feature_engineer.get_feature_importance(self.model)
        
        logger.info(f"Treinamento concluído. Acurácia: {metrics['accuracy']:.4f}")
        return metrics
    
    def _train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMClassifier:
        """Treina modelo LightGBM"""
        
        # Criar datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        # Treinar
        model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'eval'],
            num_boost_round=1000,
            callbacks=callbacks
        )
        
        return model
    
    def _evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Avalia performance do modelo"""
        y_pred = self.predict(X_val)
        y_pred_proba = self.predict_proba(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1': f1_score(y_val, y_pred, average='weighted'),
            'auc': roc_auc_score(y_val, y_pred_proba[:, 1]) if y_pred_proba.shape[1] > 1 else 0
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado")
        
        if self.model_type == "lightgbm":
            predictions = self.model.predict(X)
            return (predictions > 0.5).astype(int)
        
        return np.array([])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado")
        
        if self.model_type == "lightgbm":
            probas = self.model.predict(X)
            # LightGBM retorna probabilidade da classe positiva
            return np.column_stack([1 - probas, probas])
        
        return np.array([])
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz a partir de DataFrame OHLCV
        
        Returns:
            predictions, probabilities
        """
        # Criar features
        features_df = self.feature_engineer.create_features(df)
        
        if features_df.empty:
            return np.array([]), np.array([])
        
        # Preparar dados (sem fit do scaler)
        X, _ = self.feature_engineer.prepare_features(features_df, fit_scaler=False)
        
        # Predições
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        return predictions, probabilities
    
    def get_trading_signal(self, df: pd.DataFrame, min_confidence: float = None) -> Dict[str, Any]:
        """
        Gera sinal de trading
        
        Args:
            df: DataFrame com dados recentes
            min_confidence: Confiança mínima para sinal
            
        Returns:
            Dicionário com sinal e informações
        """
        if min_confidence is None:
            min_confidence = config.trading.min_prediction_confidence
        
        predictions, probabilities = self.predict_from_dataframe(df)
        
        if len(predictions) == 0:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'prediction': None,
                'reason': 'Dados insuficientes'
            }
        
        # Última predição
        last_prediction = predictions[-1]
        last_probability = probabilities[-1]
        
        # Confiança (probabilidade da classe predita)
        confidence = last_probability[last_prediction]
        
        # Determinar sinal
        if confidence >= min_confidence:
            signal = 'CALL' if last_prediction == 1 else 'PUT'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'prediction': last_prediction,
            'probability_up': last_probability[1],
            'probability_down': last_probability[0],
            'reason': f'Confiança: {confidence:.3f}'
        }
    
    def cross_validate(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, float]:
        """Validação cruzada temporal"""
        logger.info(f"Executando validação cruzada com {cv_folds} folds...")
        
        X, y = self.prepare_data(df)
        
        # Time Series Split
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Treinar modelo temporário
            temp_model = self._train_lightgbm(X_train, y_train, X_val, y_val)
            
            # Avaliar
            y_pred = (temp_model.predict(X_val) > 0.5).astype(int)
            
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['precision'].append(precision_score(y_val, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_val, y_pred, average='weighted'))
            scores['f1'].append(f1_score(y_val, y_pred, average='weighted'))
        
        # Médias e desvios
        cv_results = {}
        for metric, values in scores.items():
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        logger.info(f"CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        return cv_results
    
    def save_model(self, filepath: str = None):
        """Salva modelo treinado"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{config.data.models_dir}/trading_model_{timestamp}.pkl"
        
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'model_type': self.model_type,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo salvo em {filepath}")
        return filepath
    
    def load_model(self, filepath: str):
        """Carrega modelo salvo"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_engineer = model_data['feature_engineer']
            self.model_type = model_data['model_type']
            self.training_metrics = model_data.get('training_metrics', {})
            self.feature_importance = model_data.get('feature_importance', pd.DataFrame())
            self.model_params = model_data.get('model_params', {})
            
            self.is_trained = True
            logger.info(f"Modelo carregado de {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'feature_count': len(self.feature_engineer.feature_names),
            'feature_names': self.feature_engineer.feature_names[:10],  # Top 10
            'model_params': self.model_params
        }
    
    def retrain_if_needed(self, df: pd.DataFrame, performance_threshold: float = 0.55) -> bool:
        """Retreina modelo se performance estiver baixa"""
        if not self.is_trained:
            logger.info("Modelo não treinado, iniciando treinamento...")
            self.train(df)
            return True
        
        # Avaliar performance atual
        try:
            X, y = self.prepare_data(df.tail(1000))  # Últimos 1000 pontos
            current_accuracy = accuracy_score(y, self.predict(X))
            
            if current_accuracy < performance_threshold:
                logger.info(f"Performance baixa ({current_accuracy:.3f}), retreinando...")
                self.train(df)
                return True
            else:
                logger.info(f"Performance adequada ({current_accuracy:.3f})")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao avaliar performance: {e}")
            return False

# Convenience functions
def create_and_train_model(df: pd.DataFrame, model_type: str = "lightgbm") -> TradingMLModel:
    """Cria e treina modelo"""
    model = TradingMLModel(model_type)
    model.train(df)
    return model

def load_trained_model(filepath: str) -> TradingMLModel:
    """Carrega modelo treinado"""
    model = TradingMLModel()
    if model.load_model(filepath):
        return model
    return None

def get_latest_model() -> Optional[TradingMLModel]:
    """Carrega o modelo mais recente"""
    import os
    import glob
    
    model_files = glob.glob(f"{config.data.models_dir}/trading_model_*.pkl")
    if not model_files:
        return None
    
    # Modelo mais recente
    latest_model = max(model_files, key=os.path.getctime)
    return load_trained_model(latest_model)