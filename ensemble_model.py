"""
Sistema de Ensemble de Modelos para Trading
Combina múltiplos algoritmos ML para melhorar a precisão
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Classe para ensemble de modelos de trading"""
    
    def __init__(self, ensemble_type: str = "voting"):
        """
        Args:
            ensemble_type: 'voting', 'stacking', ou 'weighted'
        """
        self.ensemble_type = ensemble_type
        self.base_models = {}
        self.ensemble_model = None
        self.feature_engineer = None
        self.is_trained = False
        self.model_weights = {}
        self.performance_metrics = {}
        
    def create_base_models(self) -> Dict[str, Any]:
        """Cria os modelos base para o ensemble"""
        models = {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=50,
                random_state=42,
                verbose=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        self.base_models = models
        logger.info(f"Criados {len(models)} modelos base")
        return models
    
    def evaluate_base_models(self, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Avalia performance individual dos modelos base"""
        logger.info("Avaliando modelos base...")
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        results = {}
        
        for name, model in self.base_models.items():
            try:
                # Validação cruzada
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                
                # Treinar para métricas detalhadas
                model.fit(X, y)
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='weighted'),
                    'recall': recall_score(y, y_pred, average='weighted'),
                    'f1': f1_score(y, y_pred, average='weighted'),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                results[name] = metrics
                logger.info(f"{name}: Accuracy={metrics['accuracy']:.4f}, CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")
                
            except Exception as e:
                logger.warning(f"Erro ao avaliar {name}: {e}")
                results[name] = {'accuracy': 0.0, 'cv_mean': 0.0}
        
        self.performance_metrics = results
        return results
    
    def calculate_model_weights(self, performance_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calcula pesos dos modelos baseado na performance"""
        weights = {}
        total_score = sum(metrics['cv_mean'] for metrics in performance_metrics.values())
        
        if total_score > 0:
            for name, metrics in performance_metrics.items():
                weights[name] = metrics['cv_mean'] / total_score
        else:
            # Pesos iguais se não há performance válida
            num_models = len(performance_metrics)
            weights = {name: 1.0/num_models for name in performance_metrics.keys()}
        
        self.model_weights = weights
        logger.info(f"Pesos calculados: {weights}")
        return weights
    
    def create_voting_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """Cria ensemble por votação"""
        estimators = [(name, model) for name, model in models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Usa probabilidades
            n_jobs=-1
        )
        
        logger.info("Ensemble de votação criado")
        return ensemble
    
    def create_stacking_ensemble(self, models: Dict[str, Any]) -> StackingClassifier:
        """Cria ensemble por stacking"""
        estimators = [(name, model) for name, model in models.items()]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42)
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=TimeSeriesSplit(n_splits=3),
            n_jobs=-1
        )
        
        logger.info("Ensemble de stacking criado")
        return ensemble
    
    def create_weighted_ensemble(self, models: Dict[str, Any], 
                               weights: Dict[str, float]) -> 'WeightedEnsemble':
        """Cria ensemble com pesos customizados"""
        return WeightedEnsemble(models, weights)
    
    def train(self, df: pd.DataFrame, selected_models: List[str] = None):
        """
        Treina o ensemble
        
        Args:
            df: DataFrame com dados OHLCV
            selected_models: Lista de modelos a usar (None = todos)
        """
        logger.info("Iniciando treinamento do ensemble...")
        
        # Preparar features
        self.feature_engineer = FeatureEngineer()
        features_df = self.feature_engineer.create_features(df)
        X, y = self.feature_engineer.prepare_features(features_df)
        
        # Criar modelos base
        all_models = self.create_base_models()
        
        # Selecionar modelos
        if selected_models:
            models = {name: all_models[name] for name in selected_models if name in all_models}
        else:
            models = all_models
        
        # Avaliar modelos base
        performance = self.evaluate_base_models(X, y)
        
        # Filtrar modelos com performance mínima
        min_accuracy = 0.5  # Pelo menos 50% de acurácia
        good_models = {
            name: model for name, model in models.items()
            if performance[name]['cv_mean'] >= min_accuracy
        }
        
        if not good_models:
            logger.warning("Nenhum modelo atende critério mínimo, usando todos")
            good_models = models
        
        logger.info(f"Usando {len(good_models)} modelos: {list(good_models.keys())}")
        
        # Criar ensemble
        if self.ensemble_type == "voting":
            self.ensemble_model = self.create_voting_ensemble(good_models)
        elif self.ensemble_type == "stacking":
            self.ensemble_model = self.create_stacking_ensemble(good_models)
        elif self.ensemble_type == "weighted":
            weights = self.calculate_model_weights(performance)
            filtered_weights = {name: weights[name] for name in good_models.keys()}
            self.ensemble_model = self.create_weighted_ensemble(good_models, filtered_weights)
        
        # Treinar ensemble
        self.ensemble_model.fit(X, y)
        self.is_trained = True
        
        # Avaliar ensemble
        y_pred = self.ensemble_model.predict(X)
        ensemble_accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"Ensemble treinado! Acurácia: {ensemble_accuracy:.4f}")
        logger.info(f"Tipo: {self.ensemble_type}, Modelos: {len(good_models)}")
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Faz predições usando o ensemble"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado")
        
        features_df = self.feature_engineer.create_features(df)
        X, _ = self.feature_engineer.prepare_features(features_df)
        
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades das predições"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado")
        
        features_df = self.feature_engineer.create_features(df)
        X, _ = self.feature_engineer.prepare_features(features_df)
        
        if hasattr(self.ensemble_model, 'predict_proba'):
            return self.ensemble_model.predict_proba(X)
        else:
            # Para modelos sem predict_proba, usar predict como probabilidade
            predictions = self.ensemble_model.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[predictions == 0, 0] = 1.0
            proba[predictions == 1, 1] = 1.0
            return proba
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features (quando disponível)"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        if self.ensemble_type == "voting" and hasattr(self.ensemble_model, 'estimators_'):
            # Média das importâncias dos modelos base
            feature_names = self.feature_engineer.feature_names if hasattr(self.feature_engineer, 'feature_names') else None
            
            for name, estimator in zip(self.ensemble_model.named_estimators_.keys(), 
                                     self.ensemble_model.estimators_):
                if hasattr(estimator, 'feature_importances_'):
                    if feature_names:
                        for i, importance in enumerate(estimator.feature_importances_):
                            feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                            if feat_name not in importance_dict:
                                importance_dict[feat_name] = []
                            importance_dict[feat_name].append(importance)
            
            # Calcular média
            avg_importance = {name: np.mean(values) for name, values in importance_dict.items()}
            return avg_importance
        
        return {}
    
    def save_model(self, filepath: str = None):
        """Salva o modelo ensemble"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"models/ensemble_model_{timestamp}.joblib"
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'feature_engineer': self.feature_engineer,
            'ensemble_type': self.ensemble_type,
            'model_weights': self.model_weights,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo ensemble salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """Carrega modelo ensemble"""
        model_data = joblib.load(filepath)
        
        self.ensemble_model = model_data['ensemble_model']
        self.feature_engineer = model_data['feature_engineer']
        self.ensemble_type = model_data['ensemble_type']
        self.model_weights = model_data.get('model_weights', {})
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.is_trained = model_data.get('is_trained', False)
        
        logger.info(f"Modelo ensemble carregado de: {filepath}")

class WeightedEnsemble:
    """Ensemble com pesos customizados"""
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        self.models = models
        self.weights = weights
        self.is_fitted = False
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {name: weight/total_weight for name, weight in weights.items()}
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina todos os modelos"""
        for name, model in self.models.items():
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predição com pesos"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado")
        
        weighted_proba = np.zeros((X.shape[0], 2))
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            if weight > 0:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                    proba = np.zeros((len(pred), 2))
                    proba[pred == 0, 0] = 1.0
                    proba[pred == 1, 1] = 1.0
                
                weighted_proba += weight * proba
        
        return np.argmax(weighted_proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades com pesos"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado")
        
        weighted_proba = np.zeros((X.shape[0], 2))
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            if weight > 0:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                    proba = np.zeros((len(pred), 2))
                    proba[pred == 0, 0] = 1.0
                    proba[pred == 1, 1] = 1.0
                
                weighted_proba += weight * proba
        
        return weighted_proba

def create_ensemble_model(df: pd.DataFrame, ensemble_type: str = "voting",
                         selected_models: List[str] = None) -> EnsembleModel:
    """
    Função utilitária para criar e treinar ensemble
    
    Args:
        df: DataFrame com dados OHLCV
        ensemble_type: 'voting', 'stacking', ou 'weighted'
        selected_models: Lista de modelos a usar
    
    Returns:
        Modelo ensemble treinado
    """
    ensemble = EnsembleModel(ensemble_type)
    ensemble.train(df, selected_models)
    return ensemble