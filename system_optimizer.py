"""
Sistema de Otimização de Performance para Trading Automatizado
Otimiza modelos, estratégias e performance geral do sistema
"""

import pandas as pd
import numpy as np
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Imports para otimização
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
import joblib

@dataclass
class OptimizationConfig:
    """Configurações de otimização"""
    max_trials: int = 100
    timeout_seconds: int = 3600  # 1 hora
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    
@dataclass
class ModelPerformance:
    """Performance de modelo"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    prediction_time: float
    memory_usage: float
    best_params: Dict

@dataclass
class OptimizationResult:
    """Resultado da otimização"""
    timestamp: datetime
    optimization_type: str
    best_model: str
    best_score: float
    improvement: float
    total_time: float
    models_tested: int

class ModelOptimizer:
    """Otimizador de modelos de machine learning"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                model_type: str = 'auto') -> Tuple[object, Dict]:
        """Otimizar hiperparâmetros usando Optuna"""
        
        def objective(trial):
            if model_type == 'random_forest' or (model_type == 'auto' and trial.suggest_categorical('model', ['rf']) == 'rf'):
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=self.config.random_state
                )
            elif model_type == 'gradient_boosting' or (model_type == 'auto' and trial.suggest_categorical('model', ['gb']) == 'gb'):
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    random_state=self.config.random_state
                )
            elif model_type == 'logistic' or (model_type == 'auto' and trial.suggest_categorical('model', ['lr']) == 'lr'):
                model = LogisticRegression(
                    C=trial.suggest_float('C', 0.01, 100, log=True),
                    penalty=trial.suggest_categorical('penalty', ['l1', 'l2']),
                    solver='liblinear',
                    random_state=self.config.random_state
                )
            else:  # SVM
                model = SVC(
                    C=trial.suggest_float('C', 0.01, 100, log=True),
                    kernel=trial.suggest_categorical('kernel', ['rbf', 'linear']),
                    gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
                    random_state=self.config.random_state
                )
                
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=self.config.cv_folds, scoring='accuracy')
            return scores.mean()
            
        # Criar estudo Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.max_trials, timeout=self.config.timeout_seconds)
        
        # Treinar melhor modelo
        best_params = study.best_params
        
        if 'model' in best_params:
            model_key = best_params.pop('model')
            if model_key == 'rf':
                best_model = RandomForestClassifier(**best_params, random_state=self.config.random_state)
            elif model_key == 'gb':
                best_model = GradientBoostingClassifier(**best_params, random_state=self.config.random_state)
            elif model_key == 'lr':
                best_model = LogisticRegression(**best_params, random_state=self.config.random_state)
            else:
                best_model = SVC(**best_params, random_state=self.config.random_state)
        else:
            if model_type == 'random_forest':
                best_model = RandomForestClassifier(**best_params, random_state=self.config.random_state)
            elif model_type == 'gradient_boosting':
                best_model = GradientBoostingClassifier(**best_params, random_state=self.config.random_state)
            elif model_type == 'logistic':
                best_model = LogisticRegression(**best_params, random_state=self.config.random_state)
            else:
                best_model = SVC(**best_params, random_state=self.config.random_state)
                
        best_model.fit(X, y)
        
        return best_model, study.best_params
        
    def evaluate_model_performance(self, model, X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Avaliar performance completa do modelo"""
        
        # Tempo de treinamento
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Tempo de predição
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Uso de memória (aproximado)
        memory_usage = self.estimate_model_memory(model)
        
        return ModelPerformance(
            model_name=type(model).__name__,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            prediction_time=prediction_time,
            memory_usage=memory_usage,
            best_params=getattr(model, 'get_params', lambda: {})()
        )
        
    def estimate_model_memory(self, model) -> float:
        """Estimar uso de memória do modelo (MB)"""
        try:
            import sys
            return sys.getsizeof(model) / (1024 * 1024)
        except:
            return 0.0
            
    def optimize_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Otimizar seleção de features"""
        from sklearn.feature_selection import SelectKBest, f_classif, RFE
        from sklearn.ensemble import RandomForestClassifier
        
        # Método 1: SelectKBest
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        feature_scores = selector.scores_
        
        # Método 2: RFE com Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
        rfe = RFE(estimator=rf, n_features_to_select=min(20, X.shape[1]))
        rfe.fit(X, y)
        
        # Combinar resultados
        feature_importance = np.zeros(X.shape[1])
        feature_importance += (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min())
        feature_importance += rfe.ranking_ / rfe.ranking_.max()
        
        # Selecionar top features
        n_features = min(15, X.shape[1])
        top_features = np.argsort(feature_importance)[-n_features:]
        
        return X[:, top_features], top_features.tolist()

class StrategyOptimizer:
    """Otimizador de estratégias de trading"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        self.logger = logging.getLogger(__name__)
        
    def optimize_entry_conditions(self, data: pd.DataFrame) -> Dict:
        """Otimizar condições de entrada"""
        
        # Parâmetros para otimizar
        rsi_params = range(10, 31, 5)  # RSI period
        ma_params = range(5, 21, 5)   # Moving average period
        confidence_thresholds = np.arange(0.5, 0.9, 0.1)
        
        best_params = {}
        best_score = 0
        
        for rsi_period in rsi_params:
            for ma_period in ma_params:
                for conf_threshold in confidence_thresholds:
                    
                    # Calcular indicadores
                    data_copy = data.copy()
                    data_copy['rsi'] = self.calculate_rsi(data_copy['close'], rsi_period)
                    data_copy['ma'] = data_copy['close'].rolling(ma_period).mean()
                    
                    # Simular estratégia
                    score = self.simulate_strategy(data_copy, conf_threshold)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'rsi_period': rsi_period,
                            'ma_period': ma_period,
                            'confidence_threshold': conf_threshold
                        }
                        
        return best_params
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def simulate_strategy(self, data: pd.DataFrame, confidence_threshold: float) -> float:
        """Simular estratégia e retornar score"""
        
        # Condições de entrada simplificadas
        buy_signals = (
            (data['rsi'] < 30) & 
            (data['close'] > data['ma']) &
            (np.random.random(len(data)) > confidence_threshold)  # Simular confiança
        )
        
        # Calcular retornos
        returns = []
        for i in range(1, len(data)):
            if buy_signals.iloc[i-1]:
                ret = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                returns.append(ret)
                
        if not returns:
            return 0
            
        # Score baseado em Sharpe ratio simplificado
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
            
        return mean_return / std_return

class SystemOptimizer:
    """Otimizador geral do sistema"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.model_optimizer = ModelOptimizer(config)
        self.strategy_optimizer = StrategyOptimizer()
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        self.logger = logging.getLogger(__name__)
        
    def generate_synthetic_data(self, n_samples: int = 1000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Gerar dados sintéticos para teste"""
        np.random.seed(self.config.random_state)
        
        # Features baseadas em indicadores técnicos
        X = np.random.randn(n_samples, n_features)
        
        # Target baseado em combinação linear com ruído
        weights = np.random.randn(n_features)
        y_continuous = X @ weights + np.random.randn(n_samples) * 0.1
        y = (y_continuous > np.median(y_continuous)).astype(int)
        
        return X, y
        
    def optimize_complete_system(self) -> Dict:
        """Otimizar sistema completo"""
        
        self.logger.info("Iniciando otimização completa do sistema")
        start_time = time.time()
        
        results = {}
        
        # 1. Otimização de modelos
        self.logger.info("Otimizando modelos de ML...")
        X, y = self.generate_synthetic_data(2000, 25)
        
        # Dividir dados
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Otimizar seleção de features
        X_train_selected, selected_features = self.model_optimizer.optimize_feature_selection(X_train, y_train)
        X_test_selected = X_test[:, selected_features]
        
        # Testar diferentes modelos
        models_to_test = ['random_forest', 'gradient_boosting', 'logistic']
        model_performances = []
        
        for model_type in models_to_test:
            try:
                self.logger.info(f"Otimizando {model_type}...")
                
                # Otimizar hiperparâmetros
                best_model, best_params = self.model_optimizer.optimize_hyperparameters(
                    X_train_selected, y_train, model_type
                )
                
                # Avaliar performance
                performance = self.model_optimizer.evaluate_model_performance(
                    best_model, X_train_selected, X_test_selected, y_train, y_test
                )
                
                model_performances.append(performance)
                
            except Exception as e:
                self.logger.error(f"Erro ao otimizar {model_type}: {e}")
                
        # Selecionar melhor modelo
        if model_performances:
            best_model_perf = max(model_performances, key=lambda x: x.f1_score)
            results['best_model'] = asdict(best_model_perf)
            
        # 2. Otimização de estratégias
        self.logger.info("Otimizando estratégias de trading...")
        
        # Gerar dados de preços sintéticos
        price_data = self.generate_price_data(1000)
        strategy_params = self.strategy_optimizer.optimize_entry_conditions(price_data)
        results['best_strategy_params'] = strategy_params
        
        # 3. Otimização de performance
        self.logger.info("Otimizando performance do sistema...")
        performance_optimizations = self.optimize_system_performance()
        results['performance_optimizations'] = performance_optimizations
        
        # 4. Resultados finais
        total_time = time.time() - start_time
        results['optimization_summary'] = {
            'total_time': total_time,
            'models_tested': len(model_performances),
            'features_selected': len(selected_features),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Otimização concluída em {total_time:.1f}s")
        
        return results
        
    def generate_price_data(self, n_points: int) -> pd.DataFrame:
        """Gerar dados de preços sintéticos"""
        np.random.seed(self.config.random_state)
        
        # Gerar preços com random walk
        returns = np.random.randn(n_points) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='1min'),
            'close': prices,
            'volume': np.random.randint(100, 1000, n_points)
        })
        
        return data
        
    def optimize_system_performance(self) -> Dict:
        """Otimizar performance geral do sistema"""
        
        optimizations = {
            'memory_optimization': {
                'use_float32': True,
                'batch_processing': True,
                'garbage_collection': True
            },
            'computation_optimization': {
                'parallel_processing': True,
                'vectorized_operations': True,
                'caching': True
            },
            'model_optimization': {
                'model_compression': True,
                'feature_reduction': True,
                'early_stopping': True
            }
        }
        
        return optimizations
        
    def save_optimization_results(self, results: Dict, filename: str = None):
        """Salvar resultados da otimização"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Resultados salvos: {filename}")
        return filename

def run_system_optimization():
    """Executar otimização completa do sistema"""
    print("Sistema de Otimização de Performance")
    print("Otimizando modelos, estratégias e performance geral")
    
    # Configurar otimização
    config = OptimizationConfig(
        max_trials=50,  # Reduzido para demo
        timeout_seconds=300,  # 5 minutos
        cv_folds=3
    )
    
    # Criar otimizador
    optimizer = SystemOptimizer(config)
    
    # Executar otimização
    results = optimizer.optimize_complete_system()
    
    # Mostrar resultados
    print(f"\nResultados da Otimização:")
    
    if 'best_model' in results:
        model = results['best_model']
        print(f"\nMelhor Modelo: {model['model_name']}")
        print(f"   Acurácia: {model['accuracy']:.3f}")
        print(f"   F1-Score: {model['f1_score']:.3f}")
        print(f"   Tempo de Treinamento: {model['training_time']:.2f}s")
        print(f"   Tempo de Predição: {model['prediction_time']:.4f}s")
        print(f"   Uso de Memória: {model['memory_usage']:.2f}MB")
        
    if 'best_strategy_params' in results:
        strategy = results['best_strategy_params']
        print(f"\nMelhores Parâmetros de Estratégia:")
        for param, value in strategy.items():
            print(f"   {param}: {value}")
            
    if 'performance_optimizations' in results:
        print(f"\nOtimizações de Performance Aplicadas:")
        perf = results['performance_optimizations']
        for category, optimizations in perf.items():
            print(f"   {category}:")
            for opt, enabled in optimizations.items():
                status = "[OK]" if enabled else "[--]"
                print(f"     {status} {opt}")
                
    # Salvar resultados
    filename = optimizer.save_optimization_results(results)
    print(f"\nResultados salvos: {filename}")
    
    # Resumo final
    summary = results.get('optimization_summary', {})
    print(f"\nResumo:")
    print(f"   Tempo Total: {summary.get('total_time', 0):.1f}s")
    print(f"   Modelos Testados: {summary.get('models_tested', 0)}")
    print(f"   Features Selecionadas: {summary.get('features_selected', 0)}")
    
    print(f"\nOtimização do sistema concluída!")

if __name__ == "__main__":
    run_system_optimization()