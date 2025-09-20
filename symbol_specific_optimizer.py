#!/usr/bin/env python3
"""
Otimizador Específico por Símbolo
Sistema para otimizar estratégias individuais para cada símbolo
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

from advanced_indicators import AdvancedIndicators
from utils import setup_logging

logger = logging.getLogger(__name__)

class SymbolSpecificOptimizer:
    """Otimizador específico para cada símbolo"""
    
    def __init__(self):
        """Inicializa o otimizador"""
        self.symbols = ["R_50", "R_100", "R_25", "R_75"]
        self.optimization_results = {}
        
        # Configurar diretórios
        self.results_dir = "symbol_optimization_results"
        self.models_dir = os.path.join(self.results_dir, "models")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        
        for dir_path in [self.results_dir, self.models_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Configurações específicas por símbolo
        self.symbol_configs = {
            "R_50": {
                "volatility": 0.02,
                "trend_strength": 0.001,
                "noise_level": 0.5,
                "optimal_timeframe": "5min",
                "risk_tolerance": "medium",
                "preferred_indicators": ["rsi", "macd", "bb_position"]
            },
            "R_100": {
                "volatility": 0.015,
                "trend_strength": 0.0008,
                "noise_level": 0.3,
                "optimal_timeframe": "15min",
                "risk_tolerance": "low",
                "preferred_indicators": ["sma_20", "ema_50", "williams_r"]
            },
            "R_25": {
                "volatility": 0.025,
                "trend_strength": 0.0012,
                "noise_level": 0.7,
                "optimal_timeframe": "1min",
                "risk_tolerance": "high",
                "preferred_indicators": ["stoch_rsi_k", "cci", "volatility"]
            },
            "R_75": {
                "volatility": 0.018,
                "trend_strength": 0.0009,
                "noise_level": 0.4,
                "optimal_timeframe": "30min",
                "risk_tolerance": "medium",
                "preferred_indicators": ["macd_histogram", "bb_position", "trend_signal"]
            }
        }
        
        logger.info("SymbolSpecificOptimizer inicializado")
    
    def generate_symbol_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        Gera dados específicos para um símbolo
        
        Args:
            symbol: Símbolo para análise
            days: Número de dias
            
        Returns:
            DataFrame com dados OHLCV
        """
        try:
            config = self.symbol_configs.get(symbol, self.symbol_configs["R_50"])
            base_price = float(symbol.split('_')[1])
            
            # Determinar frequência baseada no timeframe ótimo
            timeframe_freq = {
                "1min": "1min",
                "5min": "5min", 
                "15min": "15min",
                "30min": "30min"
            }
            
            freq = timeframe_freq.get(config["optimal_timeframe"], "5min")
            
            # Calcular períodos
            periods_map = {
                "1min": days * 24 * 60,
                "5min": days * 24 * 12,
                "15min": days * 24 * 4,
                "30min": days * 24 * 2
            }
            
            periods = periods_map.get(freq, days * 24 * 12)
            
            # Gerar timestamps
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                periods=periods,
                freq=freq.replace('min', 'T')
            )
            
            # Seed baseado no símbolo
            np.random.seed(hash(symbol) % 2**32)
            
            # Componentes de preço específicos do símbolo
            volatility = config["volatility"]
            trend_strength = config["trend_strength"]
            noise_level = config["noise_level"]
            
            # Tendência de longo prazo
            long_trend = np.linspace(0, trend_strength * periods, periods)
            
            # Ciclos baseados no timeframe
            if freq == "1min":
                # Ciclos rápidos para 1min
                medium_cycles = (
                    np.sin(np.arange(periods) * 2 * np.pi / 60) * volatility * 0.3 +  # Ciclo horário
                    np.sin(np.arange(periods) * 2 * np.pi / 15) * volatility * 0.2    # Ciclo de 15min
                )
            elif freq == "5min":
                # Ciclos médios para 5min
                medium_cycles = (
                    np.sin(np.arange(periods) * 2 * np.pi / 288) * volatility * 0.4 +  # Ciclo diário
                    np.sin(np.arange(periods) * 2 * np.pi / 12) * volatility * 0.3     # Ciclo horário
                )
            else:
                # Ciclos longos para timeframes maiores
                medium_cycles = (
                    np.sin(np.arange(periods) * 2 * np.pi / 48) * volatility * 0.5 +   # Ciclo diário
                    np.sin(np.arange(periods) * 2 * np.pi / 7) * volatility * 0.2      # Ciclo semanal
                )
            
            # Ruído específico do símbolo
            short_noise = np.random.normal(0, volatility * noise_level, periods)
            
            # Eventos especiais baseados na tolerância ao risco
            event_probability = {
                "low": 0.0005,
                "medium": 0.001,
                "high": 0.002
            }.get(config["risk_tolerance"], 0.001)
            
            events = np.zeros(periods)
            event_indices = np.random.random(periods) < event_probability
            events[event_indices] = np.random.normal(0, volatility * 5, np.sum(event_indices))
            
            # Combinar componentes
            returns = long_trend + medium_cycles + short_noise + events
            
            # Calcular preços
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))
            
            # Criar dados OHLC
            data = []
            for i, price in enumerate(prices):
                volatility_factor = volatility / 100
                high_factor = 1 + abs(np.random.normal(0, volatility_factor))
                low_factor = 1 - abs(np.random.normal(0, volatility_factor))
                
                high = price * high_factor
                low = price * low_factor
                
                # Volume baseado na volatilidade e timeframe
                base_volume = {
                    "1min": 1000,
                    "5min": 3000,
                    "15min": 8000,
                    "30min": 15000
                }.get(freq, 3000)
                
                volume = int(base_volume + abs(returns[i]) * 50000 + np.random.normal(0, base_volume * 0.2))
                
                data.append({
                    'timestamp': dates[i],
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': max(volume, 100)
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Dados gerados para {symbol}: {len(df)} registros em {freq}")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados para {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_symbol_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calcula features específicas para um símbolo
        
        Args:
            data: Dados OHLCV
            symbol: Símbolo
            
        Returns:
            DataFrame com features
        """
        try:
            df = data.copy()
            config = self.symbol_configs.get(symbol, self.symbol_configs["R_50"])
            
            # Features básicas
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Médias móveis adaptadas ao símbolo
            if config["optimal_timeframe"] == "1min":
                sma_periods = [5, 10, 20]
                ema_periods = [8, 13, 21]
            elif config["optimal_timeframe"] == "5min":
                sma_periods = [10, 20, 50]
                ema_periods = [12, 26, 50]
            elif config["optimal_timeframe"] == "15min":
                sma_periods = [20, 50, 100]
                ema_periods = [21, 55, 89]
            else:  # 30min
                sma_periods = [50, 100, 200]
                ema_periods = [34, 89, 144]
            
            # Calcular médias móveis
            for period in sma_periods:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'sma_{period}_signal'] = (df['close'] > df[f'sma_{period}']).astype(int)
            
            for period in ema_periods:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'ema_{period}_signal'] = (df['close'] > df[f'ema_{period}']).astype(int)
            
            # RSI adaptado
            rsi_period = {
                "1min": 14,
                "5min": 14,
                "15min": 21,
                "30min": 28
            }.get(config["optimal_timeframe"], 14)
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi_signal'] = 0
            df.loc[df['rsi'] > 70, 'rsi_signal'] = -1  # Sobrecomprado
            df.loc[df['rsi'] < 30, 'rsi_signal'] = 1   # Sobrevendido
            
            # Bollinger Bands
            bb_period = sma_periods[1]
            bb_std_mult = {
                "low": 1.8,
                "medium": 2.0,
                "high": 2.2
            }.get(config["risk_tolerance"], 2.0)
            
            bb_middle = df['close'].rolling(bb_period).mean()
            bb_std = df['close'].rolling(bb_period).std()
            df['bb_upper'] = bb_middle + (bb_std * bb_std_mult)
            df['bb_lower'] = bb_middle - (bb_std * bb_std_mult)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_signal'] = 0
            df.loc[df['bb_position'] > 0.8, 'bb_signal'] = -1
            df.loc[df['bb_position'] < 0.2, 'bb_signal'] = 1
            
            # MACD
            ema_fast = 12 if config["optimal_timeframe"] in ["1min", "5min"] else 26
            ema_slow = 26 if config["optimal_timeframe"] in ["1min", "5min"] else 52
            
            ema_12 = df['close'].ewm(span=ema_fast).mean()
            ema_26 = df['close'].ewm(span=ema_slow).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            df['macd_cross_signal'] = (df['macd'] > df['macd_signal']).astype(int)
            
            # Indicadores avançados
            indicators = AdvancedIndicators()
            
            try:
                df['williams_r'] = indicators.williams_percent_r(df)
                df['cci'] = indicators.commodity_channel_index(df)
                
                stoch_rsi = indicators.stochastic_rsi(df)
                if stoch_rsi:
                    df['stoch_rsi_k'] = stoch_rsi.get('k', 50)
                    df['stoch_rsi_d'] = stoch_rsi.get('d', 50)
                else:
                    df['stoch_rsi_k'] = 50
                    df['stoch_rsi_d'] = 50
                    
            except Exception as e:
                logger.warning(f"Erro ao calcular indicadores avançados para {symbol}: {e}")
                df['williams_r'] = -50
                df['cci'] = 0
                df['stoch_rsi_k'] = 50
                df['stoch_rsi_d'] = 50
            
            # Features de momentum específicas
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Features de volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
            
            # Features de volatilidade
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            
            # Sinais combinados baseados nos indicadores preferidos
            preferred_indicators = config["preferred_indicators"]
            signal_columns = [col for col in df.columns if col.endswith('_signal')]
            
            # Sinal de consenso dos indicadores preferidos
            preferred_signals = []
            for indicator in preferred_indicators:
                signal_col = f"{indicator}_signal"
                if signal_col in df.columns:
                    preferred_signals.append(signal_col)
                elif indicator in df.columns:
                    # Criar sinal baseado no valor do indicador
                    if 'rsi' in indicator:
                        df[f'{indicator}_derived_signal'] = 0
                        df.loc[df[indicator] > 70, f'{indicator}_derived_signal'] = -1
                        df.loc[df[indicator] < 30, f'{indicator}_derived_signal'] = 1
                        preferred_signals.append(f'{indicator}_derived_signal')
                    elif 'bb_position' in indicator:
                        df[f'{indicator}_derived_signal'] = 0
                        df.loc[df[indicator] > 0.8, f'{indicator}_derived_signal'] = -1
                        df.loc[df[indicator] < 0.2, f'{indicator}_derived_signal'] = 1
                        preferred_signals.append(f'{indicator}_derived_signal')
            
            # Calcular sinal de consenso
            if preferred_signals:
                df['consensus_signal'] = df[preferred_signals].mean(axis=1)
            else:
                df['consensus_signal'] = 0
            
            # Target para ML (próximo movimento)
            df['future_return'] = df['close'].shift(-1) / df['close'] - 1
            df['target'] = 0
            df.loc[df['future_return'] > 0.001, 'target'] = 1  # CALL
            df.loc[df['future_return'] < -0.001, 'target'] = -1  # PUT
            
            # Limpeza
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)
            
            logger.info(f"Features calculadas para {symbol}: {len(df)} registros, {len(df.columns)} colunas")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao calcular features para {symbol}: {e}")
            return data
    
    def optimize_symbol_model(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Otimiza modelo específico para um símbolo
        
        Args:
            data: Dados com features
            symbol: Símbolo
            
        Returns:
            Resultados da otimização
        """
        try:
            logger.info(f"Otimizando modelo para {symbol}")
            
            # Preparar dados
            feature_columns = [col for col in data.columns if col not in [
                'open', 'high', 'low', 'close', 'volume', 'target', 'future_return'
            ]]
            
            X = data[feature_columns].copy()
            y = data['target'].copy()
            
            # Filtrar apenas sinais claros (não neutros)
            clear_signals = y != 0
            X_filtered = X[clear_signals]
            y_filtered = y[clear_signals]
            
            if len(X_filtered) < 100:
                logger.warning(f"Poucos dados para {symbol}: {len(X_filtered)}")
                return {'error': 'Dados insuficientes'}
            
            # Converter para binário (1 para CALL, 0 para PUT)
            y_binary = (y_filtered > 0).astype(int)
            
            # Limpeza robusta
            X_clean = X_filtered.copy()
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
            X_clean = X_clean.fillna(0)
            
            # Normalização
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_clean)
            X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
            
            # Split temporal
            split_point = int(len(X_scaled) * 0.8)
            X_train = X_scaled.iloc[:split_point]
            X_test = X_scaled.iloc[split_point:]
            y_train = y_binary.iloc[:split_point]
            y_test = y_binary.iloc[split_point:]
            
            # Modelos para testar
            models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True)
            }
            
            # Parâmetros para otimização
            param_grids = {
                'RandomForest': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                },
                'GradientBoosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                },
                'LogisticRegression': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'SVM': {
                    'C': [0.1, 1.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
            
            # Otimizar cada modelo
            best_models = {}
            results = {}
            
            for model_name, model in models.items():
                try:
                    logger.info(f"Otimizando {model_name} para {symbol}")
                    
                    # Grid search com validação temporal
                    tscv = TimeSeriesSplit(n_splits=3)
                    grid_search = GridSearchCV(
                        model,
                        param_grids[model_name],
                        cv=tscv,
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    # Melhor modelo
                    best_model = grid_search.best_estimator_
                    
                    # Predições
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                    
                    # Métricas
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Análise de features importantes
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = dict(zip(
                            X_clean.columns,
                            best_model.feature_importances_
                        ))
                        top_features = sorted(
                            feature_importance.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:10]
                    else:
                        top_features = []
                    
                    results[model_name] = {
                        'accuracy': accuracy,
                        'best_params': grid_search.best_params_,
                        'cv_score': grid_search.best_score_,
                        'top_features': top_features,
                        'predictions': y_pred.tolist(),
                        'probabilities': y_proba.tolist()
                    }
                    
                    best_models[model_name] = best_model
                    
                    logger.info(f"{model_name} para {symbol}: Acurácia = {accuracy:.3f}")
                    
                except Exception as e:
                    logger.error(f"Erro ao otimizar {model_name} para {symbol}: {e}")
                    continue
            
            # Selecionar melhor modelo
            if results:
                best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
                best_accuracy = results[best_model_name]['accuracy']
                
                optimization_result = {
                    'symbol': symbol,
                    'best_model': best_model_name,
                    'best_accuracy': best_accuracy,
                    'all_results': results,
                    'data_stats': {
                        'total_samples': len(data),
                        'training_samples': len(X_train),
                        'test_samples': len(X_test),
                        'features_count': len(feature_columns)
                    },
                    'symbol_config': self.symbol_configs.get(symbol, {}),
                    'optimization_timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Otimização concluída para {symbol}: {best_model_name} com {best_accuracy:.3f} de acurácia")
                return optimization_result
            
            else:
                return {'error': 'Nenhum modelo foi otimizado com sucesso'}
                
        except Exception as e:
            logger.error(f"Erro na otimização para {symbol}: {e}")
            return {'error': str(e)}
    
    def run_symbol_optimization(self, symbol: str) -> Dict:
        """
        Executa otimização completa para um símbolo
        
        Args:
            symbol: Símbolo para otimizar
            
        Returns:
            Resultados da otimização
        """
        try:
            logger.info(f"Iniciando otimização para {symbol}")
            
            # Gerar dados
            data = self.generate_symbol_data(symbol)
            
            if data.empty:
                return {'error': 'Falha na geração de dados'}
            
            # Calcular features
            data_with_features = self.calculate_symbol_features(data, symbol)
            
            if len(data_with_features) < 500:
                return {'error': 'Dados insuficientes após processamento'}
            
            # Otimizar modelo
            optimization_result = self.optimize_symbol_model(data_with_features, symbol)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Erro na otimização de {symbol}: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_optimization(self) -> Dict:
        """
        Executa otimização para todos os símbolos
        
        Returns:
            Resultados completos
        """
        try:
            logger.info("Iniciando otimização abrangente por símbolos")
            
            all_results = {
                'optimization_results': {},
                'summary': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for symbol in self.symbols:
                logger.info(f"Otimizando {symbol}...")
                
                symbol_result = self.run_symbol_optimization(symbol)
                
                if 'error' not in symbol_result:
                    all_results['optimization_results'][symbol] = symbol_result
                else:
                    logger.warning(f"Falha na otimização de {symbol}: {symbol_result['error']}")
            
            # Calcular resumo
            all_results['summary'] = self.calculate_optimization_summary(all_results)
            
            # Salvar resultados
            results_file = os.path.join(
                self.results_dir,
                f"symbol_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"Otimização completa. Resultados salvos em: {results_file}")
            return all_results
            
        except Exception as e:
            logger.error(f"Erro na otimização abrangente: {e}")
            return {}
    
    def calculate_optimization_summary(self, results: Dict) -> Dict:
        """
        Calcula resumo da otimização
        
        Args:
            results: Resultados da otimização
            
        Returns:
            Resumo calculado
        """
        try:
            optimization_results = results.get('optimization_results', {})
            
            if not optimization_results:
                return {}
            
            # Estatísticas gerais
            total_symbols = len(optimization_results)
            successful_optimizations = len([r for r in optimization_results.values() if 'error' not in r])
            
            # Melhores modelos por símbolo
            best_models = {}
            accuracies = []
            
            for symbol, result in optimization_results.items():
                if 'best_model' in result:
                    best_models[symbol] = {
                        'model': result['best_model'],
                        'accuracy': result['best_accuracy'],
                        'config': result.get('symbol_config', {})
                    }
                    accuracies.append(result['best_accuracy'])
            
            # Estatísticas de performance
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                min_accuracy = np.min(accuracies)
                max_accuracy = np.max(accuracies)
                std_accuracy = np.std(accuracies)
            else:
                avg_accuracy = min_accuracy = max_accuracy = std_accuracy = 0
            
            # Análise de modelos preferidos
            model_counts = {}
            for result in optimization_results.values():
                if 'best_model' in result:
                    model = result['best_model']
                    model_counts[model] = model_counts.get(model, 0) + 1
            
            # Recomendações
            recommendations = []
            
            if avg_accuracy > 0.7:
                recommendations.append(f"Excelente performance geral: {avg_accuracy:.1%} de acurácia média")
            elif avg_accuracy > 0.6:
                recommendations.append(f"Boa performance: {avg_accuracy:.1%} de acurácia média")
            else:
                recommendations.append(f"Performance moderada: {avg_accuracy:.1%} - considerar mais features")
            
            if std_accuracy < 0.05:
                recommendations.append("Consistência alta entre símbolos")
            else:
                recommendations.append("Variação significativa entre símbolos - otimização específica necessária")
            
            # Modelo mais eficaz
            if model_counts:
                best_overall_model = max(model_counts.keys(), key=lambda k: model_counts[k])
                recommendations.append(f"Modelo mais eficaz: {best_overall_model} ({model_counts[best_overall_model]} símbolos)")
            
            summary = {
                'total_symbols': total_symbols,
                'successful_optimizations': successful_optimizations,
                'success_rate': successful_optimizations / total_symbols if total_symbols > 0 else 0,
                'best_models_by_symbol': best_models,
                'performance_stats': {
                    'average_accuracy': avg_accuracy,
                    'min_accuracy': min_accuracy,
                    'max_accuracy': max_accuracy,
                    'std_accuracy': std_accuracy
                },
                'model_distribution': model_counts,
                'recommendations': recommendations
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao calcular resumo: {e}")
            return {}


def run_symbol_optimization():
    """Função principal para executar otimização por símbolos"""
    try:
        # Configurar logging
        setup_logging()
        
        print("\n" + "="*80)
        print("🎯 OTIMIZAÇÃO ESPECÍFICA POR SÍMBOLO")
        print("="*80)
        print("🔧 Otimizando modelos individuais...")
        print("📊 Testando múltiplos algoritmos...")
        print("⚡ Ajustando hiperparâmetros...")
        print("="*80)
        
        # Criar otimizador
        optimizer = SymbolSpecificOptimizer()
        
        # Executar otimização
        results = optimizer.run_comprehensive_optimization()
        
        if results and 'summary' in results:
            summary = results['summary']
            
            print("\n" + "="*80)
            print("🏆 RESULTADOS DA OTIMIZAÇÃO POR SÍMBOLO")
            print("="*80)
            
            print(f"📊 Símbolos processados: {summary.get('total_symbols', 0)}")
            print(f"✅ Otimizações bem-sucedidas: {summary.get('successful_optimizations', 0)}")
            print(f"📈 Taxa de sucesso: {summary.get('success_rate', 0):.1%}")
            
            # Performance
            perf_stats = summary.get('performance_stats', {})
            if perf_stats:
                print(f"\n🎯 ESTATÍSTICAS DE PERFORMANCE:")
                print(f"   • Acurácia média: {perf_stats.get('average_accuracy', 0):.1%}")
                print(f"   • Acurácia mínima: {perf_stats.get('min_accuracy', 0):.1%}")
                print(f"   • Acurácia máxima: {perf_stats.get('max_accuracy', 0):.1%}")
                print(f"   • Desvio padrão: {perf_stats.get('std_accuracy', 0):.1%}")
            
            # Melhores modelos
            best_models = summary.get('best_models_by_symbol', {})
            if best_models:
                print(f"\n🏅 MELHORES MODELOS POR SÍMBOLO:")
                for symbol, model_info in best_models.items():
                    model_name = model_info.get('model', 'N/A')
                    accuracy = model_info.get('accuracy', 0)
                    print(f"   • {symbol}: {model_name} ({accuracy:.1%})")
            
            # Distribuição de modelos
            model_dist = summary.get('model_distribution', {})
            if model_dist:
                print(f"\n📊 DISTRIBUIÇÃO DE MODELOS:")
                for model, count in model_dist.items():
                    print(f"   • {model}: {count} símbolos")
            
            # Recomendações
            recommendations = summary.get('recommendations', [])
            if recommendations:
                print(f"\n💡 RECOMENDAÇÕES:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print(f"\n✅ Resultados salvos em: symbol_optimization_results/")
            print("="*80)
            
        else:
            print("❌ Erro ao executar otimização por símbolos")
            
    except Exception as e:
        logger.error(f"Erro na otimização por símbolos: {e}")
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    run_symbol_optimization()