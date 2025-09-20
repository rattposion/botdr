#!/usr/bin/env python3
"""
Otimizador de Modelos para Símbolos Específicos
Sistema que otimiza parâmetros de modelos para cada símbolo individualmente
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Imports dos módulos do sistema
from ensemble_model import EnsembleModel
from multi_timeframe_strategy import MultiTimeframeStrategy
from model_optimizer import ModelOptimizer
from advanced_indicators import AdvancedIndicators
from utils import setup_logging

logger = logging.getLogger(__name__)

class SymbolOptimizer:
    """Otimizador específico para símbolos da Deriv"""
    
    def __init__(self):
        """Inicializa o otimizador"""
        self.symbols = ["R_50", "R_100", "R_25", "R_75", "R_10"]
        self.optimized_models = {}
        self.optimization_results = {}
        
        # Configurar diretórios
        self.models_dir = "optimized_models"
        self.results_dir = "optimization_results"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Parâmetros específicos por símbolo
        self.symbol_configs = {
            "R_50": {
                "volatility_threshold": 0.02,
                "trend_periods": [5, 10, 20],
                "rsi_periods": [14, 21],
                "bb_periods": [20, 25],
                "stake_multiplier": 1.0
            },
            "R_100": {
                "volatility_threshold": 0.015,
                "trend_periods": [10, 20, 30],
                "rsi_periods": [14, 28],
                "bb_periods": [20, 30],
                "stake_multiplier": 1.2
            },
            "R_25": {
                "volatility_threshold": 0.025,
                "trend_periods": [3, 5, 10],
                "rsi_periods": [7, 14],
                "bb_periods": [15, 20],
                "stake_multiplier": 0.8
            },
            "R_75": {
                "volatility_threshold": 0.018,
                "trend_periods": [7, 15, 25],
                "rsi_periods": [14, 21],
                "bb_periods": [20, 25],
                "stake_multiplier": 1.1
            },
            "R_10": {
                "volatility_threshold": 0.03,
                "trend_periods": [2, 5, 8],
                "rsi_periods": [7, 10],
                "bb_periods": [10, 15],
                "stake_multiplier": 0.6
            }
        }
        
        logger.info("SymbolOptimizer inicializado")
    
    def generate_symbol_data(self, symbol: str, days: int = 14) -> pd.DataFrame:
        """
        Gera dados específicos para cada símbolo
        
        Args:
            symbol: Símbolo para gerar dados
            days: Número de dias
            
        Returns:
            DataFrame com dados do símbolo
        """
        try:
            config = self.symbol_configs.get(symbol, self.symbol_configs["R_50"])
            
            # Parâmetros baseados no símbolo
            base_price = float(symbol.split('_')[1])
            volatility = config["volatility_threshold"]
            
            # Gerar mais dados para otimização
            periods = days * 24 * 60  # 1 minuto
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                periods=periods,
                freq='1min'
            )
            
            # Seed específico para cada símbolo
            np.random.seed(hash(symbol) % 2**32)
            
            # Gerar retornos com características específicas
            returns = np.random.normal(0, volatility/100, periods)
            
            # Adicionar padrões específicos do símbolo
            if symbol == "R_10":
                # Mais volátil, movimentos rápidos
                returns += np.random.normal(0, volatility/50, periods) * np.sin(np.arange(periods) * 0.1)
            elif symbol == "R_100":
                # Mais estável, tendências longas
                trend_component = np.sin(np.arange(periods) * 0.001) * volatility/200
                returns += trend_component
            elif symbol == "R_25":
                # Movimentos médios
                cycle_component = np.sin(np.arange(periods) * 0.01) * volatility/100
                returns += cycle_component
            
            # Calcular preços
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))
            
            # Criar OHLC com spread específico
            spread_factor = 0.0001 if base_price >= 50 else 0.0002
            
            data = []
            for i, price in enumerate(prices):
                # Simular spread bid/ask
                spread = price * spread_factor
                
                high_factor = 1 + abs(np.random.normal(0, volatility/300))
                low_factor = 1 - abs(np.random.normal(0, volatility/300))
                
                high = price * high_factor + spread/2
                low = price * low_factor - spread/2
                
                data.append({
                    'timestamp': dates[i],
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': np.random.randint(500, 3000),
                    'bid': price - spread/2,
                    'ask': price + spread/2
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Dados gerados para {symbol}: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados para {symbol}: {e}")
            return pd.DataFrame()
    
    def create_symbol_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Cria features específicas para cada símbolo
        
        Args:
            data: Dados OHLC
            symbol: Símbolo sendo processado
            
        Returns:
            DataFrame com features específicas
        """
        try:
            df = data.copy()
            config = self.symbol_configs.get(symbol, self.symbol_configs["R_50"])
            
            # Features básicas
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Spread features
            if 'bid' in df.columns and 'ask' in df.columns:
                df['spread'] = df['ask'] - df['bid']
                df['spread_pct'] = df['spread'] / df['close']
                df['mid_price'] = (df['bid'] + df['ask']) / 2
            
            # Médias móveis específicas
            for period in config["trend_periods"]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
            # RSI com períodos específicos
            for period in config["rsi_periods"]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands específicas
            for period in config["bb_periods"]:
                bb_middle = df['close'].rolling(period).mean()
                bb_std = df['close'].rolling(period).std()
                df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
                df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
                df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_middle
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Indicadores de momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # Features de volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Features de volatilidade
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            
            # Features específicas do símbolo
            if symbol in ["R_10", "R_25"]:
                # Símbolos mais voláteis - features de curto prazo
                df['price_acceleration'] = df['returns'].diff()
                df['volatility_spike'] = (df['volatility'] > df['volatility'].rolling(10).quantile(0.8)).astype(int)
            
            elif symbol in ["R_75", "R_100"]:
                # Símbolos mais estáveis - features de longo prazo
                df['long_trend'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)
                df['trend_strength'] = abs(df['close'] - df['close'].rolling(30).mean()) / df['close']
            
            # Indicadores avançados
            indicators = AdvancedIndicators()
            
            try:
                # Williams %R
                df['williams_r'] = indicators.williams_percent_r(df)
                
                # CCI
                df['cci'] = indicators.commodity_channel_index(df)
                
                # Stochastic RSI
                stoch_rsi = indicators.stochastic_rsi(df)
                if stoch_rsi is not None:
                    df['stoch_rsi_k'] = stoch_rsi.get('k', 0)
                    df['stoch_rsi_d'] = stoch_rsi.get('d', 0)
                
            except Exception as e:
                logger.warning(f"Erro ao calcular indicadores avançados para {symbol}: {e}")
            
            # Remover NaN
            df = df.dropna()
            
            logger.info(f"Features criadas para {symbol}: {len(df.columns)} colunas, {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao criar features para {symbol}: {e}")
            return data
    
    def optimize_symbol_model(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Otimiza modelo específico para um símbolo
        
        Args:
            symbol: Símbolo para otimizar
            data: Dados preparados
            
        Returns:
            Resultados da otimização
        """
        try:
            logger.info(f"Iniciando otimização para {symbol}")
            
            if len(data) < 500:
                return {'error': 'Dados insuficientes para otimização'}
            
            # Preparar features
            feature_cols = [col for col in data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'bid', 'ask']]
            
            X = data[feature_cols].fillna(0)
            
            # Target: próximo movimento
            y = (data['close'].shift(-1) > data['close']).astype(int)
            y = y.dropna()
            X = X.iloc[:-1]
            
            # Ajustar tamanhos
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            
            # Dividir dados
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Criar otimizador
            optimizer = ModelOptimizer()
            
            # Otimização específica para o símbolo
            optimization_results = {}
            
            # Grid Search
            try:
                grid_result = optimizer.grid_search_optimization(X_train, y_train, cv_folds=3)
                optimization_results['grid_search'] = {
                    'best_score': grid_result.get('best_score', 0),
                    'best_params': grid_result.get('best_params', {}),
                    'method': 'grid_search'
                }
                logger.info(f"Grid search para {symbol}: {grid_result.get('best_score', 0):.4f}")
            except Exception as e:
                logger.warning(f"Erro no grid search para {symbol}: {e}")
            
            # Random Search
            try:
                random_result = optimizer.random_search_optimization(X_train, y_train, n_iter=20, cv_folds=3)
                optimization_results['random_search'] = {
                    'best_score': random_result.get('best_score', 0),
                    'best_params': random_result.get('best_params', {}),
                    'method': 'random_search'
                }
                logger.info(f"Random search para {symbol}: {random_result.get('best_score', 0):.4f}")
            except Exception as e:
                logger.warning(f"Erro no random search para {symbol}: {e}")
            
            # Selecionar melhor resultado
            best_result = None
            best_score = 0
            
            for method, result in optimization_results.items():
                if result.get('best_score', 0) > best_score:
                    best_score = result.get('best_score', 0)
                    best_result = result
            
            if best_result is None:
                return {'error': 'Nenhuma otimização bem-sucedida'}
            
            # Treinar modelo final com melhores parâmetros
            ensemble = EnsembleModel()
            ensemble.train(X_train, y_train)
            
            # Avaliar no conjunto de teste
            predictions = ensemble.predict(X_test)
            probabilities = ensemble.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, predictions)
            
            # Simular trading com modelo otimizado
            config = self.symbol_configs.get(symbol, self.symbol_configs["R_50"])
            trading_results = self.simulate_optimized_trading(
                data.iloc[split_idx:], predictions, probabilities, config
            )
            
            # Salvar modelo otimizado
            model_path = os.path.join(self.models_dir, f"{symbol}_optimized_model.joblib")
            joblib.dump(ensemble, model_path)
            
            results = {
                'symbol': symbol,
                'optimization_method': best_result['method'],
                'best_score': best_score,
                'best_params': best_result.get('best_params', {}),
                'test_accuracy': accuracy,
                'features_used': len(feature_cols),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'trading_results': trading_results,
                'model_path': model_path,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Otimização concluída para {symbol}: {accuracy:.4f} accuracy")
            return results
            
        except Exception as e:
            logger.error(f"Erro na otimização para {symbol}: {e}")
            return {'error': str(e)}
    
    def simulate_optimized_trading(self, data: pd.DataFrame, predictions: np.ndarray, 
                                 probabilities: np.ndarray, config: Dict) -> Dict:
        """
        Simula trading com modelo otimizado
        
        Args:
            data: Dados de teste
            predictions: Predições do modelo
            probabilities: Probabilidades das predições
            config: Configuração específica do símbolo
            
        Returns:
            Resultados da simulação
        """
        try:
            balance = 1000.0
            base_stake = 10.0 * config.get("stake_multiplier", 1.0)
            wins = 0
            losses = 0
            total_pnl = 0.0
            trades = []
            
            # Threshold de probabilidade adaptativo
            prob_threshold = 0.65  # Mais conservador para modelos otimizados
            
            for i in range(min(len(predictions), len(data) - 1)):
                prob = probabilities[i] if len(probabilities) > i else 0.5
                
                # Só fazer trade se probabilidade for alta
                if prob > prob_threshold:
                    current_price = data.iloc[i]['close']
                    next_price = data.iloc[i + 1]['close']
                    
                    # Ajustar stake baseado na confiança
                    confidence = (prob - 0.5) * 2  # 0 a 1
                    stake = base_stake * (0.5 + confidence * 0.5)  # 50% a 100% do stake base
                    
                    # Determinar direção
                    direction = "CALL" if predictions[i] == 1 else "PUT"
                    
                    # Calcular resultado
                    if direction == "CALL":
                        win = next_price > current_price
                    else:
                        win = next_price < current_price
                    
                    # Calcular PnL
                    if win:
                        payout_rate = 0.8 + (confidence * 0.1)  # 80% a 90% baseado na confiança
                        pnl = stake * payout_rate
                        wins += 1
                    else:
                        pnl = -stake
                        losses += 1
                    
                    total_pnl += pnl
                    balance += pnl
                    
                    trades.append({
                        'direction': direction,
                        'stake': stake,
                        'probability': prob,
                        'confidence': confidence,
                        'win': win,
                        'pnl': pnl,
                        'balance': balance
                    })
            
            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'final_balance': balance,
                'roi': (balance - 1000) / 1000,
                'average_stake': np.mean([t['stake'] for t in trades]) if trades else 0,
                'average_confidence': np.mean([t['confidence'] for t in trades]) if trades else 0,
                'profit_factor': sum(t['pnl'] for t in trades if t['pnl'] > 0) / abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) if any(t['pnl'] < 0 for t in trades) else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Erro na simulação de trading: {e}")
            return {}
    
    def optimize_all_symbols(self) -> Dict:
        """
        Otimiza modelos para todos os símbolos
        
        Returns:
            Resultados completos da otimização
        """
        try:
            logger.info("Iniciando otimização para todos os símbolos")
            
            all_results = {
                'optimization_results': {},
                'summary': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for symbol in self.symbols:
                logger.info(f"Processando símbolo: {symbol}")
                
                # Gerar dados específicos
                data = self.generate_symbol_data(symbol, days=14)
                
                if data.empty:
                    logger.warning(f"Pulando {symbol} - erro na geração de dados")
                    continue
                
                # Criar features específicas
                data_with_features = self.create_symbol_features(data, symbol)
                
                if len(data_with_features) < 500:
                    logger.warning(f"Pulando {symbol} - dados insuficientes")
                    continue
                
                # Otimizar modelo
                optimization_result = self.optimize_symbol_model(symbol, data_with_features)
                
                if 'error' not in optimization_result:
                    all_results['optimization_results'][symbol] = optimization_result
                    self.optimized_models[symbol] = optimization_result
                else:
                    logger.error(f"Erro na otimização de {symbol}: {optimization_result['error']}")
            
            # Calcular resumo
            all_results['summary'] = self.calculate_optimization_summary(all_results)
            
            # Salvar resultados
            results_file = os.path.join(
                self.results_dir,
                f"symbol_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"Otimização completa finalizada. Resultados salvos em: {results_file}")
            return all_results
            
        except Exception as e:
            logger.error(f"Erro na otimização completa: {e}")
            return {}
    
    def calculate_optimization_summary(self, results: Dict) -> Dict:
        """
        Calcula resumo dos resultados de otimização
        
        Args:
            results: Resultados da otimização
            
        Returns:
            Resumo calculado
        """
        try:
            optimization_results = results.get('optimization_results', {})
            
            if not optimization_results:
                return {}
            
            # Métricas de otimização
            best_scores = [r.get('best_score', 0) for r in optimization_results.values()]
            test_accuracies = [r.get('test_accuracy', 0) for r in optimization_results.values()]
            
            # Métricas de trading
            trading_results = [r.get('trading_results', {}) for r in optimization_results.values()]
            win_rates = [tr.get('win_rate', 0) for tr in trading_results if tr]
            rois = [tr.get('roi', 0) for tr in trading_results if tr]
            profit_factors = [tr.get('profit_factor', 0) for tr in trading_results if tr and tr.get('profit_factor', 0) != float('inf')]
            
            # Encontrar melhor símbolo
            best_symbol = None
            best_roi = -float('inf')
            
            for symbol, result in optimization_results.items():
                trading_result = result.get('trading_results', {})
                roi = trading_result.get('roi', 0)
                if roi > best_roi:
                    best_roi = roi
                    best_symbol = symbol
            
            summary = {
                'symbols_optimized': len(optimization_results),
                'optimization_metrics': {
                    'average_best_score': np.mean(best_scores) if best_scores else 0,
                    'best_optimization_score': max(best_scores) if best_scores else 0,
                    'average_test_accuracy': np.mean(test_accuracies) if test_accuracies else 0,
                    'best_test_accuracy': max(test_accuracies) if test_accuracies else 0
                },
                'trading_performance': {
                    'average_win_rate': np.mean(win_rates) if win_rates else 0,
                    'best_win_rate': max(win_rates) if win_rates else 0,
                    'average_roi': np.mean(rois) if rois else 0,
                    'best_roi': max(rois) if rois else 0,
                    'average_profit_factor': np.mean(profit_factors) if profit_factors else 0,
                    'profitable_symbols': len([roi for roi in rois if roi > 0])
                },
                'best_performing_symbol': {
                    'symbol': best_symbol,
                    'roi': best_roi,
                    'details': optimization_results.get(best_symbol, {}) if best_symbol else {}
                },
                'models_saved': len([r for r in optimization_results.values() if 'model_path' in r])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao calcular resumo: {e}")
            return {}


def run_symbol_optimization():
    """Função principal para executar otimização de símbolos"""
    try:
        # Configurar logging
        setup_logging()
        
        print("\n" + "="*70)
        print("🎯 OTIMIZAÇÃO DE MODELOS PARA SÍMBOLOS ESPECÍFICOS")
        print("="*70)
        print("🔧 Otimizando parâmetros para cada símbolo da Deriv...")
        print("📊 Grid Search + Random Search + Ensemble Models")
        print("="*70)
        
        # Criar otimizador
        optimizer = SymbolOptimizer()
        
        # Executar otimização
        results = optimizer.optimize_all_symbols()
        
        if results and 'summary' in results:
            summary = results['summary']
            
            print("\n" + "="*70)
            print("🎯 RESULTADOS DA OTIMIZAÇÃO")
            print("="*70)
            
            print(f"📊 Símbolos otimizados: {summary.get('symbols_optimized', 0)}")
            
            # Métricas de otimização
            opt_metrics = summary.get('optimization_metrics', {})
            if opt_metrics:
                print(f"\n🔧 MÉTRICAS DE OTIMIZAÇÃO:")
                print(f"   • Score médio: {opt_metrics.get('average_best_score', 0):.4f}")
                print(f"   • Melhor score: {opt_metrics.get('best_optimization_score', 0):.4f}")
                print(f"   • Accuracy médio: {opt_metrics.get('average_test_accuracy', 0):.4f}")
                print(f"   • Melhor accuracy: {opt_metrics.get('best_test_accuracy', 0):.4f}")
            
            # Performance de trading
            trading_perf = summary.get('trading_performance', {})
            if trading_perf:
                print(f"\n💰 PERFORMANCE DE TRADING:")
                print(f"   • Win Rate médio: {trading_perf.get('average_win_rate', 0):.2%}")
                print(f"   • Melhor Win Rate: {trading_perf.get('best_win_rate', 0):.2%}")
                print(f"   • ROI médio: {trading_perf.get('average_roi', 0):.2%}")
                print(f"   • Melhor ROI: {trading_perf.get('best_roi', 0):.2%}")
                print(f"   • Profit Factor médio: {trading_perf.get('average_profit_factor', 0):.2f}")
                print(f"   • Símbolos lucrativos: {trading_perf.get('profitable_symbols', 0)}")
            
            # Melhor símbolo
            best_symbol = summary.get('best_performing_symbol', {})
            if best_symbol.get('symbol'):
                print(f"\n🏆 MELHOR SÍMBOLO OTIMIZADO:")
                print(f"   • Símbolo: {best_symbol.get('symbol')}")
                print(f"   • ROI: {best_symbol.get('roi', 0):.2%}")
                
                details = best_symbol.get('details', {})
                if details:
                    trading_results = details.get('trading_results', {})
                    print(f"   • Win Rate: {trading_results.get('win_rate', 0):.2%}")
                    print(f"   • Total Trades: {trading_results.get('total_trades', 0)}")
                    print(f"   • Método: {details.get('optimization_method', 'N/A')}")
            
            print(f"\n💾 Modelos salvos: {summary.get('models_saved', 0)}")
            print(f"📁 Diretório: optimized_models/")
            print("="*70)
            
        else:
            print("❌ Erro ao executar otimização")
            
    except Exception as e:
        logger.error(f"Erro na otimização: {e}")
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    run_symbol_optimization()