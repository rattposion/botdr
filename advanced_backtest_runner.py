#!/usr/bin/env python3
"""
Sistema Avançado de Backtesting
Integra ensemble models, multi-timeframe e otimização de hiperparâmetros
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import asyncio

# Imports dos módulos do sistema
from data_collector import DerivDataCollector
from feature_engineering import FeatureEngineer
from advanced_indicators import AdvancedIndicators
from ensemble_model import EnsembleModel
from multi_timeframe_strategy import MultiTimeframeStrategy
from model_optimizer import ModelOptimizer
from advanced_backtester import AdvancedBacktester
from utils import setup_logging

logger = logging.getLogger(__name__)

class AdvancedBacktestRunner:
    """Sistema completo de backtesting com funcionalidades avançadas"""
    
    def __init__(self, symbols: List[str] = None):
        """
        Inicializa o sistema de backtesting avançado
        
        Args:
            symbols: Lista de símbolos para testar
        """
        self.symbols = symbols or ["R_50", "R_100", "R_25", "R_75"]
        self.data_collector = DerivDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.advanced_indicators = AdvancedIndicators()
        self.results = {}
        
        # Configurar diretórios
        self.results_dir = "backtest_results/advanced"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"AdvancedBacktestRunner inicializado para símbolos: {self.symbols}")
    
    def prepare_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Prepara dados históricos com features avançadas
        
        Args:
            symbol: Símbolo para coletar dados
            days: Número de dias de histórico
            
        Returns:
            DataFrame com dados preparados
        """
        try:
            logger.info(f"Coletando dados para {symbol} ({days} dias)")
            
            # Conectar e coletar dados
            self.data_collector.connect()
            
            # Calcular quantidade de candles (assumindo 1 minuto)
            count = days * 24 * 60
            
            # Coletar dados históricos
            data = self.data_collector.get_historical_data(
                symbol=symbol,
                timeframe="1m",
                count=count
            )
            
            if data is None or len(data) < 100:
                logger.warning(f"Dados insuficientes para {symbol}")
                return pd.DataFrame()
            
            # Converter para DataFrame se necessário
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Garantir colunas necessárias
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Colunas necessárias não encontradas em {symbol}")
                return pd.DataFrame()
            
            # Converter timestamp se necessário
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Adicionar volume se não existir
            if 'volume' not in df.columns:
                df['volume'] = 1000
            
            # Gerar features básicas
            df = self.feature_engineer.create_features(df)
            
            # Adicionar indicadores avançados
            advanced_features = self.advanced_indicators.calculate_all_indicators(df)
            for name, values in advanced_features.items():
                if isinstance(values, pd.Series) and len(values) == len(df):
                    df[name] = values
            
            # Remover NaN
            df = df.dropna()
            
            logger.info(f"Dados preparados para {symbol}: {len(df)} registros, {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados para {symbol}: {e}")
            return pd.DataFrame()
        finally:
            self.data_collector.disconnect()
    
    def run_ensemble_backtest(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Executa backtest com modelo ensemble
        
        Args:
            symbol: Símbolo sendo testado
            data: Dados preparados
            
        Returns:
            Resultados do backtest ensemble
        """
        try:
            logger.info(f"Executando backtest ensemble para {symbol}")
            
            if len(data) < 200:
                logger.warning(f"Dados insuficientes para ensemble em {symbol}")
                return {}
            
            # Preparar dados para ML
            feature_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = data[feature_cols].fillna(0)
            
            # Criar target (próximo movimento)
            y = (data['close'].shift(-1) > data['close']).astype(int)
            y = y.dropna()
            X = X.iloc[:-1]  # Ajustar tamanho
            
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
            
            # Dividir dados
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Criar e treinar ensemble
            ensemble = EnsembleModel()
            ensemble.train(X_train, y_train)
            
            # Fazer predições
            predictions = ensemble.predict(X_test)
            probabilities = ensemble.predict_proba(X_test)
            
            # Executar backtest
            backtester = AdvancedBacktester(
                initial_balance=1000,
                stake_amount=10,
                symbol=symbol
            )
            
            # Simular trades baseados nas predições
            test_data = data.iloc[split_idx:split_idx+len(predictions)]
            
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if i >= len(test_data):
                    break
                    
                current_price = test_data.iloc[i]['close']
                timestamp = test_data.index[i]
                
                # Só fazer trade se probabilidade for alta
                if prob > 0.6:
                    contract_type = "CALL" if pred == 1 else "PUT"
                    
                    # Simular resultado do trade
                    if i + 1 < len(test_data):
                        next_price = test_data.iloc[i + 1]['close']
                        
                        if contract_type == "CALL":
                            win = next_price > current_price
                        else:
                            win = next_price < current_price
                        
                        # Adicionar trade ao backtester
                        backtester.add_trade_result(
                            timestamp=timestamp,
                            contract_type=contract_type,
                            stake=10,
                            payout=18 if win else 0,
                            win=win,
                            entry_price=current_price,
                            exit_price=next_price
                        )
            
            # Calcular métricas
            results = backtester.calculate_metrics()
            results['model_type'] = 'ensemble'
            results['symbol'] = symbol
            results['total_trades'] = len(backtester.trades)
            
            logger.info(f"Backtest ensemble {symbol} concluído: {results.get('win_rate', 0):.2%} win rate")
            return results
            
        except Exception as e:
            logger.error(f"Erro no backtest ensemble para {symbol}: {e}")
            return {}
    
    def run_multi_timeframe_backtest(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Executa backtest com estratégia multi-timeframe
        
        Args:
            symbol: Símbolo sendo testado
            data: Dados preparados
            
        Returns:
            Resultados do backtest multi-timeframe
        """
        try:
            logger.info(f"Executando backtest multi-timeframe para {symbol}")
            
            # Criar estratégia multi-timeframe
            strategy = MultiTimeframeStrategy(symbol=symbol)
            
            # Simular coleta de dados para diferentes timeframes
            timeframes_data = {}
            
            for tf_name, tf_seconds in strategy.timeframes.items():
                # Resample data para o timeframe
                tf_data = data.resample(f'{tf_seconds}s').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(tf_data) > 50:
                    timeframes_data[tf_name] = tf_data
            
            if not timeframes_data:
                logger.warning(f"Não foi possível criar dados multi-timeframe para {symbol}")
                return {}
            
            # Executar backtest
            backtester = AdvancedBacktester(
                initial_balance=1000,
                stake_amount=10,
                symbol=symbol
            )
            
            # Simular sinais multi-timeframe
            base_data = timeframes_data.get('1m', data)
            
            for i in range(50, len(base_data) - 1):
                current_data = base_data.iloc[:i+1]
                
                # Gerar sinal multi-timeframe (simplificado)
                signals = {}
                for tf_name, tf_data in timeframes_data.items():
                    if len(tf_data) > 10:
                        # Sinal baseado em tendência simples
                        recent_data = tf_data.tail(10)
                        trend = 1 if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else -1
                        signals[tf_name] = trend
                
                # Combinar sinais
                if signals:
                    combined_signal = sum(signals.values())
                    
                    if abs(combined_signal) >= 2:  # Pelo menos 2 timeframes concordam
                        current_price = current_data.iloc[-1]['close']
                        timestamp = current_data.index[-1]
                        
                        contract_type = "CALL" if combined_signal > 0 else "PUT"
                        
                        # Simular resultado
                        next_price = base_data.iloc[i + 1]['close']
                        
                        if contract_type == "CALL":
                            win = next_price > current_price
                        else:
                            win = next_price < current_price
                        
                        backtester.add_trade_result(
                            timestamp=timestamp,
                            contract_type=contract_type,
                            stake=10,
                            payout=18 if win else 0,
                            win=win,
                            entry_price=current_price,
                            exit_price=next_price
                        )
            
            # Calcular métricas
            results = backtester.calculate_metrics()
            results['model_type'] = 'multi_timeframe'
            results['symbol'] = symbol
            results['total_trades'] = len(backtester.trades)
            results['timeframes_used'] = list(timeframes_data.keys())
            
            logger.info(f"Backtest multi-timeframe {symbol} concluído: {results.get('win_rate', 0):.2%} win rate")
            return results
            
        except Exception as e:
            logger.error(f"Erro no backtest multi-timeframe para {symbol}: {e}")
            return {}
    
    def optimize_model_for_symbol(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Otimiza hiperparâmetros do modelo para um símbolo específico
        
        Args:
            symbol: Símbolo para otimizar
            data: Dados preparados
            
        Returns:
            Resultados da otimização
        """
        try:
            logger.info(f"Otimizando modelo para {symbol}")
            
            if len(data) < 200:
                logger.warning(f"Dados insuficientes para otimização em {symbol}")
                return {}
            
            # Preparar dados
            feature_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = data[feature_cols].fillna(0)
            y = (data['close'].shift(-1) > data['close']).astype(int).dropna()
            X = X.iloc[:-1]
            
            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
            
            # Criar otimizador
            optimizer = ModelOptimizer()
            
            # Definir espaço de parâmetros
            param_space = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            # Executar otimização (versão simplificada)
            best_params = optimizer.grid_search(X, y, param_space, cv=3)
            
            # Salvar resultados
            optimization_results = {
                'symbol': symbol,
                'best_params': best_params,
                'optimization_method': 'grid_search',
                'timestamp': datetime.now().isoformat(),
                'data_points': len(X),
                'features': len(feature_cols)
            }
            
            # Salvar em arquivo
            results_file = os.path.join(self.results_dir, f"optimization_{symbol}.json")
            with open(results_file, 'w') as f:
                json.dump(optimization_results, f, indent=2)
            
            logger.info(f"Otimização para {symbol} concluída: {best_params}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Erro na otimização para {symbol}: {e}")
            return {}
    
    def run_comprehensive_backtest(self, days: int = 30) -> Dict:
        """
        Executa backtest completo para todos os símbolos e estratégias
        
        Args:
            days: Número de dias de dados históricos
            
        Returns:
            Resultados completos dos backtests
        """
        try:
            logger.info("Iniciando backtest completo com estratégias avançadas")
            
            all_results = {
                'ensemble_results': {},
                'multi_timeframe_results': {},
                'optimization_results': {},
                'summary': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for symbol in self.symbols:
                logger.info(f"Processando símbolo: {symbol}")
                
                # Preparar dados
                data = self.prepare_data(symbol, days)
                
                if data.empty:
                    logger.warning(f"Pulando {symbol} - dados insuficientes")
                    continue
                
                # Backtest ensemble
                ensemble_results = self.run_ensemble_backtest(symbol, data)
                if ensemble_results:
                    all_results['ensemble_results'][symbol] = ensemble_results
                
                # Backtest multi-timeframe
                mtf_results = self.run_multi_timeframe_backtest(symbol, data)
                if mtf_results:
                    all_results['multi_timeframe_results'][symbol] = mtf_results
                
                # Otimização de modelo
                opt_results = self.optimize_model_for_symbol(symbol, data)
                if opt_results:
                    all_results['optimization_results'][symbol] = opt_results
            
            # Calcular resumo
            all_results['summary'] = self.calculate_summary(all_results)
            
            # Salvar resultados completos
            results_file = os.path.join(self.results_dir, f"comprehensive_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"Backtest completo finalizado. Resultados salvos em: {results_file}")
            return all_results
            
        except Exception as e:
            logger.error(f"Erro no backtest completo: {e}")
            return {}
    
    def calculate_summary(self, results: Dict) -> Dict:
        """
        Calcula resumo dos resultados
        
        Args:
            results: Resultados dos backtests
            
        Returns:
            Resumo calculado
        """
        try:
            summary = {
                'total_symbols_tested': 0,
                'ensemble_performance': {},
                'multi_timeframe_performance': {},
                'best_performing_strategy': {},
                'optimization_summary': {}
            }
            
            # Resumo ensemble
            ensemble_results = results.get('ensemble_results', {})
            if ensemble_results:
                win_rates = [r.get('win_rate', 0) for r in ensemble_results.values() if 'win_rate' in r]
                total_trades = sum(r.get('total_trades', 0) for r in ensemble_results.values())
                
                summary['ensemble_performance'] = {
                    'symbols_tested': len(ensemble_results),
                    'average_win_rate': np.mean(win_rates) if win_rates else 0,
                    'best_win_rate': max(win_rates) if win_rates else 0,
                    'total_trades': total_trades
                }
            
            # Resumo multi-timeframe
            mtf_results = results.get('multi_timeframe_results', {})
            if mtf_results:
                win_rates = [r.get('win_rate', 0) for r in mtf_results.values() if 'win_rate' in r]
                total_trades = sum(r.get('total_trades', 0) for r in mtf_results.values())
                
                summary['multi_timeframe_performance'] = {
                    'symbols_tested': len(mtf_results),
                    'average_win_rate': np.mean(win_rates) if win_rates else 0,
                    'best_win_rate': max(win_rates) if win_rates else 0,
                    'total_trades': total_trades
                }
            
            # Melhor estratégia
            all_performances = []
            
            for symbol, result in ensemble_results.items():
                if 'win_rate' in result:
                    all_performances.append({
                        'symbol': symbol,
                        'strategy': 'ensemble',
                        'win_rate': result['win_rate'],
                        'total_trades': result.get('total_trades', 0)
                    })
            
            for symbol, result in mtf_results.items():
                if 'win_rate' in result:
                    all_performances.append({
                        'symbol': symbol,
                        'strategy': 'multi_timeframe',
                        'win_rate': result['win_rate'],
                        'total_trades': result.get('total_trades', 0)
                    })
            
            if all_performances:
                best_performance = max(all_performances, key=lambda x: x['win_rate'])
                summary['best_performing_strategy'] = best_performance
            
            # Resumo otimização
            opt_results = results.get('optimization_results', {})
            summary['optimization_summary'] = {
                'symbols_optimized': len(opt_results),
                'optimization_method': 'grid_search'
            }
            
            summary['total_symbols_tested'] = len(set(
                list(ensemble_results.keys()) + 
                list(mtf_results.keys()) + 
                list(opt_results.keys())
            ))
            
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao calcular resumo: {e}")
            return {}


def run_advanced_backtests():
    """Função principal para executar backtests avançados"""
    try:
        # Configurar logging
        setup_logging()
        
        # Criar runner
        runner = AdvancedBacktestRunner()
        
        # Executar backtests
        results = runner.run_comprehensive_backtest(days=7)  # 7 dias para teste rápido
        
        if results:
            print("\n" + "="*60)
            print("🚀 RESULTADOS DO BACKTEST AVANÇADO")
            print("="*60)
            
            summary = results.get('summary', {})
            
            print(f"📊 Símbolos testados: {summary.get('total_symbols_tested', 0)}")
            
            # Ensemble performance
            ensemble_perf = summary.get('ensemble_performance', {})
            if ensemble_perf:
                print(f"\n🤖 ENSEMBLE MODEL:")
                print(f"   • Win Rate Médio: {ensemble_perf.get('average_win_rate', 0):.2%}")
                print(f"   • Melhor Win Rate: {ensemble_perf.get('best_win_rate', 0):.2%}")
                print(f"   • Total de Trades: {ensemble_perf.get('total_trades', 0)}")
            
            # Multi-timeframe performance
            mtf_perf = summary.get('multi_timeframe_performance', {})
            if mtf_perf:
                print(f"\n📈 MULTI-TIMEFRAME:")
                print(f"   • Win Rate Médio: {mtf_perf.get('average_win_rate', 0):.2%}")
                print(f"   • Melhor Win Rate: {mtf_perf.get('best_win_rate', 0):.2%}")
                print(f"   • Total de Trades: {mtf_perf.get('total_trades', 0)}")
            
            # Melhor estratégia
            best_strategy = summary.get('best_performing_strategy', {})
            if best_strategy:
                print(f"\n🏆 MELHOR ESTRATÉGIA:")
                print(f"   • Símbolo: {best_strategy.get('symbol', 'N/A')}")
                print(f"   • Estratégia: {best_strategy.get('strategy', 'N/A')}")
                print(f"   • Win Rate: {best_strategy.get('win_rate', 0):.2%}")
                print(f"   • Trades: {best_strategy.get('total_trades', 0)}")
            
            print(f"\n✅ Resultados salvos em: backtest_results/advanced/")
            
        else:
            print("❌ Erro ao executar backtests")
            
    except Exception as e:
        logger.error(f"Erro na execução dos backtests: {e}")
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    run_advanced_backtests()