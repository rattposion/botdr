#!/usr/bin/env python3
"""
Demo do Sistema de Backtesting Avançado
Executa backtests com dados simulados para demonstração
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional

# Imports dos módulos do sistema
from ensemble_model import EnsembleModel
from multi_timeframe_strategy import MultiTimeframeStrategy
from model_optimizer import ModelOptimizer
from advanced_indicators import AdvancedIndicators
from utils import setup_logging

logger = logging.getLogger(__name__)

class DemoBacktestRunner:
    """Sistema de demonstração de backtesting com dados simulados"""
    
    def __init__(self):
        """Inicializa o sistema de demo"""
        self.symbols = ["R_50", "R_100", "R_25", "R_75"]
        self.results = {}
        
        # Configurar diretórios
        self.results_dir = "backtest_results/demo"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("DemoBacktestRunner inicializado")
    
    def generate_synthetic_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Gera dados sintéticos para demonstração
        
        Args:
            symbol: Símbolo para simular
            days: Número de dias de dados
            
        Returns:
            DataFrame com dados sintéticos
        """
        try:
            # Configurar parâmetros baseados no símbolo
            if symbol == "R_50":
                base_price = 50.0
                volatility = 0.02
            elif symbol == "R_100":
                base_price = 100.0
                volatility = 0.015
            elif symbol == "R_25":
                base_price = 25.0
                volatility = 0.025
            else:  # R_75
                base_price = 75.0
                volatility = 0.018
            
            # Gerar timestamps (1 minuto)
            periods = days * 24 * 60
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                periods=periods,
                freq='1min'
            )
            
            # Gerar preços usando random walk
            np.random.seed(hash(symbol) % 2**32)  # Seed baseado no símbolo
            
            returns = np.random.normal(0, volatility/100, periods)
            
            # Adicionar tendência sutil
            trend = np.linspace(0, 0.001, periods)
            returns += trend
            
            # Calcular preços
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))  # Evitar preços negativos
            
            # Criar OHLC
            data = []
            for i in range(len(prices)):
                price = prices[i]
                
                # Simular high/low baseado na volatilidade
                high_factor = 1 + abs(np.random.normal(0, volatility/200))
                low_factor = 1 - abs(np.random.normal(0, volatility/200))
                
                high = price * high_factor
                low = price * low_factor
                
                data.append({
                    'timestamp': dates[i],
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': np.random.randint(1000, 5000)
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Dados sintéticos gerados para {symbol}: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados para {symbol}: {e}")
            return pd.DataFrame()
    
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features técnicas aos dados
        
        Args:
            data: DataFrame com dados OHLC
            
        Returns:
            DataFrame com features adicionadas
        """
        try:
            df = data.copy()
            
            # Features básicas
            df['returns'] = df['close'].pct_change()
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['volatility'] = df['returns'].rolling(20).std()
            
            # RSI simplificado
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_std_val = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD simplificado
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Indicadores avançados
            indicators = AdvancedIndicators()
            
            # Williams %R
            df['williams_r'] = indicators.williams_percent_r(df)
            
            # CCI
            df['cci'] = indicators.commodity_channel_index(df)
            
            # Ichimoku (componentes principais)
            ichimoku = indicators.ichimoku_cloud(df)
            if ichimoku:
                df['tenkan_sen'] = ichimoku.get('tenkan_sen', 0)
                df['kijun_sen'] = ichimoku.get('kijun_sen', 0)
            
            # Remover NaN
            df = df.dropna()
            
            logger.info(f"Features adicionadas: {len(df.columns)} colunas, {len(df)} registros válidos")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao adicionar features: {e}")
            return data
    
    def run_ensemble_demo(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Demonstração do modelo ensemble
        
        Args:
            symbol: Símbolo sendo testado
            data: Dados preparados
            
        Returns:
            Resultados do teste ensemble
        """
        try:
            logger.info(f"Executando demo ensemble para {symbol}")
            
            if len(data) < 200:
                return {'error': 'Dados insuficientes'}
            
            # Preparar features
            feature_cols = [col for col in data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            X = data[feature_cols].fillna(0)
            
            # Target: próximo movimento (simplificado)
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
            
            # Criar e treinar ensemble
            ensemble = EnsembleModel()
            ensemble.train(X_train, y_train)
            
            # Fazer predições
            predictions = ensemble.predict(X_test)
            probabilities = ensemble.predict_proba(X_test)
            
            # Calcular métricas
            accuracy = np.mean(predictions == y_test)
            
            # Simular trading baseado nas predições
            balance = 1000.0
            stake = 10.0
            wins = 0
            losses = 0
            total_pnl = 0.0
            
            test_data = data.iloc[split_idx:split_idx+len(predictions)]
            
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if i >= len(test_data) - 1:
                    break
                
                # Só fazer trade se probabilidade for alta
                if prob > 0.6:
                    current_price = test_data.iloc[i]['close']
                    next_price = test_data.iloc[i + 1]['close']
                    
                    # Determinar resultado
                    if pred == 1:  # CALL
                        win = next_price > current_price
                    else:  # PUT
                        win = next_price < current_price
                    
                    # Calcular PnL
                    if win:
                        pnl = stake * 0.8  # 80% payout
                        wins += 1
                    else:
                        pnl = -stake
                        losses += 1
                    
                    total_pnl += pnl
                    balance += pnl
            
            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            results = {
                'symbol': symbol,
                'model_type': 'ensemble',
                'accuracy': accuracy,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'final_balance': balance,
                'roi': (balance - 1000) / 1000,
                'features_used': len(feature_cols)
            }
            
            logger.info(f"Demo ensemble {symbol}: {win_rate:.2%} win rate, {total_trades} trades")
            return results
            
        except Exception as e:
            logger.error(f"Erro no demo ensemble para {symbol}: {e}")
            return {'error': str(e)}
    
    def run_multi_timeframe_demo(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Demonstração da estratégia multi-timeframe
        
        Args:
            symbol: Símbolo sendo testado
            data: Dados preparados
            
        Returns:
            Resultados do teste multi-timeframe
        """
        try:
            logger.info(f"Executando demo multi-timeframe para {symbol}")
            
            if len(data) < 200:
                return {'error': 'Dados insuficientes'}
            
            # Criar diferentes timeframes
            timeframes = {
                '1m': data,
                '5m': data.resample('5min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna(),
                '15m': data.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            }
            
            # Simular trading multi-timeframe
            balance = 1000.0
            stake = 10.0
            wins = 0
            losses = 0
            total_pnl = 0.0
            
            base_data = timeframes['1m']
            
            for i in range(100, len(base_data) - 1):
                current_time = base_data.index[i]
                current_price = base_data.iloc[i]['close']
                next_price = base_data.iloc[i + 1]['close']
                
                # Gerar sinais para cada timeframe
                signals = {}
                
                for tf_name, tf_data in timeframes.items():
                    if len(tf_data) > 20:
                        # Sinal baseado em tendência simples
                        recent_data = tf_data[tf_data.index <= current_time].tail(10)
                        if len(recent_data) >= 2:
                            trend = 1 if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else -1
                            signals[tf_name] = trend
                
                # Combinar sinais
                if len(signals) >= 2:
                    combined_signal = sum(signals.values())
                    
                    # Fazer trade se pelo menos 2 timeframes concordam
                    if abs(combined_signal) >= 2:
                        if combined_signal > 0:  # CALL
                            win = next_price > current_price
                        else:  # PUT
                            win = next_price < current_price
                        
                        # Calcular PnL
                        if win:
                            pnl = stake * 0.8
                            wins += 1
                        else:
                            pnl = -stake
                            losses += 1
                        
                        total_pnl += pnl
                        balance += pnl
            
            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            results = {
                'symbol': symbol,
                'model_type': 'multi_timeframe',
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'final_balance': balance,
                'roi': (balance - 1000) / 1000,
                'timeframes_used': list(timeframes.keys())
            }
            
            logger.info(f"Demo multi-timeframe {symbol}: {win_rate:.2%} win rate, {total_trades} trades")
            return results
            
        except Exception as e:
            logger.error(f"Erro no demo multi-timeframe para {symbol}: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_demo(self) -> Dict:
        """
        Executa demonstração completa para todos os símbolos
        
        Returns:
            Resultados completos da demonstração
        """
        try:
            logger.info("Iniciando demonstração completa do sistema de backtesting")
            
            all_results = {
                'ensemble_results': {},
                'multi_timeframe_results': {},
                'summary': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for symbol in self.symbols:
                logger.info(f"Processando símbolo: {symbol}")
                
                # Gerar dados sintéticos
                data = self.generate_synthetic_data(symbol, days=7)
                
                if data.empty:
                    logger.warning(f"Pulando {symbol} - erro na geração de dados")
                    continue
                
                # Adicionar features
                data_with_features = self.add_features(data)
                
                if len(data_with_features) < 100:
                    logger.warning(f"Pulando {symbol} - dados insuficientes após features")
                    continue
                
                # Demo ensemble
                ensemble_results = self.run_ensemble_demo(symbol, data_with_features)
                if 'error' not in ensemble_results:
                    all_results['ensemble_results'][symbol] = ensemble_results
                
                # Demo multi-timeframe
                mtf_results = self.run_multi_timeframe_demo(symbol, data_with_features)
                if 'error' not in mtf_results:
                    all_results['multi_timeframe_results'][symbol] = mtf_results
            
            # Calcular resumo
            all_results['summary'] = self.calculate_summary(all_results)
            
            # Salvar resultados
            results_file = os.path.join(
                self.results_dir, 
                f"demo_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"Demonstração completa finalizada. Resultados salvos em: {results_file}")
            return all_results
            
        except Exception as e:
            logger.error(f"Erro na demonstração completa: {e}")
            return {}
    
    def calculate_summary(self, results: Dict) -> Dict:
        """
        Calcula resumo dos resultados
        
        Args:
            results: Resultados dos testes
            
        Returns:
            Resumo calculado
        """
        try:
            summary = {
                'total_symbols_tested': 0,
                'ensemble_performance': {},
                'multi_timeframe_performance': {},
                'best_performing_strategy': {},
                'overall_performance': {}
            }
            
            # Resumo ensemble
            ensemble_results = results.get('ensemble_results', {})
            if ensemble_results:
                win_rates = [r.get('win_rate', 0) for r in ensemble_results.values()]
                rois = [r.get('roi', 0) for r in ensemble_results.values()]
                total_trades = sum(r.get('total_trades', 0) for r in ensemble_results.values())
                
                summary['ensemble_performance'] = {
                    'symbols_tested': len(ensemble_results),
                    'average_win_rate': np.mean(win_rates) if win_rates else 0,
                    'best_win_rate': max(win_rates) if win_rates else 0,
                    'average_roi': np.mean(rois) if rois else 0,
                    'best_roi': max(rois) if rois else 0,
                    'total_trades': total_trades
                }
            
            # Resumo multi-timeframe
            mtf_results = results.get('multi_timeframe_results', {})
            if mtf_results:
                win_rates = [r.get('win_rate', 0) for r in mtf_results.values()]
                rois = [r.get('roi', 0) for r in mtf_results.values()]
                total_trades = sum(r.get('total_trades', 0) for r in mtf_results.values())
                
                summary['multi_timeframe_performance'] = {
                    'symbols_tested': len(mtf_results),
                    'average_win_rate': np.mean(win_rates) if win_rates else 0,
                    'best_win_rate': max(win_rates) if win_rates else 0,
                    'average_roi': np.mean(rois) if rois else 0,
                    'best_roi': max(rois) if rois else 0,
                    'total_trades': total_trades
                }
            
            # Melhor estratégia
            all_performances = []
            
            for symbol, result in ensemble_results.items():
                all_performances.append({
                    'symbol': symbol,
                    'strategy': 'ensemble',
                    'win_rate': result.get('win_rate', 0),
                    'roi': result.get('roi', 0),
                    'total_trades': result.get('total_trades', 0)
                })
            
            for symbol, result in mtf_results.items():
                all_performances.append({
                    'symbol': symbol,
                    'strategy': 'multi_timeframe',
                    'win_rate': result.get('win_rate', 0),
                    'roi': result.get('roi', 0),
                    'total_trades': result.get('total_trades', 0)
                })
            
            if all_performances:
                best_performance = max(all_performances, key=lambda x: x['roi'])
                summary['best_performing_strategy'] = best_performance
                
                # Performance geral
                all_win_rates = [p['win_rate'] for p in all_performances]
                all_rois = [p['roi'] for p in all_performances]
                
                summary['overall_performance'] = {
                    'total_strategies_tested': len(all_performances),
                    'average_win_rate': np.mean(all_win_rates),
                    'average_roi': np.mean(all_rois),
                    'profitable_strategies': len([p for p in all_performances if p['roi'] > 0])
                }
            
            summary['total_symbols_tested'] = len(set(
                list(ensemble_results.keys()) + list(mtf_results.keys())
            ))
            
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao calcular resumo: {e}")
            return {}


def run_demo_backtests():
    """Função principal para executar demonstração dos backtests"""
    try:
        # Configurar logging
        setup_logging()
        
        print("\n" + "="*60)
        print("🚀 DEMONSTRAÇÃO DO SISTEMA DE BACKTESTING AVANÇADO")
        print("="*60)
        print("📊 Testando estratégias com dados sintéticos...")
        print("🤖 Ensemble Models + 📈 Multi-Timeframe Analysis")
        print("="*60)
        
        # Criar runner
        runner = DemoBacktestRunner()
        
        # Executar demonstração
        results = runner.run_comprehensive_demo()
        
        if results and 'summary' in results:
            summary = results['summary']
            
            print("\n" + "="*60)
            print("📈 RESULTADOS DA DEMONSTRAÇÃO")
            print("="*60)
            
            print(f"📊 Símbolos testados: {summary.get('total_symbols_tested', 0)}")
            
            # Ensemble performance
            ensemble_perf = summary.get('ensemble_performance', {})
            if ensemble_perf:
                print(f"\n🤖 ENSEMBLE MODELS:")
                print(f"   • Símbolos testados: {ensemble_perf.get('symbols_tested', 0)}")
                print(f"   • Win Rate Médio: {ensemble_perf.get('average_win_rate', 0):.2%}")
                print(f"   • Melhor Win Rate: {ensemble_perf.get('best_win_rate', 0):.2%}")
                print(f"   • ROI Médio: {ensemble_perf.get('average_roi', 0):.2%}")
                print(f"   • Melhor ROI: {ensemble_perf.get('best_roi', 0):.2%}")
                print(f"   • Total de Trades: {ensemble_perf.get('total_trades', 0)}")
            
            # Multi-timeframe performance
            mtf_perf = summary.get('multi_timeframe_performance', {})
            if mtf_perf:
                print(f"\n📈 MULTI-TIMEFRAME:")
                print(f"   • Símbolos testados: {mtf_perf.get('symbols_tested', 0)}")
                print(f"   • Win Rate Médio: {mtf_perf.get('average_win_rate', 0):.2%}")
                print(f"   • Melhor Win Rate: {mtf_perf.get('best_win_rate', 0):.2%}")
                print(f"   • ROI Médio: {mtf_perf.get('average_roi', 0):.2%}")
                print(f"   • Melhor ROI: {mtf_perf.get('best_roi', 0):.2%}")
                print(f"   • Total de Trades: {mtf_perf.get('total_trades', 0)}")
            
            # Melhor estratégia
            best_strategy = summary.get('best_performing_strategy', {})
            if best_strategy:
                print(f"\n🏆 MELHOR ESTRATÉGIA:")
                print(f"   • Símbolo: {best_strategy.get('symbol', 'N/A')}")
                print(f"   • Estratégia: {best_strategy.get('strategy', 'N/A')}")
                print(f"   • Win Rate: {best_strategy.get('win_rate', 0):.2%}")
                print(f"   • ROI: {best_strategy.get('roi', 0):.2%}")
                print(f"   • Trades: {best_strategy.get('total_trades', 0)}")
            
            # Performance geral
            overall = summary.get('overall_performance', {})
            if overall:
                print(f"\n📊 PERFORMANCE GERAL:")
                print(f"   • Estratégias testadas: {overall.get('total_strategies_tested', 0)}")
                print(f"   • Win Rate médio: {overall.get('average_win_rate', 0):.2%}")
                print(f"   • ROI médio: {overall.get('average_roi', 0):.2%}")
                print(f"   • Estratégias lucrativas: {overall.get('profitable_strategies', 0)}")
            
            print(f"\n✅ Resultados salvos em: backtest_results/demo/")
            print("="*60)
            
        else:
            print("❌ Erro ao executar demonstração")
            
    except Exception as e:
        logger.error(f"Erro na demonstração: {e}")
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    run_demo_backtests()