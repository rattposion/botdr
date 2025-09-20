#!/usr/bin/env python3
"""
Analisador Multi-Timeframe para Trading
Sistema para análise de múltiplos timeframes e correlações
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from advanced_indicators import AdvancedIndicators
from utils import setup_logging

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """Analisador de múltiplos timeframes"""
    
    def __init__(self):
        """Inicializa o analisador"""
        self.timeframes = ['1min', '5min', '15min', '30min', '1H']
        self.symbols = ["R_50", "R_100", "R_25", "R_75"]
        self.analysis_results = {}
        
        # Configurar diretórios
        self.results_dir = "multi_timeframe_results"
        self.plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logger.info("MultiTimeframeAnalyzer inicializado")
    
    def generate_multi_timeframe_data(self, symbol: str, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Gera dados para múltiplos timeframes
        
        Args:
            symbol: Símbolo para análise
            days: Número de dias
            
        Returns:
            Dicionário com dados por timeframe
        """
        try:
            # Parâmetros baseados no símbolo
            base_price = float(symbol.split('_')[1])
            
            # Configurações específicas por símbolo
            symbol_configs = {
                "R_50": {"volatility": 0.02, "trend_strength": 0.001, "noise_level": 0.5},
                "R_100": {"volatility": 0.015, "trend_strength": 0.0008, "noise_level": 0.3},
                "R_25": {"volatility": 0.025, "trend_strength": 0.0012, "noise_level": 0.7},
                "R_75": {"volatility": 0.018, "trend_strength": 0.0009, "noise_level": 0.4}
            }
            
            config = symbol_configs.get(symbol, symbol_configs["R_50"])
            
            # Gerar dados base (1 minuto)
            periods_1min = days * 24 * 60
            dates_1min = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                periods=periods_1min,
                freq='1min'
            )
            
            # Seed baseado no símbolo
            np.random.seed(hash(symbol) % 2**32)
            
            # Gerar componentes de preço
            long_trend = np.linspace(0, config["trend_strength"] * periods_1min, periods_1min)
            medium_cycles = np.sin(np.arange(periods_1min) * 2 * np.pi / (24 * 60)) * config["volatility"] * 0.5
            short_noise = np.random.normal(0, config["volatility"] * config["noise_level"], periods_1min)
            
            # Eventos especiais
            events = np.zeros(periods_1min)
            event_indices = np.random.random(periods_1min) < 0.001
            events[event_indices] = np.random.normal(0, config["volatility"] * 3, np.sum(event_indices))
            
            # Combinar componentes
            returns = long_trend + medium_cycles + short_noise + events
            
            # Calcular preços
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))
            
            # Criar dados OHLC base (1 minuto)
            data_1min = []
            for i, price in enumerate(prices):
                volatility_factor = config["volatility"] / 100
                high_factor = 1 + abs(np.random.normal(0, volatility_factor))
                low_factor = 1 - abs(np.random.normal(0, volatility_factor))
                
                high = price * high_factor
                low = price * low_factor
                volume = int(2000 + abs(returns[i]) * 100000 + np.random.normal(0, 500))
                
                data_1min.append({
                    'timestamp': dates_1min[i],
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': max(volume, 100)
                })
            
            df_1min = pd.DataFrame(data_1min)
            df_1min.set_index('timestamp', inplace=True)
            
            # Gerar outros timeframes
            timeframe_data = {'1min': df_1min}
            
            # Mapeamento de frequências
            freq_map = {
                '5min': '5T',
                '15min': '15T',
                '30min': '30T',
                '1H': '1H'
            }
            
            for tf, freq in freq_map.items():
                # Resample para timeframe maior
                resampled = df_1min.resample(freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                timeframe_data[tf] = resampled
            
            logger.info(f"Dados multi-timeframe gerados para {symbol}: {[f'{tf}({len(data)})' for tf, data in timeframe_data.items()]}")
            return timeframe_data
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados multi-timeframe para {symbol}: {e}")
            return {}
    
    def calculate_timeframe_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calcula indicadores para um timeframe específico
        
        Args:
            data: Dados OHLCV
            timeframe: Timeframe (1min, 5min, etc.)
            
        Returns:
            DataFrame com indicadores
        """
        try:
            df = data.copy()
            
            # Indicadores básicos
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Médias móveis adaptadas ao timeframe
            if timeframe in ['1min', '5min']:
                sma_periods = [10, 20, 50]
                rsi_period = 14
            elif timeframe in ['15min', '30min']:
                sma_periods = [20, 50, 100]
                rsi_period = 21
            else:  # 1H
                sma_periods = [50, 100, 200]
                rsi_period = 28
            
            # Médias móveis
            for period in sma_periods:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = sma_periods[1]  # Período médio
            bb_middle = df['close'].rolling(bb_period).mean()
            bb_std = df['close'].rolling(bb_period).std()
            df['bb_upper'] = bb_middle + (bb_std * 2)
            df['bb_lower'] = bb_middle - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD
            ema_fast = 12 if timeframe in ['1min', '5min'] else 26
            ema_slow = 26 if timeframe in ['1min', '5min'] else 52
            
            ema_12 = df['close'].ewm(span=ema_fast).mean()
            ema_26 = df['close'].ewm(span=ema_slow).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Indicadores avançados
            indicators = AdvancedIndicators()
            
            try:
                df['williams_r'] = indicators.williams_percent_r(df)
                df['cci'] = indicators.commodity_channel_index(df)
                
                stoch_rsi = indicators.stochastic_rsi(df)
                if stoch_rsi:
                    df['stoch_rsi_k'] = stoch_rsi.get('k', 0)
                    df['stoch_rsi_d'] = stoch_rsi.get('d', 0)
                
            except Exception as e:
                logger.warning(f"Erro ao calcular indicadores avançados para {timeframe}: {e}")
            
            # Sinais de trading
            df['trend_signal'] = 0
            df.loc[df['close'] > df[f'sma_{sma_periods[0]}'], 'trend_signal'] = 1
            df.loc[df['close'] < df[f'sma_{sma_periods[0]}'], 'trend_signal'] = -1
            
            df['momentum_signal'] = 0
            df.loc[df['rsi'] > 70, 'momentum_signal'] = -1  # Sobrecomprado
            df.loc[df['rsi'] < 30, 'momentum_signal'] = 1   # Sobrevendido
            
            df['volatility_signal'] = 0
            vol_threshold = df['volatility'].rolling(50).mean()
            df.loc[df['volatility'] > vol_threshold * 1.5, 'volatility_signal'] = 1  # Alta volatilidade
            
            # Limpeza
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            logger.info(f"Indicadores calculados para {timeframe}: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores para {timeframe}: {e}")
            return data
    
    def analyze_timeframe_correlations(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analisa correlações entre timeframes
        
        Args:
            timeframe_data: Dados por timeframe
            
        Returns:
            Análise de correlações
        """
        try:
            correlations = {}
            
            # Preparar dados para correlação
            correlation_data = {}
            
            for tf, data in timeframe_data.items():
                if len(data) > 50:  # Mínimo de dados
                    # Resample para frequência comum (5min)
                    if tf != '5min':
                        resampled = data.resample('5T').agg({
                            'close': 'last',
                            'volume': 'sum',
                            'rsi': 'last',
                            'macd': 'last'
                        }).dropna()
                    else:
                        resampled = data[['close', 'volume', 'rsi', 'macd']].copy()
                    
                    correlation_data[tf] = resampled
            
            # Calcular correlações entre timeframes
            if len(correlation_data) >= 2:
                timeframes = list(correlation_data.keys())
                
                for i, tf1 in enumerate(timeframes):
                    for tf2 in timeframes[i+1:]:
                        # Alinhar dados por timestamp
                        data1 = correlation_data[tf1]
                        data2 = correlation_data[tf2]
                        
                        # Encontrar timestamps comuns
                        common_index = data1.index.intersection(data2.index)
                        
                        if len(common_index) > 20:
                            aligned_data1 = data1.loc[common_index]
                            aligned_data2 = data2.loc[common_index]
                            
                            # Calcular correlações
                            price_corr = aligned_data1['close'].corr(aligned_data2['close'])
                            volume_corr = aligned_data1['volume'].corr(aligned_data2['volume'])
                            rsi_corr = aligned_data1['rsi'].corr(aligned_data2['rsi'])
                            macd_corr = aligned_data1['macd'].corr(aligned_data2['macd'])
                            
                            correlations[f'{tf1}_vs_{tf2}'] = {
                                'price_correlation': price_corr,
                                'volume_correlation': volume_corr,
                                'rsi_correlation': rsi_corr,
                                'macd_correlation': macd_corr,
                                'data_points': len(common_index)
                            }
            
            # Análise de divergências
            divergences = self.detect_timeframe_divergences(timeframe_data)
            
            return {
                'correlations': correlations,
                'divergences': divergences,
                'timeframes_analyzed': list(timeframe_data.keys())
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de correlações: {e}")
            return {}
    
    def detect_timeframe_divergences(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Detecta divergências entre timeframes
        
        Args:
            timeframe_data: Dados por timeframe
            
        Returns:
            Divergências detectadas
        """
        try:
            divergences = {}
            
            # Comparar sinais entre timeframes
            signals = {}
            
            for tf, data in timeframe_data.items():
                if len(data) > 10:
                    latest_data = data.tail(10)
                    
                    # Sinais atuais
                    trend_signal = latest_data['trend_signal'].iloc[-1] if 'trend_signal' in latest_data.columns else 0
                    momentum_signal = latest_data['momentum_signal'].iloc[-1] if 'momentum_signal' in latest_data.columns else 0
                    
                    signals[tf] = {
                        'trend': trend_signal,
                        'momentum': momentum_signal,
                        'price_change': (latest_data['close'].iloc[-1] / latest_data['close'].iloc[0] - 1) * 100
                    }
            
            # Detectar divergências
            timeframes = list(signals.keys())
            
            for i, tf1 in enumerate(timeframes):
                for tf2 in timeframes[i+1:]:
                    signal1 = signals[tf1]
                    signal2 = signals[tf2]
                    
                    # Divergência de tendência
                    trend_divergence = signal1['trend'] != signal2['trend']
                    
                    # Divergência de momentum
                    momentum_divergence = signal1['momentum'] != signal2['momentum']
                    
                    # Divergência de preço
                    price_diff = abs(signal1['price_change'] - signal2['price_change'])
                    price_divergence = price_diff > 2.0  # Mais de 2% de diferença
                    
                    if trend_divergence or momentum_divergence or price_divergence:
                        divergences[f'{tf1}_vs_{tf2}'] = {
                            'trend_divergence': trend_divergence,
                            'momentum_divergence': momentum_divergence,
                            'price_divergence': price_divergence,
                            'price_difference': price_diff,
                            'signals': {tf1: signal1, tf2: signal2}
                        }
            
            return divergences
            
        except Exception as e:
            logger.error(f"Erro na detecção de divergências: {e}")
            return {}
    
    def create_consensus_signal(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Cria sinal de consenso entre timeframes
        
        Args:
            timeframe_data: Dados por timeframe
            
        Returns:
            Sinal de consenso
        """
        try:
            # Pesos por timeframe (timeframes maiores têm mais peso)
            timeframe_weights = {
                '1min': 0.1,
                '5min': 0.15,
                '15min': 0.2,
                '30min': 0.25,
                '1H': 0.3
            }
            
            signals = {}
            weighted_signals = {'trend': 0, 'momentum': 0, 'volatility': 0}
            total_weight = 0
            
            for tf, data in timeframe_data.items():
                if len(data) > 5 and tf in timeframe_weights:
                    latest = data.tail(1).iloc[0]
                    weight = timeframe_weights[tf]
                    
                    # Extrair sinais
                    trend_signal = latest.get('trend_signal', 0)
                    momentum_signal = latest.get('momentum_signal', 0)
                    volatility_signal = latest.get('volatility_signal', 0)
                    
                    signals[tf] = {
                        'trend': trend_signal,
                        'momentum': momentum_signal,
                        'volatility': volatility_signal,
                        'weight': weight
                    }
                    
                    # Calcular média ponderada
                    weighted_signals['trend'] += trend_signal * weight
                    weighted_signals['momentum'] += momentum_signal * weight
                    weighted_signals['volatility'] += volatility_signal * weight
                    total_weight += weight
            
            # Normalizar sinais
            if total_weight > 0:
                for signal_type in weighted_signals:
                    weighted_signals[signal_type] /= total_weight
            
            # Determinar sinal final
            consensus_signal = 0
            confidence = 0
            
            # Lógica de consenso
            trend_consensus = weighted_signals['trend']
            momentum_consensus = weighted_signals['momentum']
            
            if trend_consensus > 0.5 and momentum_consensus >= 0:
                consensus_signal = 1  # CALL
                confidence = min(trend_consensus, 0.8)
            elif trend_consensus < -0.5 and momentum_consensus <= 0:
                consensus_signal = -1  # PUT
                confidence = min(abs(trend_consensus), 0.8)
            else:
                consensus_signal = 0  # HOLD
                confidence = 0.5
            
            # Ajustar confiança baseada na volatilidade
            volatility_factor = abs(weighted_signals['volatility'])
            if volatility_factor > 0.5:
                confidence *= 0.8  # Reduzir confiança em alta volatilidade
            
            return {
                'consensus_signal': consensus_signal,
                'confidence': confidence,
                'individual_signals': signals,
                'weighted_signals': weighted_signals,
                'signal_interpretation': {
                    1: 'CALL',
                    -1: 'PUT',
                    0: 'HOLD'
                }.get(consensus_signal, 'UNKNOWN')
            }
            
        except Exception as e:
            logger.error(f"Erro ao criar sinal de consenso: {e}")
            return {}
    
    def run_multi_timeframe_analysis(self, symbol: str) -> Dict:
        """
        Executa análise completa multi-timeframe
        
        Args:
            symbol: Símbolo para análise
            
        Returns:
            Resultados da análise
        """
        try:
            logger.info(f"Iniciando análise multi-timeframe para {symbol}")
            
            # Gerar dados
            timeframe_data = self.generate_multi_timeframe_data(symbol)
            
            if not timeframe_data:
                return {'error': 'Falha na geração de dados'}
            
            # Calcular indicadores para cada timeframe
            processed_data = {}
            for tf, data in timeframe_data.items():
                processed_data[tf] = self.calculate_timeframe_indicators(data, tf)
            
            # Análise de correlações
            correlations = self.analyze_timeframe_correlations(processed_data)
            
            # Sinal de consenso
            consensus = self.create_consensus_signal(processed_data)
            
            # Estatísticas por timeframe
            timeframe_stats = {}
            for tf, data in processed_data.items():
                if len(data) > 0:
                    timeframe_stats[tf] = {
                        'data_points': len(data),
                        'avg_volatility': data['volatility'].mean() if 'volatility' in data.columns else 0,
                        'current_rsi': data['rsi'].iloc[-1] if 'rsi' in data.columns and len(data) > 0 else 50,
                        'trend_direction': data['trend_signal'].iloc[-1] if 'trend_signal' in data.columns and len(data) > 0 else 0,
                        'price_change_24h': ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100 if len(data) > 0 else 0
                    }
            
            results = {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'timeframe_stats': timeframe_stats,
                'correlations': correlations,
                'consensus_signal': consensus,
                'data_quality': {
                    'timeframes_processed': len(processed_data),
                    'total_data_points': sum(len(data) for data in processed_data.values()),
                    'analysis_period_days': 30
                }
            }
            
            logger.info(f"Análise multi-timeframe concluída para {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise multi-timeframe para {symbol}: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Executa análise abrangente para todos os símbolos
        
        Returns:
            Resultados completos
        """
        try:
            logger.info("Iniciando análise multi-timeframe abrangente")
            
            all_results = {
                'analysis_results': {},
                'summary': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for symbol in self.symbols:
                logger.info(f"Analisando {symbol}...")
                
                symbol_results = self.run_multi_timeframe_analysis(symbol)
                
                if 'error' not in symbol_results:
                    all_results['analysis_results'][symbol] = symbol_results
            
            # Calcular resumo
            all_results['summary'] = self.calculate_analysis_summary(all_results)
            
            # Salvar resultados
            results_file = os.path.join(
                self.results_dir,
                f"multi_timeframe_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"Análise multi-timeframe completa. Resultados salvos em: {results_file}")
            return all_results
            
        except Exception as e:
            logger.error(f"Erro na análise abrangente: {e}")
            return {}
    
    def calculate_analysis_summary(self, results: Dict) -> Dict:
        """
        Calcula resumo da análise
        
        Args:
            results: Resultados da análise
            
        Returns:
            Resumo calculado
        """
        try:
            analysis_results = results.get('analysis_results', {})
            
            if not analysis_results:
                return {}
            
            # Estatísticas gerais
            total_symbols = len(analysis_results)
            
            # Sinais de consenso
            consensus_signals = {}
            confidence_levels = []
            
            for symbol, symbol_data in analysis_results.items():
                consensus = symbol_data.get('consensus_signal', {})
                signal = consensus.get('consensus_signal', 0)
                confidence = consensus.get('confidence', 0)
                
                signal_name = consensus.get('signal_interpretation', 'UNKNOWN')
                consensus_signals[symbol] = {
                    'signal': signal,
                    'signal_name': signal_name,
                    'confidence': confidence
                }
                
                confidence_levels.append(confidence)
            
            # Correlações médias
            all_correlations = []
            for symbol_data in analysis_results.values():
                correlations = symbol_data.get('correlations', {}).get('correlations', {})
                for corr_data in correlations.values():
                    price_corr = corr_data.get('price_correlation', 0)
                    if not np.isnan(price_corr):
                        all_correlations.append(abs(price_corr))
            
            avg_correlation = np.mean(all_correlations) if all_correlations else 0
            
            # Recomendações
            recommendations = []
            
            # Contar sinais
            call_signals = sum(1 for s in consensus_signals.values() if s['signal'] > 0)
            put_signals = sum(1 for s in consensus_signals.values() if s['signal'] < 0)
            hold_signals = sum(1 for s in consensus_signals.values() if s['signal'] == 0)
            
            if call_signals > put_signals:
                recommendations.append(f"Tendência geral BULLISH: {call_signals} sinais de CALL vs {put_signals} de PUT")
            elif put_signals > call_signals:
                recommendations.append(f"Tendência geral BEARISH: {put_signals} sinais de PUT vs {call_signals} de CALL")
            else:
                recommendations.append(f"Mercado NEUTRO: sinais equilibrados")
            
            # Confiança média
            avg_confidence = np.mean(confidence_levels) if confidence_levels else 0
            if avg_confidence > 0.7:
                recommendations.append("Alta confiança nos sinais - condições favoráveis para trading")
            elif avg_confidence < 0.5:
                recommendations.append("Baixa confiança nos sinais - aguardar melhores condições")
            
            # Correlações
            if avg_correlation > 0.8:
                recommendations.append("Alta correlação entre timeframes - sinais consistentes")
            elif avg_correlation < 0.5:
                recommendations.append("Baixa correlação entre timeframes - mercado fragmentado")
            
            summary = {
                'total_symbols_analyzed': total_symbols,
                'consensus_signals': consensus_signals,
                'signal_distribution': {
                    'call_signals': call_signals,
                    'put_signals': put_signals,
                    'hold_signals': hold_signals
                },
                'confidence_stats': {
                    'average_confidence': avg_confidence,
                    'min_confidence': min(confidence_levels) if confidence_levels else 0,
                    'max_confidence': max(confidence_levels) if confidence_levels else 0
                },
                'correlation_stats': {
                    'average_correlation': avg_correlation,
                    'total_correlations_analyzed': len(all_correlations)
                },
                'recommendations': recommendations
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao calcular resumo: {e}")
            return {}


def run_multi_timeframe_analysis():
    """Função principal para executar análise multi-timeframe"""
    try:
        # Configurar logging
        setup_logging()
        
        print("\n" + "="*80)
        print("📊 ANÁLISE MULTI-TIMEFRAME AVANÇADA")
        print("="*80)
        print("⏰ Analisando múltiplos timeframes...")
        print("🔄 Detectando correlações e divergências...")
        print("🎯 Gerando sinais de consenso...")
        print("="*80)
        
        # Criar analisador
        analyzer = MultiTimeframeAnalyzer()
        
        # Executar análise
        results = analyzer.run_comprehensive_analysis()
        
        if results and 'summary' in results:
            summary = results['summary']
            
            print("\n" + "="*80)
            print("📈 RESULTADOS DA ANÁLISE MULTI-TIMEFRAME")
            print("="*80)
            
            print(f"📊 Símbolos analisados: {summary.get('total_symbols_analyzed', 0)}")
            
            # Distribuição de sinais
            signal_dist = summary.get('signal_distribution', {})
            if signal_dist:
                print(f"\n🎯 DISTRIBUIÇÃO DE SINAIS:")
                print(f"   • Sinais CALL: {signal_dist.get('call_signals', 0)}")
                print(f"   • Sinais PUT: {signal_dist.get('put_signals', 0)}")
                print(f"   • Sinais HOLD: {signal_dist.get('hold_signals', 0)}")
            
            # Sinais de consenso
            consensus_signals = summary.get('consensus_signals', {})
            if consensus_signals:
                print(f"\n📡 SINAIS DE CONSENSO:")
                for symbol, signal_data in consensus_signals.items():
                    signal_name = signal_data.get('signal_name', 'UNKNOWN')
                    confidence = signal_data.get('confidence', 0)
                    print(f"   • {symbol}: {signal_name} (Confiança: {confidence:.2%})")
            
            # Estatísticas de confiança
            confidence_stats = summary.get('confidence_stats', {})
            if confidence_stats:
                print(f"\n🎯 ESTATÍSTICAS DE CONFIANÇA:")
                print(f"   • Confiança média: {confidence_stats.get('average_confidence', 0):.2%}")
                print(f"   • Confiança mínima: {confidence_stats.get('min_confidence', 0):.2%}")
                print(f"   • Confiança máxima: {confidence_stats.get('max_confidence', 0):.2%}")
            
            # Correlações
            corr_stats = summary.get('correlation_stats', {})
            if corr_stats:
                print(f"\n🔄 ESTATÍSTICAS DE CORRELAÇÃO:")
                print(f"   • Correlação média: {corr_stats.get('average_correlation', 0):.2%}")
                print(f"   • Correlações analisadas: {corr_stats.get('total_correlations_analyzed', 0)}")
            
            # Recomendações
            recommendations = summary.get('recommendations', [])
            if recommendations:
                print(f"\n💡 RECOMENDAÇÕES:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print(f"\n✅ Resultados salvos em: multi_timeframe_results/")
            print("="*80)
            
        else:
            print("❌ Erro ao executar análise multi-timeframe")
            
    except Exception as e:
        logger.error(f"Erro na análise multi-timeframe: {e}")
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    run_multi_timeframe_analysis()