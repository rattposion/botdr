"""
Sistema de Análise Multi-Timeframe para Trading
Combina sinais de diferentes timeframes para decisões mais robustas
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from config import config
from data_collector import DerivDataCollector
from feature_engineering import FeatureEngineer
from ml_model import TradingMLModel
from ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)

class MultiTimeframeStrategy:
    """Estratégia de trading multi-timeframe"""
    
    def __init__(self, symbol: str = "R_50"):
        self.symbol = symbol
        self.timeframes = {
            '1m': 60,      # 1 minuto
            '5m': 300,     # 5 minutos  
            '15m': 900,    # 15 minutos
            '1h': 3600,    # 1 hora
            '4h': 14400    # 4 horas
        }
        self.models = {}
        self.data_collector = None
        self.feature_engineers = {}
        self.current_data = {}
        self.signals = {}
        self.weights = {
            '1m': 0.1,
            '5m': 0.2,
            '15m': 0.3,
            '1h': 0.3,
            '4h': 0.1
        }
        
    async def initialize(self):
        """Inicializa o sistema multi-timeframe"""
        logger.info("Inicializando sistema multi-timeframe...")
        
        # Inicializar coletor de dados
        self.data_collector = DerivDataCollector()
        self.data_collector.connect()
        
        # Criar feature engineers para cada timeframe
        for tf in self.timeframes.keys():
            self.feature_engineers[tf] = FeatureEngineer()
        
        logger.info(f"Sistema inicializado para {len(self.timeframes)} timeframes")
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Reamostra dados para o timeframe especificado
        
        Args:
            df: DataFrame com dados tick/1min
            timeframe: Timeframe alvo ('1m', '5m', '15m', '1h', '4h')
        """
        if timeframe == '1m':
            return df  # Já está em 1 minuto
        
        # Definir regras de agregação
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Converter timeframe para pandas frequency
        freq_map = {
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '4h': '4H'
        }
        
        freq = freq_map.get(timeframe, '1T')
        
        # Garantir que o índice seja datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Reamostrar
        resampled = df.resample(freq).agg(agg_rules)
        resampled.dropna(inplace=True)
        
        logger.debug(f"Dados reamostrados para {timeframe}: {len(resampled)} candles")
        return resampled
    
    async def collect_timeframe_data(self, timeframe: str, 
                                   count: int = 1000) -> pd.DataFrame:
        """Coleta dados para um timeframe específico"""
        try:
            # Coletar dados base (1 minuto)
            base_data = self.data_collector.get_historical_data(
                symbol=self.symbol,
                timeframe="1m",  # 1 minuto
                count=count * self.timeframes[timeframe] // 60  # Ajustar quantidade
            )
            
            if base_data.empty:
                logger.warning(f"Nenhum dado coletado para {timeframe}")
                return pd.DataFrame()
            
            # Reamostrar para o timeframe desejado
            tf_data = self.resample_data(base_data, timeframe)
            self.current_data[timeframe] = tf_data
            
            logger.debug(f"Coletados {len(tf_data)} candles para {timeframe}")
            return tf_data
            
        except Exception as e:
            logger.error(f"Erro ao coletar dados para {timeframe}: {e}")
            return pd.DataFrame()
    
    async def collect_all_timeframes(self, count: int = 1000):
        """Coleta dados para todos os timeframes"""
        logger.info("Coletando dados para todos os timeframes...")
        
        tasks = []
        for tf in self.timeframes.keys():
            task = self.collect_timeframe_data(tf, count)
            tasks.append(task)
        
        # Executar coletas em paralelo
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verificar resultados
        for tf, result in zip(self.timeframes.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Erro na coleta {tf}: {result}")
            elif not result.empty:
                logger.info(f"Timeframe {tf}: {len(result)} candles coletados")
    
    def train_timeframe_model(self, timeframe: str, 
                            model_type: str = "ensemble") -> Union[TradingMLModel, EnsembleModel]:
        """Treina modelo para um timeframe específico"""
        logger.info(f"Treinando modelo para timeframe {timeframe}...")
        
        if timeframe not in self.current_data or self.current_data[timeframe].empty:
            logger.warning(f"Sem dados para treinar modelo {timeframe}")
            return None
        
        df = self.current_data[timeframe]
        
        try:
            if model_type == "ensemble":
                # Usar ensemble de modelos
                from ensemble_model import create_ensemble_model
                model = create_ensemble_model(df, ensemble_type="voting")
            else:
                # Usar modelo simples
                model = TradingMLModel(model_type)
                model.train(df)
            
            self.models[timeframe] = model
            logger.info(f"Modelo {timeframe} treinado com sucesso")
            return model
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo {timeframe}: {e}")
            return None
    
    def train_all_models(self, model_type: str = "ensemble"):
        """Treina modelos para todos os timeframes"""
        logger.info("Treinando modelos para todos os timeframes...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for tf in self.timeframes.keys():
                if tf in self.current_data and not self.current_data[tf].empty:
                    future = executor.submit(self.train_timeframe_model, tf, model_type)
                    futures.append((tf, future))
            
            # Aguardar conclusão
            for tf, future in futures:
                try:
                    model = future.result(timeout=300)  # 5 minutos timeout
                    if model:
                        logger.info(f"Modelo {tf} treinado")
                except Exception as e:
                    logger.error(f"Erro no treinamento {tf}: {e}")
    
    def generate_timeframe_signal(self, timeframe: str) -> Dict[str, Any]:
        """Gera sinal para um timeframe específico"""
        if timeframe not in self.models or not self.models[timeframe]:
            return {'signal': 'HOLD', 'confidence': 0.0, 'error': 'Modelo não disponível'}
        
        if timeframe not in self.current_data or self.current_data[timeframe].empty:
            return {'signal': 'HOLD', 'confidence': 0.0, 'error': 'Dados não disponíveis'}
        
        try:
            model = self.models[timeframe]
            df = self.current_data[timeframe].tail(100)  # Últimos 100 candles
            
            # Gerar predição
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df)
                if len(probabilities) > 0:
                    last_proba = probabilities[-1]
                    confidence = max(last_proba)
                    signal = 'CALL' if last_proba[1] > last_proba[0] else 'PUT'
                else:
                    signal, confidence = 'HOLD', 0.0
            else:
                predictions = model.predict(df)
                if len(predictions) > 0:
                    last_pred = predictions[-1]
                    signal = 'CALL' if last_pred == 1 else 'PUT'
                    confidence = 0.7  # Confiança padrão
                else:
                    signal, confidence = 'HOLD', 0.0
            
            return {
                'signal': signal,
                'confidence': confidence,
                'timeframe': timeframe,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinal {timeframe}: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'error': str(e)}
    
    def generate_all_signals(self) -> Dict[str, Dict[str, Any]]:
        """Gera sinais para todos os timeframes"""
        signals = {}
        
        for tf in self.timeframes.keys():
            signals[tf] = self.generate_timeframe_signal(tf)
        
        self.signals = signals
        return signals
    
    def combine_signals(self, signals: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combina sinais de múltiplos timeframes
        
        Args:
            signals: Sinais por timeframe (usa self.signals se None)
        
        Returns:
            Sinal combinado com confiança
        """
        if signals is None:
            signals = self.signals
        
        if not signals:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Nenhum sinal disponível'}
        
        # Calcular scores ponderados
        call_score = 0.0
        put_score = 0.0
        total_weight = 0.0
        
        valid_signals = 0
        signal_details = {}
        
        for tf, signal_data in signals.items():
            if 'error' in signal_data:
                continue
                
            signal = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('confidence', 0.0)
            weight = self.weights.get(tf, 0.1)
            
            if signal != 'HOLD' and confidence > 0:
                weighted_confidence = confidence * weight
                
                if signal == 'CALL':
                    call_score += weighted_confidence
                elif signal == 'PUT':
                    put_score += weighted_confidence
                
                total_weight += weight
                valid_signals += 1
                
                signal_details[tf] = {
                    'signal': signal,
                    'confidence': confidence,
                    'weight': weight,
                    'weighted_score': weighted_confidence
                }
        
        if valid_signals == 0:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Nenhum sinal válido'}
        
        # Determinar sinal final
        if call_score > put_score:
            final_signal = 'CALL'
            final_confidence = call_score / total_weight if total_weight > 0 else 0.0
        elif put_score > call_score:
            final_signal = 'PUT'
            final_confidence = put_score / total_weight if total_weight > 0 else 0.0
        else:
            final_signal = 'HOLD'
            final_confidence = 0.0
        
        # Aplicar filtros de confiança
        min_confidence = config.trading.min_prediction_confidence
        if final_confidence < min_confidence:
            final_signal = 'HOLD'
            reason = f'Confiança {final_confidence:.3f} < {min_confidence}'
        else:
            reason = f'Consenso de {valid_signals} timeframes'
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'reason': reason,
            'valid_signals': valid_signals,
            'total_timeframes': len(signals),
            'call_score': call_score,
            'put_score': put_score,
            'signal_details': signal_details,
            'timestamp': datetime.now()
        }
    
    def get_trend_alignment(self) -> Dict[str, Any]:
        """Analisa alinhamento de tendências entre timeframes"""
        if not self.signals:
            return {'alignment': 'UNKNOWN', 'strength': 0.0}
        
        call_count = 0
        put_count = 0
        hold_count = 0
        
        for signal_data in self.signals.values():
            signal = signal_data.get('signal', 'HOLD')
            if signal == 'CALL':
                call_count += 1
            elif signal == 'PUT':
                put_count += 1
            else:
                hold_count += 1
        
        total_signals = len(self.signals)
        
        if call_count > put_count and call_count > hold_count:
            alignment = 'BULLISH'
            strength = call_count / total_signals
        elif put_count > call_count and put_count > hold_count:
            alignment = 'BEARISH'
            strength = put_count / total_signals
        else:
            alignment = 'NEUTRAL'
            strength = max(call_count, put_count, hold_count) / total_signals
        
        return {
            'alignment': alignment,
            'strength': strength,
            'call_count': call_count,
            'put_count': put_count,
            'hold_count': hold_count,
            'total_signals': total_signals
        }
    
    async def update_data_and_signals(self):
        """Atualiza dados e gera novos sinais"""
        logger.info("Atualizando dados e sinais multi-timeframe...")
        
        # Atualizar dados
        await self.collect_all_timeframes(count=500)
        
        # Gerar novos sinais
        signals = self.generate_all_signals()
        
        # Combinar sinais
        combined = self.combine_signals(signals)
        
        # Analisar tendências
        trend = self.get_trend_alignment()
        
        logger.info(f"Sinal combinado: {combined['signal']} (confiança: {combined['confidence']:.3f})")
        logger.info(f"Alinhamento: {trend['alignment']} (força: {trend['strength']:.3f})")
        
        return {
            'individual_signals': signals,
            'combined_signal': combined,
            'trend_analysis': trend
        }
    
    def set_timeframe_weights(self, weights: Dict[str, float]):
        """Define pesos customizados para timeframes"""
        # Normalizar pesos
        total = sum(weights.values())
        if total > 0:
            self.weights = {tf: weight/total for tf, weight in weights.items()}
            logger.info(f"Pesos atualizados: {self.weights}")
    
    async def cleanup(self):
        """Limpa recursos"""
        if self.data_collector:
            self.data_collector.disconnect()
        logger.info("Sistema multi-timeframe finalizado")

async def create_multi_timeframe_strategy(symbol: str = "R_50") -> MultiTimeframeStrategy:
    """
    Função utilitária para criar estratégia multi-timeframe
    
    Args:
        symbol: Símbolo para trading
    
    Returns:
        Estratégia multi-timeframe inicializada
    """
    strategy = MultiTimeframeStrategy(symbol)
    await strategy.initialize()
    return strategy