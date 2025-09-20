#!/usr/bin/env python3
"""
Indicadores Técnicos Avançados para Trading
Implementa indicadores mais sofisticados como Ichimoku, Williams %R, CCI, etc.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import talib

logger = logging.getLogger(__name__)

class AdvancedIndicators:
    """Classe para calcular indicadores técnicos avançados"""
    
    def __init__(self):
        """Inicializa a classe de indicadores avançados"""
        self.indicators_cache = {}
        logger.info("AdvancedIndicators inicializado")
    
    def ichimoku_cloud(self, data: pd.DataFrame, 
                      tenkan_period: int = 9,
                      kijun_period: int = 26,
                      senkou_span_b_period: int = 52) -> Dict[str, pd.Series]:
        """
        Calcula o Ichimoku Cloud
        
        Args:
            data: DataFrame com dados OHLC
            tenkan_period: Período para Tenkan-sen
            kijun_period: Período para Kijun-sen
            senkou_span_b_period: Período para Senkou Span B
            
        Returns:
            Dict com componentes do Ichimoku
        """
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Tenkan-sen (Conversion Line)
            tenkan_high = high.rolling(window=tenkan_period).max()
            tenkan_low = low.rolling(window=tenkan_period).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = high.rolling(window=kijun_period).max()
            kijun_low = low.rolling(window=kijun_period).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
            
            # Senkou Span B (Leading Span B)
            senkou_high = high.rolling(window=senkou_span_b_period).max()
            senkou_low = low.rolling(window=senkou_span_b_period).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun_period)
            
            # Chikou Span (Lagging Span)
            chikou_span = close.shift(-kijun_period)
            
            return {
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_span_a': senkou_span_a,
                'senkou_span_b': senkou_span_b,
                'chikou_span': chikou_span,
                'cloud_top': np.maximum(senkou_span_a, senkou_span_b),
                'cloud_bottom': np.minimum(senkou_span_a, senkou_span_b)
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular Ichimoku Cloud: {e}")
            return {}
    
    def williams_percent_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula Williams %R
        
        Args:
            data: DataFrame com dados OHLC
            period: Período para cálculo
            
        Returns:
            Series com valores Williams %R
        """
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            
            return williams_r
            
        except Exception as e:
            logger.error(f"Erro ao calcular Williams %R: {e}")
            return pd.Series()
    
    def commodity_channel_index(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calcula Commodity Channel Index (CCI)
        
        Args:
            data: DataFrame com dados OHLC
            period: Período para cálculo
            
        Returns:
            Series com valores CCI
        """
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Typical Price
            tp = (high + low + close) / 3
            
            # Simple Moving Average do Typical Price
            sma_tp = tp.rolling(window=period).mean()
            
            # Mean Deviation
            mad = tp.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            # CCI
            cci = (tp - sma_tp) / (0.015 * mad)
            
            return cci
            
        except Exception as e:
            logger.error(f"Erro ao calcular CCI: {e}")
            return pd.Series()
    
    def stochastic_rsi(self, data: pd.DataFrame, 
                      rsi_period: int = 14,
                      stoch_period: int = 14,
                      k_period: int = 3,
                      d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calcula Stochastic RSI
        
        Args:
            data: DataFrame com dados OHLC
            rsi_period: Período para RSI
            stoch_period: Período para Stochastic
            k_period: Período para %K
            d_period: Período para %D
            
        Returns:
            Dict com %K e %D do Stochastic RSI
        """
        try:
            close = data['close']
            
            # Calcular RSI
            rsi = talib.RSI(close.values, timeperiod=rsi_period)
            rsi_series = pd.Series(rsi, index=close.index)
            
            # Calcular Stochastic do RSI
            rsi_min = rsi_series.rolling(window=stoch_period).min()
            rsi_max = rsi_series.rolling(window=stoch_period).max()
            
            stoch_rsi = (rsi_series - rsi_min) / (rsi_max - rsi_min) * 100
            
            # %K e %D
            k_percent = stoch_rsi.rolling(window=k_period).mean()
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'stoch_rsi': stoch_rsi,
                'k_percent': k_percent,
                'd_percent': d_percent
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular Stochastic RSI: {e}")
            return {}
    
    def awesome_oscillator(self, data: pd.DataFrame,
                          fast_period: int = 5,
                          slow_period: int = 34) -> pd.Series:
        """
        Calcula Awesome Oscillator
        
        Args:
            data: DataFrame com dados OHLC
            fast_period: Período rápido
            slow_period: Período lento
            
        Returns:
            Series com valores AO
        """
        try:
            high = data['high']
            low = data['low']
            
            # Median Price
            median_price = (high + low) / 2
            
            # SMAs
            sma_fast = median_price.rolling(window=fast_period).mean()
            sma_slow = median_price.rolling(window=slow_period).mean()
            
            # Awesome Oscillator
            ao = sma_fast - sma_slow
            
            return ao
            
        except Exception as e:
            logger.error(f"Erro ao calcular Awesome Oscillator: {e}")
            return pd.Series()
    
    def money_flow_index(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula Money Flow Index (MFI)
        
        Args:
            data: DataFrame com dados OHLCV
            period: Período para cálculo
            
        Returns:
            Series com valores MFI
        """
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data.get('volume', pd.Series(1, index=data.index))
            
            # Typical Price
            tp = (high + low + close) / 3
            
            # Raw Money Flow
            rmf = tp * volume
            
            # Positive and Negative Money Flow
            positive_mf = rmf.where(tp > tp.shift(1), 0)
            negative_mf = rmf.where(tp < tp.shift(1), 0)
            
            # Money Flow Ratio
            positive_mf_sum = positive_mf.rolling(window=period).sum()
            negative_mf_sum = negative_mf.rolling(window=period).sum()
            
            mfr = positive_mf_sum / negative_mf_sum
            
            # Money Flow Index
            mfi = 100 - (100 / (1 + mfr))
            
            return mfi
            
        except Exception as e:
            logger.error(f"Erro ao calcular MFI: {e}")
            return pd.Series()
    
    def vortex_indicator(self, data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calcula Vortex Indicator
        
        Args:
            data: DataFrame com dados OHLC
            period: Período para cálculo
            
        Returns:
            Dict com VI+ e VI-
        """
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Vortex Movements
            vm_plus = np.abs(high - low.shift(1))
            vm_minus = np.abs(low - high.shift(1))
            
            # Somas dos períodos
            tr_sum = tr.rolling(window=period).sum()
            vm_plus_sum = vm_plus.rolling(window=period).sum()
            vm_minus_sum = vm_minus.rolling(window=period).sum()
            
            # Vortex Indicators
            vi_plus = vm_plus_sum / tr_sum
            vi_minus = vm_minus_sum / tr_sum
            
            return {
                'vi_plus': vi_plus,
                'vi_minus': vi_minus
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular Vortex Indicator: {e}")
            return {}
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Calcula todos os indicadores avançados
        
        Args:
            data: DataFrame com dados OHLC
            
        Returns:
            Dict com todos os indicadores
        """
        try:
            indicators = {}
            
            # Ichimoku Cloud
            ichimoku = self.ichimoku_cloud(data)
            indicators.update({f'ichimoku_{k}': v for k, v in ichimoku.items()})
            
            # Williams %R
            indicators['williams_r'] = self.williams_percent_r(data)
            
            # CCI
            indicators['cci'] = self.commodity_channel_index(data)
            
            # Stochastic RSI
            stoch_rsi = self.stochastic_rsi(data)
            indicators.update({f'stoch_rsi_{k}': v for k, v in stoch_rsi.items()})
            
            # Awesome Oscillator
            indicators['awesome_oscillator'] = self.awesome_oscillator(data)
            
            # Money Flow Index
            indicators['mfi'] = self.money_flow_index(data)
            
            # Vortex Indicator
            vortex = self.vortex_indicator(data)
            indicators.update({f'vortex_{k}': v for k, v in vortex.items()})
            
            logger.info(f"Calculados {len(indicators)} indicadores avançados")
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores avançados: {e}")
            return {}
    
    def get_trading_signals(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Gera sinais de trading baseados nos indicadores avançados
        
        Args:
            data: DataFrame com dados OHLC
            
        Returns:
            Dict com sinais (-1, 0, 1) para cada indicador
        """
        try:
            signals = {}
            
            # Ichimoku signals
            ichimoku = self.ichimoku_cloud(data)
            if ichimoku:
                close = data['close']
                tenkan = ichimoku['tenkan_sen']
                kijun = ichimoku['kijun_sen']
                cloud_top = ichimoku['cloud_top']
                cloud_bottom = ichimoku['cloud_bottom']
                
                # Sinal baseado na posição do preço em relação à nuvem
                above_cloud = close > cloud_top
                below_cloud = close < cloud_bottom
                tenkan_above_kijun = tenkan > kijun
                
                ichimoku_signal = np.where(
                    above_cloud & tenkan_above_kijun, 1,
                    np.where(below_cloud & ~tenkan_above_kijun, -1, 0)
                )
                signals['ichimoku'] = ichimoku_signal[-1] if len(ichimoku_signal) > 0 else 0
            
            # Williams %R signals
            williams_r = self.williams_percent_r(data)
            if not williams_r.empty:
                latest_wr = williams_r.iloc[-1]
                if latest_wr > -20:  # Sobrecomprado
                    signals['williams_r'] = -1
                elif latest_wr < -80:  # Sobrevendido
                    signals['williams_r'] = 1
                else:
                    signals['williams_r'] = 0
            
            # CCI signals
            cci = self.commodity_channel_index(data)
            if not cci.empty:
                latest_cci = cci.iloc[-1]
                if latest_cci > 100:  # Sobrecomprado
                    signals['cci'] = -1
                elif latest_cci < -100:  # Sobrevendido
                    signals['cci'] = 1
                else:
                    signals['cci'] = 0
            
            # Stochastic RSI signals
            stoch_rsi = self.stochastic_rsi(data)
            if stoch_rsi and 'k_percent' in stoch_rsi:
                k_percent = stoch_rsi['k_percent']
                d_percent = stoch_rsi['d_percent']
                
                if not k_percent.empty and not d_percent.empty:
                    latest_k = k_percent.iloc[-1]
                    latest_d = d_percent.iloc[-1]
                    
                    if latest_k > latest_d and latest_k < 20:  # Cruzamento para cima em área sobrevendida
                        signals['stoch_rsi'] = 1
                    elif latest_k < latest_d and latest_k > 80:  # Cruzamento para baixo em área sobrecomprada
                        signals['stoch_rsi'] = -1
                    else:
                        signals['stoch_rsi'] = 0
            
            logger.info(f"Gerados sinais para {len(signals)} indicadores avançados")
            return signals
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinais de trading: {e}")
            return {}


def create_advanced_indicators() -> AdvancedIndicators:
    """
    Função utilitária para criar uma instância de AdvancedIndicators
    
    Returns:
        Instância configurada de AdvancedIndicators
    """
    return AdvancedIndicators()


# Instância global para uso direto
advanced_indicators = create_advanced_indicators()