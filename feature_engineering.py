"""
Pipeline de Feature Engineering para Trading
Gera features técnicas e estatísticas para modelos de ML
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from config import config

logger = logging.getLogger(__name__)

# Implementações de indicadores técnicos usando pandas/numpy
def sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period).mean()

def rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    """MACD Indicator"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series, period=20, std_dev=2):
    """Bollinger Bands"""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def williams_r(high, low, close, period=14):
    """Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))

def atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def roc(series, period=10):
    """Rate of Change"""
    return ((series - series.shift(period)) / series.shift(period)) * 100

def momentum(series, period=10):
    """Momentum"""
    return series - series.shift(period)

def cci(high, low, close, period=20):
    """Commodity Channel Index"""
    tp = (high + low + close) / 3  # Typical Price
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - sma_tp) / (0.015 * mad)

def ichimoku_cloud(high, low, close, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    senkou_span_b = ((high.rolling(window=senkou_b_period).max() + 
                     low.rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-kijun_period)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def parabolic_sar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Parabolic SAR"""
    length = len(close)
    psar = np.zeros(length)
    af = af_start
    ep = 0
    trend = 1  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    psar[0] = low.iloc[0]
    
    for i in range(1, length):
        if trend == 1:  # Uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_increment, af_max)
            
            if low.iloc[i] <= psar[i]:
                trend = -1
                psar[i] = ep
                af = af_start
                ep = low.iloc[i]
        else:  # Downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_increment, af_max)
            
            if high.iloc[i] >= psar[i]:
                trend = 1
                psar[i] = ep
                af = af_start
                ep = high.iloc[i]
    
    return pd.Series(psar, index=close.index)

def adx(high, low, close, period=14):
    """Average Directional Index"""
    tr = atr(high, low, close, 1)  # True Range for 1 period
    
    # Directional Movement
    dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
    dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
    
    dm_plus = pd.Series(dm_plus, index=high.index)
    dm_minus = pd.Series(dm_minus, index=low.index)
    
    # Smoothed values
    tr_smooth = tr.rolling(window=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    
    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_value = dx.rolling(window=period).mean()
    
    return adx_value, di_plus, di_minus

def obv(close, volume):
    """On-Balance Volume"""
    obv_values = np.zeros(len(close))
    obv_values[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv_values[i] = obv_values[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv_values[i] = obv_values[i-1] - volume.iloc[i]
        else:
            obv_values[i] = obv_values[i-1]
    
    return pd.Series(obv_values, index=close.index)

def vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

class FeatureEngineer:
    """Classe para engenharia de features"""
    
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.feature_names = []
        self.target_column = "target"
        
    def create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Cria todas as features a partir dos dados OHLCV
        
        Args:
            df: DataFrame com colunas [open, high, low, close, volume]
            symbol: Símbolo do ativo (opcional)
            
        Returns:
            DataFrame com features
        """
        if df.empty:
            return df
        
        logger.info(f"Criando features para {len(df)} registros")
        
        # Cópia dos dados originais
        features_df = df.copy()
        
        # 1. Features básicas de preço
        features_df = self._add_price_features(features_df)
        
        # 2. Indicadores técnicos
        features_df = self._add_technical_indicators(features_df)
        
        # 3. Features estatísticas
        features_df = self._add_statistical_features(features_df)
        
        # 4. Features de volatilidade
        features_df = self._add_volatility_features(features_df)
        
        # 5. Features de momentum
        features_df = self._add_momentum_features(features_df)
        
        # 6. Features de volume
        features_df = self._add_volume_features(features_df)
        
        # 7. Features de padrões de candlestick
        features_df = self._add_candlestick_patterns(features_df)
        
        # 8. Features de lag
        features_df = self._add_lag_features(features_df)
        
        # 9. Features de tempo
        features_df = self._add_time_features(features_df)
        
        # 10. Target (próxima direção)
        features_df = self._add_target(features_df)
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        # Armazena nomes das features
        self.feature_names = [col for col in features_df.columns if col != self.target_column]
        
        logger.info(f"Features criadas: {len(self.feature_names)} features, {len(features_df)} registros válidos")
        
        return features_df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features básicas de preço"""
        # Retornos
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Preço relativo
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        # Range
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Gaps
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores técnicos"""
        try:
            # Médias móveis
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = sma(df['close'], period)
                df[f'ema_{period}'] = ema(df['close'], period)
                
                # Posição relativa à média
                df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
                df[f'close_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
            
            # RSI
            for period in [14, 21]:
                df[f'rsi_{period}'] = rsi(df['close'], period)
            
            # MACD
            macd_line, macd_signal, macd_hist = macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            stoch_k, stoch_d = stochastic(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Williams %R
            df['williams_r'] = williams_r(df['high'], df['low'], df['close'])
            
            # ATR
            df['atr'] = atr(df['high'], df['low'], df['close'])
            
            # CCI (Commodity Channel Index)
            df['cci'] = cci(df['high'], df['low'], df['close'])
            
            # Ichimoku Cloud
            tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku_cloud(df['high'], df['low'], df['close'])
            df['ichimoku_tenkan'] = tenkan
            df['ichimoku_kijun'] = kijun
            df['ichimoku_senkou_a'] = senkou_a
            df['ichimoku_senkou_b'] = senkou_b
            df['ichimoku_chikou'] = chikou
            
            # Ichimoku signals
            df['ichimoku_cloud_green'] = (senkou_a > senkou_b).astype(int)
            df['ichimoku_above_cloud'] = (df['close'] > senkou_a) & (df['close'] > senkou_b)
            df['ichimoku_below_cloud'] = (df['close'] < senkou_a) & (df['close'] < senkou_b)
            
            # Parabolic SAR
            df['psar'] = parabolic_sar(df['high'], df['low'], df['close'])
            df['psar_signal'] = (df['close'] > df['psar']).astype(int)  # 1 for bullish, 0 for bearish
            
            # ADX (Average Directional Index)
            adx_val, di_plus, di_minus = adx(df['high'], df['low'], df['close'])
            df['adx'] = adx_val
            df['di_plus'] = di_plus
            df['di_minus'] = di_minus
            df['adx_trend_strength'] = (df['adx'] > 25).astype(int)  # Strong trend when ADX > 25
            
            # Volume indicators
            if 'volume' in df.columns:
                df['obv'] = obv(df['close'], df['volume'])
                df['vwap'] = vwap(df['high'], df['low'], df['close'], df['volume'])
                df['close_vwap_ratio'] = df['close'] / df['vwap']
            
            # Rate of Change
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = roc(df['close'], period)
                df[f'momentum_{period}'] = momentum(df['close'], period)
            
        except Exception as e:
            logger.warning(f"Erro ao calcular indicadores técnicos: {e}")
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features estatísticas"""
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window).max()
            df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
            df[f'close_kurt_{window}'] = df['close'].rolling(window).kurt()
            
            # Z-score
            df[f'close_zscore_{window}'] = (df['close'] - df[f'close_mean_{window}']) / df[f'close_std_{window}']
            
            # Percentile position
            df[f'close_percentile_{window}'] = df['close'].rolling(window).rank(pct=True)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volatilidade"""
        # Volatilidade realizada
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(252)
            
        # Volatilidade intraday
        df['intraday_volatility'] = np.log(df['high'] / df['low'])
        
        # Garman-Klass volatility
        df['gk_volatility'] = 0.5 * np.log(df['high'] / df['low'])**2 - (2*np.log(2) - 1) * np.log(df['close'] / df['open'])**2
        
        # Parkinson volatility
        df['parkinson_volatility'] = np.log(df['high'] / df['low'])**2 / (4 * np.log(2))
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de momentum"""
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = roc(df['close'], period)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = momentum(df['close'], period)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volume"""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            # Se não há dados de volume, criar features dummy
            df['volume'] = 1
        
        # Volume médio
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # On Balance Volume (implementação simples)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        
        # Volume Price Trend
        df['vpt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
        
        # Accumulation/Distribution Line (implementação simples)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        df['ad'] = (clv * df['volume']).cumsum()
        
        return df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona padrões de candlestick básicos"""
        try:
            # Padrões básicos de candlestick (implementação simples)
            
            # Doji
            body_size = abs(df['close'] - df['open'])
            range_size = df['high'] - df['low']
            df['doji'] = (body_size / range_size < 0.1).astype(int)
            
            # Hammer
            lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
            upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
            df['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
            
            # Shooting Star
            df['shooting_star'] = ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)
            
            # Engulfing patterns
            bullish_engulfing = (
                (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle bearish
                (df['close'] > df['open']) &  # Current candle bullish
                (df['open'] < df['close'].shift(1)) &  # Current open below previous close
                (df['close'] > df['open'].shift(1))  # Current close above previous open
            )
            df['bullish_engulfing'] = bullish_engulfing.astype(int)
            
            bearish_engulfing = (
                (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle bullish
                (df['close'] < df['open']) &  # Current candle bearish
                (df['open'] > df['close'].shift(1)) &  # Current open above previous close
                (df['close'] < df['open'].shift(1))  # Current close below previous open
            )
            df['bearish_engulfing'] = bearish_engulfing.astype(int)
            
        except Exception as e:
            logger.warning(f"Erro ao calcular padrões de candlestick: {e}")
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de lag (valores passados)"""
        # Lags do preço de fechamento
        for lag in config.ml.lookback_periods:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['return_1'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Diferenças
        for lag in [1, 5, 10]:
            df[f'close_diff_{lag}'] = df['close'] - df['close'].shift(lag)
            df[f'close_pct_diff_{lag}'] = (df['close'] - df['close'].shift(lag)) / df['close'].shift(lag)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features temporais"""
        if df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Features cíclicas
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona variável target (direção do próximo movimento)"""
        # Target: 1 se próximo close > close atual, 0 caso contrário
        df[self.target_column] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Target alternativo: retorno futuro
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        
        # Target para múltiplos períodos
        for period in [1, 3, 5]:
            df[f'target_{period}'] = (df['close'].shift(-period) > df['close']).astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara features para treinamento/predição
        
        Args:
            df: DataFrame com features
            fit_scaler: Se deve ajustar o scaler
            
        Returns:
            X, y arrays
        """
        # Separar features e target
        feature_cols = [col for col in df.columns if col not in [self.target_column, 'future_return'] 
                       and not col.startswith('target_')]
        
        X = df[feature_cols].values
        y = df[self.target_column].values
        
        # Remover infinitos e NaN
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        # Normalização
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)
        
        logger.info(f"Features preparadas: {X.shape[0]} amostras, {X.shape[1]} features")
        
        return X, y
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """Retorna importância das features"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        
        return pd.DataFrame()
    
    def reduce_features(self, X: np.ndarray, n_components: int = 50, fit_pca: bool = True) -> np.ndarray:
        """Reduz dimensionalidade usando PCA"""
        if fit_pca:
            self.pca = PCA(n_components=n_components)
            X_reduced = self.pca.fit_transform(X)
        elif self.pca is not None:
            X_reduced = self.pca.transform(X)
        else:
            return X
        
        logger.info(f"Dimensionalidade reduzida de {X.shape[1]} para {X_reduced.shape[1]}")
        return X_reduced

# Convenience functions
def create_features_from_data(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Função de conveniência para criar features"""
    engineer = FeatureEngineer()
    return engineer.create_features(df, symbol)

def prepare_ml_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, FeatureEngineer]:
    """Prepara dados para ML"""
    engineer = FeatureEngineer()
    features_df = engineer.create_features(df)
    X, y = engineer.prepare_features(features_df)
    return X, y, engineer