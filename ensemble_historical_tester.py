#!/usr/bin/env python3
"""
Testador de Estratégias Ensemble em Dados Históricos
Sistema para testar e validar estratégias ensemble com dados históricos simulados
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
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Imports dos módulos do sistema
from advanced_indicators import AdvancedIndicators
from utils import setup_logging

logger = logging.getLogger(__name__)

class EnsembleHistoricalTester:
    """Testador de estratégias ensemble com dados históricos"""
    
    def __init__(self):
        """Inicializa o testador"""
        self.symbols = ["R_50", "R_100", "R_25", "R_75"]
        self.test_results = {}
        
        # Configurar diretórios
        self.results_dir = "ensemble_test_results"
        self.plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Configurações de teste
        self.test_periods = [7, 14, 30]  # dias
        self.ensemble_configs = [
            {
                'name': 'conservative',
                'prob_threshold': 0.7,
                'stake_multiplier': 0.8,
                'max_consecutive_losses': 3
            },
            {
                'name': 'balanced',
                'prob_threshold': 0.6,
                'stake_multiplier': 1.0,
                'max_consecutive_losses': 5
            },
            {
                'name': 'aggressive',
                'prob_threshold': 0.55,
                'stake_multiplier': 1.2,
                'max_consecutive_losses': 7
            }
        ]
        
        logger.info("EnsembleHistoricalTester inicializado")
    
    def create_simple_ensemble(self):
        """
        Cria ensemble simplificado usando VotingClassifier
        
        Returns:
            Ensemble treinado
        """
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Modelos base
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('svm', SVC(probability=True, random_state=42, kernel='rbf'))
        ]
        
        # Criar ensemble voting
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft'  # Usar probabilidades
        )
        
        return ensemble
    
    def generate_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Gera dados históricos simulados com padrões realistas
        
        Args:
            symbol: Símbolo para simular
            days: Número de dias
            
        Returns:
            DataFrame com dados históricos
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
            
            # Gerar timestamps (5 minutos para mais dados)
            periods = days * 24 * 12  # 12 períodos de 5 min por hora
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                periods=periods,
                freq='5min'
            )
            
            # Seed baseado no símbolo e período
            np.random.seed(hash(f"{symbol}_{days}") % 2**32)
            
            # Gerar componentes de preço
            # 1. Tendência de longo prazo
            long_trend = np.linspace(0, config["trend_strength"] * periods, periods)
            
            # 2. Ciclos de médio prazo
            medium_cycles = np.sin(np.arange(periods) * 2 * np.pi / (24 * 60)) * config["volatility"] * 0.5
            
            # 3. Ruído de curto prazo
            short_noise = np.random.normal(0, config["volatility"] * config["noise_level"], periods)
            
            # 4. Eventos especiais (spikes ocasionais)
            events = np.zeros(periods)
            event_probability = 0.001  # 0.1% chance por minuto
            event_indices = np.random.random(periods) < event_probability
            events[event_indices] = np.random.normal(0, config["volatility"] * 3, np.sum(event_indices))
            
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
                # Simular variação intrabar
                volatility_factor = config["volatility"] / 100
                high_factor = 1 + abs(np.random.normal(0, volatility_factor))
                low_factor = 1 - abs(np.random.normal(0, volatility_factor))
                
                high = price * high_factor
                low = price * low_factor
                
                # Volume correlacionado com volatilidade
                base_volume = 2000
                volatility_volume = abs(returns[i]) * 100000
                volume = int(base_volume + volatility_volume + np.random.normal(0, 500))
                
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
            
            logger.info(f"Dados históricos gerados para {symbol}: {len(df)} registros ({days} dias)")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados históricos para {symbol}: {e}")
            return pd.DataFrame()
    
    def add_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features abrangentes para ensemble
        
        Args:
            data: Dados OHLC
            
        Returns:
            DataFrame com features completas
        """
        try:
            df = data.copy()
            
            # Features básicas
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Múltiplas médias móveis
            periods = [5, 10, 20, 50, 100]
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_vs_sma_{period}'] = (df['close'] / df[f'sma_{period}'] - 1) * 100
            
            # RSI múltiplos
            rsi_periods = [7, 14, 21, 28]
            for period in rsi_periods:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands múltiplas
            bb_periods = [15, 20, 25]
            for period in bb_periods:
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
            
            # Momentum indicators
            momentum_periods = [3, 5, 10, 20]
            for period in momentum_periods:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'roc_{period}'] = df['close'].pct_change(period)
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
            df['volume_volatility'] = df['volume'].rolling(20).std()
            
            # Volatility features
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['close_open_ratio'] = df['close'] / df['open'] - 1
            
            # Advanced technical indicators
            indicators = AdvancedIndicators()
            
            try:
                # Williams %R
                df['williams_r'] = indicators.williams_percent_r(df)
                
                # CCI
                df['cci'] = indicators.commodity_channel_index(df)
                
                # Stochastic RSI
                stoch_rsi = indicators.stochastic_rsi(df)
                if stoch_rsi:
                    df['stoch_rsi_k'] = stoch_rsi.get('k', 0)
                    df['stoch_rsi_d'] = stoch_rsi.get('d', 0)
                
                # Ichimoku components
                ichimoku = indicators.ichimoku_cloud(df)
                if ichimoku:
                    df['tenkan_sen'] = ichimoku.get('tenkan_sen', 0)
                    df['kijun_sen'] = ichimoku.get('kijun_sen', 0)
                    df['senkou_span_a'] = ichimoku.get('senkou_span_a', 0)
                    df['senkou_span_b'] = ichimoku.get('senkou_span_b', 0)
                
            except Exception as e:
                logger.warning(f"Erro ao calcular indicadores avançados: {e}")
            
            # Lag features
            lag_periods = [1, 2, 3, 5]
            for lag in lag_periods:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # Rolling statistics
            rolling_periods = [10, 20, 50]
            for period in rolling_periods:
                df[f'close_mean_{period}'] = df['close'].rolling(period).mean()
                df[f'close_std_{period}'] = df['close'].rolling(period).std()
                df[f'close_min_{period}'] = df['close'].rolling(period).min()
                df[f'close_max_{period}'] = df['close'].rolling(period).max()
                df[f'close_skew_{period}'] = df['close'].rolling(period).skew()
            
            # Time-based features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
            
            # Limpeza de dados
            # Substituir infinitos por NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Remove NaN values
            df = df.dropna()
            
            # Verificar se ainda há valores problemáticos
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any() or np.isinf(df[col]).any():
                    df[col] = df[col].fillna(df[col].median())
                    df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
            
            # Normalizar valores extremos (outliers)
            for col in numeric_cols:
                if col not in ['timestamp', 'hour', 'minute', 'day_of_week', 'is_weekend']:
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    df[col] = df[col].clip(lower=q01, upper=q99)
            
            logger.info(f"Features abrangentes criadas: {len(df.columns)} colunas, {len(df)} registros válidos")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao criar features: {e}")
            return data
    
    def test_ensemble_strategy(self, symbol: str, data: pd.DataFrame, config: Dict, period_days: int) -> Dict:
        """
        Testa estratégia ensemble com configuração específica
        
        Args:
            symbol: Símbolo sendo testado
            data: Dados preparados
            config: Configuração da estratégia
            period_days: Período de teste em dias
            
        Returns:
            Resultados do teste
        """
        try:
            logger.info(f"Testando ensemble {config['name']} para {symbol} ({period_days} dias)")
            
            if len(data) < 500:
                return {'error': 'Dados insuficientes'}
            
            # Preparar features
            feature_cols = [col for col in data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            X = data[feature_cols].copy()
            
            # Limpeza agressiva de dados
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            # Verificar e corrigir tipos de dados
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Normalização robusta para evitar valores extremos
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Target: próximo movimento
            y = (data['close'].shift(-1) > data['close']).astype(int)
            y = y.dropna()
            X = X.iloc[:-1]
            
            # Ajustar tamanhos
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            
            # Dividir dados (70% treino, 30% teste)
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Treinar ensemble simplificado
            ensemble = self.create_simple_ensemble()
            ensemble.fit(X_train.values, y_train.values)
            
            # Fazer predições
            predictions = ensemble.predict(X_test.values)
            probabilities = ensemble.predict_proba(X_test.values)[:, 1]  # Probabilidade da classe positiva
            
            # Métricas de classificação
            accuracy = np.mean(predictions == y_test)
            
            # Simular trading
            trading_results = self.simulate_ensemble_trading(
                data.iloc[split_idx:], predictions, probabilities, config
            )
            
            # Análise detalhada
            analysis = self.analyze_predictions(y_test, predictions, probabilities)
            
            results = {
                'symbol': symbol,
                'config_name': config['name'],
                'period_days': period_days,
                'accuracy': accuracy,
                'features_used': len(feature_cols),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'trading_results': trading_results,
                'prediction_analysis': analysis,
                'test_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Teste ensemble {config['name']} para {symbol}: {accuracy:.4f} accuracy, {trading_results.get('win_rate', 0):.2%} win rate")
            return results
            
        except Exception as e:
            logger.error(f"Erro no teste ensemble para {symbol}: {e}")
            return {'error': str(e)}
    
    def simulate_ensemble_trading(self, data: pd.DataFrame, predictions: np.ndarray, 
                                probabilities: np.ndarray, config: Dict) -> Dict:
        """
        Simula trading com estratégia ensemble
        
        Args:
            data: Dados de teste
            predictions: Predições do modelo
            probabilities: Probabilidades
            config: Configuração da estratégia
            
        Returns:
            Resultados da simulação
        """
        try:
            balance = 1000.0
            base_stake = 10.0 * config.get('stake_multiplier', 1.0)
            wins = 0
            losses = 0
            consecutive_losses = 0
            max_consecutive_losses = config.get('max_consecutive_losses', 5)
            prob_threshold = config.get('prob_threshold', 0.6)
            
            trades = []
            balance_history = [balance]
            
            for i in range(min(len(predictions), len(data) - 1)):
                prob = probabilities[i] if len(probabilities) > i else 0.5
                
                # Verificar condições de entrada
                if (prob > prob_threshold and 
                    consecutive_losses < max_consecutive_losses and
                    balance > base_stake):
                    
                    current_price = data.iloc[i]['close']
                    next_price = data.iloc[i + 1]['close']
                    
                    # Ajustar stake baseado na confiança e perdas consecutivas
                    confidence = (prob - 0.5) * 2
                    loss_penalty = max(0.5, 1 - (consecutive_losses * 0.1))
                    stake = base_stake * confidence * loss_penalty
                    stake = min(stake, balance * 0.1)  # Máximo 10% do saldo
                    
                    # Determinar direção
                    direction = "CALL" if predictions[i] == 1 else "PUT"
                    
                    # Calcular resultado
                    if direction == "CALL":
                        win = next_price > current_price
                    else:
                        win = next_price < current_price
                    
                    # Calcular PnL
                    if win:
                        payout_rate = 0.8 + (confidence * 0.1)  # 80% a 90%
                        pnl = stake * payout_rate
                        wins += 1
                        consecutive_losses = 0
                    else:
                        pnl = -stake
                        losses += 1
                        consecutive_losses += 1
                    
                    balance += pnl
                    balance_history.append(balance)
                    
                    trades.append({
                        'timestamp': data.index[i],
                        'direction': direction,
                        'stake': stake,
                        'probability': prob,
                        'confidence': confidence,
                        'win': win,
                        'pnl': pnl,
                        'balance': balance,
                        'consecutive_losses': consecutive_losses
                    })
            
            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # Calcular métricas avançadas
            if trades:
                pnls = [t['pnl'] for t in trades]
                winning_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
                losing_trades = [t['pnl'] for t in trades if t['pnl'] < 0]
                
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')
                
                # Drawdown
                peak = balance_history[0]
                max_drawdown = 0
                for balance_point in balance_history:
                    if balance_point > peak:
                        peak = balance_point
                    drawdown = (peak - balance_point) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Sharpe ratio simplificado
                returns = np.diff(balance_history) / balance_history[:-1]
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                avg_win = avg_loss = profit_factor = max_drawdown = sharpe_ratio = 0
            
            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': balance - 1000,
                'final_balance': balance,
                'roi': (balance - 1000) / 1000,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'max_consecutive_losses_hit': max(t.get('consecutive_losses', 0) for t in trades) if trades else 0,
                'balance_history': balance_history
            }
            
        except Exception as e:
            logger.error(f"Erro na simulação de trading: {e}")
            return {}
    
    def analyze_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> Dict:
        """
        Analisa qualidade das predições
        
        Args:
            y_true: Valores reais
            y_pred: Predições
            probabilities: Probabilidades
            
        Returns:
            Análise das predições
        """
        try:
            # Métricas básicas
            accuracy = np.mean(y_pred == y_true)
            
            # Distribuição de probabilidades
            prob_stats = {
                'mean': np.mean(probabilities),
                'std': np.std(probabilities),
                'min': np.min(probabilities),
                'max': np.max(probabilities),
                'q25': np.percentile(probabilities, 25),
                'q50': np.percentile(probabilities, 50),
                'q75': np.percentile(probabilities, 75)
            }
            
            # Análise por faixas de probabilidade
            prob_ranges = [
                (0.5, 0.6, 'low_confidence'),
                (0.6, 0.7, 'medium_confidence'),
                (0.7, 0.8, 'high_confidence'),
                (0.8, 1.0, 'very_high_confidence')
            ]
            
            range_analysis = {}
            for min_prob, max_prob, label in prob_ranges:
                mask = (probabilities >= min_prob) & (probabilities < max_prob)
                if np.sum(mask) > 0:
                    range_accuracy = np.mean(y_pred[mask] == y_true[mask])
                    range_analysis[label] = {
                        'count': np.sum(mask),
                        'accuracy': range_accuracy,
                        'percentage': np.sum(mask) / len(probabilities)
                    }
            
            # Calibração (reliability)
            calibration_bins = 10
            bin_boundaries = np.linspace(0.5, 1.0, calibration_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_data = []
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
                if np.sum(in_bin) > 0:
                    prob_in_bin = np.mean(probabilities[in_bin])
                    accuracy_in_bin = np.mean(y_true[in_bin])
                    calibration_data.append({
                        'bin_lower': bin_lower,
                        'bin_upper': bin_upper,
                        'prob_in_bin': prob_in_bin,
                        'accuracy_in_bin': accuracy_in_bin,
                        'count': np.sum(in_bin)
                    })
            
            return {
                'overall_accuracy': accuracy,
                'probability_stats': prob_stats,
                'confidence_ranges': range_analysis,
                'calibration_data': calibration_data,
                'total_predictions': len(y_pred)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de predições: {e}")
            return {}
    
    def run_comprehensive_ensemble_test(self) -> Dict:
        """
        Executa teste abrangente de estratégias ensemble
        
        Returns:
            Resultados completos dos testes
        """
        try:
            logger.info("Iniciando teste abrangente de estratégias ensemble")
            
            all_results = {
                'test_results': {},
                'summary': {},
                'timestamp': datetime.now().isoformat()
            }
            
            total_tests = len(self.symbols) * len(self.test_periods) * len(self.ensemble_configs)
            current_test = 0
            
            for symbol in self.symbols:
                all_results['test_results'][symbol] = {}
                
                for period_days in self.test_periods:
                    logger.info(f"Processando {symbol} - {period_days} dias")
                    
                    # Gerar dados históricos
                    data = self.generate_historical_data(symbol, period_days)
                    
                    if data.empty:
                        logger.warning(f"Pulando {symbol} ({period_days} dias) - erro na geração de dados")
                        continue
                    
                    # Adicionar features
                    data_with_features = self.add_comprehensive_features(data)
                    
                    if len(data_with_features) < 500:
                        logger.warning(f"Pulando {symbol} ({period_days} dias) - dados insuficientes")
                        continue
                    
                    all_results['test_results'][symbol][f'{period_days}d'] = {}
                    
                    # Testar cada configuração ensemble
                    for config in self.ensemble_configs:
                        current_test += 1
                        progress = (current_test / total_tests) * 100
                        
                        logger.info(f"Progresso: {progress:.1f}% - Testando {config['name']} para {symbol} ({period_days}d)")
                        
                        # Executar teste
                        test_result = self.test_ensemble_strategy(symbol, data_with_features, config, period_days)
                        
                        if 'error' not in test_result:
                            all_results['test_results'][symbol][f'{period_days}d'][config['name']] = test_result
            
            # Calcular resumo
            all_results['summary'] = self.calculate_ensemble_summary(all_results)
            
            # Salvar resultados
            results_file = os.path.join(
                self.results_dir,
                f"ensemble_historical_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"Teste ensemble completo finalizado. Resultados salvos em: {results_file}")
            return all_results
            
        except Exception as e:
            logger.error(f"Erro no teste ensemble completo: {e}")
            return {}
    
    def calculate_ensemble_summary(self, results: Dict) -> Dict:
        """
        Calcula resumo dos resultados ensemble
        
        Args:
            results: Resultados dos testes
            
        Returns:
            Resumo calculado
        """
        try:
            test_results = results.get('test_results', {})
            
            if not test_results:
                return {}
            
            # Coletar todas as métricas
            all_metrics = []
            
            for symbol, symbol_data in test_results.items():
                for period, period_data in symbol_data.items():
                    for config_name, config_data in period_data.items():
                        if 'error' not in config_data:
                            trading_results = config_data.get('trading_results', {})
                            
                            all_metrics.append({
                                'symbol': symbol,
                                'period': period,
                                'config': config_name,
                                'accuracy': config_data.get('accuracy', 0),
                                'win_rate': trading_results.get('win_rate', 0),
                                'roi': trading_results.get('roi', 0),
                                'total_trades': trading_results.get('total_trades', 0),
                                'profit_factor': trading_results.get('profit_factor', 0),
                                'max_drawdown': trading_results.get('max_drawdown', 0),
                                'sharpe_ratio': trading_results.get('sharpe_ratio', 0)
                            })
            
            if not all_metrics:
                return {}
            
            # Calcular estatísticas gerais
            accuracies = [m['accuracy'] for m in all_metrics]
            win_rates = [m['win_rate'] for m in all_metrics]
            rois = [m['roi'] for m in all_metrics]
            profit_factors = [m['profit_factor'] for m in all_metrics if m['profit_factor'] != float('inf')]
            
            # Encontrar melhores performances
            best_roi = max(all_metrics, key=lambda x: x['roi'])
            best_win_rate = max(all_metrics, key=lambda x: x['win_rate'])
            best_accuracy = max(all_metrics, key=lambda x: x['accuracy'])
            
            # Análise por configuração
            config_analysis = {}
            for config_name in ['conservative', 'balanced', 'aggressive']:
                config_metrics = [m for m in all_metrics if m['config'] == config_name]
                if config_metrics:
                    config_analysis[config_name] = {
                        'tests_count': len(config_metrics),
                        'avg_accuracy': np.mean([m['accuracy'] for m in config_metrics]),
                        'avg_win_rate': np.mean([m['win_rate'] for m in config_metrics]),
                        'avg_roi': np.mean([m['roi'] for m in config_metrics]),
                        'profitable_tests': len([m for m in config_metrics if m['roi'] > 0]),
                        'best_roi': max(m['roi'] for m in config_metrics)
                    }
            
            # Análise por símbolo
            symbol_analysis = {}
            for symbol in self.symbols:
                symbol_metrics = [m for m in all_metrics if m['symbol'] == symbol]
                if symbol_metrics:
                    symbol_analysis[symbol] = {
                        'tests_count': len(symbol_metrics),
                        'avg_accuracy': np.mean([m['accuracy'] for m in symbol_metrics]),
                        'avg_win_rate': np.mean([m['win_rate'] for m in symbol_metrics]),
                        'avg_roi': np.mean([m['roi'] for m in symbol_metrics]),
                        'best_config': max(symbol_metrics, key=lambda x: x['roi'])['config'],
                        'best_roi': max(m['roi'] for m in symbol_metrics)
                    }
            
            summary = {
                'total_tests': len(all_metrics),
                'overall_performance': {
                    'avg_accuracy': np.mean(accuracies),
                    'avg_win_rate': np.mean(win_rates),
                    'avg_roi': np.mean(rois),
                    'best_roi': max(rois),
                    'profitable_tests': len([roi for roi in rois if roi > 0]),
                    'avg_profit_factor': np.mean(profit_factors) if profit_factors else 0
                },
                'best_performances': {
                    'best_roi': best_roi,
                    'best_win_rate': best_win_rate,
                    'best_accuracy': best_accuracy
                },
                'config_analysis': config_analysis,
                'symbol_analysis': symbol_analysis,
                'recommendations': self.generate_recommendations(config_analysis, symbol_analysis)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao calcular resumo ensemble: {e}")
            return {}
    
    def generate_recommendations(self, config_analysis: Dict, symbol_analysis: Dict) -> List[str]:
        """
        Gera recomendações baseadas nos resultados
        
        Args:
            config_analysis: Análise por configuração
            symbol_analysis: Análise por símbolo
            
        Returns:
            Lista de recomendações
        """
        recommendations = []
        
        try:
            # Melhor configuração geral
            if config_analysis:
                best_config = max(config_analysis.items(), key=lambda x: x[1]['avg_roi'])
                recommendations.append(f"Configuração mais lucrativa: {best_config[0]} (ROI médio: {best_config[1]['avg_roi']:.2%})")
            
            # Melhor símbolo
            if symbol_analysis:
                best_symbol = max(symbol_analysis.items(), key=lambda x: x[1]['avg_roi'])
                recommendations.append(f"Símbolo mais lucrativo: {best_symbol[0]} (ROI médio: {best_symbol[1]['avg_roi']:.2%})")
            
            # Recomendações específicas
            for config_name, config_data in config_analysis.items():
                if config_data['avg_roi'] > 0.1:  # ROI > 10%
                    recommendations.append(f"Configuração {config_name} mostra resultados consistentes com {config_data['profitable_tests']} testes lucrativos")
            
            # Alertas
            for symbol, symbol_data in symbol_analysis.items():
                if symbol_data['avg_roi'] < 0:
                    recommendations.append(f"ATENÇÃO: {symbol} apresenta ROI médio negativo - revisar estratégia")
            
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {e}")
        
        return recommendations


def run_ensemble_historical_tests():
    """Função principal para executar testes ensemble históricos"""
    try:
        # Configurar logging
        setup_logging()
        
        print("\n" + "="*80)
        print("🧪 TESTE DE ESTRATÉGIAS ENSEMBLE EM DADOS HISTÓRICOS")
        print("="*80)
        print("📊 Testando múltiplas configurações ensemble...")
        print("⏱️  Múltiplos períodos de tempo...")
        print("🎯 Análise abrangente de performance...")
        print("="*80)
        
        # Criar testador
        tester = EnsembleHistoricalTester()
        
        # Executar testes
        results = tester.run_comprehensive_ensemble_test()
        
        if results and 'summary' in results:
            summary = results['summary']
            
            print("\n" + "="*80)
            print("📈 RESULTADOS DOS TESTES ENSEMBLE")
            print("="*80)
            
            print(f"📊 Total de testes: {summary.get('total_tests', 0)}")
            
            # Performance geral
            overall = summary.get('overall_performance', {})
            if overall:
                print(f"\n🎯 PERFORMANCE GERAL:")
                print(f"   • Accuracy médio: {overall.get('avg_accuracy', 0):.2%}")
                print(f"   • Win Rate médio: {overall.get('avg_win_rate', 0):.2%}")
                print(f"   • ROI médio: {overall.get('avg_roi', 0):.2%}")
                print(f"   • Melhor ROI: {overall.get('best_roi', 0):.2%}")
                print(f"   • Testes lucrativos: {overall.get('profitable_tests', 0)}")
                print(f"   • Profit Factor médio: {overall.get('avg_profit_factor', 0):.2f}")
            
            # Melhores performances
            best_perfs = summary.get('best_performances', {})
            if best_perfs:
                print(f"\n🏆 MELHORES PERFORMANCES:")
                
                best_roi = best_perfs.get('best_roi', {})
                if best_roi:
                    print(f"   • Melhor ROI: {best_roi.get('symbol', 'N/A')} - {best_roi.get('config', 'N/A')} ({best_roi.get('period', 'N/A')})")
                    print(f"     ROI: {best_roi.get('roi', 0):.2%}, Win Rate: {best_roi.get('win_rate', 0):.2%}")
                
                best_wr = best_perfs.get('best_win_rate', {})
                if best_wr:
                    print(f"   • Melhor Win Rate: {best_wr.get('symbol', 'N/A')} - {best_wr.get('config', 'N/A')} ({best_wr.get('period', 'N/A')})")
                    print(f"     Win Rate: {best_wr.get('win_rate', 0):.2%}, ROI: {best_wr.get('roi', 0):.2%}")
            
            # Análise por configuração
            config_analysis = summary.get('config_analysis', {})
            if config_analysis:
                print(f"\n⚙️ ANÁLISE POR CONFIGURAÇÃO:")
                for config_name, config_data in config_analysis.items():
                    print(f"   • {config_name.upper()}:")
                    print(f"     - Testes: {config_data.get('tests_count', 0)}")
                    print(f"     - ROI médio: {config_data.get('avg_roi', 0):.2%}")
                    print(f"     - Win Rate médio: {config_data.get('avg_win_rate', 0):.2%}")
                    print(f"     - Testes lucrativos: {config_data.get('profitable_tests', 0)}")
            
            # Análise por símbolo
            symbol_analysis = summary.get('symbol_analysis', {})
            if symbol_analysis:
                print(f"\n📊 ANÁLISE POR SÍMBOLO:")
                for symbol, symbol_data in symbol_analysis.items():
                    print(f"   • {symbol}:")
                    print(f"     - ROI médio: {symbol_data.get('avg_roi', 0):.2%}")
                    print(f"     - Melhor config: {symbol_data.get('best_config', 'N/A')}")
                    print(f"     - Melhor ROI: {symbol_data.get('best_roi', 0):.2%}")
            
            # Recomendações
            recommendations = summary.get('recommendations', [])
            if recommendations:
                print(f"\n💡 RECOMENDAÇÕES:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print(f"\n✅ Resultados salvos em: ensemble_test_results/")
            print("="*80)
            
        else:
            print("❌ Erro ao executar testes ensemble")
            
    except Exception as e:
        logger.error(f"Erro nos testes ensemble: {e}")
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    run_ensemble_historical_tests()