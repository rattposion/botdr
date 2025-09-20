"""
Módulo de Backtest para Validação de Estratégias
Simula trading histórico para avaliar performance de estratégias
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from ml_model import TradingMLModel
from feature_engineering import FeatureEngineer
from utils import get_logger, calculate_sharpe_ratio, calculate_max_drawdown, format_currency, format_percentage
from config import config

@dataclass
class BacktestTrade:
    """Representa um trade no backtest"""
    timestamp: datetime
    signal: str
    confidence: float
    stake: float
    entry_price: float
    exit_price: float
    duration: int
    pnl: float
    pnl_percentage: float
    martingale_step: int
    balance_before: float
    balance_after: float
    features: Dict[str, Any]

@dataclass
class BacktestResults:
    """Resultados do backtest"""
    # Métricas básicas
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Métricas financeiras
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    
    # Métricas de risco
    max_consecutive_losses: int
    max_consecutive_wins: int
    largest_win: float
    largest_loss: float
    
    # Dados para análise
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    
    # Configurações
    initial_balance: float
    final_balance: float
    start_date: datetime
    end_date: datetime

class Backtester:
    """Motor de backtest"""
    
    def __init__(self):
        self.logger = get_logger('backtester')
        self.feature_engineer = FeatureEngineer()
        self.model = None
        
        # Configurações de backtest
        self.initial_balance = config.trading.initial_balance
        self.initial_stake = config.trading.initial_stake
        self.max_stake = config.trading.max_stake
        self.min_confidence = config.trading.min_prediction_confidence
        self.enable_martingale = config.trading.enable_martingale
        self.martingale_multiplier = config.trading.martingale_multiplier
        self.max_martingale_steps = config.trading.max_martingale_steps
        self.max_daily_trades = config.trading.max_daily_trades
        self.max_daily_loss = config.trading.max_daily_loss
        self.contract_duration = config.trading.contract_duration
        
        # Estado do backtest
        self.current_balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self.martingale_step = 0
        self.daily_trades = 0
        self.daily_pnl = 0
        self.last_trade_date = None
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    model: TradingMLModel,
                    start_date: str = None,
                    end_date: str = None,
                    commission: float = 0.0) -> BacktestResults:
        """
        Executa backtest completo
        
        Args:
            data: DataFrame com dados históricos (timestamp, quote)
            model: Modelo ML treinado
            start_date: Data de início (formato YYYY-MM-DD)
            end_date: Data de fim (formato YYYY-MM-DD)
            commission: Comissão por trade
        """
        self.logger.info("Iniciando backtest...")
        
        # Preparar dados
        data = self._prepare_data(data, start_date, end_date)
        if data.empty:
            raise ValueError("Dados insuficientes para backtest")
        
        self.model = model
        
        # Reset estado
        self._reset_state()
        
        # Gerar features
        self.logger.info("Gerando features...")
        features_df = self.feature_engineer.create_features(data)
        
        if features_df.empty:
            raise ValueError("Falha ao gerar features")
        
        # Executar simulação
        self.logger.info(f"Executando simulação de {len(features_df)} pontos...")
        self._run_simulation(features_df, commission)
        
        # Calcular resultados
        results = self._calculate_results(data.iloc[0]['timestamp'], data.iloc[-1]['timestamp'])
        
        self.logger.info(f"Backtest concluído: {results.total_trades} trades, "
                        f"Win Rate: {results.win_rate:.1%}, "
                        f"Total Return: {results.total_return:.1%}")
        
        return results
    
    def _prepare_data(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Prepara dados para backtest"""
        # Converter timestamp se necessário
        if 'timestamp' not in data.columns:
            if 'epoch' in data.columns:
                data['timestamp'] = pd.to_datetime(data['epoch'], unit='s')
            else:
                raise ValueError("Coluna 'timestamp' ou 'epoch' não encontrada")
        
        # Garantir que timestamp é datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Filtrar por datas
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data['timestamp'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data['timestamp'] <= end_date]
        
        # Ordenar por timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Verificar colunas necessárias
        required_columns = ['timestamp', 'quote']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Colunas obrigatórias não encontradas: {missing_columns}")
        
        return data
    
    def _reset_state(self):
        """Reset estado do backtest"""
        self.current_balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self.martingale_step = 0
        self.daily_trades = 0
        self.daily_pnl = 0
        self.last_trade_date = None
    
    def _run_simulation(self, features_df: pd.DataFrame, commission: float):
        """Executa simulação principal"""
        for i in range(len(features_df)):
            row = features_df.iloc[i]
            
            # Reset contadores diários
            self._reset_daily_counters(row['timestamp'])
            
            # Verificar se pode fazer trade
            if not self._can_trade():
                continue
            
            # Gerar sinal
            signal = self._generate_signal(row)
            
            if signal['type'] != 'HOLD':
                # Executar trade
                trade = self._execute_backtest_trade(row, signal, commission)
                if trade:
                    self.trades.append(trade)
                    self._update_equity_curve(trade.timestamp, trade.balance_after)
    
    def _reset_daily_counters(self, current_timestamp: pd.Timestamp):
        """Reset contadores diários"""
        current_date = current_timestamp.date()
        
        if self.last_trade_date is None or current_date != self.last_trade_date:
            self.daily_trades = 0
            self.daily_pnl = 0
            self.last_trade_date = current_date
    
    def _can_trade(self) -> bool:
        """Verifica se pode fazer trade"""
        # Verificar limite de trades diários
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        # Verificar stop loss diário
        if self.daily_pnl <= -self.max_daily_loss:
            return False
        
        # Verificar limite de martingale
        if self.enable_martingale and self.martingale_step >= self.max_martingale_steps:
            return False
        
        # Verificar saldo mínimo
        min_stake = self._calculate_stake()
        if self.current_balance < min_stake:
            return False
        
        return True
    
    def _generate_signal(self, row: pd.Series) -> Dict[str, Any]:
        """Gera sinal usando modelo ML"""
        try:
            # Preparar dados para predição
            features = row.drop(['timestamp', 'target'] if 'target' in row.index else ['timestamp'])
            features_df = pd.DataFrame([features])
            
            # Fazer predição
            prediction = self.model.predict(features_df)
            
            if prediction is None:
                return {'type': 'HOLD', 'confidence': 0}
            
            signal_type, confidence = prediction
            
            # Verificar confiança mínima
            if confidence < self.min_confidence:
                return {'type': 'HOLD', 'confidence': confidence}
            
            return {
                'type': signal_type,
                'confidence': confidence,
                'features': features.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal: {e}")
            return {'type': 'HOLD', 'confidence': 0}
    
    def _execute_backtest_trade(self, row: pd.Series, signal: Dict[str, Any], commission: float) -> Optional[BacktestTrade]:
        """Executa trade no backtest"""
        try:
            # Calcular stake
            stake = self._calculate_stake()
            
            # Simular entrada
            entry_price = row['quote']
            entry_time = row['timestamp']
            
            # Simular saída (simplificado - assume que o sinal está correto)
            # Em um backtest real, você precisaria dos dados futuros para determinar o resultado
            exit_price = self._simulate_exit_price(entry_price, signal['type'], signal['confidence'])
            
            # Calcular PnL
            pnl = self._calculate_pnl(stake, entry_price, exit_price, signal['type'], commission)
            
            # Atualizar saldo
            balance_before = self.current_balance
            self.current_balance += pnl
            
            # Atualizar contadores
            self.daily_trades += 1
            self.daily_pnl += pnl
            
            # Atualizar martingale
            if pnl > 0:
                self.martingale_step = 0
            elif self.enable_martingale:
                self.martingale_step += 1
            
            # Criar trade
            trade = BacktestTrade(
                timestamp=entry_time,
                signal=signal['type'],
                confidence=signal['confidence'],
                stake=stake,
                entry_price=entry_price,
                exit_price=exit_price,
                duration=self.contract_duration,
                pnl=pnl,
                pnl_percentage=(pnl / stake) * 100,
                martingale_step=self.martingale_step,
                balance_before=balance_before,
                balance_after=self.current_balance,
                features=signal['features']
            )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Erro ao executar trade no backtest: {e}")
            return None
    
    def _calculate_stake(self) -> float:
        """Calcula stake para o trade"""
        base_stake = self.initial_stake
        
        if self.enable_martingale and self.martingale_step > 0:
            stake = base_stake * (self.martingale_multiplier ** self.martingale_step)
            return min(stake, self.max_stake)
        
        return base_stake
    
    def _simulate_exit_price(self, entry_price: float, signal_type: str, confidence: float) -> float:
        """
        Simula preço de saída (simplificado)
        Em um backtest real, você usaria dados futuros reais
        """
        # Simulação baseada na confiança do modelo
        # Maior confiança = maior probabilidade de acerto
        
        # Gerar resultado aleatório baseado na confiança
        win_probability = confidence
        is_win = np.random.random() < win_probability
        
        if signal_type == 'CALL':
            if is_win:
                return entry_price * (1 + np.random.uniform(0.001, 0.01))  # Ganho de 0.1% a 1%
            else:
                return entry_price * (1 - np.random.uniform(0.001, 0.01))  # Perda de 0.1% a 1%
        else:  # PUT
            if is_win:
                return entry_price * (1 - np.random.uniform(0.001, 0.01))  # Ganho (preço caiu)
            else:
                return entry_price * (1 + np.random.uniform(0.001, 0.01))  # Perda (preço subiu)
    
    def _calculate_pnl(self, stake: float, entry_price: float, exit_price: float, signal_type: str, commission: float) -> float:
        """Calcula PnL do trade"""
        # Para contratos binários, o PnL é tipicamente fixo
        # Simplificação: ganho = 80% do stake, perda = -100% do stake
        
        if signal_type == 'CALL':
            is_win = exit_price > entry_price
        else:  # PUT
            is_win = exit_price < entry_price
        
        if is_win:
            pnl = stake * 0.8  # 80% de retorno
        else:
            pnl = -stake  # Perda total do stake
        
        # Subtrair comissão
        pnl -= commission
        
        return pnl
    
    def _update_equity_curve(self, timestamp: pd.Timestamp, balance: float):
        """Atualiza curva de equity"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': balance
        })
    
    def _calculate_results(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> BacktestResults:
        """Calcula resultados finais do backtest"""
        if not self.trades:
            return self._empty_results(start_date, end_date)
        
        # Métricas básicas
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Métricas financeiras
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Criar série de equity
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_series = equity_df.set_index('timestamp')['balance']
            max_drawdown = calculate_max_drawdown(equity_series)
            
            # Calcular retornos diários
            daily_returns = equity_series.resample('D').last().pct_change().dropna()
            sharpe_ratio = calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 1 else 0
        else:
            equity_series = pd.Series()
            daily_returns = pd.Series()
            max_drawdown = 0
            sharpe_ratio = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Métricas de risco
        consecutive_wins = self._calculate_consecutive_wins()
        consecutive_losses = self._calculate_consecutive_losses()
        largest_win = max((t.pnl for t in self.trades), default=0)
        largest_loss = min((t.pnl for t in self.trades), default=0)
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            max_consecutive_losses=consecutive_losses,
            max_consecutive_wins=consecutive_wins,
            largest_win=largest_win,
            largest_loss=largest_loss,
            trades=self.trades,
            equity_curve=equity_series,
            daily_returns=daily_returns,
            initial_balance=self.initial_balance,
            final_balance=self.current_balance,
            start_date=start_date,
            end_date=end_date
        )
    
    def _empty_results(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> BacktestResults:
        """Retorna resultados vazios"""
        return BacktestResults(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            total_return=0,
            max_drawdown=0,
            sharpe_ratio=0,
            profit_factor=0,
            max_consecutive_losses=0,
            max_consecutive_wins=0,
            largest_win=0,
            largest_loss=0,
            trades=[],
            equity_curve=pd.Series(),
            daily_returns=pd.Series(),
            initial_balance=self.initial_balance,
            final_balance=self.initial_balance,
            start_date=start_date,
            end_date=end_date
        )
    
    def _calculate_consecutive_wins(self) -> int:
        """Calcula máximo de vitórias consecutivas"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self) -> int:
        """Calcula máximo de perdas consecutivas"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

class BacktestAnalyzer:
    """Analisador de resultados de backtest"""
    
    def __init__(self):
        self.logger = get_logger('backtest_analyzer')
    
    def generate_report(self, results: BacktestResults) -> str:
        """Gera relatório detalhado"""
        report = f"""
        📊 RELATÓRIO DE BACKTEST
        ========================
        
        📅 Período: {results.start_date.strftime('%Y-%m-%d')} a {results.end_date.strftime('%Y-%m-%d')}
        
        💰 PERFORMANCE FINANCEIRA
        -------------------------
        Saldo Inicial: {format_currency(results.initial_balance)}
        Saldo Final: {format_currency(results.final_balance)}
        PnL Total: {format_currency(results.total_pnl)}
        Retorno Total: {format_percentage(results.total_return)}
        
        📈 MÉTRICAS DE TRADING
        ----------------------
        Total de Trades: {results.total_trades}
        Trades Vencedores: {results.winning_trades}
        Trades Perdedores: {results.losing_trades}
        Taxa de Acerto: {format_percentage(results.win_rate)}
        
        📊 MÉTRICAS DE RISCO
        --------------------
        Máximo Drawdown: {format_percentage(results.max_drawdown)}
        Sharpe Ratio: {results.sharpe_ratio:.2f}
        Profit Factor: {results.profit_factor:.2f}
        
        🎯 EXTREMOS
        -----------
        Maior Ganho: {format_currency(results.largest_win)}
        Maior Perda: {format_currency(results.largest_loss)}
        Máx. Vitórias Consecutivas: {results.max_consecutive_wins}
        Máx. Perdas Consecutivas: {results.max_consecutive_losses}
        """
        
        return report
    
    def plot_results(self, results: BacktestResults, save_path: str = None):
        """Gera gráficos dos resultados"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise de Backtest', fontsize=16)
        
        # 1. Curva de Equity
        if not results.equity_curve.empty:
            axes[0, 0].plot(results.equity_curve.index, results.equity_curve.values)
            axes[0, 0].set_title('Curva de Equity')
            axes[0, 0].set_ylabel('Saldo ($)')
            axes[0, 0].grid(True)
        
        # 2. Distribuição de PnL
        if results.trades:
            pnl_values = [t.pnl for t in results.trades]
            axes[0, 1].hist(pnl_values, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Distribuição de PnL')
            axes[0, 1].set_xlabel('PnL ($)')
            axes[0, 1].set_ylabel('Frequência')
            axes[0, 1].grid(True)
        
        # 3. Retornos Diários
        if not results.daily_returns.empty:
            axes[1, 0].plot(results.daily_returns.index, results.daily_returns.values)
            axes[1, 0].set_title('Retornos Diários')
            axes[1, 0].set_ylabel('Retorno (%)')
            axes[1, 0].grid(True)
        
        # 4. Métricas Resumo
        axes[1, 1].axis('off')
        metrics_text = f"""
        Total Trades: {results.total_trades}
        Win Rate: {results.win_rate:.1%}
        Total Return: {results.total_return:.1%}
        Max Drawdown: {results.max_drawdown:.1%}
        Sharpe Ratio: {results.sharpe_ratio:.2f}
        Profit Factor: {results.profit_factor:.2f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Gráficos salvos em: {save_path}")
        
        plt.show()
    
    def export_trades(self, results: BacktestResults, filename: str):
        """Exporta trades para CSV"""
        if not results.trades:
            self.logger.warning("Nenhum trade para exportar")
            return
        
        trades_data = []
        for trade in results.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'signal': trade.signal,
                'confidence': trade.confidence,
                'stake': trade.stake,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'duration': trade.duration,
                'pnl': trade.pnl,
                'pnl_percentage': trade.pnl_percentage,
                'martingale_step': trade.martingale_step,
                'balance_before': trade.balance_before,
                'balance_after': trade.balance_after
            })
        
        df = pd.DataFrame(trades_data)
        df.to_csv(filename, index=False)
        self.logger.info(f"Trades exportados para: {filename}")

# Funções de conveniência
def run_simple_backtest(data: pd.DataFrame, model: TradingMLModel) -> BacktestResults:
    """Executa backtest simples"""
    backtester = Backtester()
    return backtester.run_backtest(data, model)

def analyze_backtest_results(results: BacktestResults, show_plots: bool = True) -> str:
    """Analisa resultados de backtest"""
    analyzer = BacktestAnalyzer()
    report = analyzer.generate_report(results)
    
    if show_plots:
        analyzer.plot_results(results)
    
    return report

def compare_strategies(results_list: List[BacktestResults], strategy_names: List[str]) -> pd.DataFrame:
    """Compara múltiplas estratégias"""
    comparison_data = []
    
    for results, name in zip(results_list, strategy_names):
        comparison_data.append({
            'Strategy': name,
            'Total Trades': results.total_trades,
            'Win Rate': results.win_rate,
            'Total Return': results.total_return,
            'Max Drawdown': results.max_drawdown,
            'Sharpe Ratio': results.sharpe_ratio,
            'Profit Factor': results.profit_factor
        })
    
    return pd.DataFrame(comparison_data)