"""
Sistema Avançado de Backtesting para Estratégias de Trading com IA
Simula trading histórico com métricas detalhadas de performance
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

from data_collector import data_collector
from ml_model import TradingMLModel
from feature_engineering import FeatureEngineer
from config import config
from utils import get_logger

@dataclass
class BacktestTrade:
    """Representa um trade no backtesting"""
    entry_time: datetime
    exit_time: datetime
    signal: str  # CALL, PUT
    entry_price: float
    exit_price: float
    stake: float
    payout: float
    profit: float
    confidence: float
    duration: int
    symbol: str
    won: bool

@dataclass
class BacktestMetrics:
    """Métricas de performance do backtesting"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    max_drawdown: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_profit_per_trade: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float
    expectancy: float
    kelly_criterion: float

class AdvancedBacktester:
    """Sistema avançado de backtesting"""
    
    def __init__(self):
        self.logger = get_logger('backtester')
        self.feature_engineer = FeatureEngineer()
        
        # Configurações de backtesting
        self.initial_balance = 1000.0
        self.default_stake = 1.0
        self.payout_ratio = 0.85  # 85% payout típico da Deriv
        self.commission = 0.0  # Sem comissão na Deriv
        
        # Resultados
        self.trades = []
        self.balance_history = []
        self.equity_curve = []
        self.metrics = None
        
    def run_backtest(self, 
                    data: pd.DataFrame,
                    model: TradingMLModel,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    initial_balance: float = 1000.0,
                    stake_amount: float = 1.0,
                    min_confidence: float = 0.6,
                    max_trades_per_day: int = 50,
                    trade_duration: int = 5) -> Dict[str, Any]:
        """
        Executa backtesting completo
        
        Args:
            data: DataFrame com dados históricos
            model: Modelo de ML treinado
            start_date: Data de início (opcional)
            end_date: Data de fim (opcional)
            initial_balance: Saldo inicial
            stake_amount: Valor base do stake
            min_confidence: Confiança mínima para trade
            max_trades_per_day: Máximo de trades por dia
            trade_duration: Duração do trade em ticks/minutos
        
        Returns:
            Dict com resultados do backtesting
        """
        try:
            self.logger.info("Iniciando backtesting avançado...")
            
            # Resetar estado
            self._reset_state()
            self.initial_balance = initial_balance
            self.default_stake = stake_amount
            
            # Filtrar dados por período se especificado
            if start_date or end_date:
                data = self._filter_data_by_period(data, start_date, end_date)
            
            if data.empty:
                raise ValueError("Dados insuficientes para backtesting")
            
            # Preparar dados com features
            self.logger.info("Preparando features...")
            features_data = self.feature_engineer.create_features(data)
            
            if features_data.empty:
                raise ValueError("Falha ao criar features")
            
            # Executar simulação
            self.logger.info("Executando simulação...")
            self._simulate_trading(features_data, model, min_confidence, 
                                 max_trades_per_day, trade_duration)
            
            # Calcular métricas
            self.logger.info("Calculando métricas...")
            self.metrics = self._calculate_metrics()
            
            # Gerar relatório
            report = self._generate_report()
            
            self.logger.info("Backtesting concluído!")
            
            return {
                "success": True,
                "metrics": self.metrics.__dict__,
                "trades": [trade.__dict__ for trade in self.trades],
                "balance_history": self.balance_history,
                "equity_curve": self.equity_curve,
                "report": report
            }
            
        except Exception as e:
            self.logger.error(f"Erro no backtesting: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": None
            }
    
    def _reset_state(self):
        """Reseta estado do backtester"""
        self.trades.clear()
        self.balance_history.clear()
        self.equity_curve.clear()
        self.metrics = None
    
    def _filter_data_by_period(self, data: pd.DataFrame, 
                              start_date: Optional[datetime],
                              end_date: Optional[datetime]) -> pd.DataFrame:
        """Filtra dados por período"""
        filtered_data = data.copy()
        
        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]
        
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]
        
        return filtered_data
    
    def _simulate_trading(self, data: pd.DataFrame, model: TradingMLModel,
                         min_confidence: float, max_trades_per_day: int,
                         trade_duration: int):
        """Simula trading com o modelo"""
        current_balance = self.initial_balance
        daily_trades = 0
        last_trade_date = None
        
        # Janela deslizante para predições
        window_size = 100
        
        for i in range(window_size, len(data)):
            current_time = data.index[i]
            current_date = current_time.date()
            
            # Resetar contador diário
            if last_trade_date != current_date:
                daily_trades = 0
                last_trade_date = current_date
            
            # Verificar limite diário
            if daily_trades >= max_trades_per_day:
                continue
            
            # Preparar dados para predição (janela histórica)
            window_data = data.iloc[i-window_size:i].copy()
            
            try:
                # Gerar sinal
                signal = model.get_trading_signal(window_data, min_confidence)
                
                if signal['action'] != 'HOLD' and signal['confidence'] >= min_confidence:
                    # Calcular stake baseado na confiança
                    stake = self._calculate_dynamic_stake(
                        current_balance, signal['confidence'], self.default_stake
                    )
                    
                    # Verificar se tem saldo suficiente
                    if stake > current_balance:
                        continue
                    
                    # Simular entrada do trade
                    entry_price = data.iloc[i]['close']
                    entry_time = current_time
                    
                    # Simular saída do trade (após duração especificada)
                    exit_index = min(i + trade_duration, len(data) - 1)
                    exit_price = data.iloc[exit_index]['close']
                    exit_time = data.index[exit_index]
                    
                    # Determinar resultado
                    if signal['action'] == 'CALL':
                        won = exit_price > entry_price
                    else:  # PUT
                        won = exit_price < entry_price
                    
                    # Calcular payout e lucro
                    if won:
                        payout = stake * (1 + self.payout_ratio)
                        profit = payout - stake
                    else:
                        payout = 0
                        profit = -stake
                    
                    # Atualizar saldo
                    current_balance += profit
                    
                    # Criar registro do trade
                    trade = BacktestTrade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        signal=signal['action'],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stake=stake,
                        payout=payout,
                        profit=profit,
                        confidence=signal['confidence'],
                        duration=trade_duration,
                        symbol=data.get('symbol', ['R_10'])[0] if 'symbol' in data.columns else 'R_10',
                        won=won
                    )
                    
                    self.trades.append(trade)
                    daily_trades += 1
                    
                    # Registrar saldo
                    self.balance_history.append({
                        'timestamp': current_time,
                        'balance': current_balance,
                        'trade_profit': profit
                    })
                    
            except Exception as e:
                self.logger.warning(f"Erro ao processar sinal em {current_time}: {e}")
                continue
        
        # Criar curva de equity
        self._create_equity_curve()
    
    def _calculate_dynamic_stake(self, balance: float, confidence: float, base_stake: float) -> float:
        """Calcula stake dinâmico baseado na confiança e saldo"""
        # Kelly Criterion simplificado
        win_prob = confidence
        loss_prob = 1 - confidence
        
        # Assumir payout ratio
        b = self.payout_ratio  # odds
        
        # Kelly fraction
        kelly_fraction = (win_prob * (1 + b) - loss_prob) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Limitar a 25%
        
        # Calcular stake
        kelly_stake = balance * kelly_fraction
        
        # Usar o menor entre Kelly e stake baseado na confiança
        confidence_stake = base_stake * (1 + (confidence - 0.5) * 2)  # Aumentar com confiança
        
        final_stake = min(kelly_stake, confidence_stake, balance * 0.05)  # Max 5% do saldo
        
        return max(final_stake, 0.1)  # Mínimo $0.10
    
    def _create_equity_curve(self):
        """Cria curva de equity"""
        if not self.balance_history:
            return
        
        balance_df = pd.DataFrame(self.balance_history)
        balance_df.set_index('timestamp', inplace=True)
        
        # Resample para intervalos regulares
        daily_balance = balance_df.resample('D')['balance'].last().fillna(method='ffill')
        
        self.equity_curve = [
            {'date': date, 'balance': balance, 'return': (balance / self.initial_balance - 1) * 100}
            for date, balance in daily_balance.items()
        ]
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calcula métricas detalhadas de performance"""
        if not self.trades:
            return BacktestMetrics(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
                total_profit=0, total_loss=0, net_profit=0, max_drawdown=0,
                max_consecutive_wins=0, max_consecutive_losses=0, avg_profit_per_trade=0,
                profit_factor=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                recovery_factor=0, expectancy=0, kelly_criterion=0
            )
        
        # Métricas básicas
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.won)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Lucros e perdas
        profits = [trade.profit for trade in self.trades if trade.profit > 0]
        losses = [abs(trade.profit) for trade in self.trades if trade.profit < 0]
        
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        net_profit = total_profit - total_loss
        
        # Profit Factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Sequências consecutivas
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_streaks()
        
        # Médias
        avg_profit_per_trade = net_profit / total_trades if total_trades > 0 else 0
        avg_win = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)
        
        # Ratios de risco
        returns = [trade.profit / trade.stake for trade in self.trades]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = (net_profit / self.initial_balance) / (max_drawdown / 100) if max_drawdown > 0 else 0
        recovery_factor = net_profit / max_drawdown if max_drawdown > 0 else 0
        
        # Kelly Criterion
        kelly_criterion = self._calculate_kelly_criterion(win_rate / 100, avg_win, avg_loss)
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            max_drawdown=max_drawdown,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            avg_profit_per_trade=avg_profit_per_trade,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            recovery_factor=recovery_factor,
            expectancy=expectancy,
            kelly_criterion=kelly_criterion
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calcula máximo drawdown"""
        if not self.balance_history:
            return 0
        
        balances = [entry['balance'] for entry in self.balance_history]
        peak = balances[0]
        max_dd = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            
            drawdown = (peak - balance) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_consecutive_streaks(self) -> Tuple[int, int]:
        """Calcula sequências consecutivas de vitórias e derrotas"""
        if not self.trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.won:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calcula Sharpe Ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return / std_return if std_return > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calcula Sortino Ratio"""
        if not returns:
            return 0
        
        mean_return = np.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        
        return mean_return / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calcula Kelly Criterion"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss  # odds
        p = win_rate  # probability of winning
        
        kelly = p - ((1 - p) / b)
        return max(0, kelly)  # Não permitir valores negativos
    
    def _generate_report(self) -> str:
        """Gera relatório detalhado do backtesting"""
        if not self.metrics:
            return "Nenhuma métrica disponível"
        
        m = self.metrics
        
        report = f"""
📊 RELATÓRIO DE BACKTESTING AVANÇADO
{'='*60}

📈 PERFORMANCE GERAL
• Total de Trades: {m.total_trades}
• Trades Vencedores: {m.winning_trades}
• Trades Perdedores: {m.losing_trades}
• Taxa de Acerto: {m.win_rate:.2f}%
• Lucro Líquido: ${m.net_profit:.2f}
• Lucro Médio por Trade: ${m.avg_profit_per_trade:.2f}

💰 ANÁLISE FINANCEIRA
• Lucro Total: ${m.total_profit:.2f}
• Perda Total: ${m.total_loss:.2f}
• Profit Factor: {m.profit_factor:.2f}
• Expectancy: ${m.expectancy:.2f}
• Máximo Drawdown: {m.max_drawdown:.2f}%

🎯 ANÁLISE DE RISCO
• Sharpe Ratio: {m.sharpe_ratio:.3f}
• Sortino Ratio: {m.sortino_ratio:.3f}
• Calmar Ratio: {m.calmar_ratio:.3f}
• Recovery Factor: {m.recovery_factor:.2f}
• Kelly Criterion: {m.kelly_criterion:.3f}

📊 SEQUÊNCIAS
• Máx. Vitórias Consecutivas: {m.max_consecutive_wins}
• Máx. Derrotas Consecutivas: {m.max_consecutive_losses}

💡 RECOMENDAÇÕES
"""
        
        # Adicionar recomendações baseadas nas métricas
        if m.win_rate >= 60:
            report += "✅ Excelente taxa de acerto\n"
        elif m.win_rate >= 50:
            report += "⚠️ Taxa de acerto aceitável\n"
        else:
            report += "❌ Taxa de acerto baixa - revisar estratégia\n"
        
        if m.profit_factor >= 1.5:
            report += "✅ Excelente profit factor\n"
        elif m.profit_factor >= 1.2:
            report += "⚠️ Profit factor aceitável\n"
        else:
            report += "❌ Profit factor baixo\n"
        
        if m.max_drawdown <= 10:
            report += "✅ Drawdown controlado\n"
        elif m.max_drawdown <= 20:
            report += "⚠️ Drawdown moderado\n"
        else:
            report += "❌ Drawdown alto - revisar gerenciamento de risco\n"
        
        report += f"\n{'='*60}"
        
        return report
    
    def save_results(self, filepath: str):
        """Salva resultados do backtesting"""
        try:
            results = {
                "metrics": self.metrics.__dict__ if self.metrics else {},
                "trades": [trade.__dict__ for trade in self.trades],
                "balance_history": self.balance_history,
                "equity_curve": self.equity_curve,
                "config": {
                    "initial_balance": self.initial_balance,
                    "default_stake": self.default_stake,
                    "payout_ratio": self.payout_ratio
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Resultados salvos em {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados: {e}")
    
    def plot_results(self, save_path: Optional[str] = None):
        """Gera gráficos dos resultados"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Curva de Equity
            if self.equity_curve:
                dates = [entry['date'] for entry in self.equity_curve]
                balances = [entry['balance'] for entry in self.equity_curve]
                
                axes[0, 0].plot(dates, balances, linewidth=2)
                axes[0, 0].set_title('Curva de Equity')
                axes[0, 0].set_ylabel('Saldo ($)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Distribuição de Lucros/Perdas
            if self.trades:
                profits = [trade.profit for trade in self.trades]
                axes[0, 1].hist(profits, bins=30, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Distribuição de Lucros/Perdas')
                axes[0, 1].set_xlabel('Lucro ($)')
                axes[0, 1].set_ylabel('Frequência')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Trades por Dia
            if self.trades:
                trade_dates = [trade.entry_time.date() for trade in self.trades]
                daily_counts = pd.Series(trade_dates).value_counts().sort_index()
                
                axes[1, 0].plot(daily_counts.index, daily_counts.values, marker='o')
                axes[1, 0].set_title('Trades por Dia')
                axes[1, 0].set_ylabel('Número de Trades')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Win Rate por Confiança
            if self.trades:
                confidence_bins = np.arange(0.5, 1.01, 0.05)
                win_rates = []
                
                for i in range(len(confidence_bins) - 1):
                    bin_trades = [t for t in self.trades 
                                if confidence_bins[i] <= t.confidence < confidence_bins[i+1]]
                    if bin_trades:
                        win_rate = sum(1 for t in bin_trades if t.won) / len(bin_trades) * 100
                        win_rates.append(win_rate)
                    else:
                        win_rates.append(0)
                
                bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
                axes[1, 1].bar(bin_centers, win_rates, width=0.04, alpha=0.7)
                axes[1, 1].set_title('Win Rate por Nível de Confiança')
                axes[1, 1].set_xlabel('Confiança')
                axes[1, 1].set_ylabel('Win Rate (%)')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Gráficos salvos em {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar gráficos: {e}")

# Instância global
advanced_backtester = AdvancedBacktester()

# Funções de conveniência
def run_strategy_backtest(data: pd.DataFrame, model: TradingMLModel, **kwargs) -> Dict[str, Any]:
    """Executa backtesting de uma estratégia"""
    return advanced_backtester.run_backtest(data, model, **kwargs)

def quick_backtest(symbol: str = "R_10", days: int = 30, **kwargs) -> Dict[str, Any]:
    """Executa backtesting rápido com dados recentes"""
    try:
        # Coletar dados históricos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        historical_data = data_collector.get_historical_data(symbol, "1m", days * 1440)
        
        if historical_data.empty:
            return {"success": False, "error": "Dados insuficientes"}
        
        # Criar e treinar modelo rápido
        from ml_model import create_and_train_model
        model = create_and_train_model(historical_data)
        
        # Executar backtesting
        return advanced_backtester.run_backtest(historical_data, model, **kwargs)
        
    except Exception as e:
        return {"success": False, "error": str(e)}