"""
Utilitários do Sistema de Trading
Inclui logging, relatórios, persistência de dados e funções auxiliares
"""
import os
import csv
import json
import logging
import logging.handlers
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import colorlog

from config import config

class TradingLogger:
    """Sistema de logging personalizado para trading"""
    
    def __init__(self):
        self.logger = None
        self.setup_logging()
    
    def setup_logging(self):
        """Configura sistema de logging"""
        # Criar diretório de logs
        os.makedirs(config.data.logs_dir, exist_ok=True)
        
        # Configurar logger principal
        self.logger = logging.getLogger('trading_bot')
        self.logger.setLevel(getattr(logging, config.logging.log_level))
        
        # Remover handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Handler para arquivo
        file_handler = logging.handlers.RotatingFileHandler(
            config.logging.log_file,
            maxBytes=config.logging.max_log_size,
            backupCount=config.logging.backup_count
        )
        file_formatter = logging.Formatter(config.logging.log_format)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para console (com cores se habilitado)
        if config.logging.console_logging:
            console_handler = logging.StreamHandler()
            
            if config.logging.colored_logs:
                console_formatter = colorlog.ColoredFormatter(
                    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                )
            else:
                console_formatter = logging.Formatter(config.logging.log_format)
            
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Configurar loggers de bibliotecas
        logging.getLogger('websocket').setLevel(logging.WARNING)
        logging.getLogger('lightgbm').setLevel(logging.WARNING)
        
        self.logger.info("Sistema de logging inicializado")
    
    def get_logger(self, name: str = None):
        """Retorna logger"""
        if name:
            return logging.getLogger(f'trading_bot.{name}')
        return self.logger

class TradeRecorder:
    """Registra trades em CSV para análise"""
    
    def __init__(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{config.data.reports_dir}/trades_{timestamp}.csv"
        
        self.filename = filename
        self.fieldnames = [
            'timestamp', 'symbol', 'signal', 'stake', 'contract_type',
            'entry_price', 'exit_price', 'duration', 'pnl', 'pnl_percentage',
            'confidence', 'features_used', 'martingale_step', 'balance_before',
            'balance_after', 'status', 'error_message'
        ]
        
        # Criar arquivo se não existir
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Garante que o arquivo CSV existe com headers"""
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Registra um trade"""
        # Adicionar timestamp se não existir
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
        
        # Garantir que todos os campos existem
        for field in self.fieldnames:
            if field not in trade_data:
                trade_data[field] = None
        
        # Escrever no CSV
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(trade_data)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Retorna trades como DataFrame"""
        try:
            df = pd.read_csv(self.filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            return pd.DataFrame(columns=self.fieldnames)
    
    def get_daily_summary(self, date: datetime = None) -> Dict[str, Any]:
        """Retorna resumo diário"""
        if date is None:
            date = datetime.now().date()
        
        df = self.get_trades_df()
        if df.empty:
            return self._empty_summary()
        
        # Filtrar por data
        df['date'] = df['timestamp'].dt.date
        daily_df = df[df['date'] == date]
        
        if daily_df.empty:
            return self._empty_summary()
        
        # Calcular métricas
        total_trades = len(daily_df)
        winning_trades = len(daily_df[daily_df['pnl'] > 0])
        losing_trades = len(daily_df[daily_df['pnl'] < 0])
        
        total_pnl = daily_df['pnl'].sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = daily_df[daily_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = daily_df[daily_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        return {
            'date': date.isoformat(),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0
        }
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Retorna resumo vazio"""
        return {
            'date': datetime.now().date().isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }

class PerformanceTracker:
    """Rastreia performance do sistema"""
    
    def __init__(self):
        self.metrics_file = f"{config.data.reports_dir}/performance_metrics.json"
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Carrega métricas salvas"""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'start_date': datetime.now().isoformat(),
                'total_trades': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'max_profit': 0,
                'daily_metrics': {}
            }
    
    def _save_metrics(self):
        """Salva métricas"""
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def update_trade(self, pnl: float, balance: float):
        """Atualiza métricas com novo trade"""
        self.metrics['total_trades'] += 1
        self.metrics['total_pnl'] += pnl
        
        # Atualizar máximos
        if pnl > self.metrics['max_profit']:
            self.metrics['max_profit'] = pnl
        
        # Calcular drawdown (simplificado)
        if pnl < 0 and abs(pnl) > self.metrics['max_drawdown']:
            self.metrics['max_drawdown'] = abs(pnl)
        
        # Métricas diárias
        today = datetime.now().date().isoformat()
        if today not in self.metrics['daily_metrics']:
            self.metrics['daily_metrics'][today] = {
                'trades': 0,
                'pnl': 0,
                'balance': balance
            }
        
        self.metrics['daily_metrics'][today]['trades'] += 1
        self.metrics['daily_metrics'][today]['pnl'] += pnl
        self.metrics['daily_metrics'][today]['balance'] = balance
        
        self._save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais"""
        return self.metrics.copy()

class DataManager:
    """Gerencia persistência de dados"""
    
    def __init__(self):
        self.data_dir = config.data.data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, format: str = None):
        """Salva DataFrame em arquivo"""
        if format is None:
            format = config.data.data_format
        
        filepath = os.path.join(self.data_dir, filename)
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'parquet':
            df.to_parquet(filepath)
        elif format == 'hdf5':
            df.to_hdf(filepath, key='data', mode='w')
        else:
            raise ValueError(f"Formato não suportado: {format}")
    
    def load_dataframe(self, filename: str, format: str = None) -> pd.DataFrame:
        """Carrega DataFrame de arquivo"""
        if format is None:
            format = config.data.data_format
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return pd.DataFrame()
        
        if format == 'csv':
            return pd.read_csv(filepath)
        elif format == 'parquet':
            return pd.read_parquet(filepath)
        elif format == 'hdf5':
            return pd.read_hdf(filepath, key='data')
        else:
            raise ValueError(f"Formato não suportado: {format}")
    
    def save_json(self, data: Dict[str, Any], filename: str):
        """Salva dados em JSON"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """Carrega dados de JSON"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

class RiskManager:
    """Gerencia riscos e limites"""
    
    def __init__(self):
        self.daily_pnl = 0
        self.daily_trades = 0
        self.current_balance = 0
        self.martingale_step = 0
        self.last_reset = datetime.now().date()
    
    def reset_daily_counters(self):
        """Reseta contadores diários"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0
            self.daily_trades = 0
            self.martingale_step = 0
            self.last_reset = today
    
    def can_trade(self) -> Tuple[bool, str]:
        """Verifica se pode fazer trade"""
        self.reset_daily_counters()
        
        # Verificar limite de trades diários
        if self.daily_trades >= config.trading.max_daily_trades:
            return False, f"Limite diário de trades atingido ({config.trading.max_daily_trades})"
        
        # Verificar stop loss diário
        if self.daily_pnl <= -config.trading.max_daily_loss:
            return False, f"Stop loss diário atingido ({config.trading.max_daily_loss})"
        
        # Verificar limite de martingale
        if config.trading.enable_martingale and self.martingale_step >= config.trading.max_martingale_steps:
            return False, f"Limite de martingale atingido ({config.trading.max_martingale_steps})"
        
        return True, "OK"
    
    def calculate_stake(self, is_martingale: bool = False) -> float:
        """Calcula stake para próximo trade"""
        base_stake = config.trading.initial_stake
        
        if is_martingale and config.trading.enable_martingale:
            stake = base_stake * (config.trading.martingale_multiplier ** self.martingale_step)
            return min(stake, config.trading.max_stake)
        
        return base_stake
    
    def update_trade_result(self, pnl: float, balance: float):
        """Atualiza resultado do trade"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.current_balance = balance
        
        # Resetar martingale se ganhou
        if pnl > 0:
            self.martingale_step = 0
        elif config.trading.enable_martingale:
            self.martingale_step += 1
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Retorna status de risco"""
        self.reset_daily_counters()
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'martingale_step': self.martingale_step,
            'can_trade': self.can_trade()[0],
            'remaining_trades': max(0, config.trading.max_daily_trades - self.daily_trades),
            'remaining_loss_limit': max(0, config.trading.max_daily_loss + self.daily_pnl),
            'current_balance': self.current_balance
        }

# Instâncias globais
trading_logger = TradingLogger()
trade_recorder = TradeRecorder()
performance_tracker = PerformanceTracker()
data_manager = DataManager()
risk_manager = RiskManager()

# Funções de conveniência
def get_logger(name: str = None):
    """Retorna logger configurado"""
    return trading_logger.get_logger(name)

def log_trade(trade_data: Dict[str, Any]):
    """Registra trade"""
    trade_recorder.record_trade(trade_data)
    performance_tracker.update_trade(
        trade_data.get('pnl', 0),
        trade_data.get('balance_after', 0)
    )

def get_daily_report(date: datetime = None) -> Dict[str, Any]:
    """Gera relatório diário"""
    summary = trade_recorder.get_daily_summary(date)
    risk_status = risk_manager.get_risk_status()
    
    return {
        'trading_summary': summary,
        'risk_status': risk_status,
        'performance_metrics': performance_tracker.get_metrics()
    }

def save_market_data(df: pd.DataFrame, symbol: str, timeframe: str):
    """Salva dados de mercado"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"market_data_{symbol}_{timeframe}_{timestamp}.csv"
    data_manager.save_dataframe(df, filename)

def cleanup_old_files(days: int = 30):
    """Remove arquivos antigos"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for directory in [config.data.data_dir, config.data.logs_dir, config.data.reports_dir]:
        if not os.path.exists(directory):
            continue
            
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                if file_time < cutoff_date:
                    try:
                        os.remove(filepath)
                        get_logger().info(f"Arquivo removido: {filepath}")
                    except Exception as e:
                        get_logger().error(f"Erro ao remover {filepath}: {e}")

def format_currency(value: float, currency: str = "USD") -> str:
    """Formata valor monetário"""
    return f"{value:+.2f} {currency}"

def format_percentage(value: float) -> str:
    """Formata porcentagem"""
    return f"{value:+.2%}"

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calcula Sharpe Ratio"""
    if returns.std() == 0:
        return 0
    return (returns.mean() - risk_free_rate / 252) / returns.std() * np.sqrt(252)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calcula máximo drawdown"""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def send_notification(message: str, level: str = "info"):
    """Envia notificação (placeholder para futuras integrações)"""
    logger = get_logger('notifications')
    
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)
    
    # Aqui você pode adicionar integrações com:
    # - Telegram Bot
    # - Discord Webhook
    # - Email
    # - SMS
    # etc.