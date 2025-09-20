"""
Configurações do Sistema de Trading Automatizado Deriv
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

@dataclass
class DerivConfig:
    """Configurações da API Deriv"""
    # API Credentials - Configuradas diretamente
    app_id: str = "101918"  # App ID configurado
    api_token: str = "cuCpkc00HgKXvym"  # Token configurado
    
    # WebSocket URLs
    websocket_url: str = "wss://ws.binaryws.com/websockets/v3"
    
    # Trading parameters
    default_symbol: str = "R_10"  # Volatility 10 Index
    default_duration: int = 5  # 5 ticks
    default_duration_unit: str = "t"  # ticks
    
    # Account type
    account_type: str = "demo"  # "demo" or "real"

@dataclass
class TradingConfig:
    """Configurações de Trading"""
    # Risk Management
    initial_stake: float = 1.0  # Stake inicial em USD
    max_stake: float = 100.0  # Stake máximo
    stop_loss_percentage: float = 0.05  # 5% stop loss
    stop_gain_percentage: float = 0.10  # 10% stop gain
    max_daily_loss: float = 50.0  # Perda máxima diária
    max_daily_trades: int = 100  # Máximo de trades por dia
    
    # Martingale
    enable_martingale: bool = False
    martingale_multiplier: float = 2.0
    max_martingale_steps: int = 3
    
    # Model thresholds
    min_prediction_confidence: float = 0.6  # Confiança mínima para trade
    
    # Symbols to trade
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]

@dataclass
class MLConfig:
    """Configurações do Modelo de Machine Learning"""
    # Model parameters
    model_type: str = "lightgbm"  # "lightgbm", "lstm", "ensemble"
    
    # LightGBM parameters
    lgb_params: Dict = None
    
    # Feature engineering
    lookback_periods: List[int] = None
    technical_indicators: List[str] = None
    
    # Training
    train_test_split: float = 0.8
    validation_split: float = 0.2
    min_training_samples: int = 1000
    retrain_frequency: str = "daily"  # "hourly", "daily", "weekly"
    
    # Data
    max_data_points: int = 10000  # Máximo de pontos para treinamento
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50]
            
        if self.technical_indicators is None:
            self.technical_indicators = [
                'rsi', 'macd', 'bollinger', 'sma', 'ema', 
                'stochastic', 'williams_r', 'atr', 'adx'
            ]

@dataclass
class DataConfig:
    """Configurações de Dados"""
    # Data storage
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    reports_dir: str = "reports"
    
    # Data collection
    tick_buffer_size: int = 1000
    candle_timeframes: List[str] = None
    
    # File formats
    data_format: str = "csv"  # "csv", "parquet", "hdf5"
    
    def __post_init__(self):
        if self.candle_timeframes is None:
            self.candle_timeframes = ["1m", "5m", "15m", "1h"]
        
        # Create directories
        for directory in [self.data_dir, self.models_dir, self.logs_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)

@dataclass
class LoggingConfig:
    """Configurações de Logging"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/bot.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Console logging
    console_logging: bool = True
    colored_logs: bool = True

@dataclass
class DashboardConfig:
    """Configurações do Dashboard"""
    host: str = os.getenv('HOST', '0.0.0.0')  # Railway requires 0.0.0.0
    port: int = int(os.getenv('PORT', '8501'))  # Railway sets PORT automatically
    title: str = "Deriv AI Trading Bot"
    refresh_interval: int = 5  # seconds
    
    # Chart settings
    max_chart_points: int = 500
    default_timeframe: str = "1h"

# Main configuration class
@dataclass
class Config:
    """Configuração Principal do Sistema"""
    deriv: DerivConfig = None
    trading: TradingConfig = None
    ml: MLConfig = None
    data: DataConfig = None
    logging: LoggingConfig = None
    dashboard: DashboardConfig = None
    
    # Environment - Auto-detect Railway production environment
    environment: str = os.getenv('RAILWAY_ENVIRONMENT', 'development')  # Railway sets this
    debug: bool = os.getenv('DEBUG', 'True').lower() in ('true', '1', 'yes')
    
    def __post_init__(self):
        if self.deriv is None:
            self.deriv = DerivConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.dashboard is None:
            self.dashboard = DashboardConfig()

# Global configuration instance
config = Config()

# Validation functions
def validate_config():
    """Valida as configurações"""
    errors = []
    
    # Check API token
    if not config.deriv.api_token:
        errors.append("DERIV_API_TOKEN não configurado. Configure via variável de ambiente.")
    
    # Check trading parameters
    if config.trading.initial_stake <= 0:
        errors.append("initial_stake deve ser maior que 0")
    
    if config.trading.min_prediction_confidence < 0.5 or config.trading.min_prediction_confidence > 1.0:
        errors.append("min_prediction_confidence deve estar entre 0.5 e 1.0")
    
    # Check ML parameters
    if config.ml.min_training_samples < 100:
        errors.append("min_training_samples deve ser pelo menos 100")
    
    if errors:
        raise ValueError(f"Erros de configuração: {'; '.join(errors)}")
    
    return True

def load_config_from_env():
    """Carrega configurações de variáveis de ambiente"""
    # Deriv API
    if os.getenv('DERIV_API_TOKEN'):
        config.deriv.api_token = os.getenv('DERIV_API_TOKEN')
    
    if os.getenv('DERIV_APP_ID'):
        config.deriv.app_id = os.getenv('DERIV_APP_ID')
    
    # Trading
    if os.getenv('INITIAL_STAKE'):
        config.trading.initial_stake = float(os.getenv('INITIAL_STAKE'))
    
    if os.getenv('MAX_DAILY_LOSS'):
        config.trading.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS'))
    
    # Environment
    if os.getenv('ENVIRONMENT'):
        config.environment = os.getenv('ENVIRONMENT')
        config.debug = config.environment == "development"

# Load environment variables on import
load_config_from_env()