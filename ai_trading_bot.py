"""
Bot de Trading Automatizado com Inteligência Artificial para Deriv
Integra coleta de dados, análise técnica, ML e execução de trades
"""
import asyncio
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple, Any
import pandas as pd
import numpy as np
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from data_collector import data_collector
from ml_model import TradingMLModel, create_and_train_model
from feature_engineering import FeatureEngineer
from auth_manager import auth_manager
from config import config
from utils import get_logger, log_trade, risk_manager

class BotStatus(Enum):
    """Status do bot"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class SignalStrength(Enum):
    """Força do sinal"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class AITradingBot:
    """Bot de Trading com IA para Deriv"""
    
    def __init__(self, config_override: Dict = None):
        self.logger = get_logger('ai_trading_bot')
        
        # Configurações
        import copy
        self.config = copy.deepcopy(config)
        if config_override:
            # Aplicar overrides aos atributos do config
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Componentes principais
        self.data_collector = data_collector
        self.ml_model = None
        self.feature_engineer = FeatureEngineer()
        
        # Estado do bot
        self.status = BotStatus.STOPPED
        self.is_running = False
        self.start_time = None
        self.last_prediction_time = None
        
        # Dados em tempo real
        self.current_data = pd.DataFrame()
        self.tick_buffer = []
        self.prediction_buffer = []
        
        # Estatísticas de trading
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "current_balance": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_profit_per_trade": 0.0,
            "predictions_made": 0,
            "correct_predictions": 0,
            "prediction_accuracy": 0.0
        }
        
        # Controle de risco
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = None
        
        # Configurações de trading
        self.symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]  # Synthetic Indices
        self.timeframes = ["1m", "5m"]
        self.min_confidence = 0.65  # Confiança mínima para trade
        self.stake_amount = 1.0  # Valor base do stake
        self.max_daily_trades = 50
        self.max_daily_loss = 100.0
        self.max_consecutive_losses = 5
        
    async def initialize(self) -> bool:
        """Inicializa o bot e todos os componentes"""
        try:
            self.logger.info("🤖 Inicializando AI Trading Bot...")
            self.status = BotStatus.STARTING
            
            # 1. Verificar autenticação
            if not auth_manager.is_authenticated():
                self.logger.error("❌ Bot não autenticado. Faça login primeiro.")
                self.status = BotStatus.ERROR
                return False
            
            # 2. Conectar data collector
            if not self.data_collector.is_connected:
                if not self.data_collector.connect():
                    self.logger.error("❌ Falha ao conectar data collector")
                    self.status = BotStatus.ERROR
                    return False
            
            # 3. Carregar ou treinar modelo
            await self._initialize_ml_model()
            
            # 4. Configurar callbacks de dados
            self._setup_data_callbacks()
            
            # 5. Iniciar coleta de dados
            await self._start_data_collection()
            
            # 6. Atualizar saldo inicial
            await self._update_balance()
            
            self.logger.info("✅ AI Trading Bot inicializado com sucesso!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização: {e}")
            self.status = BotStatus.ERROR
            return False
    
    async def _initialize_ml_model(self):
        """Inicializa ou carrega modelo de ML"""
        try:
            # Tentar carregar modelo existente
            model_path = "models/latest_trading_model.joblib"
            try:
                self.ml_model = TradingMLModel()
                self.ml_model.load_model(model_path)
                self.logger.info("📊 Modelo ML carregado com sucesso")
            except:
                self.logger.info("🔄 Treinando novo modelo ML...")
                
                # Coletar dados históricos para treinamento
                historical_data = await self._collect_training_data()
                
                if len(historical_data) < 1000:
                    self.logger.warning("⚠️ Poucos dados para treinamento. Usando modelo padrão.")
                    self.ml_model = TradingMLModel()
                else:
                    # Treinar novo modelo
                    self.ml_model = create_and_train_model(historical_data)
                    self.ml_model.save_model(model_path)
                    self.logger.info("✅ Novo modelo treinado e salvo")
                    
        except Exception as e:
            self.logger.error(f"❌ Erro ao inicializar modelo ML: {e}")
            # Usar modelo padrão em caso de erro
            self.ml_model = TradingMLModel()
    
    async def _collect_training_data(self) -> pd.DataFrame:
        """Coleta dados históricos para treinamento"""
        all_data = []
        
        for symbol in self.symbols[:2]:  # Usar apenas 2 símbolos para treinamento inicial
            try:
                # Coletar 5000 pontos históricos
                historical = self.data_collector.get_historical_data(symbol, "1m", 5000)
                if not historical.empty:
                    historical['symbol'] = symbol
                    all_data.append(historical)
                    self.logger.info(f"📈 Coletados {len(historical)} pontos para {symbol}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao coletar dados de {symbol}: {e}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"📊 Total de dados coletados: {len(combined_data)} pontos")
            return combined_data
        
        return pd.DataFrame()
    
    def _setup_data_callbacks(self):
        """Configura callbacks para dados em tempo real"""
        def on_tick_received(tick_data):
            """Callback para novos ticks"""
            self.tick_buffer.append(tick_data)
            # Manter apenas os últimos 1000 ticks
            if len(self.tick_buffer) > 1000:
                self.tick_buffer = self.tick_buffer[-1000:]
        
        def on_candle_received(candle_data):
            """Callback para novas velas"""
            asyncio.create_task(self._process_new_candle(candle_data))
        
        # Registrar callbacks
        self.data_collector.callbacks['tick'] = on_tick_received
        self.data_collector.callbacks['candle'] = on_candle_received
    
    async def _start_data_collection(self):
        """Inicia coleta de dados para os símbolos configurados"""
        for symbol in self.symbols:
            # Subscrever ticks
            self.data_collector.subscribe_ticks(symbol)
            
            # Subscrever candles
            for timeframe in self.timeframes:
                self.data_collector.subscribe_candles(symbol, timeframe)
        
        self.logger.info(f"📡 Coleta iniciada para {len(self.symbols)} símbolos")
    
    async def start_trading(self):
        """Inicia o trading automatizado"""
        if self.status != BotStatus.STOPPED:
            self.logger.warning("⚠️ Bot já está rodando ou em erro")
            return False
        
        if not await self.initialize():
            return False
        
        self.is_running = True
        self.status = BotStatus.RUNNING
        self.start_time = datetime.now()
        
        self.logger.info("🚀 Trading automatizado iniciado!")
        
        # Iniciar loop principal
        asyncio.create_task(self._main_trading_loop())
        return True
    
    async def stop_trading(self):
        """Para o trading automatizado"""
        self.is_running = False
        self.status = BotStatus.STOPPED
        
        # Aguardar trades ativos terminarem
        # TODO: Implementar monitoramento de contratos ativos
        
        self.logger.info("🛑 Trading automatizado parado")
        self._generate_session_report()
    
    async def _main_trading_loop(self):
        """Loop principal de trading"""
        while self.is_running:
            try:
                # Verificar condições de parada
                if await self._should_stop_trading():
                    await self.stop_trading()
                    break
                
                # Gerar predição se tiver dados suficientes
                if await self._should_make_prediction():
                    prediction = await self._generate_prediction()
                    
                    if prediction and prediction['confidence'] >= self.min_confidence:
                        await self._execute_trade_from_prediction(prediction)
                
                # Aguardar próximo ciclo
                await asyncio.sleep(5)  # Verificar a cada 5 segundos
                
            except Exception as e:
                self.logger.error(f"❌ Erro no loop principal: {e}")
                await asyncio.sleep(10)  # Aguardar mais tempo em caso de erro
    
    async def _process_new_candle(self, candle_data: Dict):
        """Processa nova vela recebida"""
        try:
            # Atualizar dados atuais
            # TODO: Implementar atualização do DataFrame atual
            pass
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao processar vela: {e}")
    
    async def _should_make_prediction(self) -> bool:
        """Verifica se deve fazer uma nova predição"""
        # Verificar se passou tempo suficiente desde a última predição
        if self.last_prediction_time:
            time_since_last = datetime.now() - self.last_prediction_time
            if time_since_last.total_seconds() < 30:  # Mínimo 30 segundos
                return False
        
        # Verificar se tem dados suficientes
        if len(self.tick_buffer) < 100:
            return False
        
        # Verificar limites diários
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        if self.daily_loss >= self.max_daily_loss:
            return False
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        
        return True
    
    async def _generate_prediction(self) -> Optional[Dict]:
        """Gera predição usando o modelo de ML"""
        try:
            if not self.ml_model or not self.ml_model.is_trained:
                return None
            
            # Preparar dados recentes para predição
            recent_data = await self._prepare_recent_data()
            if recent_data.empty:
                return None
            
            # Gerar predição
            signal = self.ml_model.get_trading_signal(recent_data, self.min_confidence)
            
            if signal['action'] != 'HOLD':
                self.stats['predictions_made'] += 1
                self.last_prediction_time = datetime.now()
                
                prediction = {
                    'signal': signal['action'],
                    'confidence': signal['confidence'],
                    'symbol': signal.get('symbol', 'R_10'),
                    'timestamp': datetime.now(),
                    'features_used': signal.get('features_used', []),
                    'model_version': self.ml_model.get_model_info().get('version', '1.0')
                }
                
                self.prediction_buffer.append(prediction)
                self.logger.info(f"🎯 Predição: {prediction['signal']} (confiança: {prediction['confidence']:.2%})")
                
                return prediction
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar predição: {e}")
            return None
    
    async def _prepare_recent_data(self) -> pd.DataFrame:
        """Prepara dados recentes para predição"""
        try:
            # Converter ticks recentes em DataFrame
            if len(self.tick_buffer) < 50:
                return pd.DataFrame()
            
            recent_ticks = self.tick_buffer[-100:]  # Últimos 100 ticks
            
            df_data = []
            for tick in recent_ticks:
                df_data.append({
                    'timestamp': tick.get('timestamp'),
                    'datetime': tick.get('datetime'),
                    'close': tick.get('price'),
                    'symbol': tick.get('symbol', 'R_10')
                })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao preparar dados: {e}")
            return pd.DataFrame()
    
    async def _execute_trade_from_prediction(self, prediction: Dict):
        """Executa trade baseado na predição"""
        try:
            # Calcular stake baseado no gerenciamento de risco
            stake = self._calculate_stake(prediction['confidence'])
            
            # TODO: Implementar execução real do trade via API Deriv
            # Por enquanto, apenas simular
            
            trade_data = {
                'signal': prediction['signal'],
                'confidence': prediction['confidence'],
                'stake': stake,
                'symbol': prediction['symbol'],
                'timestamp': datetime.now(),
                'status': 'executed'
            }
            
            self.daily_trades += 1
            self.stats['total_trades'] += 1
            
            self.logger.info(f"💰 Trade executado: {trade_data['signal']} ${stake} em {trade_data['symbol']}")
            
            # Log do trade
            log_trade(trade_data)
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao executar trade: {e}")
    
    def _calculate_stake(self, confidence: float) -> float:
        """Calcula valor do stake baseado na confiança"""
        base_stake = self.stake_amount
        
        # Ajustar stake baseado na confiança
        if confidence >= 0.8:
            multiplier = 1.5  # Aumentar stake para alta confiança
        elif confidence >= 0.7:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        # Aplicar gerenciamento de risco
        max_stake = self.stats['current_balance'] * 0.02  # Máximo 2% do saldo
        calculated_stake = min(base_stake * multiplier, max_stake)
        
        return round(calculated_stake, 2)
    
    async def _should_stop_trading(self) -> bool:
        """Verifica se deve parar o trading"""
        # Verificar limites diários
        if self.daily_loss >= self.max_daily_loss:
            self.logger.warning(f"🛑 Limite de perda diária atingido: ${self.daily_loss}")
            return True
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(f"🛑 Muitas perdas consecutivas: {self.consecutive_losses}")
            return True
        
        # Verificar horário de trading (opcional)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Parar entre 22h e 6h
            return True
        
        return False
    
    async def _update_balance(self):
        """Atualiza saldo atual"""
        try:
            balance_data = await self.data_collector.get_balance()
            if balance_data and 'balance' in balance_data:
                self.stats['current_balance'] = balance_data['balance']['balance']
                self.logger.debug(f"💰 Saldo atualizado: ${self.stats['current_balance']}")
        except Exception as e:
            self.logger.error(f"❌ Erro ao atualizar saldo: {e}")
    
    def _generate_session_report(self):
        """Gera relatório da sessão de trading"""
        if self.start_time:
            session_duration = datetime.now() - self.start_time
            
            # Calcular estatísticas
            if self.stats['total_trades'] > 0:
                self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
                self.stats['avg_profit_per_trade'] = self.stats['total_profit'] / self.stats['total_trades']
            
            if self.stats['predictions_made'] > 0:
                self.stats['prediction_accuracy'] = (self.stats['correct_predictions'] / self.stats['predictions_made']) * 100
            
            report = f"""
📊 RELATÓRIO DA SESSÃO DE TRADING
{'='*50}
⏱️ Duração: {session_duration}
💰 Saldo Final: ${self.stats['current_balance']:.2f}
📈 Total de Trades: {self.stats['total_trades']}
✅ Trades Vencedores: {self.stats['winning_trades']}
❌ Trades Perdedores: {self.stats['losing_trades']}
📊 Taxa de Acerto: {self.stats['win_rate']:.1f}%
💵 Lucro Total: ${self.stats['total_profit']:.2f}
📊 Lucro Médio por Trade: ${self.stats['avg_profit_per_trade']:.2f}
🎯 Predições Feitas: {self.stats['predictions_made']}
✅ Predições Corretas: {self.stats['correct_predictions']}
🎯 Precisão do Modelo: {self.stats['prediction_accuracy']:.1f}%
{'='*50}
            """
            
            self.logger.info(report)
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do bot"""
        return {
            'status': self.status.value,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stats': self.stats.copy(),
            'daily_trades': self.daily_trades,
            'daily_loss': self.daily_loss,
            'consecutive_losses': self.consecutive_losses,
            'symbols': self.symbols,
            'min_confidence': self.min_confidence,
            'model_info': self.ml_model.get_model_info() if self.ml_model else None
        }

# Instância global do bot
ai_bot = AITradingBot()

# Funções de conveniência
async def start_ai_trading(config_override: Dict = None) -> bool:
    """Inicia o bot de trading com IA"""
    if config_override:
        ai_bot.config.update(config_override)
    
    return await ai_bot.start_trading()

async def stop_ai_trading():
    """Para o bot de trading com IA"""
    await ai_bot.stop_trading()

def get_ai_bot_status() -> Dict[str, Any]:
    """Retorna status do bot de IA"""
    return ai_bot.get_status()