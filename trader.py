"""
Executor de Trading em Tempo Real
Gerencia execução de trades, contratos e gerenciamento de risco
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from enum import Enum

from data_collector import DerivDataCollector
from ml_model import TradingMLModel
from feature_engineering import FeatureEngineer
from utils import get_logger, log_trade, risk_manager, send_notification
from config import config

class TradeStatus(Enum):
    """Status do trade"""
    PENDING = "pending"
    ACTIVE = "active"
    WON = "won"
    LOST = "lost"
    ERROR = "error"

class SignalType(Enum):
    """Tipo de sinal"""
    CALL = "CALL"
    PUT = "PUT"
    HOLD = "HOLD"

class TradingExecutor:
    """Executor principal de trading"""
    
    def __init__(self):
        self.logger = get_logger('trader')
        self.data_collector = None
        self.ml_model = None
        self.feature_engineer = None
        
        # Estado do trading
        self.is_running = False
        self.current_balance = 0
        self.active_contracts = {}
        self.last_signal_time = None
        
        # Dados em tempo real
        self.current_tick = None
        self.tick_history = []
        self.candle_history = []
        
        # Performance
        self.session_stats = {
            'start_time': None,
            'trades_count': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0
        }
    
    async def initialize(self):
        """Inicializa componentes"""
        try:
            self.logger.info("Inicializando executor de trading...")
            
            # Inicializar coletor de dados
            self.data_collector = DerivDataCollector()
            await self.data_collector.connect()
            
            # Carregar modelo ML
            self.ml_model = TradingMLModel()
            if not self.ml_model.load_model():
                self.logger.warning("Modelo ML não encontrado. Será necessário treinar primeiro.")
                return False
            
            # Inicializar feature engineer
            self.feature_engineer = FeatureEngineer()
            
            # Obter saldo atual
            await self._update_balance()
            
            # Configurar callbacks
            self.data_collector.set_tick_callback(self._on_tick_received)
            self.data_collector.set_candle_callback(self._on_candle_received)
            
            self.logger.info("Executor inicializado com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar executor: {e}")
            return False
    
    async def start_trading(self):
        """Inicia trading automático"""
        if not await self.initialize():
            return False
        
        self.is_running = True
        self.session_stats['start_time'] = datetime.now()
        
        self.logger.info("Iniciando trading automático...")
        send_notification("🚀 Trading automático iniciado", "info")
        
        try:
            # Subscrever a ticks
            await self.data_collector.subscribe_ticks(config.trading.symbol)
            
            # Loop principal
            while self.is_running:
                await self._trading_loop()
                await asyncio.sleep(1)  # Evitar uso excessivo de CPU
                
        except Exception as e:
            self.logger.error(f"Erro no loop de trading: {e}")
            send_notification(f"❌ Erro no trading: {e}", "error")
        finally:
            await self.stop_trading()
    
    async def stop_trading(self):
        """Para trading automático"""
        self.is_running = False
        
        # Aguardar contratos ativos
        if self.active_contracts:
            self.logger.info("Aguardando contratos ativos finalizarem...")
            await self._wait_for_active_contracts()
        
        # Desconectar
        if self.data_collector:
            await self.data_collector.disconnect()
        
        # Relatório final
        self._generate_session_report()
        
        self.logger.info("Trading parado")
        send_notification("⏹️ Trading automático parado", "info")
    
    async def _trading_loop(self):
        """Loop principal de trading"""
        try:
            # Verificar se pode fazer trade
            can_trade, reason = risk_manager.can_trade()
            if not can_trade:
                if reason != "OK":
                    self.logger.warning(f"Trading bloqueado: {reason}")
                return
            
            # Verificar se há dados suficientes
            if len(self.tick_history) < config.ml.min_training_samples:
                return
            
            # Verificar intervalo mínimo entre trades
            if self._should_wait_for_next_signal():
                return
            
            # Gerar sinal
            signal = await self._generate_signal()
            
            if signal['type'] != SignalType.HOLD:
                await self._execute_trade(signal)
                
        except Exception as e:
            self.logger.error(f"Erro no loop de trading: {e}")
    
    async def _generate_signal(self) -> Dict[str, Any]:
        """Gera sinal de trading usando ML"""
        try:
            # Preparar dados
            df = pd.DataFrame(self.tick_history[-config.ml.min_training_samples:])
            
            # Gerar features
            features_df = self.feature_engineer.create_features(df)
            
            if features_df.empty:
                return {'type': SignalType.HOLD, 'confidence': 0}
            
            # Fazer previsão
            prediction = self.ml_model.predict(features_df.iloc[-1:])
            
            if prediction is None:
                return {'type': SignalType.HOLD, 'confidence': 0}
            
            signal_type, confidence = prediction
            
            # Verificar confiança mínima
            if confidence < config.trading.min_prediction_confidence:
                return {'type': SignalType.HOLD, 'confidence': confidence}
            
            self.logger.info(f"Sinal gerado: {signal_type} (confiança: {confidence:.3f})")
            
            return {
                'type': SignalType(signal_type),
                'confidence': confidence,
                'features': features_df.iloc[-1].to_dict(),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal: {e}")
            return {'type': SignalType.HOLD, 'confidence': 0}
    
    async def _execute_trade(self, signal: Dict[str, Any]):
        """Executa trade baseado no sinal"""
        try:
            # Calcular stake
            is_martingale = risk_manager.martingale_step > 0
            stake = risk_manager.calculate_stake(is_martingale)
            
            # Preparar parâmetros do contrato
            contract_params = {
                'contract_type': signal['type'].value,
                'symbol': config.trading.symbol,
                'amount': stake,
                'duration': config.trading.contract_duration,
                'duration_unit': config.trading.duration_unit,
                'basis': 'stake'
            }
            
            self.logger.info(f"Executando trade: {contract_params}")
            
            # Comprar contrato
            contract_result = await self.data_collector.buy_contract(contract_params)
            
            if contract_result and 'buy' in contract_result:
                contract_id = contract_result['buy']['contract_id']
                
                # Registrar contrato ativo
                self.active_contracts[contract_id] = {
                    'signal': signal,
                    'params': contract_params,
                    'start_time': datetime.now(),
                    'buy_price': contract_result['buy']['buy_price'],
                    'longcode': contract_result['buy']['longcode']
                }
                
                # Atualizar estatísticas
                self.session_stats['trades_count'] += 1
                self.last_signal_time = datetime.now()
                
                # Monitorar contrato
                asyncio.create_task(self._monitor_contract(contract_id))
                
                self.logger.info(f"Trade executado: ID {contract_id}")
                send_notification(f"📈 Trade executado: {signal['type'].value} - Stake: ${stake}", "info")
                
            else:
                self.logger.error(f"Falha ao executar trade: {contract_result}")
                
        except Exception as e:
            self.logger.error(f"Erro ao executar trade: {e}")
            send_notification(f"❌ Erro ao executar trade: {e}", "error")
    
    async def _monitor_contract(self, contract_id: str):
        """Monitora contrato ativo"""
        try:
            contract_info = self.active_contracts[contract_id]
            
            # Aguardar resultado
            while contract_id in self.active_contracts:
                # Verificar status do contrato
                status_result = await self.data_collector.get_contract_status(contract_id)
                
                if status_result and 'proposal_open_contract' in status_result:
                    contract_data = status_result['proposal_open_contract']
                    
                    if contract_data.get('is_settled'):
                        await self._handle_contract_result(contract_id, contract_data)
                        break
                
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Erro ao monitorar contrato {contract_id}: {e}")
            if contract_id in self.active_contracts:
                del self.active_contracts[contract_id]
    
    async def _handle_contract_result(self, contract_id: str, contract_data: Dict[str, Any]):
        """Processa resultado do contrato"""
        try:
            if contract_id not in self.active_contracts:
                return
            
            contract_info = self.active_contracts[contract_id]
            
            # Calcular resultado
            buy_price = contract_info['buy_price']
            sell_price = contract_data.get('sell_price', 0)
            pnl = sell_price - buy_price
            
            # Determinar status
            is_win = pnl > 0
            status = TradeStatus.WON if is_win else TradeStatus.LOST
            
            # Atualizar estatísticas
            if is_win:
                self.session_stats['wins'] += 1
            else:
                self.session_stats['losses'] += 1
            
            self.session_stats['total_pnl'] += pnl
            
            # Atualizar saldo
            await self._update_balance()
            
            # Atualizar gerenciador de risco
            risk_manager.update_trade_result(pnl, self.current_balance)
            
            # Registrar trade
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': contract_info['params']['symbol'],
                'signal': contract_info['signal']['type'].value,
                'stake': contract_info['params']['amount'],
                'contract_type': contract_info['params']['contract_type'],
                'entry_price': contract_data.get('entry_tick', 0),
                'exit_price': contract_data.get('exit_tick', 0),
                'duration': contract_info['params']['duration'],
                'pnl': pnl,
                'pnl_percentage': (pnl / buy_price) * 100 if buy_price > 0 else 0,
                'confidence': contract_info['signal']['confidence'],
                'features_used': json.dumps(contract_info['signal'].get('features', {})),
                'martingale_step': risk_manager.martingale_step,
                'balance_before': self.current_balance - pnl,
                'balance_after': self.current_balance,
                'status': status.value
            }
            
            log_trade(trade_data)
            
            # Notificação
            result_emoji = "✅" if is_win else "❌"
            send_notification(
                f"{result_emoji} Trade finalizado: {status.value.upper()} - PnL: ${pnl:.2f}",
                "info"
            )
            
            self.logger.info(f"Contrato {contract_id} finalizado: {status.value} - PnL: ${pnl:.2f}")
            
            # Remover contrato ativo
            del self.active_contracts[contract_id]
            
        except Exception as e:
            self.logger.error(f"Erro ao processar resultado do contrato {contract_id}: {e}")
    
    async def _update_balance(self):
        """Atualiza saldo atual"""
        try:
            balance_result = await self.data_collector.get_balance()
            if balance_result and 'balance' in balance_result:
                self.current_balance = balance_result['balance']['balance']
                self.logger.debug(f"Saldo atualizado: ${self.current_balance:.2f}")
        except Exception as e:
            self.logger.error(f"Erro ao atualizar saldo: {e}")
    
    def _should_wait_for_next_signal(self) -> bool:
        """Verifica se deve aguardar antes do próximo sinal"""
        if self.last_signal_time is None:
            return False
        
        time_since_last = datetime.now() - self.last_signal_time
        min_interval = timedelta(seconds=config.trading.min_trade_interval)
        
        return time_since_last < min_interval
    
    async def _wait_for_active_contracts(self, timeout: int = 300):
        """Aguarda contratos ativos finalizarem"""
        start_time = time.time()
        
        while self.active_contracts and (time.time() - start_time) < timeout:
            self.logger.info(f"Aguardando {len(self.active_contracts)} contratos ativos...")
            await asyncio.sleep(5)
        
        if self.active_contracts:
            self.logger.warning(f"Timeout: {len(self.active_contracts)} contratos ainda ativos")
    
    def _generate_session_report(self):
        """Gera relatório da sessão"""
        if self.session_stats['start_time'] is None:
            return
        
        duration = datetime.now() - self.session_stats['start_time']
        win_rate = (self.session_stats['wins'] / self.session_stats['trades_count']) * 100 if self.session_stats['trades_count'] > 0 else 0
        
        report = f"""
        📊 RELATÓRIO DA SESSÃO
        =====================
        Duração: {duration}
        Total de Trades: {self.session_stats['trades_count']}
        Vitórias: {self.session_stats['wins']}
        Derrotas: {self.session_stats['losses']}
        Taxa de Acerto: {win_rate:.1f}%
        PnL Total: ${self.session_stats['total_pnl']:.2f}
        Saldo Final: ${self.current_balance:.2f}
        """
        
        self.logger.info(report)
        send_notification(report, "info")
    
    async def _on_tick_received(self, tick_data: Dict[str, Any]):
        """Callback para tick recebido"""
        self.current_tick = tick_data
        self.tick_history.append(tick_data)
        
        # Manter apenas os últimos N ticks
        max_history = config.ml.max_history_size
        if len(self.tick_history) > max_history:
            self.tick_history = self.tick_history[-max_history:]
    
    async def _on_candle_received(self, candle_data: Dict[str, Any]):
        """Callback para candle recebido"""
        self.candle_history.append(candle_data)
        
        # Manter apenas os últimos N candles
        max_history = config.ml.max_history_size
        if len(self.candle_history) > max_history:
            self.candle_history = self.candle_history[-max_history:]
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do trader"""
        return {
            'is_running': self.is_running,
            'current_balance': self.current_balance,
            'active_contracts': len(self.active_contracts),
            'session_stats': self.session_stats.copy(),
            'risk_status': risk_manager.get_risk_status(),
            'last_tick': self.current_tick,
            'data_points': len(self.tick_history)
        }

# Funções de conveniência
async def start_automated_trading():
    """Inicia trading automatizado"""
    trader = TradingExecutor()
    await trader.start_trading()
    return trader

async def execute_single_trade(signal_type: str, stake: float = None) -> Dict[str, Any]:
    """Executa um trade único"""
    trader = TradingExecutor()
    
    if not await trader.initialize():
        return {'success': False, 'error': 'Falha na inicialização'}
    
    try:
        # Preparar sinal manual
        signal = {
            'type': SignalType(signal_type),
            'confidence': 1.0,
            'features': {},
            'timestamp': datetime.now()
        }
        
        # Executar trade
        await trader._execute_trade(signal)
        
        return {'success': True, 'message': 'Trade executado com sucesso'}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        await trader.data_collector.disconnect()

def get_trading_status() -> Dict[str, Any]:
    """Retorna status do trading (para uso em dashboard)"""
    return {
        'risk_status': risk_manager.get_risk_status(),
        'current_time': datetime.now().isoformat()
    }