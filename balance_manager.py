"""
Gerenciador de Saldo em Tempo Real
Mantém o saldo atualizado automaticamente
"""
import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from data_collector import data_collector
from config import config

logger = logging.getLogger(__name__)

# Import auth_manager para verificar status de autenticação
try:
    from auth_manager import auth_manager
    AUTH_MANAGER_AVAILABLE = True
except ImportError:
    AUTH_MANAGER_AVAILABLE = False

class BalanceManager:
    """Gerencia saldo em tempo real"""
    
    def __init__(self):
        self.current_balance = 0.0
        self.last_update = None
        self.is_connected = False
        self.is_updating = False
        self.update_interval = 30  # Atualizar a cada 30 segundos
        self.error_message = None
        self.connection_status = "disconnected"
        
        # Thread para atualização automática
        self._update_thread = None
        self._stop_event = threading.Event()
        
    def start_auto_update(self):
        """Inicia atualização automática do saldo"""
        if self._update_thread and self._update_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Atualização automática de saldo iniciada")
    
    def stop_auto_update(self):
        """Para atualização automática do saldo"""
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("Atualização automática de saldo parada")
    
    def _auto_update_loop(self):
        """Loop de atualização automática"""
        while not self._stop_event.is_set():
            try:
                # Executar atualização assíncrona
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.update_balance())
                loop.close()
                
                # Aguardar próxima atualização
                self._stop_event.wait(self.update_interval)
                
            except Exception as e:
                logger.error(f"Erro na atualização automática: {e}")
                self.error_message = str(e)
                self._stop_event.wait(10)  # Aguardar menos tempo em caso de erro
    
    async def update_balance(self) -> bool:
        """Atualiza saldo atual"""
        if self.is_updating:
            return False
            
        self.is_updating = True
        
        try:
            # Verificar se está conectado
            if not data_collector.is_connected:
                self.connection_status = "connecting"
                success = data_collector.connect()
                if not success:
                    self.connection_status = "failed"
                    self.error_message = "Falha na conexão"
                    return False
                
                # Aguardar conexão
                await asyncio.sleep(2)
            
            # Verificar autorização
            if not data_collector.is_authorized:
                self.connection_status = "authorizing"
                
                # Verificar se há token disponível (auth_manager ou config)
                has_token = False
                if AUTH_MANAGER_AVAILABLE:
                    has_token = auth_manager.get_api_token() is not None
                if not has_token:
                    has_token = config.deriv.api_token is not None
                
                if not has_token:
                    self.connection_status = "no_token"
                    self.error_message = "Token da API não configurado. Use a aba Login para autenticar."
                    return False
                
                authorized = data_collector.authorize()
                if not authorized:
                    self.connection_status = "auth_failed"
                    self.error_message = "Falha na autorização"
                    return False
                
                # Aguardar autorização
                await asyncio.sleep(2)
            
            # Buscar saldo
            self.connection_status = "fetching_balance"
            balance_result = await data_collector.get_balance()
            
            if balance_result and 'error' not in balance_result:
                if 'balance' in balance_result:
                    # Extrair saldo da resposta
                    balance_data = balance_result['balance']
                    if isinstance(balance_data, dict):
                        self.current_balance = float(balance_data.get('balance', 0))
                    else:
                        self.current_balance = float(balance_data)
                    
                    self.last_update = datetime.now()
                    self.connection_status = "connected"
                    self.error_message = None
                    self.is_connected = True
                    
                    logger.debug(f"Saldo atualizado: ${self.current_balance:.2f}")
                    return True
                else:
                    self.error_message = "Formato de resposta inválido"
                    self.connection_status = "error"
                    return False
            else:
                # Tratar erro da API
                if balance_result and 'error' in balance_result:
                    error_msg = balance_result['error'].get('message', 'Erro desconhecido')
                    self.error_message = error_msg
                    
                    if 'oauth token' in error_msg.lower() or 'permission' in error_msg.lower():
                        self.connection_status = "invalid_token"
                    else:
                        self.connection_status = "api_error"
                else:
                    self.error_message = "Timeout ao obter saldo"
                    self.connection_status = "timeout"
                
                self.is_connected = False
                return False
                
        except Exception as e:
            logger.error(f"Erro ao atualizar saldo: {e}")
            self.error_message = str(e)
            self.connection_status = "error"
            self.is_connected = False
            return False
        finally:
            self.is_updating = False
    
    def get_balance_info(self) -> Dict[str, Any]:
        """Retorna informações completas do saldo"""
        return {
            'balance': self.current_balance,
            'last_update': self.last_update,
            'is_connected': self.is_connected,
            'is_updating': self.is_updating,
            'connection_status': self.connection_status,
            'error_message': self.error_message,
            'update_interval': self.update_interval
        }
    
    def get_status_emoji(self) -> str:
        """Retorna emoji baseado no status"""
        status_emojis = {
            'connected': '🟢',
            'connecting': '🟡',
            'authorizing': '🟡',
            'fetching_balance': '🟡',
            'disconnected': '🔴',
            'failed': '🔴',
            'auth_failed': '🔴',
            'invalid_token': '🔴',
            'api_error': '🔴',
            'timeout': '🟠',
            'error': '🔴',
            'no_token': '🔴'
        }
        return status_emojis.get(self.connection_status, '⚪')
    
    def get_status_text(self) -> str:
        """Retorna texto do status"""
        status_texts = {
            'connected': 'Conectado',
            'connecting': 'Conectando...',
            'authorizing': 'Autorizando...',
            'fetching_balance': 'Obtendo saldo...',
            'disconnected': 'Desconectado',
            'failed': 'Falha na conexão',
            'auth_failed': 'Falha na autorização',
            'invalid_token': 'Token inválido',
            'api_error': 'Erro da API',
            'timeout': 'Timeout',
            'error': 'Erro',
            'no_token': 'Token não configurado'
        }
        return status_texts.get(self.connection_status, 'Status desconhecido')
    
    def force_update(self):
        """Força atualização imediata do saldo"""
        if not self.is_updating:
            # Executar em thread separada para não bloquear
            thread = threading.Thread(target=self._force_update_sync, daemon=True)
            thread.start()
    
    def _force_update_sync(self):
        """Executa atualização forçada de forma síncrona"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.update_balance())
            loop.close()
        except Exception as e:
            logger.error(f"Erro na atualização forçada: {e}")

# Instância global
balance_manager = BalanceManager()