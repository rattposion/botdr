"""
Gerenciador de Tokens Automático
Renova tokens automaticamente antes do vencimento
"""
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TokenManager:
    """Gerencia renovação automática de tokens"""
    
    def __init__(self):
        self.auth_manager = None
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.renewal_margin = 300  # Renovar 5 minutos antes do vencimento
        
        # Tentar importar auth_manager
        try:
            from auth_manager import auth_manager
            self.auth_manager = auth_manager
            logger.info("✅ Token manager conectado ao auth_manager")
        except ImportError:
            logger.warning("⚠️ Auth manager não disponível")
    
    def start_monitoring(self):
        """Inicia monitoramento automático de tokens"""
        if not self.auth_manager:
            logger.warning("Auth manager não disponível - monitoramento desabilitado")
            return False
            
        if self.is_monitoring:
            return True
            
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.is_monitoring = True
        
        logger.info("🔄 Monitoramento automático de tokens iniciado")
        return True
    
    def stop_monitoring(self):
        """Para monitoramento automático"""
        self.stop_event.set()
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        logger.info("⏹️ Monitoramento automático de tokens parado")
    
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while not self.stop_event.is_set():
            try:
                self._check_and_renew_token()
                
                # Verificar a cada 60 segundos
                for _ in range(60):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Erro no monitoramento de tokens: {e}")
                time.sleep(30)  # Aguardar antes de tentar novamente
    
    def _check_and_renew_token(self):
        """Verifica se o token precisa ser renovado"""
        if not self.auth_manager:
            return
            
        try:
            # Verificar se há token válido
            if not self.auth_manager.is_authenticated:
                logger.debug("Nenhum token ativo para monitorar")
                return
            
            # Obter informações do token
            token_info = self.auth_manager.get_token_info()
            if not token_info:
                logger.debug("Informações do token não disponíveis")
                return
            
            # Verificar se o token está próximo do vencimento
            expires_at = token_info.get('expires_at')
            if not expires_at:
                logger.debug("Data de expiração do token não disponível")
                return
            
            # Calcular tempo restante
            if isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            
            now = datetime.now(expires_at.tzinfo) if expires_at.tzinfo else datetime.now()
            time_until_expiry = (expires_at - now).total_seconds()
            
            # Verificar se precisa renovar
            if time_until_expiry <= self.renewal_margin:
                logger.info(f"🔄 Token expira em {time_until_expiry:.0f}s - iniciando renovação")
                self._renew_token()
            else:
                logger.debug(f"Token válido por mais {time_until_expiry:.0f}s")
                
        except Exception as e:
            logger.error(f"Erro ao verificar token: {e}")
    
    def _renew_token(self):
        """Renova o token atual"""
        try:
            success = self.auth_manager.refresh_token()
            
            if success:
                logger.info("✅ Token renovado com sucesso")
                
                # Notificar outros componentes sobre a renovação
                self._notify_token_renewed()
            else:
                logger.warning("⚠️ Falha na renovação do token")
                
        except Exception as e:
            logger.error(f"Erro ao renovar token: {e}")
    
    def _notify_token_renewed(self):
        """Notifica outros componentes sobre renovação do token"""
        try:
            # Notificar data_collector para re-autorizar
            from data_collector import data_collector
            if data_collector.is_connected:
                logger.info("🔄 Re-autorizando data_collector com novo token")
                data_collector.is_authorized = False  # Forçar nova autorização
                
        except Exception as e:
            logger.error(f"Erro ao notificar renovação: {e}")
    
    def force_renewal(self) -> bool:
        """Força renovação imediata do token"""
        if not self.auth_manager:
            return False
            
        try:
            return self._renew_token()
        except Exception as e:
            logger.error(f"Erro na renovação forçada: {e}")
            return False
    
    def get_status(self) -> dict:
        """Retorna status do gerenciador de tokens"""
        if not self.auth_manager:
            return {
                'available': False,
                'monitoring': False,
                'error': 'Auth manager não disponível'
            }
        
        status = {
            'available': True,
            'monitoring': self.is_monitoring,
            'authenticated': self.auth_manager.is_authenticated
        }
        
        if self.auth_manager.is_authenticated:
            token_info = self.auth_manager.get_token_info()
            if token_info and 'expires_at' in token_info:
                expires_at = token_info['expires_at']
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                
                now = datetime.now(expires_at.tzinfo) if expires_at.tzinfo else datetime.now()
                time_until_expiry = (expires_at - now).total_seconds()
                
                status.update({
                    'expires_at': expires_at.isoformat(),
                    'expires_in_seconds': max(0, time_until_expiry),
                    'needs_renewal': time_until_expiry <= self.renewal_margin
                })
        
        return status

# Instância global
token_manager = TokenManager()