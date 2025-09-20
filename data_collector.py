"""
Módulo de Coleta de Dados da API Deriv
Coleta ticks e candles em tempo real via WebSocket
"""
import asyncio
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import websocket
from collections import deque

from config import config

logger = logging.getLogger(__name__)

# Import auth_manager para obter token automaticamente
try:
    from auth_manager import auth_manager
    AUTH_MANAGER_AVAILABLE = True
except ImportError:
    AUTH_MANAGER_AVAILABLE = False
    logger.warning("Auth manager não disponível - usando token manual")

class DerivDataCollector:
    """Coletor de dados da API Deriv via WebSocket"""
    
    def __init__(self, app_id=None, api_token=None):
        logger.info("Inicializando Data Collector...")
        
        # Configurações básicas
        self.app_id = app_id or "1089"  # App ID padrão para demo
        self.api_token = api_token
        self.ws = None
        self.is_connected = False
        self.is_authorized = False
        self.subscriptions = {}
        self.tick_buffer = deque(maxlen=config.data.tick_buffer_size)
        self.candle_buffer = {}
        self.callbacks = {}
        self.lock = threading.Lock()
        
        # Request ID counter
        self.req_id = 1
        
        # Balance response storage
        self._balance_response = None
        
        # Informações da conta autorizada
        self.current_loginid = None
        self.account_info = None
        
        # Configurações de coleta
        self.max_buffer_size = 10000
        self.save_interval = 300  # 5 minutos
        self.data_directory = "data"
        
        logger.info("Data Collector inicializado")
        
    def connect(self):
        """Conecta ao WebSocket da Deriv"""
        try:
            logger.info("Conectando ao WebSocket Deriv...")
            
            # Usar app_id do config se disponível, senão usar o padrão
            app_id = getattr(config.deriv, 'app_id', self.app_id)
            
            # Construir URL com app_id
            url = f"{config.deriv.websocket_url}?app_id={app_id}"
            logger.info(f"Conectando à API Deriv: {url}")
            
            websocket.enableTrace(config.debug)
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start connection in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not self.is_connected:
                raise ConnectionError("Falha ao conectar ao WebSocket")
                
            logger.info("✅ Conectado ao WebSocket da Deriv")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            return False
    
    def disconnect(self):
        """Desconecta do WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False
            self.is_authorized = False
            logger.info("Desconectado do WebSocket")
    
    def authorize(self):
        """Autoriza a conexão com o token da API"""
        # Tentar obter token do auth_manager primeiro
        api_token = None
        
        if AUTH_MANAGER_AVAILABLE:
            api_token = auth_manager.get_api_token()
            if api_token:
                logger.info("✅ Usando token do auth_manager")
            else:
                logger.info("⚠️ Token do auth_manager não disponível, usando configuração manual")
        
        # Fallback para token manual
        if not api_token:
            api_token = config.deriv.api_token
            
        if not api_token:
            logger.warning("❌ Nenhum token da API disponível")
            return False
            
        request = {
            "authorize": api_token,
            "req_id": self._get_req_id()
        }
        
        self._send_request(request)
        
        # Wait for authorization
        timeout = 5
        start_time = time.time()
        while not self.is_authorized and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        return self.is_authorized
    
    def subscribe_ticks(self, symbol: str, callback: Optional[Callable] = None):
        """Subscreve aos ticks de um símbolo"""
        req_id = self._get_req_id()
        
        request = {
            "ticks": symbol,
            "subscribe": 1,
            "req_id": req_id
        }
        
        if callback:
            self.callbacks[req_id] = callback
        
        self.subscriptions[req_id] = {
            "type": "ticks",
            "symbol": symbol,
            "active": True
        }
        
        self._send_request(request)
        logger.info(f"Subscrito aos ticks de {symbol}")
        return req_id
    
    def subscribe_candles(self, symbol: str, timeframe: str = "1m", callback: Optional[Callable] = None):
        """Subscreve às velas de um símbolo"""
        req_id = self._get_req_id()
        
        # Convert timeframe to Deriv format
        granularity = self._timeframe_to_granularity(timeframe)
        
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 1000,
            "end": "latest",
            "granularity": granularity,
            "style": "candles",
            "subscribe": 1,
            "req_id": req_id
        }
        
        if callback:
            self.callbacks[req_id] = callback
        
        self.subscriptions[req_id] = {
            "type": "candles",
            "symbol": symbol,
            "timeframe": timeframe,
            "active": True
        }
        
        if symbol not in self.candle_buffer:
            self.candle_buffer[symbol] = {}
        if timeframe not in self.candle_buffer[symbol]:
            self.candle_buffer[symbol][timeframe] = deque(maxlen=1000)
        
        self._send_request(request)
        logger.info(f"Subscrito às velas {timeframe} de {symbol}")
        return req_id
    
    def unsubscribe(self, req_id: int):
        """Remove subscrição"""
        if req_id in self.subscriptions:
            request = {
                "forget": req_id,
                "req_id": self._get_req_id()
            }
            self._send_request(request)
            
            self.subscriptions[req_id]["active"] = False
            if req_id in self.callbacks:
                del self.callbacks[req_id]
            
            logger.info(f"Subscrição {req_id} removida")
    
    def get_historical_data(self, symbol: str, timeframe: str = "1m", count: int = 1000) -> pd.DataFrame:
        """Obtém dados históricos"""
        granularity = self._timeframe_to_granularity(timeframe)
        
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles",
            "req_id": self._get_req_id()
        }
        
        # Send request and wait for response
        response_event = threading.Event()
        response_data = {}
        
        def temp_callback(data):
            response_data.update(data)
            response_event.set()
        
        req_id = request["req_id"]
        self.callbacks[req_id] = temp_callback
        
        self._send_request(request)
        
        # Wait for response
        if response_event.wait(timeout=10):
            if req_id in self.callbacks:
                del self.callbacks[req_id]
            
            if "candles" in response_data:
                return self._candles_to_dataframe(response_data["candles"])
            elif "history" in response_data:
                return self._history_to_dataframe(response_data["history"])
        
        logger.error(f"Timeout ao obter dados históricos para {symbol}")
        return pd.DataFrame()
    
    def get_latest_ticks(self, count: int = 100) -> List[Dict]:
        """Retorna os últimos ticks"""
        with self.lock:
            return list(self.tick_buffer)[-count:]
    
    def get_latest_candles(self, symbol: str, timeframe: str = "1m", count: int = 100) -> pd.DataFrame:
        """Retorna as últimas velas"""
        with self.lock:
            if symbol in self.candle_buffer and timeframe in self.candle_buffer[symbol]:
                candles = list(self.candle_buffer[symbol][timeframe])[-count:]
                return pd.DataFrame(candles)
            return pd.DataFrame()
    
    def save_data_to_csv(self, symbol: str, timeframe: str = "1m", filename: Optional[str] = None):
        """Salva dados em CSV"""
        df = self.get_latest_candles(symbol, timeframe)
        
        if df.empty:
            logger.warning(f"Nenhum dado para salvar: {symbol} {timeframe}")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.data.data_dir}/{symbol}_{timeframe}_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        logger.info(f"Dados salvos em {filename}")
    
    async def get_balance(self) -> Optional[Dict]:
        """Obtém o saldo da conta"""
        if not self.is_connected:
            logger.error("WebSocket não conectado")
            return None
            
        if not self.is_authorized:
            logger.error("Não autorizado - verifique se o token da API é válido")
            return None
            
        try:
            # Limpar resposta anterior
            self._balance_response = None
            
            # Determinar qual conta usar para saldo
            # Se temos OAuth token, podemos usar "all", senão usar loginid específico
            has_oauth = AUTH_MANAGER_AVAILABLE and auth_manager.get_api_token() is not None
            
            if has_oauth:
                account_param = "all"
                logger.debug("Usando OAuth token - consultando todas as contas")
            else:
                account_param = self.current_loginid or "current"
                logger.debug(f"Usando API token - consultando conta: {account_param}")
            
            # Enviar requisição de saldo
            request = {
                "balance": 1,
                "account": account_param,
                "req_id": self._get_req_id()
            }
            
            self._send_request(request)
            
            # Aguardar resposta
            timeout = 10
            start_time = time.time()
            while self._balance_response is None and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if self._balance_response:
                # Verificar se há erro na resposta
                if 'error' in self._balance_response:
                    error_msg = self._balance_response['error'].get('message', 'Erro desconhecido')
                    logger.error(f"Erro da API ao obter saldo: {error_msg}")
                    
                    # Se for erro de permissão, sugerir verificar token
                    if 'oauth token' in error_msg.lower() or 'permission' in error_msg.lower():
                        logger.error("Token da API inválido ou expirado. Verifique o DERIV_API_TOKEN no arquivo .env")
                    
                    return None
                
                return self._balance_response
            else:
                logger.error("Timeout ao obter saldo")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            return None
    
    def _on_open(self, ws):
        """Callback quando WebSocket abre"""
        self.is_connected = True
        logger.info("WebSocket conectado")
        # Autorizar automaticamente após conexão
        self.authorize()
    
    def _on_message(self, ws, message):
        """Callback para mensagens recebidas"""
        try:
            data = json.loads(message)
            self._handle_message(data)
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
    
    def _on_error(self, ws, error):
        """Callback para erros"""
        logger.error(f"Erro WebSocket: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback quando WebSocket fecha"""
        self.is_connected = False
        self.is_authorized = False
        logger.info("WebSocket desconectado")
    
    def _handle_message(self, data: Dict):
        """Processa mensagens recebidas"""
        msg_type = data.get("msg_type")
        req_id = data.get("req_id")
        
        # Handle authorization
        if msg_type == "authorize":
            if data.get("authorize"):
                self.is_authorized = True
                auth_data = data.get("authorize", {})
                self.current_loginid = auth_data.get("loginid")
                self.account_info = auth_data
                logger.info(f"Autorização bem-sucedida - LoginID: {self.current_loginid}")
            else:
                logger.error("Falha na autorização")
        
        # Handle tick data
        elif msg_type == "tick":
            self._handle_tick(data)
        
        # Handle candle data
        elif msg_type == "candles":
            self._handle_candles(data)
        
        # Handle historical data
        elif msg_type == "history":
            self._handle_history(data)
        
        # Handle balance data
        elif msg_type == "balance":
            self._balance_response = data
            logger.debug(f"Saldo recebido: {data}")
        
        # Handle errors
        elif msg_type == "error":
            logger.error(f"Erro da API: {data.get('error', {}).get('message', 'Erro desconhecido')}")
        
        # Call custom callback if exists
        if req_id and req_id in self.callbacks:
            try:
                self.callbacks[req_id](data)
            except Exception as e:
                logger.error(f"Erro no callback {req_id}: {e}")
    
    def _handle_tick(self, data: Dict):
        """Processa dados de tick"""
        tick_data = {
            "symbol": data.get("echo_req", {}).get("ticks"),
            "timestamp": data.get("tick", {}).get("epoch"),
            "price": data.get("tick", {}).get("quote"),
            "datetime": datetime.fromtimestamp(data.get("tick", {}).get("epoch", 0))
        }
        
        with self.lock:
            self.tick_buffer.append(tick_data)
    
    def _handle_candles(self, data: Dict):
        """Processa dados de velas"""
        candles = data.get("candles", [])
        if not candles:
            return
        
        # Get symbol from echo_req
        symbol = data.get("echo_req", {}).get("ticks_history")
        if not symbol:
            return
        
        # Convert to standard format
        processed_candles = []
        for candle in candles:
            processed_candle = {
                "timestamp": candle.get("epoch"),
                "datetime": datetime.fromtimestamp(candle.get("epoch", 0)),
                "open": candle.get("open"),
                "high": candle.get("high"),
                "low": candle.get("low"),
                "close": candle.get("close"),
                "volume": candle.get("volume", 0)
            }
            processed_candles.append(processed_candle)
        
        # Store in buffer
        req_id = data.get("req_id")
        if req_id in self.subscriptions:
            timeframe = self.subscriptions[req_id].get("timeframe", "1m")
            
            with self.lock:
                if symbol not in self.candle_buffer:
                    self.candle_buffer[symbol] = {}
                if timeframe not in self.candle_buffer[symbol]:
                    self.candle_buffer[symbol][timeframe] = deque(maxlen=1000)
                
                self.candle_buffer[symbol][timeframe].extend(processed_candles)
    
    def _handle_history(self, data: Dict):
        """Processa dados históricos"""
        # Similar to candles but for historical requests
        self._handle_candles(data)
    
    def _send_request(self, request: Dict):
        """Envia requisição via WebSocket"""
        if self.ws and self.is_connected:
            message = json.dumps(request)
            self.ws.send(message)
            logger.debug(f"Enviado: {message}")
        else:
            logger.error("WebSocket não conectado")
    
    def _get_req_id(self) -> int:
        """Gera ID único para requisições"""
        req_id = self.req_id
        self.req_id += 1
        return req_id
    
    def _timeframe_to_granularity(self, timeframe: str) -> int:
        """Converte timeframe para granularidade da Deriv"""
        timeframe_map = {
            "1m": 60,
            "2m": 120,
            "3m": 180,
            "5m": 300,
            "10m": 600,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "8h": 28800,
            "1d": 86400
        }
        return timeframe_map.get(timeframe, 60)
    
    def _candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """Converte velas para DataFrame"""
        if not candles:
            return pd.DataFrame()
        
        df_data = []
        for candle in candles:
            df_data.append({
                "timestamp": candle.get("epoch"),
                "datetime": pd.to_datetime(candle.get("epoch"), unit="s"),
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0)),
                "volume": float(candle.get("volume", 0))
            })
        
        df = pd.DataFrame(df_data)
        df.set_index("datetime", inplace=True)
        return df
    
    def _history_to_dataframe(self, history: Dict) -> pd.DataFrame:
        """Converte histórico para DataFrame"""
        times = history.get("times", [])
        prices = history.get("prices", [])
        
        if len(times) != len(prices):
            logger.error("Tamanhos incompatíveis entre times e prices")
            return pd.DataFrame()
        
        df_data = []
        for timestamp, price in zip(times, prices):
            df_data.append({
                "timestamp": timestamp,
                "datetime": pd.to_datetime(timestamp, unit="s"),
                "price": float(price)
            })
        
        df = pd.DataFrame(df_data)
        df.set_index("datetime", inplace=True)
        return df

# Singleton instance
data_collector = DerivDataCollector()

# Convenience functions
def connect_to_deriv():
    """Conecta ao Deriv e autoriza"""
    if data_collector.connect():
        return data_collector.authorize()
    return False

def get_live_data(symbol: str, timeframe: str = "1m", count: int = 100) -> pd.DataFrame:
    """Obtém dados em tempo real"""
    if not data_collector.is_connected:
        if not connect_to_deriv():
            return pd.DataFrame()
    
    return data_collector.get_historical_data(symbol, timeframe, count)

def start_data_collection(symbols: List[str], timeframes: List[str] = None):
    """Inicia coleta de dados para múltiplos símbolos"""
    if timeframes is None:
        timeframes = ["1m", "5m"]
    
    if not data_collector.is_connected:
        if not connect_to_deriv():
            return False
    
    for symbol in symbols:
        # Subscribe to ticks
        data_collector.subscribe_ticks(symbol)
        
        # Subscribe to candles
        for timeframe in timeframes:
            data_collector.subscribe_candles(symbol, timeframe)
    
    logger.info(f"Coleta iniciada para {len(symbols)} símbolos")
    return True