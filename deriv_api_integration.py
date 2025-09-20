"""
Integração com API Real da Deriv para Execução de Trades
Sistema completo de conexão, autenticação e execução de trades
"""

import asyncio
import websockets
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DerivConfig:
    """Configurações da API Deriv"""
    app_id: str = "1089"  # App ID público para demo
    api_url: str = "wss://ws.binaryws.com/websockets/v3"
    demo_account: bool = True
    api_token: Optional[str] = None  # Token para conta real
    
@dataclass
class MarketData:
    """Dados de mercado em tempo real"""
    symbol: str
    timestamp: datetime
    price: float
    volume: Optional[float] = None

@dataclass
class TradeRequest:
    """Solicitação de trade"""
    symbol: str
    direction: str  # 'CALL' ou 'PUT'
    amount: float
    duration: int = 60  # segundos
    duration_unit: str = 's'
    barrier: Optional[float] = None

@dataclass
class TradeResult:
    """Resultado do trade"""
    contract_id: str
    symbol: str
    direction: str
    amount: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = 'OPEN'  # OPEN, WIN, LOSS
    timestamp: datetime = None

class DerivAPIClient:
    """Cliente para API da Deriv"""
    
    def __init__(self, config: DerivConfig):
        self.config = config
        self.websocket = None
        self.is_connected = False
        self.is_authorized = False
        self.request_id = 1
        self.callbacks = {}
        self.market_data = {}
        self.active_contracts = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    async def connect(self):
        """Conectar à API da Deriv"""
        try:
            url = f"{self.config.api_url}?app_id={self.config.app_id}"
            self.websocket = await websockets.connect(url)
            self.is_connected = True
            self.logger.info("Conectado à API da Deriv")
            
            # Iniciar loop de recebimento de mensagens
            asyncio.create_task(self.message_handler())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao conectar: {e}")
            return False
            
    async def disconnect(self):
        """Desconectar da API"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.is_authorized = False
            self.logger.info("Desconectado da API da Deriv")
            
    async def send_request(self, request: Dict, callback: Callable = None) -> int:
        """Enviar solicitação para API"""
        if not self.is_connected:
            raise Exception("Não conectado à API")
            
        request_id = self.request_id
        self.request_id += 1
        
        request['req_id'] = request_id
        
        if callback:
            self.callbacks[request_id] = callback
            
        await self.websocket.send(json.dumps(request))
        self.logger.debug(f"Enviado: {request}")
        
        return request_id
        
    async def message_handler(self):
        """Manipular mensagens recebidas"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                self.logger.debug(f"Recebido: {data}")
                
                # Verificar se há callback para esta resposta
                req_id = data.get('req_id')
                if req_id and req_id in self.callbacks:
                    callback = self.callbacks.pop(req_id)
                    await callback(data)
                    
                # Processar tipos específicos de mensagem
                await self.process_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Conexão WebSocket fechada")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Erro no handler de mensagens: {e}")
            
    async def process_message(self, data: Dict):
        """Processar mensagens específicas"""
        msg_type = data.get('msg_type')
        
        if msg_type == 'tick':
            await self.process_tick(data)
        elif msg_type == 'proposal_open_contract':
            await self.process_contract_update(data)
        elif msg_type == 'buy':
            await self.process_buy_response(data)
        elif msg_type == 'authorize':
            await self.process_authorize_response(data)
            
    async def process_tick(self, data: Dict):
        """Processar tick de preço"""
        tick = data.get('tick', {})
        symbol = tick.get('symbol')
        
        if symbol:
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(tick.get('epoch', 0)),
                price=float(tick.get('quote', 0))
            )
            self.market_data[symbol] = market_data
            
    async def process_contract_update(self, data: Dict):
        """Processar atualização de contrato"""
        contract = data.get('proposal_open_contract', {})
        contract_id = contract.get('contract_id')
        
        if contract_id and contract_id in self.active_contracts:
            trade = self.active_contracts[contract_id]
            
            # Atualizar status do contrato
            if contract.get('is_sold'):
                trade.exit_price = float(contract.get('exit_tick', 0))
                trade.pnl = float(contract.get('profit', 0))
                trade.status = 'WIN' if trade.pnl > 0 else 'LOSS'
                self.logger.info(f"Contrato {contract_id} finalizado: {trade.status} PnL: {trade.pnl}")
                
    async def process_buy_response(self, data: Dict):
        """Processar resposta de compra"""
        buy_info = data.get('buy', {})
        contract_id = buy_info.get('contract_id')
        
        if contract_id:
            self.logger.info(f"Trade executado: Contrato {contract_id}")
            
    async def process_authorize_response(self, data: Dict):
        """Processar resposta de autorização"""
        if data.get('authorize'):
            self.is_authorized = True
            self.logger.info("Autorização bem-sucedida")
        else:
            self.logger.error("Falha na autorização")
            
    async def authorize(self, api_token: str):
        """Autorizar com token da API"""
        request = {
            "authorize": api_token
        }
        
        await self.send_request(request)
        
        # Aguardar autorização
        for _ in range(50):  # 5 segundos
            if self.is_authorized:
                return True
            await asyncio.sleep(0.1)
            
        return False
        
    async def subscribe_ticks(self, symbols: List[str]):
        """Subscrever ticks de símbolos"""
        for symbol in symbols:
            request = {
                "ticks": symbol,
                "subscribe": 1
            }
            await self.send_request(request)
            self.logger.info(f"Subscrito aos ticks de {symbol}")
            
    async def get_active_symbols(self) -> List[Dict]:
        """Obter símbolos ativos"""
        request = {
            "active_symbols": "brief",
            "product_type": "basic"
        }
        
        response_data = {}
        
        async def callback(data):
            response_data['symbols'] = data.get('active_symbols', [])
            
        await self.send_request(request, callback)
        
        # Aguardar resposta
        for _ in range(50):
            if 'symbols' in response_data:
                return response_data['symbols']
            await asyncio.sleep(0.1)
            
        return []
        
    async def get_proposal(self, trade_request: TradeRequest) -> Optional[Dict]:
        """Obter proposta de trade"""
        request = {
            "proposal": 1,
            "amount": trade_request.amount,
            "basis": "stake",
            "contract_type": trade_request.direction,
            "currency": "USD",
            "duration": trade_request.duration,
            "duration_unit": trade_request.duration_unit,
            "symbol": trade_request.symbol
        }
        
        if trade_request.barrier:
            request["barrier"] = trade_request.barrier
            
        response_data = {}
        
        async def callback(data):
            response_data['proposal'] = data.get('proposal')
            
        await self.send_request(request, callback)
        
        # Aguardar resposta
        for _ in range(50):
            if 'proposal' in response_data:
                return response_data['proposal']
            await asyncio.sleep(0.1)
            
        return None
        
    async def buy_contract(self, proposal_id: str, price: float) -> Optional[str]:
        """Comprar contrato"""
        request = {
            "buy": proposal_id,
            "price": price
        }
        
        response_data = {}
        
        async def callback(data):
            buy_info = data.get('buy')
            if buy_info:
                response_data['contract_id'] = buy_info.get('contract_id')
                
        await self.send_request(request, callback)
        
        # Aguardar resposta
        for _ in range(50):
            if 'contract_id' in response_data:
                return response_data['contract_id']
            await asyncio.sleep(0.1)
            
        return None
        
    async def execute_trade(self, trade_request: TradeRequest) -> Optional[TradeResult]:
        """Executar trade completo"""
        try:
            # Obter proposta
            proposal = await self.get_proposal(trade_request)
            if not proposal:
                self.logger.error("Falha ao obter proposta")
                return None
                
            proposal_id = proposal.get('id')
            ask_price = float(proposal.get('ask_price', 0))
            
            # Comprar contrato
            contract_id = await self.buy_contract(proposal_id, ask_price)
            if not contract_id:
                self.logger.error("Falha ao comprar contrato")
                return None
                
            # Criar resultado do trade
            trade_result = TradeResult(
                contract_id=contract_id,
                symbol=trade_request.symbol,
                direction=trade_request.direction,
                amount=trade_request.amount,
                entry_price=ask_price,
                timestamp=datetime.now()
            )
            
            # Adicionar aos contratos ativos
            self.active_contracts[contract_id] = trade_result
            
            # Subscrever atualizações do contrato
            await self.subscribe_contract(contract_id)
            
            self.logger.info(f"Trade executado: {contract_id}")
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Erro ao executar trade: {e}")
            return None
            
    async def subscribe_contract(self, contract_id: str):
        """Subscrever atualizações de contrato"""
        request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1
        }
        await self.send_request(request)
        
    async def get_balance(self) -> Optional[float]:
        """Obter saldo da conta"""
        if not self.is_authorized:
            self.logger.error("Não autorizado para obter saldo")
            return None
            
        request = {
            "balance": 1,
            "subscribe": 1
        }
        
        response_data = {}
        
        async def callback(data):
            balance_info = data.get('balance')
            if balance_info:
                response_data['balance'] = float(balance_info.get('balance', 0))
                
        await self.send_request(request, callback)
        
        # Aguardar resposta
        for _ in range(50):
            if 'balance' in response_data:
                return response_data['balance']
            await asyncio.sleep(0.1)
            
        return None

class DerivTradingBot:
    """Bot de trading integrado com API da Deriv"""
    
    def __init__(self, config: DerivConfig):
        self.config = config
        self.client = DerivAPIClient(config)
        self.is_running = False
        self.trade_history = []
        self.setup_logging()
        
        # Configurações de trading
        self.default_stake = 1.0
        self.default_duration = 5  # 5 ticks
        self.max_concurrent_trades = 3
        
    def setup_logging(self):
        """Configurar logging"""
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Iniciar bot"""
        self.logger.info("🚀 Iniciando Bot de Trading Deriv")
        
        # Conectar à API
        if not await self.client.connect():
            self.logger.error("Falha ao conectar à API")
            return False
            
        # Autorizar se token fornecido
        if self.config.api_token:
            if not await self.client.authorize(self.config.api_token):
                self.logger.error("Falha na autorização")
                return False
                
        # Obter símbolos ativos
        symbols = await self.client.get_active_symbols()
        active_symbols = [s['symbol'] for s in symbols if s.get('exchange_is_open')]
        self.logger.info(f"Símbolos ativos: {len(active_symbols)}")
        
        # Subscrever ticks dos principais símbolos
        main_symbols = ['R_10', 'R_25', 'R_50', 'R_75', 'R_100']
        available_symbols = [s for s in main_symbols if s in active_symbols]
        
        if available_symbols:
            await self.client.subscribe_ticks(available_symbols)
            
        self.is_running = True
        return True
        
    async def stop(self):
        """Parar bot"""
        self.is_running = False
        await self.client.disconnect()
        self.logger.info("Bot parado")
        
    async def execute_trade_with_signal(self, symbol: str, direction: str, 
                                      confidence: float, amount: float = 1.0) -> bool:
        """Executar trade baseado em sinal"""
        try:
            # Criar solicitação de trade
            trade_request = TradeRequest(
                symbol=symbol,
                direction=direction,
                amount=amount,
                duration=60
            )
            
            # Executar trade
            result = await self.client.execute_trade(trade_request)
            
            if result:
                self.trade_history.append(result)
                self.logger.info(f"Trade executado: {symbol} {direction} ${amount}")
                return True
            else:
                self.logger.error("Falha ao executar trade")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao executar trade: {e}")
            return False
            
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Obter dados de mercado"""
        return self.client.market_data.get(symbol)
        
    async def get_account_balance(self) -> Optional[float]:
        """Obter saldo da conta"""
        return await self.client.get_balance()
        
    def get_trade_statistics(self) -> Dict:
        """Obter estatísticas de trades"""
        if not self.trade_history:
            return {}
            
        completed_trades = [t for t in self.trade_history if t.status in ['WIN', 'LOSS']]
        
        if not completed_trades:
            return {'total_trades': len(self.trade_history), 'completed_trades': 0}
            
        wins = [t for t in completed_trades if t.status == 'WIN']
        total_pnl = sum(t.pnl for t in completed_trades if t.pnl)
        
        return {
            'total_trades': len(self.trade_history),
            'completed_trades': len(completed_trades),
            'wins': len(wins),
            'win_rate': len(wins) / len(completed_trades),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(completed_trades) if completed_trades else 0
        }

async def demo_trading_session():
    """Demonstração de sessão de trading"""
    print("🔗 Iniciando Integração com API da Deriv")
    
    # Configurar para conta demo
    config = DerivConfig(
        demo_account=True,
        app_id="1089"  # App ID público
    )
    
    # Criar bot
    bot = DerivTradingBot(config)
    
    try:
        # Iniciar bot
        if not await bot.start():
            print("❌ Falha ao iniciar bot")
            return
            
        print("✅ Bot conectado com sucesso")
        
        # Aguardar alguns ticks
        print("📊 Aguardando dados de mercado...")
        await asyncio.sleep(5)
        
        # Verificar dados de mercado
        symbols = ['R_10', 'R_25', 'R_50']
        for symbol in symbols:
            market_data = await bot.get_market_data(symbol)
            if market_data:
                print(f"{symbol}: ${market_data.price:.5f} ({market_data.timestamp})")
            else:
                print(f"{symbol}: Sem dados")
                
        # Simular alguns trades (apenas em demo)
        if config.demo_account:
            print("\n🎯 Executando trades de demonstração...")
            
            # Trade 1: CALL em R_10
            success = await bot.execute_trade_with_signal('R_10', 'CALL', 0.75, 1.0)
            if success:
                print("✅ Trade 1 executado: R_10 CALL")
            else:
                print("❌ Falha no Trade 1")
                
            await asyncio.sleep(2)
            
            # Trade 2: PUT em R_25
            success = await bot.execute_trade_with_signal('R_25', 'PUT', 0.68, 1.5)
            if success:
                print("✅ Trade 2 executado: R_25 PUT")
            else:
                print("❌ Falha no Trade 2")
                
        # Aguardar um pouco para ver resultados
        print("\n⏳ Aguardando resultados dos trades...")
        await asyncio.sleep(10)
        
        # Mostrar estatísticas
        stats = bot.get_trade_statistics()
        print(f"\n📊 Estatísticas da Sessão:")
        print(f"Total de Trades: {stats.get('total_trades', 0)}")
        print(f"Trades Completados: {stats.get('completed_trades', 0)}")
        print(f"Taxa de Vitória: {stats.get('win_rate', 0):.1%}")
        print(f"PnL Total: ${stats.get('total_pnl', 0):.2f}")
        
        # Verificar saldo (se autorizado)
        balance = await bot.get_account_balance()
        if balance is not None:
            print(f"Saldo da Conta: ${balance:.2f}")
            
    except Exception as e:
        print(f"❌ Erro na sessão: {e}")
        
    finally:
        # Parar bot
        await bot.stop()
        print("🔚 Sessão finalizada")

def run_deriv_integration():
    """Executar integração com Deriv"""
    print("🌐 Sistema de Integração com API da Deriv")
    print("⚠️  ATENÇÃO: Este é um sistema para conta DEMO")
    print("⚠️  Para conta real, configure o API token adequadamente")
    
    # Executar demonstração
    asyncio.run(demo_trading_session())

if __name__ == "__main__":
    run_deriv_integration()