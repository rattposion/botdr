"""
Executor de Trades para API Deriv
Gerencia execução de contratos, monitoramento e resultados
"""
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid

from data_collector import data_collector
from auth_manager import auth_manager
from config import config
from utils import get_logger, log_trade

class ContractType(Enum):
    """Tipos de contrato disponíveis"""
    CALL = "CALL"
    PUT = "PUT"
    DIGITEVEN = "DIGITEVEN"
    DIGITODD = "DIGITODD"
    DIGITOVER = "DIGITOVER"
    DIGITUNDER = "DIGITUNDER"

class ContractStatus(Enum):
    """Status do contrato"""
    PENDING = "pending"
    ACTIVE = "active"
    WON = "won"
    LOST = "lost"
    CANCELLED = "cancelled"
    ERROR = "error"

class TradeExecutor:
    """Executor de trades para Deriv"""
    
    def __init__(self):
        self.logger = get_logger('trade_executor')
        self.data_collector = data_collector
        
        # Contratos ativos
        self.active_contracts = {}
        self.contract_history = []
        
        # Configurações de trading
        self.default_duration = 5  # 5 ticks
        self.default_stake = 1.0
        self.max_concurrent_trades = 3
        
        # Estatísticas
        self.stats = {
            "total_contracts": 0,
            "won_contracts": 0,
            "lost_contracts": 0,
            "total_stake": 0.0,
            "total_payout": 0.0,
            "net_profit": 0.0,
            "win_rate": 0.0,
            "avg_payout": 0.0
        }
        
        # Request ID para tracking
        self.req_id_counter = 1000
    
    async def buy_contract(self, 
                          contract_type: str,
                          symbol: str,
                          stake: float,
                          duration: int = None,
                          duration_unit: str = "t",  # t=ticks, s=seconds, m=minutes
                          barrier: float = None) -> Dict[str, Any]:
        """
        Compra um contrato na Deriv
        
        Args:
            contract_type: Tipo do contrato (CALL, PUT, etc.)
            symbol: Símbolo para trading (R_10, R_25, etc.)
            stake: Valor do stake
            duration: Duração do contrato
            duration_unit: Unidade da duração (t, s, m)
            barrier: Barreira para contratos que precisam
        
        Returns:
            Dict com resultado da compra
        """
        try:
            # Verificar se pode fazer mais trades
            if len(self.active_contracts) >= self.max_concurrent_trades:
                return {
                    "success": False,
                    "error": "Máximo de trades simultâneos atingido",
                    "contract_id": None
                }
            
            # Verificar autenticação
            if not auth_manager.is_authenticated():
                return {
                    "success": False,
                    "error": "Não autenticado",
                    "contract_id": None
                }
            
            # Preparar parâmetros do contrato
            if duration is None:
                duration = self.default_duration
            
            contract_params = {
                "buy": "1",
                "price": stake,
                "parameters": {
                    "contract_type": contract_type,
                    "symbol": symbol,
                    "duration": duration,
                    "duration_unit": duration_unit
                },
                "req_id": self._get_req_id()
            }
            
            # Adicionar barreira se necessário
            if barrier is not None:
                contract_params["parameters"]["barrier"] = barrier
            
            # Enviar requisição
            self.data_collector._send_request(contract_params)
            
            # Criar registro do contrato
            contract_id = str(uuid.uuid4())
            contract_data = {
                "contract_id": contract_id,
                "req_id": contract_params["req_id"],
                "type": contract_type,
                "symbol": symbol,
                "stake": stake,
                "duration": duration,
                "duration_unit": duration_unit,
                "barrier": barrier,
                "status": ContractStatus.PENDING,
                "buy_time": datetime.now(),
                "buy_price": None,
                "sell_time": None,
                "sell_price": None,
                "payout": 0.0,
                "profit": 0.0
            }
            
            self.active_contracts[contract_id] = contract_data
            self.stats["total_contracts"] += 1
            self.stats["total_stake"] += stake
            
            self.logger.info(f"💰 Contrato enviado: {contract_type} {symbol} ${stake}")
            
            # Configurar callback para resultado
            self.data_collector.callbacks[contract_params["req_id"]] = \
                lambda data: asyncio.create_task(self._handle_buy_response(contract_id, data))
            
            return {
                "success": True,
                "contract_id": contract_id,
                "req_id": contract_params["req_id"],
                "message": f"Contrato {contract_type} enviado para {symbol}"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao comprar contrato: {e}")
            return {
                "success": False,
                "error": str(e),
                "contract_id": None
            }
    
    async def _handle_buy_response(self, contract_id: str, response_data: Dict):
        """Processa resposta da compra de contrato"""
        try:
            if contract_id not in self.active_contracts:
                return
            
            contract = self.active_contracts[contract_id]
            
            if response_data.get("msg_type") == "buy":
                buy_data = response_data.get("buy", {})
                
                if buy_data:
                    # Contrato comprado com sucesso
                    contract["status"] = ContractStatus.ACTIVE
                    contract["buy_price"] = buy_data.get("buy_price")
                    contract["contract_id"] = buy_data.get("contract_id")
                    contract["payout"] = buy_data.get("payout", 0)
                    
                    self.logger.info(f"✅ Contrato ativo: {contract['type']} {contract['symbol']} "
                                   f"ID: {buy_data.get('contract_id')}")
                    
                    # Iniciar monitoramento
                    asyncio.create_task(self._monitor_contract(contract_id))
                    
                else:
                    # Erro na compra
                    contract["status"] = ContractStatus.ERROR
                    error_msg = response_data.get("error", {}).get("message", "Erro desconhecido")
                    self.logger.error(f"❌ Erro na compra: {error_msg}")
            
            elif response_data.get("msg_type") == "error":
                # Erro na requisição
                contract["status"] = ContractStatus.ERROR
                error_msg = response_data.get("error", {}).get("message", "Erro desconhecido")
                self.logger.error(f"❌ Erro na requisição: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao processar resposta de compra: {e}")
    
    async def _monitor_contract(self, contract_id: str):
        """Monitora um contrato ativo até o resultado"""
        try:
            if contract_id not in self.active_contracts:
                return
            
            contract = self.active_contracts[contract_id]
            deriv_contract_id = contract.get("contract_id")
            
            if not deriv_contract_id:
                return
            
            # Subscrever para atualizações do contrato
            proposal_open_contract = {
                "proposal_open_contract": "1",
                "contract_id": deriv_contract_id,
                "req_id": self._get_req_id()
            }
            
            self.data_collector._send_request(proposal_open_contract)
            
            # Configurar callback para atualizações
            self.data_collector.callbacks[proposal_open_contract["req_id"]] = \
                lambda data: asyncio.create_task(self._handle_contract_update(contract_id, data))
            
            self.logger.debug(f"👁️ Monitorando contrato {deriv_contract_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao monitorar contrato: {e}")
    
    async def _handle_contract_update(self, contract_id: str, update_data: Dict):
        """Processa atualizações do contrato"""
        try:
            if contract_id not in self.active_contracts:
                return
            
            contract = self.active_contracts[contract_id]
            
            if update_data.get("msg_type") == "proposal_open_contract":
                poc_data = update_data.get("proposal_open_contract", {})
                
                # Verificar se o contrato terminou
                if poc_data.get("is_sold"):
                    await self._finalize_contract(contract_id, poc_data)
                else:
                    # Atualizar informações do contrato
                    contract["current_spot"] = poc_data.get("current_spot")
                    contract["bid_price"] = poc_data.get("bid_price")
                    
        except Exception as e:
            self.logger.error(f"❌ Erro ao processar atualização: {e}")
    
    async def _finalize_contract(self, contract_id: str, final_data: Dict):
        """Finaliza um contrato e calcula resultados"""
        try:
            if contract_id not in self.active_contracts:
                return
            
            contract = self.active_contracts[contract_id]
            
            # Atualizar dados finais
            contract["sell_time"] = datetime.now()
            contract["sell_price"] = final_data.get("sell_price", 0)
            contract["final_spot"] = final_data.get("current_spot")
            
            # Calcular resultado
            payout = final_data.get("sell_price", 0)
            stake = contract["stake"]
            profit = payout - stake
            
            contract["payout"] = payout
            contract["profit"] = profit
            
            # Determinar se ganhou ou perdeu
            if profit > 0:
                contract["status"] = ContractStatus.WON
                self.stats["won_contracts"] += 1
                self.logger.info(f"🎉 GANHOU: {contract['type']} {contract['symbol']} "
                               f"Lucro: ${profit:.2f}")
            else:
                contract["status"] = ContractStatus.LOST
                self.stats["lost_contracts"] += 1
                self.logger.info(f"😞 PERDEU: {contract['type']} {contract['symbol']} "
                               f"Perda: ${abs(profit):.2f}")
            
            # Atualizar estatísticas
            self.stats["total_payout"] += payout
            self.stats["net_profit"] += profit
            
            if self.stats["total_contracts"] > 0:
                self.stats["win_rate"] = (self.stats["won_contracts"] / self.stats["total_contracts"]) * 100
                self.stats["avg_payout"] = self.stats["total_payout"] / self.stats["total_contracts"]
            
            # Mover para histórico
            self.contract_history.append(contract.copy())
            del self.active_contracts[contract_id]
            
            # Log do trade
            log_trade({
                "contract_id": contract_id,
                "type": contract["type"],
                "symbol": contract["symbol"],
                "stake": stake,
                "payout": payout,
                "profit": profit,
                "status": contract["status"].value,
                "duration": (contract["sell_time"] - contract["buy_time"]).total_seconds(),
                "timestamp": contract["sell_time"]
            })
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao finalizar contrato: {e}")
    
    async def sell_contract(self, contract_id: str) -> Dict[str, Any]:
        """Vende um contrato antes do vencimento"""
        try:
            if contract_id not in self.active_contracts:
                return {
                    "success": False,
                    "error": "Contrato não encontrado"
                }
            
            contract = self.active_contracts[contract_id]
            deriv_contract_id = contract.get("contract_id")
            
            if not deriv_contract_id:
                return {
                    "success": False,
                    "error": "ID do contrato Deriv não encontrado"
                }
            
            # Enviar requisição de venda
            sell_request = {
                "sell": deriv_contract_id,
                "price": 0,  # Vender pelo preço atual
                "req_id": self._get_req_id()
            }
            
            self.data_collector._send_request(sell_request)
            
            self.logger.info(f"📤 Vendendo contrato {deriv_contract_id}")
            
            return {
                "success": True,
                "message": f"Venda solicitada para contrato {deriv_contract_id}"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao vender contrato: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_active_contracts(self) -> List[Dict]:
        """Retorna lista de contratos ativos"""
        return list(self.active_contracts.values())
    
    def get_contract_history(self, limit: int = 50) -> List[Dict]:
        """Retorna histórico de contratos"""
        return self.contract_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de trading"""
        return self.stats.copy()
    
    def _get_req_id(self) -> int:
        """Gera ID único para requisições"""
        req_id = self.req_id_counter
        self.req_id_counter += 1
        return req_id
    
    def reset_statistics(self):
        """Reseta estatísticas de trading"""
        self.stats = {
            "total_contracts": 0,
            "won_contracts": 0,
            "lost_contracts": 0,
            "total_stake": 0.0,
            "total_payout": 0.0,
            "net_profit": 0.0,
            "win_rate": 0.0,
            "avg_payout": 0.0
        }
        self.contract_history.clear()
        self.logger.info("📊 Estatísticas resetadas")

# Instância global
trade_executor = TradeExecutor()

# Funções de conveniência
async def buy_call(symbol: str, stake: float, duration: int = 5) -> Dict[str, Any]:
    """Compra contrato CALL"""
    return await trade_executor.buy_contract("CALL", symbol, stake, duration)

async def buy_put(symbol: str, stake: float, duration: int = 5) -> Dict[str, Any]:
    """Compra contrato PUT"""
    return await trade_executor.buy_contract("PUT", symbol, stake, duration)

async def buy_digit_even(symbol: str, stake: float, duration: int = 5) -> Dict[str, Any]:
    """Compra contrato DIGIT EVEN"""
    return await trade_executor.buy_contract("DIGITEVEN", symbol, stake, duration)

async def buy_digit_odd(symbol: str, stake: float, duration: int = 5) -> Dict[str, Any]:
    """Compra contrato DIGIT ODD"""
    return await trade_executor.buy_contract("DIGITODD", symbol, stake, duration)

def get_trading_stats() -> Dict[str, Any]:
    """Retorna estatísticas de trading"""
    return trade_executor.get_statistics()

def get_active_trades() -> List[Dict]:
    """Retorna trades ativos"""
    return trade_executor.get_active_contracts()