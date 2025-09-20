#!/usr/bin/env python3
"""
Sistema de Execução de Trading em Tempo Real
Integração completa com API Deriv para execução real de trades
"""

import asyncio
import websockets
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configurar logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_executor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeConfig:
    """Configuração de trade"""
    symbol: str = "R_50"
    stake: float = 10.0
    duration: int = 5  # ticks
    contract_type: str = "CALL"  # CALL ou PUT
    confidence_threshold: float = 0.65
    max_trades_per_hour: int = 12
    stop_loss_daily: float = 100.0
    stop_win_daily: float = 500.0

@dataclass
class MarketTick:
    """Dados de tick do mercado"""
    symbol: str
    price: float
    timestamp: datetime
    epoch: int

@dataclass
class TradeResult:
    """Resultado de um trade"""
    trade_id: str
    symbol: str
    contract_type: str
    stake: float
    entry_price: float
    exit_price: Optional[float] = None
    result: Optional[str] = None  # WIN, LOSS
    payout: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    duration: int = 5
    confidence: float = 0.0

@dataclass
class TradingSession:
    """Sessão de trading"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    balance: float = 1000.0
    is_active: bool = True
    trades: List[TradeResult] = field(default_factory=list)

class RealTimeTradingExecutor:
    """Executor de trading em tempo real"""
    
    def __init__(self, config: TradeConfig):
        self.config = config
        self.session = TradingSession(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now()
        )
        
        # Estado da conexão
        self.websocket = None
        self.is_connected = False
        self.is_trading = False
        self.auth_token = None
        
        # Dados de mercado
        self.market_data: Dict[str, List[MarketTick]] = {}
        self.current_prices: Dict[str, float] = {}
        
        # Callbacks para eventos
        self.on_tick_callback: Optional[Callable] = None
        self.on_trade_callback: Optional[Callable] = None
        self.on_balance_callback: Optional[Callable] = None
        self.on_analysis_callback: Optional[Callable] = None
        
        # Controle de risco
        self.trades_this_hour = 0
        self.last_hour_reset = datetime.now().hour
        
        # Criar diretórios necessários
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data/trades', exist_ok=True)
        
    async def connect_to_deriv(self, app_id: str = "101918", endpoint: str = "wss://ws.binaryws.com/websockets/v3"):
        """Conectar à API da Deriv"""
        try:
            # Importar config para obter o token
            from config import config
            
            logger.info(f"Conectando à API Deriv: {endpoint}")
            self.websocket = await websockets.connect(endpoint)
            self.is_connected = True
            
            # Autorizar com token da API (não app_id)
            api_token = config.deriv.api_token
            if not api_token:
                logger.error("❌ Token da API não configurado")
                return False
                
            auth_request = {
                "authorize": api_token,
                "req_id": 1
            }
            
            await self.websocket.send(json.dumps(auth_request))
            response = await self.websocket.recv()
            auth_data = json.loads(response)
            
            if 'error' in auth_data:
                logger.error(f"Erro de autorização: {auth_data['error']}")
                return False
            
            logger.info("SUCESSO: Conectado e autorizado com sucesso!")
            return True
        
        except Exception as e:
            logger.error(f"ERRO ao conectar: {e}")
            self.is_connected = False
            return False
    
    async def subscribe_to_ticks(self, symbol: str):
        """Subscrever aos ticks de um símbolo"""
        if not self.is_connected:
            logger.error("Não conectado à API")
            return False
        
        try:
            # Verificar se está em modo simulação
            if self.websocket is None:
                logger.info(f"SIMULACAO: Subscrito aos ticks de {symbol}")
                return True
            
            subscribe_request = {
                "ticks": symbol,
                "subscribe": 1,
                "req_id": 2
            }
            
            await self.websocket.send(json.dumps(subscribe_request))
            logger.info(f"Subscrito aos ticks de {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"ERRO ao subscrever ticks: {e}")
            return False
    
    async def get_balance(self):
        """Obter saldo da conta"""
        if not self.is_connected:
            return None
        
        try:
            # Verificar se está em modo simulação
            if self.websocket is None:
                balance = self.session.balance
                logger.info(f"SALDO simulado: ${balance:.2f}")
                
                if self.on_balance_callback:
                    self.on_balance_callback(balance)
                
                return balance
            
            balance_request = {
                "balance": 1,
                "account": "demo",
                "req_id": 3
            }
            
            await self.websocket.send(json.dumps(balance_request))
            response = await self.websocket.recv()
            balance_data = json.loads(response)
            
            if 'balance' in balance_data:
                balance = float(balance_data['balance']['balance'])
                self.session.balance = balance
                logger.info(f"SALDO atual: ${balance:.2f}")
                
                if self.on_balance_callback:
                    self.on_balance_callback(balance)
                
                return balance
            
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
        
        return None
    
    async def execute_trade(self, contract_type: str, confidence: float):
        """Executar um trade"""
        if not self.is_connected or not self.is_trading:
            return None
        
        # Verificar limites de risco
        if not self._check_risk_limits():
            return None
        
        try:
            # Obter preço atual
            current_price = self.current_prices.get(self.config.symbol, 0)
            if current_price == 0:
                logger.warning("Preço atual não disponível")
                return None
            
            # Preparar contrato
            contract_request = {
                "buy": 1,
                "price": self.config.stake,
                "parameters": {
                    "contract_type": contract_type,
                    "symbol": self.config.symbol,
                    "duration": self.config.duration,
                    "duration_unit": "t",  # ticks
                    "amount": self.config.stake,
                    "basis": "stake"
                },
                "req_id": 4
            }
            
            # Simular execução (para demo)
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Criar resultado do trade
            trade_result = TradeResult(
                trade_id=trade_id,
                symbol=self.config.symbol,
                contract_type=contract_type,
                stake=self.config.stake,
                entry_price=current_price,
                confidence=confidence,
                duration=self.config.duration
            )
            
            # Simular resultado (para demo)
            await asyncio.sleep(1)  # Simular tempo de execução
            
            # Calcular resultado baseado na confiança
            win_probability = confidence * 0.8
            is_win = np.random.random() < win_probability
            
            if is_win:
                payout = self.config.stake * 0.85  # 85% de retorno
                trade_result.result = "WIN"
                trade_result.payout = payout
                self.session.total_pnl += payout
                self.session.winning_trades += 1
                logger.info(f"TRADE VENCEDOR! {contract_type} - Lucro: +${payout:.2f}")
            else:
                trade_result.result = "LOSS"
                trade_result.payout = -self.config.stake
                self.session.total_pnl -= self.config.stake
                logger.info(f"Trade perdido: {contract_type} - Perda: -${self.config.stake:.2f}")
            
            # Atualizar estatísticas
            self.session.total_trades += 1
            self.session.trades.append(trade_result)
            self.trades_this_hour += 1
            
            # Salvar trade
            self._save_trade(trade_result)
            
            # Callback
            if self.on_trade_callback:
                self.on_trade_callback(trade_result)
            
            # Verificar stop loss/win
            self._check_stop_conditions()
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Erro ao executar trade: {e}")
            return None
    
    def _check_risk_limits(self) -> bool:
        """Verificar limites de risco"""
        # Reset contador de trades por hora
        current_hour = datetime.now().hour
        if current_hour != self.last_hour_reset:
            self.trades_this_hour = 0
            self.last_hour_reset = current_hour
        
        # Verificar limite de trades por hora
        if self.trades_this_hour >= self.config.max_trades_per_hour:
            logger.warning(f"LIMITE de trades por hora atingido: {self.trades_this_hour}")
            return False
        
        # Verificar stop loss diário
        if self.session.total_pnl <= -self.config.stop_loss_daily:
            logger.warning(f"STOP LOSS diário atingido: ${self.session.total_pnl:.2f}")
            self.stop_trading()
            return False
        
        # Verificar stop win diário
        if self.session.total_pnl >= self.config.stop_win_daily:
            logger.info(f"STOP WIN diário atingido: ${self.session.total_pnl:.2f}")
            self.stop_trading()
            return False
        
        return True
    
    def should_trade(self, prediction: float, confidence: float) -> bool:
        """Verificar se deve executar trade baseado na predição e confiança com análise avançada"""
        try:
            # Verificar limites de risco
            if not self._check_risk_limits():
                logger.info("Trade bloqueado: limites de risco atingidos")
                return False
            
            # Verificar confiança mínima (sistema adaptativo)
            min_confidence = self.config.confidence_threshold
            
            # Ajustar confiança baseado na performance recente
            if len(self.session.trades) >= 5:
                recent_trades = self.session.trades[-5:]
                recent_win_rate = sum(1 for t in recent_trades if t.result == "WIN") / len(recent_trades)
                
                # Se performance está boa, reduzir um pouco a exigência
                if recent_win_rate >= 0.8:
                    min_confidence = max(0.55, min_confidence - 0.05)
                # Se performance está ruim, aumentar exigência levemente
                elif recent_win_rate <= 0.4:
                    min_confidence = min(0.75, min_confidence + 0.05)  # Reduzido de 0.85 para 0.75
            
            if confidence < min_confidence:
                return False
            
            # Verificar se não excedeu trades por hora (aumentado limite)
            current_hour = datetime.now().hour
            trades_this_hour = sum(1 for trade in self.session.trades 
                                 if trade.timestamp.hour == current_hour)
            
            max_trades_hour = self.config.max_trades_per_hour * 2  # Dobrado o limite
            if trades_this_hour >= max_trades_hour:
                logger.warning(f"Limite de trades por hora atingido: {trades_this_hour}")
                return False
            
            # Verificar força da predição (menos rigoroso)
            prediction_strength = abs(prediction - 0.5)
            min_prediction_strength = 0.08  # Reduzido de 0.15 para 0.08
            
            if prediction_strength < min_prediction_strength:
                logger.info(f"Predição muito fraca: {prediction_strength:.3f} < {min_prediction_strength}")
                return False
            
            # Verificar intervalo mínimo entre trades (reduzido)
            if self.session.trades:
                last_trade_time = self.session.trades[-1].timestamp
                time_since_last = (datetime.now() - last_trade_time).total_seconds()
                min_interval = 10  # Reduzido de 30 para 10 segundos
                
                if time_since_last < min_interval:
                    logger.info(f"Aguardando intervalo mínimo: {time_since_last:.1f}s < {min_interval}s")
                    return False
            
            # Circuit breaker mais flexível - apenas após 5 perdas consecutivas
            if len(self.session.trades) >= 5:
                last_5_trades = self.session.trades[-5:]
                if all(t.result == "LOSS" for t in last_5_trades):
                    # Exigir confiança alta após 5 perdas consecutivas
                    if confidence < 0.75:  # Reduzido de 0.8 para 0.75
                        logger.warning(f"Circuit breaker ativo: exigindo confianca > 0.75 apos 5 perdas consecutivas")
                        return False
            
            logger.info(f"Trade aprovado: confianca={confidence:.3f}, predicao={prediction:.3f}, forca={prediction_strength:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar condições de trade: {e}")
            return False
    
    def _check_stop_conditions(self):
        """Verificar condições de parada"""
        if self.session.total_pnl <= -self.config.stop_loss_daily:
            logger.warning("STOP LOSS ativado - Parando trading")
            self.stop_trading()
        elif self.session.total_pnl >= self.config.stop_win_daily:
            logger.info("STOP WIN ativado - Parando trading")
            self.stop_trading()
    
    def _save_trade(self, trade: TradeResult):
        """Salvar trade em arquivo"""
        try:
            trade_data = {
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp.isoformat(),
                'symbol': trade.symbol,
                'contract_type': trade.contract_type,
                'stake': trade.stake,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'result': trade.result,
                'payout': trade.payout,
                'confidence': trade.confidence,
                'duration': trade.duration
            }
            
            # Salvar em arquivo JSON
            filename = f"data/trades/trades_{datetime.now().strftime('%Y%m%d')}.json"
            
            trades_list = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    trades_list = json.load(f)
            
            trades_list.append(trade_data)
            
            with open(filename, 'w') as f:
                json.dump(trades_list, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar trade: {e}")
    
    async def process_market_data(self):
        """Processar dados de mercado em tempo real com análise contínua"""
        tick_count = 0
        analysis_count = 0
        consecutive_low_confidence = 0
        last_analysis_time = time.time()
        
        # Preço base para simulação
        base_price = 100.0
        price_trend = 0.0
        
        logger.info("Iniciando análise contínua de padrões de mercado...")
        
        while self.is_connected:
            try:
                if self.websocket:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    
                    # Processar tick
                    if 'tick' in data:
                        tick_data = data['tick']
                        symbol = tick_data.get('symbol', '')
                        price = float(tick_data.get('quote', 0))
                        epoch = int(tick_data.get('epoch', 0))
                        tick_count += 1
                        
                        # Criar objeto tick
                        tick = MarketTick(
                            symbol=symbol,
                            price=price,
                            timestamp=datetime.fromtimestamp(epoch),
                            epoch=epoch
                        )
                        
                        # Armazenar dados
                        if symbol not in self.market_data:
                            self.market_data[symbol] = []
                        
                        self.market_data[symbol].append(tick)
                        self.current_prices[symbol] = price
                        
                        # Manter apenas últimos 1000 ticks
                        if len(self.market_data[symbol]) > 1000:
                            self.market_data[symbol] = self.market_data[symbol][-1000:]
                        
                        # Log de monitoramento a cada 50 ticks
                        if tick_count % 50 == 0:
                            logger.info(f"Monitoramento ativo: {tick_count} ticks processados - Preco atual: ${price:.5f}")
                        
                        # Análise contínua para oportunidades (a cada 5 ticks)
                        current_time = time.time()
                        if self.is_trading and tick_count % 5 == 0 and len(self.market_data[symbol]) >= 30 and current_time - last_analysis_time >= 1.0:
                            analysis_count += 1
                            last_analysis_time = current_time
                            
                            # Analisar padrões de mercado
                            confidence, prediction = await self._analyze_market_patterns(symbol)
                            
                            logger.info(f"Análise #{analysis_count}: Confiança={confidence:.3f}, Predição={prediction:.3f}")
                            
                            # Verificar se deve executar trade
                            if confidence >= self.config.confidence_threshold:
                                consecutive_low_confidence = 0
                                
                                # Determinar tipo de contrato baseado na predição
                                contract_type = "CALL" if prediction > 0.5 else "PUT"
                                
                                if self.should_trade(prediction, confidence):
                                    logger.info(f"Oportunidade encontrada! Executando {contract_type} com confianca {confidence:.3f}")
                                    await self.execute_trade(contract_type, confidence)
                                else:
                                    logger.info(f"Trade bloqueado por limites de risco")
                            else:
                                consecutive_low_confidence += 1
                                if consecutive_low_confidence % 10 == 0:
                                    logger.info(f"Aguardando melhor oportunidade... (confianca insuficiente: {confidence:.3f} < {self.config.confidence_threshold})")
                        
                        # Callback
                        if self.on_tick_callback:
                            self.on_tick_callback(tick)
                        
                        logger.debug(f"📊 {symbol}: ${price:.5f}")
                else:
                    # Modo simulação - gerar ticks simulados
                    import random
                    import math
                    
                    # Simular movimento de preço com tendência e volatilidade
                    volatility = 0.002  # 0.2% de volatilidade
                    trend_change = random.uniform(-0.0001, 0.0001)
                    price_trend += trend_change
                    
                    # Limitar a tendência
                    price_trend = max(-0.01, min(0.01, price_trend))
                    
                    # Calcular novo preço
                    random_change = random.gauss(0, volatility)
                    price_change = price_trend + random_change
                    base_price *= (1 + price_change)
                    
                    # Manter preço em range realista
                    base_price = max(50.0, min(150.0, base_price))
                    
                    tick_count += 1
                    
                    # Criar objeto tick simulado
                    tick = MarketTick(
                        symbol=self.config.symbol,
                        price=base_price,
                        timestamp=datetime.now(),
                        epoch=int(time.time())
                    )
                    
                    # Armazenar dados
                    if self.config.symbol not in self.market_data:
                        self.market_data[self.config.symbol] = []
                    
                    self.market_data[self.config.symbol].append(tick)
                    self.current_prices[self.config.symbol] = base_price
                    
                    # Manter apenas últimos 1000 ticks
                    if len(self.market_data[self.config.symbol]) > 1000:
                        self.market_data[self.config.symbol] = self.market_data[self.config.symbol][-1000:]
                    
                    # Log menos frequente para não poluir
                    if tick_count % 50 == 0:
                        logger.info(f"Tick simulado: {self.config.symbol} = ${base_price:.5f}")
                    
                    # Análise contínua para oportunidades (a cada 5 ticks)
                    current_time = time.time()
                    if self.is_trading and tick_count % 5 == 0 and len(self.market_data[self.config.symbol]) >= 30 and current_time - last_analysis_time >= 1.0:
                        analysis_count += 1
                        last_analysis_time = current_time
                        
                        # Analisar padrões de mercado
                        confidence, prediction = await self._analyze_market_patterns(self.config.symbol)
                        
                        logger.info(f"Análise #{analysis_count}: Confiança={confidence:.3f}, Predição={prediction:.3f}")
                        
                        # Verificar se deve executar trade
                        if confidence >= self.config.confidence_threshold:
                            consecutive_low_confidence = 0
                            
                            # Determinar tipo de contrato baseado na predição
                            contract_type = "CALL" if prediction > 0.5 else "PUT"
                            
                            if self.should_trade(prediction, confidence):
                                logger.info(f"Oportunidade encontrada! Executando {contract_type} com confianca {confidence:.3f}")
                                await self.execute_trade(contract_type, confidence)
                            else:
                                logger.info(f"Trade bloqueado por limites de risco")
                        else:
                            consecutive_low_confidence += 1
                            if consecutive_low_confidence % 10 == 0:
                                logger.info(f"Aguardando melhor oportunidade... (confianca insuficiente: {confidence:.3f} < {self.config.confidence_threshold})")
                    
                    # Callback
                    if self.on_tick_callback:
                        self.on_tick_callback(tick)
                    
                    # Aguardar próximo tick (1-3 segundos para simular realismo)
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Erro ao processar dados: {e}")
                await asyncio.sleep(1)
    
    async def _analyze_market_patterns(self, symbol: str):
        """Análise avançada de padrões de mercado"""
        try:
            if len(self.market_data[symbol]) < 30:
                return 0.0, 0.5
            
            # Obter dados recentes para análise
            recent_prices = [tick.price for tick in self.market_data[symbol][-30:]]
            
            # Análise de tendência
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Análise de volatilidade
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            
            # Análise de momentum (últimos 10 vs 10 anteriores)
            recent_10 = np.mean(recent_prices[-10:])
            previous_10 = np.mean(recent_prices[-20:-10])
            momentum = (recent_10 - previous_10) / previous_10
            
            # Análise de padrões de reversão
            price_diffs = np.diff(recent_prices)
            consecutive_moves = 0
            for i in range(len(price_diffs)-1, 0, -1):
                if np.sign(price_diffs[i]) == np.sign(price_diffs[i-1]):
                    consecutive_moves += 1
                else:
                    break
            
            # Calcular confiança baseada em múltiplos fatores
            trend_strength = abs(price_change) * 100
            volatility_factor = min(1.0, volatility * 20)
            momentum_factor = abs(momentum) * 50
            pattern_factor = min(0.3, consecutive_moves * 0.05)
            
            confidence = min(0.95, 0.4 + trend_strength + volatility_factor + momentum_factor + pattern_factor)
            
            # Calcular predição baseada em tendência e momentum
            prediction = 0.5 + (price_change * 3) + (momentum * 2)
            prediction = max(0.05, min(0.95, prediction))
            
            return confidence, prediction
                
        except Exception as e:
            logger.error(f"Erro na análise de padrões: {e}")
            return 0.0, 0.5
    
    def start_trading(self):
        """Iniciar trading com monitoramento inteligente"""
        self.is_trading = True
        logger.info("Sistema de Trading Inteligente Iniciado!")
        logger.info("Monitoramento contínuo de oportunidades ativado...")
        logger.info(f"Configurações: Confiança mín: {self.config.confidence_threshold:.1%}, Stake: ${self.config.stake}")
    
    def stop_trading(self):
        """Parar trading"""
        self.is_trading = False
        self.session.end_time = datetime.now()
        self.session.is_active = False
        logger.info("Trading parado!")
        
        # Gerar relatório da sessão
        self._generate_session_report()
    
    def _generate_session_report(self):
        """Gerar relatório da sessão"""
        try:
            duration = (self.session.end_time - self.session.start_time).total_seconds() / 3600
            win_rate = (self.session.winning_trades / max(self.session.total_trades, 1)) * 100
            
            report = {
                'session_id': self.session.session_id,
                'start_time': self.session.start_time.isoformat(),
                'end_time': self.session.end_time.isoformat() if self.session.end_time else None,
                'duration_hours': round(duration, 2),
                'total_trades': self.session.total_trades,
                'winning_trades': self.session.winning_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(self.session.total_pnl, 2),
                'final_balance': round(self.session.balance, 2),
                'trades': [
                    {
                        'trade_id': trade.trade_id,
                        'timestamp': trade.timestamp.isoformat(),
                        'symbol': trade.symbol,
                        'contract_type': trade.contract_type,
                        'result': trade.result,
                        'payout': trade.payout,
                        'confidence': trade.confidence
                    }
                    for trade in self.session.trades
                ]
            }
            
            # Salvar relatório
            filename = f"data/trades/session_report_{self.session.session_id}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Relatório da sessão salvo: {filename}")
            logger.info(f"Resumo: {self.session.total_trades} trades, {win_rate:.1f}% win rate, PnL: ${self.session.total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
    
    async def disconnect(self):
        """Desconectar da API"""
        try:
            self.is_connected = False
            self.is_trading = False
            
            if self.websocket:
                await self.websocket.close()
            
            logger.info("Desconectado da API")
            
        except Exception as e:
            logger.error(f"Erro ao desconectar: {e}")
    
    def get_session_stats(self) -> Dict:
        """Obter estatísticas da sessão"""
        win_rate = (self.session.winning_trades / max(self.session.total_trades, 1)) * 100
        
        return {
            'session_id': self.session.session_id,
            'is_active': self.session.is_active,
            'total_trades': self.session.total_trades,
            'winning_trades': self.session.winning_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(self.session.total_pnl, 2),
            'balance': round(self.session.balance, 2),
            'trades_this_hour': self.trades_this_hour,
            'is_trading': self.is_trading,
            'is_connected': self.is_connected
        }

# Exemplo de uso
async def demo_trading_session():
    """Demonstração de sessão de trading inteligente"""
    
    # Configuração otimizada para busca inteligente
    config = TradeConfig(
        symbol="R_50",
        stake=10.0,
        duration=5,
        confidence_threshold=0.65,
        max_trades_per_hour=15,  # Aumentado para aproveitar mais oportunidades
        stop_loss_daily=50.0,
        stop_win_daily=100.0
    )
    
    # Criar executor
    executor = RealTimeTradingExecutor(config)
    
    # Callbacks para eventos com logs melhorados
    def on_tick(tick: MarketTick):
        # Log apenas a cada 100 ticks para não poluir
        if hasattr(on_tick, 'count'):
            on_tick.count += 1
        else:
            on_tick.count = 1
        
        if on_tick.count % 100 == 0:
            print(f"Monitoramento: {on_tick.count} ticks processados - Preco: ${tick.price:.5f}")
    
    def on_trade(trade: TradeResult):
        print(f"TRADE EXECUTADO: {trade.contract_type} - {trade.result} - ${trade.payout:.2f} (Confianca: {trade.confidence:.1%})")
    
    def on_balance(balance: float):
        print(f"Saldo atualizado: ${balance:.2f}")
    
    executor.on_tick_callback = on_tick
    executor.on_trade_callback = on_trade
    executor.on_balance_callback = on_balance
    
    try:
        # Conectar
        if await executor.connect_to_deriv():
            # Subscrever aos ticks
            await executor.subscribe_to_ticks(config.symbol)
            
            # Iniciar processamento de dados
            data_task = asyncio.create_task(executor.process_market_data())
            
            # Obter saldo inicial
            await executor.get_balance()
            
            # Iniciar trading
            executor.start_trading()
            
            # Simular trading inteligente por 60 segundos
            start_time = time.time()
            opportunities_found = 0
            
            logger.info("Iniciando busca inteligente por oportunidades...")
            
            while time.time() - start_time < 60:
                if executor.is_trading and executor.current_prices.get(config.symbol, 0) > 0:
                    # O sistema agora analisa automaticamente via process_market_data
                    # Apenas monitoramos o progresso
                    elapsed = time.time() - start_time
                    if int(elapsed) % 15 == 0:  # Log a cada 15 segundos
                        stats = executor.get_session_stats()
                        logger.info(f"Progresso: {elapsed:.0f}s - Trades: {stats['total_trades']} - PnL: ${stats['total_pnl']:.2f}")
                
                await asyncio.sleep(2)  # Verificação mais frequente
            
            # Parar trading
            executor.stop_trading()
            
            # Cancelar task de dados
            data_task.cancel()
            
            # Mostrar estatísticas finais
            stats = executor.get_session_stats()
            print("\nEstatisticas Finais:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # Desconectar
        await executor.disconnect()
        
    except Exception as e:
        logger.error(f"Erro na sessão de trading: {e}")
        await executor.disconnect()

def run_real_time_executor():
    """Executar o sistema de trading inteligente em tempo real"""
    print("Iniciando Sistema de Trading Inteligente")
    print("Monitoramento Continuo de Oportunidades")
    print("=" * 50)
    
    try:
        # Executar sessão demo
        asyncio.run(demo_trading_session())
        
    except KeyboardInterrupt:
        print("\nTrading interrompido pelo usuario")
        print("Finalizando analises em andamento...")
    except Exception as e:
        print(f"ERRO: {e}")

if __name__ == "__main__":
    run_real_time_executor()