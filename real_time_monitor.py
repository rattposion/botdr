#!/usr/bin/env python3
"""
Monitor de Trading em Tempo Real
Sistema integrado para monitoramento de todas as estratégias e análises
"""

import pandas as pd
import numpy as np
import logging
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Estrutura para sinais de trading"""
    symbol: str
    signal_type: str  # CALL, PUT, HOLD
    confidence: float
    source: str  # ensemble, multi_timeframe, symbol_specific
    timestamp: datetime
    price: float
    indicators: Dict[str, float]
    risk_level: str  # low, medium, high

@dataclass
class MarketStatus:
    """Status do mercado"""
    symbol: str
    current_price: float
    volatility: float
    trend: str  # bullish, bearish, neutral
    volume: int
    last_update: datetime

@dataclass
class PerformanceMetrics:
    """Métricas de performance"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_profit: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float

class RealTimeMonitor:
    """Monitor de trading em tempo real"""
    
    def __init__(self):
        """Inicializa o monitor"""
        self.symbols = ["R_50", "R_100", "R_25", "R_75"]
        self.is_running = False
        self.signals_history = []
        self.market_status = {}
        self.performance_metrics = {}
        
        # Configurar diretórios
        self.monitor_dir = "real_time_monitoring"
        self.logs_dir = os.path.join(self.monitor_dir, "logs")
        self.signals_dir = os.path.join(self.monitor_dir, "signals")
        
        for dir_path in [self.monitor_dir, self.logs_dir, self.signals_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Configurações de trading
        self.trading_config = {
            "min_confidence": 0.65,
            "max_trades_per_hour": 10,
            "risk_per_trade": 0.02,  # 2% do capital
            "stop_loss": 0.05,       # 5%
            "take_profit": 0.10,     # 10%
            "max_daily_loss": 0.20   # 20%
        }
        
        # Carregar resultados das análises anteriores
        self.ensemble_results = self.load_analysis_results("ensemble_test_results")
        self.multi_timeframe_results = self.load_analysis_results("multi_timeframe_results")
        self.symbol_optimization_results = self.load_analysis_results("symbol_optimization_results")
        
        logger.info("RealTimeMonitor inicializado")
    
    def load_analysis_results(self, results_dir: str) -> Dict:
        """
        Carrega resultados de análises anteriores
        
        Args:
            results_dir: Diretório dos resultados
            
        Returns:
            Resultados carregados
        """
        try:
            if not os.path.exists(results_dir):
                return {}
            
            # Encontrar arquivo mais recente
            files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            if not files:
                return {}
            
            latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
            file_path = os.path.join(results_dir, latest_file)
            
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Resultados carregados de {results_dir}: {latest_file}")
            return results
            
        except Exception as e:
            logger.error(f"Erro ao carregar resultados de {results_dir}: {e}")
            return {}
    
    def generate_market_data(self, symbol: str) -> MarketStatus:
        """
        Gera dados de mercado simulados
        
        Args:
            symbol: Símbolo
            
        Returns:
            Status do mercado
        """
        try:
            base_price = float(symbol.split('_')[1])
            
            # Simular variação de preço
            if symbol in self.market_status:
                last_price = self.market_status[symbol].current_price
                price_change = np.random.normal(0, 0.001)  # 0.1% de volatilidade
                current_price = last_price * (1 + price_change)
            else:
                current_price = base_price + np.random.normal(0, 0.5)
            
            # Simular outros dados
            volatility = abs(np.random.normal(0.02, 0.005))
            volume = int(np.random.normal(5000, 1000))
            
            # Determinar tendência
            if symbol in self.market_status:
                price_history = getattr(self.market_status[symbol], 'price_history', [current_price])
                price_history.append(current_price)
                if len(price_history) > 20:
                    price_history = price_history[-20:]
                
                recent_trend = (price_history[-1] - price_history[0]) / price_history[0]
                if recent_trend > 0.002:
                    trend = "bullish"
                elif recent_trend < -0.002:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"
                price_history = [current_price]
            
            market_status = MarketStatus(
                symbol=symbol,
                current_price=current_price,
                volatility=volatility,
                trend=trend,
                volume=volume,
                last_update=datetime.now()
            )
            
            # Armazenar histórico de preços
            market_status.price_history = price_history
            
            return market_status
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados de mercado para {symbol}: {e}")
            return None
    
    def analyze_ensemble_signal(self, symbol: str, market_data: MarketStatus) -> Optional[TradingSignal]:
        """
        Analisa sinal baseado nos resultados ensemble
        
        Args:
            symbol: Símbolo
            market_data: Dados de mercado
            
        Returns:
            Sinal de trading ou None
        """
        try:
            if not self.ensemble_results or 'summary' not in self.ensemble_results:
                return None
            
            summary = self.ensemble_results['summary']
            
            # Simular análise ensemble baseada nos resultados
            avg_accuracy = summary.get('average_accuracy', 0.5)
            avg_win_rate = summary.get('average_win_rate', 0.5)
            
            # Gerar sinal baseado na performance histórica
            confidence = min(avg_accuracy * avg_win_rate, 0.95)
            
            # Determinar tipo de sinal baseado na tendência
            if market_data.trend == "bullish" and confidence > self.trading_config["min_confidence"]:
                signal_type = "CALL"
            elif market_data.trend == "bearish" and confidence > self.trading_config["min_confidence"]:
                signal_type = "PUT"
            else:
                signal_type = "HOLD"
                confidence = 0.5
            
            # Indicadores simulados
            indicators = {
                "ensemble_accuracy": avg_accuracy,
                "ensemble_win_rate": avg_win_rate,
                "market_volatility": market_data.volatility,
                "trend_strength": abs(np.random.normal(0.5, 0.2))
            }
            
            # Nível de risco baseado na volatilidade
            if market_data.volatility > 0.03:
                risk_level = "high"
            elif market_data.volatility > 0.015:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                source="ensemble",
                timestamp=datetime.now(),
                price=market_data.current_price,
                indicators=indicators,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Erro na análise ensemble para {symbol}: {e}")
            return None
    
    def analyze_multi_timeframe_signal(self, symbol: str, market_data: MarketStatus) -> Optional[TradingSignal]:
        """
        Analisa sinal baseado na análise multi-timeframe
        
        Args:
            symbol: Símbolo
            market_data: Dados de mercado
            
        Returns:
            Sinal de trading ou None
        """
        try:
            if not self.multi_timeframe_results or 'summary' not in self.multi_timeframe_results:
                return None
            
            summary = self.multi_timeframe_results['summary']
            consensus_signals = summary.get('consensus_signals', {})
            
            if symbol not in consensus_signals:
                return None
            
            symbol_signal = consensus_signals[symbol]
            signal_name = symbol_signal.get('signal_name', 'HOLD')
            confidence = symbol_signal.get('confidence', 0.5)
            
            # Ajustar confiança baseada na correlação
            correlation_stats = summary.get('correlation_stats', {})
            avg_correlation = correlation_stats.get('average_correlation', 0.5)
            
            # Aumentar confiança se correlações são altas
            if avg_correlation > 0.8:
                confidence = min(confidence * 1.2, 0.95)
            
            indicators = {
                "multi_timeframe_confidence": confidence,
                "average_correlation": avg_correlation,
                "consensus_signal": signal_name,
                "market_trend": market_data.trend
            }
            
            # Determinar nível de risco
            if confidence > 0.8:
                risk_level = "low"
            elif confidence > 0.6:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_name,
                confidence=confidence,
                source="multi_timeframe",
                timestamp=datetime.now(),
                price=market_data.current_price,
                indicators=indicators,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Erro na análise multi-timeframe para {symbol}: {e}")
            return None
    
    def analyze_symbol_specific_signal(self, symbol: str, market_data: MarketStatus) -> Optional[TradingSignal]:
        """
        Analisa sinal baseado na otimização específica do símbolo
        
        Args:
            symbol: Símbolo
            market_data: Dados de mercado
            
        Returns:
            Sinal de trading ou None
        """
        try:
            if not self.symbol_optimization_results or 'optimization_results' not in self.symbol_optimization_results:
                return None
            
            optimization_results = self.symbol_optimization_results['optimization_results']
            
            if symbol not in optimization_results:
                return None
            
            symbol_result = optimization_results[symbol]
            best_accuracy = symbol_result.get('best_accuracy', 0.5)
            best_model = symbol_result.get('best_model', 'Unknown')
            
            # Simular predição do modelo otimizado
            confidence = best_accuracy
            
            # Determinar sinal baseado no modelo e tendência
            if best_model == "LogisticRegression" and confidence > self.trading_config["min_confidence"]:
                if market_data.trend == "bullish":
                    signal_type = "CALL"
                elif market_data.trend == "bearish":
                    signal_type = "PUT"
                else:
                    signal_type = "HOLD"
                    confidence = 0.5
            else:
                signal_type = "HOLD"
                confidence = 0.5
            
            indicators = {
                "model_accuracy": best_accuracy,
                "best_model": best_model,
                "optimization_confidence": confidence,
                "symbol_trend": market_data.trend
            }
            
            # Nível de risco baseado na acurácia do modelo
            if best_accuracy > 0.8:
                risk_level = "low"
            elif best_accuracy > 0.6:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                source="symbol_specific",
                timestamp=datetime.now(),
                price=market_data.current_price,
                indicators=indicators,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Erro na análise específica para {symbol}: {e}")
            return None
    
    def generate_consensus_signal(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """
        Gera sinal de consenso entre diferentes análises
        
        Args:
            signals: Lista de sinais
            
        Returns:
            Sinal de consenso
        """
        try:
            if not signals:
                return None
            
            symbol = signals[0].symbol
            
            # Pesos por fonte
            source_weights = {
                "ensemble": 0.4,
                "multi_timeframe": 0.35,
                "symbol_specific": 0.25
            }
            
            # Calcular consenso ponderado
            weighted_signals = {"CALL": 0, "PUT": 0, "HOLD": 0}
            total_weight = 0
            combined_confidence = 0
            combined_indicators = {}
            
            for signal in signals:
                weight = source_weights.get(signal.source, 0.1)
                weighted_signals[signal.signal_type] += signal.confidence * weight
                combined_confidence += signal.confidence * weight
                total_weight += weight
                
                # Combinar indicadores
                for key, value in signal.indicators.items():
                    if key not in combined_indicators:
                        combined_indicators[key] = []
                    combined_indicators[key].append(value)
            
            # Normalizar
            if total_weight > 0:
                for signal_type in weighted_signals:
                    weighted_signals[signal_type] /= total_weight
                combined_confidence /= total_weight
            
            # Determinar sinal final
            consensus_signal_type = max(weighted_signals.keys(), key=lambda k: weighted_signals[k])
            consensus_confidence = weighted_signals[consensus_signal_type]
            
            # Se não há consenso claro, manter HOLD
            if consensus_confidence < self.trading_config["min_confidence"]:
                consensus_signal_type = "HOLD"
                consensus_confidence = 0.5
            
            # Calcular indicadores médios
            avg_indicators = {}
            for key, values in combined_indicators.items():
                if isinstance(values[0], (int, float)):
                    avg_indicators[key] = np.mean(values)
                else:
                    avg_indicators[key] = values[0]  # Usar primeiro valor para strings
            
            # Determinar nível de risco
            risk_levels = [s.risk_level for s in signals]
            if "high" in risk_levels:
                consensus_risk = "high"
            elif "medium" in risk_levels:
                consensus_risk = "medium"
            else:
                consensus_risk = "low"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=consensus_signal_type,
                confidence=consensus_confidence,
                source="consensus",
                timestamp=datetime.now(),
                price=signals[0].price,
                indicators=avg_indicators,
                risk_level=consensus_risk
            )
            
        except Exception as e:
            logger.error(f"Erro ao gerar consenso: {e}")
            return None
    
    def monitor_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """
        Monitora um símbolo específico
        
        Args:
            symbol: Símbolo para monitorar
            
        Returns:
            Sinal de trading ou None
        """
        try:
            # Gerar dados de mercado
            market_data = self.generate_market_data(symbol)
            if not market_data:
                return None
            
            # Atualizar status do mercado
            self.market_status[symbol] = market_data
            
            # Analisar sinais de diferentes fontes
            signals = []
            
            # Sinal ensemble
            ensemble_signal = self.analyze_ensemble_signal(symbol, market_data)
            if ensemble_signal:
                signals.append(ensemble_signal)
            
            # Sinal multi-timeframe
            mt_signal = self.analyze_multi_timeframe_signal(symbol, market_data)
            if mt_signal:
                signals.append(mt_signal)
            
            # Sinal específico do símbolo
            specific_signal = self.analyze_symbol_specific_signal(symbol, market_data)
            if specific_signal:
                signals.append(specific_signal)
            
            # Gerar consenso
            if signals:
                consensus_signal = self.generate_consensus_signal(signals)
                if consensus_signal:
                    self.signals_history.append(consensus_signal)
                    
                    # Salvar sinal
                    self.save_signal(consensus_signal)
                    
                    return consensus_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao monitorar {symbol}: {e}")
            return None
    
    def save_signal(self, signal: TradingSignal):
        """
        Salva sinal em arquivo
        
        Args:
            signal: Sinal para salvar
        """
        try:
            signal_file = os.path.join(
                self.signals_dir,
                f"signals_{datetime.now().strftime('%Y%m%d')}.json"
            )
            
            signal_data = asdict(signal)
            signal_data['timestamp'] = signal.timestamp.isoformat()
            
            # Carregar sinais existentes
            if os.path.exists(signal_file):
                with open(signal_file, 'r') as f:
                    signals = json.load(f)
            else:
                signals = []
            
            signals.append(signal_data)
            
            # Salvar
            with open(signal_file, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Erro ao salvar sinal: {e}")
    
    def run_monitoring_cycle(self):
        """Executa um ciclo de monitoramento"""
        try:
            logger.info("Executando ciclo de monitoramento...")
            
            active_signals = []
            
            for symbol in self.symbols:
                signal = self.monitor_symbol(symbol)
                if signal and signal.signal_type != "HOLD":
                    active_signals.append(signal)
                    logger.info(f"Sinal ativo: {symbol} - {signal.signal_type} (Confiança: {signal.confidence:.2%})")
            
            # Relatório do ciclo
            if active_signals:
                print(f"\n🚨 SINAIS ATIVOS ({datetime.now().strftime('%H:%M:%S')})")
                print("-" * 60)
                for signal in active_signals:
                    print(f"📊 {signal.symbol}: {signal.signal_type} | Confiança: {signal.confidence:.1%} | Risco: {signal.risk_level}")
                    print(f"   💰 Preço: {signal.price:.4f} | Fonte: {signal.source}")
                print("-" * 60)
            else:
                print(f"⏳ Aguardando sinais... ({datetime.now().strftime('%H:%M:%S')})")
            
            return active_signals
            
        except Exception as e:
            logger.error(f"Erro no ciclo de monitoramento: {e}")
            return []
    
    def start_monitoring(self, duration_minutes: int = 60):
        """
        Inicia monitoramento em tempo real
        
        Args:
            duration_minutes: Duração em minutos
        """
        try:
            self.is_running = True
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            print("\n" + "="*80)
            print("🔴 MONITOR DE TRADING EM TEMPO REAL INICIADO")
            print("="*80)
            print(f"⏰ Duração: {duration_minutes} minutos")
            print(f"📊 Símbolos: {', '.join(self.symbols)}")
            print(f"🎯 Confiança mínima: {self.trading_config['min_confidence']:.0%}")
            print("="*80)
            
            cycle_count = 0
            total_signals = 0
            
            while self.is_running and datetime.now() < end_time:
                cycle_count += 1
                
                # Executar ciclo
                active_signals = self.run_monitoring_cycle()
                total_signals += len(active_signals)
                
                # Aguardar próximo ciclo (30 segundos)
                time.sleep(30)
            
            print("\n" + "="*80)
            print("⏹️ MONITORAMENTO FINALIZADO")
            print("="*80)
            print(f"📊 Ciclos executados: {cycle_count}")
            print(f"🚨 Total de sinais: {total_signals}")
            print(f"📁 Logs salvos em: {self.logs_dir}")
            print(f"📈 Sinais salvos em: {self.signals_dir}")
            print("="*80)
            
        except KeyboardInterrupt:
            print("\n⏹️ Monitoramento interrompido pelo usuário")
        except Exception as e:
            logger.error(f"Erro no monitoramento: {e}")
        finally:
            self.is_running = False


def run_real_time_monitoring():
    """Função principal para executar monitoramento em tempo real"""
    try:
        # Configurar logging
        setup_logging()
        
        # Criar monitor
        monitor = RealTimeMonitor()
        
        # Iniciar monitoramento (5 minutos para demonstração)
        monitor.start_monitoring(duration_minutes=5)
        
    except Exception as e:
        logger.error(f"Erro no monitoramento em tempo real: {e}")
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    run_real_time_monitoring()