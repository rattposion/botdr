"""
Sistema Avançado de Gerenciamento de Risco para Trading Automatizado
Implementa múltiplas camadas de proteção e controle de risco
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RiskLimits:
    """Configurações de limites de risco"""
    max_daily_loss: float = 100.0  # Perda máxima diária
    max_consecutive_losses: int = 5  # Máximo de perdas consecutivas
    max_position_size: float = 10.0  # Tamanho máximo da posição
    max_drawdown: float = 20.0  # Drawdown máximo (%)
    max_trades_per_hour: int = 10  # Máximo de trades por hora
    max_trades_per_day: int = 50  # Máximo de trades por dia
    min_win_rate: float = 0.4  # Taxa de vitória mínima
    max_risk_per_trade: float = 2.0  # Risco máximo por trade (%)
    volatility_threshold: float = 0.05  # Limite de volatilidade
    correlation_limit: float = 0.8  # Limite de correlação entre trades

@dataclass
class TradeRecord:
    """Registro de trade para análise de risco"""
    timestamp: datetime
    symbol: str
    direction: str  # CALL/PUT
    stake: float
    result: str  # WIN/LOSS
    pnl: float
    confidence: float
    volatility: float

@dataclass
class RiskMetrics:
    """Métricas de risco calculadas"""
    current_drawdown: float
    consecutive_losses: int
    daily_pnl: float
    hourly_trades: int
    daily_trades: int
    win_rate: float
    avg_volatility: float
    portfolio_correlation: float
    risk_score: float

class RiskManager:
    """Sistema avançado de gerenciamento de risco"""
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.trade_history: List[TradeRecord] = []
        self.daily_stats = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def add_trade(self, trade: TradeRecord):
        """Adicionar trade ao histórico"""
        self.trade_history.append(trade)
        self.update_daily_stats(trade)
        
    def update_daily_stats(self, trade: TradeRecord):
        """Atualizar estatísticas diárias"""
        date_key = trade.timestamp.date()
        
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'trades': 0,
                'pnl': 0.0,
                'wins': 0,
                'losses': 0
            }
            
        stats = self.daily_stats[date_key]
        stats['trades'] += 1
        stats['pnl'] += trade.pnl
        
        if trade.result == 'WIN':
            stats['wins'] += 1
        else:
            stats['losses'] += 1
            
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calcular métricas de risco atuais"""
        if not self.trade_history:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
            
        now = datetime.now()
        today = now.date()
        hour_ago = now - timedelta(hours=1)
        
        # Trades recentes
        recent_trades = [t for t in self.trade_history if t.timestamp.date() == today]
        hourly_trades = [t for t in self.trade_history if t.timestamp >= hour_ago]
        
        # Calcular drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in self.trade_history])
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (peak - cumulative_pnl) / np.maximum(peak, 1) * 100
        current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0
        
        # Perdas consecutivas
        consecutive_losses = 0
        for trade in reversed(self.trade_history):
            if trade.result == 'LOSS':
                consecutive_losses += 1
            else:
                break
                
        # PnL diário
        daily_pnl = sum(t.pnl for t in recent_trades)
        
        # Taxa de vitória
        if recent_trades:
            wins = sum(1 for t in recent_trades if t.result == 'WIN')
            win_rate = wins / len(recent_trades)
        else:
            win_rate = 0
            
        # Volatilidade média
        if recent_trades:
            avg_volatility = np.mean([t.volatility for t in recent_trades])
        else:
            avg_volatility = 0
            
        # Correlação do portfólio (simplificada)
        portfolio_correlation = self.calculate_portfolio_correlation()
        
        # Score de risco (0-100)
        risk_score = self.calculate_risk_score(
            current_drawdown, consecutive_losses, daily_pnl,
            len(hourly_trades), len(recent_trades), win_rate,
            avg_volatility, portfolio_correlation
        )
        
        return RiskMetrics(
            current_drawdown=current_drawdown,
            consecutive_losses=consecutive_losses,
            daily_pnl=daily_pnl,
            hourly_trades=len(hourly_trades),
            daily_trades=len(recent_trades),
            win_rate=win_rate,
            avg_volatility=avg_volatility,
            portfolio_correlation=portfolio_correlation,
            risk_score=risk_score
        )
        
    def calculate_portfolio_correlation(self) -> float:
        """Calcular correlação do portfólio"""
        if len(self.trade_history) < 10:
            return 0.0
            
        # Agrupar por símbolo
        symbols = {}
        for trade in self.trade_history[-50:]:  # Últimos 50 trades
            if trade.symbol not in symbols:
                symbols[trade.symbol] = []
            symbols[trade.symbol].append(trade.pnl)
            
        if len(symbols) < 2:
            return 0.0
            
        # Calcular correlação média entre símbolos
        correlations = []
        symbol_list = list(symbols.keys())
        
        for i in range(len(symbol_list)):
            for j in range(i + 1, len(symbol_list)):
                s1_pnl = symbols[symbol_list[i]]
                s2_pnl = symbols[symbol_list[j]]
                
                min_len = min(len(s1_pnl), len(s2_pnl))
                if min_len > 3:
                    corr = np.corrcoef(s1_pnl[-min_len:], s2_pnl[-min_len:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                        
        return np.mean(correlations) if correlations else 0.0
        
    def calculate_risk_score(self, drawdown: float, consecutive_losses: int,
                           daily_pnl: float, hourly_trades: int, daily_trades: int,
                           win_rate: float, volatility: float, correlation: float) -> float:
        """Calcular score de risco (0-100)"""
        score = 0
        
        # Drawdown (0-25 pontos)
        score += min(25, (drawdown / self.limits.max_drawdown) * 25)
        
        # Perdas consecutivas (0-20 pontos)
        score += min(20, (consecutive_losses / self.limits.max_consecutive_losses) * 20)
        
        # PnL diário negativo (0-15 pontos)
        if daily_pnl < 0:
            score += min(15, (abs(daily_pnl) / self.limits.max_daily_loss) * 15)
            
        # Overtrading (0-15 pontos)
        score += min(10, (hourly_trades / self.limits.max_trades_per_hour) * 10)
        score += min(5, (daily_trades / self.limits.max_trades_per_day) * 5)
        
        # Taxa de vitória baixa (0-10 pontos)
        if win_rate < self.limits.min_win_rate:
            score += min(10, ((self.limits.min_win_rate - win_rate) / self.limits.min_win_rate) * 10)
            
        # Volatilidade alta (0-10 pontos)
        score += min(10, (volatility / self.limits.volatility_threshold) * 10)
        
        # Correlação alta (0-5 pontos)
        score += min(5, (correlation / self.limits.correlation_limit) * 5)
        
        return min(100, score)
        
    def check_trade_permission(self, symbol: str, stake: float, 
                             confidence: float, volatility: float) -> Tuple[bool, str]:
        """Verificar se o trade é permitido"""
        metrics = self.calculate_risk_metrics()
        
        # Verificar limites críticos
        if metrics.current_drawdown > self.limits.max_drawdown:
            return False, f"Drawdown máximo excedido: {metrics.current_drawdown:.1f}%"
            
        if metrics.consecutive_losses >= self.limits.max_consecutive_losses:
            return False, f"Muitas perdas consecutivas: {metrics.consecutive_losses}"
            
        if abs(metrics.daily_pnl) > self.limits.max_daily_loss:
            return False, f"Perda diária máxima excedida: {metrics.daily_pnl:.2f}"
            
        if metrics.hourly_trades >= self.limits.max_trades_per_hour:
            return False, f"Limite de trades por hora excedido: {metrics.hourly_trades}"
            
        if metrics.daily_trades >= self.limits.max_trades_per_day:
            return False, f"Limite de trades diários excedido: {metrics.daily_trades}"
            
        if stake > self.limits.max_position_size:
            return False, f"Tamanho da posição muito grande: {stake}"
            
        if volatility > self.limits.volatility_threshold:
            return False, f"Volatilidade muito alta: {volatility:.3f}"
            
        # Verificar score de risco
        if metrics.risk_score > 80:
            return False, f"Score de risco muito alto: {metrics.risk_score:.1f}"
            
        # Verificar taxa de vitória
        if metrics.win_rate < self.limits.min_win_rate and metrics.daily_trades > 10:
            return False, f"Taxa de vitória muito baixa: {metrics.win_rate:.2f}"
            
        return True, "Trade aprovado"
        
    def suggest_position_size(self, confidence: float, volatility: float) -> float:
        """Sugerir tamanho da posição baseado no risco"""
        metrics = self.calculate_risk_metrics()
        
        # Tamanho base
        base_size = self.limits.max_position_size * 0.1
        
        # Ajustar por confiança
        confidence_multiplier = min(2.0, confidence * 2)
        
        # Ajustar por volatilidade (inverso)
        volatility_multiplier = max(0.5, 1 - (volatility / self.limits.volatility_threshold))
        
        # Ajustar por score de risco (inverso)
        risk_multiplier = max(0.3, 1 - (metrics.risk_score / 100))
        
        # Ajustar por drawdown
        drawdown_multiplier = max(0.5, 1 - (metrics.current_drawdown / self.limits.max_drawdown))
        
        suggested_size = base_size * confidence_multiplier * volatility_multiplier * risk_multiplier * drawdown_multiplier
        
        return min(self.limits.max_position_size, max(1.0, suggested_size))
        
    def generate_risk_report(self) -> Dict:
        """Gerar relatório de risco"""
        metrics = self.calculate_risk_metrics()
        
        # Análise de tendências
        recent_trades = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
        
        trend_analysis = {
            'improving': False,
            'stable': False,
            'deteriorating': False
        }
        
        if len(recent_trades) >= 10:
            first_half = recent_trades[:len(recent_trades)//2]
            second_half = recent_trades[len(recent_trades)//2:]
            
            first_win_rate = sum(1 for t in first_half if t.result == 'WIN') / len(first_half)
            second_win_rate = sum(1 for t in second_half if t.result == 'WIN') / len(second_half)
            
            if second_win_rate > first_win_rate + 0.1:
                trend_analysis['improving'] = True
            elif abs(second_win_rate - first_win_rate) <= 0.1:
                trend_analysis['stable'] = True
            else:
                trend_analysis['deteriorating'] = True
                
        # Recomendações
        recommendations = []
        
        if metrics.risk_score > 70:
            recommendations.append("Reduzir tamanho das posições")
            recommendations.append("Pausar trading por algumas horas")
            
        if metrics.consecutive_losses >= 3:
            recommendations.append("Revisar estratégia de entrada")
            recommendations.append("Considerar pausa no trading")
            
        if metrics.win_rate < 0.4 and metrics.daily_trades > 5:
            recommendations.append("Aumentar critério de confiança mínima")
            recommendations.append("Focar em setups de maior qualidade")
            
        if metrics.avg_volatility > self.limits.volatility_threshold * 0.8:
            recommendations.append("Evitar mercados muito voláteis")
            recommendations.append("Reduzir exposição durante alta volatilidade")
            
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics),
            'limits': asdict(self.limits),
            'trend_analysis': trend_analysis,
            'recommendations': recommendations,
            'total_trades': len(self.trade_history),
            'daily_stats': {str(k): v for k, v in self.daily_stats.items()}
        }
        
    def save_risk_report(self, filename: str = None):
        """Salvar relatório de risco"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_report_{timestamp}.json"
            
        report = self.generate_risk_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Relatório de risco salvo: {filename}")
        return filename

def run_risk_analysis():
    """Executar análise de risco com dados simulados"""
    print("🛡️ Iniciando Sistema Avançado de Gerenciamento de Risco")
    
    # Configurar limites personalizados
    limits = RiskLimits(
        max_daily_loss=50.0,
        max_consecutive_losses=3,
        max_position_size=5.0,
        max_drawdown=15.0,
        max_trades_per_hour=5,
        max_trades_per_day=25,
        min_win_rate=0.45
    )
    
    risk_manager = RiskManager(limits)
    
    # Simular histórico de trades
    symbols = ['R_10', 'R_25', 'R_50', 'R_75']
    
    print("\n📊 Simulando histórico de trades...")
    
    for i in range(100):
        symbol = np.random.choice(symbols)
        direction = np.random.choice(['CALL', 'PUT'])
        stake = np.random.uniform(1.0, 5.0)
        confidence = np.random.uniform(0.5, 0.9)
        volatility = np.random.uniform(0.01, 0.08)
        
        # Simular resultado baseado na confiança
        win_prob = confidence * 0.8  # Ajustar probabilidade
        result = 'WIN' if np.random.random() < win_prob else 'LOSS'
        pnl = stake * 0.8 if result == 'WIN' else -stake
        
        trade = TradeRecord(
            timestamp=datetime.now() - timedelta(hours=np.random.randint(0, 72)),
            symbol=symbol,
            direction=direction,
            stake=stake,
            result=result,
            pnl=pnl,
            confidence=confidence,
            volatility=volatility
        )
        
        risk_manager.add_trade(trade)
    
    # Calcular métricas
    metrics = risk_manager.calculate_risk_metrics()
    
    print(f"\n📈 Métricas de Risco Atuais:")
    print(f"Drawdown Atual: {metrics.current_drawdown:.1f}%")
    print(f"Perdas Consecutivas: {metrics.consecutive_losses}")
    print(f"PnL Diário: ${metrics.daily_pnl:.2f}")
    print(f"Trades por Hora: {metrics.hourly_trades}")
    print(f"Trades Diários: {metrics.daily_trades}")
    print(f"Taxa de Vitória: {metrics.win_rate:.1%}")
    print(f"Volatilidade Média: {metrics.avg_volatility:.3f}")
    print(f"Correlação do Portfólio: {metrics.portfolio_correlation:.3f}")
    print(f"Score de Risco: {metrics.risk_score:.1f}/100")
    
    # Testar permissões de trade
    print(f"\n🔍 Testando Permissões de Trade:")
    
    test_cases = [
        ('R_10', 2.0, 0.8, 0.03),
        ('R_25', 6.0, 0.9, 0.02),  # Stake muito alto
        ('R_50', 3.0, 0.6, 0.1),   # Volatilidade alta
        ('R_75', 1.5, 0.85, 0.025)
    ]
    
    for symbol, stake, confidence, volatility in test_cases:
        allowed, reason = risk_manager.check_trade_permission(symbol, stake, confidence, volatility)
        suggested_size = risk_manager.suggest_position_size(confidence, volatility)
        
        status = "✅ APROVADO" if allowed else "❌ REJEITADO"
        print(f"{status} - {symbol}: ${stake:.1f} -> {reason}")
        print(f"   Tamanho Sugerido: ${suggested_size:.1f}")
    
    # Gerar e salvar relatório
    report_file = risk_manager.save_risk_report()
    print(f"\n📋 Relatório de risco salvo: {report_file}")
    
    # Mostrar recomendações
    report = risk_manager.generate_risk_report()
    if report['recommendations']:
        print(f"\n💡 Recomendações:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    print(f"\n✅ Análise de risco concluída!")
    print(f"Total de trades analisados: {len(risk_manager.trade_history)}")

if __name__ == "__main__":
    run_risk_analysis()