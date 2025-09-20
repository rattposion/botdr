"""
Teste do Risk Manager
Verifica se o risk manager está funcionando corretamente
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import risk_manager, get_logger
from config import config

logger = get_logger(__name__)

def test_risk_manager():
    """Testa o risk manager"""
    print("🔍 Testando Risk Manager...")
    
    # Resetar contadores
    risk_manager.daily_pnl = 0
    risk_manager.daily_trades = 0
    risk_manager.martingale_step = 0
    risk_manager.current_balance = 1000.0
    
    print(f"📊 Configurações atuais:")
    print(f"   - Max trades diários: {config.trading.max_daily_trades}")
    print(f"   - Max perda diária: ${config.trading.max_daily_loss}")
    print(f"   - Martingale habilitado: {config.trading.enable_martingale}")
    print(f"   - Stake inicial: ${config.trading.initial_stake}")
    print(f"   - Confiança mínima: {config.trading.min_prediction_confidence}")
    
    # Teste 1: Estado inicial
    print(f"\n🧪 Teste 1: Estado inicial")
    can_trade, reason = risk_manager.can_trade()
    print(f"   - Pode fazer trade: {can_trade}")
    print(f"   - Razão: {reason}")
    
    # Teste 2: Simular alguns trades
    print(f"\n🧪 Teste 2: Simulando trades")
    for i in range(5):
        # Simular trade perdedor
        risk_manager.update_trade_result(-2.0, 1000.0 - (i+1)*2)
        can_trade, reason = risk_manager.can_trade()
        print(f"   - Trade {i+1}: PnL=-$2.00, Pode trade: {can_trade}, Razão: {reason}")
    
    # Teste 3: Status de risco
    print(f"\n🧪 Teste 3: Status de risco")
    status = risk_manager.get_risk_status()
    for key, value in status.items():
        print(f"   - {key}: {value}")
    
    # Teste 4: Calcular stake
    print(f"\n🧪 Teste 4: Cálculo de stake")
    stake_normal = risk_manager.calculate_stake(False)
    stake_martingale = risk_manager.calculate_stake(True)
    print(f"   - Stake normal: ${stake_normal}")
    print(f"   - Stake martingale: ${stake_martingale}")
    
    print(f"\n✅ Teste do Risk Manager concluído!")

if __name__ == "__main__":
    test_risk_manager()