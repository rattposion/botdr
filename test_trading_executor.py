#!/usr/bin/env python3
"""
Script de teste para verificar se o TradingExecutor está funcionando corretamente
"""

import asyncio
import logging
from trader import TradingExecutor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_trading_executor():
    """Testa o TradingExecutor de forma isolada"""
    print("🧪 Iniciando teste do TradingExecutor...")
    
    try:
        # Criar instância
        print("🔧 Criando instância do TradingExecutor...")
        trader = TradingExecutor()
        print("✅ TradingExecutor criado com sucesso")
        
        # Testar inicialização
        print("🔄 Testando inicialização...")
        await trader.initialize()
        print("✅ Inicialização concluída")
        
        # Verificar estado
        print(f"📊 Estado do trader:")
        print(f"  - Modelo ML carregado: {trader.ml_model is not None}")
        print(f"  - Data collector: {trader.data_collector is not None}")
        print(f"  - Feature engineer: {trader.feature_engineer is not None}")
        print(f"  - Tick history: {len(trader.tick_history)} ticks")
        
        # Testar start_trading por alguns segundos
        print("🚀 Testando start_trading por 10 segundos...")
        
        # Criar uma task para parar após 10 segundos
        async def stop_after_delay():
            await asyncio.sleep(10)
            trader.is_running = False
            print("⏹️ Parando trading após 10 segundos...")
        
        # Executar ambas as tasks
        await asyncio.gather(
            trader.start_trading(),
            stop_after_delay()
        )
        
        print("✅ Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_trading_executor())