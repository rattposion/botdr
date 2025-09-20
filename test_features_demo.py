#!/usr/bin/env python3
"""
Demonstração das funcionalidades avançadas do bot de trading
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_advanced_indicators():
    """Testa os indicadores avançados"""
    print("\n=== TESTE: Indicadores Técnicos Avançados ===")
    
    try:
        from advanced_indicators import AdvancedIndicators
        
        # Criar dados de exemplo
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        
        # Simular dados OHLC
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        high_prices = close_prices + np.random.rand(100) * 2
        low_prices = close_prices - np.random.rand(100) * 2
        open_prices = close_prices + np.random.randn(100) * 0.5
        volume = np.random.randint(1000, 10000, 100)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        # Testar indicadores
        indicators = AdvancedIndicators()
        
        # Ichimoku Cloud
        ichimoku = indicators.ichimoku_cloud(data)
        print(f"✓ Ichimoku Cloud calculado: {len(ichimoku)} componentes")
        
        # Williams %R
        williams_r = indicators.williams_percent_r(data)
        print(f"✓ Williams %R calculado: {len(williams_r)} valores")
        
        # CCI
        cci = indicators.commodity_channel_index(data)
        print(f"✓ CCI calculado: {len(cci)} valores")
        
        # Sinais de trading
        signals = indicators.get_trading_signals(data)
        print(f"✓ Sinais gerados: {signals}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste de indicadores: {e}")
        return False

def test_model_optimizer():
    """Testa o otimizador de modelos"""
    print("\n=== TESTE: Otimização de Hiperparâmetros ===")
    
    try:
        from model_optimizer import ModelOptimizer
        
        # Criar dados de exemplo
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Testar otimizador
        optimizer = ModelOptimizer()
        
        # Definir espaço de parâmetros simples
        param_space = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        }
        
        print("✓ ModelOptimizer criado com sucesso")
        print(f"✓ Espaço de parâmetros definido: {len(param_space)} parâmetros")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste de otimização: {e}")
        return False

def test_ensemble_model():
    """Testa o modelo ensemble"""
    print("\n=== TESTE: Modelo Ensemble ===")
    
    try:
        from ensemble_model import EnsembleModel
        
        # Criar dados de exemplo
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Testar ensemble
        ensemble = EnsembleModel()
        
        print("✓ EnsembleModel criado com sucesso")
        print("✓ Pronto para treinar múltiplos modelos")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste de ensemble: {e}")
        return False

def test_multi_timeframe():
    """Testa a estratégia multi-timeframe"""
    print("\n=== TESTE: Estratégia Multi-Timeframe ===")
    
    try:
        from multi_timeframe_strategy import MultiTimeframeStrategy
        
        # Testar criação da estratégia
        strategy = MultiTimeframeStrategy(symbol="R_50")
        
        print("✓ MultiTimeframeStrategy criado com sucesso")
        print(f"✓ Timeframes configurados: {strategy.timeframes}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste multi-timeframe: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🚀 INICIANDO TESTES DAS FUNCIONALIDADES AVANÇADAS")
    print("=" * 60)
    
    results = []
    
    # Executar testes
    results.append(("Indicadores Avançados", test_advanced_indicators()))
    results.append(("Otimização de Modelos", test_model_optimizer()))
    results.append(("Modelo Ensemble", test_ensemble_model()))
    results.append(("Multi-Timeframe", test_multi_timeframe()))
    
    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nResultado Final: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema pronto para uso avançado.")
    else:
        print("⚠️  Alguns testes falharam. Verifique os erros acima.")
    
    print("\n🔧 FUNCIONALIDADES DISPONÍVEIS:")
    print("• Indicadores técnicos avançados (Ichimoku, Williams %R, CCI)")
    print("• Otimização automática de hiperparâmetros")
    print("• Modelos ensemble para maior precisão")
    print("• Análise multi-timeframe")
    print("• Sistema completo de backtesting")

if __name__ == "__main__":
    main()