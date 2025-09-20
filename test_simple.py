#!/usr/bin/env python3
"""
Teste simples para verificar se os módulos estão funcionando
"""

print("Iniciando teste simples...")

try:
    from advanced_indicators import AdvancedIndicators
    print("✓ AdvancedIndicators importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar AdvancedIndicators: {e}")

try:
    from model_optimizer import ModelOptimizer
    print("✓ ModelOptimizer importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar ModelOptimizer: {e}")

try:
    from ensemble_model import EnsembleModel
    print("✓ EnsembleModel importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar EnsembleModel: {e}")

try:
    from multi_timeframe_strategy import MultiTimeframeStrategy
    print("✓ MultiTimeframeStrategy importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar MultiTimeframeStrategy: {e}")

print("Teste simples concluído!")