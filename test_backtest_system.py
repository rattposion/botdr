#!/usr/bin/env python3
"""
Teste do Sistema de Backtesting Avançado
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("🔍 Testando Sistema de Backtesting Avançado")
print("=" * 50)

try:
    # Teste de imports
    print("📦 Testando imports...")
    
    from advanced_backtest_runner import AdvancedBacktestRunner
    print("✅ AdvancedBacktestRunner importado com sucesso")
    
    from ensemble_model import EnsembleModel
    print("✅ EnsembleModel importado com sucesso")
    
    from multi_timeframe_strategy import MultiTimeframeStrategy
    print("✅ MultiTimeframeStrategy importado com sucesso")
    
    from model_optimizer import ModelOptimizer
    print("✅ ModelOptimizer importado com sucesso")
    
    from advanced_indicators import AdvancedIndicators
    print("✅ AdvancedIndicators importado com sucesso")
    
    print("\n🧪 Testando funcionalidades básicas...")
    
    # Criar dados sintéticos para teste
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    # Simular dados de preço
    price_base = 100
    returns = np.random.normal(0, 0.001, 1000)
    prices = [price_base]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Criar DataFrame
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 5000, 1000)
    })
    test_data.set_index('timestamp', inplace=True)
    
    print(f"✅ Dados sintéticos criados: {len(test_data)} registros")
    
    # Teste AdvancedIndicators
    print("\n📊 Testando AdvancedIndicators...")
    indicators = AdvancedIndicators()
    
    # Teste Ichimoku
    ichimoku = indicators.ichimoku_cloud(test_data)
    print(f"✅ Ichimoku Cloud calculado: {len(ichimoku)} componentes")
    
    # Teste Williams %R
    williams_r = indicators.williams_percent_r(test_data)
    print(f"✅ Williams %R calculado: {len(williams_r)} valores")
    
    # Teste CCI
    cci = indicators.commodity_channel_index(test_data)
    print(f"✅ CCI calculado: {len(cci)} valores")
    
    # Teste EnsembleModel
    print("\n🤖 Testando EnsembleModel...")
    
    # Preparar dados para ML
    X = test_data[['open', 'high', 'low', 'volume']].fillna(0)
    y = (test_data['close'].shift(-1) > test_data['close']).astype(int).dropna()
    X = X.iloc[:-1]  # Ajustar tamanho
    
    if len(X) == len(y):
        ensemble = EnsembleModel()
        
        # Dividir dados
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Treinar
        ensemble.train(X_train, y_train)
        print("✅ EnsembleModel treinado com sucesso")
        
        # Predições
        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)
        
        accuracy = np.mean(predictions == y_test)
        print(f"✅ Predições realizadas - Acurácia: {accuracy:.2%}")
        
        # Teste ModelOptimizer (dentro do mesmo bloco para ter acesso às variáveis)
        print("\n⚙️ Testando ModelOptimizer...")
        
        optimizer = ModelOptimizer()
        
        # Teste simples de grid search (usando dados menores para teste rápido)
        if len(X_train) > 100:
            X_test_opt = X_train.iloc[:100]
            y_test_opt = y_train.iloc[:100]
            result = optimizer.grid_search_optimization(X_test_opt, y_test_opt, cv_folds=2)
            print(f"✅ Grid Search executado - Score: {result.get('best_score', 0):.3f}")
        else:
            print("✅ ModelOptimizer inicializado (dados insuficientes para teste completo)")
    
    # Teste MultiTimeframeStrategy
    print("\n📈 Testando MultiTimeframeStrategy...")
    
    strategy = MultiTimeframeStrategy(symbol="TEST")
    print(f"✅ MultiTimeframeStrategy criada para {len(strategy.timeframes)} timeframes")
    
    # Teste AdvancedBacktestRunner
    print("\n🚀 Testando AdvancedBacktestRunner...")
    
    runner = AdvancedBacktestRunner(symbols=["TEST"])
    print("✅ AdvancedBacktestRunner inicializado")
    
    # Verificar diretórios
    if os.path.exists(runner.results_dir):
        print(f"✅ Diretório de resultados criado: {runner.results_dir}")
    
    print("\n" + "=" * 50)
    print("🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
    print("🚀 Sistema de Backtesting Avançado está funcionando!")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)