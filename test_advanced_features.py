"""
Teste das Funcionalidades Avançadas do Bot de Trading
- Novos indicadores técnicos
- Otimização de hiperparâmetros  
- Ensemble de modelos
- Estratégia multi-timeframe
"""
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports dos módulos
from config import config
from data_collector import DerivDataCollector
from feature_engineering import FeatureEngineer
from model_optimizer import ModelOptimizer, optimize_trading_model
from ensemble_model import EnsembleModel, create_ensemble_model
from multi_timeframe_strategy import MultiTimeframeStrategy

async def test_new_technical_indicators():
    """Testa os novos indicadores técnicos"""
    logger.info("=== TESTE: Novos Indicadores Técnicos ===")
    
    try:
        # Conectar e coletar dados
        collector = DerivDataCollector()
        collector.connect()
        
        # Coletar dados históricos
        df = collector.get_historical_data("R_50", timeframe="1m", count=500)
        logger.info(f"Dados coletados: {len(df)} candles")
        
        # Criar features com novos indicadores
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(df)
        
        logger.info(f"Features criadas: {features_df.shape}")
        logger.info(f"Colunas disponíveis: {len(features_df.columns)}")
        
        # Verificar novos indicadores
        new_indicators = [
            'cci_14', 'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a',
            'ichimoku_senkou_b', 'ichimoku_signal', 'parabolic_sar', 'adx_14',
            'obv', 'vwap', 'roc_5', 'roc_10', 'momentum_5', 'momentum_10'
        ]
        
        found_indicators = []
        for indicator in new_indicators:
            if indicator in features_df.columns:
                found_indicators.append(indicator)
                # Mostrar estatísticas básicas
                values = features_df[indicator].dropna()
                if len(values) > 0:
                    logger.info(f"{indicator}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
        
        logger.info(f"Novos indicadores encontrados: {len(found_indicators)}/{len(new_indicators)}")
        logger.info(f"Indicadores: {found_indicators}")
        
        collector.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Erro no teste de indicadores: {e}")
        return False

async def test_model_optimization():
    """Testa a otimização de hiperparâmetros"""
    logger.info("=== TESTE: Otimização de Hiperparâmetros ===")
    
    try:
        # Conectar e coletar dados
        collector = DerivDataCollector()
        collector.connect()
        
        # Coletar dados para otimização
        df = collector.get_historical_data("R_50", timeframe="1m", count=1000)
        logger.info(f"Dados para otimização: {len(df)} candles")
        
        # Teste com Random Search (mais rápido)
        logger.info("Iniciando otimização Random Search...")
        optimizer = ModelOptimizer("lightgbm")
        
        # Preparar dados
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(df)
        X, y = feature_engineer.prepare_features(features_df)
        
        logger.info(f"Dados preparados: X={X.shape}, y={y.shape}")
        
        # Otimização rápida (poucos iterations para teste)
        results = optimizer.random_search_optimization(X, y, n_iter=10, cv_folds=3)
        
        logger.info(f"Melhor score: {results['best_score']:.4f}")
        logger.info(f"Melhores parâmetros: {results['best_params']}")
        
        # Teste modelo otimizado
        optimized_model = optimize_trading_model(df, method="random", n_iter=5)
        logger.info("Modelo otimizado criado com sucesso!")
        
        collector.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Erro no teste de otimização: {e}")
        return False

async def test_ensemble_model():
    """Testa o sistema de ensemble"""
    logger.info("=== TESTE: Ensemble de Modelos ===")
    
    try:
        # Conectar e coletar dados
        collector = DerivDataCollector()
        collector.connect()
        
        # Coletar dados
        df = collector.get_historical_data("R_50", timeframe="1m", count=800)
        logger.info(f"Dados para ensemble: {len(df)} candles")
        
        # Criar ensemble
        logger.info("Criando ensemble de modelos...")
        ensemble = EnsembleModel("voting")
        
        # Selecionar alguns modelos para teste rápido
        selected_models = ['lightgbm', 'random_forest', 'gradient_boosting']
        ensemble.train(df, selected_models)
        
        logger.info("Ensemble treinado com sucesso!")
        
        # Testar predições
        predictions = ensemble.predict(df.tail(10))
        probabilities = ensemble.predict_proba(df.tail(10))
        
        logger.info(f"Predições: {predictions}")
        logger.info(f"Probabilidades shape: {probabilities.shape}")
        
        # Mostrar performance dos modelos base
        if ensemble.performance_metrics:
            logger.info("Performance dos modelos base:")
            for model_name, metrics in ensemble.performance_metrics.items():
                logger.info(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, CV={metrics['cv_mean']:.4f}")
        
        collector.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Erro no teste de ensemble: {e}")
        return False

async def test_multi_timeframe_strategy():
    """Testa a estratégia multi-timeframe"""
    logger.info("=== TESTE: Estratégia Multi-Timeframe ===")
    
    try:
        # Criar estratégia
        strategy = MultiTimeframeStrategy("R_50")
        await strategy.initialize()
        
        logger.info("Estratégia multi-timeframe inicializada")
        
        # Coletar dados para alguns timeframes (teste rápido)
        logger.info("Coletando dados para timeframes...")
        await strategy.collect_timeframe_data('1m', count=200)
        await strategy.collect_timeframe_data('5m', count=100)
        await strategy.collect_timeframe_data('15m', count=50)
        
        # Verificar dados coletados
        for tf, data in strategy.current_data.items():
            if not data.empty:
                logger.info(f"Timeframe {tf}: {len(data)} candles")
        
        # Treinar modelos (apenas para timeframes com dados)
        logger.info("Treinando modelos para timeframes...")
        for tf in ['1m', '5m', '15m']:
            if tf in strategy.current_data and not strategy.current_data[tf].empty:
                model = strategy.train_timeframe_model(tf, model_type="lightgbm")
                if model:
                    logger.info(f"Modelo {tf} treinado com sucesso")
        
        # Gerar sinais
        logger.info("Gerando sinais multi-timeframe...")
        signals = strategy.generate_all_signals()
        
        for tf, signal_data in signals.items():
            if 'error' not in signal_data:
                logger.info(f"Sinal {tf}: {signal_data['signal']} (confiança: {signal_data['confidence']:.3f})")
            else:
                logger.warning(f"Erro no sinal {tf}: {signal_data['error']}")
        
        # Combinar sinais
        combined = strategy.combine_signals(signals)
        logger.info(f"Sinal combinado: {combined['signal']} (confiança: {combined['confidence']:.3f})")
        logger.info(f"Razão: {combined['reason']}")
        
        # Análise de tendência
        trend = strategy.get_trend_alignment()
        logger.info(f"Alinhamento: {trend['alignment']} (força: {trend['strength']:.3f})")
        
        await strategy.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Erro no teste multi-timeframe: {e}")
        return False

async def test_complete_integration():
    """Teste de integração completa"""
    logger.info("=== TESTE: Integração Completa ===")
    
    try:
        # Conectar
        collector = DerivDataCollector()
        collector.connect()
        
        # Coletar dados
        df = collector.get_historical_data("R_50", timeframe="1m", count=600)
        logger.info(f"Dados coletados: {len(df)} candles")
        
        # 1. Testar features avançadas
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(df)
        logger.info(f"Features criadas: {features_df.shape[1]} colunas")
        
        # 2. Criar ensemble otimizado
        logger.info("Criando ensemble com modelos selecionados...")
        ensemble = create_ensemble_model(
            df, 
            ensemble_type="voting",
            selected_models=['lightgbm', 'random_forest']
        )
        
        # 3. Testar predições
        test_data = df.tail(20)
        predictions = ensemble.predict(test_data)
        probabilities = ensemble.predict_proba(test_data)
        
        # Analisar resultados
        call_signals = np.sum(predictions == 1)
        put_signals = np.sum(predictions == 0)
        avg_confidence = np.mean(np.max(probabilities, axis=1))
        
        logger.info(f"Sinais gerados: {call_signals} CALL, {put_signals} PUT")
        logger.info(f"Confiança média: {avg_confidence:.3f}")
        
        # 4. Verificar configurações
        min_conf = config.trading.min_prediction_confidence
        valid_signals = np.sum(np.max(probabilities, axis=1) >= min_conf)
        logger.info(f"Sinais válidos (>= {min_conf}): {valid_signals}/{len(predictions)}")
        
        collector.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Erro no teste de integração: {e}")
        return False

async def main():
    """Executa todos os testes"""
    logger.info("INICIANDO TESTES DAS FUNCIONALIDADES AVANÇADAS")
    logger.info("=" * 60)
    
    tests = [
        ("Novos Indicadores Técnicos", test_new_technical_indicators),
        ("Otimização de Hiperparâmetros", test_model_optimization),
        ("Ensemble de Modelos", test_ensemble_model),
        ("Estratégia Multi-Timeframe", test_multi_timeframe_strategy),
        ("Integração Completa", test_complete_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nExecutando: {test_name}")
        logger.info("-" * 40)
        
        try:
            start_time = datetime.now()
            success = await test_func()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results[test_name] = {
                'success': success,
                'duration': duration
            }
            
            status = "SUCESSO" if success else "FALHOU"
            logger.info(f"{status} - Duração: {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"ERRO CRÍTICO em {test_name}: {e}")
            results[test_name] = {
                'success': False,
                'duration': 0,
                'error': str(e)
            }
    
    # Resumo final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMO DOS TESTES")
    logger.info("=" * 60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r['success'])
    total_duration = sum(r['duration'] for r in results.values())
    
    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        duration = result['duration']
        logger.info(f"{status} {test_name}: {duration:.2f}s")
        
        if 'error' in result:
            logger.info(f"   Erro: {result['error']}")
    
    logger.info("-" * 60)
    logger.info(f"Testes executados: {total_tests}")
    logger.info(f"Sucessos: {successful_tests}")
    logger.info(f"Falhas: {total_tests - successful_tests}")
    logger.info(f"Taxa de sucesso: {successful_tests/total_tests*100:.1f}%")
    logger.info(f"Tempo total: {total_duration:.2f}s")
    
    if successful_tests == total_tests:
        logger.info("TODOS OS TESTES PASSARAM!")
    else:
        logger.info("Alguns testes falharam - verificar logs acima")

if __name__ == "__main__":
    asyncio.run(main())