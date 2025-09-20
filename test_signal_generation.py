#!/usr/bin/env python3
"""
Teste de Geração de Sinais - Bot Trading Deriv
Testa se o modelo ML está gerando sinais corretamente
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_logger
from data_collector import DerivDataCollector
from ai_trading_bot import AITradingBot
from config import config

logger = get_logger(__name__)

def test_signal_generation():
    """Testa a geração de sinais do modelo ML"""
    try:
        print("🤖 Testando Geração de Sinais...")
        
        # 1. Conectar ao data collector
        print("📡 Conectando ao data collector...")
        data_collector = DerivDataCollector()
        connection_result = data_collector.connect()
        print(f"   - Conexão: {connection_result}")
        
        # Aguardar um pouco para estabilizar a conexão
        if connection_result:
            time.sleep(2)
            print("   - Conexão estabilizada")
        
        # 2. Criar dados simulados para teste (mais confiável)
        print("📊 Criando dados simulados para teste...")
        dates = pd.date_range(start=datetime.now() - timedelta(hours=6), 
                            end=datetime.now(), freq='1min')
        
        # Simular preços com movimento browniano
        np.random.seed(42)
        base_price = 100
        returns = np.random.randn(len(dates)) * 0.002  # 0.2% volatilidade
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Criar DataFrame com OHLCV
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.rand(len(dates)) * 0.005),
            'low': prices * (1 - np.random.rand(len(dates)) * 0.005),
            'close': prices * (1 + np.random.randn(len(dates)) * 0.001),
            'volume': np.random.randint(100, 1000, len(dates))
        })
        
        print(f"   - Dados criados: {len(data)} candles")
        print(f"   - Período: {data['timestamp'].min()} até {data['timestamp'].max()}")
        print(f"   - Preço inicial: {data['close'].iloc[0]:.4f}")
        print(f"   - Preço final: {data['close'].iloc[-1]:.4f}")
        
        # 3. Inicializar o bot de trading
        print("🤖 Inicializando AI Trading Bot...")
        bot = AITradingBot()
        
        # Inicializar o modelo ML manualmente
        from ml_model import TradingMLModel
        bot.ml_model = TradingMLModel()
        
        # 4. Testar feature engineering
        print("🔧 Testando feature engineering...")
        try:
            features = bot.feature_engineer.create_features(data)
            print(f"   - Features criadas: {features.shape}")
            print(f"   - Colunas: {list(features.columns)[:10]}...")  # Primeiras 10 colunas
            
            # Verificar se há dados suficientes
            if len(features) < 50:
                print("   ⚠️ Poucos dados para análise confiável")
            else:
                print("   ✅ Dados suficientes para análise")
                
        except Exception as e:
            print(f"   ❌ Erro no feature engineering: {e}")
            return False
        
        # 5. Testar modelo ML
        print("🧠 Testando modelo ML...")
        try:
            # Verificar se existe modelo treinado
            model_path = "models/lightgbm_model.pkl"
            if os.path.exists(model_path):
                print("   - Modelo encontrado, carregando...")
                bot.ml_model.load_model(model_path)
                print("   ✅ Modelo carregado")
            else:
                print("   - Modelo não encontrado, treinando...")
                # Treinar com dados simulados
                X = features.dropna()
                if len(X) > 50:
                    # Criar target simulado (próximo movimento)
                    y = (data['close'].shift(-1) > data['close']).astype(int)
                    y = y.iloc[:len(X)]
                    
                    # Converter para tipos numéricos e remover colunas não numéricas
                    X_numeric = X.select_dtypes(include=[np.number])
                    if 'timestamp' in X_numeric.columns:
                        X_numeric = X_numeric.drop('timestamp', axis=1)
                    
                    # Garantir que não há valores infinitos ou NaN
                    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)
                    y_clean = y.fillna(0)
                    
                    print(f"   - Dados para treino: {X_numeric.shape}, Target: {len(y_clean)}")
                    
                    # Treinar diretamente sem validação de tamanho mínimo
                    from sklearn.model_selection import train_test_split
                    import lightgbm as lgb
                    
                    # Split dos dados
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_numeric, y_clean, test_size=0.2, random_state=42
                    )
                    
                    # Treinar LightGBM diretamente
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        verbose=-1
                    )
                    
                    model.fit(X_train, y_train)
                    bot.ml_model.model = model
                    bot.ml_model.is_trained = True
                    bot.ml_model.feature_names = list(X_numeric.columns)
                    
                    # Salvar dados processados para uso posterior
                    X_processed = X_numeric
                    
                    # Avaliar
                    from sklearn.metrics import accuracy_score
                    y_pred = model.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred)
                    print(f"   ✅ Modelo treinado - Acurácia: {accuracy:.3f}")
                else:
                    print("   ⚠️ Dados insuficientes para treinar modelo")
                    return False
                    
        except Exception as e:
            print(f"   ❌ Erro no modelo ML: {e}")
            return False
        
        # 6. Testar geração de sinais
        print("📈 Testando geração de sinais...")
        try:
            # Usar dados já processados para predição direta
            recent_X = X_processed[-10:]  # Últimas 10 amostras processadas
            
            # Fazer predição direta
            predictions = bot.ml_model.predict(recent_X)
            probabilities = bot.ml_model.predict_proba(recent_X)
            
            # Última predição
            last_prediction = predictions[-1]
            last_probability = probabilities[-1]
            confidence = last_probability[last_prediction]
            
            print(f"   - Predição: {last_prediction} ({'CALL' if last_prediction == 1 else 'PUT'})")
            print(f"   - Confiança: {confidence:.3f}")
            print(f"   - Prob UP: {last_probability[1]:.3f}")
            print(f"   - Prob DOWN: {last_probability[0]:.3f}")
            
            # Verificar se o sinal é válido
            min_confidence = 0.6  # Confiança mínima
            if confidence >= min_confidence:
                signal_type = 'CALL' if last_prediction == 1 else 'PUT'
                print(f"   ✅ Sinal válido: {signal_type} (confiança: {confidence:.3f})")
            else:
                print(f"   ⚠️ Sinal com baixa confiança: {confidence:.3f} < {min_confidence}")
                
        except Exception as e:
            print(f"   ❌ Erro na geração de sinais: {e}")
            return False
        
        # 7. Testar múltiplos sinais
        print("🔄 Testando múltiplos sinais...")
        signals_count = 0
        valid_signals = 0
        
        for i in range(5):
            try:
                # Usar diferentes partes dos dados processados
                start_idx = max(0, len(X_processed) - 50 - i*10)
                end_idx = len(X_processed) - i*2
                test_X = X_processed[start_idx:end_idx]
                
                if len(test_X) > 0:
                    predictions = bot.ml_model.predict(test_X[-10:])  # Últimas 10 amostras
                    probabilities = bot.ml_model.predict_proba(test_X[-10:])
                    
                    last_prediction = predictions[-1]
                    last_probability = probabilities[-1]
                    confidence = last_probability[last_prediction]
                    
                    signals_count += 1
                    
                    if confidence >= 0.6:  # Confiança mínima
                        valid_signals += 1
                    
                time.sleep(0.1)  # Pequena pausa
                
            except Exception as e:
                print(f"   ⚠️ Erro no teste {i+1}: {e}")
        
        print(f"   - Sinais testados: {signals_count}")
        print(f"   - Sinais válidos: {valid_signals}")
        print(f"   - Taxa de sucesso: {(valid_signals/signals_count)*100:.1f}%" if signals_count > 0 else "N/A")
        
        # 8. Verificar configurações
        print("⚙️ Verificando configurações...")
        print(f"   - Threshold mínimo: {config.trading.min_prediction_confidence}")
        print(f"   - Stake inicial: ${config.trading.initial_stake}")
        print(f"   - Max trades diários: {config.trading.max_daily_trades}")
        print(f"   - Stop loss: ${config.trading.max_daily_loss}")
        
        # 9. Desconectar
        if connection_result:
            data_collector.disconnect()
            print("📡 Desconectado do WebSocket")
        
        print("\n✅ Teste de geração de sinais concluído!")
        return True
        
    except Exception as e:
        logger.error(f"Erro no teste de sinais: {e}")
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    test_signal_generation()