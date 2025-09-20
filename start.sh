#!/bin/bash

# Script de inicialização para Railway
# Este script configura o ambiente e inicia o bot de trading

set -e  # Exit on any error

echo "🚀 Iniciando Deriv AI Trading Bot no Railway..."

# Verificar variáveis de ambiente essenciais
if [ -z "$PORT" ]; then
    echo "⚠️ PORT não definida, usando 8080"
    export PORT=8080
fi

if [ -z "$DERIV_API_TOKEN" ]; then
    echo "⚠️ DERIV_API_TOKEN não configurado - funcionalidade limitada"
fi

if [ -z "$DERIV_APP_ID" ]; then
    echo "⚠️ DERIV_APP_ID não configurado, usando padrão"
    export DERIV_APP_ID=1089
fi

# Configurar ambiente de produção
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export RAILWAY_ENVIRONMENT=production

# Configurações do Streamlit para produção
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

echo "📊 Configurações:"
echo "   - PORT: $PORT"
echo "   - HOST: 0.0.0.0"
echo "   - ENVIRONMENT: production"
echo "   - DERIV_APP_ID: $DERIV_APP_ID"

# Criar diretórios necessários
mkdir -p logs data models backtest_results reports

# Verificar se o Python está disponível
if ! command -v python &> /dev/null; then
    echo "❌ Python não encontrado"
    exit 1
fi

echo "✅ Ambiente configurado com sucesso"

# Iniciar aplicação
echo "🎯 Iniciando dashboard..."
exec python main.py --mode dashboard --host 0.0.0.0 --port $PORT