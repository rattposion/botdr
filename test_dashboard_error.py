"""
Script para testar tratamento de erro no dashboard
Simula erro de token inválido
"""
import streamlit as st
import sys
import os

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simulate_api_error():
    """Simula erro de API para testar o dashboard"""
    # Simular erro de token OAuth
    st.session_state.api_error = "PermissionDenied: An oauth token is required to access account balance."
    
    print("✅ Erro de API simulado no session_state")
    print("🔄 Agora abra o dashboard para ver o tratamento de erro")
    print("📱 URL: http://localhost:8501")

if __name__ == "__main__":
    # Configurar Streamlit para não mostrar warnings
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Simular erro
    simulate_api_error()
    
    # Importar e executar dashboard
    try:
        from dashboard import main
        main()
    except Exception as e:
        print(f"Erro ao executar dashboard: {e}")