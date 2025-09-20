#!/usr/bin/env python3
"""
Teste para simular o ambiente do Streamlit e verificar o problema do .env
"""
import os
import sys

def test_streamlit_environment():
    """Simula o ambiente do Streamlit"""
    print("🔍 Testando ambiente similar ao Streamlit...")
    
    # Simular o que acontece no dashboard.py
    print(f"📁 Diretório atual: {os.getcwd()}")
    print(f"📄 __file__: {__file__}")
    print(f"📁 dirname(__file__): {os.path.dirname(__file__)}")
    
    # Testar o caminho usado no dashboard.py
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    print(f"📍 Caminho do .env: {env_file}")
    print(f"📄 Arquivo existe: {os.path.exists(env_file)}")
    
    # Testar abertura do arquivo
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
        print(f"✅ Arquivo lido com sucesso! {len(lines)} linhas")
        
        # Testar escrita
        with open(env_file, 'w') as f:
            f.writelines(lines)
        print("✅ Arquivo escrito com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print(f"🔍 Tipo do erro: {type(e)}")
        import traceback
        traceback.print_exc()
    
    # Verificar se há alguma configuração específica
    print("\n🌍 Verificando configurações do sistema...")
    print(f"🔑 sys.path[0]: {sys.path[0]}")
    print(f"🔑 os.getcwd(): {os.getcwd()}")
    
    # Verificar se há alguma variável de ambiente que possa estar interferindo
    print("\n🔍 Verificando variáveis de ambiente...")
    for key, value in os.environ.items():
        if any(term in key.lower() for term in ['path', 'env', 'app', 'deriv']):
            print(f"🔑 {key}: {value}")

if __name__ == "__main__":
    test_streamlit_environment()