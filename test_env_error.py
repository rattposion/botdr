#!/usr/bin/env python3
"""
Script de teste para reproduzir o erro do arquivo .env
"""
import os
import sys
from dotenv import load_dotenv, set_key

def test_env_operations():
    """Testa operações com arquivo .env"""
    print("🔍 Testando operações com arquivo .env...")
    
    # Verificar diretório atual
    print(f"📁 Diretório atual: {os.getcwd()}")
    
    # Verificar se .env existe
    env_file = ".env"
    print(f"📄 Arquivo .env existe: {os.path.exists(env_file)}")
    
    # Tentar carregar .env
    print("📥 Carregando .env...")
    result = load_dotenv(env_file, verbose=True)
    print(f"✅ Resultado do load_dotenv: {result}")
    
    # Verificar variáveis carregadas
    print(f"🔑 DERIV_APP_ID: {os.getenv('DERIV_APP_ID')}")
    print(f"🔑 DERIV_API_TOKEN: {os.getenv('DERIV_API_TOKEN', 'Não definido')}")
    
    # Tentar salvar uma nova variável
    print("\n💾 Testando salvamento de nova variável...")
    try:
        # Usar caminho absoluto
        env_path = os.path.abspath(env_file)
        print(f"📍 Caminho absoluto: {env_path}")
        
        # Tentar salvar
        result = set_key(env_path, "TEST_VAR", "test_value")
        print(f"✅ Resultado do set_key: {result}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        print(f"🔍 Tipo do erro: {type(e)}")
        import traceback
        traceback.print_exc()
    
    # Verificar se há alguma configuração de ambiente que possa estar interferindo
    print("\n🌍 Verificando variáveis de ambiente relevantes...")
    env_vars = ['PYTHONPATH', 'HOME', 'USERPROFILE', 'PWD', 'RAILWAY_ENVIRONMENT']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"🔑 {var}: {value}")

if __name__ == "__main__":
    test_env_operations()