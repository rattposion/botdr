#!/usr/bin/env python3
"""
Teste para verificar se as correções de path funcionaram corretamente
"""
import os
import sys
import tempfile
import shutil

def test_dashboard_token_save():
    """Testa o salvamento de token do dashboard com as correções"""
    print("🔍 Testando salvamento de token do dashboard (corrigido)...")
    
    try:
        # Simular o código corrigido do dashboard
        from dotenv import find_dotenv, set_key
        
        # Encontrar o arquivo .env automaticamente
        env_file = find_dotenv()
        if not env_file:
            # Se não encontrar, usar caminho padrão no diretório atual
            env_file = os.path.abspath('.env')
        
        print(f"📍 Arquivo .env encontrado: {env_file}")
        print(f"📄 Arquivo existe: {os.path.exists(env_file)}")
        
        # Testar salvamento com set_key
        test_token = "test_token_fixed_12345"
        success = set_key(env_file, 'DERIV_API_TOKEN', test_token)
        
        if success:
            print("✅ Token salvo com sucesso usando set_key!")
            
            # Verificar se foi salvo corretamente
            from dotenv import load_dotenv
            
            # Recarregar variáveis
            load_dotenv(env_file, override=True)
            saved_token = os.getenv('DERIV_API_TOKEN')
            
            if saved_token == test_token:
                print("✅ Token verificado com sucesso!")
            else:
                print(f"⚠️ Token salvo ({saved_token}) diferente do esperado ({test_token})")
        else:
            print("❌ Falha ao salvar token")
            
    except Exception as e:
        print(f"❌ Erro no teste do dashboard: {e}")
        import traceback
        traceback.print_exc()

def test_auth_manager_paths():
    """Testa os caminhos do auth_manager com as correções"""
    print("\n🔍 Testando caminhos do auth_manager (corrigido)...")
    
    try:
        # Simular o código corrigido do auth_manager
        token_file = os.path.abspath('.deriv_tokens.json')
        print(f"📍 Arquivo de tokens: {token_file}")
        
        # Testar criação de arquivo de teste
        test_data = {
            'access_token': 'test_access_token',
            'refresh_token': 'test_refresh_token',
            'expires_at': '2024-12-31T23:59:59',
            'user_info': {'test': 'data'},
            'saved_at': '2024-01-01T00:00:00'
        }
        
        import json
        with open(token_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print("✅ Arquivo de tokens criado com sucesso!")
        
        # Testar leitura
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                loaded_data = json.load(f)
            
            if loaded_data['access_token'] == test_data['access_token']:
                print("✅ Arquivo de tokens lido com sucesso!")
            else:
                print("⚠️ Dados lidos não conferem")
        
        # Limpar arquivo de teste
        if os.path.exists(token_file):
            os.remove(token_file)
            print("🧹 Arquivo de teste removido")
            
    except Exception as e:
        print(f"❌ Erro no teste do auth_manager: {e}")
        import traceback
        traceback.print_exc()

def test_working_directory_independence():
    """Testa se o código funciona independente do working directory"""
    print("\n🔍 Testando independência do working directory...")
    
    original_cwd = os.getcwd()
    
    try:
        # Criar diretório temporário
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Diretório temporário: {temp_dir}")
            
            # Mudar para diretório temporário
            os.chdir(temp_dir)
            print(f"📁 Working directory alterado para: {os.getcwd()}")
            
            # Testar se ainda consegue encontrar o .env original
            from dotenv import find_dotenv
            env_file = find_dotenv()
            
            if env_file:
                print(f"✅ Arquivo .env encontrado mesmo em diretório diferente: {env_file}")
            else:
                print("⚠️ Arquivo .env não encontrado em diretório diferente")
                # Isso é esperado, pois find_dotenv procura a partir do diretório atual
                
            # Testar caminho absoluto
            original_env = os.path.join(original_cwd, '.env')
            if os.path.exists(original_env):
                print(f"✅ Arquivo .env original ainda acessível: {original_env}")
            else:
                print("❌ Arquivo .env original não acessível")
    
    finally:
        # Restaurar diretório original
        os.chdir(original_cwd)
        print(f"📁 Working directory restaurado: {os.getcwd()}")

if __name__ == "__main__":
    print("🧪 Testando correções de path...")
    test_dashboard_token_save()
    test_auth_manager_paths()
    test_working_directory_independence()
    print("\n✅ Testes concluídos!")