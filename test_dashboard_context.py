#!/usr/bin/env python3
"""
Teste para reproduzir o contexto específico do dashboard e identificar o problema
"""
import os
import sys
import traceback

def test_dashboard_context():
    """Testa o contexto específico do dashboard"""
    print("🔍 Testando contexto específico do dashboard...")
    
    # Simular exatamente o que acontece no dashboard.py
    print(f"📁 Diretório atual: {os.getcwd()}")
    print(f"📄 __file__: {__file__}")
    print(f"📁 dirname(__file__): {os.path.dirname(__file__)}")
    
    # Testar o caminho exato usado no dashboard.py linha 333
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    print(f"📍 Caminho do .env: {env_file}")
    print(f"📄 Arquivo existe: {os.path.exists(env_file)}")
    print(f"📍 Caminho absoluto: {os.path.abspath(env_file)}")
    
    # Simular exatamente o código do dashboard.py
    try:
        print("\n💾 Simulando salvamento de token (código do dashboard)...")
        
        # Código exato das linhas 333-345 do dashboard.py
        with open(env_file, 'r') as f:
            lines = f.readlines()
        print(f"✅ Arquivo lido com sucesso! {len(lines)} linhas")
        
        # Simular token manual
        manual_token = "test_token_12345"
        
        # Atualizar ou adicionar linha do token
        token_found = False
        for i, line in enumerate(lines):
            if line.startswith('DERIV_API_TOKEN='):
                lines[i] = f'DERIV_API_TOKEN={manual_token}\n'
                token_found = True
                break
        
        if not token_found:
            lines.append(f'DERIV_API_TOKEN={manual_token}\n')
        
        print(f"🔍 Token encontrado: {token_found}")
        print(f"📝 Total de linhas após modificação: {len(lines)}")
        
        # Tentar escrever (esta é a linha que está falhando)
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        print("✅ Arquivo escrito com sucesso!")
        
    except Exception as e:
        print(f"❌ ERRO REPRODUZIDO: {e}")
        print(f"🔍 Tipo do erro: {type(e)}")
        print(f"📍 Caminho que causou erro: {env_file}")
        traceback.print_exc()
        
        # Verificar detalhes do erro
        if hasattr(e, 'filename'):
            print(f"🔍 Filename do erro: {e.filename}")
        if hasattr(e, 'errno'):
            print(f"🔍 Errno: {e.errno}")
    
    # Verificar se há alguma diferença no ambiente
    print("\n🌍 Verificando ambiente atual...")
    print(f"🔑 sys.executable: {sys.executable}")
    print(f"🔑 sys.prefix: {sys.prefix}")
    print(f"🔑 os.name: {os.name}")
    
    # Verificar permissões do arquivo
    print(f"\n🔒 Verificando permissões...")
    try:
        stat_info = os.stat(env_file)
        print(f"🔍 Stat info: {stat_info}")
        print(f"🔍 Arquivo é legível: {os.access(env_file, os.R_OK)}")
        print(f"🔍 Arquivo é gravável: {os.access(env_file, os.W_OK)}")
    except Exception as e:
        print(f"❌ Erro ao verificar permissões: {e}")

if __name__ == "__main__":
    test_dashboard_context()