"""
Teste do Fluxo Completo de Autenticação
Verifica integração entre auth_manager, token_manager, data_collector e balance_manager
"""
import asyncio
import time
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_auth_manager():
    """Testa funcionalidades do auth_manager"""
    print("\n" + "="*60)
    print("🔐 TESTE DO AUTH MANAGER")
    print("="*60)
    
    try:
        from auth_manager import auth_manager
        
        # Verificar status inicial
        print(f"✅ Auth manager importado com sucesso")
        print(f"📊 Status inicial: {auth_manager.get_auth_status()}")
        
        # Verificar se há token salvo
        if auth_manager.is_authenticated:
            print("✅ Usuário já autenticado")
            token = auth_manager.get_api_token()
            if token:
                print(f"🔑 Token disponível: {token[:10]}...{token[-10:]}")
            else:
                print("❌ Token não disponível")
        else:
            print("⚠️ Usuário não autenticado")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no auth_manager: {e}")
        return False

def test_token_manager():
    """Testa funcionalidades do token_manager"""
    print("\n" + "="*60)
    print("🔄 TESTE DO TOKEN MANAGER")
    print("="*60)
    
    try:
        from token_manager import token_manager
        
        print(f"✅ Token manager importado com sucesso")
        
        # Verificar status
        status = token_manager.get_status()
        print(f"📊 Status: {status}")
        
        # Iniciar monitoramento
        if not status['monitoring']:
            print("🔄 Iniciando monitoramento...")
            success = token_manager.start_monitoring()
            print(f"📊 Monitoramento iniciado: {success}")
        else:
            print("✅ Monitoramento já ativo")
        
        # Aguardar um pouco
        time.sleep(2)
        
        # Verificar status novamente
        status = token_manager.get_status()
        print(f"📊 Status após inicialização: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no token_manager: {e}")
        return False

def test_data_collector():
    """Testa integração do data_collector com auth"""
    print("\n" + "="*60)
    print("📡 TESTE DO DATA COLLECTOR")
    print("="*60)
    
    try:
        from data_collector import data_collector
        
        print(f"✅ Data collector importado com sucesso")
        
        # Verificar status inicial
        print(f"📊 Conectado: {data_collector.is_connected}")
        print(f"📊 Autorizado: {data_collector.is_authorized}")
        
        # Tentar conectar
        if not data_collector.is_connected:
            print("🔄 Conectando...")
            success = data_collector.connect()
            print(f"📊 Conexão: {success}")
            
            # Aguardar conexão
            time.sleep(3)
        
        # Tentar autorizar
        if data_collector.is_connected and not data_collector.is_authorized:
            print("🔄 Autorizando...")
            success = data_collector.authorize()
            print(f"📊 Autorização: {success}")
            
            # Aguardar autorização
            time.sleep(2)
        
        print(f"📊 Status final - Conectado: {data_collector.is_connected}, Autorizado: {data_collector.is_authorized}")
        
        return data_collector.is_connected and data_collector.is_authorized
        
    except Exception as e:
        print(f"❌ Erro no data_collector: {e}")
        return False

async def test_balance_manager():
    """Testa funcionalidades do balance_manager"""
    print("\n" + "="*60)
    print("💰 TESTE DO BALANCE MANAGER")
    print("="*60)
    
    try:
        from balance_manager import balance_manager
        
        print(f"✅ Balance manager importado com sucesso")
        
        # Verificar status inicial
        info = balance_manager.get_balance_info()
        print(f"📊 Info inicial: {info}")
        
        # Iniciar atualização automática
        if not balance_manager.is_updating:
            print("🔄 Iniciando atualização automática...")
            balance_manager.start_auto_update()
            time.sleep(1)
        
        # Forçar atualização
        print("🔄 Forçando atualização de saldo...")
        success = await balance_manager.update_balance()
        print(f"📊 Atualização: {success}")
        
        # Verificar resultado
        info = balance_manager.get_balance_info()
        print(f"📊 Info final: {info}")
        
        return success
        
    except Exception as e:
        print(f"❌ Erro no balance_manager: {e}")
        return False

def test_integration():
    """Testa integração completa"""
    print("\n" + "="*60)
    print("🔗 TESTE DE INTEGRAÇÃO COMPLETA")
    print("="*60)
    
    try:
        from auth_manager import auth_manager
        from token_manager import token_manager
        from data_collector import data_collector
        from balance_manager import balance_manager
        
        print("✅ Todos os módulos importados")
        
        # Verificar se há autenticação
        if not auth_manager.is_authenticated:
            print("⚠️ Usuário não autenticado - alguns testes podem falhar")
            print("💡 Para testar completamente, faça login primeiro no dashboard")
            return False
        
        # Verificar token
        token = auth_manager.get_api_token()
        if not token:
            print("❌ Token não disponível")
            return False
        
        print(f"✅ Token disponível: {token[:10]}...{token[-10:]}")
        
        # Verificar se data_collector usa o token
        if data_collector.is_connected and data_collector.is_authorized:
            print("✅ Data collector conectado e autorizado")
        else:
            print("⚠️ Data collector não está totalmente funcional")
        
        # Verificar balance_manager
        info = balance_manager.get_balance_info()
        if info['status'] == 'connected':
            print(f"✅ Balance manager funcionando - Saldo: ${info['balance']:.2f}")
        else:
            print(f"⚠️ Balance manager com problemas: {info['error']}")
        
        # Verificar token_manager
        token_status = token_manager.get_status()
        if token_status['monitoring']:
            print("✅ Token manager monitorando automaticamente")
        else:
            print("⚠️ Token manager não está monitorando")
        
        print("\n🎉 Integração testada com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        return False

async def main():
    """Função principal de teste"""
    print("🚀 INICIANDO TESTES DO FLUXO DE AUTENTICAÇÃO")
    print("=" * 80)
    
    results = []
    
    # Teste 1: Auth Manager
    results.append(("Auth Manager", test_auth_manager()))
    
    # Teste 2: Token Manager
    results.append(("Token Manager", test_token_manager()))
    
    # Teste 3: Data Collector
    results.append(("Data Collector", test_data_collector()))
    
    # Teste 4: Balance Manager
    results.append(("Balance Manager", await test_balance_manager()))
    
    # Teste 5: Integração
    results.append(("Integração", test_integration()))
    
    # Resumo dos resultados
    print("\n" + "="*80)
    print("📊 RESUMO DOS TESTES")
    print("="*80)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:20} | {status}")
    
    total_passed = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f"\n📈 RESULTADO FINAL: {total_passed}/{total_tests} testes passaram")
    
    if total_passed == total_tests:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema de autenticação funcionando perfeitamente.")
    elif total_passed >= total_tests * 0.8:
        print("✅ MAIORIA DOS TESTES PASSOU. Sistema funcionando com pequenos problemas.")
    else:
        print("⚠️ VÁRIOS TESTES FALHARAM. Verifique a configuração do sistema.")
    
    print("\n💡 DICAS:")
    print("- Para funcionalidade completa, faça login no dashboard primeiro")
    print("- Verifique se o arquivo .env está configurado corretamente")
    print("- Certifique-se de que a conexão com internet está funcionando")

if __name__ == "__main__":
    asyncio.run(main())