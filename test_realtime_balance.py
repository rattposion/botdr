"""
Teste do Sistema de Saldo em Tempo Real
Verifica se o balance_manager está funcionando corretamente
"""
import asyncio
import time
from datetime import datetime
from balance_manager import balance_manager
from config import config

def test_balance_manager():
    """Testa o gerenciador de saldo"""
    print("🧪 Testando Balance Manager...")
    print("=" * 50)
    
    # 1. Verificar configuração inicial
    print("1. Verificando configuração inicial...")
    balance_info = balance_manager.get_balance_info()
    print(f"   Status inicial: {balance_manager.get_status_text()}")
    print(f"   Emoji: {balance_manager.get_status_emoji()}")
    print(f"   Saldo: ${balance_info['balance']:.2f}")
    print(f"   Conectado: {balance_info['is_connected']}")
    print(f"   Atualizando: {balance_info['is_updating']}")
    print()
    
    # 2. Verificar token da API
    print("2. Verificando token da API...")
    if config.deriv.api_token:
        print(f"   ✅ Token configurado: {config.deriv.api_token[:10]}...")
    else:
        print("   ❌ Token não configurado")
    print()
    
    # 3. Iniciar atualização automática
    print("3. Iniciando atualização automática...")
    balance_manager.start_auto_update()
    print("   ✅ Atualização automática iniciada")
    print()
    
    # 4. Aguardar algumas atualizações
    print("4. Monitorando atualizações por 60 segundos...")
    start_time = time.time()
    last_status = None
    update_count = 0
    
    while time.time() - start_time < 60:
        balance_info = balance_manager.get_balance_info()
        current_status = balance_info['connection_status']
        
        if current_status != last_status:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"   [{timestamp}] Status: {balance_manager.get_status_emoji()} {balance_manager.get_status_text()}")
            
            if balance_info['error_message']:
                print(f"   [{timestamp}] Erro: {balance_info['error_message']}")
            
            if balance_info['balance'] > 0:
                print(f"   [{timestamp}] Saldo: ${balance_info['balance']:.2f}")
                update_count += 1
            
            last_status = current_status
        
        time.sleep(2)
    
    print(f"   📊 Total de atualizações de saldo: {update_count}")
    print()
    
    # 5. Teste de atualização forçada
    print("5. Testando atualização forçada...")
    print("   Forçando atualização...")
    balance_manager.force_update()
    
    # Aguardar resultado
    time.sleep(5)
    balance_info = balance_manager.get_balance_info()
    print(f"   Status após força: {balance_manager.get_status_text()}")
    print(f"   Saldo: ${balance_info['balance']:.2f}")
    print()
    
    # 6. Parar atualização automática
    print("6. Parando atualização automática...")
    balance_manager.stop_auto_update()
    print("   ✅ Atualização automática parada")
    print()
    
    # 7. Resumo final
    print("7. Resumo final...")
    final_info = balance_manager.get_balance_info()
    print(f"   Status final: {balance_manager.get_status_text()}")
    print(f"   Saldo final: ${final_info['balance']:.2f}")
    print(f"   Última atualização: {final_info['last_update']}")
    print(f"   Total de erros: {1 if final_info['error_message'] else 0}")
    
    # Resultado do teste
    print("\n" + "=" * 50)
    if final_info['balance'] > 0:
        print("✅ TESTE PASSOU - Saldo obtido com sucesso!")
    elif final_info['connection_status'] == 'no_token':
        print("⚠️  TESTE PARCIAL - Token não configurado (esperado)")
    elif final_info['connection_status'] == 'invalid_token':
        print("⚠️  TESTE PARCIAL - Token inválido (configure um token válido)")
    else:
        print("❌ TESTE FALHOU - Erro na conexão")
    
    return final_info['balance'] > 0

async def test_async_balance():
    """Testa a função assíncrona de saldo"""
    print("\n🔄 Testando função assíncrona...")
    
    try:
        from data_collector import data_collector
        
        # Conectar
        print("   Conectando...")
        connected = data_collector.connect()
        if not connected:
            print("   ❌ Falha na conexão")
            return False
        
        # Autorizar
        print("   Autorizando...")
        authorized = data_collector.authorize()
        if not authorized:
            print("   ❌ Falha na autorização")
            return False
        
        # Obter saldo
        print("   Obtendo saldo...")
        balance_result = await data_collector.get_balance()
        
        if balance_result and 'error' not in balance_result:
            print("   ✅ Saldo obtido com sucesso!")
            print(f"   Resposta: {balance_result}")
            return True
        else:
            print("   ❌ Erro ao obter saldo")
            if balance_result:
                print(f"   Erro: {balance_result}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exceção: {e}")
        return False

def main():
    """Função principal de teste"""
    print("🚀 Iniciando Teste do Sistema de Saldo em Tempo Real")
    print("=" * 60)
    
    # Teste 1: Balance Manager
    success1 = test_balance_manager()
    
    # Teste 2: Função Assíncrona
    success2 = asyncio.run(test_async_balance())
    
    # Resultado final
    print("\n" + "=" * 60)
    print("📋 RESULTADO FINAL:")
    print(f"   Balance Manager: {'✅ PASSOU' if success1 else '❌ FALHOU'}")
    print(f"   Função Assíncrona: {'✅ PASSOU' if success2 else '❌ FALHOU'}")
    
    if success1 and success2:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("   O sistema de saldo em tempo real está funcionando!")
    elif success1 or success2:
        print("\n⚠️  TESTES PARCIAIS PASSARAM")
        print("   Verifique a configuração do token da API")
    else:
        print("\n❌ TESTES FALHARAM")
        print("   Verifique a configuração e conexão")

if __name__ == "__main__":
    main()