#!/usr/bin/env python3
"""
Script de diagnóstico para problemas de autenticação Deriv
Testa diferentes cenários de conexão e autenticação
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

# Configurações de teste
CONFIGS_TO_TEST = [
    {
        "name": "App ID Demo Padrão (1089)",
        "app_id": "1089",
        "url": "wss://ws.binaryws.com/websockets/v3",
        "token": None
    },
    {
        "name": "App ID Configurado (101918)",
        "app_id": "101918", 
        "url": "wss://ws.binaryws.com/websockets/v3",
        "token": None
    },
    {
        "name": "App ID Configurado com Token",
        "app_id": "101918",
        "url": "wss://ws.binaryws.com/websockets/v3", 
        "token": "cuCpkc00HgKXvym"
    },
    {
        "name": "URL Alternativa (derivws.com)",
        "app_id": "1089",
        "url": "wss://ws.derivws.com/websockets/v3",
        "token": None
    }
]

async def test_connection(config):
    """Testa uma configuração específica"""
    print(f"\n🔄 Testando: {config['name']}")
    print(f"   App ID: {config['app_id']}")
    print(f"   URL: {config['url']}")
    print(f"   Token: {'Sim' if config['token'] else 'Não'}")
    
    try:
        # Construir URL com app_id
        url = f"{config['url']}?app_id={config['app_id']}"
        print(f"   URL completa: {url}")
        
        # Conectar ao WebSocket
        async with websockets.connect(url) as websocket:
            print("   ✅ Conectado ao WebSocket!")
            
            # Teste 1: Ping
            ping_request = {
                "ping": 1,
                "req_id": 1
            }
            
            await websocket.send(json.dumps(ping_request))
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            ping_data = json.loads(response)
            
            if "pong" in ping_data:
                print("   ✅ Ping/Pong funcionando!")
            else:
                print(f"   ⚠️ Resposta inesperada do ping: {ping_data}")
            
            # Teste 2: Autorização (se token disponível)
            if config['token']:
                auth_request = {
                    "authorize": config['token'],
                    "req_id": 2
                }
                
                await websocket.send(json.dumps(auth_request))
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                auth_data = json.loads(response)
                
                if 'error' in auth_data:
                    print(f"   ❌ Erro de autorização: {auth_data['error']}")
                    print(f"      Código: {auth_data['error'].get('code', 'N/A')}")
                    print(f"      Mensagem: {auth_data['error'].get('message', 'N/A')}")
                elif 'authorize' in auth_data:
                    print("   ✅ Autorização bem-sucedida!")
                    print(f"      Login ID: {auth_data['authorize'].get('loginid', 'N/A')}")
                    print(f"      Moeda: {auth_data['authorize'].get('currency', 'N/A')}")
                else:
                    print(f"   ⚠️ Resposta inesperada da autorização: {auth_data}")
            
            # Teste 3: Obter tempo do servidor
            time_request = {
                "time": 1,
                "req_id": 3
            }
            
            await websocket.send(json.dumps(time_request))
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            time_data = json.loads(response)
            
            if 'time' in time_data:
                server_time = datetime.fromtimestamp(time_data['time'])
                print(f"   ✅ Tempo do servidor: {server_time}")
            else:
                print(f"   ⚠️ Erro ao obter tempo: {time_data}")
            
            return True
            
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"   ❌ Erro de status HTTP: {e}")
        print(f"      Status Code: {e.status_code}")
        if e.status_code == 401:
            print("      🔍 HTTP 401: Problema de autenticação")
            print("         - Verifique se o App ID está correto")
            print("         - Verifique se o token não expirou")
            print("         - Tente usar o App ID padrão 1089 para demo")
        return False
        
    except asyncio.TimeoutError:
        print("   ❌ Timeout na conexão")
        return False
        
    except Exception as e:
        print(f"   ❌ Erro inesperado: {e}")
        return False

async def main():
    """Executa todos os testes"""
    print("🔍 Diagnóstico de Autenticação Deriv")
    print("=" * 50)
    
    results = []
    
    for config in CONFIGS_TO_TEST:
        success = await test_connection(config)
        results.append((config['name'], success))
        time.sleep(1)  # Pausa entre testes
    
    # Resumo dos resultados
    print("\n📊 Resumo dos Testes:")
    print("=" * 50)
    
    for name, success in results:
        status = "✅ SUCESSO" if success else "❌ FALHOU"
        print(f"{status}: {name}")
    
    # Recomendações
    print("\n💡 Recomendações:")
    print("=" * 50)
    
    successful_configs = [name for name, success in results if success]
    
    if successful_configs:
        print("✅ Configurações que funcionaram:")
        for config in successful_configs:
            print(f"   - {config}")
        print("\n🔧 Use uma das configurações que funcionaram no seu código.")
    else:
        print("❌ Nenhuma configuração funcionou.")
        print("\n🔧 Possíveis soluções:")
        print("   1. Verifique sua conexão com a internet")
        print("   2. Verifique se não há firewall bloqueando")
        print("   3. Tente gerar um novo token na Deriv")
        print("   4. Use o App ID padrão 1089 para testes")

if __name__ == "__main__":
    asyncio.run(main())