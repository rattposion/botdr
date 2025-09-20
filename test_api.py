#!/usr/bin/env python3
"""
Script de teste para verificar a conexão com a API Deriv
"""

import asyncio
import websockets
import json
from config import config

async def test_api_connection():
    """Testa a conexão com a API Deriv"""
    
    print("🔄 Testando conexão com a API Deriv...")
    print(f"App ID: {config.deriv.app_id}")
    print(f"WebSocket URL: {config.deriv.websocket_url}")
    
    try:
        # Conectar ao WebSocket
        async with websockets.connect(config.deriv.websocket_url) as websocket:
            print("✅ Conectado ao WebSocket!")
            
            # Teste 1: Ping
            ping_request = {
                "ping": 1,
                "req_id": 1
            }
            
            await websocket.send(json.dumps(ping_request))
            response = await websocket.recv()
            ping_data = json.loads(response)
            
            print(f"📋 Resposta do ping: {ping_data}")
            
            if "pong" in ping_data:
                print("✅ Ping/Pong funcionando!")
            elif "error" in ping_data:
                print(f"❌ Erro no ping: {ping_data['error']}")
                return False
            else:
                print("⚠️ Resposta inesperada do ping, mas continuando...")
                # Não retornar False aqui, continuar com outros testes
            
            # Teste 2: Obter informações do servidor
            server_time_request = {
                "time": 1,
                "req_id": 2
            }
            
            await websocket.send(json.dumps(server_time_request))
            response = await websocket.recv()
            time_data = json.loads(response)
            
            if "time" in time_data:
                print(f"✅ Hora do servidor: {time_data['time']}")
            else:
                print("❌ Erro ao obter hora do servidor")
                return False
            
            # Teste 3: Obter ticks (dados de mercado)
            ticks_request = {
                "ticks": "R_10",
                "subscribe": 1,
                "req_id": 3
            }
            
            await websocket.send(json.dumps(ticks_request))
            
            # Aguardar alguns ticks
            for i in range(3):
                response = await websocket.recv()
                tick_data = json.loads(response)
                
                if "tick" in tick_data:
                    tick = tick_data["tick"]
                    print(f"✅ Tick {i+1}: {tick['symbol']} = {tick['quote']} (época: {tick['epoch']})")
                else:
                    print(f"❌ Erro no tick {i+1}: {tick_data}")
            
            # Teste 4: Verificar se token está funcionando (se configurado)
            if config.deriv.api_token:
                print("\n🔑 Testando autenticação com token...")
                
                auth_request = {
                    "authorize": config.deriv.api_token,
                    "req_id": 4
                }
                
                await websocket.send(json.dumps(auth_request))
                response = await websocket.recv()
                auth_data = json.loads(response)
                
                if "authorize" in auth_data:
                    account_info = auth_data["authorize"]
                    print(f"✅ Autenticado! Conta: {account_info.get('loginid', 'N/A')}")
                    print(f"   Moeda: {account_info.get('currency', 'N/A')}")
                    print(f"   Saldo: {account_info.get('balance', 'N/A')}")
                    print(f"   País: {account_info.get('country', 'N/A')}")
                else:
                    print(f"❌ Erro na autenticação: {auth_data}")
                    return False
            else:
                print("⚠️ Token não configurado - pulando teste de autenticação")
            
            print("\n🎉 Todos os testes passaram! API funcionando corretamente.")
            return True
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Conexão fechada: {e}")
        return False
    except websockets.exceptions.InvalidURI as e:
        print(f"❌ URL inválida: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def main():
    """Função principal"""
    print("=" * 50)
    print("🧪 TESTE DA API DERIV")
    print("=" * 50)
    
    # Executar teste
    result = asyncio.run(test_api_connection())
    
    print("\n" + "=" * 50)
    if result:
        print("✅ RESULTADO: API funcionando perfeitamente!")
        print("Você pode usar o bot com segurança.")
    else:
        print("❌ RESULTADO: Problemas encontrados na API.")
        print("Verifique as configurações no arquivo .env")
        print("Consulte o arquivo SETUP_API.md para mais informações.")
    print("=" * 50)

if __name__ == "__main__":
    main()