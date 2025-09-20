"""
Script de teste para verificar se a função get_balance funciona
"""
import asyncio
import logging
from data_collector import data_collector
from config import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_balance():
    """Testa a função get_balance"""
    try:
        print("=== TESTE DE CONEXÃO E SALDO DERIV ===")
        print(f"App ID: {config.deriv.app_id}")
        print(f"Token configurado: {'Sim' if config.deriv.api_token else 'Não'}")
        print(f"URL WebSocket: {config.deriv.websocket_url}")
        print()
        
        # Conectar
        print("1. Conectando à API da Deriv...")
        success = data_collector.connect()
        if not success:
            print("❌ Falha na conexão")
            return
        print("✅ Conectado com sucesso")
        
        # Aguardar conexão
        await asyncio.sleep(2)
        
        # Autorizar
        print("2. Autorizando...")
        if not config.deriv.api_token:
            print("❌ Token da API não configurado no arquivo .env")
            return
            
        authorized = data_collector.authorize()
        if not authorized:
            print("❌ Falha na autorização - verifique o token")
            return
        print("✅ Autorizado com sucesso")
        
        # Aguardar autorização
        await asyncio.sleep(2)
        
        # Obter saldo
        print("3. Obtendo saldo...")
        balance = await data_collector.get_balance()
        
        if balance and 'error' not in balance:
            print("✅ Saldo obtido com sucesso!")
            if 'balance' in balance:
                print(f"💰 Saldo: {balance['balance']}")
            else:
                print(f"📊 Resposta completa: {balance}")
        else:
            print("❌ Não foi possível obter o saldo")
            if balance and 'error' in balance:
                error_msg = balance['error'].get('message', 'Erro desconhecido')
                print(f"🚫 Erro: {error_msg}")
                
                if 'oauth token' in error_msg.lower():
                    print("\n💡 SOLUÇÃO:")
                    print("1. Acesse https://app.deriv.com/account/api-token")
                    print("2. Gere um novo token da API")
                    print("3. Atualize o DERIV_API_TOKEN no arquivo .env")
                    print("4. Reinicie o dashboard")
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
    finally:
        # Desconectar
        print("4. Desconectando...")
        data_collector.disconnect()
        print("✅ Desconectado")

if __name__ == "__main__":
    asyncio.run(test_balance())