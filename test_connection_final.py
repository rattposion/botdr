#!/usr/bin/env python3
"""
Teste final para verificar se a conexão WebSocket está funcionando
"""

import asyncio
import time
from data_collector import data_collector

async def test_final_connection():
    """Teste completo da conexão"""
    print("🔄 Iniciando teste final da conexão...")
    
    try:
        # Conectar
        print("1. Conectando...")
        result = data_collector.connect()
        print(f"   Resultado da conexão: {result}")
        
        if result:
            print("✅ Conexão estabelecida com sucesso!")
            
            # Aguardar um pouco para garantir que a autorização foi processada
            print("2. Aguardando autorização...")
            await asyncio.sleep(3)
            
            # Verificar se está autorizado
            if hasattr(data_collector, 'is_authorized') and data_collector.is_authorized:
                print("✅ Autorização confirmada!")
            else:
                print("⚠️  Status de autorização não confirmado")
            
            # Testar uma operação simples
            print("3. Testando operação básica...")
            try:
                # Tentar obter o tempo do servidor
                server_time = data_collector.get_server_time()
                if server_time:
                    print(f"✅ Tempo do servidor obtido: {server_time}")
                else:
                    print("⚠️  Não foi possível obter o tempo do servidor")
            except Exception as e:
                print(f"⚠️  Erro ao obter tempo do servidor: {e}")
            
            # Desconectar
            print("4. Desconectando...")
            data_collector.disconnect()
            print("✅ Desconectado com sucesso!")
            
        else:
            print("❌ Falha na conexão")
            
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Função principal"""
    print("=" * 50)
    print("TESTE FINAL DE CONEXÃO DERIV WEBSOCKET")
    print("=" * 50)
    
    # Executar teste
    asyncio.run(test_final_connection())
    
    print("\n" + "=" * 50)
    print("TESTE CONCLUÍDO")
    print("=" * 50)

if __name__ == "__main__":
    main()