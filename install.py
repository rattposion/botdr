"""
Script de Instalação - Deriv AI Trading Bot
Automatiza a configuração inicial do sistema
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Exibe banner de instalação"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 INSTALAÇÃO - DERIV AI TRADING BOT            ║
    ║                                                              ║
    ║  🚀 Configuração automática do ambiente de trading          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Verifica versão do Python"""
    print("🐍 Verificando versão do Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário!")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")

def create_directories():
    """Cria diretórios necessários"""
    print("📁 Criando estrutura de diretórios...")
    
    directories = [
        "data",
        "models", 
        "logs",
        "logs/daily_reports",
        "logs/performance",
        "backtest_results",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")

def install_dependencies():
    """Instala dependências"""
    print("📦 Instalando dependências...")
    
    try:
        # Atualizar pip
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True)
        
        # Instalar requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ Dependências instaladas com sucesso!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        sys.exit(1)

def setup_environment_file():
    """Configura arquivo .env"""
    print("⚙️ Configurando arquivo de ambiente...")
    
    if os.path.exists(".env"):
        print("   ⚠️  Arquivo .env já existe, pulando...")
        return
    
    if not os.path.exists(".env.example"):
        print("   ❌ Arquivo .env.example não encontrado!")
        return
    
    # Copiar .env.example para .env
    shutil.copy(".env.example", ".env")
    print("   ✅ Arquivo .env criado a partir do .env.example")
    
    # Solicitar configurações básicas
    print("\n🔧 Configuração básica (pressione Enter para usar padrão):")
    
    configs = {}
    
    # App ID
    app_id = input("   Deriv App ID (obrigatório): ").strip()
    if app_id:
        configs["DERIV_APP_ID"] = app_id
    
    # API Token
    api_token = input("   Deriv API Token (opcional para demo): ").strip()
    if api_token:
        configs["DERIV_API_TOKEN"] = api_token
    
    # Stake inicial
    stake = input("   Stake inicial em $ (padrão: 1.0): ").strip()
    if stake:
        try:
            float(stake)
            configs["INITIAL_STAKE"] = stake
        except ValueError:
            print("   ⚠️  Valor inválido para stake, usando padrão")
    
    # Perda máxima
    max_loss = input("   Perda máxima diária em $ (padrão: 50.0): ").strip()
    if max_loss:
        try:
            float(max_loss)
            configs["MAX_DAILY_LOSS"] = max_loss
        except ValueError:
            print("   ⚠️  Valor inválido para perda máxima, usando padrão")
    
    # Ambiente
    env = input("   Ambiente (demo/real) (padrão: demo): ").strip().lower()
    if env in ["demo", "real"]:
        configs["ENVIRONMENT"] = env
    
    # Atualizar arquivo .env
    if configs:
        update_env_file(configs)

def update_env_file(configs):
    """Atualiza arquivo .env com configurações"""
    try:
        with open(".env", "r") as f:
            content = f.read()
        
        for key, value in configs.items():
            # Substituir ou adicionar configuração
            if f"{key}=" in content:
                # Substituir linha existente
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith(f"{key}="):
                        lines[i] = f"{key}={value}"
                        break
                content = "\n".join(lines)
            else:
                # Adicionar nova linha
                content += f"\n{key}={value}"
        
        with open(".env", "w") as f:
            f.write(content)
        
        print("   ✅ Configurações salvas no arquivo .env")
        
    except Exception as e:
        print(f"   ❌ Erro ao atualizar .env: {e}")

def test_installation():
    """Testa a instalação"""
    print("🧪 Testando instalação...")
    
    try:
        # Testar imports principais
        import pandas
        import numpy
        import lightgbm
        import streamlit
        import websockets
        print("   ✅ Imports principais - OK")
        
        # Testar configuração
        from config import config
        print("   ✅ Configuração - OK")
        
        # Testar módulos locais
        from utils import trading_logger
        from data_collector import DerivDataCollector
        from ml_model import TradingMLModel
        print("   ✅ Módulos locais - OK")
        
        print("✅ Instalação testada com sucesso!")
        
    except ImportError as e:
        print(f"   ❌ Erro de import: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Erro no teste: {e}")
        return False
    
    return True

def show_next_steps():
    """Mostra próximos passos"""
    print("\n🎉 Instalação concluída com sucesso!")
    print("\n📋 Próximos passos:")
    print("   1. Configure suas credenciais Deriv no arquivo .env")
    print("   2. Verifique o status: python main.py --mode status")
    print("   3. Colete dados: python main.py --mode collect")
    print("   4. Treine o modelo: python main.py --mode train")
    print("   5. Execute backtest: python main.py --mode backtest")
    print("   6. Inicie trading: python main.py --mode trade")
    print("   7. Abra dashboard: python main.py --mode dashboard")
    
    print("\n📚 Documentação:")
    print("   - README.md: Documentação completa")
    print("   - .env.example: Exemplo de configurações")
    
    print("\n🔗 Links úteis:")
    print("   - Deriv API: https://developers.deriv.com/")
    print("   - App Registration: https://app.deriv.com/account/api-token")
    
    print("\n⚠️  IMPORTANTE:")
    print("   - Sempre teste em conta DEMO primeiro")
    print("   - Nunca invista mais do que pode perder")
    print("   - Monitore constantemente o sistema")

def main():
    """Função principal de instalação"""
    print_banner()
    
    try:
        # Verificações
        check_python_version()
        
        # Configuração
        create_directories()
        install_dependencies()
        setup_environment_file()
        
        # Teste
        if test_installation():
            show_next_steps()
        else:
            print("\n❌ Instalação falhou nos testes!")
            print("   Verifique os erros acima e tente novamente")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Instalação cancelada pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erro durante instalação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()