"""
Script Principal - Deriv AI Trading Bot
Ponto de entrada principal do sistema de trading automatizado
"""
import asyncio
import argparse
import sys
import os
from datetime import datetime
from typing import Optional

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports locais
from config import config, load_config_from_env, validate_config
from utils import trading_logger, cleanup_old_files, get_logger
from trader import TradingExecutor, start_automated_trading
from data_collector import DerivDataCollector, start_data_collection
from ml_model import TradingMLModel, create_and_train_model
from backtester import run_simple_backtest
from dashboard import main as run_dashboard

def print_banner():
    """Exibe banner do sistema"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    DERIV AI TRADING BOT                      ║
    ║                                                              ║
    ║  🤖 Sistema de Trading Automatizado com Inteligência        ║
    ║     Artificial para a plataforma Deriv                      ║
    ║                                                              ║
    ║  Desenvolvido com: Python, LightGBM, WebSocket, Streamlit   ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def setup_environment():
    """Configura ambiente inicial"""
    print("🔧 Configurando ambiente...")
    
    # Setup de logging
    trading_logger.setup_logging()
    get_logger().info("Sistema iniciado")
    
    # Limpeza de arquivos antigos
    cleanup_old_files()
    
    # Carregar configurações do ambiente
    load_config_from_env()
    
    # Verificar configurações
    try:
        validate_config()
    except ValueError as e:
        get_logger().error(f"Configurações inválidas: {e}")
        print(f"❌ {e}")
        sys.exit(1)
    
    print("✅ Ambiente configurado com sucesso!")

async def collect_data_mode():
    """Modo de coleta de dados"""
    print("📊 Iniciando coleta de dados...")
    
    try:
        collector = DerivDataCollector()
        
        # Conectar
        await collector.connect()
        
        # Autorizar se necessário
        if config.deriv.api_token:
            await collector.authorize()
        
        # Iniciar coleta
        await start_data_collection(
            symbol=config.trading.symbol,
            duration_hours=24  # Coletar por 24 horas
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ Coleta interrompida pelo usuário")
    except Exception as e:
        get_logger().error(f"Erro na coleta de dados: {e}")
        print(f"❌ Erro na coleta: {e}")

def train_model_mode():
    """Modo de treinamento do modelo"""
    print("🧠 Iniciando treinamento do modelo...")
    
    try:
        # Verificar se existem dados
        data_file = f"data/{config.trading.symbol}_ticks.csv"
        
        if not os.path.exists(data_file):
            print(f"❌ Arquivo de dados não encontrado: {data_file}")
            print("💡 Execute primeiro o modo de coleta de dados")
            return
        
        # Treinar modelo
        model = create_and_train_model(data_file)
        
        if model:
            print("✅ Modelo treinado com sucesso!")
            
            # Mostrar informações do modelo
            info = model.get_model_info()
            print(f"📊 Acurácia: {info['accuracy']:.2%}")
            print(f"📈 Features importantes: {', '.join(info['top_features'][:5])}")
        else:
            print("❌ Falha no treinamento do modelo")
            
    except Exception as e:
        get_logger().error(f"Erro no treinamento: {e}")
        print(f"❌ Erro no treinamento: {e}")

def backtest_mode():
    """Modo de backtest"""
    print("🧪 Iniciando backtest...")
    
    try:
        # Verificar se existem dados
        data_file = f"data/{config.trading.symbol}_ticks.csv"
        
        if not os.path.exists(data_file):
            print(f"❌ Arquivo de dados não encontrado: {data_file}")
            return
        
        # Executar backtest
        results = run_simple_backtest(
            data_file=data_file,
            start_date=datetime.now().date().replace(day=1),  # Início do mês
            end_date=datetime.now().date()
        )
        
        if results:
            print("✅ Backtest concluído!")
            print(f"📊 Total de trades: {results.total_trades}")
            print(f"💰 PnL total: ${results.total_pnl:.2f}")
            print(f"📈 Win rate: {results.win_rate:.2%}")
            print(f"📉 Max drawdown: {results.max_drawdown:.2%}")
        else:
            print("❌ Falha no backtest")
            
    except Exception as e:
        get_logger().error(f"Erro no backtest: {e}")
        print(f"❌ Erro no backtest: {e}")

async def trading_mode():
    """Modo de trading automático"""
    print("🚀 Iniciando trading automático...")
    
    try:
        # Verificar se modelo existe
        model_file = f"models/{config.trading.symbol}_model.pkl"
        
        if not os.path.exists(model_file):
            print("❌ Modelo não encontrado!")
            print("💡 Execute primeiro o treinamento do modelo")
            return
        
        print("⚠️  ATENÇÃO: Trading automático ativo!")
        print(f"💰 Stake inicial: ${config.trading.initial_stake}")
        print(f"🛡️  Perda máxima diária: ${config.trading.max_daily_loss}")
        print(f"📊 Trades máximos diários: {config.trading.max_daily_trades}")
        
        # Confirmar início
        if config.environment == "real":
            confirm = input("\n🔴 CONTA REAL! Confirma início do trading? (sim/não): ")
            if confirm.lower() not in ['sim', 's', 'yes', 'y']:
                print("❌ Trading cancelado pelo usuário")
                return
        
        # Iniciar trading
        await start_automated_trading()
        
    except KeyboardInterrupt:
        print("\n⏹️ Trading interrompido pelo usuário")
    except Exception as e:
        get_logger().error(f"Erro no trading: {e}")
        print(f"❌ Erro no trading: {e}")

def dashboard_mode(host="0.0.0.0", port=8501):
    """Modo dashboard"""
    print("📊 Iniciando dashboard...")
    
    try:
        # Executar dashboard Streamlit
        import subprocess
        
        dashboard_file = os.path.join(os.path.dirname(__file__), "dashboard.py")
        
        print("🌐 Abrindo dashboard no navegador...")
        print(f"🔗 URL: http://{host}:{port}")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
    except Exception as e:
        get_logger().error(f"Erro no dashboard: {e}")
        print(f"❌ Erro no dashboard: {e}")

def status_mode():
    """Modo de status do sistema"""
    print("📋 Status do Sistema")
    print("=" * 50)
    
    # Verificar arquivos essenciais
    symbol = config.trading.symbols[0] if config.trading.symbols else config.deriv.default_symbol
    files_to_check = [
        ("Configuração", ".env"),
        ("Dados", f"data/{symbol}_ticks.csv"),
        ("Modelo", f"models/{symbol}_model.pkl"),
        ("Logs", "logs/bot.log")
    ]
    
    for name, file_path in files_to_check:
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"{status} {name}: {file_path}")
    
    print("\n📊 Configurações Atuais:")
    print(f"🎯 Símbolo: {symbol}")
    print(f"💰 Stake: ${config.trading.initial_stake}")
    print(f"🛡️  Max Loss: ${config.trading.max_daily_loss}")
    print(f"📈 Max Trades: {config.trading.max_daily_trades}")
    print(f"🎲 Martingale: {'Ativo' if config.trading.enable_martingale else 'Inativo'}")
    print(f"🌍 Ambiente: {config.environment}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Deriv AI Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --mode collect     # Coletar dados
  python main.py --mode train       # Treinar modelo
  python main.py --mode backtest    # Executar backtest
  python main.py --mode trade       # Trading automático
  python main.py --mode dashboard   # Abrir dashboard
  python main.py --mode status      # Ver status do sistema
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["collect", "train", "backtest", "trade", "dashboard", "status"],
        default="status",
        help="Modo de operação (padrão: status)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Arquivo de configuração personalizado"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        help="Símbolo para trading (ex: R_50)"
    )
    
    parser.add_argument(
        "--stake",
        type=float,
        help="Stake inicial personalizado"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=config.dashboard.host,
        help="Host para o dashboard (padrão: config.dashboard.host)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=config.dashboard.port,
        help="Porta para o dashboard (padrão: config.dashboard.port)"
    )
    
    args = parser.parse_args()
    
    # Banner
    print_banner()
    
    # Carregar configuração personalizada se especificada
    if args.config:
        load_config_from_env()
    
    # Sobrescrever configurações via argumentos
    if args.symbol:
        config.trading.symbol = args.symbol
    
    if args.stake:
        config.trading.initial_stake = args.stake
    
    # Setup do ambiente
    setup_environment()
    
    # Executar modo selecionado
    try:
        if args.mode == "collect":
            asyncio.run(collect_data_mode())
        
        elif args.mode == "train":
            train_model_mode()
        
        elif args.mode == "backtest":
            backtest_mode()
        
        elif args.mode == "trade":
            asyncio.run(trading_mode())
        
        elif args.mode == "dashboard":
            dashboard_mode(host=args.host, port=args.port)
        
        elif args.mode == "status":
            status_mode()
        
    except KeyboardInterrupt:
        print("\n👋 Sistema encerrado pelo usuário")
    except Exception as e:
        get_logger().error(f"Erro fatal: {e}")
        print(f"💥 Erro fatal: {e}")
        sys.exit(1)
    
    print("\n✅ Execução finalizada")

if __name__ == "__main__":
    main()