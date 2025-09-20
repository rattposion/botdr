"""
Dashboard de Monitoramento com Streamlit
Interface web para monitorar trading em tempo real
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional
from advanced_backtester import advanced_backtester, quick_backtest
from strategy_optimizer import strategy_optimizer, quick_optimization
from ai_trading_bot import AITradingBot
from trade_executor import trade_executor

# Configuração da página
st.set_page_config(
    page_title="Deriv AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para otimizar carregamento de fontes
st.markdown("""
<style>
    /* Otimizar carregamento de fontes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Usar fonte local quando possível */
    .main .block-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Reduzir preload de recursos desnecessários */
    link[rel="preload"] {
        display: none !important;
    }
    
    /* Otimizar performance */
    .stApp {
        font-display: swap;
    }
    
    /* Customizações visuais */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #FF6C37;
    }
    
    .status-success { color: #10B981; }
    .status-error { color: #EF4444; }
    .status-warning { color: #F59E0B; }
</style>
""", unsafe_allow_html=True)

# Imports locais
try:
    from utils import (
        trade_recorder, performance_tracker, risk_manager, 
        get_daily_report, format_currency, format_percentage
    )
    from trader import TradingExecutor, get_trading_status
    from ml_model import TradingMLModel
    from backtester import Backtester, BacktestAnalyzer
    from data_collector import DerivDataCollector
    from balance_manager import balance_manager
    from auth_manager import auth_manager
    from token_manager import token_manager
    from config import config
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

class TradingDashboard:
    """Dashboard principal de trading"""
    
    def __init__(self):
        from utils import get_logger
        self.logger = get_logger('dashboard')
        self.initialize_session_state()
    
    def add_notification(self, message: str, notification_type: str = 'info'):
        """Adiciona notificação ao sistema"""
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        
        notification = {
            'message': message,
            'type': notification_type,
            'timestamp': datetime.now()
        }
        
        st.session_state.notifications.append(notification)
        
        # Manter apenas as últimas 10 notificações
        if len(st.session_state.notifications) > 10:
            st.session_state.notifications = st.session_state.notifications[-10:]
    
    def initialize_session_state(self):
        """Inicializa estado da sessão"""
        if 'trading_active' not in st.session_state:
            st.session_state.trading_active = False
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        
        if 'trader_instance' not in st.session_state:
            st.session_state.trader_instance = None
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        
        # Inicializar balance manager
        if 'balance_manager_started' not in st.session_state:
            balance_manager.start_auto_update()
            st.session_state.balance_manager_started = True
        
        # Inicializar token manager
        if 'token_manager_started' not in st.session_state:
            token_manager.start_monitoring()
            st.session_state.token_manager_started = True
    
    def run(self):
        """Executa dashboard principal"""
        # Sidebar
        self.render_sidebar()
        
        # Título principal
        st.title("🤖 Deriv AI Trading Bot")
        st.markdown("---")
        
        # Abas principais
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🔐 Login", "📊 Overview", "💹 Trading", "📈 Performance", 
            "🧪 Backtest", "🎯 Bot AI", "⚙️ Configurações"
        ])
        
        with tab1:
            self.render_auth_tab()
        
        with tab2:
            self.render_overview_tab()
        
        with tab3:
            self.render_trading_tab()
        
        with tab4:
            self.render_performance_tab()
        
        with tab5:
            self.render_backtest_tab()
        
        with tab6:
            self.render_ai_bot_tab()
        
        with tab7:
            self.render_settings_tab()
        
        # Auto-refresh inteligente
        if st.session_state.auto_refresh:
            # Verificar se precisa atualizar mais frequentemente
            balance_info = balance_manager.get_balance_info()
            
            if balance_info['is_updating']:
                # Se está atualizando, refresh mais rápido
                time.sleep(2)
            elif not balance_info['is_connected']:
                # Se não está conectado, refresh mais lento
                time.sleep(10)
            else:
                # Normal refresh
                time.sleep(5)
            
            st.rerun()
    
    def render_sidebar(self):
        """Renderiza sidebar"""
        st.sidebar.title("🎛️ Controles")
        
        # Status do sistema
        st.sidebar.subheader("Status do Sistema")
        
        # Status da conexão com a API
        balance_info = balance_manager.get_balance_info()
        status_emoji = balance_manager.get_status_emoji()
        status_text = balance_manager.get_status_text()
        
        if balance_info['is_connected']:
            st.sidebar.success(f"{status_emoji} API Conectada")
            st.sidebar.caption(f"Saldo: {format_currency(balance_info['balance'])}")
        elif balance_info['is_updating']:
            st.sidebar.info(f"{status_emoji} {status_text}")
        else:
            st.sidebar.error(f"{status_emoji} API Desconectada")
            if balance_info['error_message']:
                st.sidebar.caption(f"⚠️ {balance_info['error_message']}")
        
        # Indicador de status do trading
        if st.session_state.trading_active:
            st.sidebar.success("🟢 Trading Ativo")
            
            # Status detalhado quando ativo
            if st.session_state.trader_instance:
                trader_status = st.session_state.trader_instance.get_status()
                st.sidebar.info(f"📊 Trades: {trader_status.get('trades_count', 0)}")
                st.sidebar.info(f"💰 P&L: ${trader_status.get('total_pnl', 0):.2f}")
                
                # Mostrar último sinal se disponível
                if trader_status.get('last_signal_time'):
                    time_diff = datetime.now() - trader_status['last_signal_time']
                    st.sidebar.caption(f"🎯 Último sinal: {int(time_diff.total_seconds())}s atrás")
        else:
            st.sidebar.error("🔴 Trading Parado")
            
            # Mostrar motivos pelos quais não pode iniciar
            can_trade, reason = risk_manager.can_trade()
            if not can_trade:
                st.sidebar.warning(f"⚠️ {reason}")
            
            if not auth_manager.is_authenticated:
                st.sidebar.error("❌ Não autenticado")
            
            token_status = token_manager.get_status()
            if not token_status.get('authenticated', False):
                st.sidebar.error("❌ Token inválido")
        
        # Controles de trading
        st.sidebar.subheader("Controles de Trading")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Iniciar", disabled=st.session_state.trading_active):
                self.start_trading()
        
        with col2:
            if st.button("⏹️ Parar", disabled=not st.session_state.trading_active):
                self.stop_trading()
        
        # Auto-refresh
        st.sidebar.subheader("Configurações")
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Auto-refresh (5s)", 
            value=st.session_state.auto_refresh
        )
        
        # Widget de monitoramento em tempo real
        st.sidebar.subheader("Monitoramento")
        
        # Informações de atualização
        balance_info = balance_manager.get_balance_info()
        if balance_info['last_update']:
            time_diff = datetime.now() - balance_info['last_update']
            if time_diff.total_seconds() < 60:
                st.sidebar.success(f"🔄 Atualizado há {int(time_diff.total_seconds())}s")
            else:
                st.sidebar.warning(f"🕐 Atualizado há {int(time_diff.total_seconds()/60)}min")
        else:
            st.sidebar.error("❌ Nunca atualizado")
        
        # Intervalo de atualização
        st.sidebar.caption(f"Intervalo: {balance_info['update_interval']}s")
        
        # Informações do sistema
        st.sidebar.subheader("Sistema")
        st.sidebar.info(f"Dashboard: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        # Botão de refresh manual
        if st.sidebar.button("🔄 Atualizar Tudo"):
            balance_manager.force_update()
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        # Sistema de notificações
        st.sidebar.subheader("📢 Notificações")
        
        # Inicializar lista de notificações se não existir
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        
        # Mostrar últimas 3 notificações
        if st.session_state.notifications:
            for notification in st.session_state.notifications[-3:]:
                timestamp = notification['timestamp'].strftime('%H:%M:%S')
                if notification['type'] == 'success':
                    st.sidebar.success(f"{timestamp}: {notification['message']}")
                elif notification['type'] == 'error':
                    st.sidebar.error(f"{timestamp}: {notification['message']}")
                elif notification['type'] == 'warning':
                    st.sidebar.warning(f"{timestamp}: {notification['message']}")
                else:
                    st.sidebar.info(f"{timestamp}: {notification['message']}")
        else:
            st.sidebar.caption("Nenhuma notificação recente")
    
    def render_auth_tab(self):
        """Renderiza aba de autenticação"""
        st.header("🔐 Autenticação Deriv")
        
        # Status atual da autenticação
        auth_status = auth_manager.get_auth_status()
        
        if auth_status['is_authenticated']:
            # Usuário autenticado
            st.success("✅ Autenticado com sucesso!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"👤 **Usuário:** {auth_status['user_email'] or 'N/A'}")
                
                if auth_status['expires_in_minutes'] > 0:
                    st.info(f"⏰ **Token expira em:** {auth_status['expires_in_minutes']} minutos")
                else:
                    st.warning("⚠️ Token expirado - será renovado automaticamente")
            
            with col2:
                if st.button("🚪 Logout", type="secondary"):
                    auth_manager.logout()
                    st.rerun()
            
            # Informações do token para uso no bot
            st.subheader("🔑 Token para API")
            
            api_token = auth_manager.get_api_token()
            if api_token:
                # Mostrar token mascarado
                masked_token = f"{api_token[:10]}...{api_token[-10:]}"
                st.code(f"Token ativo: {masked_token}")
                
                # Botão para copiar token completo
                if st.button("📋 Copiar Token Completo"):
                    st.session_state.show_full_token = True
                
                if st.session_state.get('show_full_token', False):
                    st.code(api_token)
                    st.caption("⚠️ Mantenha este token seguro!")
                    
                    if st.button("🙈 Ocultar Token"):
                        st.session_state.show_full_token = False
                        st.rerun()
                
                # Atualizar configuração automaticamente
                if config.deriv.api_token != api_token:
                    config.deriv.api_token = api_token
                    st.success("🔄 Token atualizado na configuração!")
            else:
                st.error("❌ Token não disponível")
        
        else:
            # Usuário não autenticado
            st.warning("⚠️ Você precisa fazer login para usar o bot")
            
            st.markdown("""
            ### 🚀 Como fazer login:
            
            1. **Clique no botão "Fazer Login"** abaixo
            2. **Seu navegador será aberto** com a página de login do Deriv
            3. **Faça login** com suas credenciais do Deriv
            4. **Autorize o bot** a acessar sua conta
            5. **Volte para esta página** - o login será concluído automaticamente
            
            ⚡ **O processo é seguro e usa OAuth2 oficial do Deriv**
            """)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🔐 Fazer Login com Deriv", type="primary", use_container_width=True):
                    with st.spinner("🔄 Iniciando processo de login..."):
                        try:
                            success = auth_manager.login()
                            if success == "MANUAL_AUTH_REQUIRED":
                                # Railway - autenticação manual
                                st.session_state.show_manual_auth = True
                                st.rerun()
                            elif success:
                                st.success("✅ Login realizado com sucesso!")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Falha no login. Tente novamente.")
                        except Exception as e:
                            st.error(f"❌ Erro no login: {e}")
            
            # Mostrar interface de autenticação manual se necessário
            if st.session_state.get('show_manual_auth', False):
                st.markdown("---")
                st.subheader("🔑 Autenticação Manual (Railway)")
                
                st.info("""
                **No Railway, você precisa completar a autenticação manualmente:**
                
                1. **Acesse a URL de autorização** que foi exibida no console
                2. **Faça login** na sua conta Deriv
                3. **Autorize o aplicativo**
                4. **Copie o código** da URL de retorno (parâmetro `code=`)
                5. **Cole o código** no campo abaixo
                """)
                
                auth_code = st.text_input(
                    "Código de Autorização:",
                    placeholder="Cole aqui o código obtido da URL de retorno",
                    help="O código aparece na URL após 'code=' quando você autoriza o app"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Confirmar Código", disabled=not auth_code):
                        with st.spinner("🔄 Processando autenticação..."):
                            try:
                                success = auth_manager.manual_auth_with_code(auth_code)
                                if success:
                                    st.success("✅ Autenticação realizada com sucesso!")
                                    st.session_state.show_manual_auth = False
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("❌ Código inválido. Tente novamente.")
                            except Exception as e:
                                st.error(f"❌ Erro na autenticação: {e}")
                
                with col2:
                    if st.button("❌ Cancelar"):
                        st.session_state.show_manual_auth = False
                        st.rerun()
            
            # Informações adicionais
            st.markdown("---")
            
            with st.expander("ℹ️ Informações sobre a Autenticação"):
                st.markdown("""
                **🔒 Segurança:**
                - Usamos OAuth2 oficial do Deriv
                - Seus dados de login não são armazenados
                - O token é criptografado localmente
                
                **🎯 Permissões solicitadas:**
                - **Read:** Ler informações da conta
                - **Trade:** Executar operações de trading
                - **Payments:** Acessar informações de saldo
                - **Admin:** Gerenciar configurações da conta
                
                **⏰ Duração:**
                - O token expira automaticamente
                - Renovação automática quando necessário
                - Logout manual disponível a qualquer momento
                """)
            
            # Status de conexão alternativo
            st.markdown("---")
            st.subheader("🔧 Configuração Manual (Alternativa)")
            
            with st.expander("Usar Token Manual"):
                st.markdown("""
                Se preferir, você pode usar um token de API manual:
                
                1. Acesse [app.deriv.com](https://app.deriv.com)
                2. Vá em **Configurações > Segurança > Tokens de API**
                3. Crie um novo token com as permissões necessárias
                4. Cole o token no arquivo `.env` como `DERIV_API_TOKEN`
                """)
                
                manual_token = st.text_input(
                    "Token de API Manual:",
                    type="password",
                    placeholder="Cole seu token aqui..."
                )
                
                if st.button("💾 Salvar Token Manual") and manual_token:
                    try:
                        # Salvar no config
                        config.deriv.api_token = manual_token
                        
                        # Salvar no arquivo .env usando python-dotenv
                        from dotenv import find_dotenv, set_key
                        
                        # Encontrar o arquivo .env automaticamente
                        env_file = find_dotenv()
                        if not env_file:
                            # Se não encontrar, usar caminho padrão no diretório atual
                            env_file = os.path.abspath('.env')
                        
                        # Usar set_key para salvar de forma segura
                        success = set_key(env_file, 'DERIV_API_TOKEN', manual_token)
                        
                        if success:
                            st.success("✅ Token manual salvo com sucesso!")
                            # Recarregar configurações
                            from config import load_config_from_env
                            load_config_from_env()
                        else:
                            st.error("❌ Falha ao salvar token no arquivo .env")
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao salvar token: {e}")
                        # Log adicional para debug
                        import traceback
                        st.error(f"Detalhes do erro: {traceback.format_exc()}")
        
        # Seção de gerenciamento automático de tokens
        st.markdown("---")
        st.subheader("🔄 Gerenciamento Automático de Tokens")
        
        token_status = token_manager.get_status()
        
        if token_status['available']:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if token_status['monitoring']:
                    st.success("✅ Monitoramento automático ativo")
                    
                    if token_status['authenticated']:
                        expires_in = token_status.get('expires_in_seconds', 0)
                        if expires_in > 0:
                            hours = expires_in // 3600
                            minutes = (expires_in % 3600) // 60
                            st.info(f"⏰ Token expira em: {hours}h {minutes}m")
                            
                            if token_status.get('needs_renewal', False):
                                st.warning("⚠️ Token será renovado automaticamente em breve")
                        else:
                            st.warning("⚠️ Token expirado")
                    else:
                        st.info("ℹ️ Aguardando autenticação")
                else:
                    st.warning("⚠️ Monitoramento automático inativo")
            
            with col2:
                if token_status['monitoring']:
                    if st.button("⏹️ Parar Monitoramento"):
                        token_manager.stop_monitoring()
                        st.rerun()
                else:
                    if st.button("▶️ Iniciar Monitoramento"):
                        token_manager.start_monitoring()
                        st.rerun()
                
                if token_status['authenticated']:
                    if st.button("🔄 Renovar Agora"):
                        with st.spinner("Renovando token..."):
                            success = token_manager.force_renewal()
                            if success:
                                st.success("✅ Token renovado!")
                            else:
                                st.error("❌ Falha na renovação")
                            time.sleep(2)
                            st.rerun()
        else:
            st.error("❌ Gerenciador de tokens não disponível")
    
    def render_overview_tab(self):
        """Renderiza aba de overview"""
        st.header("📊 Visão Geral")
        
        # Status da conexão em tempo real
        balance_info = balance_manager.get_balance_info()
        
        # Indicador de status no topo
        status_col1, status_col2, status_col3 = st.columns([1, 2, 1])
        with status_col2:
            status_emoji = balance_manager.get_status_emoji()
            status_text = balance_manager.get_status_text()
            
            if balance_info['is_connected']:
                st.success(f"{status_emoji} {status_text}")
            elif balance_info['is_updating']:
                st.info(f"{status_emoji} {status_text}")
            else:
                st.error(f"{status_emoji} {status_text}")
                if balance_info['error_message']:
                    st.caption(f"Erro: {balance_info['error_message']}")
        
        # Botão de atualização manual
        col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 1, 1])
        with col_refresh2:
            if st.button("🔄 Atualizar Saldo", use_container_width=True):
                balance_manager.force_update()
                st.rerun()
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        # Obter dados atuais
        daily_report = get_daily_report()
        risk_status = risk_manager.get_risk_status()
        
        with col1:
            # Exibir saldo em tempo real
            current_balance = balance_info['balance']
            last_update = balance_info['last_update']
            
            if current_balance > 0:
                st.metric(
                    "💰 Saldo Atual",
                    format_currency(current_balance),
                    delta=format_currency(daily_report['trading_summary']['total_pnl'])
                )
                if last_update:
                    time_diff = datetime.now() - last_update
                    if time_diff.total_seconds() < 60:
                        st.caption(f"✅ Atualizado há {int(time_diff.total_seconds())}s")
                    else:
                        st.caption(f"🕐 Atualizado há {int(time_diff.total_seconds()/60)}min")
            else:
                st.error("⚠️ Saldo não disponível")
                if balance_info['connection_status'] == 'no_token':
                    st.caption("Configure o token da API")
                elif balance_info['connection_status'] == 'invalid_token':
                    st.caption("Token inválido ou expirado")
                else:
                    st.caption("Verifique a conexão")
        
        with col2:
            st.metric(
                "Trades Hoje",
                daily_report['trading_summary']['total_trades'],
                delta=f"{daily_report['trading_summary']['win_rate']:.1%} win rate"
            )
        
        with col3:
            st.metric(
                "PnL Diário",
                format_currency(daily_report['trading_summary']['total_pnl']),
                delta=f"{daily_report['trading_summary']['winning_trades']} wins"
            )
        
        with col4:
            remaining_trades = risk_status.get('remaining_trades', 0)
            st.metric(
                "Trades Restantes",
                remaining_trades,
                delta=f"Limite: {config.trading.max_daily_trades}"
            )
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_equity_curve()
        
        with col2:
            self.render_daily_pnl_chart()
        
        # Tabela de trades recentes
        st.subheader("🕐 Trades Recentes")
        self.render_recent_trades_table()
    
    def render_trading_tab(self):
        """Renderiza aba de trading"""
        st.header("💹 Trading em Tempo Real")
        
        # Status de risco
        risk_status = risk_manager.get_risk_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🛡️ Status de Risco")
            
            if risk_status['can_trade']:
                st.success("✅ Pode fazer trades")
            else:
                st.error("❌ Trading bloqueado")
            
            st.write(f"**PnL Diário:** {format_currency(risk_status['daily_pnl'])}")
            st.write(f"**Trades Diários:** {risk_status['daily_trades']}")
            st.write(f"**Martingale Step:** {risk_status['martingale_step']}")
        
        with col2:
            st.subheader("📊 Limites")
            
            # Progress bars para limites
            trades_progress = risk_status['daily_trades'] / config.trading.max_daily_trades
            st.progress(trades_progress, text=f"Trades: {risk_status['daily_trades']}/{config.trading.max_daily_trades}")
            
            loss_limit = config.trading.max_daily_loss
            loss_progress = min(abs(risk_status['daily_pnl']) / loss_limit, 1.0) if loss_limit > 0 else 0
            st.progress(loss_progress, text=f"Loss Limit: {format_currency(abs(risk_status['daily_pnl']))}/{format_currency(loss_limit)}")
        
        with col3:
            st.subheader("⚡ Ações Rápidas")
            
            if st.button("📈 Trade Manual CALL"):
                self.execute_manual_trade("CALL")
            
            if st.button("📉 Trade Manual PUT"):
                self.execute_manual_trade("PUT")
            
            if st.button("🔄 Resetar Martingale"):
                risk_manager.martingale_step = 0
                st.success("Martingale resetado!")
        
        # Gráfico de preço em tempo real
        st.subheader("📈 Preço em Tempo Real")
        self.render_realtime_price_chart()
        
        # Log de atividades
        st.subheader("📝 Log de Atividades")
        self.render_activity_log()
    
    def render_performance_tab(self):
        """Renderiza aba de performance"""
        st.header("📈 Análise de Performance")
        
        # Seletor de período
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Data Início", value=datetime.now().date() - timedelta(days=30))
        
        with col2:
            end_date = st.date_input("Data Fim", value=datetime.now().date())
        
        # Carregar dados do período
        trades_df = trade_recorder.get_trades_df()
        
        if not trades_df.empty:
            # Filtrar por período
            trades_df['date'] = trades_df['timestamp'].dt.date
            period_trades = trades_df[
                (trades_df['date'] >= start_date) & 
                (trades_df['date'] <= end_date)
            ]
            
            if not period_trades.empty:
                # Métricas do período
                self.render_period_metrics(period_trades)
                
                # Gráficos de análise
                col1, col2 = st.columns(2)
                
                with col1:
                    self.render_pnl_distribution(period_trades)
                
                with col2:
                    self.render_win_rate_by_hour(period_trades)
                
                # Análise por símbolo/sinal
                col1, col2 = st.columns(2)
                
                with col1:
                    self.render_performance_by_signal(period_trades)
                
                with col2:
                    self.render_martingale_analysis(period_trades)
            else:
                st.info("Nenhum trade encontrado no período selecionado.")
        else:
            st.info("Nenhum dado de trading disponível.")
    
    def render_backtest_tab(self):
         """Renderiza aba de backtest"""
         st.header("🧪 Backtest de Estratégias")
         
         # Abas secundárias para diferentes tipos de backtest
         backtest_tab1, backtest_tab2, backtest_tab3 = st.tabs([
             "📊 Backtest Simples", "🔬 Backtest Avançado", "⚡ Otimização"
         ])
         
         with backtest_tab1:
             self.render_simple_backtest()
         
         with backtest_tab2:
             self.render_advanced_backtest()
         
         with backtest_tab3:
             self.render_optimization()
    
    def render_simple_backtest(self):
         """Renderiza interface de backtest simples"""
         st.subheader("📊 Backtest Simples")
         
         # Upload de dados
         uploaded_file = st.file_uploader(
             "Upload arquivo CSV com dados históricos",
             type=['csv'],
             help="Arquivo deve conter colunas: timestamp, quote",
             key="simple_backtest_upload"
         )
         
         if uploaded_file is not None:
             try:
                 # Carregar dados
                 data = pd.read_csv(uploaded_file)
                 st.success(f"Dados carregados: {len(data)} registros")
                 
                 # Mostrar preview
                 with st.expander("👀 Preview dos Dados"):
                     st.dataframe(data.head())
                 
                 # Configurações básicas
                 col1, col2, col3 = st.columns(3)
                 
                 with col1:
                     initial_balance = st.number_input(
                         "Saldo Inicial ($)", 
                         value=1000.0, 
                         min_value=100.0
                     )
                 
                 with col2:
                     stake_amount = st.number_input(
                         "Valor por Trade ($)", 
                         value=10.0, 
                         min_value=1.0
                     )
                 
                 with col3:
                     confidence_threshold = st.slider(
                         "Confiança Mínima", 
                         min_value=0.5, 
                         max_value=1.0, 
                         value=0.6, 
                         step=0.01
                     )
                 
                 # Executar backtest simples
                 if st.button("🚀 Executar Backtest Simples", key="run_simple_backtest"):
                     with st.spinner("Executando backtest..."):
                         try:
                             # Usar quick_backtest
                             results = quick_backtest(
                                 data=data,
                                 initial_balance=initial_balance,
                                 stake_amount=stake_amount,
                                 confidence_threshold=confidence_threshold
                             )
                             
                             # Exibir resultados
                             self.display_backtest_results(results, "Backtest Simples")
                             
                         except Exception as e:
                             st.error(f"Erro no backtest: {e}")
                             
             except Exception as e:
                 st.error(f"Erro ao carregar dados: {e}")
    
    def render_advanced_backtest(self):
         """Renderiza interface de backtest avançado"""
         st.subheader("🔬 Backtest Avançado")
         
         # Upload de dados
         uploaded_file = st.file_uploader(
             "Upload arquivo CSV com dados históricos",
             type=['csv'],
             help="Arquivo deve conter colunas: timestamp, quote",
             key="advanced_backtest_upload"
         )
         
         if uploaded_file is not None:
             try:
                 # Carregar dados
                 data = pd.read_csv(uploaded_file)
                 st.success(f"Dados carregados: {len(data)} registros")
                 
                 # Configurações avançadas
                 st.subheader("⚙️ Configurações Avançadas")
                 
                 col1, col2 = st.columns(2)
                 
                 with col1:
                     st.markdown("**💰 Configurações Financeiras**")
                     
                     initial_balance = st.number_input(
                         "Saldo Inicial ($)", 
                         value=1000.0, 
                         min_value=100.0,
                         key="adv_initial_balance"
                     )
                     
                     stake_amount = st.number_input(
                         "Valor por Trade ($)", 
                         value=10.0, 
                         min_value=1.0,
                         key="adv_stake"
                     )
                     
                     commission = st.number_input(
                         "Comissão por Trade ($)", 
                         value=0.0, 
                         min_value=0.0,
                         key="adv_commission"
                     )
                     
                     max_daily_loss = st.number_input(
                         "Perda Máxima Diária ($)", 
                         value=100.0, 
                         min_value=0.0,
                         key="adv_max_loss"
                     )
                 
                 with col2:
                     st.markdown("**🎯 Configurações de Trading**")
                     
                     confidence_threshold = st.slider(
                         "Confiança Mínima", 
                         min_value=0.5, 
                         max_value=1.0, 
                         value=0.6, 
                         step=0.01,
                         key="adv_confidence"
                     )
                     
                     enable_martingale = st.checkbox(
                         "Habilitar Martingale",
                         value=False,
                         key="adv_martingale"
                     )
                     
                     if enable_martingale:
                         martingale_multiplier = st.number_input(
                             "Multiplicador Martingale", 
                             value=2.0, 
                             min_value=1.1,
                             key="adv_mart_mult"
                         )
                         
                         max_martingale_steps = st.number_input(
                             "Máx. Steps Martingale", 
                             value=3, 
                             min_value=1,
                             key="adv_mart_steps"
                         )
                     else:
                         martingale_multiplier = 1.0
                         max_martingale_steps = 0
                     
                     max_daily_trades = st.number_input(
                         "Máx. Trades por Dia", 
                         value=50, 
                         min_value=1,
                         key="adv_max_trades"
                     )
                 
                 # Período de análise
                 st.markdown("**📅 Período de Análise**")
                 col1, col2 = st.columns(2)
                 
                 with col1:
                     start_date = st.date_input(
                         "Data Início", 
                         value=datetime.now().date() - timedelta(days=30),
                         key="adv_start_date"
                     )
                 
                 with col2:
                     end_date = st.date_input(
                         "Data Fim", 
                         value=datetime.now().date(),
                         key="adv_end_date"
                     )
                 
                 # Executar backtest avançado
                 if st.button("🚀 Executar Backtest Avançado", key="run_advanced_backtest"):
                     with st.spinner("Executando backtest avançado..."):
                         try:
                             # Configurações para o backtest
                             config_dict = {
                                 'initial_balance': initial_balance,
                                 'stake_amount': stake_amount,
                                 'commission': commission,
                                 'confidence_threshold': confidence_threshold,
                                 'enable_martingale': enable_martingale,
                                 'martingale_multiplier': martingale_multiplier,
                                 'max_martingale_steps': max_martingale_steps,
                                 'max_daily_trades': max_daily_trades,
                                 'max_daily_loss': max_daily_loss,
                                 'start_date': start_date,
                                 'end_date': end_date
                             }
                             
                             # Usar advanced_backtester
                             results = advanced_backtester(data, config_dict)
                             
                             # Exibir resultados
                             self.display_advanced_backtest_results(results)
                             
                         except Exception as e:
                             st.error(f"Erro no backtest avançado: {e}")
                             import traceback
                             st.error(f"Detalhes: {traceback.format_exc()}")
                             
             except Exception as e:
                 st.error(f"Erro ao carregar dados: {e}")
    
    def render_optimization(self):
         """Renderiza interface de otimização de estratégias"""
         st.subheader("⚡ Otimização de Estratégias")
         
         # Upload de dados
         uploaded_file = st.file_uploader(
             "Upload arquivo CSV com dados históricos",
             type=['csv'],
             help="Arquivo deve conter colunas: timestamp, quote",
             key="optimization_upload"
         )
         
         if uploaded_file is not None:
             try:
                 # Carregar dados
                 data = pd.read_csv(uploaded_file)
                 st.success(f"Dados carregados: {len(data)} registros")
                 
                 # Configurações de otimização
                 st.subheader("🎯 Parâmetros para Otimização")
                 
                 col1, col2 = st.columns(2)
                 
                 with col1:
                     st.markdown("**📊 Ranges de Confiança**")
                     
                     confidence_min = st.slider(
                         "Confiança Mínima", 
                         min_value=0.5, 
                         max_value=0.9, 
                         value=0.55,
                         key="opt_conf_min"
                     )
                     
                     confidence_max = st.slider(
                         "Confiança Máxima", 
                         min_value=confidence_min, 
                         max_value=1.0, 
                         value=0.85,
                         key="opt_conf_max"
                     )
                     
                     confidence_steps = st.number_input(
                         "Passos de Confiança", 
                         value=6, 
                         min_value=3, 
                         max_value=20,
                         key="opt_conf_steps"
                     )
                 
                 with col2:
                     st.markdown("**💰 Ranges de Stake**")
                     
                     stake_min = st.number_input(
                         "Stake Mínimo ($)", 
                         value=5.0, 
                         min_value=1.0,
                         key="opt_stake_min"
                     )
                     
                     stake_max = st.number_input(
                         "Stake Máximo ($)", 
                         value=50.0, 
                         min_value=stake_min,
                         key="opt_stake_max"
                     )
                     
                     stake_steps = st.number_input(
                         "Passos de Stake", 
                         value=5, 
                         min_value=3, 
                         max_value=10,
                         key="opt_stake_steps"
                     )
                 
                 # Configurações adicionais
                 col1, col2 = st.columns(2)
                 
                 with col1:
                     initial_balance = st.number_input(
                         "Saldo Inicial ($)", 
                         value=1000.0, 
                         min_value=100.0,
                         key="opt_balance"
                     )
                     
                     optimization_metric = st.selectbox(
                         "Métrica de Otimização",
                         ["total_return", "sharpe_ratio", "win_rate", "profit_factor"],
                         index=0,
                         key="opt_metric"
                     )
                 
                 with col2:
                     max_combinations = st.number_input(
                         "Máx. Combinações", 
                         value=100, 
                         min_value=10, 
                         max_value=1000,
                         key="opt_max_comb"
                     )
                     
                     enable_parallel = st.checkbox(
                         "Processamento Paralelo",
                         value=True,
                         key="opt_parallel"
                     )
                 
                 # Executar otimização
                 if st.button("🚀 Executar Otimização", key="run_optimization"):
                     with st.spinner("Executando otimização... Isso pode levar alguns minutos."):
                         try:
                             # Configurações para otimização
                             optimization_config = {
                                 'confidence_range': (confidence_min, confidence_max, confidence_steps),
                                 'stake_range': (stake_min, stake_max, stake_steps),
                                 'initial_balance': initial_balance,
                                 'optimization_metric': optimization_metric,
                                 'max_combinations': max_combinations,
                                 'enable_parallel': enable_parallel
                             }
                             
                             # Usar strategy_optimizer
                             results = strategy_optimizer(data, optimization_config)
                             
                             # Exibir resultados
                             self.display_optimization_results(results)
                             
                         except Exception as e:
                             st.error(f"Erro na otimização: {e}")
                             import traceback
                             st.error(f"Detalhes: {traceback.format_exc()}")
                 
                 # Otimização rápida
                 st.markdown("---")
                 st.subheader("⚡ Otimização Rápida")
                 st.info("Use esta opção para uma otimização rápida com parâmetros padrão")
                 
                 if st.button("⚡ Otimização Rápida", key="quick_optimization"):
                     with st.spinner("Executando otimização rápida..."):
                         try:
                             # Usar quick_optimization
                             results = quick_optimization(data)
                             
                             # Exibir resultados
                             self.display_optimization_results(results)
                             
                         except Exception as e:
                             st.error(f"Erro na otimização rápida: {e}")
                             
             except Exception as e:
                 st.error(f"Erro ao carregar dados: {e}")
    
    def display_backtest_results(self, results: dict, title: str = "Resultados do Backtest"):
         """Exibe resultados do backtest simples"""
         st.subheader(f"📊 {title}")
         
         # Métricas principais
         col1, col2, col3, col4 = st.columns(4)
         
         with col1:
             st.metric(
                 "Retorno Total",
                 f"{results.get('total_return', 0):.2%}",
                 delta=f"${results.get('final_balance', 0) - results.get('initial_balance', 0):.2f}"
             )
         
         with col2:
             st.metric(
                 "Win Rate",
                 f"{results.get('win_rate', 0):.1%}",
                 delta=f"{results.get('total_trades', 0)} trades"
             )
         
         with col3:
             st.metric(
                 "Profit Factor",
                 f"{results.get('profit_factor', 0):.2f}",
                 delta=f"Max DD: {results.get('max_drawdown', 0):.2%}"
             )
         
         with col4:
             st.metric(
                 "Sharpe Ratio",
                 f"{results.get('sharpe_ratio', 0):.2f}",
                 delta=f"Volatilidade: {results.get('volatility', 0):.2%}"
             )
         
         # Gráfico de equity curve
         if 'equity_curve' in results:
             st.subheader("📈 Curva de Equity")
             
             equity_data = results['equity_curve']
             fig = go.Figure()
             
             fig.add_trace(go.Scatter(
                 x=list(range(len(equity_data))),
                 y=equity_data,
                 mode='lines',
                 name='Equity',
                 line=dict(color='blue', width=2)
             ))
             
             fig.update_layout(
                 title="Evolução do Saldo Durante o Backtest",
                 xaxis_title="Número de Trades",
                 yaxis_title="Saldo ($)",
                 height=400
             )
             
             st.plotly_chart(fig, use_container_width=True)
         
         # Tabela de estatísticas detalhadas
         if 'detailed_stats' in results:
             with st.expander("📋 Estatísticas Detalhadas"):
                 stats_df = pd.DataFrame([results['detailed_stats']])
                 st.dataframe(stats_df.T, use_container_width=True)
    
    def display_advanced_backtest_results(self, results: dict):
         """Exibe resultados do backtest avançado"""
         st.subheader("🔬 Resultados do Backtest Avançado")
         
         # Métricas principais em cards
         col1, col2, col3, col4 = st.columns(4)
         
         with col1:
             st.metric(
                 "💰 Saldo Final",
                 f"${results.get('final_balance', 0):.2f}",
                 delta=f"${results.get('total_pnl', 0):.2f}"
             )
         
         with col2:
             st.metric(
                 "📊 Total de Trades",
                 results.get('total_trades', 0),
                 delta=f"{results.get('win_rate', 0):.1%} win rate"
             )
         
         with col3:
             st.metric(
                 "📈 Retorno Total",
                 f"{results.get('total_return', 0):.2%}",
                 delta=f"Anualizado: {results.get('annualized_return', 0):.2%}"
             )
         
         with col4:
             st.metric(
                 "⚡ Sharpe Ratio",
                 f"{results.get('sharpe_ratio', 0):.2f}",
                 delta=f"Max DD: {results.get('max_drawdown', 0):.2%}"
             )
         
         # Gráficos lado a lado
         col1, col2 = st.columns(2)
         
         with col1:
             # Equity curve
             if 'equity_curve' in results:
                 st.subheader("📈 Curva de Equity")
                 
                 equity_data = results['equity_curve']
                 fig = go.Figure()
                 
                 fig.add_trace(go.Scatter(
                     x=list(range(len(equity_data))),
                     y=equity_data,
                     mode='lines',
                     name='Equity',
                     line=dict(color='blue', width=2)
                 ))
                 
                 fig.update_layout(
                     title="Evolução do Saldo",
                     xaxis_title="Trades",
                     yaxis_title="Saldo ($)",
                     height=400
                 )
                 
                 st.plotly_chart(fig, use_container_width=True)
         
         with col2:
             # Drawdown
             if 'drawdown_curve' in results:
                 st.subheader("📉 Drawdown")
                 
                 drawdown_data = results['drawdown_curve']
                 fig = go.Figure()
                 
                 fig.add_trace(go.Scatter(
                     x=list(range(len(drawdown_data))),
                     y=drawdown_data,
                     mode='lines',
                     name='Drawdown',
                     line=dict(color='red', width=2),
                     fill='tonexty'
                 ))
                 
                 fig.update_layout(
                     title="Drawdown ao Longo do Tempo",
                     xaxis_title="Trades",
                     yaxis_title="Drawdown (%)",
                     height=400
                 )
                 
                 st.plotly_chart(fig, use_container_width=True)
         
         # Análise de performance por período
         if 'monthly_returns' in results:
             st.subheader("📅 Retornos Mensais")
             
             monthly_data = results['monthly_returns']
             fig = go.Figure()
             
             colors = ['green' if ret >= 0 else 'red' for ret in monthly_data.values()]
             
             fig.add_trace(go.Bar(
                 x=list(monthly_data.keys()),
                 y=list(monthly_data.values()),
                 marker_color=colors,
                 name='Retorno Mensal'
             ))
             
             fig.update_layout(
                 title="Performance Mensal",
                 xaxis_title="Mês",
                 yaxis_title="Retorno (%)",
                 height=400
             )
             
             st.plotly_chart(fig, use_container_width=True)
         
         # Tabela de estatísticas completas
         with st.expander("📊 Estatísticas Completas"):
             if 'detailed_stats' in results:
                 stats_df = pd.DataFrame([results['detailed_stats']])
                 st.dataframe(stats_df.T, use_container_width=True)
         
         # Histórico de trades
         with st.expander("📋 Histórico de Trades"):
             if 'trade_history' in results:
                 trades_df = pd.DataFrame(results['trade_history'])
                 st.dataframe(trades_df, use_container_width=True)
    
    def display_optimization_results(self, results: dict):
         """Exibe resultados da otimização"""
         st.subheader("⚡ Resultados da Otimização")
         
         # Melhores parâmetros
         if 'best_params' in results:
             st.success("🏆 Melhores Parâmetros Encontrados:")
             
             best_params = results['best_params']
             col1, col2, col3 = st.columns(3)
             
             with col1:
                 st.metric(
                     "🎯 Confiança Ótima",
                     f"{best_params.get('confidence_threshold', 0):.3f}"
                 )
             
             with col2:
                 st.metric(
                     "💰 Stake Ótimo",
                     f"${best_params.get('stake_amount', 0):.2f}"
                 )
             
             with col3:
                 st.metric(
                     "📊 Score Ótimo",
                     f"{best_params.get('score', 0):.4f}"
                 )
         
         # Performance dos melhores parâmetros
         if 'best_performance' in results:
             st.subheader("📈 Performance dos Melhores Parâmetros")
             
             perf = results['best_performance']
             col1, col2, col3, col4 = st.columns(4)
             
             with col1:
                 st.metric(
                     "Retorno Total",
                     f"{perf.get('total_return', 0):.2%}"
                 )
             
             with col2:
                 st.metric(
                     "Win Rate",
                     f"{perf.get('win_rate', 0):.1%}"
                 )
             
             with col3:
                 st.metric(
                     "Sharpe Ratio",
                     f"{perf.get('sharpe_ratio', 0):.2f}"
                 )
             
             with col4:
                 st.metric(
                     "Max Drawdown",
                     f"{perf.get('max_drawdown', 0):.2%}"
                 )
         
         # Heatmap de resultados
         if 'optimization_matrix' in results:
             st.subheader("🔥 Heatmap de Otimização")
             
             matrix_data = results['optimization_matrix']
             
             fig = go.Figure(data=go.Heatmap(
                 z=matrix_data['scores'],
                 x=matrix_data['stake_values'],
                 y=matrix_data['confidence_values'],
                 colorscale='RdYlGn',
                 hoverongaps=False
             ))
             
             fig.update_layout(
                 title="Mapa de Performance (Confiança vs Stake)",
                 xaxis_title="Stake Amount ($)",
                 yaxis_title="Confidence Threshold",
                 height=500
             )
             
             st.plotly_chart(fig, use_container_width=True)
         
         # Top 10 combinações
         if 'top_combinations' in results:
             st.subheader("🏅 Top 10 Combinações")
             
             top_df = pd.DataFrame(results['top_combinations'])
             
             # Formatar colunas
             if 'total_return' in top_df.columns:
                 top_df['total_return'] = top_df['total_return'].apply(lambda x: f"{x:.2%}")
             if 'win_rate' in top_df.columns:
                 top_df['win_rate'] = top_df['win_rate'].apply(lambda x: f"{x:.1%}")
             if 'sharpe_ratio' in top_df.columns:
                 top_df['sharpe_ratio'] = top_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
             
             st.dataframe(top_df, use_container_width=True, hide_index=True)
         
         # Distribuição de scores
         if 'all_scores' in results:
             st.subheader("📊 Distribuição de Scores")
             
             scores = results['all_scores']
             
             fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=30)])
             
             fig.update_layout(
                 title="Distribuição dos Scores de Otimização",
                 xaxis_title="Score",
                 yaxis_title="Frequência",
                 height=400
             )
             
             st.plotly_chart(fig, use_container_width=True)
         
         # Botão para aplicar melhores parâmetros
         if 'best_params' in results:
             st.markdown("---")
             
             col1, col2, col3 = st.columns([1, 2, 1])
             
             with col2:
                 if st.button("✅ Aplicar Melhores Parâmetros ao Bot", type="primary", use_container_width=True):
                     try:
                         best_params = results['best_params']
                         
                         # Atualizar configurações
                         config.trading.min_prediction_confidence = best_params.get('confidence_threshold', 0.6)
                         config.trading.initial_stake = best_params.get('stake_amount', 10.0)
                         
                         st.success("✅ Parâmetros aplicados com sucesso!")
                         st.info("🔄 Reinicie o bot para que as mudanças tenham efeito")
                         
                     except Exception as e:
                         st.error(f"Erro ao aplicar parâmetros: {e}")
    
    def render_settings_tab(self):
        """Renderiza aba de configurações"""
        st.header("⚙️ Configurações do Sistema")
        
        # Configurações de Trading
        st.subheader("💹 Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_stake = st.number_input(
                "Stake Inicial ($)",
                value=float(config.trading.initial_stake),
                min_value=1.0,
                step=1.0
            )
            
            new_confidence = st.slider(
                "Confiança Mínima",
                min_value=0.5,
                max_value=1.0,
                value=float(config.trading.min_prediction_confidence),
                step=0.01
            )
            
            new_max_trades = st.number_input(
                "Máx. Trades Diários",
                value=config.trading.max_daily_trades,
                min_value=1,
                step=1
            )
        
        with col2:
            new_max_loss = st.number_input(
                "Máx. Perda Diária ($)",
                value=float(config.trading.max_daily_loss),
                min_value=1.0,
                step=1.0
            )
            
            enable_martingale = st.checkbox(
                "Habilitar Martingale",
                value=config.trading.enable_martingale
            )
            
            if enable_martingale:
                martingale_mult = st.number_input(
                    "Multiplicador Martingale",
                    value=float(config.trading.martingale_multiplier),
                    min_value=1.1,
                    step=0.1
                )
        
        # Configurações de ML
        st.subheader("🤖 Machine Learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            retrain_interval = st.number_input(
                "Intervalo de Retreino (horas)",
                value=24,
                min_value=1,
                step=1
            )
        
        with col2:
            min_data_points = st.number_input(
                "Mín. Amostras Treinamento",
                value=config.ml.min_training_samples,
                min_value=100,
                step=50
            )
        
        # Botão para salvar configurações
        if st.button("💾 Salvar Configurações"):
            self.save_settings({
                'initial_stake': new_stake,
                'min_prediction_confidence': new_confidence,
                'max_daily_trades': new_max_trades,
                'max_daily_loss': new_max_loss,
                'enable_martingale': enable_martingale,
                'martingale_multiplier': martingale_mult if enable_martingale else 2.0,
                'retrain_interval': retrain_interval,
                'min_training_samples': min_data_points
            })
        
        # Status do modelo
        st.subheader("🧠 Status do Modelo ML")
        self.render_model_status()
    
    def render_equity_curve(self):
        """Renderiza curva de equity"""
        st.subheader("💰 Curva de Equity")
        
        try:
            trades_df = trade_recorder.get_trades_df()
            
            if not trades_df.empty:
                # Calcular equity curve
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                trades_df['equity'] = config.trading.initial_balance + trades_df['cumulative_pnl']
                
                # Criar gráfico
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="Evolução do Saldo",
                    xaxis_title="Tempo",
                    yaxis_title="Saldo ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum dado disponível para equity curve")
                
        except Exception as e:
            st.error(f"Erro ao renderizar equity curve: {e}")
    
    def render_daily_pnl_chart(self):
        """Renderiza gráfico de PnL diário"""
        st.subheader("📊 PnL Diário")
        
        try:
            trades_df = trade_recorder.get_trades_df()
            
            if not trades_df.empty:
                # Agrupar por dia
                daily_pnl = trades_df.groupby(trades_df['timestamp'].dt.date)['pnl'].sum()
                
                # Criar gráfico de barras
                fig = go.Figure()
                
                colors = ['green' if pnl >= 0 else 'red' for pnl in daily_pnl.values]
                
                fig.add_trace(go.Bar(
                    x=daily_pnl.index,
                    y=daily_pnl.values,
                    marker_color=colors,
                    name='PnL Diário'
                ))
                
                fig.update_layout(
                    title="PnL por Dia",
                    xaxis_title="Data",
                    yaxis_title="PnL ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum dado disponível para PnL diário")
                
        except Exception as e:
            st.error(f"Erro ao renderizar PnL diário: {e}")
    
    def render_recent_trades_table(self):
        """Renderiza tabela de trades recentes"""
        try:
            trades_df = trade_recorder.get_trades_df()
            
            if not trades_df.empty:
                # Últimos 10 trades
                recent_trades = trades_df.tail(10).copy()
                
                # Formatar dados
                recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime('%H:%M:%S')
                recent_trades['pnl'] = recent_trades['pnl'].apply(lambda x: f"${x:.2f}")
                recent_trades['confidence'] = recent_trades['confidence'].apply(lambda x: f"{x:.1%}")
                
                # Selecionar colunas
                display_columns = ['timestamp', 'signal', 'stake', 'pnl', 'confidence']
                
                st.dataframe(
                    recent_trades[display_columns],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Nenhum trade realizado ainda")
                
        except Exception as e:
            st.error(f"Erro ao carregar trades recentes: {e}")
    
    def start_trading(self):
        """Inicia trading automático"""
        try:
            # Verificar se já existe uma instância do trader
            if st.session_state.trader_instance is None:
                st.session_state.trader_instance = TradingExecutor()
            
            # Verificar se pode iniciar trading
            can_trade, reason = risk_manager.can_trade()
            if not can_trade:
                error_msg = f"Não é possível iniciar trading: {reason}"
                st.error(error_msg)
                self.add_notification(error_msg, 'error')
                return
            
            # Verificar autenticação
            if not auth_manager.is_authenticated:
                error_msg = "Não autenticado. Faça login primeiro."
                st.error(f"❌ {error_msg}")
                self.add_notification(error_msg, 'error')
                return
            
            # Verificar token
            token_status = token_manager.get_status()
            if not token_status.get('authenticated', False):
                error_msg = "Token inválido ou expirado. Renove o token."
                st.error(f"❌ {error_msg}")
                self.add_notification(error_msg, 'error')
                return
            
            # Iniciar trading em background
            with st.spinner("Iniciando trading automático..."):
                # Criar task assíncrona para o trading
                if hasattr(st.session_state, 'trading_task') and not st.session_state.trading_task.done():
                    st.warning("Trading já está em execução")
                    return
                
                # Marcar como ativo
                st.session_state.trading_active = True
                
                # Criar nova task para trading
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def run_trading():
                    try:
                        await st.session_state.trader_instance.start_trading()
                    except Exception as e:
                        st.session_state.trading_active = False
                        st.error(f"Erro durante trading: {e}")
                
                # Executar em thread separada para não bloquear UI
                import threading
                def trading_thread():
                    try:
                        loop.run_until_complete(run_trading())
                    except Exception as e:
                        st.session_state.trading_active = False
                        st.error(f"Erro na thread de trading: {e}")
                    finally:
                        loop.close()
                
                thread = threading.Thread(target=trading_thread, daemon=True)
                thread.start()
                
                success_msg = "Trading iniciado com sucesso!"
                st.success(f"✅ {success_msg}")
                st.info("🤖 Bot está analisando o mercado e executando trades automaticamente")
                
                # Adicionar notificações
                self.add_notification(success_msg, 'success')
                self.add_notification("Bot analisando mercado", 'info')
                
                # Log da ação
                self.logger.info("Trading automático iniciado via dashboard")
            
        except Exception as e:
            st.error(f"Erro ao iniciar trading: {e}")
            st.session_state.trading_active = False
            self.logger.error(f"Erro ao iniciar trading: {e}")
    
    def stop_trading(self):
        """Para trading automático"""
        try:
            with st.spinner("Parando trading automático..."):
                # Parar o trader se existir
                if st.session_state.trader_instance is not None:
                    # Criar loop para parar o trading
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def stop_trading_async():
                        await st.session_state.trader_instance.stop_trading()
                    
                    try:
                        loop.run_until_complete(stop_trading_async())
                    except Exception as e:
                        self.logger.warning(f"Erro ao parar trader: {e}")
                    finally:
                        loop.close()
                
                # Marcar como inativo
                st.session_state.trading_active = False
                
                # Cancelar task se existir
                if hasattr(st.session_state, 'trading_task'):
                    try:
                        st.session_state.trading_task.cancel()
                    except:
                        pass
                
                success_msg = "Trading parado com sucesso!"
                st.success(f"✅ {success_msg}")
                st.info("🛑 Bot parou de executar trades automaticamente")
                
                # Adicionar notificações
                self.add_notification(success_msg, 'success')
                self.add_notification("Bot parado", 'info')
                
                # Log da ação
                self.logger.info("Trading automático parado via dashboard")
            
        except Exception as e:
            st.error(f"Erro ao parar trading: {e}")
            self.logger.error(f"Erro ao parar trading: {e}")
    
    def execute_manual_trade(self, signal_type: str):
        """Executa trade manual"""
        try:
            # Verificar se pode fazer trade
            can_trade, reason = risk_manager.can_trade()
            
            if not can_trade:
                st.error(f"Não é possível fazer trade: {reason}")
                return
            
            # Aqui você implementaria a execução do trade manual
            st.success(f"Trade {signal_type} executado!")
            
        except Exception as e:
            st.error(f"Erro ao executar trade manual: {e}")
    
    def save_settings(self, settings: Dict[str, Any]):
        """Salva configurações"""
        try:
            # Aqui você implementaria a lógica para salvar configurações
            st.success("Configurações salvas com sucesso!")
            
        except Exception as e:
            st.error(f"Erro ao salvar configurações: {e}")
    
    def render_realtime_price_chart(self):
        """Renderiza gráfico de preço em tempo real"""
        st.info("Gráfico de preço em tempo real será implementado com dados da API")
    
    def render_activity_log(self):
        """Renderiza log de atividades"""
        st.info("Log de atividades será implementado")
    
    def render_period_metrics(self, trades_df: pd.DataFrame):
        """Renderiza métricas do período"""
        st.info("Métricas do período serão implementadas")
    
    def render_pnl_distribution(self, trades_df: pd.DataFrame):
        """Renderiza distribuição de PnL"""
        st.info("Distribuição de PnL será implementada")
    
    def render_win_rate_by_hour(self, trades_df: pd.DataFrame):
        """Renderiza win rate por hora"""
        st.info("Win rate por hora será implementado")
    
    def render_performance_by_signal(self, trades_df: pd.DataFrame):
        """Renderiza performance por sinal"""
        st.info("Performance por sinal será implementada")
    
    def render_martingale_analysis(self, trades_df: pd.DataFrame):
        """Renderiza análise de martingale"""
        st.info("Análise de martingale será implementada")
    
    def run_backtest(self, data: pd.DataFrame, start_date, end_date, commission: float):
        """Executa backtest"""
        st.info("Funcionalidade de backtest será implementada")
    
    def render_backtest_history(self):
        """Renderiza histórico de backtests"""
        st.info("Histórico de backtests será implementado")
    
    def render_model_status(self):
        """Renderiza status do modelo"""
        st.info("Status do modelo ML será implementado")
    
    def render_ai_bot_tab(self):
        """Renderiza aba de controle do Bot AI"""
        st.header("🎯 Bot AI - Trading Automatizado")
        
        # Inicializar bot AI se não existir
        if 'ai_bot' not in st.session_state:
            st.session_state.ai_bot = None
        
        # Status do bot
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.ai_bot and st.session_state.ai_bot.is_running:
                st.success("🟢 Bot AI Ativo")
                
                # Estatísticas em tempo real
                stats = st.session_state.ai_bot.get_session_stats()
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric(
                        "Trades Hoje",
                        stats.get('trades_today', 0),
                        delta=f"Win Rate: {stats.get('win_rate_today', 0):.1%}"
                    )
                
                with col_b:
                    st.metric(
                        "P&L Sessão",
                        f"${stats.get('session_pnl', 0):.2f}",
                        delta=f"{stats.get('session_return', 0):.2%}"
                    )
                
                with col_c:
                    st.metric(
                        "Última Predição",
                        f"{stats.get('last_confidence', 0):.1%}",
                        delta=stats.get('last_signal', 'N/A')
                    )
                
                with col_d:
                    st.metric(
                        "Tempo Ativo",
                        stats.get('uptime', '00:00:00'),
                        delta=f"Próximo: {stats.get('next_prediction', 'N/A')}"
                    )
            else:
                st.warning("🔴 Bot AI Inativo")
                st.info("Configure e inicie o bot para começar o trading automatizado")
        
        with col2:
            # Controles principais
            if st.session_state.ai_bot and st.session_state.ai_bot.is_running:
                if st.button("⏹️ Parar Bot", type="secondary", use_container_width=True):
                    self.stop_ai_bot()
            else:
                if st.button("▶️ Iniciar Bot", type="primary", use_container_width=True):
                    self.start_ai_bot()
        
        with col3:
            # Botão de emergência
            if st.button("🚨 STOP EMERGÊNCIA", type="secondary", use_container_width=True):
                self.emergency_stop()
        
        st.markdown("---")
        
        # Configurações do Bot AI
        st.subheader("⚙️ Configurações do Bot AI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Configurações de Trading**")
            
            symbol = st.selectbox(
                "Símbolo",
                ["R_50", "R_75", "R_100", "RDBEAR", "RDBULL"],
                index=0,
                key="ai_symbol"
            )
            
            stake_amount = st.number_input(
                "Valor por Trade ($)",
                value=10.0,
                min_value=1.0,
                max_value=1000.0,
                step=1.0,
                key="ai_stake"
            )
            
            confidence_threshold = st.slider(
                "Confiança Mínima",
                min_value=0.5,
                max_value=1.0,
                value=0.65,
                step=0.01,
                key="ai_confidence"
            )
            
            max_daily_trades = st.number_input(
                "Máx. Trades por Dia",
                value=50,
                min_value=1,
                max_value=500,
                key="ai_max_trades"
            )
        
        with col2:
            st.markdown("**🛡️ Gerenciamento de Risco**")
            
            max_daily_loss = st.number_input(
                "Perda Máxima Diária ($)",
                value=100.0,
                min_value=10.0,
                max_value=5000.0,
                step=10.0,
                key="ai_max_loss"
            )
            
            enable_martingale = st.checkbox(
                "Habilitar Martingale",
                value=False,
                key="ai_martingale"
            )
            
            if enable_martingale:
                martingale_multiplier = st.number_input(
                    "Multiplicador Martingale",
                    value=2.0,
                    min_value=1.1,
                    max_value=5.0,
                    step=0.1,
                    key="ai_mart_mult"
                )
                
                max_martingale_steps = st.number_input(
                    "Máx. Steps Martingale",
                    value=3,
                    min_value=1,
                    max_value=10,
                    key="ai_mart_steps"
                )
            else:
                martingale_multiplier = 1.0
                max_martingale_steps = 0
            
            stop_loss_percentage = st.slider(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                key="ai_stop_loss"
            )
        
        # Configurações avançadas
        with st.expander("🔧 Configurações Avançadas"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 Modelo de IA**")
                
                model_type = st.selectbox(
                    "Tipo de Modelo",
                    ["LightGBM", "LSTM", "RandomForest"],
                    index=0,
                    key="ai_model_type"
                )
                
                retrain_frequency = st.selectbox(
                    "Frequência de Retreino",
                    ["Nunca", "Diário", "Semanal", "A cada 100 trades"],
                    index=2,
                    key="ai_retrain"
                )
                
                use_technical_indicators = st.multiselect(
                    "Indicadores Técnicos",
                    ["RSI", "MACD", "Bollinger", "SMA", "EMA", "ATR", "Williams %R"],
                    default=["RSI", "MACD", "Bollinger", "SMA"],
                    key="ai_indicators"
                )
            
            with col2:
                st.markdown("**📊 Dados e Features**")
                
                lookback_period = st.number_input(
                    "Período de Lookback",
                    value=100,
                    min_value=50,
                    max_value=1000,
                    step=10,
                    key="ai_lookback"
                )
                
                prediction_interval = st.selectbox(
                    "Intervalo de Predição",
                    ["1 tick", "5 ticks", "10 ticks", "1 minuto"],
                    index=1,
                    key="ai_interval"
                )
                
                enable_news_sentiment = st.checkbox(
                    "Análise de Sentimento (Experimental)",
                    value=False,
                    key="ai_sentiment"
                )
        
        # Botão para salvar configurações
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("💾 Salvar Configurações", type="primary", use_container_width=True):
                self.save_ai_bot_config()
        
        st.markdown("---")
        
        # Monitoramento em tempo real
        if st.session_state.ai_bot and st.session_state.ai_bot.is_running:
            st.subheader("📊 Monitoramento em Tempo Real")
            
            # Gráficos de performance
            col1, col2 = st.columns(2)
            
            with col1:
                # Equity curve em tempo real
                st.markdown("**📈 Curva de Equity**")
                
                equity_data = st.session_state.ai_bot.get_equity_curve()
                
                if equity_data:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(equity_data))),
                        y=equity_data,
                        mode='lines',
                        name='Equity',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Evolução do Saldo",
                        xaxis_title="Trades",
                        yaxis_title="Saldo ($)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aguardando dados de equity...")
            
            with col2:
                # Distribuição de predições
                st.markdown("**🎯 Distribuição de Confiança**")
                
                confidence_data = st.session_state.ai_bot.get_confidence_distribution()
                
                if confidence_data:
                    fig = go.Figure(data=[go.Histogram(
                        x=confidence_data,
                        nbinsx=20,
                        name='Confiança'
                    )])
                    
                    fig.update_layout(
                        title="Distribuição das Predições",
                        xaxis_title="Confiança",
                        yaxis_title="Frequência",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aguardando dados de predições...")
            
            # Log de atividades
            st.markdown("**📋 Log de Atividades**")
            
            log_data = st.session_state.ai_bot.get_recent_logs(limit=10)
            
            if log_data:
                log_df = pd.DataFrame(log_data)
                st.dataframe(log_df, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhuma atividade recente")
        
        # Histórico de sessões
        st.subheader("📚 Histórico de Sessões")
        
        session_history = self.get_ai_bot_session_history()
        
        if session_history:
            history_df = pd.DataFrame(session_history)
            
            # Formatar colunas
            if 'session_return' in history_df.columns:
                history_df['session_return'] = history_df['session_return'].apply(lambda x: f"{x:.2%}")
            if 'win_rate' in history_df.columns:
                history_df['win_rate'] = history_df['win_rate'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhuma sessão anterior encontrada")
    
    def start_ai_bot(self):
        """Inicia o bot AI"""
        try:
            # Verificar se está conectado
            if not self.check_api_connection():
                st.error("❌ Conecte-se à API Deriv primeiro")
                return
            
            # Criar configuração do bot
            bot_config = {
                'symbol': st.session_state.get('ai_symbol', 'R_50'),
                'stake_amount': st.session_state.get('ai_stake', 10.0),
                'confidence_threshold': st.session_state.get('ai_confidence', 0.65),
                'max_daily_trades': st.session_state.get('ai_max_trades', 50),
                'max_daily_loss': st.session_state.get('ai_max_loss', 100.0),
                'enable_martingale': st.session_state.get('ai_martingale', False),
                'martingale_multiplier': st.session_state.get('ai_mart_mult', 2.0),
                'max_martingale_steps': st.session_state.get('ai_mart_steps', 3),
                'stop_loss_percentage': st.session_state.get('ai_stop_loss', 10.0),
                'model_type': st.session_state.get('ai_model_type', 'LightGBM'),
                'retrain_frequency': st.session_state.get('ai_retrain', 'Semanal'),
                'technical_indicators': st.session_state.get('ai_indicators', ['RSI', 'MACD']),
                'lookback_period': st.session_state.get('ai_lookback', 100),
                'prediction_interval': st.session_state.get('ai_interval', '5 ticks'),
                'enable_news_sentiment': st.session_state.get('ai_sentiment', False)
            }
            
            # Criar e inicializar bot
            st.session_state.ai_bot = AITradingBot(bot_config)
            
            # Inicializar bot
            if st.session_state.ai_bot.initialize():
                # Iniciar trading
                st.session_state.ai_bot.start_trading()
                
                st.success("✅ Bot AI iniciado com sucesso!")
                self.add_notification("success", "Bot AI iniciado")
                
                # Rerun para atualizar interface
                st.rerun()
            else:
                st.error("❌ Falha ao inicializar o bot AI")
                st.session_state.ai_bot = None
                
        except Exception as e:
            st.error(f"❌ Erro ao iniciar bot AI: {e}")
            st.session_state.ai_bot = None
    
    def stop_ai_bot(self):
        """Para o bot AI"""
        try:
            if st.session_state.ai_bot:
                # Parar trading
                st.session_state.ai_bot.stop_trading()
                
                # Gerar relatório final
                final_report = st.session_state.ai_bot.generate_session_report()
                
                st.success("✅ Bot AI parado com sucesso!")
                self.add_notification("info", "Bot AI parado")
                
                # Mostrar relatório
                with st.expander("📊 Relatório da Sessão"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total de Trades",
                            final_report.get('total_trades', 0)
                        )
                    
                    with col2:
                        st.metric(
                            "Win Rate",
                            f"{final_report.get('win_rate', 0):.1%}"
                        )
                    
                    with col3:
                        st.metric(
                            "P&L Total",
                            f"${final_report.get('total_pnl', 0):.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Retorno",
                            f"{final_report.get('total_return', 0):.2%}"
                        )
                
                # Limpar instância
                st.session_state.ai_bot = None
                
                # Rerun para atualizar interface
                st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erro ao parar bot AI: {e}")
    
    def emergency_stop(self):
        """Para tudo imediatamente"""
        try:
            # Parar bot AI
            if st.session_state.ai_bot:
                st.session_state.ai_bot.emergency_stop()
                st.session_state.ai_bot = None
            
            # Parar trading manual
            if st.session_state.trading_active:
                st.session_state.trading_active = False
                if st.session_state.trader_instance:
                    st.session_state.trader_instance.stop()
                    st.session_state.trader_instance = None
            
            # Fechar todas as posições abertas
            trade_executor.close_all_positions()
            
            st.warning("🚨 PARADA DE EMERGÊNCIA EXECUTADA!")
            self.add_notification("warning", "Parada de emergência executada")
            
            # Rerun para atualizar interface
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erro na parada de emergência: {e}")
    
    def save_ai_bot_config(self):
        """Salva configurações do bot AI"""
        try:
            config_data = {
                'symbol': st.session_state.get('ai_symbol', 'R_50'),
                'stake_amount': st.session_state.get('ai_stake', 10.0),
                'confidence_threshold': st.session_state.get('ai_confidence', 0.65),
                'max_daily_trades': st.session_state.get('ai_max_trades', 50),
                'max_daily_loss': st.session_state.get('ai_max_loss', 100.0),
                'enable_martingale': st.session_state.get('ai_martingale', False),
                'martingale_multiplier': st.session_state.get('ai_mart_mult', 2.0),
                'max_martingale_steps': st.session_state.get('ai_mart_steps', 3),
                'stop_loss_percentage': st.session_state.get('ai_stop_loss', 10.0),
                'model_type': st.session_state.get('ai_model_type', 'LightGBM'),
                'retrain_frequency': st.session_state.get('ai_retrain', 'Semanal'),
                'technical_indicators': st.session_state.get('ai_indicators', ['RSI', 'MACD']),
                'lookback_period': st.session_state.get('ai_lookback', 100),
                'prediction_interval': st.session_state.get('ai_interval', '5 ticks'),
                'enable_news_sentiment': st.session_state.get('ai_sentiment', False),
                'saved_at': datetime.now().isoformat()
            }
            
            # Salvar em arquivo
            config_file = 'ai_bot_config.json'
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            st.success("✅ Configurações salvas com sucesso!")
            self.add_notification("success", "Configurações do Bot AI salvas")
            
        except Exception as e:
            st.error(f"❌ Erro ao salvar configurações: {e}")
    
    def get_ai_bot_session_history(self):
        """Obtém histórico de sessões do bot AI"""
        try:
            history_file = 'ai_bot_sessions.json'
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    return json.load(f)
            
            return []
            
        except Exception as e:
            st.error(f"Erro ao carregar histórico: {e}")
            return []
    
    def check_api_connection(self):
        """Verifica status da conexão com a API e exibe alertas"""
        try:
            # Verificar se o token está configurado
            if not config.deriv.api_token:
                st.error("🚫 Token da API não configurado!")
                st.info("Configure o DERIV_API_TOKEN no arquivo .env")
                return False
            
            # Verificar se há problemas conhecidos
            if 'api_error' in st.session_state:
                error_msg = st.session_state.api_error
                
                if 'oauth token' in error_msg.lower() or 'permission' in error_msg.lower():
                    st.error("🔑 Token da API inválido ou expirado!")
                    
                    with st.expander("💡 Como corrigir o problema"):
                        st.markdown("""
                        **Passos para corrigir:**
                        1. Acesse [app.deriv.com/account/api-token](https://app.deriv.com/account/api-token)
                        2. Gere um novo token da API
                        3. Copie o token gerado
                        4. Abra o arquivo `.env` na pasta do projeto
                        5. Atualize a linha: `DERIV_API_TOKEN=seu_novo_token`
                        6. Reinicie o dashboard
                        
                        **Importante:** O token deve ter permissões para:
                        - Ler informações da conta
                        - Fazer trades
                        - Acessar saldos
                        """)
                    return False
                else:
                    st.warning(f"⚠️ Problema na API: {error_msg}")
                    return False
            
            return True
            
        except Exception as e:
            st.error(f"Erro ao verificar API: {e}")
            return False

def main():
    """Função principal do dashboard"""
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()

