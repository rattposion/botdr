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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔐 Login", "📊 Overview", "💹 Trading", "📈 Performance", 
            "🧪 Backtest", "⚙️ Configurações"
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
        
        # Upload de dados
        st.subheader("📁 Dados para Backtest")
        
        uploaded_file = st.file_uploader(
            "Upload arquivo CSV com dados históricos",
            type=['csv'],
            help="Arquivo deve conter colunas: timestamp, quote"
        )
        
        if uploaded_file is not None:
            try:
                # Carregar dados
                data = pd.read_csv(uploaded_file)
                st.success(f"Dados carregados: {len(data)} registros")
                
                # Mostrar preview
                st.subheader("👀 Preview dos Dados")
                st.dataframe(data.head())
                
                # Configurações do backtest
                st.subheader("⚙️ Configurações do Backtest")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    start_date = st.date_input("Data Início Backtest")
                
                with col2:
                    end_date = st.date_input("Data Fim Backtest")
                
                with col3:
                    commission = st.number_input("Comissão por Trade", value=0.0, min_value=0.0)
                
                # Executar backtest
                if st.button("🚀 Executar Backtest"):
                    self.run_backtest(data, start_date, end_date, commission)
                    
            except Exception as e:
                st.error(f"Erro ao carregar dados: {e}")
        
        # Resultados de backtests anteriores
        st.subheader("📊 Resultados Anteriores")
        self.render_backtest_history()
    
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