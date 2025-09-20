"""
Sistema de Autenticação OAuth2 para Deriv
Gerencia login, autorização e renovação de tokens automaticamente
"""
import os
import json
import time
import hashlib
import secrets
import base64
import urllib.parse
import webbrowser
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import requests
import streamlit as st
from config import config

class DerivAuthManager:
    """Gerenciador de autenticação OAuth2 para Deriv"""
    
    def __init__(self):
        self.client_id = "64394"  # App ID público do Deriv para OAuth
        self.redirect_uri = "http://localhost:8502/callback"
        self.auth_url = "https://oauth.deriv.com/oauth2/authorize"
        self.token_url = "https://oauth.deriv.com/oauth2/token"
        self.api_url = "https://api.deriv.com"
        
        # Estado da autenticação
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.user_info = None
        self.is_authenticated = False
        
        # Servidor de callback
        self.callback_server = None
        self.callback_received = False
        self.auth_code = None
        self.auth_error = None
        
        # Carregar tokens salvos
        self._load_saved_tokens()
    
    def _load_saved_tokens(self):
        """Carrega tokens salvos do arquivo"""
        try:
            # Usar caminho absoluto mais robusto
            token_file = os.path.abspath('.deriv_tokens.json')
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    data = json.load(f)
                
                self.access_token = data.get('access_token')
                self.refresh_token = data.get('refresh_token')
                expires_str = data.get('expires_at')
                
                if expires_str:
                    self.token_expires_at = datetime.fromisoformat(expires_str)
                
                self.user_info = data.get('user_info')
                
                # Verificar se o token ainda é válido
                if self.access_token and self.token_expires_at:
                    if datetime.now() < self.token_expires_at:
                        self.is_authenticated = True
                        print("✅ Tokens carregados com sucesso")
                    else:
                        print("⚠️ Token expirado, será necessário renovar")
                        
        except Exception as e:
            print(f"❌ Erro ao carregar tokens: {e}")
    
    def _save_tokens(self):
        """Salva tokens no arquivo"""
        try:
            # Usar caminho absoluto mais robusto
            token_file = os.path.abspath('.deriv_tokens.json')
            data = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'expires_at': self.token_expires_at.isoformat() if self.token_expires_at else None,
                'user_info': self.user_info,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(token_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print("✅ Tokens salvos com sucesso")
            
        except Exception as e:
            print(f"❌ Erro ao salvar tokens: {e}")
    
    def generate_auth_url(self) -> Tuple[str, str]:
        """Gera URL de autorização OAuth2"""
        # Gerar state para segurança
        state = secrets.token_urlsafe(32)
        
        # Gerar code_verifier e code_challenge para PKCE
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')
        
        # Parâmetros da URL de autorização
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'read,trade,payments,admin',
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }
        
        auth_url = f"{self.auth_url}?{urllib.parse.urlencode(params)}"
        
        # Salvar code_verifier para usar na troca do token
        self.code_verifier = code_verifier
        self.auth_state = state
        
        return auth_url, state
    
    def start_callback_server(self):
        """Inicia servidor para receber callback OAuth"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import socketserver
            
            class CallbackHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    # Parse da URL
                    parsed_url = urllib.parse.urlparse(self.path)
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    
                    # Verificar se recebeu código de autorização
                    if 'code' in query_params:
                        auth_manager.auth_code = query_params['code'][0]
                        auth_manager.callback_received = True
                        
                        # Resposta de sucesso
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        success_html = """
                        <html>
                        <head><title>Autorização Concluída</title></head>
                        <body style="font-family: Arial; text-align: center; padding: 50px;">
                            <h1 style="color: green;">✅ Autorização Concluída!</h1>
                            <p>Você pode fechar esta janela e voltar ao bot.</p>
                            <script>setTimeout(() => window.close(), 3000);</script>
                        </body>
                        </html>
                        """
                        self.wfile.write(success_html.encode())
                        
                    elif 'error' in query_params:
                        auth_manager.auth_error = query_params['error'][0]
                        auth_manager.callback_received = True
                        
                        # Resposta de erro
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        error_html = f"""
                        <html>
                        <head><title>Erro na Autorização</title></head>
                        <body style="font-family: Arial; text-align: center; padding: 50px;">
                            <h1 style="color: red;">❌ Erro na Autorização</h1>
                            <p>Erro: {auth_manager.auth_error}</p>
                            <p>Tente novamente no bot.</p>
                        </body>
                        </html>
                        """
                        self.wfile.write(error_html.encode())
                
                def log_message(self, format, *args):
                    # Suprimir logs do servidor
                    pass
            
            # Referência para o auth_manager
            auth_manager = self
            
            # Iniciar servidor na porta 8502
            server = HTTPServer(('localhost', 8502), CallbackHandler)
            self.callback_server = server
            
            # Executar em thread separada
            def run_server():
                print("🌐 Servidor de callback iniciado em http://localhost:8502")
                server.serve_forever()
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao iniciar servidor de callback: {e}")
            return False
    
    def stop_callback_server(self):
        """Para o servidor de callback"""
        if self.callback_server:
            self.callback_server.shutdown()
            self.callback_server = None
            print("🛑 Servidor de callback parado")
    
    def exchange_code_for_token(self, auth_code: str) -> bool:
        """Troca código de autorização por token de acesso"""
        try:
            data = {
                'grant_type': 'authorization_code',
                'client_id': self.client_id,
                'code': auth_code,
                'redirect_uri': self.redirect_uri,
                'code_verifier': self.code_verifier
            }
            
            response = requests.post(self.token_url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                
                self.access_token = token_data['access_token']
                self.refresh_token = token_data.get('refresh_token')
                
                # Calcular expiração
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                self.is_authenticated = True
                
                # Obter informações do usuário
                self._get_user_info()
                
                # Salvar tokens
                self._save_tokens()
                
                print("✅ Token obtido com sucesso!")
                return True
            else:
                print(f"❌ Erro ao obter token: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erro na troca do código: {e}")
            return False
    
    def _get_user_info(self):
        """Obtém informações do usuário autenticado"""
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            response = requests.get(f"{self.api_url}/v3/user", headers=headers)
            
            if response.status_code == 200:
                self.user_info = response.json()
                print(f"✅ Usuário autenticado: {self.user_info.get('email', 'N/A')}")
            else:
                print(f"⚠️ Não foi possível obter informações do usuário: {response.text}")
                
        except Exception as e:
            print(f"❌ Erro ao obter informações do usuário: {e}")
    
    def refresh_access_token(self) -> bool:
        """Renova o token de acesso usando refresh token"""
        if not self.refresh_token:
            print("❌ Refresh token não disponível")
            return False
        
        try:
            data = {
                'grant_type': 'refresh_token',
                'client_id': self.client_id,
                'refresh_token': self.refresh_token
            }
            
            response = requests.post(self.token_url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                
                self.access_token = token_data['access_token']
                if 'refresh_token' in token_data:
                    self.refresh_token = token_data['refresh_token']
                
                # Calcular nova expiração
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                self.is_authenticated = True
                
                # Salvar tokens atualizados
                self._save_tokens()
                
                print("✅ Token renovado com sucesso!")
                return True
            else:
                print(f"❌ Erro ao renovar token: {response.text}")
                self.logout()
                return False
                
        except Exception as e:
            print(f"❌ Erro na renovação do token: {e}")
            self.logout()
            return False
    
    def check_token_validity(self) -> bool:
        """Verifica se o token ainda é válido"""
        if not self.access_token or not self.token_expires_at:
            return False
        
        # Verificar se expira em menos de 5 minutos
        if datetime.now() >= (self.token_expires_at - timedelta(minutes=5)):
            print("⚠️ Token expirando, tentando renovar...")
            return self.refresh_access_token()
        
        return True
    
    def get_api_token(self) -> Optional[str]:
        """Retorna token válido para uso na API"""
        if self.check_token_validity():
            return self.access_token
        return None
    
    def refresh_token(self) -> bool:
        """Método público para renovar token (usado pelo token_manager)"""
        return self.refresh_access_token()
    
    def get_token_info(self) -> Optional[Dict]:
        """Retorna informações detalhadas do token"""
        if not self.is_authenticated:
            return None
        
        return {
            'access_token': self.access_token,
            'expires_at': self.token_expires_at.isoformat() if self.token_expires_at else None,
            'user_email': self.user_info.get('email') if self.user_info else None,
            'has_refresh_token': bool(self.refresh_token)
        }
    
    def login(self) -> bool:
        """Inicia processo de login OAuth2"""
        try:
            # Verificar se já está autenticado
            if self.is_authenticated and self.check_token_validity():
                print("✅ Já autenticado!")
                return True
            
            # Iniciar servidor de callback
            if not self.start_callback_server():
                return False
            
            # Gerar URL de autorização
            auth_url, state = self.generate_auth_url()
            
            print("🔐 Iniciando processo de login...")
            print(f"📱 Abrindo navegador: {auth_url}")
            
            # Abrir navegador
            webbrowser.open(auth_url)
            
            # Aguardar callback
            print("⏳ Aguardando autorização...")
            timeout = 300  # 5 minutos
            start_time = time.time()
            
            while not self.callback_received and (time.time() - start_time) < timeout:
                time.sleep(1)
            
            # Parar servidor
            self.stop_callback_server()
            
            if self.auth_error:
                print(f"❌ Erro na autorização: {self.auth_error}")
                return False
            
            if not self.auth_code:
                print("❌ Timeout ou autorização cancelada")
                return False
            
            # Trocar código por token
            return self.exchange_code_for_token(self.auth_code)
            
        except Exception as e:
            print(f"❌ Erro no login: {e}")
            return False
    
    def logout(self):
        """Faz logout e limpa tokens"""
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.user_info = None
        self.is_authenticated = False
        
        # Remover arquivo de tokens
        try:
            token_file = os.path.join(os.path.dirname(__file__), '.deriv_tokens.json')
            if os.path.exists(token_file):
                os.remove(token_file)
        except:
            pass
        
        print("👋 Logout realizado com sucesso")
    
    def get_auth_status(self) -> Dict:
        """Retorna status da autenticação"""
        return {
            'is_authenticated': self.is_authenticated,
            'user_email': self.user_info.get('email') if self.user_info else None,
            'token_expires_at': self.token_expires_at.isoformat() if self.token_expires_at else None,
            'expires_in_minutes': int((self.token_expires_at - datetime.now()).total_seconds() / 60) if self.token_expires_at else 0
        }

# Instância global
auth_manager = DerivAuthManager()