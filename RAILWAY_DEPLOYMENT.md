# 🚀 Deployment no Railway - Bot Trader Deriv

## 📋 Visão Geral

Este guia explica como fazer o deployment do Bot Trader Deriv no Railway, incluindo a configuração específica para OAuth2 que funciona na plataforma.

## 🔧 Configuração no Railway

### 1. Variáveis de Ambiente

Configure as seguintes variáveis no **Railway Dashboard**:

```env
# OAuth2 Deriv (OBRIGATÓRIO)
DERIV_APP_ID=64394
DERIV_CLIENT_SECRET=seu_client_secret_aqui

# API Token (OPCIONAL - será substituído pelo OAuth)
DERIV_API_TOKEN=seu_token_manual_aqui

# Railway Detection (JÁ CONFIGURADO)
RAILWAY=true
```

### 2. Como Configurar as Variáveis

1. **Acesse seu projeto no Railway Dashboard**
2. **Vá para a aba "Variables"**
3. **Adicione cada variável individualmente:**
   - `DERIV_APP_ID`: `64394`
   - `DERIV_CLIENT_SECRET`: Seu client secret do Deriv
   - `DERIV_API_TOKEN`: Seu token da API (opcional)

### 3. Obter Credenciais do Deriv

#### App ID e Client Secret:
1. Acesse [app.deriv.com/account/api-token](https://app.deriv.com/account/api-token)
2. Clique em **"Register application"**
3. Configure:
   - **Name**: `Deriv AI Trading Bot`
   - **Scopes**: `read,trade,payments,admin`
   - **Redirect URL**: `https://seu-app.railway.app/auth/callback`
4. **Copie o App ID e Client Secret**

## 🔐 Processo de Login no Railway

### Diferenças do Ambiente Local

No Railway, o processo de login é **manual** devido às limitações de porta:

1. **Clique em "Fazer Login"** no dashboard
2. **Uma URL será exibida** no console/logs
3. **Acesse a URL manualmente** no seu navegador
4. **Faça login** na sua conta Deriv
5. **Autorize o aplicativo**
6. **Copie o código** da URL de retorno
7. **Cole o código** no campo do dashboard

### Exemplo de URL de Retorno:
```
https://seu-app.railway.app/auth/callback?code=ABC123XYZ&state=...
```
**Copie apenas**: `ABC123XYZ`

## 🛠️ Arquivos de Configuração

### railway.toml
```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/_stcore/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3

[env]
RAILWAY = "true"
STREAMLIT_SERVER_PORT = "$PORT"
STREAMLIT_SERVER_ADDRESS = "0.0.0.0"
# ... outras configurações
```

### Dockerfile
```dockerfile
# Configuração otimizada para Railway
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Expor porta
EXPOSE 8080

# Comando de inicialização
CMD streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

## 🔍 Troubleshooting

### Problema: "Address already in use"
**Solução**: ✅ **Resolvido automaticamente**
- O sistema detecta o ambiente Railway
- Usa autenticação manual em vez de servidor local
- Não há mais conflito de portas

### Problema: "OAuth token required"
**Solução**: 
1. Verifique se `DERIV_APP_ID` está configurado
2. Complete o processo de login manual
3. Verifique se o código foi inserido corretamente

### Problema: "Invalid client"
**Solução**:
1. Verifique se o `DERIV_CLIENT_SECRET` está correto
2. Confirme se o App ID é `64394`
3. Verifique se a URL de redirect está correta

## 📊 Monitoramento

### Logs do Railway
```bash
# Ver logs em tempo real
railway logs --follow

# Filtrar logs de autenticação
railway logs | grep "auth_manager\|OAuth"
```

### Status da Aplicação
- **Health Check**: `https://seu-app.railway.app/_stcore/health`
- **Dashboard**: `https://seu-app.railway.app`

## 🚀 Deploy Automático

### Via GitHub
1. **Conecte seu repositório** ao Railway
2. **Configure as variáveis** de ambiente
3. **Push para main** - deploy automático

### Via CLI
```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway up
```

## 🔒 Segurança

### Boas Práticas:
- ✅ **Nunca commite** client secrets no código
- ✅ **Use variáveis de ambiente** do Railway
- ✅ **Configure HTTPS** (automático no Railway)
- ✅ **Monitore logs** regularmente

### Variáveis Sensíveis:
- `DERIV_CLIENT_SECRET`: **NUNCA** exponha publicamente
- `DERIV_API_TOKEN`: Mantenha seguro
- Tokens OAuth: Renovados automaticamente

## 📞 Suporte

### Problemas Comuns:
1. **Erro de porta**: Resolvido automaticamente
2. **OAuth falha**: Verifique credenciais
3. **Deploy falha**: Verifique logs do Railway

### Recursos:
- [Railway Docs](https://docs.railway.app)
- [Deriv API Docs](https://developers.deriv.com)
- [Streamlit Docs](https://docs.streamlit.io)