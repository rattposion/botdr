# 🔧 Configuração da API Deriv

## Como obter e configurar seu App ID da Deriv

### 1. Criar uma conta na Deriv
- Acesse [https://app.deriv.com](https://app.deriv.com)
- Crie uma conta ou faça login

### 2. Registrar uma aplicação
- Vá para **Configurações da Conta** → **API Token**
- Clique em **"Create new token"** ou **"Register new application"**
- Preencha os dados:
  - **Name**: Nome da sua aplicação (ex: "AI Trading Bot")
  - **Scopes**: Selecione as permissões necessárias:
    - ✅ **Read**: Para ler dados de mercado
    - ✅ **Trade**: Para executar trades (se for usar conta real)
    - ✅ **Payments**: Para operações financeiras
    - ✅ **Trading Information**: Para informações de trading
    - ✅ **Admin**: Para operações administrativas

### 3. Obter o App ID
Após registrar a aplicação, você receberá:
- **App ID**: Um número (ex: 12345)
- **API Token**: Uma string longa (ex: abc123xyz...)

### 4. Configurar no projeto

#### Opção 1: Arquivo .env (Recomendado)
Edite o arquivo `.env` na raiz do projeto:

```env
# Configurações da API Deriv
DERIV_APP_ID=SEU_APP_ID_AQUI
DERIV_API_TOKEN=SEU_TOKEN_AQUI

# Configurações de Trading
INITIAL_STAKE=1.0
MAX_DAILY_LOSS=50.0

# Ambiente
ENVIRONMENT=development
```

#### Opção 2: Diretamente no código
Edite o arquivo `config.py`:

```python
@dataclass
class DerivConfig:
    app_id: str = "SEU_APP_ID_AQUI"
    api_token: str = "SEU_TOKEN_AQUI"
    # ... resto da configuração
```

### 5. Testar a conexão
1. Reinicie o dashboard: `streamlit run dashboard.py`
2. Vá para a aba **"⚙️ Configurações"**
3. Clique em **"Testar Conexão API"**
4. Verifique se aparece ✅ **"Conectado com sucesso"**

## 🔒 Segurança

### Para desenvolvimento/teste:
- Use o App ID padrão: `1089` (já configurado)
- Funciona apenas com conta demo
- Não requer token API

### Para produção:
- **NUNCA** compartilhe seu token API
- Use variáveis de ambiente (arquivo .env)
- Mantenha o arquivo .env no .gitignore
- Use conta demo primeiro para testar

## 🚨 Solução de Problemas

### Erro: "The request is missing a valid app_id"
- ✅ Verifique se o App ID está configurado no .env
- ✅ Reinicie o servidor após alterar o .env
- ✅ Use o App ID padrão `1089` para testes

### Erro: "Invalid token"
- ✅ Verifique se o token está correto
- ✅ Verifique se o token não expirou
- ✅ Gere um novo token se necessário

### Erro de conexão WebSocket
- ✅ Verifique sua conexão com a internet
- ✅ Tente usar o App ID padrão `1089`
- ✅ Verifique se não há firewall bloqueando

## 📞 Suporte

Se ainda tiver problemas:
1. Verifique os logs no terminal
2. Teste com o App ID padrão `1089`
3. Consulte a [documentação oficial da Deriv API](https://developers.deriv.com/)
4. Entre em contato com o suporte da Deriv

---

**⚠️ Importante**: Sempre teste com conta demo antes de usar conta real!