# 🚀 Guia de Deploy no Railway - Deriv AI Trading Bot

Este guia explica como fazer deploy do Deriv AI Trading Bot na plataforma Railway.

## 📋 Pré-requisitos

1. **Conta no Railway**: [railway.app](https://railway.app)
2. **Conta no GitHub**: Para conectar o repositório
3. **Token da API Deriv**: Obtenha em [app.deriv.com](https://app.deriv.com)

## 🔧 Preparação do Projeto

### 1. Arquivos de Configuração Criados

✅ **Dockerfile** - Containerização da aplicação
✅ **railway.toml** - Configurações específicas do Railway
✅ **.dockerignore** - Otimização do build
✅ **start.sh** - Script de inicialização
✅ **requirements.txt** - Dependências fixas para produção

### 2. Configurações Ajustadas

- **config.py**: Suporte a variáveis de ambiente de produção
- **main.py**: Argumentos de linha de comando para host/port
- **Streamlit**: Configurado para produção

## 🚀 Deploy Passo a Passo

### Passo 1: Preparar Repositório

```bash
# 1. Inicializar git (se ainda não foi feito)
git init

# 2. Adicionar arquivos
git add .

# 3. Commit inicial
git commit -m "feat: configuração para deploy no Railway"

# 4. Conectar ao GitHub
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

### Passo 2: Criar Projeto no Railway

1. **Acesse**: [railway.app](https://railway.app)
2. **Login** com GitHub
3. **New Project** → **Deploy from GitHub repo**
4. **Selecione** seu repositório
5. **Deploy Now**

### Passo 3: Configurar Variáveis de Ambiente

No painel do Railway, vá em **Variables** e adicione:

#### 🔑 Variáveis Obrigatórias

```env
# API Deriv
DERIV_API_TOKEN=seu_token_aqui
DERIV_APP_ID=1089

# Ambiente
RAILWAY_ENVIRONMENT=production
DEBUG=false

# Streamlit
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

#### 🔧 Variáveis Opcionais

```env
# Trading (opcional)
TRADING_INITIAL_STAKE=1.0
TRADING_MAX_DAILY_LOSS=50.0
TRADING_ENABLE_MARTINGALE=false

# ML Model (opcional)
ML_MODEL_TYPE=lightgbm
ML_MIN_PREDICTION_CONFIDENCE=0.6
```

### Passo 4: Verificar Deploy

1. **Aguarde** o build completar (5-10 minutos)
2. **Acesse** a URL fornecida pelo Railway
3. **Teste** o dashboard

## 🔍 Verificação e Monitoramento

### Health Check

O Railway verificará automaticamente:
- **Endpoint**: `/_stcore/health`
- **Timeout**: 300 segundos
- **Restart**: Automático em caso de falha

### Logs

```bash
# Ver logs em tempo real
railway logs

# Ver logs específicos
railway logs --tail 100
```

### Métricas

No painel Railway você pode monitorar:
- **CPU Usage**
- **Memory Usage**
- **Network Traffic**
- **Response Time**

## 🛠️ Comandos Úteis

### Railway CLI

```bash
# Instalar CLI
npm install -g @railway/cli

# Login
railway login

# Conectar ao projeto
railway link

# Ver status
railway status

# Redeploy
railway up
```

### Desenvolvimento Local

```bash
# Testar configuração de produção localmente
export RAILWAY_ENVIRONMENT=production
export PORT=8080
python main.py --mode dashboard --host 0.0.0.0 --port 8080
```

## 🔧 Configurações Avançadas

### Custom Domain

1. **Railway Dashboard** → **Settings** → **Domains**
2. **Add Custom Domain**
3. **Configure DNS** conforme instruções

### Scaling

```toml
# railway.toml
[deploy]
replicas = 1
restartPolicyType = "on_failure"
```

### Environment-specific Configs

```env
# Produção
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Staging
ENVIRONMENT=staging
LOG_LEVEL=DEBUG
DEBUG=true
```

## 🚨 Troubleshooting

### Problemas Comuns

#### 1. Build Falha
```bash
# Verificar logs
railway logs --deployment

# Soluções:
- Verificar Dockerfile
- Conferir requirements.txt
- Checar variáveis de ambiente
```

#### 2. App Não Inicia
```bash
# Verificar se PORT está configurada
echo $PORT

# Verificar se host está correto
# Deve ser 0.0.0.0, não localhost
```

#### 3. Timeout no Health Check
```bash
# Aumentar timeout no railway.toml
[deploy]
healthcheckTimeout = 600
```

#### 4. Erro de Permissão
```bash
# Verificar se start.sh tem permissão
chmod +x start.sh
```

### Debug Avançado

```bash
# Conectar ao container
railway shell

# Verificar variáveis
env | grep -E "(DERIV|RAILWAY|STREAMLIT)"

# Testar manualmente
python main.py --mode status
```

## 📊 Monitoramento de Performance

### Métricas Importantes

- **Response Time**: < 2s
- **Memory Usage**: < 512MB
- **CPU Usage**: < 50%
- **Error Rate**: < 1%

### Alertas

Configure alertas no Railway para:
- **High Memory Usage**
- **High CPU Usage**
- **Application Crashes**
- **Failed Deployments**

## 🔐 Segurança

### Boas Práticas

1. **Nunca** commite tokens no código
2. **Use** variáveis de ambiente para secrets
3. **Configure** CORS adequadamente
4. **Monitore** logs para atividades suspeitas

### Variáveis Sensíveis

```env
# ❌ NUNCA faça isso
DERIV_API_TOKEN=abc123  # No código

# ✅ Sempre use variáveis de ambiente
DERIV_API_TOKEN=  # Configurar no Railway
```

## 📈 Otimizações

### Performance

1. **Docker Multi-stage**: Para builds menores
2. **Caching**: Dependências Python
3. **Compression**: Assets estáticos
4. **CDN**: Para assets globais

### Custos

- **Starter Plan**: Gratuito (limitado)
- **Developer Plan**: $5/mês
- **Team Plan**: $20/mês

## 🎯 Próximos Passos

Após deploy bem-sucedido:

1. **Configure** domínio personalizado
2. **Setup** monitoramento avançado
3. **Implemente** CI/CD pipeline
4. **Configure** backup de dados
5. **Setup** staging environment

## 📞 Suporte

- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **GitHub Issues**: Para problemas específicos do bot

---

## ✅ Checklist de Deploy

- [ ] Repositório no GitHub
- [ ] Projeto criado no Railway
- [ ] Variáveis de ambiente configuradas
- [ ] Build completado com sucesso
- [ ] Health check passando
- [ ] Dashboard acessível
- [ ] Logs sem erros críticos
- [ ] Autenticação funcionando
- [ ] Monitoramento configurado

**🎉 Parabéns! Seu Deriv AI Trading Bot está rodando em produção!**