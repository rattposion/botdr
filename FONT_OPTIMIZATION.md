# Otimizações de Fontes e Preload - Deriv AI Trading Bot

## Problema Identificado
O dashboard estava gerando avisos no console do navegador sobre recursos de fontes sendo pré-carregados mas não utilizados:

```
The resource https://botdr-production.up.railway.app/static/media/SourceSansPro-Bold.118dea98980e20a81ced.woff2 was preloaded using link preload but not used within a few seconds from the window's load event.
```

## Soluções Implementadas

### 1. Configuração do Streamlit (`.streamlit/config.toml`)
```toml
[global]
developmentMode = false

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6C37"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[runner]
magicEnabled = true

[logger]
level = "info"
messageFormat = "%(asctime)s %(message)s"
```

### 2. CSS Customizado no Dashboard
Adicionado CSS otimizado no `dashboard.py`:

```css
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
```

### 3. Configurações do Railway (`railway.toml`)
Adicionadas variáveis de ambiente para otimização:

```toml
STREAMLIT_GLOBAL_DEVELOPMENT_MODE = "false"
STREAMLIT_CLIENT_CACHING = "true"
STREAMLIT_RUNNER_INSTALL_TRACER = "false"
```

### 4. Exclusões no Docker (`.dockerignore`)
Adicionadas exclusões para arquivos de fonte que podem causar problemas:

```
# Font and static files that cause preload warnings
*.woff
*.woff2
*.ttf
*.eot
static/
assets/fonts/

# Streamlit cache
.streamlit/secrets.toml
```

## Benefícios das Otimizações

1. **Redução de Avisos**: Eliminação dos avisos de preload no console do navegador
2. **Performance Melhorada**: Carregamento mais rápido da interface
3. **Uso de Fontes Locais**: Priorização de fontes do sistema quando disponíveis
4. **Configuração Limpa**: Remoção de configurações desnecessárias do Streamlit

## Verificação
- ✅ Dashboard rodando sem avisos de configuração inválida
- ✅ Interface carregando corretamente
- ✅ Fontes otimizadas implementadas
- ✅ Configurações de produção aplicadas

## Comandos para Deploy
```bash
# Local
streamlit run dashboard.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

# Railway (automático com as configurações do railway.toml)
railway up
```

## Notas Técnicas
- As configurações são compatíveis com Streamlit 1.x
- O CSS customizado não interfere na funcionalidade do bot
- As otimizações são aplicadas tanto em desenvolvimento quanto em produção
- O arquivo de configuração `.streamlit/config.toml` é automaticamente carregado pelo Streamlit