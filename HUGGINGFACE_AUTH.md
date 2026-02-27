# Autenticação HuggingFace para TabPFN v2.5

## Problema
O modelo TabPFN v2.5 é um modelo "gated" no HuggingFace que requer autenticação para download. Sem a autenticação, você receberá o seguinte erro:

```
RuntimeError: Failed to download TabPFN ModelVersion.V2_5 model 'tabpfn-v2.5-regressor-v2.5_default.ckpt'.
Details and instructions:
HuggingFace authentication error downloading from 'Prior-Labs/tabpfn_2_5'.
This model is gated and requires you to accept its terms.
```

## Solução

### Passo 1: Aceitar os Termos na HuggingFace

1. Visite https://huggingface.co/Prior-Labs/tabpfn_2_5
2. Faça login na sua conta HuggingFace ou crie sua conta e confirme o email de verificação
3. Volte para https://huggingface.co/Prior-Labs/tabpfn_2_5 e clique em "Agree and access repository" para aceitar os termos de uso

### Passo 2: Obter seu Token HuggingFace

1. Vá para https://huggingface.co/settings/tokens
2. Clique em "New token" (Nova Token)
3. Crie um token com acesso "read" (leitura)
4. Copie o token gerado

### Passo 3: Configurar o Environment

#### crie um arquivo `.env` na raiz do projeto:
```
HF_TOKEN=seu_token_aqui
```

Depois carregue as variáveis:
```bash
source .env  # Linux/Mac
set -a; source .env; set +a  # Se usar bash
```

### Passo 4: Executar o Docker

Com o `HF_TOKEN` configurado, execute:

```bash
docker compose up -d --build
```

O token será passado automaticamente durante o build e será disponível para o container em tempo de execução.

## Método Alternativo (Sem Token no Build)

Se você preferir não passar o token durante o build, o código tentará fazer login automaticamente em tempo de execução usando a variável de ambiente `HF_TOKEN`. Basta garantir que ela esteja disponível:

```bash
docker compose up
```

## Método Manual (Teste Local)

Para testar a autenticação localmente sem Docker:

```bash
from huggingface_hub import login
login(token="seu_token_aqui")
```

## Segurança

⚠️ **Importante**: Nunca submeta seu token ao controle de versão (git).

- Adicione `.env` ao `.gitignore`
- Use variáveis de ambiente do seu CI/CD (GitHub Actions, GitLab CI, etc.)
- Em produção, use secrets seguros do seu orquestrador (Kubernetes secrets, Docker Swarm secrets, etc.)

## Para Uso Comercial

Se você pretende usar o TabPFN v2.5 comercialmente, entre em contato com a Prior Labs em sales@priorlabs.ai para opções de download alternativas.
