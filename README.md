# MIST_Download
Este repositório é para que aqueles que desejarem baixar nossa ferramenta MIST-Saúde, que será apresentada na SBCAS 2026

## Passos para rodar a ferramente:

 - TabPFN
O modelo TabPFN v2.5 é um modelo "gated" no HuggingFace que requer autenticação para download. Sem a autenticação, você receberá um erro, os passos abaixo são necessários para a execução sem erro:
```

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

 - Acesse localhost:3000
 - Após o uso, digite no terminal:

 ```bash
 docker compose down
 ```
 para parar e remover os containers.