# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Instala dependências de build necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências em um diretório temporário
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copia apenas os pacotes instalados do builder
COPY --from=builder /root/.local /root/.local

# Atualiza PATH para incluir os pacotes do usuário
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Copia código da aplicação
COPY . .

EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]