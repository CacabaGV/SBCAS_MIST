FROM python:3.10-slim

WORKDIR /app

# Instala ferramentas básicas do Linux
RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

# Baixa a versão leve do PyTorch primeiro (apenas CPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do seu amigo e a sua API
COPY sbcas_imputacao /app/sbcas_imputacao
COPY api.py .

# Libera a porta do servidor
EXPOSE 8000

# Inicia o servidor FastAPI em vez do script antigo
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]