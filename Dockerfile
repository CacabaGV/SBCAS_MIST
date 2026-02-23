FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia a nossa API
COPY api.py .

# Expõe a porta para a internet/navegador
EXPOSE 8000

# Liga o servidor web
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
