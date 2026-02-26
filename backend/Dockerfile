FROM python:3.11

WORKDIR /app

# copia requirements para /app/requirements.txt
COPY requirements.txt .

# executa dependencias, na hora que faz o build da imagem
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# executado no run do docker, como se digitasse isso no terminal do container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]