# API de Imputação de Dados (Machine Learning)

Esta é uma API REST construída com **FastAPI** e conteinerizada com **Docker**. O objetivo deste serviço é receber bases de dados (`.csv` ou `.parquet`) com valores nulos (missing values) e devolver o arquivo preenchido utilizando algoritmos avançados de Machine Learning, evitando o vazamento de dados (*data leakage*).

## 🚀 Como Iniciar o Servidor

Para rodar esta API na sua máquina local utilizando o Docker, abra o terminal na pasta onde os arquivos estão e execute os seguintes comandos:

### 1. Construir a Imagem Docker (Build)
Este comando prepara o ambiente isolado, instalando o Python, o FastAPI e todas as bibliotecas de Machine Learning necessárias (Pandas, Scikit-Learn, TabPFN, etc.).

sudo docker build -t api-imputacao .

*(Nota: Você só precisa rodar este comando a primeira vez ou se alterar o código no arquivo api.py ou requirements.txt).*

### 2. Ligar a API (Run)
Este comando inicia o servidor web e faz a ponte da porta 8000 do container para o seu computador.

sudo docker run -p 8000:8000 api-imputacao

---

## 💻 Como Utilizar (Interface Gráfica)

O FastAPI gera automaticamente uma interface visual interativa para testar a API sem precisar programar nada.

1. Com o servidor ligado, abra o seu navegador de internet.
2. Acesse o endereço: **http://localhost:8000/docs**
3. Clique na barra verde **`POST /imputar`**.
4. Clique no botão **"Try it out"** no canto superior direito.
5. Preencha o formulário:
   - **arquivo**: Faça o upload do seu arquivo `.csv` ou `.parquet`.
   - **metodo**: Escolha o algoritmo (ex: `mice`, `knn`, `media`, `missforest`, `tabpfn`).
   - **ignorar**: (Opcional) Digite o nome das colunas que o modelo **não** deve utilizar para aprender, separadas por vírgula (ex: `id, target`). Isso previne o *data leakage*.
   - **feature_alvo**: (Obrigatório apenas se o método for `tabpfn`).
6. Clique em **"Execute"**.
7. Na seção "Responses" mais abaixo, clique em **"Download file"** para baixar a base de dados tratada.

---
**Para desligar o servidor:** Pressione `Ctrl + C` no terminal onde o Docker está rodando.
