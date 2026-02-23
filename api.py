from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import pandas as pd
import torch
import os
from tabpfn import TabPFNRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

app = FastAPI(title="API de Imputação de Dados")

@app.post("/imputar")
async def imputar_dados(
    arquivo: UploadFile = File(...),
    metodo: str = Form("mice"),
    feature_alvo: str = Form(None),
    ignorar: str = Form(None)
):
    # Salva o arquivo recebido pelo navegador temporariamente no container
    caminho_entrada = f"temp_{arquivo.filename}"
    with open(caminho_entrada, "wb") as f:
        f.write(await arquivo.read())
        
    # Lê os dados
    if caminho_entrada.endswith('.parquet'):
        df = pd.read_parquet(caminho_entrada)
    else:
        df = pd.read_csv(caminho_entrada)

    # ---> CORREÇÃO: SALVA A ORDEM E QUAIS ERAM AS COLUNAS ORIGINAIS <---
    colunas_originais = df.columns.tolist()

    # --- Lógica de isolar colunas ---
    colunas_ignoradas = {}
    if ignorar:
        colunas_para_ignorar = [col.strip() for col in ignorar.split(',')]
        for col in colunas_para_ignorar:
            if col in df.columns:
                colunas_ignoradas[col] = df[col].copy()
                df = df.drop(columns=[col])

    # --- Lógica de Imputação ---
    if metodo == 'media':
        df_imputed = df.copy()
        for col in df_imputed.columns:
            if df_imputed[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
                
    elif metodo == 'knn':
        imputer = KNNImputer(n_neighbors=10)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        
    elif metodo == 'mice':
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=7)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        
    elif metodo == 'missforest':
        imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=7), max_iter=20, random_state=7)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        
    elif metodo == 'tabpfn':
        df_imputed = df.copy()
        df_train = df[df[feature_alvo].notna()]
        df_missing = df[df[feature_alvo].isna()]
        
        if len(df_missing) > 0:
            X_train = df_train.drop(columns=[feature_alvo])
            y_train = df_train[feature_alvo]
            X_test = df_missing.drop(columns=[feature_alvo])
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            reg = TabPFNRegressor(device=device)
            reg.fit(X_train.values, y_train.values)
            preds = reg.predict(X_test.values)
            df_imputed.loc[df_imputed[feature_alvo].isna(), feature_alvo] = preds

    # --- Devolve as colunas e reorganiza a ordem original ---
    for col, valores in colunas_ignoradas.items():
        df_imputed[col] = valores
        
    # ---> CORREÇÃO: REORGANIZA USANDO A LISTA QUE SALVAMOS LÁ EM CIMA <---
    df_imputed = df_imputed[colunas_originais]

    # Salva o resultado e prepara para enviar de volta ao usuário
    caminho_saida = f"imputado_{arquivo.filename}"
    if caminho_saida.endswith('.parquet'):
        df_imputed.to_parquet(caminho_saida, index=False)
    else:
        df_imputed.to_csv(caminho_saida, index=False)
        
    # Limpa o arquivo de entrada para não lotar o container
    os.remove(caminho_entrada)
    
    # O FastAPI empacota o arquivo e faz o download automático no navegador
    return FileResponse(path=caminho_saida, filename=caminho_saida)