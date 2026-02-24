from fastapi import FastAPI, File, UploadFile, Form
import pandas as pd
import os
import io
import base64
from fastapi.responses import FileResponse, JSONResponse
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware

from sbcas_imputacao.benchmarking.experiment_runner import ExperimentRunner

app = FastAPI(title="API de Imputação de Dados")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para desenvolvimento
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/imputar")
async def imputar_dados(
    arquivo: UploadFile = File(...),
    metodo: str = Form("mice"),
    features_a_imputar: str = Form("features_a_imputadar"), 
    ignorar: str = Form(None)
):
    caminho_entrada = f"temp_{arquivo.filename}"
    with open(caminho_entrada, "wb") as f:
        f.write(await arquivo.read())
        
    if caminho_entrada.endswith('.parquet'):
        df = pd.read_parquet(caminho_entrada)
    else:
        df = pd.read_csv(caminho_entrada)

    colunas_originais = df.columns.tolist()

    colunas_ignoradas = {}
    if ignorar:
        colunas_para_ignorar = [col.strip() for col in ignorar.split(',')]
        for col in colunas_para_ignorar:
            if col in df.columns:
                colunas_ignoradas[col] = df[col].copy()
                df = df.drop(columns=[col])

    if features_a_imputar:
        colunas_alvo = [col.strip() for col in features_a_imputar.split(',')]
    else:
        colunas_alvo = df.columns[df.isna().any()].tolist()

    runner = ExperimentRunner()
    df_imputed = df.copy()
    
    for col in colunas_alvo:
        if col in df_imputed.columns and df_imputed[col].isna().any():
            df_imputed = runner.imputar(df=df_imputed, algoritmo=metodo, feature=col)

    for col, valores in colunas_ignoradas.items():
        df_imputed[col] = valores
        
    df_imputed = df_imputed[colunas_originais]

    caminho_saida = f"imputado_{arquivo.filename}"
    if caminho_saida.endswith('.parquet'):
        df_imputed.to_parquet(caminho_saida, index=False)
    else:
        df_imputed.to_csv(caminho_saida, index=False)
        
    os.remove(caminho_entrada)
    
    return FileResponse(path=caminho_saida, filename=caminho_saida)


# ---> NOVA ROTA: DESCRIBE E GRAFICOS <---
@app.post("/describe")
async def descrever_dados(
    arquivo: UploadFile = File(...),
    feature_grafico: str = Form("feature_grafico")
):
    caminho_entrada = f"temp_desc_{arquivo.filename}"
    with open(caminho_entrada, "wb") as f:
        f.write(await arquivo.read())
        
    if caminho_entrada.endswith('.parquet'):
        df = pd.read_parquet(caminho_entrada)
    else:
        df = pd.read_csv(caminho_entrada)

    runner = ExperimentRunner()
    
    # 1. Pega as informações básicas da tabela
    info_basica = runner.describe(df)
    linhas, colunas = info_basica["shape"]
    estatisticas = info_basica["describe"].fillna("NaN").to_dict() # fillna evita erros no JSON

    # 2. Roda o experimento pesado para gerar o grafico
    resultado_exp = runner.run(df=df, feature=feature_grafico)
    fig = resultado_exp["imagem"]
    erros = resultado_exp["erro_medio"].fillna("NaN").to_dict()

    # 3. Converte a imagem para Base64 (Formato ideal para o Front-end)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    
    # Limpa a memória do servidor para não travar o Docker
    plt.close(fig)
    os.remove(caminho_entrada)

    # 4. Devolve tudo num pacote JSON organizado
    return JSONResponse(content={
        "informacoes_gerais": {
            "total_linhas": linhas,
            "total_colunas": colunas
        },
        "estatisticas_descritivas": estatisticas,
        "erros_por_quadrante": erros,
        "grafico_quadrantes_base64": f"data:image/png;base64,{img_base64}"
    })