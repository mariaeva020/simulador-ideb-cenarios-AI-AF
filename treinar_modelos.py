# ============================================================
# TREINAMENTO DOS MODELOS PARA O SIMULADOR DE CENÁRIOS DO IDEB
# ============================================================
#
# Objetivo:
# Este script realiza o processamento das bases dos Anos Iniciais
# e dos Anos Finais do Ensino Fundamental, treina um modelo XGBoost
# para cada etapa e salva os modelos treinados na pasta "models".
#
# Observação metodológica:
# A aplicação não realiza previsão futura nem inferência causal.
# O modelo é utilizado para simular cenários condicionais sobre
# os dados observados em 2023, conforme a lógica definida no artigo.
#
# Etapas contempladas:
# 1. Anos Iniciais do Ensino Fundamental
# 2. Anos Finais do Ensino Fundamental
#
# O Ensino Médio foi removido porque não faz parte do escopo desta
# versão da aplicação.
# ============================================================


# ============================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import os
import random
import warnings
import joblib

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

# Semente utilizada para reprodutibilidade dos resultados.
SEED = 42

# Define a semente para o NumPy.
np.random.seed(SEED)

# Define a semente para operações aleatórias do Python.
random.seed(SEED)

# Oculta avisos que não comprometem a execução do script.
warnings.filterwarnings("ignore")


# ============================================================
# CAMINHOS DOS ARQUIVOS
# ============================================================

# Pasta onde estão as bases de dados.
PASTA_DADOS = "data"

# Pasta onde os modelos e métricas serão salvos.
PASTA_MODELOS = "models"

# Cria a pasta "models", caso ela ainda não exista.
os.makedirs(PASTA_MODELOS, exist_ok=True)

# Caminhos das bases utilizadas no artigo e na aplicação.
CAMINHOS_BASES = {
    "Anos Iniciais": os.path.join(PASTA_DADOS, "base_anos_iniciais.xlsx"),
    "Anos Finais": os.path.join(PASTA_DADOS, "base_anos_finais.xlsx")
}


# ============================================================
# VARIÁVEIS UTILIZADAS NO MODELO FINAL
# ============================================================
#
# Estas são as variáveis utilizadas no modelo XGBoost para Anos
# Iniciais e Anos Finais, conforme a estrutura presente no script
# original de simulação.
# ============================================================

VARIAVEIS_MODELO = [
    "PIB per capita",
    "taxa_distorcao_idade_serie",
    "Grupo 5-adeq form docente",
    "Área plantada ou destinada à colheita -lavour",
    "Valor da produção na extração vegetal",
    "Valor repassado-Criança Feliz"
]


# ============================================================
# FUNÇÃO DE PROCESSAMENTO DAS BASES
# ============================================================

def processar_base(caminho_arquivo: str) -> pd.DataFrame:
    """
    Lê, limpa e prepara a base de dados para treinamento do modelo.

    Parâmetros
    ----------
    caminho_arquivo : str
        Caminho do arquivo Excel contendo a base de dados.

    Retorno
    -------
    pd.DataFrame
        Base tratada e pronta para modelagem.
    """

    # Verifica se o arquivo informado existe.
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(
            f"O arquivo não foi encontrado: {caminho_arquivo}. "
            "Verifique se a base foi colocada corretamente na pasta 'data'."
        )

    # Lê a base em formato Excel.
    df = pd.read_excel(caminho_arquivo)

    # Remove colunas SAEB específicas que apresentavam alta proporção
    # de dados ausentes no fluxo original.
    colunas_remover = [
        "Nota SAEB em Matemática",
        "Nota SAEB em Língua Portuguesa"
    ]

    df.drop(
        columns=[col for col in colunas_remover if col in df.columns],
        inplace=True,
        errors="ignore"
    )

    # Remove registros sem IDEB e sem Nota Média Padronizada.
    # Esses registros não podem ser usados no treinamento porque a
    # variável-alvo IDEB precisa estar disponível.
    colunas_obrigatorias = [
        "IDEB",
        "Nota SAEB - Nota Média Padronizada (N)"
    ]

    for coluna in colunas_obrigatorias:
        if coluna not in df.columns:
            raise ValueError(
                f"A coluna obrigatória '{coluna}' não foi encontrada na base."
            )

    df = df[df[colunas_obrigatorias].notnull().all(axis=1)].copy()

    # Lista de variáveis que podem receber imputação por mediana, caso
    # estejam presentes na base.
    variaveis_para_imputar = [
        "Valor Total repassado ao BPC",
        "Total de Beneficiários do BPC",
        "Idosos beneficiários do BPC",
        "Pessoas com Deficiência beneficiárias do BPC",
        "Valor Repassado a PCDs pelo RMV",
        "Valor Repassado a Idosos pelo RMV",
        "Valor Repassado a Idosos pelo BPC",
        "Valor Repassado a PCDs pelo BPC",
        "iptu",
        "itbi",
        "irrf",
        "iss",
        "cota parte icms",
        "cota parte ipi-exp",
        "cota-parte ipva",
        "fpm",
        "salário-educação",
        "pdde",
        "pnae",
        "pnate",
        "convênios",
        "operação de créditos",
        "contribuição na formação do fundef/fundeb – destinada",
        "cota-parte iof-ouro",
        "receita recebida na redistribuição interna do fundeb (transferência de recursos do fundeb)",
        "resultado líquido (ganhos ou perdas = recebido - enviado)",
        "receita da aplicação financeira do fundeb",
        "complementação da união",
        "vaaf",
        "total_composiçâo_complementaçâo_fundeb",
        "receitas recebidas do fundeb (fundo estadual)",
        "receitas destinadas ao fundeb (fundo estadual)",
        "valor aplicado em mde",
        "educação infantil",
        "creche",
        "pré-escola",
        "ensino fundamental",
        "ensino profissional não integrado ao ensino regular",
        "ensino médio",
        "quota do salário-educação",
        "despesas com profissionais da educação básica",
        "Valor da produção -lavour",
        "Área colhida -lavour"
    ]

    # Imputa valores ausentes pela mediana de cada variável.
    # A mediana é menos sensível a valores extremos do que a média.
    for variavel in variaveis_para_imputar:
        if variavel in df.columns:
            mediana = df[variavel].median()
            df[variavel] = df[variavel].fillna(mediana)

    # Verifica se todas as variáveis do modelo estão presentes.
    variaveis_ausentes = [
        var for var in VARIAVEIS_MODELO if var not in df.columns
    ]

    if variaveis_ausentes:
        raise ValueError(
            "As seguintes variáveis do modelo não foram encontradas na base: "
            + ", ".join(variaveis_ausentes)
        )

    # Remove linhas com valores ausentes nas variáveis finais do modelo.
    # Isso evita erro no treinamento do XGBoost.
    df = df.dropna(subset=VARIAVEIS_MODELO + ["IDEB"]).copy()

    return df


# ============================================================
# FUNÇÃO DE TREINAMENTO E AVALIAÇÃO DO MODELO
# ============================================================

def treinar_e_avaliar_modelo(df: pd.DataFrame) -> tuple:
    """
    Treina o modelo XGBoost e calcula métricas de avaliação.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base tratada contendo as variáveis explicativas e a variável-alvo.

    Retorno
    -------
    tuple
        Modelo treinado e dicionário com métricas de avaliação.
    """

    # Define a matriz de variáveis explicativas.
    X = df[VARIAVEIS_MODELO].copy()

    # Define a variável-alvo.
    y = df["IDEB"].copy()

    # Divide os dados em treinamento e teste.
    # A proporção de teste é 30%, conforme o fluxo original.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED
    )

    # Define o modelo XGBoost com os hiperparâmetros utilizados
    # no script original.
    modelo = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        n_jobs=-1,
        random_state=SEED
    )

    # Treina o modelo com os dados de treinamento.
    modelo.fit(X_train, y_train)

    # Gera previsões para treinamento e teste.
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # Calcula as métricas no conjunto de treinamento.
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Calcula as métricas no conjunto de teste.
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # Validação cruzada com 5 folds.
    # Usa apenas X_train e y_train para evitar vazamento do conjunto de teste.
    scores_r2 = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    scores_mae = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    scores_rmse = np.sqrt(
        -cross_val_score(
            modelo,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_squared_error"
        )
    )

    # Organiza as métricas em um dicionário.
    metricas = {
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "r2_train": r2_train,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test,
        "rmse_cv": np.mean(scores_rmse),
        "mae_cv": abs(np.mean(scores_mae)),
        "r2_cv": np.mean(scores_r2),
        "rmse_std": np.std(scores_rmse),
        "mae_std": np.std(scores_mae),
        "r2_std": np.std(scores_r2),
        "n_linhas": df.shape[0],
        "n_colunas": df.shape[1]
    }

    return modelo, metricas


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    """
    Executa o treinamento dos modelos para as duas etapas de ensino.
    """

    print("=" * 80)
    print("TREINAMENTO DOS MODELOS DO SIMULADOR DE CENÁRIOS DO IDEB")
    print("=" * 80)

    for etapa, caminho in CAMINHOS_BASES.items():
        print(f"\nProcessando etapa: {etapa}")
        print("-" * 80)

        # Processa a base da etapa selecionada.
        df = processar_base(caminho)

        # Treina o modelo e calcula as métricas.
        modelo, metricas = treinar_e_avaliar_modelo(df)

        # Define nomes de arquivos padronizados.
        if etapa == "Anos Iniciais":
            nome_modelo = "modelo_xgb_anos_iniciais.pkl"
            nome_metricas = "metricas_anos_iniciais.pkl"
        else:
            nome_modelo = "modelo_xgb_anos_finais.pkl"
            nome_metricas = "metricas_anos_finais.pkl"

        # Salva o modelo treinado.
        caminho_modelo = os.path.join(PASTA_MODELOS, nome_modelo)
        joblib.dump(modelo, caminho_modelo)

        # Salva as métricas do modelo.
        caminho_metricas = os.path.join(PASTA_MODELOS, nome_metricas)
        joblib.dump(metricas, caminho_metricas)

        # Exibe um resumo das métricas no terminal.
        print(f"Modelo salvo em: {caminho_modelo}")
        print(f"Métricas salvas em: {caminho_metricas}")
        print(f"Número de registros utilizados: {metricas['n_linhas']}")
        print(f"RMSE teste: {metricas['rmse_test']:.4f}")
        print(f"MAE teste: {metricas['mae_test']:.4f}")
        print(f"R² teste: {metricas['r2_test']:.4f}")
        print(
            f"Validação cruzada: "
            f"RMSE={metricas['rmse_cv']:.4f} ± {metricas['rmse_std']:.4f}; "
            f"MAE={metricas['mae_cv']:.4f} ± {metricas['mae_std']:.4f}; "
            f"R²={metricas['r2_cv']:.4f} ± {metricas['r2_std']:.4f}"
        )

    print("\nTreinamento concluído com sucesso.")


# ============================================================
# EXECUÇÃO DO SCRIPT
# ============================================================

if __name__ == "__main__":
    main()