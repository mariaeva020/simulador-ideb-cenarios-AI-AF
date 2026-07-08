# -*- coding: utf-8 -*-
"""
GERAR ARTEFATOS DOS MODELOS FINAIS POR ETAPA
============================================

Este script gera os artefatos .pkl usados pelo simulador Streamlit, sem refazer
VIF, LassoCV, SHAP, Boruta, estabilidade, ranking multicritério, Friedman ou
Wilcoxon.

Ele replica o tratamento de modelagem do script metodológico completo:
1) lê a base por etapa;
2) remove registros com IDEB ausente;
3) define X e y removendo IDEB e colunas de identificação;
4) mantém apenas preditores numéricos;
5) separa treino e teste em 80/20 com random_state=42;
6) ajusta a imputação por mediana apenas no treino e aplica ao teste;
7) ajusta a remoção de variância zero apenas no treino e aplica ao teste;
8) usa diretamente as variáveis finais já definidas para cada etapa;
9) treina o modelo final correto de cada etapa;
10) salva o artefato completo com modelo, imputador, seletor de variância,
    variáveis e métricas.

Modelos finais considerados:
- Anos Iniciais: CatBoost | SHAP
- Anos Finais: EBM | SHAP


"""

import os
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

SEED = 42
TEST_SIZE = 0.20

PASTA_DADOS = "data"
PASTA_MODELOS = "models"
PASTA_OUTPUTS = "outputs"

os.makedirs(PASTA_MODELOS, exist_ok=True)
os.makedirs(PASTA_OUTPUTS, exist_ok=True)


CONFIG_ETAPAS = {
    "anos_iniciais": {
        "nome_etapa": "Anos Iniciais",
        "arquivo_base": os.path.join(PASTA_DADOS, "base_anos_iniciais.xlsx"),
        "arquivo_saida": os.path.join(PASTA_MODELOS, "artefato_modelo_final_anos_iniciais.pkl"),
    },
    "anos_finais": {
        "nome_etapa": "Anos Finais",
        "arquivo_base": os.path.join(PASTA_DADOS, "base_anos_finais.xlsx"),
        "arquivo_saida": os.path.join(PASTA_MODELOS, "artefato_modelo_final_anos_finais.pkl"),
    },
}


COLUNAS_IDENTIFICACAO = [
    "ano",
    "cod_municipio",
    "nome_do_municipio",
    "nome_municipio",
]


# ============================================================
# MODELOS E VARIÁVEIS FINAIS POR ETAPA
# ============================================================

CONFIG_MODELOS_FINAIS = {
    "anos_iniciais": {
        "nome_modelo": "CatBoost",
        "nome_conjunto": "SHAP",
        "variaveis": [
            "taxa_distorcao_idade_serie",
            "grupo_5_adeq_form_docente",
            "percentual_docente_curso_superior",
            "valor_repassado_crianca_feliz",
            "grupo_1_adeq_form_docente",
            "pib_per_capita",
            "iptu",
            "cota_parte_ipva",
            "valor_aplicado_em_mde",
            "nivel_3_esforco_docente",
            "valor_da_producao_na_extracao_vegetal",
            "creche",
            "cota_parte_ipi_exp",
            "receita_da_aplicacao_financeira_do_fundeb",
            "contribuicao_na_formacao_do_fundef_fundeb_–_destinada",
            "nivel_1_gestao_escola",
            "valor_repassado_protecao_social_basica",
            "cota_parte_icms",
            "area_colhida_lavour",
            "nivel_5_gestao_escola",
            "pre_escola",
            "receitas_destinadas_ao_fundeb_fundo_estadual",
            "nivel_3_gestao_escola",
            "quantidade_de_matriculas",
            "valor_repassado_gestao_suas",
            "qt_salas_utiliza_climatizadas",
            "media_alunos_turma",
            "nivel_4_gestao_escola",
            "nivel_5_esforco_docente",
            "media_baixa_regularidade",
        ],
    },
    "anos_finais": {
        "nome_modelo": "EBM",
        "nome_conjunto": "SHAP",
        "variaveis": [
            "taxa_distorcao_idade_serie",
            "qt_salas_utiliza_climatizadas",
            "qt_prof_secretario",
            "qt_escolas_com_agua_potavel",
            "qt_prof_pedagogia",
            "receita_da_aplicacao_financeira_do_fundeb",
            "qt_escolas_com_orgao_conselho_escolar",
            "pib_per_capita",
            "area_colhida_lavour",
            "grupo_5_adeq_form_docente",
            "iptu",
            "valor_da_producao_na_extracao_vegetal",
            "percentual_docente_curso_superior",
            "valor_repassado_crianca_feliz",
            "media_alta_regularidade",
            "pnate",
            "qt_escolas_com_acessibilidade_rampas",
            "grupo_3_adeq_form_docente",
            "valor_repassado_protecao_social_basica",
            "grupo_1_adeq_form_docente",
            "nivel_2_gestao_escola",
            "nivel_5_esforco_docente",
            "nivel_4_esforco_docente",
            "qt_desktop_aluno",
            "media_horas_aula",
            "valor_da_producao_prod_origem_animal",
            "grupo_2_adeq_form_docente",
            "creche",
            "nivel_4_gestao_escola",
            "nivel_3_gestao_escola",
        ],
    },
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================


def validar_configuracoes() -> None:
    """Confere se cada etapa possui 30 variáveis e sem duplicidade."""
    for chave_etapa, config in CONFIG_MODELOS_FINAIS.items():
        variaveis = config["variaveis"]

        if len(variaveis) != 30:
            raise ValueError(
                f"A etapa {chave_etapa} possui {len(variaveis)} variáveis. "
                "O esperado é exatamente 30 variáveis."
            )

        duplicadas = pd.Series(variaveis)[pd.Series(variaveis).duplicated()].tolist()
        if duplicadas:
            raise ValueError(
                f"A etapa {chave_etapa} possui variáveis duplicadas: {duplicadas}"
            )


def criar_modelo_final(nome_modelo: str):
    """Cria o modelo final correspondente à etapa."""
    if nome_modelo == "CatBoost":
        return CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=10,
            loss_function="RMSE",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )

    if nome_modelo == "EBM":
        return ExplainableBoostingRegressor(
            random_state=SEED,
            interactions=10,
        )

    raise ValueError(f"Modelo não configurado: {nome_modelo}")


def identificar_coluna_ideb(df: pd.DataFrame) -> str:
    """Identifica a coluna da variável-alvo IDEB."""
    candidatas = ["ideb", "IDEB", "Ideb"]

    for coluna in candidatas:
        if coluna in df.columns:
            return coluna

    raise ValueError(
        "A variável-alvo IDEB não foi encontrada. "
        "Verifique se a coluna se chama 'ideb', 'IDEB' ou 'Ideb'."
    )


def calcular_rmse(y_real, y_pred) -> float:
    """Calcula a raiz do erro quadrático médio."""
    return float(np.sqrt(mean_squared_error(y_real, y_pred)))


def preparar_base_modelagem(
    df_original: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    """
    Replica a preparação de modelagem do script completo.

    Retorna:
    - X: matriz de preditores numéricos;
    - y: variável-alvo;
    - df_modelagem: base após remoção de IDEB ausente;
    - coluna_ideb: nome da coluna alvo.
    """
    df = df_original.copy()
    coluna_ideb = identificar_coluna_ideb(df)

    linhas_antes = df.shape[0]
    df_modelagem = df[df[coluna_ideb].notnull()].copy()
    linhas_depois = df_modelagem.shape[0]

    print("Linhas antes da remoção de IDEB ausente:", linhas_antes)
    print("Linhas após a remoção de IDEB ausente:", linhas_depois)
    print("Registros removidos:", linhas_antes - linhas_depois)

    colunas_remover = [coluna_ideb] + COLUNAS_IDENTIFICACAO
    colunas_remover = [col for col in colunas_remover if col in df_modelagem.columns]

    X = df_modelagem.drop(columns=colunas_remover, errors="ignore")
    y = df_modelagem[coluna_ideb]

    X = X.select_dtypes(include=[np.number])

    print("Total de variáveis candidatas em X:", X.shape[1])
    print("Total de registros em y:", y.shape[0])
    print("Valores ausentes em y:", y.isna().sum())
    print("Valores ausentes em X antes da divisão:", X.isna().sum().sum())

    return X, y, df_modelagem, coluna_ideb


def treinar_e_salvar_artefato(chave_etapa: str, config_etapa: Dict[str, Any]) -> Dict[str, Any]:
    """Treina o modelo final da etapa e salva o respectivo artefato."""
    print("\n" + "=" * 90)
    print(f"PROCESSANDO: {config_etapa['nome_etapa'].upper()}")
    print("=" * 90)

    config_modelo = CONFIG_MODELOS_FINAIS[chave_etapa]

    nome_modelo = config_modelo["nome_modelo"]
    nome_conjunto = config_modelo["nome_conjunto"]
    variaveis_modelo = config_modelo["variaveis"]

    arquivo_base = config_etapa["arquivo_base"]
    arquivo_saida = config_etapa["arquivo_saida"]

    if not os.path.exists(arquivo_base):
        raise FileNotFoundError(f"Base não encontrada: {arquivo_base}")

    df_original = pd.read_excel(arquivo_base)
    print("Base carregada:", arquivo_base)
    print("Dimensão original:", df_original.shape)

    # 1. Prepara X/y exatamente como no script completo.
    X, y, df_modelagem, coluna_ideb = preparar_base_modelagem(df_original)

    # 2. Usa as variáveis finais já definidas para a etapa.
    print(f"\nModelo final da etapa: {nome_modelo}")
    print(f"Conjunto final da etapa: {nome_conjunto}")
    print("\nVariáveis finais definidas no script:")
    for i, var in enumerate(variaveis_modelo, start=1):
        print(f"{i:02d}. {var}")

    variaveis_ausentes_em_x = [var for var in variaveis_modelo if var not in X.columns]
    if variaveis_ausentes_em_x:
        raise ValueError(
            "As seguintes variáveis finais não foram encontradas em X. "
            "Confira se os nomes estão exatamente iguais aos nomes da base:\n"
            f"{variaveis_ausentes_em_x}"
        )

    # 3. Divide treino e teste 80/20, como no script completo.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    # 4. Imputação por mediana aprendida apenas no treino.
    imputador = SimpleImputer(strategy="median")

    X_train_imp_array = imputador.fit_transform(X_train)
    X_test_imp_array = imputador.transform(X_test)

    X_train_imp = pd.DataFrame(
        X_train_imp_array,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_imp = pd.DataFrame(
        X_test_imp_array,
        columns=X_test.columns,
        index=X_test.index,
    )

    print("\nValores ausentes em X_train_imp:", X_train_imp.isna().sum().sum())
    print("Valores ausentes em X_test_imp:", X_test_imp.isna().sum().sum())

    # 5. Remoção de variância zero aprendida apenas no treino.
    seletor_variancia = VarianceThreshold(threshold=0)

    X_train_var_array = seletor_variancia.fit_transform(X_train_imp)
    X_test_var_array = seletor_variancia.transform(X_test_imp)

    variaveis_pos_variancia = X_train_imp.columns[
        seletor_variancia.get_support()
    ].tolist()

    X_train_var = pd.DataFrame(
        X_train_var_array,
        columns=variaveis_pos_variancia,
        index=X_train_imp.index,
    )

    X_test_var = pd.DataFrame(
        X_test_var_array,
        columns=variaveis_pos_variancia,
        index=X_test_imp.index,
    )

    print("Variáveis após remoção de variância zero:", len(variaveis_pos_variancia))

    variaveis_removidas_por_variancia = [
        var for var in variaveis_modelo
        if var not in variaveis_pos_variancia
    ]

    if variaveis_removidas_por_variancia:
        raise ValueError(
            "Alguma(s) variável(is) final(is) foi(ram) removida(s) por "
            "variância zero no treino. Isso indica inconsistência entre a base "
            "e as variáveis fixas do modelo:\n"
            f"{variaveis_removidas_por_variancia}"
        )

    X_train_final = X_train_var[variaveis_modelo].copy()
    X_test_final = X_test_var[variaveis_modelo].copy()

    # 6. Treina o modelo final correto da etapa.
    modelo_final = criar_modelo_final(nome_modelo)
    modelo_final.fit(X_train_final, y_train)

    y_pred_train = modelo_final.predict(X_train_final)
    y_pred_test = modelo_final.predict(X_test_final)

    mae_treino = mean_absolute_error(y_train, y_pred_train)
    rmse_treino = calcular_rmse(y_train, y_pred_train)
    r2_treino = r2_score(y_train, y_pred_train)

    mae_teste = mean_absolute_error(y_test, y_pred_test)
    rmse_teste = calcular_rmse(y_test, y_pred_test)
    r2_teste = r2_score(y_test, y_pred_test)

    metricas_modelo_final = pd.DataFrame({
        "base": ["Treino", "Teste"],
        "MAE": [mae_treino, mae_teste],
        "RMSE": [rmse_treino, rmse_teste],
        "R2": [r2_treino, r2_teste],
    })

    # 7. Validação cruzada simples 5-fold no treino, como resumo operacional.
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2": "r2",
    }

    resultados_cv = cross_validate(
        estimator=criar_modelo_final(nome_modelo),
        X=X_train_final,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
        error_score=np.nan,
    )

    mae_cv_folds = -resultados_cv["test_MAE"]
    rmse_cv_folds = np.sqrt(-resultados_cv["test_MSE"])
    r2_cv_folds = resultados_cv["test_R2"]

    mae_cv = float(np.nanmean(mae_cv_folds))
    rmse_cv = float(np.nanmean(rmse_cv_folds))
    r2_cv = float(np.nanmean(r2_cv_folds))

    mae_cv_desvio = float(np.nanstd(mae_cv_folds))
    rmse_cv_desvio = float(np.nanstd(rmse_cv_folds))
    r2_cv_desvio = float(np.nanstd(r2_cv_folds))

    modelo_final_escolhido = {
        "conjunto": nome_conjunto,
        "modelo": nome_modelo,
        "modelo_conjunto": f"{nome_modelo} | {nome_conjunto}",
        "n_variaveis": len(variaveis_modelo),
        "MAE_treino": mae_treino,
        "RMSE_treino": rmse_treino,
        "R2_treino": r2_treino,
        "MAE_cv": mae_cv,
        "MAE_cv_desvio": mae_cv_desvio,
        "RMSE_cv": rmse_cv,
        "RMSE_cv_desvio": rmse_cv_desvio,
        "R2_cv": r2_cv,
        "R2_cv_desvio": r2_cv_desvio,
        "MAE_teste": mae_teste,
        "RMSE_teste": rmse_teste,
        "R2_teste": r2_teste,
        "gap_R2_treino_cv": r2_treino - r2_cv,
        "gap_R2_treino_teste": r2_treino - r2_teste,
        "gap_RMSE_cv_teste": rmse_teste - rmse_cv,
        "RMSE_repeated_medio": np.nan,
        "RMSE_repeated_desvio": np.nan,
        "R2_repeated_medio": np.nan,
    }

    ranking_modelos_geral = pd.DataFrame([modelo_final_escolhido])

    resumo_repeated = pd.DataFrame({
        "modelo_conjunto": [f"{nome_modelo} | {nome_conjunto}"],
        "RMSE_repeated_medio": [np.nan],
        "RMSE_repeated_desvio": [np.nan],
        "MAE_repeated_medio": [np.nan],
        "R2_repeated_medio": [np.nan],
        "n_variaveis": [len(variaveis_modelo)],
    })

    resultado_friedman = pd.DataFrame({
        "teste": ["Friedman"],
        "base_comparacao": ["não reexecutado no script operacional"],
        "estatistica": [np.nan],
        "p_valor": [np.nan],
        "significativo_5%": [np.nan],
    })

    resultados_wilcoxon = pd.DataFrame(columns=[
        "modelo_a",
        "modelo_b",
        "estatistica",
        "p_valor",
        "p_valor_corrigido_holm",
        "diferenca_significativa",
    ])

    candidatos_final_proximos = pd.DataFrame([modelo_final_escolhido])

    print("\nMétricas finais:")
    print(metricas_modelo_final)

    print("\nResumo de validação cruzada operacional:")
    print(pd.DataFrame([{
        "MAE_cv": mae_cv,
        "MAE_cv_desvio": mae_cv_desvio,
        "RMSE_cv": rmse_cv,
        "RMSE_cv_desvio": rmse_cv_desvio,
        "R2_cv": r2_cv,
        "R2_cv_desvio": r2_cv_desvio,
    }]))

    # 8. Artefato completo compatível com o app Streamlit.
    artefato_modelo_final = {
        "modelo_final": modelo_final,
        "nome_modelo": nome_modelo,
        "nome_conjunto": nome_conjunto,
        "variaveis_modelo": variaveis_modelo,
        "imputador": imputador,
        "seletor_variancia": seletor_variancia,
        "variaveis_originais_X": X.columns.tolist(),
        "variaveis_pos_variancia": variaveis_pos_variancia,
        "metricas_modelo_final": metricas_modelo_final,
        "ranking_modelos_geral": ranking_modelos_geral,
        "modelo_final_escolhido": modelo_final_escolhido,
        "resumo_repeated": resumo_repeated,
        "resultado_friedman": resultado_friedman,
        "resultados_wilcoxon": resultados_wilcoxon,
        "candidatos_final_proximos": candidatos_final_proximos,
        "etapa_ensino": config_etapa["nome_etapa"],
        "coluna_ideb": coluna_ideb,
        "arquivo_base": arquivo_base,
    }

    joblib.dump(artefato_modelo_final, arquivo_saida)

    print("\nArtefato salvo com sucesso:")
    print(arquivo_saida)

    return {
        "etapa": config_etapa["nome_etapa"],
        "arquivo_base": arquivo_base,
        "arquivo_saida": arquivo_saida,
        "modelo": nome_modelo,
        "conjunto": nome_conjunto,
        "n_variaveis": len(variaveis_modelo),
        "MAE_teste": mae_teste,
        "RMSE_teste": rmse_teste,
        "R2_teste": r2_teste,
        "RMSE_cv": rmse_cv,
        "R2_cv": r2_cv,
    }


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    validar_configuracoes()

    resumos = []

    for chave_etapa, config_etapa in CONFIG_ETAPAS.items():
        resumo = treinar_e_salvar_artefato(chave_etapa, config_etapa)
        resumos.append(resumo)

    resumo_execucao = pd.DataFrame(resumos)
    caminho_resumo = os.path.join(
        PASTA_OUTPUTS,
        "resumo_artefatos_modelos_finais_por_etapa.xlsx",
    )
    resumo_execucao.to_excel(caminho_resumo, index=False)

    print("\n" + "=" * 90)
    print("PROCESSO CONCLUÍDO COM SUCESSO")
    print("Resumo da execução salvo em:")
    print(caminho_resumo)
    print("=" * 90)
