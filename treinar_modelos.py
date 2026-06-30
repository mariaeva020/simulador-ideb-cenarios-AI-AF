# ============================================================
# VALIDAÇÃO DOS ARTEFATOS DO SIMULADOR DE CENÁRIOS DO IDEB
# ============================================================
#
# Observação importante:
# Este arquivo substitui a versão anterior que treinava modelos XGBoost.
# Como o artigo passou a adotar o modelo final selecionado no notebook
# metodológico, o simulador deve carregar o artefato final exportado pelo
# notebook, e não treinar um modelo diferente apenas para a aplicação.
#
# Objetivo deste script:
# 1. Verificar se os artefatos finais existem na pasta "models".
# 2. Conferir se cada artefato contém as chaves necessárias para o app.py.
# 3. Gerar um resumo técnico dos modelos disponíveis para o simulador.
#
# Artefatos esperados:
# - models/artefato_modelo_final_anos_iniciais.pkl
# - models/artefato_modelo_final_anos_finais.pkl
#
# Para Anos Iniciais, também é aceito o nome genérico:
# - models/artefato_modelo_final_ideb.pkl
# ============================================================

import os
import joblib
import pandas as pd


# ============================================================
# CAMINHOS
# ============================================================

PASTA_MODELOS = "models"
PASTA_SAIDA = "outputs"

os.makedirs(PASTA_MODELOS, exist_ok=True)
os.makedirs(PASTA_SAIDA, exist_ok=True)

CONFIG_ARTEFATOS = {
    "Anos Iniciais": [
        os.path.join(PASTA_MODELOS, "artefato_modelo_final_anos_iniciais.pkl"),
        os.path.join(PASTA_MODELOS, "artefato_modelo_final_ideb.pkl")
    ],
    "Anos Finais": [
        os.path.join(PASTA_MODELOS, "artefato_modelo_final_anos_finais.pkl")
    ]
}

CHAVES_OBRIGATORIAS = [
    "modelo_final",
    "nome_modelo",
    "nome_conjunto",
    "variaveis_modelo",
    "imputador",
    "seletor_variancia",
    "variaveis_originais_X",
    "variaveis_pos_variancia"
]

CHAVES_RECOMENDADAS = [
    "metricas_modelo_final",
    "ranking_modelos_geral",
    "modelo_final_escolhido",
    "resumo_repeated",
    "resultado_friedman",
    "resultados_wilcoxon",
    "candidatos_final_proximos"
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def localizar_artefato(caminhos_possiveis: list[str]) -> str | None:
    """Retorna o primeiro artefato existente entre os caminhos possíveis."""
    for caminho in caminhos_possiveis:
        if os.path.exists(caminho):
            return caminho
    return None


def carregar_e_validar_artefato(etapa: str, caminho: str) -> dict:
    """Carrega um artefato e valida se ele contém as chaves necessárias."""
    artefato = joblib.load(caminho)

    chaves_ausentes = [chave for chave in CHAVES_OBRIGATORIAS if chave not in artefato]
    chaves_recomendadas_ausentes = [chave for chave in CHAVES_RECOMENDADAS if chave not in artefato]

    if chaves_ausentes:
        raise ValueError(
            f"O artefato da etapa '{etapa}' está incompleto. "
            "Chaves obrigatórias ausentes: " + ", ".join(chaves_ausentes)
        )

    resumo = {
        "etapa": etapa,
        "caminho_artefato": caminho,
        "modelo": artefato.get("nome_modelo"),
        "conjunto_variaveis": artefato.get("nome_conjunto"),
        "n_variaveis_modelo": len(artefato.get("variaveis_modelo", [])),
        "n_variaveis_originais": len(artefato.get("variaveis_originais_X", [])),
        "n_variaveis_pos_variancia": len(artefato.get("variaveis_pos_variancia", [])),
        "chaves_obrigatorias_ok": True,
        "chaves_recomendadas_ausentes": "; ".join(chaves_recomendadas_ausentes)
    }

    modelo_final_escolhido = artefato.get("modelo_final_escolhido", None)

    if isinstance(modelo_final_escolhido, pd.Series):
        modelo_final_escolhido = modelo_final_escolhido.to_dict()

    if isinstance(modelo_final_escolhido, dict):
        resumo["RMSE_cv"] = modelo_final_escolhido.get("RMSE_cv")
        resumo["R2_cv"] = modelo_final_escolhido.get("R2_cv")
        resumo["RMSE_teste"] = modelo_final_escolhido.get("RMSE_teste")
        resumo["R2_teste"] = modelo_final_escolhido.get("R2_teste")
        resumo["RMSE_repeated_medio"] = modelo_final_escolhido.get("RMSE_repeated_medio")
        resumo["RMSE_repeated_desvio"] = modelo_final_escolhido.get("RMSE_repeated_desvio")
        resumo["R2_repeated_medio"] = modelo_final_escolhido.get("R2_repeated_medio")

    return resumo


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    print("=" * 80)
    print("VALIDAÇÃO DOS ARTEFATOS DO SIMULADOR DE CENÁRIOS DO IDEB")
    print("=" * 80)

    resumos = []

    for etapa, caminhos in CONFIG_ARTEFATOS.items():
        print(f"\nEtapa: {etapa}")

        caminho_artefato = localizar_artefato(caminhos)

        if caminho_artefato is None:
            print("Artefato não encontrado para esta etapa.")
            print("Caminhos esperados:")
            for caminho in caminhos:
                print(" -", caminho)
            continue

        print("Artefato encontrado:", caminho_artefato)

        resumo = carregar_e_validar_artefato(etapa, caminho_artefato)
        resumos.append(resumo)

        print("Modelo:", resumo["modelo"])
        print("Conjunto de variáveis:", resumo["conjunto_variaveis"])
        print("Número de variáveis do modelo:", resumo["n_variaveis_modelo"])
        print("Chaves obrigatórias: OK")

        if resumo["chaves_recomendadas_ausentes"]:
            print("Atenção: algumas chaves recomendadas não foram encontradas:")
            print(resumo["chaves_recomendadas_ausentes"])

    if not resumos:
        raise FileNotFoundError(
            "Nenhum artefato válido foi encontrado. "
            "Exporte o artefato final do notebook metodológico para a pasta 'models'."
        )

    tabela_resumo = pd.DataFrame(resumos)
    caminho_saida = os.path.join(PASTA_SAIDA, "resumo_artefatos_simulador.xlsx")
    tabela_resumo.to_excel(caminho_saida, index=False)

    print("\nResumo salvo em:", caminho_saida)
    print("Validação concluída com sucesso.")


if __name__ == "__main__":
    main()
