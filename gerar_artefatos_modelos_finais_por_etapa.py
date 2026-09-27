# -*- coding: utf-8 -*-
"""
ORGANIZAÇÃO DOS ARTEFATOS FINAIS POR ETAPA PARA O SIMULADOR
============================================================

Este script NÃO treina modelos e NÃO refaz a seleção de variáveis.

A fonte metodológica dos modelos é constituída pelos notebooks finais do artigo:
- Anos Iniciais: XGBoost, com estratégia SHAP Top-30;
- Anos Finais: CatBoost, com estratégia Frequencia_2_ou_mais.

Depois de encerrada a avaliação científica, os próprios notebooks reajustam
os modelos de implantação com todos os dados disponíveis de 2013 a 2023 e
exportam os componentes necessários ao simulador.

Função deste script:
1. localizar as pastas exportadas pelos dois notebooks;
2. validar os arquivos necessários;
3. conferir a coerência mínima dos metadados;
4. testar o pipeline de predição de implantação;
5. copiar os componentes operacionais para models/<etapa>/;
6. copiar a base de referência de 2023 para data/;
7. gerar um resumo auditável da integração em outputs/.

Uso esperado
------------
Coloque, temporariamente, na raiz do projeto:

simulador_ideb/
├── artefatos_anos_iniciais/
├── artefatos_anos_finais/
├── app.py
├── data/
├── models/
├── outputs/
└── gerar_artefatos_modelos_finais_por_etapa.py

Depois execute:

    python gerar_artefatos_modelos_finais_por_etapa.py

Ao final, o simulador utilizará:

data/
├── base_referencia_2023_anos_iniciais.csv
└── base_referencia_2023_anos_finais.csv

models/
├── anos_iniciais/
└── anos_finais/

IMPORTANTE
----------
As métricas do artigo pertencem ao modelo de avaliação científica.
O modelo usado pelo simulador é o modelo de implantação reajustado após o
encerramento da avaliação independente. Este script preserva essa separação.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

RAIZ_PROJETO = Path(__file__).resolve().parent
PASTA_DADOS = RAIZ_PROJETO / "data"
PASTA_MODELOS = RAIZ_PROJETO / "models"
PASTA_OUTPUTS = RAIZ_PROJETO / "outputs"

PASTA_DADOS.mkdir(parents=True, exist_ok=True)
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
PASTA_OUTPUTS.mkdir(parents=True, exist_ok=True)


CONFIG_ETAPAS = {
    "anos_iniciais": {
        "nome_etapa": "Anos Iniciais",
        "pasta_origem": RAIZ_PROJETO / "artefatos_anos_iniciais",
        "arquivo_modelo": "modelo_implantacao.ubj",
        "formato_modelo": "xgboost_nativo",
        "pasta_destino_modelo": PASTA_MODELOS / "anos_iniciais",
        "arquivo_destino_base": (
            PASTA_DADOS / "base_referencia_2023_anos_iniciais.csv"
        ),
        "modelo_esperado": "XGBoost",
        "n_variaveis_avaliacao_esperado": 30,
        "n_variaveis_implantacao_esperado": 30,
    },
    "anos_finais": {
        "nome_etapa": "Anos Finais",
        "pasta_origem": RAIZ_PROJETO / "artefatos_anos_finais",
        "arquivo_modelo": "modelo_implantacao.joblib",
        "formato_modelo": "joblib",
        "pasta_destino_modelo": PASTA_MODELOS / "anos_finais",
        "arquivo_destino_base": (
            PASTA_DADOS / "base_referencia_2023_anos_finais.csv"
        ),
        "modelo_esperado": "CatBoost",
        "n_variaveis_avaliacao_esperado": 21,
        "n_variaveis_implantacao_esperado": 23,
    },
}


# Arquivos comuns às duas etapas. O arquivo do modelo é definido em CONFIG_ETAPAS.
ARQUIVOS_COMUNS_SIMULADOR = [
    "imputador_implantacao.joblib",
    "seletor_variancia_implantacao.joblib",
    "variaveis_pos_variancia.joblib",
    "variaveis_modelo.joblib",
    "colunas_entrada_modelo.joblib",
    "metadados.json",
    "resumo_tecnico_modelo_final.csv",
    "suporte_empirico_variaveis.csv",
]

ARQUIVO_BASE_REFERENCIA = "base_referencia_2023.csv"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def calcular_sha256(caminho: Path) -> str:
    """Calcula SHA-256 de um arquivo para auditoria."""
    hash_obj = hashlib.sha256()

    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            hash_obj.update(bloco)

    return hash_obj.hexdigest()


def carregar_json(caminho: Path) -> dict[str, Any]:
    """Carrega arquivo JSON em UTF-8."""
    with caminho.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def obter_valor_metadado(
    metadados: dict[str, Any],
    *chaves: str,
) -> Any:
    """Retorna o primeiro valor existente entre chaves alternativas."""
    for chave in chaves:
        if chave in metadados:
            return metadados[chave]

    return None


def validar_arquivos_origem(
    chave_etapa: str,
    config: dict[str, Any],
) -> None:
    """Confere se a pasta exportada pelo notebook possui os arquivos exigidos."""
    pasta_origem = config["pasta_origem"]

    if not pasta_origem.exists():
        raise FileNotFoundError(
            f"Pasta de origem não encontrada para {config['nome_etapa']}: "
            f"{pasta_origem.resolve()}\n"
            "Baixe a pasta de artefatos gerada pelo respectivo notebook e "
            "coloque-a na raiz do projeto."
        )

    arquivos_necessarios = (
        [config["arquivo_modelo"]]
        + ARQUIVOS_COMUNS_SIMULADOR
        + [ARQUIVO_BASE_REFERENCIA]
    )

    ausentes = [
        arquivo
        for arquivo in arquivos_necessarios
        if not (pasta_origem / arquivo).exists()
    ]

    if ausentes:
        raise FileNotFoundError(
            f"Arquivos ausentes em {pasta_origem} para {chave_etapa}:\n"
            + "\n".join(f"- {arquivo}" for arquivo in ausentes)
        )


def validar_metadados(
    config: dict[str, Any],
    metadados: dict[str, Any],
    variaveis_modelo: list[str],
) -> dict[str, Any]:
    """Valida modelo e quantidades registradas no pipeline final."""
    modelo = obter_valor_metadado(
        metadados,
        "modelo",
        "nome_modelo",
    )

    n_avaliacao = obter_valor_metadado(
        metadados,
        "n_variaveis_modelo_avaliacao",
    )

    n_implantacao = obter_valor_metadado(
        metadados,
        "n_variaveis_modelo_implantacao",
    )

    if modelo != config["modelo_esperado"]:
        raise ValueError(
            f"Modelo incompatível em {config['nome_etapa']}. "
            f"Esperado: {config['modelo_esperado']}. Encontrado: {modelo}."
        )

    if (
        n_avaliacao is not None
        and int(n_avaliacao)
        != int(config["n_variaveis_avaliacao_esperado"])
    ):
        raise ValueError(
            f"Número de variáveis do modelo de avaliação incompatível em "
            f"{config['nome_etapa']}. Esperado: "
            f"{config['n_variaveis_avaliacao_esperado']}. "
            f"Encontrado: {n_avaliacao}."
        )

    if (
        n_implantacao is not None
        and int(n_implantacao)
        != int(config["n_variaveis_implantacao_esperado"])
    ):
        raise ValueError(
            f"Número de variáveis do modelo de implantação incompatível em "
            f"{config['nome_etapa']}. Esperado: "
            f"{config['n_variaveis_implantacao_esperado']}. "
            f"Encontrado: {n_implantacao}."
        )

    if len(variaveis_modelo) != int(
        config["n_variaveis_implantacao_esperado"]
    ):
        raise ValueError(
            f"O arquivo variaveis_modelo.joblib contém "
            f"{len(variaveis_modelo)} variáveis em {config['nome_etapa']}, "
            f"mas o esperado para implantação é "
            f"{config['n_variaveis_implantacao_esperado']}."
        )

    return {
        "modelo": modelo,
        "n_variaveis_avaliacao": n_avaliacao,
        "n_variaveis_implantacao": len(variaveis_modelo),
        "estrategia": obter_valor_metadado(
            metadados,
            "estrategia_selecao_descricao",
            "estrategia_selecao",
            "estrategia_selecao_codigo",
        ),
    }


def validar_base_referencia(
    base: pd.DataFrame,
    colunas_entrada: list[str],
) -> None:
    """Valida a base de referência de 2023."""
    if base.empty:
        raise ValueError(
            "A base_referencia_2023.csv está vazia."
        )

    if "ano" in base.columns:
        anos = (
            pd.to_numeric(
                base["ano"],
                errors="coerce",
            )
            .dropna()
            .unique()
            .tolist()
        )

        if anos and set(anos) != {2023}:
            raise ValueError(
                "A base de referência contém anos diferentes de 2023: "
                f"{sorted(anos)}"
            )

    coluna_ideb = next(
        (
            coluna
            for coluna in ["ideb", "IDEB", "Ideb"]
            if coluna in base.columns
        ),
        None,
    )

    if coluna_ideb is None:
        raise ValueError(
            "A base de referência não contém a coluna do IDEB."
        )

    if "ideb_predito_referencia" not in base.columns:
        raise ValueError(
            "A base de referência não contém a coluna "
            "'ideb_predito_referencia', necessária para validar a "
            "reprodução das predições exportadas pelo notebook."
        )

    colunas_ausentes = [
        coluna
        for coluna in colunas_entrada
        if coluna not in base.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            "A base de referência não contém todas as colunas de entrada "
            "do modelo:\n"
            + "\n".join(f"- {coluna}" for coluna in colunas_ausentes)
        )


def carregar_modelo_implantacao(
    pasta_modelo: Path,
    config: dict[str, Any],
) -> Any:
    """Carrega o modelo de implantação no formato definido para cada etapa."""
    caminho_modelo = pasta_modelo / config["arquivo_modelo"]

    if config["formato_modelo"] == "xgboost_nativo":
        modelo = xgb.XGBRegressor()
        modelo.load_model(caminho_modelo)
        return modelo

    if config["formato_modelo"] == "joblib":
        return joblib.load(caminho_modelo)

    raise ValueError(
        f"Formato de modelo não reconhecido em {config['nome_etapa']}: "
        f"{config['formato_modelo']}"
    )


def preparar_entrada_implantacao(
    dados: pd.DataFrame,
    imputador: Any,
    seletor_variancia: Any,
    colunas_entrada: list[str],
    variaveis_pos_variancia: list[str],
    variaveis_modelo: list[str],
) -> pd.DataFrame:
    """
    Reproduz o pré-processamento já definido no notebook final.

    Não há qualquer novo ajuste de imputador ou seletor nesta função.
    """
    X_novo = (
        dados[colunas_entrada]
        .copy()
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    X_imp = pd.DataFrame(
        imputador.transform(
            X_novo
        ),
        columns=colunas_entrada,
        index=X_novo.index,
    )

    X_var = pd.DataFrame(
        seletor_variancia.transform(
            X_imp
        ),
        columns=variaveis_pos_variancia,
        index=X_novo.index,
    )

    ausentes_pos_processamento = [
        variavel
        for variavel in variaveis_modelo
        if variavel not in X_var.columns
    ]

    if ausentes_pos_processamento:
        raise ValueError(
            "Variáveis do modelo ausentes após o pré-processamento:\n"
            + "\n".join(
                f"- {variavel}"
                for variavel in ausentes_pos_processamento
            )
        )

    return X_var[
        variaveis_modelo
    ].copy()


def testar_pipeline_implantacao(
    pasta_origem: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Reproduz o pipeline de implantação e confere as predições contra a
    referência exportada pelo notebook final.
    """
    modelo = carregar_modelo_implantacao(
        pasta_modelo=pasta_origem,
        config=config,
    )

    imputador = joblib.load(
        pasta_origem / "imputador_implantacao.joblib"
    )

    seletor_variancia = joblib.load(
        pasta_origem / "seletor_variancia_implantacao.joblib"
    )

    variaveis_pos_variancia = list(
        joblib.load(
            pasta_origem / "variaveis_pos_variancia.joblib"
        )
    )

    variaveis_modelo = list(
        joblib.load(
            pasta_origem / "variaveis_modelo.joblib"
        )
    )

    colunas_entrada = list(
        joblib.load(
            pasta_origem / "colunas_entrada_modelo.joblib"
        )
    )

    base = pd.read_csv(
        pasta_origem / ARQUIVO_BASE_REFERENCIA
    )

    validar_base_referencia(
        base=base,
        colunas_entrada=colunas_entrada,
    )

    X_final = preparar_entrada_implantacao(
        dados=base,
        imputador=imputador,
        seletor_variancia=seletor_variancia,
        colunas_entrada=colunas_entrada,
        variaveis_pos_variancia=variaveis_pos_variancia,
        variaveis_modelo=variaveis_modelo,
    )

    predicoes = np.asarray(
        modelo.predict(X_final)
    ).ravel()

    if predicoes.shape[0] != len(base):
        raise RuntimeError(
            "O número de predições não corresponde ao número de registros "
            "da base de referência."
        )

    if not np.all(np.isfinite(predicoes)):
        raise RuntimeError(
            "O pipeline de implantação produziu predições não finitas."
        )

    referencia = pd.to_numeric(
        base["ideb_predito_referencia"],
        errors="coerce",
    ).to_numpy()

    if not np.all(np.isfinite(referencia)):
        raise ValueError(
            "A coluna 'ideb_predito_referencia' contém valores ausentes "
            "ou não finitos."
        )

    diferencas = np.abs(
        predicoes - referencia
    )

    coincidem = np.allclose(
        predicoes,
        referencia,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )

    if not coincidem:
        raise RuntimeError(
            f"As predições recalculadas para {config['nome_etapa']} não "
            "reproduzem 'ideb_predito_referencia'. "
            f"Diferença absoluta máxima: {float(np.max(diferencas)):.12g}."
        )

    suporte = pd.read_csv(
        pasta_origem / "suporte_empirico_variaveis.csv"
    )

    if "variavel" not in suporte.columns:
        raise ValueError(
            "suporte_empirico_variaveis.csv não contém a coluna 'variavel'."
        )

    variaveis_sem_suporte = [
        variavel
        for variavel in variaveis_modelo
        if variavel not in set(
            suporte["variavel"].astype(str)
        )
    ]

    if variaveis_sem_suporte:
        raise ValueError(
            "Há variáveis do modelo sem registro no suporte empírico:\n"
            + "\n".join(
                f"- {variavel}"
                for variavel in variaveis_sem_suporte
            )
        )

    return {
        "n_registros_validacao": int(len(base)),
        "n_colunas_entrada": len(colunas_entrada),
        "n_variaveis_pos_variancia": len(variaveis_pos_variancia),
        "n_variaveis_modelo": len(variaveis_modelo),
        "predicao_min": float(np.min(predicoes)),
        "predicao_max": float(np.max(predicoes)),
        "diferenca_referencia_max": float(np.max(diferencas)),
        "diferenca_referencia_media": float(np.mean(diferencas)),
        "predicoes_referencia_coincidem": True,
    }



def copiar_arquivo(
    origem: Path,
    destino: Path,
) -> None:
    """Copia arquivo preservando metadados básicos."""
    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        origem,
        destino,
    )


def integrar_etapa(
    chave_etapa: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Valida e instala os artefatos de uma etapa."""
    print(
        "\n"
        + "=" * 78
    )
    print(
        f"INTEGRANDO: {config['nome_etapa'].upper()}"
    )
    print(
        "=" * 78
    )

    validar_arquivos_origem(
        chave_etapa=chave_etapa,
        config=config,
    )

    pasta_origem = config[
        "pasta_origem"
    ]

    metadados = carregar_json(
        pasta_origem
        / "metadados.json"
    )

    variaveis_modelo = list(
        joblib.load(
            pasta_origem
            / "variaveis_modelo.joblib"
        )
    )

    info_metadados = validar_metadados(
        config=config,
        metadados=metadados,
        variaveis_modelo=variaveis_modelo,
    )

    resultado_teste = testar_pipeline_implantacao(
        pasta_origem=pasta_origem,
        config=config,
    )

    pasta_destino_modelo = config[
        "pasta_destino_modelo"
    ]

    pasta_destino_modelo.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifesto = []

    arquivos_modelo_etapa = (
        [config["arquivo_modelo"]]
        + ARQUIVOS_COMUNS_SIMULADOR
    )

    for nome_arquivo in arquivos_modelo_etapa:
        origem = (
            pasta_origem
            / nome_arquivo
        )

        destino = (
            pasta_destino_modelo
            / nome_arquivo
        )

        copiar_arquivo(
            origem=origem,
            destino=destino,
        )

        manifesto.append({
            "etapa": chave_etapa,
            "tipo": "modelo",
            "arquivo": nome_arquivo,
            "destino": str(
                destino
            ),
            "sha256": calcular_sha256(
                destino
            ),
        })

    origem_base = (
        pasta_origem
        / ARQUIVO_BASE_REFERENCIA
    )

    destino_base = config[
        "arquivo_destino_base"
    ]

    copiar_arquivo(
        origem=origem_base,
        destino=destino_base,
    )

    manifesto.append({
        "etapa": chave_etapa,
        "tipo": "base_referencia",
        "arquivo": destino_base.name,
        "destino": str(
            destino_base
        ),
        "sha256": calcular_sha256(
            destino_base
        ),
    })

    resumo = {
        "etapa": config[
            "nome_etapa"
        ],
        "modelo": info_metadados[
            "modelo"
        ],
        "estrategia": info_metadados[
            "estrategia"
        ],
        "n_variaveis_avaliacao": info_metadados[
            "n_variaveis_avaliacao"
        ],
        "n_variaveis_implantacao": resultado_teste[
            "n_variaveis_modelo"
        ],
        "n_colunas_entrada": resultado_teste[
            "n_colunas_entrada"
        ],
        "arquivo_modelo_implantacao": config["arquivo_modelo"],
        "formato_modelo_implantacao": config["formato_modelo"],
        "n_registros_validacao_referencia": resultado_teste[
            "n_registros_validacao"
        ],
        "predicao_min": resultado_teste["predicao_min"],
        "predicao_max": resultado_teste["predicao_max"],
        "diferenca_referencia_max": resultado_teste[
            "diferenca_referencia_max"
        ],
        "diferenca_referencia_media": resultado_teste[
            "diferenca_referencia_media"
        ],
        "predicoes_referencia_coincidem": resultado_teste[
            "predicoes_referencia_coincidem"
        ],
        "teste_pipeline": "aprovado",
        "pasta_modelo": str(
            pasta_destino_modelo
        ),
        "base_referencia": str(
            destino_base
        ),
    }

    print(
        "Modelo:",
        resumo[
            "modelo"
        ],
    )
    print(
        "Estratégia:",
        resumo[
            "estrategia"
        ],
    )
    print(
        "Variáveis de avaliação:",
        resumo[
            "n_variaveis_avaliacao"
        ],
    )
    print(
        "Variáveis de implantação:",
        resumo[
            "n_variaveis_implantacao"
        ],
    )
    print(
        "Validação contra ideb_predito_referencia: APROVADA"
    )
    print(
        "Diferença absoluta máxima:",
        resultado_teste["diferenca_referencia_max"],
    )
    print(
        "Destino dos modelos:",
        pasta_destino_modelo,
    )
    print(
        "Base de referência:",
        destino_base,
    )

    return (
        resumo,
        manifesto,
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    resumos = []
    manifesto_completo = []

    for (
        chave_etapa,
        config,
    ) in CONFIG_ETAPAS.items():

        (
            resumo,
            manifesto,
        ) = integrar_etapa(
            chave_etapa=chave_etapa,
            config=config,
        )

        resumos.append(
            resumo
        )

        manifesto_completo.extend(
            manifesto
        )

    tabela_resumo = pd.DataFrame(
        resumos
    )

    tabela_manifesto = pd.DataFrame(
        manifesto_completo
    )

    arquivo_resumo = (
        PASTA_OUTPUTS
        / "resumo_integracao_simulador.csv"
    )

    arquivo_manifesto = (
        PASTA_OUTPUTS
        / "manifesto_artefatos_simulador.csv"
    )

    tabela_resumo.to_csv(
        arquivo_resumo,
        index=False,
        encoding="utf-8-sig",
    )

    tabela_manifesto.to_csv(
        arquivo_manifesto,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "INTEGRAÇÃO CONCLUÍDA COM SUCESSO"
    )
    print(
        "=" * 78
    )
    print(
        "\nResumo:"
    )
    print(
        tabela_resumo.to_string(
            index=False
        )
    )
    print(
        "\nArquivos de auditoria:"
    )
    print(
        "-",
        arquivo_resumo,
    )
    print(
        "-",
        arquivo_manifesto,
    )
