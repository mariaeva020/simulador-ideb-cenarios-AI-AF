# -*- coding: utf-8 -*-
"""
VALIDAÇÃO DOS ARTEFATOS INSTALADOS DO SIMULADOR DE CENÁRIOS DO IDEB
===================================================================

Apesar do nome histórico deste arquivo ("treinar_modelos.py"), este script
NÃO treina modelos.

A fonte metodológica de verdade são os notebooks finais do artigo. Neles:
- o modelo de avaliação é submetido ao protocolo científico;
- após o encerramento da avaliação independente, o modelo de implantação
  é reajustado com todos os dados disponíveis de 2013 a 2023;
- os artefatos de implantação são exportados para uso no simulador.

Este script executa uma verificação independente da instalação local:
1. confere se as bases de referência de 2023 estão presentes;
2. confere se os artefatos de implantação estão presentes;
3. valida os metadados;
4. reproduz o pré-processamento de implantação;
5. executa predições de teste;
6. verifica se as predições são finitas;
7. gera um resumo auditável em outputs/.

Estrutura esperada
------------------
data/
├── base_referencia_2023_anos_iniciais.csv
└── base_referencia_2023_anos_finais.csv

models/
├── anos_iniciais/
│   ├── modelo_implantacao.joblib
│   ├── imputador_implantacao.joblib
│   ├── seletor_variancia_implantacao.joblib
│   ├── variaveis_pos_variancia.joblib
│   ├── variaveis_modelo.joblib
│   ├── colunas_entrada_modelo.joblib
│   ├── metadados.json
│   ├── resumo_tecnico_modelo_final.csv
│   └── suporte_empirico_variaveis.csv
└── anos_finais/
    └── mesmos arquivos
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS = Path("data")
PASTA_MODELOS = Path("models")
PASTA_OUTPUTS = Path("outputs")

PASTA_OUTPUTS.mkdir(
    parents=True,
    exist_ok=True,
)

CONFIG_ETAPAS = {
    "anos_iniciais": {
        "nome_etapa": "Anos Iniciais",
        "base": PASTA_DADOS / "base_referencia_2023_anos_iniciais.csv",
        "pasta_modelo": PASTA_MODELOS / "anos_iniciais",
        "modelo_esperado": "XGBoost",
        "n_variaveis_avaliacao_esperado": 30,
        "n_variaveis_implantacao_esperado": 30,
    },
    "anos_finais": {
        "nome_etapa": "Anos Finais",
        "base": PASTA_DADOS / "base_referencia_2023_anos_finais.csv",
        "pasta_modelo": PASTA_MODELOS / "anos_finais",
        "modelo_esperado": "CatBoost",
        "n_variaveis_avaliacao_esperado": 21,
        "n_variaveis_implantacao_esperado": 23,
    },
}

ARQUIVOS_OBRIGATORIOS = [
    "modelo_implantacao.joblib",
    "imputador_implantacao.joblib",
    "seletor_variancia_implantacao.joblib",
    "variaveis_pos_variancia.joblib",
    "variaveis_modelo.joblib",
    "colunas_entrada_modelo.joblib",
    "metadados.json",
    "resumo_tecnico_modelo_final.csv",
    "suporte_empirico_variaveis.csv",
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def carregar_json(caminho: Path) -> dict[str, Any]:
    """Carrega um JSON em UTF-8."""
    with caminho.open(
        "r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(
            arquivo
        )


def primeiro_valor(
    dicionario: dict[str, Any],
    *chaves: str,
) -> Any:
    """Retorna o primeiro valor existente entre chaves alternativas."""
    for chave in chaves:
        if chave in dicionario:
            return dicionario[chave]

    return None


def validar_estrutura(
    config: dict[str, Any],
) -> None:
    """Verifica a presença da base e dos arquivos de modelo."""
    ausentes = []

    if not config["base"].exists():
        ausentes.append(
            str(
                config["base"]
            )
        )

    for arquivo in ARQUIVOS_OBRIGATORIOS:
        caminho = (
            config["pasta_modelo"]
            / arquivo
        )

        if not caminho.exists():
            ausentes.append(
                str(
                    caminho
                )
            )

    if ausentes:
        raise FileNotFoundError(
            f"Arquivos ausentes para {config['nome_etapa']}:\n"
            + "\n".join(
                f"- {arquivo}"
                for arquivo in ausentes
            )
        )


def validar_metadados(
    config: dict[str, Any],
    metadados: dict[str, Any],
    variaveis_modelo: list[str],
) -> dict[str, Any]:
    """Confere a coerência mínima dos metadados."""
    modelo = primeiro_valor(
        metadados,
        "modelo",
        "nome_modelo",
    )

    estrategia = primeiro_valor(
        metadados,
        "estrategia_selecao_descricao",
        "estrategia_selecao",
        "estrategia_selecao_codigo",
    )

    n_avaliacao = primeiro_valor(
        metadados,
        "n_variaveis_modelo_avaliacao",
    )

    n_implantacao_metadados = primeiro_valor(
        metadados,
        "n_variaveis_modelo_implantacao",
    )

    if modelo != config["modelo_esperado"]:
        raise ValueError(
            f"Modelo incompatível em {config['nome_etapa']}. "
            f"Esperado: {config['modelo_esperado']}. "
            f"Encontrado: {modelo}."
        )

    if (
        n_avaliacao is not None
        and int(
            n_avaliacao
        )
        != int(
            config[
                "n_variaveis_avaliacao_esperado"
            ]
        )
    ):
        raise ValueError(
            f"Número de variáveis da avaliação incompatível em "
            f"{config['nome_etapa']}. Esperado: "
            f"{config['n_variaveis_avaliacao_esperado']}. "
            f"Encontrado: {n_avaliacao}."
        )

    if len(
        variaveis_modelo
    ) != int(
        config[
            "n_variaveis_implantacao_esperado"
        ]
    ):
        raise ValueError(
            f"Número de variáveis do modelo de implantação incompatível "
            f"em {config['nome_etapa']}. Esperado: "
            f"{config['n_variaveis_implantacao_esperado']}. "
            f"Encontrado: {len(variaveis_modelo)}."
        )

    if (
        n_implantacao_metadados
        is not None
        and int(
            n_implantacao_metadados
        )
        != len(
            variaveis_modelo
        )
    ):
        raise ValueError(
            f"Os metadados informam {n_implantacao_metadados} variáveis "
            f"de implantação, mas variaveis_modelo.joblib contém "
            f"{len(variaveis_modelo)}."
        )

    return {
        "modelo": modelo,
        "estrategia": estrategia,
        "n_variaveis_avaliacao": n_avaliacao,
        "n_variaveis_implantacao": len(
            variaveis_modelo
        ),
    }


def validar_base(
    base: pd.DataFrame,
    colunas_entrada: list[str],
) -> dict[str, Any]:
    """Verifica a base de referência de 2023."""
    if base.empty:
        raise ValueError(
            "A base de referência está vazia."
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

        if anos and set(
            anos
        ) != {2023}:
            raise ValueError(
                "Foram encontrados anos diferentes de 2023 na base de referência: "
                + ", ".join(
                    str(
                        int(
                            ano
                        )
                    )
                    for ano in sorted(
                        anos
                    )
                )
            )

    coluna_ideb = next(
        (
            coluna
            for coluna in [
                "ideb",
                "IDEB",
                "Ideb",
            ]
            if coluna
            in base.columns
        ),
        None,
    )

    if coluna_ideb is None:
        raise ValueError(
            "A base de referência não possui coluna do IDEB."
        )

    colunas_ausentes = [
        coluna
        for coluna in colunas_entrada
        if coluna
        not in base.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            "A base de referência não contém todas as colunas esperadas "
            "pelo pipeline de implantação:\n"
            + "\n".join(
                f"- {coluna}"
                for coluna
                in colunas_ausentes
            )
        )

    return {
        "n_registros_base": int(
            len(
                base
            )
        ),
        "coluna_ideb": coluna_ideb,
        "n_ideb_ausente": int(
            base[
                coluna_ideb
            ].isna().sum()
        ),
    }


def preparar_entrada(
    dados: pd.DataFrame,
    imputador: Any,
    seletor_variancia: Any,
    colunas_entrada: list[str],
    variaveis_pos_variancia: list[str],
    variaveis_modelo: list[str],
) -> pd.DataFrame:
    """Reproduz o pré-processamento do modelo de implantação."""
    X = (
        dados[
            colunas_entrada
        ]
        .copy()
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    X_imp = pd.DataFrame(
        imputador.transform(
            X
        ),
        columns=colunas_entrada,
        index=X.index,
    )

    X_var = pd.DataFrame(
        seletor_variancia.transform(
            X_imp
        ),
        columns=variaveis_pos_variancia,
        index=X.index,
    )

    variaveis_ausentes = [
        variavel
        for variavel
        in variaveis_modelo
        if variavel
        not in X_var.columns
    ]

    if variaveis_ausentes:
        raise ValueError(
            "Variáveis finais ausentes após o pré-processamento:\n"
            + "\n".join(
                f"- {variavel}"
                for variavel
                in variaveis_ausentes
            )
        )

    return X_var[
        variaveis_modelo
    ].copy()


def validar_suporte_empirico(
    suporte: pd.DataFrame,
    variaveis_modelo: list[str],
) -> None:
    """Confere se todas as variáveis do modelo possuem suporte empírico registrado."""
    colunas_necessarias = {
        "variavel",
        "min",
        "p01",
        "p99",
        "max",
    }

    ausentes = (
        colunas_necessarias
        - set(
            suporte.columns
        )
    )

    if ausentes:
        raise ValueError(
            "suporte_empirico_variaveis.csv não contém as colunas: "
            + ", ".join(
                sorted(
                    ausentes
                )
            )
        )

    conjunto_suporte = set(
        suporte[
            "variavel"
        ].astype(
            str
        )
    )

    sem_suporte = [
        variavel
        for variavel
        in variaveis_modelo
        if variavel
        not in conjunto_suporte
    ]

    if sem_suporte:
        raise ValueError(
            "Há variáveis do modelo sem suporte empírico registrado:\n"
            + "\n".join(
                f"- {variavel}"
                for variavel
                in sem_suporte
            )
        )


def extrair_metricas_resumo(
    resumo: pd.DataFrame,
) -> dict[str, Any]:
    """Extrai métricas principais do resumo técnico, quando disponíveis."""
    if not {
        "item",
        "valor",
    }.issubset(
        resumo.columns
    ):
        return {}

    dicionario = {
        str(
            item
        ).strip(): valor
        for (
            item,
            valor,
        ) in zip(
            resumo[
                "item"
            ],
            resumo[
                "valor"
            ],
        )
    }

    def numero(
        chave: str,
    ) -> float | None:
        valor = dicionario.get(
            chave
        )

        if valor is None:
            return None

        try:
            return float(
                str(
                    valor
                )
                .replace(
                    ",",
                    ".",
                )
            )
        except ValueError:
            return None

    return {
        "MAE_teste_territorial": numero(
            "MAE - teste territorial"
        ),
        "RMSE_teste_territorial": numero(
            "RMSE - teste territorial"
        ),
        "R2_teste_territorial": numero(
            "R² - teste territorial"
        ),
        "MAE_validacao_temporal_2023": numero(
            "MAE - validação temporal 2023"
        ),
        "RMSE_validacao_temporal_2023": numero(
            "RMSE - validação temporal 2023"
        ),
        "R2_validacao_temporal_2023": numero(
            "R² - validação temporal 2023"
        ),
    }


def validar_etapa(
    chave_etapa: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Executa todas as verificações de uma etapa."""
    print(
        "\n"
        + "=" * 80
    )
    print(
        f"VALIDANDO: {config['nome_etapa'].upper()}"
    )
    print(
        "=" * 80
    )

    validar_estrutura(
        config
    )

    pasta_modelo = config[
        "pasta_modelo"
    ]

    modelo = joblib.load(
        pasta_modelo
        / "modelo_implantacao.joblib"
    )

    imputador = joblib.load(
        pasta_modelo
        / "imputador_implantacao.joblib"
    )

    seletor_variancia = joblib.load(
        pasta_modelo
        / "seletor_variancia_implantacao.joblib"
    )

    variaveis_pos_variancia = list(
        joblib.load(
            pasta_modelo
            / "variaveis_pos_variancia.joblib"
        )
    )

    variaveis_modelo = list(
        joblib.load(
            pasta_modelo
            / "variaveis_modelo.joblib"
        )
    )

    colunas_entrada = list(
        joblib.load(
            pasta_modelo
            / "colunas_entrada_modelo.joblib"
        )
    )

    metadados = carregar_json(
        pasta_modelo
        / "metadados.json"
    )

    resumo_tecnico = pd.read_csv(
        pasta_modelo
        / "resumo_tecnico_modelo_final.csv"
    )

    suporte = pd.read_csv(
        pasta_modelo
        / "suporte_empirico_variaveis.csv"
    )

    base = pd.read_csv(
        config[
            "base"
        ]
    )

    info_metadados = validar_metadados(
        config=config,
        metadados=metadados,
        variaveis_modelo=variaveis_modelo,
    )

    info_base = validar_base(
        base=base,
        colunas_entrada=colunas_entrada,
    )

    validar_suporte_empirico(
        suporte=suporte,
        variaveis_modelo=variaveis_modelo,
    )

    n_teste = min(
        20,
        len(
            base
        ),
    )

    dados_teste = base.iloc[
        :n_teste
    ].copy()

    X_final = preparar_entrada(
        dados=dados_teste,
        imputador=imputador,
        seletor_variancia=seletor_variancia,
        colunas_entrada=colunas_entrada,
        variaveis_pos_variancia=variaveis_pos_variancia,
        variaveis_modelo=variaveis_modelo,
    )

    predicoes = np.asarray(
        modelo.predict(
            X_final
        )
    ).ravel()

    if len(
        predicoes
    ) != n_teste:
        raise RuntimeError(
            "Quantidade de predições diferente da quantidade de registros testados."
        )

    if not np.all(
        np.isfinite(
            predicoes
        )
    ):
        raise RuntimeError(
            "Foram produzidas predições não finitas."
        )

    metricas = extrair_metricas_resumo(
        resumo_tecnico
    )

    resultado = {
        "etapa": config[
            "nome_etapa"
        ],
        "status": "aprovado",
        "modelo": info_metadados[
            "modelo"
        ],
        "estrategia": info_metadados[
            "estrategia"
        ],
        "n_variaveis_avaliacao": info_metadados[
            "n_variaveis_avaliacao"
        ],
        "n_variaveis_implantacao": info_metadados[
            "n_variaveis_implantacao"
        ],
        "n_colunas_entrada": len(
            colunas_entrada
        ),
        "n_variaveis_pos_variancia": len(
            variaveis_pos_variancia
        ),
        "n_registros_base_2023": info_base[
            "n_registros_base"
        ],
        "n_ideb_ausente_base_2023": info_base[
            "n_ideb_ausente"
        ],
        "n_registros_teste_pipeline": n_teste,
        "predicao_teste_min": float(
            np.min(
                predicoes
            )
        ),
        "predicao_teste_max": float(
            np.max(
                predicoes
            )
        ),
        **metricas,
    }

    print(
        "Modelo:",
        resultado[
            "modelo"
        ],
    )
    print(
        "Estratégia:",
        resultado[
            "estrategia"
        ],
    )
    print(
        "Variáveis na avaliação:",
        resultado[
            "n_variaveis_avaliacao"
        ],
    )
    print(
        "Variáveis na implantação:",
        resultado[
            "n_variaveis_implantacao"
        ],
    )
    print(
        "Registros testados:",
        n_teste,
    )
    print(
        "Status: APROVADO"
    )

    return resultado


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    """Valida os artefatos instalados das duas etapas."""
    print(
        "=" * 80
    )
    print(
        "VALIDAÇÃO DOS ARTEFATOS INSTALADOS DO SIMULADOR DO IDEB"
    )
    print(
        "=" * 80
    )
    print(
        "Este script não treina modelos."
    )

    resultados = []

    for (
        chave_etapa,
        config,
    ) in CONFIG_ETAPAS.items():

        resultado = validar_etapa(
            chave_etapa=chave_etapa,
            config=config,
        )

        resultados.append(
            resultado
        )

    tabela_resumo = pd.DataFrame(
        resultados
    )

    caminho_saida = (
        PASTA_OUTPUTS
        / "resumo_validacao_artefatos_simulador.csv"
    )

    tabela_resumo.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "VALIDAÇÃO CONCLUÍDA COM SUCESSO"
    )
    print(
        "=" * 80
    )
    print(
        tabela_resumo.to_string(
            index=False
        )
    )
    print(
        "\nResumo salvo em:",
        caminho_saida,
    )


if __name__ == "__main__":
    main()
