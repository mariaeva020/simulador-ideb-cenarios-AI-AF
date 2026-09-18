import json
import os
from html import escape

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Simulador de Cenários do IDEB",
    layout="wide",
    initial_sidebar_state="expanded",
)

PASTA_DADOS = "data"
PASTA_MODELOS = "models"
PASTA_ASSETS = "assets"

CONFIG_ETAPAS = {
    "Anos Iniciais": {
        "base": os.path.join(PASTA_DADOS, "base_referencia_2023_anos_iniciais.csv"),
        "pasta_modelo": os.path.join(PASTA_MODELOS, "anos_iniciais"),
    },
    "Anos Finais": {
        "base": os.path.join(PASTA_DADOS, "base_referencia_2023_anos_finais.csv"),
        "pasta_modelo": os.path.join(PASTA_MODELOS, "anos_finais"),
    },
}

ARQUIVOS_MODELO = [
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


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .titulo-principal {font-size: 2.1rem; font-weight: 700; color: #23406E; margin-bottom: .3rem;}
    .subtitulo {font-size: 1.05rem; color: #4B5563; margin-bottom: 1.2rem;}
    .caixa-aviso, .alerta-metodologico {
        background-color: #FFF7E6; border-left: 6px solid #E3A72F;
        padding: 1rem 1.2rem; border-radius: 12px; color: #374151;
        line-height: 1.6; margin-bottom: 1.2rem;
    }
    .nota-metodologica {
        background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
        padding: 1.2rem 1.4rem; margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(15,23,42,.05);
    }
    .nota-metodologica h4 {margin-top: 0; margin-bottom: .5rem; color: #1F2937;}
    .nota-metodologica p {color: #374151; line-height: 1.65; margin-bottom: 0; text-align: justify;}
    .metric-card-grid {
        display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
        gap:1rem; margin:1.2rem 0 1.4rem 0;
    }
    .metric-card {
        background:#FFF; border:1px solid #E5E7EB; border-radius:14px;
        padding:1rem 1.2rem; box-shadow:0 1px 4px rgba(15,23,42,.06);
    }
    .metric-card-label {font-size:.85rem; color:#6B7280; margin-bottom:.35rem;}
    .metric-card-value {font-size:1.28rem; font-weight:700; color:#1F2937;}
    .tabela-artigo table {width:100%; border-collapse:collapse; margin:.8rem 0 1.2rem; font-size:.95rem;}
    .tabela-artigo th {background:#F3F4F6; font-weight:700; text-align:center; border:1px solid #D1D5DB; padding:.65rem;}
    .tabela-artigo td {border:1px solid #D1D5DB; padding:.6rem; text-align:center;}
    </style>
    """,
    unsafe_allow_html=True,
)


def formatar_numero(valor, casas=4):
    try:
        if valor is None or pd.isna(valor):
            return "Não disponível"
        return f"{float(valor):.{casas}f}".replace(".", ",")
    except Exception:
        return str(valor)


def obter_coluna_existente(df, candidatas):
    for coluna in candidatas:
        if coluna in df.columns:
            return coluna
    return None


def obter_coluna_municipio(df):
    coluna = obter_coluna_existente(
        df,
        ["nome_do_municipio", "nome_municipio", "municipio", "Município", "Nome do Município"],
    )
    if coluna is None:
        raise ValueError("A base de referência não contém a coluna de município.")
    return coluna


def caminho(pasta, arquivo):
    return os.path.join(pasta, arquivo)


def verificar_etapa(config):
    ausentes = []
    if not os.path.exists(config["base"]):
        ausentes.append(config["base"])
    for arquivo in ARQUIVOS_MODELO:
        arq = caminho(config["pasta_modelo"], arquivo)
        if not os.path.exists(arq):
            ausentes.append(arq)
    return ausentes


def detectar_etapas_disponiveis():
    return {
        etapa: config
        for etapa, config in CONFIG_ETAPAS.items()
        if not verificar_etapa(config)
    }


@st.cache_data
def carregar_csv(caminho_arquivo):
    return pd.read_csv(caminho_arquivo)


@st.cache_data
def carregar_json(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


@st.cache_resource
def carregar_joblib(caminho_arquivo):
    return joblib.load(caminho_arquivo)


@st.cache_resource
def carregar_artefatos(pasta_modelo):
    return {
        "modelo_implantacao": carregar_joblib(caminho(pasta_modelo, "modelo_implantacao.joblib")),
        "imputador_implantacao": carregar_joblib(caminho(pasta_modelo, "imputador_implantacao.joblib")),
        "seletor_variancia_implantacao": carregar_joblib(caminho(pasta_modelo, "seletor_variancia_implantacao.joblib")),
        "variaveis_pos_variancia": carregar_joblib(caminho(pasta_modelo, "variaveis_pos_variancia.joblib")),
        "variaveis_modelo": carregar_joblib(caminho(pasta_modelo, "variaveis_modelo.joblib")),
        "colunas_entrada_modelo": carregar_joblib(caminho(pasta_modelo, "colunas_entrada_modelo.joblib")),
        "metadados": carregar_json(caminho(pasta_modelo, "metadados.json")),
        "resumo_tecnico": carregar_csv(caminho(pasta_modelo, "resumo_tecnico_modelo_final.csv")),
        "suporte_empirico": carregar_csv(caminho(pasta_modelo, "suporte_empirico_variaveis.csv")),
    }


def construir_rotulos(artefatos):
    suporte = artefatos["suporte_empirico"]
    if {"variavel", "rotulo"}.issubset(suporte.columns):
        return dict(zip(suporte["variavel"].astype(str), suporte["rotulo"].astype(str)))
    return {
        v: str(v).replace("_", " ").strip().capitalize()
        for v in artefatos["variaveis_modelo"]
    }


def resumo_em_dict(artefatos):
    resumo = artefatos["resumo_tecnico"]
    if not {"item", "valor"}.issubset(resumo.columns):
        return {}
    saida = {}
    for item, valor in zip(resumo["item"], resumo["valor"]):
        texto = str(valor).strip().replace(",", ".")
        try:
            saida[str(item).strip()] = float(texto)
        except ValueError:
            saida[str(item).strip()] = str(valor)
    return saida


def preparar_entrada(dados, artefatos):
    colunas = list(artefatos["colunas_entrada_modelo"])
    pos_var = list(artefatos["variaveis_pos_variancia"])
    variaveis = list(artefatos["variaveis_modelo"])

    ausentes = [c for c in colunas if c not in dados.columns]
    if ausentes:
        raise ValueError("Colunas ausentes na base: " + ", ".join(ausentes))

    X = dados[colunas].copy().apply(pd.to_numeric, errors="coerce")

    X_imp = pd.DataFrame(
        artefatos["imputador_implantacao"].transform(X),
        columns=colunas,
        index=X.index,
    )

    X_var = pd.DataFrame(
        artefatos["seletor_variancia_implantacao"].transform(X_imp),
        columns=pos_var,
        index=X.index,
    )

    faltantes = [v for v in variaveis if v not in X_var.columns]
    if faltantes:
        raise ValueError("Variáveis finais ausentes: " + ", ".join(faltantes))

    return X_var[variaveis].copy()


def prever(dados, artefatos):
    X_final = preparar_entrada(dados, artefatos)
    return np.asarray(
        artefatos["modelo_implantacao"].predict(X_final)
    ).ravel()


def aplicar_alteracoes(dados, alteracoes):
    cenario = dados.copy()
    for item in alteracoes:
        var = item["variavel_tecnica"]
        if var not in cenario.columns:
            raise ValueError(f"A variável '{var}' não existe na base de referência.")
        serie = pd.to_numeric(cenario[var], errors="coerce")
        p = float(item["Percentual"]) / 100
        fator = 1 + p if item["Operação"] == "Aumento" else 1 - p
        cenario[var] = serie * fator
    return cenario


def avaliar_extrapolacao(dados_cenario, alteracoes, artefatos):
    suporte = artefatos["suporte_empirico"]
    if "variavel" not in suporte.columns:
        return []

    suporte = suporte.set_index("variavel")
    alertas = []

    for item in alteracoes:
        var = item["variavel_tecnica"]
        if var not in suporte.index or var not in dados_cenario.columns:
            continue

        linha = suporte.loc[var]
        serie = pd.to_numeric(dados_cenario[var], errors="coerce").dropna()
        if serie.empty:
            continue

        rotulo = linha.get("rotulo", var)
        minimo, maximo = float(serie.min()), float(serie.max())

        def num(nome):
            return pd.to_numeric(pd.Series([linha.get(nome)]), errors="coerce").iloc[0]

        min_obs, max_obs, p01, p99 = num("min"), num("max"), num("p01"), num("p99")

        fora_extremos = (
            (pd.notna(min_obs) and minimo < min_obs)
            or (pd.notna(max_obs) and maximo > max_obs)
        )
        fora_central = (
            (pd.notna(p01) and minimo < p01)
            or (pd.notna(p99) and maximo > p99)
        )

        if fora_extremos:
            alertas.append(
                f"{rotulo}: há valores além do intervalo mínimo-máximo observado."
            )
        elif fora_central:
            alertas.append(
                f"{rotulo}: há valores além da faixa entre os percentis 1 e 99."
            )

    return alertas


def renderizar_tabela(tabela):
    if tabela.empty:
        st.info("Não há informações disponíveis.")
        return
    st.markdown(
        f'<div class="tabela-artigo">{tabela.to_html(index=False, escape=True)}</div>',
        unsafe_allow_html=True,
    )


def tabela_metricas(artefatos):
    r = resumo_em_dict(artefatos)
    tabela = pd.DataFrame(
        [
            {
                "Avaliação": "Teste territorial independente",
                "MAE": r.get("MAE - teste territorial", np.nan),
                "RMSE": r.get("RMSE - teste territorial", np.nan),
                "R²": r.get("R² - teste territorial", np.nan),
            },
            {
                "Avaliação": "Validação temporal em 2023",
                "MAE": r.get("MAE - validação temporal 2023", np.nan),
                "RMSE": r.get("RMSE - validação temporal 2023", np.nan),
                "R²": r.get("R² - validação temporal 2023", np.nan),
            },
        ]
    )
    for col in ["MAE", "RMSE", "R²"]:
        tabela[col] = tabela[col].apply(lambda x: formatar_numero(x, 4))
    return tabela


def renderizar_cartoes(artefatos):
    r = resumo_em_dict(artefatos)
    m = artefatos["metadados"]

    modelo = r.get("Modelo final", m.get("modelo", "Não disponível"))

    estrategia = (
        m.get("estrategia_selecao_descricao")
        or r.get("Estratégia de seleção")
        or m.get("estrategia_selecao")
        or m.get("estrategia_selecao_codigo")
        or "Não disponível"
    )
    n_avaliacao = m.get(
        "n_variaveis_modelo_avaliacao",
        r.get("Número de variáveis", "Não disponível"),
    )
    n_implantacao = m.get(
        "n_variaveis_modelo_implantacao",
        len(artefatos["variaveis_modelo"]),
    )

    st.markdown(
        f"""
        <div class="metric-card-grid">
            <div class="metric-card"><div class="metric-card-label">Modelo avaliado</div>
            <div class="metric-card-value">{escape(str(modelo))}</div></div>
            <div class="metric-card"><div class="metric-card-label">Estratégia de seleção</div>
            <div class="metric-card-value">{escape(str(estrategia))}</div></div>
            <div class="metric-card"><div class="metric-card-label">Variáveis na avaliação</div>
            <div class="metric-card-value">{escape(str(n_avaliacao))}</div></div>
            <div class="metric-card"><div class="metric-card-label">Variáveis na implantação</div>
            <div class="metric-card-value">{escape(str(n_implantacao))}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


for chave, valor in {
    "alteracoes_atuais": [],
    "etapa_alteracoes": None,
    "resultados_cenarios": [],
    "resultados_municipais_cenarios": [],
}.items():
    if chave not in st.session_state:
        st.session_state[chave] = valor


logo = os.path.join(PASTA_ASSETS, "logo_simulador_ideb.png")
if os.path.exists(logo):
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.image(logo, use_container_width=True)

st.markdown(
    '<div class="titulo-principal">Simulador de Cenários Educacionais para o IDEB Municipal</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitulo">Aplicação para exploração de cenários hipotéticos a partir dos modelos de implantação produzidos no estudo.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="caixa-aviso">
    Esta aplicação não realiza previsão futura nem inferência causal.
    Os resultados representam respostas condicionais dos modelos de implantação
    a alterações hipotéticas aplicadas aos registros observados de 2023.
    </div>
    """,
    unsafe_allow_html=True,
)


ETAPAS = detectar_etapas_disponiveis()

if not ETAPAS:
    st.error(
        "Nenhuma etapa está pronta para uso. Copie as bases de referência para "
        "'data' e os artefatos de cada etapa para as respectivas subpastas de 'models'."
    )
    with st.expander("Arquivos ausentes"):
        for etapa, config in CONFIG_ETAPAS.items():
            st.markdown(f"**{etapa}**")
            st.code("\n".join(verificar_etapa(config)) or "Todos os arquivos foram encontrados.")
    st.stop()


aba_simulacao, aba_resultados, aba_municipios, aba_metricas, aba_metodologia = st.tabs(
    ["Simulação", "Resultados gerais", "Resultados municipais", "Métricas do modelo", "Notas metodológicas"]
)


with aba_simulacao:
    st.subheader("Construção do cenário")

    c1, c2 = st.columns([1.2, 2])
    with c1:
        etapa = st.selectbox("Selecione a etapa de ensino", list(ETAPAS.keys()))
    with c2:
        st.info(
            "As alterações são feitas na escala original dos dados de 2023 e, "
            "depois, o mesmo pré-processamento do modelo de implantação é reaplicado."
        )

    config = ETAPAS[etapa]
    base_2023 = carregar_csv(config["base"])
    artefatos = carregar_artefatos(config["pasta_modelo"])
    variaveis = list(artefatos["variaveis_modelo"])
    rotulos = construir_rotulos(artefatos)

    if st.session_state.etapa_alteracoes not in (None, etapa):
        st.session_state.alteracoes_atuais = []
    st.session_state.etapa_alteracoes = etapa

    coluna_ideb = obter_coluna_existente(base_2023, ["ideb", "IDEB", "Ideb"])
    if coluna_ideb is None:
        st.error("A base de referência não contém a coluna do IDEB.")
        st.stop()

    base_2023 = base_2023[base_2023[coluna_ideb].notna()].copy()
    if base_2023.empty:
        st.error("Não há registros válidos de 2023 com IDEB observado.")
        st.stop()

    st.markdown("### Variáveis do cenário")
    c1, c2, c3 = st.columns([2.2, 1, 1])

    with c1:
        variavel = st.selectbox(
            "Variável a alterar",
            variaveis,
            format_func=lambda x: rotulos.get(x, str(x).replace("_", " ").capitalize()),
        )
    with c2:
        operacao = st.radio("Operação", ["Aumento", "Redução"], horizontal=True)
    with c3:
        percentual = st.number_input(
            "Percentual", min_value=0.0, max_value=100.0, value=10.0, step=0.5
        )

    b1, b2 = st.columns(2)
    with b1:
        adicionar = st.button("Adicionar variável ao cenário", use_container_width=True)
    with b2:
        limpar = st.button("Limpar variáveis do cenário", use_container_width=True)

    if adicionar:
        existentes = {x["variavel_tecnica"] for x in st.session_state.alteracoes_atuais}
        if variavel in existentes:
            st.warning("Essa variável já foi adicionada ao cenário atual.")
        else:
            st.session_state.alteracoes_atuais.append(
                {
                    "Variável": rotulos.get(variavel, variavel),
                    "variavel_tecnica": variavel,
                    "Operação": operacao,
                    "Percentual": percentual,
                }
            )

    if limpar:
        st.session_state.alteracoes_atuais = []

    if st.session_state.alteracoes_atuais:
        tabela_alt = pd.DataFrame(st.session_state.alteracoes_atuais)[
            ["Variável", "Operação", "Percentual"]
        ]
        st.dataframe(tabela_alt, use_container_width=True, hide_index=True)
    else:
        st.caption("Nenhuma variável foi adicionada ao cenário atual.")

    st.markdown("### Identificação do cenário")
    nome_cenario = st.text_input("Nome do cenário", value="Cenário personalizado")
    gerar = st.button("Gerar cenário", type="primary", use_container_width=True)

    if gerar:
        if not st.session_state.alteracoes_atuais:
            st.warning("Adicione pelo menos uma variável antes de gerar o cenário.")
        elif not nome_cenario.strip():
            st.warning("Informe um nome para o cenário.")
        else:
            alteracoes_usadas = [dict(x) for x in st.session_state.alteracoes_atuais]

            try:
                dados_cenario = aplicar_alteracoes(base_2023, alteracoes_usadas)
                pred_base = prever(base_2023, artefatos)
                pred_cenario = prever(dados_cenario, artefatos)
            except Exception as erro:
                st.error(f"Não foi possível gerar o cenário. Detalhe técnico: {erro}")
                st.stop()

            ideb_real = pd.to_numeric(base_2023[coluna_ideb], errors="coerce").mean()
            previsto_base = float(np.mean(pred_base))
            previsto_cenario = float(np.mean(pred_cenario))

            descricao = "; ".join(
                f"{x['Variável']} ({x['Operação']} de {x['Percentual']:.1f}%)"
                for x in alteracoes_usadas
            )

            resultado = {
                "Etapa": etapa,
                "Cenário": nome_cenario.strip(),
                "Variáveis alteradas": descricao,
                "IDEB real médio em 2023": round(ideb_real, 4),
                "IDEB previsto sem alteração": round(previsto_base, 4),
                "IDEB previsto no cenário": round(previsto_cenario, 4),
                "Diferença em relação ao previsto sem alteração": round(previsto_cenario - previsto_base, 4),
                "Diferença em relação ao IDEB real": round(previsto_cenario - ideb_real, 4),
            }
            st.session_state.resultados_cenarios.append(resultado)

            coluna_municipio = obter_coluna_municipio(base_2023)
            municipal = base_2023[[coluna_municipio, coluna_ideb]].copy().rename(
                columns={coluna_municipio: "Município", coluna_ideb: "IDEB real em 2023"}
            )
            municipal["Etapa"] = etapa
            municipal["Cenário"] = nome_cenario.strip()
            municipal["Variáveis alteradas"] = descricao
            municipal["IDEB previsto sem alteração"] = np.round(pred_base, 4)
            municipal["IDEB previsto no cenário"] = np.round(pred_cenario, 4)
            municipal["Diferença em relação ao previsto sem alteração"] = np.round(
                pred_cenario - pred_base, 4
            )
            municipal["Diferença em relação ao IDEB real"] = np.round(
                pred_cenario
                - pd.to_numeric(municipal["IDEB real em 2023"], errors="coerce").to_numpy(),
                4,
            )
            st.session_state.resultados_municipais_cenarios.append(municipal)

            alertas = avaliar_extrapolacao(dados_cenario, alteracoes_usadas, artefatos)
            st.session_state.alteracoes_atuais = []

            st.success("Cenário gerado com sucesso.")
            st.dataframe(pd.DataFrame([resultado]), use_container_width=True, hide_index=True)

            if alertas:
                st.warning(
                    "O cenário contém valores fora do suporte empírico mais representativo do conjunto utilizado no ajuste."
                )
                for alerta in alertas:
                    st.write(f"- {alerta}")


with aba_resultados:
    st.subheader("Resultados gerais dos cenários simulados")

    if not st.session_state.resultados_cenarios:
        st.info("Nenhum cenário foi gerado ainda.")
    else:
        resultados = pd.DataFrame(st.session_state.resultados_cenarios)
        st.dataframe(resultados, use_container_width=True, hide_index=True)

        ultimo = resultados.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IDEB real médio em 2023", formatar_numero(ultimo["IDEB real médio em 2023"]))
        c2.metric("Previsto sem alteração", formatar_numero(ultimo["IDEB previsto sem alteração"]))
        c3.metric("Previsto no cenário", formatar_numero(ultimo["IDEB previsto no cenário"]))
        c4.metric(
            "Diferença estimada",
            formatar_numero(ultimo["Diferença em relação ao previsto sem alteração"]),
        )

        fig = px.bar(
            resultados,
            x="Cenário",
            y="IDEB previsto no cenário",
            color="Etapa",
            text="IDEB previsto no cenário",
            title="IDEB médio previsto por cenário",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            resultados,
            x="Cenário",
            y="Diferença em relação ao previsto sem alteração",
            color="Etapa",
            text="Diferença em relação ao previsto sem alteração",
            title="Diferença estimada em relação ao IDEB previsto sem alteração",
        )
        fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig2.add_hline(y=0, line_width=1, line_color="black")
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "Baixar resultados gerais em CSV",
            resultados.to_csv(index=False, encoding="utf-8-sig"),
            "resultados_gerais_cenarios_ideb.csv",
            "text/csv",
        )


with aba_municipios:
    st.subheader("Resultados municipais dos cenários simulados")

    if not st.session_state.resultados_municipais_cenarios:
        st.info("Nenhum resultado municipal foi gerado ainda.")
    else:
        municipal = pd.concat(
            st.session_state.resultados_municipais_cenarios, ignore_index=True
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            filtro_etapa = st.selectbox(
                "Filtrar por etapa de ensino",
                ["Todas"] + sorted(municipal["Etapa"].dropna().unique().tolist()),
                key="filtro_etapa_municipal",
            )
        with c2:
            filtro_cenario = st.selectbox(
                "Filtrar por cenário",
                ["Todos"] + sorted(municipal["Cenário"].dropna().unique().tolist()),
                key="filtro_cenario_municipal",
            )

        filtrada = municipal.copy()
        if filtro_etapa != "Todas":
            filtrada = filtrada[filtrada["Etapa"] == filtro_etapa]
        if filtro_cenario != "Todos":
            filtrada = filtrada[filtrada["Cenário"] == filtro_cenario]

        st.dataframe(filtrada, use_container_width=True, hide_index=True)

        positivo = filtrada.sort_values(
            "Diferença em relação ao previsto sem alteração", ascending=False
        ).head(15)
        figp = px.bar(
            positivo,
            x="Diferença em relação ao previsto sem alteração",
            y="Município",
            color="Cenário",
            orientation="h",
            text="Diferença em relação ao previsto sem alteração",
            title="Maiores variações positivas estimadas",
        )
        figp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        figp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figp, use_container_width=True)

        negativo = filtrada.sort_values(
            "Diferença em relação ao previsto sem alteração", ascending=True
        ).head(15)
        fign = px.bar(
            negativo,
            x="Diferença em relação ao previsto sem alteração",
            y="Município",
            color="Cenário",
            orientation="h",
            text="Diferença em relação ao previsto sem alteração",
            title="Maiores variações negativas estimadas",
        )
        fign.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fign.update_layout(yaxis={"categoryorder": "total descending"})
        st.plotly_chart(fign, use_container_width=True)

        st.download_button(
            "Baixar resultados municipais consolidados em CSV",
            municipal.to_csv(index=False, encoding="utf-8-sig"),
            "resultados_municipais_cenarios_ideb.csv",
            "text/csv",
        )


with aba_metricas:
    st.subheader("Desempenho do modelo avaliado")

    etapa_m = st.selectbox(
        "Selecione a etapa de ensino", list(ETAPAS.keys()), key="etapa_metricas"
    )
    art_m = carregar_artefatos(ETAPAS[etapa_m]["pasta_modelo"])

    renderizar_cartoes(art_m)
    st.markdown("#### Avaliação preditiva")
    renderizar_tabela(tabela_metricas(art_m))

    st.markdown(
        """
        <div class="alerta-metodologico">
        As métricas apresentadas pertencem ao modelo submetido ao protocolo de
        avaliação científica. O modelo utilizado nas simulações foi reajustado
        posteriormente com todos os dados disponíveis de 2013 a 2023 e não é
        utilizado para estimar desempenho de generalização.
        </div>
        """,
        unsafe_allow_html=True,
    )


with aba_metodologia:
    st.subheader("Notas metodológicas da aplicação")

    etapa_n = st.selectbox(
        "Selecione a etapa de ensino", list(ETAPAS.keys()), key="etapa_metodologia"
    )
    art_n = carregar_artefatos(ETAPAS[etapa_n]["pasta_modelo"])
    r = resumo_em_dict(art_n)
    m = art_n["metadados"]

    modelo = r.get("Modelo final", m.get("modelo", "Não disponível"))

    estrategia = (
        m.get("estrategia_selecao_descricao")
        or r.get("Estratégia de seleção")
        or m.get("estrategia_selecao")
        or m.get("estrategia_selecao_codigo")
        or "Não disponível"
    )
    n_av = m.get("n_variaveis_modelo_avaliacao", r.get("Número de variáveis", "Não disponível"))
    n_imp = m.get("n_variaveis_modelo_implantacao", len(art_n["variaveis_modelo"]))

    st.markdown(
        f"""
        <div class="nota-metodologica">
        <h4>Escopo da aplicação</h4>
        <p>
        Para a etapa selecionada, o estudo avaliou o algoritmo
        <strong>{escape(str(modelo))}</strong> com a estratégia
        <strong>{escape(str(estrategia))}</strong>. O modelo submetido à avaliação
        científica utilizou <strong>{escape(str(n_av))}</strong> variáveis.
        Após o encerramento da avaliação, o modelo de implantação foi reajustado
        com todos os dados disponíveis de 2013 a 2023, resultando em
        <strong>{escape(str(n_imp))}</strong> variáveis para uso no simulador.
        </p></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="nota-metodologica"><h4>Procedimento de simulação</h4><p>
        A aplicação utiliza a base de referência de 2023 exportada pelo pipeline
        metodológico. As alterações são aplicadas às variáveis em sua escala
        original. Depois, o sistema reaplica o imputador, o seletor de variância
        e o subconjunto de variáveis definidos para implantação antes da predição.
        </p></div>
        <div class="nota-metodologica"><h4>Interpretação dos resultados</h4><p>
        A diferença entre o IDEB previsto sem alteração e o previsto no cenário
        representa uma variação preditiva estimada. Ela não deve ser interpretada
        como efeito causal. A aplicação não estima efeitos de tratamento nem
        identifica mecanismos causais.
        </p></div>
        <div class="nota-metodologica"><h4>Suporte empírico</h4><p>
        A aplicação compara os valores simulados com o suporte empírico das
        variáveis. Valores além dos percentis 1 e 99 ou do intervalo mínimo-máximo
        observado são sinalizados como extrapolações.
        </p></div>
        """,
        unsafe_allow_html=True,
    )

    rotulos_n = construir_rotulos(art_n)
    vars_n = list(art_n["variaveis_modelo"])
    tabela_vars = pd.DataFrame(
        {
            "Nº": range(1, len(vars_n) + 1),
            "Variável exibida no simulador": [
                rotulos_n.get(v, str(v).replace("_", " ").capitalize()) for v in vars_n
            ],
            "Nome técnico no artefato": vars_n,
        }
    )
    st.markdown("#### Variáveis disponíveis para simulação")
    renderizar_tabela(tabela_vars)
