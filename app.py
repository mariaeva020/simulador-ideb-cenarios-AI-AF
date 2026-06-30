# ============================================================
# APLICAÇÃO WEB: SIMULADOR DE CENÁRIOS EDUCACIONAIS DO IDEB
# ============================================================
#
# Objetivo:
# Esta aplicação permite simular cenários hipotéticos para o IDEB
# municipal a partir do artefato final do modelo selecionado no artigo.
#
# Observação metodológica:
# O simulador não realiza previsão futura nem inferência causal.
# Os resultados representam simulações condicionais do modelo, obtidas
# por análise de sensibilidade sobre os registros observados de 2023.
#
# Modelo:
# O modelo é carregado a partir do artefato final gerado no notebook
# metodológico do artigo. Esse artefato deve conter o modelo final,
# as variáveis selecionadas, o imputador, o seletor de variância e as
# métricas utilizadas na avaliação.
# ============================================================

import os
import re
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# CONFIGURAÇÕES DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Simulador de Cenários do IDEB",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTES DO PROJETO
# ============================================================

PASTA_DADOS = "data"
PASTA_MODELOS = "models"
PASTA_ASSETS = "assets"

# A aplicação procura, para cada etapa, um artefato final exportado pelo notebook.
# Para Anos Iniciais, também aceita o nome genérico artefato_modelo_final_ideb.pkl.
CONFIG_ETAPAS = {
    "Anos Iniciais": {
        "base": os.path.join(PASTA_DADOS, "base_anos_iniciais.xlsx"),
        "artefatos_possiveis": [
            os.path.join(PASTA_MODELOS, "artefato_modelo_final_anos_iniciais.pkl"),
            os.path.join(PASTA_MODELOS, "artefato_modelo_final_ideb.pkl")
        ]
    },
    "Anos Finais": {
        "base": os.path.join(PASTA_DADOS, "base_anos_finais.xlsx"),
        "artefatos_possiveis": [
            os.path.join(PASTA_MODELOS, "artefato_modelo_final_anos_finais.pkl")
        ]
    }
}


# ============================================================
# ESTILO VISUAL
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    .titulo-principal {
        font-size: 2.1rem;
        font-weight: 700;
        color: #23406E;
        margin-bottom: 0.3rem;
    }

    .subtitulo {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }

    .caixa-aviso {
        background-color: #FFF7E6;
        border-left: 6px solid #E3A72F;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: #374151;
        font-size: 0.98rem;
        margin-bottom: 1.5rem;
    }

    .texto-pequeno {
        color: #6B7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNÇÕES AUXILIARES GERAIS
# ============================================================

def formatar_numero(valor, casas: int = 4) -> str:
    """Formata números no padrão brasileiro."""
    try:
        if pd.isna(valor):
            return "Não disponível"
        return f"{float(valor):.{casas}f}".replace(".", ",")
    except Exception:
        return str(valor)


def nome_variavel_texto(nome: str) -> str:
    """
    Converte o nome técnico da variável em texto mais legível para a interface.
    Exemplo: taxa_distorcao_idade_serie -> Taxa distorcao idade serie.
    """
    texto = str(nome)
    texto = texto.replace("_", " ")
    texto = texto.replace("-", " - ")
    texto = re.sub(r"\s+", " ", texto).strip()
    if not texto:
        return str(nome)
    return texto[:1].upper() + texto[1:]


def obter_coluna_existente(df: pd.DataFrame, candidatas: list[str]) -> str | None:
    """Retorna a primeira coluna existente entre as candidatas informadas."""
    for coluna in candidatas:
        if coluna in df.columns:
            return coluna
    return None


def obter_coluna_municipio(df: pd.DataFrame) -> str:
    """Identifica a coluna de município na base."""
    possiveis_colunas = [
        "nome_do_municipio",
        "nome_municipio",
        "Nome do Município",
        "Município",
        "municipio",
        "Nome Município"
    ]

    coluna = obter_coluna_existente(df, possiveis_colunas)

    if coluna is not None:
        return coluna

    df["Município"] = [f"Município {i + 1}" for i in range(df.shape[0])]
    return "Município"


def localizar_artefato(caminhos_possiveis: list[str]) -> str | None:
    """Localiza o primeiro artefato existente entre os caminhos possíveis."""
    for caminho in caminhos_possiveis:
        if os.path.exists(caminho):
            return caminho
    return None


def detectar_etapas_disponiveis() -> dict:
    """Retorna apenas as etapas que possuem base e artefato disponíveis."""
    etapas = {}

    for etapa, config in CONFIG_ETAPAS.items():
        caminho_base = config["base"]
        caminho_artefato = localizar_artefato(config["artefatos_possiveis"])

        if os.path.exists(caminho_base) and caminho_artefato is not None:
            etapas[etapa] = {
                "base": caminho_base,
                "artefato": caminho_artefato
            }

    return etapas


# ============================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================

@st.cache_data
def carregar_base(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega a base em Excel sem refazer o pré-processamento do treinamento."""
    if not os.path.exists(caminho_arquivo):
        st.error(
            f"A base de dados não foi encontrada em: {caminho_arquivo}. "
            "Verifique se o arquivo foi colocado corretamente na pasta 'data'."
        )
        st.stop()

    return pd.read_excel(caminho_arquivo)


@st.cache_resource
def carregar_artefato(caminho_artefato: str) -> dict:
    """Carrega o artefato final do modelo selecionado no artigo."""
    if not os.path.exists(caminho_artefato):
        st.error(
            f"O artefato do modelo não foi encontrado em: {caminho_artefato}. "
            "Exporte o artefato final do notebook metodológico para a pasta 'models'."
        )
        st.stop()

    artefato = joblib.load(caminho_artefato)

    chaves_obrigatorias = [
        "modelo_final",
        "variaveis_modelo",
        "imputador",
        "seletor_variancia",
        "variaveis_originais_X",
        "variaveis_pos_variancia"
    ]

    chaves_ausentes = [chave for chave in chaves_obrigatorias if chave not in artefato]

    if chaves_ausentes:
        st.error(
            "O artefato carregado não contém todas as informações necessárias. "
            "Chaves ausentes: " + ", ".join(chaves_ausentes)
        )
        st.stop()

    return artefato


# ============================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO E PREDIÇÃO
# ============================================================

def preparar_entrada_modelo(dados: pd.DataFrame, artefato: dict) -> pd.DataFrame:
    """
    Reaplica o mesmo pré-processamento usado no treinamento:
    seleção das variáveis originais, imputação, remoção de variância zero
    e seleção das variáveis finais do modelo.
    """
    dados = dados.copy()

    variaveis_originais = artefato["variaveis_originais_X"]
    variaveis_pos_variancia = artefato["variaveis_pos_variancia"]
    variaveis_modelo = artefato["variaveis_modelo"]
    imputador = artefato["imputador"]
    seletor_variancia = artefato["seletor_variancia"]

    for variavel in variaveis_originais:
        if variavel not in dados.columns:
            dados[variavel] = np.nan

    X_original = dados[variaveis_originais].copy()
    X_original = X_original.apply(pd.to_numeric, errors="coerce")

    X_imp_array = imputador.transform(X_original)
    X_imp = pd.DataFrame(
        X_imp_array,
        columns=variaveis_originais,
        index=dados.index
    )

    X_var_array = seletor_variancia.transform(X_imp)
    X_var = pd.DataFrame(
        X_var_array,
        columns=variaveis_pos_variancia,
        index=dados.index
    )

    variaveis_ausentes = [var for var in variaveis_modelo if var not in X_var.columns]
    if variaveis_ausentes:
        st.error(
            "As seguintes variáveis finais do modelo não foram encontradas após o pré-processamento: "
            + ", ".join(variaveis_ausentes)
        )
        st.stop()

    return X_var[variaveis_modelo].copy()


def aplicar_alteracoes_cenario(X_base: pd.DataFrame, alteracoes: list[dict]) -> pd.DataFrame:
    """Aplica alterações percentuais às variáveis finais do modelo."""
    X_cenario = X_base.copy()

    for item in alteracoes:
        variavel = item["variavel_tecnica"]
        operacao = item["Operação"]
        percentual = item["Percentual"]

        fator = 1 + percentual / 100 if operacao == "Aumento" else 1 - percentual / 100
        X_cenario[variavel] = X_cenario[variavel] * fator

        # Variáveis proporcionais ou percentuais não devem extrapolar limites plausíveis.
        nome_lower = variavel.lower()
        if (
            "percentual" in nome_lower
            or "taxa" in nome_lower
            or "grupo" in nome_lower
            or "proporcao" in nome_lower
            or "proporção" in nome_lower
        ):
            X_cenario[variavel] = X_cenario[variavel].clip(lower=0)
        else:
            X_cenario[variavel] = X_cenario[variavel].clip(lower=0)

    return X_cenario


def criar_tabela_metricas_artefato(artefato: dict) -> pd.DataFrame:
    """Cria uma tabela resumida com as métricas disponíveis no artefato."""
    modelo_final_escolhido = artefato.get("modelo_final_escolhido", {})
    metricas_modelo_final = artefato.get("metricas_modelo_final", None)
    resultado_friedman = artefato.get("resultado_friedman", None)

    if isinstance(modelo_final_escolhido, pd.Series):
        modelo_final_escolhido = modelo_final_escolhido.to_dict()

    linhas = [
        {"Indicador": "Modelo final", "Valor": artefato.get("nome_modelo", "Não disponível")},
        {"Indicador": "Conjunto de variáveis", "Valor": artefato.get("nome_conjunto", "Não disponível")},
        {"Indicador": "Número de variáveis", "Valor": len(artefato.get("variaveis_modelo", []))},
        {"Indicador": "RMSE na validação cruzada inicial", "Valor": formatar_numero(modelo_final_escolhido.get("RMSE_cv"))},
        {"Indicador": "R² na validação cruzada inicial", "Valor": formatar_numero(modelo_final_escolhido.get("R2_cv"))},
        {"Indicador": "RMSE no teste", "Valor": formatar_numero(modelo_final_escolhido.get("RMSE_teste"))},
        {"Indicador": "R² no teste", "Valor": formatar_numero(modelo_final_escolhido.get("R2_teste"))},
        {"Indicador": "RMSE médio no RepeatedKFold", "Valor": formatar_numero(modelo_final_escolhido.get("RMSE_repeated_medio"))},
        {"Indicador": "Desvio-padrão do RMSE no RepeatedKFold", "Valor": formatar_numero(modelo_final_escolhido.get("RMSE_repeated_desvio"))},
        {"Indicador": "R² médio no RepeatedKFold", "Valor": formatar_numero(modelo_final_escolhido.get("R2_repeated_medio"))}
    ]

    if isinstance(metricas_modelo_final, pd.DataFrame) and not metricas_modelo_final.empty:
        for _, linha in metricas_modelo_final.iterrows():
            base = linha.get("base", "")
            linhas.append({"Indicador": f"MAE final - {base}", "Valor": formatar_numero(linha.get("MAE"))})
            linhas.append({"Indicador": f"RMSE final - {base}", "Valor": formatar_numero(linha.get("RMSE"))})
            linhas.append({"Indicador": f"R² final - {base}", "Valor": formatar_numero(linha.get("R2"))})

    if isinstance(resultado_friedman, pd.DataFrame) and not resultado_friedman.empty:
        linhas.append({"Indicador": "Teste de Friedman - p-valor", "Valor": formatar_numero(resultado_friedman.iloc[0].get("p_valor"), 8)})
        linhas.append({"Indicador": "Teste de Friedman significativo a 5%", "Valor": str(resultado_friedman.iloc[0].get("significativo_5%"))})

    return pd.DataFrame(linhas)


# ============================================================
# ESTADO DA SESSÃO
# ============================================================

if "alteracoes_atuais" not in st.session_state:
    st.session_state.alteracoes_atuais = []

if "resultados_cenarios" not in st.session_state:
    st.session_state.resultados_cenarios = []

if "resultado_municipal" not in st.session_state:
    st.session_state.resultado_municipal = pd.DataFrame()


# ============================================================
# CABEÇALHO
# ============================================================

logo_path = os.path.join(PASTA_ASSETS, "logo_simulador_ideb.png")
if os.path.exists(logo_path):
    col_logo1, col_logo2, col_logo3 = st.columns([1, 6, 1])
    with col_logo2:
        st.image(logo_path, use_container_width=True)

st.markdown(
    '<div class="titulo-principal">Simulador de Cenários Educacionais para o IDEB Municipal</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitulo">'
    'Aplicação baseada no artefato final do modelo selecionado no artigo para simulação '
    'de alterações hipotéticas em variáveis associadas ao IDEB municipal.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="caixa-aviso">
    Esta aplicação não realiza previsão futura nem inferência causal. Os resultados representam
    simulações condicionais do modelo a partir dos dados observados em 2023. Portanto, as variações
    estimadas devem ser interpretadas como análise de sensibilidade, e não como efeito causal direto
    das variáveis sobre o IDEB.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# VERIFICAÇÃO DE ETAPAS DISPONÍVEIS
# ============================================================

ETAPAS_DISPONIVEIS = detectar_etapas_disponiveis()

if not ETAPAS_DISPONIVEIS:
    st.error(
        "Nenhuma etapa está pronta para uso. Verifique se a pasta 'data' contém a base Excel "
        "e se a pasta 'models' contém o artefato final exportado pelo notebook metodológico."
    )
    st.stop()


# ============================================================
# ABAS DA APLICAÇÃO
# ============================================================

aba_simulacao, aba_resultados, aba_municipios, aba_metricas, aba_metodologia = st.tabs(
    [
        "Simulação",
        "Resultados gerais",
        "Resultados municipais",
        "Métricas do modelo",
        "Notas metodológicas"
    ]
)


# ============================================================
# ABA 1: SIMULAÇÃO
# ============================================================

with aba_simulacao:
    st.subheader("Construção do cenário")

    col_etapa, col_info = st.columns([1.2, 2])

    with col_etapa:
        etapa = st.selectbox(
            "Selecione a etapa de ensino",
            list(ETAPAS_DISPONIVEIS.keys())
        )

    with col_info:
        st.info(
            "A simulação utiliza os registros de 2023. As alterações são aplicadas às "
            "variáveis finais do modelo e o sistema estima a resposta condicional do IDEB."
        )

    caminho_base = ETAPAS_DISPONIVEIS[etapa]["base"]
    caminho_artefato = ETAPAS_DISPONIVEIS[etapa]["artefato"]

    df = carregar_base(caminho_base)
    artefato = carregar_artefato(caminho_artefato)
    modelo = artefato["modelo_final"]
    variaveis_modelo = artefato["variaveis_modelo"]

    coluna_ano = obter_coluna_existente(df, ["ano", "Ano", "ANO"])
    coluna_ideb = obter_coluna_existente(df, ["ideb", "IDEB", "Ideb"])

    if coluna_ano is None:
        st.error("Não foi encontrada coluna de ano na base. Use 'ano' ou 'Ano'.")
        st.stop()

    if coluna_ideb is None:
        st.error("Não foi encontrada coluna de IDEB na base. Use 'ideb' ou 'IDEB'.")
        st.stop()

    df_2023 = df[df[coluna_ano] == 2023].copy()
    df_2023 = df_2023[df_2023[coluna_ideb].notnull()].copy()

    if df_2023.empty:
        st.error("Não foram encontrados registros válidos referentes ao ano de 2023 na base.")
        st.stop()

    X_base = preparar_entrada_modelo(df_2023, artefato)

    rotulos_variaveis = {var: nome_variavel_texto(var) for var in variaveis_modelo}

    st.markdown("### Variáveis do cenário")

    col1, col2, col3 = st.columns([2.2, 1, 1])

    with col1:
        variavel = st.selectbox(
            "Variável a alterar",
            variaveis_modelo,
            format_func=lambda x: rotulos_variaveis.get(x, x)
        )

    with col2:
        operacao = st.radio(
            "Operação",
            ["Aumento", "Redução"],
            horizontal=True
        )

    with col3:
        percentual = st.number_input(
            "Percentual",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5
        )

    col_botao1, col_botao2 = st.columns([1, 1])

    with col_botao1:
        adicionar = st.button(
            "Adicionar variável ao cenário",
            type="secondary",
            use_container_width=True
        )

    with col_botao2:
        limpar = st.button(
            "Limpar variáveis do cenário",
            use_container_width=True
        )

    if adicionar:
        variaveis_ja_adicionadas = [
            item["variavel_tecnica"] for item in st.session_state.alteracoes_atuais
        ]

        if variavel in variaveis_ja_adicionadas:
            st.warning("Essa variável já foi adicionada ao cenário atual.")
        else:
            st.session_state.alteracoes_atuais.append(
                {
                    "Variável": rotulos_variaveis[variavel],
                    "variavel_tecnica": variavel,
                    "Operação": operacao,
                    "Percentual": percentual
                }
            )
            st.success("Variável adicionada ao cenário.")

    if limpar:
        st.session_state.alteracoes_atuais = []
        st.success("Lista de variáveis do cenário limpa.")

    if st.session_state.alteracoes_atuais:
        st.markdown("#### Alterações adicionadas ao cenário")
        tabela_alteracoes = pd.DataFrame(st.session_state.alteracoes_atuais)
        tabela_alteracoes = tabela_alteracoes[["Variável", "Operação", "Percentual"]]
        st.dataframe(tabela_alteracoes, use_container_width=True)
    else:
        st.caption("Nenhuma variável foi adicionada ao cenário atual.")

    st.markdown("### Identificação do cenário")

    nome_cenario = st.text_input(
        "Nome do cenário",
        value="Cenário personalizado"
    )

    gerar = st.button(
        "Gerar cenário",
        type="primary",
        use_container_width=True
    )

    if gerar:
        if not st.session_state.alteracoes_atuais:
            st.warning("Adicione pelo menos uma variável antes de gerar o cenário.")
        elif not nome_cenario.strip():
            st.warning("Informe um nome para o cenário.")
        else:
            X_cenario = aplicar_alteracoes_cenario(
                X_base,
                st.session_state.alteracoes_atuais
            )

            pred_sem_alteracao = modelo.predict(X_base)
            pred_cenario = modelo.predict(X_cenario)

            ideb_real_medio = df_2023[coluna_ideb].mean()
            ideb_previsto_sem_alteracao = pred_sem_alteracao.mean()
            ideb_previsto_cenario = pred_cenario.mean()

            delta_previsto = ideb_previsto_cenario - ideb_previsto_sem_alteracao
            delta_real = ideb_previsto_cenario - ideb_real_medio

            descricao_variaveis = "; ".join(
                [
                    f"{item['Variável']} ({item['Operação']} de {item['Percentual']:.1f}%)"
                    for item in st.session_state.alteracoes_atuais
                ]
            )

            resultado_geral = {
                "Etapa": etapa,
                "Cenário": nome_cenario.strip(),
                "Variáveis alteradas": descricao_variaveis,
                "IDEB real médio em 2023": round(ideb_real_medio, 4),
                "IDEB previsto sem alteração": round(ideb_previsto_sem_alteracao, 4),
                "IDEB previsto no cenário": round(ideb_previsto_cenario, 4),
                "Diferença em relação ao previsto sem alteração": round(delta_previsto, 4),
                "Diferença em relação ao IDEB real": round(delta_real, 4)
            }

            st.session_state.resultados_cenarios.append(resultado_geral)

            coluna_municipio = obter_coluna_municipio(df_2023)

            tabela_municipal = df_2023[[coluna_municipio, coluna_ideb]].copy()
            tabela_municipal = tabela_municipal.rename(
                columns={
                    coluna_municipio: "Município",
                    coluna_ideb: "IDEB real em 2023"
                }
            )

            tabela_municipal["Etapa"] = etapa
            tabela_municipal["Cenário"] = nome_cenario.strip()
            tabela_municipal["Variáveis alteradas"] = descricao_variaveis
            tabela_municipal["IDEB previsto sem alteração"] = np.round(pred_sem_alteracao, 4)
            tabela_municipal["IDEB previsto no cenário"] = np.round(pred_cenario, 4)
            tabela_municipal["Diferença em relação ao previsto sem alteração"] = np.round(
                tabela_municipal["IDEB previsto no cenário"]
                - tabela_municipal["IDEB previsto sem alteração"],
                4
            )
            tabela_municipal["Diferença em relação ao IDEB real"] = np.round(
                tabela_municipal["IDEB previsto no cenário"]
                - tabela_municipal["IDEB real em 2023"],
                4
            )

            st.session_state.resultado_municipal = tabela_municipal.copy()
            st.session_state.alteracoes_atuais = []

            st.success("Cenário gerado com sucesso.")
            st.dataframe(pd.DataFrame([resultado_geral]), use_container_width=True)


# ============================================================
# ABA 2: RESULTADOS GERAIS
# ============================================================

with aba_resultados:
    st.subheader("Resultados gerais dos cenários simulados")

    if not st.session_state.resultados_cenarios:
        st.info("Nenhum cenário foi gerado ainda.")
    else:
        df_resultados = pd.DataFrame(st.session_state.resultados_cenarios)
        st.dataframe(df_resultados, use_container_width=True)

        st.markdown("### Indicadores do último cenário gerado")
        ultimo = df_resultados.iloc[-1]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric("IDEB real médio em 2023", formatar_numero(ultimo["IDEB real médio em 2023"]))
        with col_m2:
            st.metric("Previsto sem alteração", formatar_numero(ultimo["IDEB previsto sem alteração"]))
        with col_m3:
            st.metric("Previsto no cenário", formatar_numero(ultimo["IDEB previsto no cenário"]))
        with col_m4:
            st.metric("Diferença estimada", formatar_numero(ultimo["Diferença em relação ao previsto sem alteração"]))

        st.markdown("### IDEB médio previsto por cenário")
        fig_ideb = px.bar(
            df_resultados,
            x="Cenário",
            y="IDEB previsto no cenário",
            color="Etapa",
            text="IDEB previsto no cenário",
            title="IDEB médio previsto por cenário"
        )
        fig_ideb.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_ideb.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font_color="#23406E",
            xaxis_title="Cenário",
            yaxis_title="IDEB médio previsto",
            legend_title="Etapa de ensino"
        )
        st.plotly_chart(fig_ideb, use_container_width=True)

        st.markdown("### Variação em relação ao cenário sem alteração")
        fig_delta = px.bar(
            df_resultados,
            x="Cenário",
            y="Diferença em relação ao previsto sem alteração",
            color="Etapa",
            text="Diferença em relação ao previsto sem alteração",
            title="Diferença estimada em relação ao IDEB previsto sem alteração"
        )
        fig_delta.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_delta.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font_color="#23406E",
            xaxis_title="Cenário",
            yaxis_title="Diferença estimada",
            legend_title="Etapa de ensino"
        )
        fig_delta.add_hline(y=0, line_width=1, line_color="black")
        st.plotly_chart(fig_delta, use_container_width=True)

        csv_resultados = df_resultados.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Baixar resultados gerais em CSV",
            data=csv_resultados,
            file_name="resultados_gerais_cenarios_ideb.csv",
            mime="text/csv"
        )


# ============================================================
# ABA 3: RESULTADOS MUNICIPAIS
# ============================================================

with aba_municipios:
    st.subheader("Resultados municipais do último cenário gerado")

    if st.session_state.resultado_municipal.empty:
        st.info("Nenhum resultado municipal foi gerado ainda.")
    else:
        tabela_municipal = st.session_state.resultado_municipal.copy()
        st.dataframe(tabela_municipal, use_container_width=True)

        st.markdown("### Municípios com maiores variações positivas")
        ranking_positivo = tabela_municipal.sort_values(
            by="Diferença em relação ao previsto sem alteração",
            ascending=False
        ).head(15)

        fig_pos = px.bar(
            ranking_positivo,
            x="Diferença em relação ao previsto sem alteração",
            y="Município",
            orientation="h",
            text="Diferença em relação ao previsto sem alteração",
            title="Maiores variações positivas estimadas"
        )
        fig_pos.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_pos.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font_color="#23406E",
            xaxis_title="Diferença estimada",
            yaxis_title="Município",
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig_pos, use_container_width=True)

        st.markdown("### Municípios com maiores variações negativas")
        ranking_negativo = tabela_municipal.sort_values(
            by="Diferença em relação ao previsto sem alteração",
            ascending=True
        ).head(15)

        fig_neg = px.bar(
            ranking_negativo,
            x="Diferença em relação ao previsto sem alteração",
            y="Município",
            orientation="h",
            text="Diferença em relação ao previsto sem alteração",
            title="Maiores variações negativas estimadas"
        )
        fig_neg.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_neg.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font_color="#23406E",
            xaxis_title="Diferença estimada",
            yaxis_title="Município",
            yaxis={"categoryorder": "total descending"}
        )
        st.plotly_chart(fig_neg, use_container_width=True)

        csv_municipal = tabela_municipal.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Baixar resultados municipais em CSV",
            data=csv_municipal,
            file_name="resultados_municipais_cenario_ideb.csv",
            mime="text/csv"
        )


# ============================================================
# ABA 4: MÉTRICAS DO MODELO
# ============================================================

with aba_metricas:
    st.subheader("Métricas e informações do modelo")

    etapa_metricas = st.selectbox(
        "Selecione a etapa de ensino",
        list(ETAPAS_DISPONIVEIS.keys()),
        key="etapa_metricas"
    )

    artefato_metricas = carregar_artefato(ETAPAS_DISPONIVEIS[etapa_metricas]["artefato"])
    tabela_metricas = criar_tabela_metricas_artefato(artefato_metricas)

    st.dataframe(tabela_metricas, use_container_width=True)

    st.caption(
        "As métricas exibidas são aquelas armazenadas no artefato final do modelo, "
        "exportado a partir do notebook metodológico do artigo."
    )


# ============================================================
# ABA 5: NOTAS METODOLÓGICAS
# ============================================================

with aba_metodologia:
    st.subheader("Notas metodológicas da aplicação")

    etapa_metodologia = st.selectbox(
        "Selecione a etapa de ensino",
        list(ETAPAS_DISPONIVEIS.keys()),
        key="etapa_metodologia"
    )

    artefato_metodologia = carregar_artefato(ETAPAS_DISPONIVEIS[etapa_metodologia]["artefato"])
    variaveis_modelo_metodologia = artefato_metodologia["variaveis_modelo"]

    st.markdown(
        """
        Esta aplicação operacionaliza a simulação de cenários descrita no artigo.
        O sistema utiliza o artefato final do modelo selecionado no processo metodológico,
        preservando o pré-processamento aplicado no treinamento, incluindo imputação,
        remoção de variáveis de variância zero e seleção do subconjunto final de variáveis.

        A simulação é realizada sobre os registros de 2023. O usuário seleciona uma ou mais
        variáveis explicativas, define percentuais de aumento ou redução e o sistema recalcula
        o IDEB previsto pelo modelo para o novo cenário.

        A diferença entre o IDEB previsto sem alteração e o IDEB previsto no cenário representa
        uma variação estimada pelo modelo. Essa diferença não deve ser interpretada como efeito
        causal, pois a aplicação não identifica mecanismos causais, não controla confundimento
        por desenho causal e não estima efeitos de tratamento.
        """
    )

    st.markdown("### Variáveis disponíveis para simulação")

    tabela_variaveis = pd.DataFrame(
        {
            "Variável exibida no simulador": [nome_variavel_texto(v) for v in variaveis_modelo_metodologia],
            "Nome técnico no artefato": variaveis_modelo_metodologia
        }
    )

    st.dataframe(tabela_variaveis, use_container_width=True)

    st.markdown(
        """
        A interpretação dos cenários deve considerar as limitações dos dados, a agregação
        municipal, a qualidade das variáveis disponíveis e o fato de que modelos preditivos
        capturam padrões estatísticos presentes na base, mas não substituem análise educacional,
        institucional e territorial.
        """
    )
