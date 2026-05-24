# ============================================================
# APLICAÇÃO WEB: SIMULADOR DE CENÁRIOS EDUCACIONAIS DO IDEB
# ============================================================
#
# Objetivo:
# Esta aplicação permite simular cenários hipotéticos para o IDEB
# municipal dos Anos Iniciais e dos Anos Finais do Ensino Fundamental.
#
# A aplicação utiliza modelos XGBoost previamente treinados e salvos
# pelo script "treinar_modelos.py".
#
# Escopo:
# 1. Anos Iniciais do Ensino Fundamental
# 2. Anos Finais do Ensino Fundamental
#
# Observação metodológica:
# Os resultados não devem ser interpretados como previsão futura nem
# como efeito causal. A aplicação realiza análise de sensibilidade
# baseada em alterações hipotéticas nas variáveis explicativas.
# ============================================================


# ============================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import os
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
    layout="wide"
)


# ============================================================
# CONSTANTES DO PROJETO
# ============================================================

# Pasta onde estão as bases de dados.
PASTA_DADOS = "data"

# Pasta onde estão os modelos treinados.
PASTA_MODELOS = "models"

# Variáveis utilizadas no modelo final.
VARIAVEIS_MODELO = [
    "PIB per capita",
    "taxa_distorcao_idade_serie",
    "Grupo 5-adeq form docente",
    "Área plantada ou destinada à colheita -lavour",
    "Valor da produção na extração vegetal",
    "Valor repassado-Criança Feliz"
]

# Caminhos das bases de dados.
CAMINHOS_BASES = {
    "Anos Iniciais": os.path.join(PASTA_DADOS, "base_anos_iniciais.xlsx"),
    "Anos Finais": os.path.join(PASTA_DADOS, "base_anos_finais.xlsx")
}

# Caminhos dos modelos treinados.
CAMINHOS_MODELOS = {
    "Anos Iniciais": os.path.join(PASTA_MODELOS, "modelo_xgb_anos_iniciais.pkl"),
    "Anos Finais": os.path.join(PASTA_MODELOS, "modelo_xgb_anos_finais.pkl")
}

# Caminhos das métricas salvas.
CAMINHOS_METRICAS = {
    "Anos Iniciais": os.path.join(PASTA_MODELOS, "metricas_anos_iniciais.pkl"),
    "Anos Finais": os.path.join(PASTA_MODELOS, "metricas_anos_finais.pkl")
}


# ============================================================
# ESTILO VISUAL DA APLICAÇÃO
# ============================================================
#
# O CSS abaixo define uma identidade visual mais limpa, com fundo
# claro, cards brancos, título em azul escuro e componentes com
# cantos arredondados.
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
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

    .card-metrica {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #E5E7EB;
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
# FUNÇÃO DE LEITURA E TRATAMENTO DAS BASES
# ============================================================

@st.cache_data
def carregar_base(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega e trata a base de dados usada na aplicação.

    Parâmetros
    ----------
    caminho_arquivo : str
        Caminho do arquivo Excel.

    Retorno
    -------
    pd.DataFrame
        Base tratada.
    """

    # Verifica se a base existe no caminho esperado.
    if not os.path.exists(caminho_arquivo):
        st.error(
            f"A base de dados não foi encontrada em: {caminho_arquivo}. "
            "Verifique se o arquivo foi colocado corretamente na pasta 'data'."
        )
        st.stop()

    # Lê a base em Excel.
    df = pd.read_excel(caminho_arquivo)

    # Remove colunas SAEB específicas que não são utilizadas no modelo.
    colunas_remover = [
        "Nota SAEB em Matemática",
        "Nota SAEB em Língua Portuguesa"
    ]

    df.drop(
        columns=[col for col in colunas_remover if col in df.columns],
        inplace=True,
        errors="ignore"
    )

    # Garante que as colunas necessárias para filtragem existem.
    colunas_obrigatorias = [
        "IDEB",
        "Nota SAEB - Nota Média Padronizada (N)"
    ]

    for coluna in colunas_obrigatorias:
        if coluna not in df.columns:
            st.error(f"A coluna obrigatória '{coluna}' não foi encontrada na base.")
            st.stop()

    # Remove registros sem IDEB e sem Nota Média Padronizada.
    df = df[df[colunas_obrigatorias].notnull().all(axis=1)].copy()

    # Lista de variáveis com possível imputação por mediana.
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

    # Imputação por mediana nas variáveis disponíveis.
    for variavel in variaveis_para_imputar:
        if variavel in df.columns:
            mediana = df[variavel].median()
            df[variavel] = df[variavel].fillna(mediana)

    # Verifica se todas as variáveis do modelo existem na base.
    variaveis_ausentes = [
        var for var in VARIAVEIS_MODELO if var not in df.columns
    ]

    if variaveis_ausentes:
        st.error(
            "As seguintes variáveis do modelo não foram encontradas na base: "
            + ", ".join(variaveis_ausentes)
        )
        st.stop()

    # Remove linhas sem valores nas variáveis do modelo ou no IDEB.
    df = df.dropna(subset=VARIAVEIS_MODELO + ["IDEB"]).copy()

    return df


# ============================================================
# FUNÇÃO PARA CARREGAR MODELO
# ============================================================

@st.cache_resource
def carregar_modelo(caminho_modelo: str):
    """
    Carrega o modelo treinado salvo em formato pkl.

    Parâmetros
    ----------
    caminho_modelo : str
        Caminho do arquivo do modelo.

    Retorno
    -------
    object
        Modelo treinado.
    """

    if not os.path.exists(caminho_modelo):
        st.error(
            f"O modelo não foi encontrado em: {caminho_modelo}. "
            "Execute primeiro o script 'treinar_modelos.py'."
        )
        st.stop()

    return joblib.load(caminho_modelo)


# ============================================================
# FUNÇÃO PARA CARREGAR MÉTRICAS
# ============================================================

@st.cache_data
def carregar_metricas(caminho_metricas: str) -> dict:
    """
    Carrega as métricas salvas do modelo.

    Parâmetros
    ----------
    caminho_metricas : str
        Caminho do arquivo de métricas.

    Retorno
    -------
    dict
        Dicionário com métricas de avaliação.
    """

    if not os.path.exists(caminho_metricas):
        st.error(
            f"As métricas não foram encontradas em: {caminho_metricas}. "
            "Execute primeiro o script 'treinar_modelos.py'."
        )
        st.stop()

    return joblib.load(caminho_metricas)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def formatar_numero(valor: float, casas: int = 4) -> str:
    """
    Formata números no padrão brasileiro.

    Parâmetros
    ----------
    valor : float
        Valor numérico a ser formatado.

    casas : int
        Número de casas decimais.

    Retorno
    -------
    str
        Valor formatado com vírgula decimal.
    """

    if pd.isna(valor):
        return ""

    return f"{valor:.{casas}f}".replace(".", ",")


def obter_coluna_municipio(df: pd.DataFrame) -> str:
    """
    Identifica a coluna de município na base.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base de dados.

    Retorno
    -------
    str
        Nome da coluna de município.
    """

    possiveis_colunas = [
        "Nome do Município",
        "Município",
        "municipio",
        "nome_municipio",
        "Nome Município"
    ]

    for coluna in possiveis_colunas:
        if coluna in df.columns:
            return coluna

    # Caso nenhuma coluna seja encontrada, a aplicação cria uma
    # identificação genérica para evitar erro.
    df["Município"] = [f"Município {i + 1}" for i in range(df.shape[0])]
    return "Município"


def criar_tabela_metricas(metricas: dict) -> pd.DataFrame:
    """
    Cria uma tabela organizada com as métricas do modelo.

    Parâmetros
    ----------
    metricas : dict
        Dicionário de métricas salvo no treinamento.

    Retorno
    -------
    pd.DataFrame
        Tabela formatada com métricas.
    """

    tabela = pd.DataFrame(
        [
            {
                "Conjunto": "Treinamento",
                "RMSE": metricas["rmse_train"],
                "MAE": metricas["mae_train"],
                "R²": metricas["r2_train"]
            },
            {
                "Conjunto": "Teste",
                "RMSE": metricas["rmse_test"],
                "MAE": metricas["mae_test"],
                "R²": metricas["r2_test"]
            },
            {
                "Conjunto": "Validação cruzada",
                "RMSE": f"{metricas['rmse_cv']:.4f} ± {metricas['rmse_std']:.4f}",
                "MAE": f"{metricas['mae_cv']:.4f} ± {metricas['mae_std']:.4f}",
                "R²": f"{metricas['r2_cv']:.4f} ± {metricas['r2_std']:.4f}"
            }
        ]
    )

    return tabela


def aplicar_alteracoes_cenario(
    X_base: pd.DataFrame,
    alteracoes: list
) -> pd.DataFrame:
    """
    Aplica as alterações percentuais definidas pelo usuário.

    Parâmetros
    ----------
    X_base : pd.DataFrame
        Matriz original de variáveis explicativas.

    alteracoes : list
        Lista de dicionários contendo variável, operação e percentual.

    Retorno
    -------
    pd.DataFrame
        Matriz alterada conforme o cenário.
    """

    # Cria uma cópia para preservar a base original.
    X_cenario = X_base.copy()

    # Aplica cada alteração acumulada no cenário.
    for item in alteracoes:
        variavel = item["Variável"]
        operacao = item["Operação"]
        percentual = item["Percentual"]

        # Define o fator multiplicativo.
        if operacao == "Aumento":
            fator = 1 + percentual / 100
        else:
            fator = 1 - percentual / 100

        # Aplica a alteração na variável selecionada.
        X_cenario[variavel] = X_cenario[variavel] * fator

        # Regra de segurança para variáveis proporcionais de formação docente.
        # Como são proporções, não devem ultrapassar 1.
        if "form docente" in variavel.lower():
            X_cenario[variavel] = X_cenario[variavel].clip(upper=1)

        # Evita valores negativos em variáveis quantitativas após reduções.
        X_cenario[variavel] = X_cenario[variavel].clip(lower=0)

    return X_cenario


# ============================================================
# ESTADO DA SESSÃO
# ============================================================
#
# O Streamlit reexecuta o script a cada interação.
# Por isso, o session_state é usado para guardar os cenários
# e as variáveis adicionadas pelo usuário.
# ============================================================

if "alteracoes_atuais" not in st.session_state:
    st.session_state.alteracoes_atuais = []

if "resultados_cenarios" not in st.session_state:
    st.session_state.resultados_cenarios = []

if "resultado_municipal" not in st.session_state:
    st.session_state.resultado_municipal = pd.DataFrame()


# ============================================================
# CABEÇALHO DA APLICAÇÃO
# ============================================================

st.markdown(
    '<div class="titulo-principal">Simulador de Cenários Educacionais para o IDEB Municipal</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitulo">'
    'Aplicação baseada em modelo XGBoost para simulação de alterações hipotéticas '
    'em variáveis associadas ao IDEB municipal.'
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
            list(CAMINHOS_BASES.keys())
        )

    with col_info:
        st.info(
            "A simulação utiliza apenas os registros de 2023. "
            "As alterações são aplicadas às variáveis explicativas e o modelo estima "
            "o IDEB médio resultante do cenário."
        )

    # Carrega a base, o modelo e as métricas da etapa selecionada.
    df = carregar_base(CAMINHOS_BASES[etapa])
    modelo = carregar_modelo(CAMINHOS_MODELOS[etapa])
    metricas = carregar_metricas(CAMINHOS_METRICAS[etapa])

    # Filtra os dados de 2023.
    if "Ano" not in df.columns:
        st.error("A coluna 'Ano' não foi encontrada na base.")
        st.stop()

    df_2023 = df[df["Ano"] == 2023].copy()

    if df_2023.empty:
        st.error("Não foram encontrados registros referentes ao ano de 2023 na base.")
        st.stop()

    st.markdown("### Variáveis do cenário")

    col1, col2, col3 = st.columns([2.2, 1, 1])

    with col1:
        variavel = st.selectbox(
            "Variável a alterar",
            VARIAVEIS_MODELO
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
        nova_alteracao = {
            "Variável": variavel,
            "Operação": operacao,
            "Percentual": percentual
        }

        # Evita duplicidade da mesma variável no mesmo cenário.
        variaveis_ja_adicionadas = [
            item["Variável"] for item in st.session_state.alteracoes_atuais
        ]

        if variavel in variaveis_ja_adicionadas:
            st.warning("Essa variável já foi adicionada ao cenário atual.")
        else:
            st.session_state.alteracoes_atuais.append(nova_alteracao)
            st.success("Variável adicionada ao cenário.")

    if limpar:
        st.session_state.alteracoes_atuais = []
        st.success("Lista de variáveis do cenário limpa.")

    if st.session_state.alteracoes_atuais:
        st.markdown("#### Alterações adicionadas ao cenário")
        st.dataframe(
            pd.DataFrame(st.session_state.alteracoes_atuais),
            use_container_width=True
        )
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
            # Matriz original das variáveis explicativas em 2023.
            X_base = df_2023[VARIAVEIS_MODELO].copy()

            # Matriz alterada conforme as variáveis adicionadas.
            X_cenario = aplicar_alteracoes_cenario(
                X_base,
                st.session_state.alteracoes_atuais
            )

            # Predição sem alteração.
            pred_sem_alteracao = modelo.predict(X_base)

            # Predição com alterações do cenário.
            pred_cenario = modelo.predict(X_cenario)

            # IDEB real médio de 2023.
            ideb_real_medio = df_2023["IDEB"].mean()

            # IDEB previsto sem alteração.
            ideb_previsto_sem_alteracao = pred_sem_alteracao.mean()

            # IDEB previsto no cenário.
            ideb_previsto_cenario = pred_cenario.mean()

            # Diferenças estimadas.
            delta_previsto = ideb_previsto_cenario - ideb_previsto_sem_alteracao
            delta_real = ideb_previsto_cenario - ideb_real_medio

            # Descrição das variáveis alteradas.
            descricao_variaveis = "; ".join(
                [
                    f"{item['Variável']} ({item['Operação']} de {item['Percentual']:.1f}%)"
                    for item in st.session_state.alteracoes_atuais
                ]
            )

            # Resultado consolidado do cenário.
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

            # Armazena o resultado geral na sessão.
            st.session_state.resultados_cenarios.append(resultado_geral)

            # Identifica a coluna de município.
            coluna_municipio = obter_coluna_municipio(df_2023)

            # Cria tabela municipal.
            tabela_municipal = df_2023[[coluna_municipio, "IDEB"]].copy()
            tabela_municipal = tabela_municipal.rename(
                columns={
                    coluna_municipio: "Município",
                    "IDEB": "IDEB real em 2023"
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

            # Guarda a tabela municipal da última simulação.
            st.session_state.resultado_municipal = tabela_municipal.copy()

            # Limpa as alterações atuais após gerar o cenário.
            st.session_state.alteracoes_atuais = []

            st.success("Cenário gerado com sucesso.")
            st.dataframe(
                pd.DataFrame([resultado_geral]),
                use_container_width=True
            )


# ============================================================
# ABA 2: RESULTADOS GERAIS
# ============================================================

with aba_resultados:
    st.subheader("Resultados gerais dos cenários simulados")

    if not st.session_state.resultados_cenarios:
        st.info("Nenhum cenário foi gerado ainda.")
    else:
        df_resultados = pd.DataFrame(st.session_state.resultados_cenarios)

        st.dataframe(
            df_resultados,
            use_container_width=True
        )

        st.markdown("### Indicadores do último cenário gerado")

        ultimo = df_resultados.iloc[-1]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric(
                "IDEB real médio em 2023",
                formatar_numero(ultimo["IDEB real médio em 2023"])
            )

        with col_m2:
            st.metric(
                "Previsto sem alteração",
                formatar_numero(ultimo["IDEB previsto sem alteração"])
            )

        with col_m3:
            st.metric(
                "Previsto no cenário",
                formatar_numero(ultimo["IDEB previsto no cenário"])
            )

        with col_m4:
            st.metric(
                "Diferença estimada",
                formatar_numero(
                    ultimo["Diferença em relação ao previsto sem alteração"]
                )
            )

        st.markdown("### IDEB médio previsto por cenário")

        fig_ideb = px.bar(
            df_resultados,
            x="Cenário",
            y="IDEB previsto no cenário",
            color="Etapa",
            text="IDEB previsto no cenário",
            title="IDEB médio previsto por cenário"
        )

        fig_ideb.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

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

        fig_delta.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

        fig_delta.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font_color="#23406E",
            xaxis_title="Cenário",
            yaxis_title="Diferença estimada",
            legend_title="Etapa de ensino"
        )

        fig_delta.add_hline(
            y=0,
            line_width=1,
            line_color="black"
        )

        st.plotly_chart(fig_delta, use_container_width=True)

        # Exportação dos resultados gerais.
        csv_resultados = df_resultados.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

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

        st.dataframe(
            tabela_municipal,
            use_container_width=True
        )

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

        fig_pos.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

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

        fig_neg.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

        fig_neg.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font_color="#23406E",
            xaxis_title="Diferença estimada",
            yaxis_title="Município",
            yaxis={"categoryorder": "total descending"}
        )

        st.plotly_chart(fig_neg, use_container_width=True)

        # Exportação da tabela municipal.
        csv_municipal = tabela_municipal.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

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
    st.subheader("Métricas de avaliação do modelo")

    etapa_metricas = st.selectbox(
        "Selecione a etapa de ensino",
        list(CAMINHOS_METRICAS.keys()),
        key="etapa_metricas"
    )

    metricas_etapa = carregar_metricas(CAMINHOS_METRICAS[etapa_metricas])

    tabela_metricas = criar_tabela_metricas(metricas_etapa)

    st.dataframe(
        tabela_metricas,
        use_container_width=True
    )

    st.markdown("### Informações da base utilizada no treinamento")

    col_base1, col_base2 = st.columns(2)

    with col_base1:
        st.metric(
            "Número de registros",
            int(metricas_etapa["n_linhas"])
        )

    with col_base2:
        st.metric(
            "Número de colunas",
            int(metricas_etapa["n_colunas"])
        )

    st.caption(
        "As métricas foram calculadas no treinamento do modelo. "
        "A validação cruzada foi realizada com 5 folds no conjunto de treinamento."
    )


# ============================================================
# ABA 5: NOTAS METODOLÓGICAS
# ============================================================

with aba_metodologia:
    st.subheader("Notas metodológicas da aplicação")

    st.markdown(
        """
        Esta aplicação foi construída para operacionalizar a simulação de cenários
        descrita no artigo. O procedimento parte de modelos preditivos treinados
        com dados educacionais, socioeconômicos e territoriais agregados em nível
        municipal.

        A simulação é realizada apenas sobre os registros de 2023. O usuário seleciona
        uma ou mais variáveis explicativas, define o percentual de aumento ou redução
        e o sistema recalcula o IDEB previsto pelo modelo para esse novo cenário.

        A diferença entre o IDEB previsto sem alteração e o IDEB previsto no cenário
        representa uma variação estimada pelo modelo. Essa diferença não deve ser
        interpretada como efeito causal, pois a aplicação não identifica mecanismos
        causais, não controla confundimento por desenho causal e não estima efeitos
        de tratamento.

        A utilidade da aplicação está na análise exploratória e na avaliação de
        sensibilidade do modelo. Portanto, os resultados podem apoiar a formulação
        de hipóteses, a comparação de cenários hipotéticos e a visualização de
        possíveis respostas do modelo diante de alterações nas variáveis.
        """
    )

    st.markdown("### Variáveis disponíveis para simulação")

    tabela_variaveis = pd.DataFrame(
        {
            "Variável": VARIAVEIS_MODELO,
            "Descrição operacional": [
                "Indicador econômico municipal utilizado como proxy de contexto socioeconômico.",
                "Indicador educacional associado à proporção de estudantes em defasagem idade-série.",
                "Indicador de adequação da formação docente.",
                "Indicador territorial e produtivo relacionado à área agrícola.",
                "Indicador econômico associado à produção na extração vegetal.",
                "Valor repassado no âmbito do Programa Criança Feliz."
            ]
        }
    )

    st.dataframe(
        tabela_variaveis,
        use_container_width=True
    )

    st.markdown(
        """
        A interpretação dos cenários deve considerar as limitações dos dados, a agregação
        municipal, a qualidade das variáveis disponíveis e o fato de que modelos preditivos
        capturam padrões estatísticos presentes na base, mas não substituem análise
        educacional, institucional e territorial.
        """
    )