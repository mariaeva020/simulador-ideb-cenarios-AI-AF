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
from html import escape

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

    .metric-card-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 1.2rem 0 1.4rem 0;
    }

    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
    }

    .metric-card-label {
        font-size: 0.85rem;
        color: #6B7280;
        margin-bottom: 0.35rem;
    }

    .metric-card-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1F2937;
    }

    .tabela-artigo table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
        font-size: 0.95rem;
    }

    .tabela-artigo th {
        background-color: #F3F4F6;
        color: #111827;
        font-weight: 700;
        text-align: center;
        border: 1px solid #D1D5DB;
        padding: 0.65rem;
    }

    .tabela-artigo td {
        border: 1px solid #D1D5DB;
        padding: 0.6rem;
        text-align: center;
        color: #111827;
    }

    .tabela-artigo .celula-modelo {
        vertical-align: middle;
        font-weight: 700;
        background-color: #FFFFFF;
    }

    .tabela-artigo td:first-child,
    .tabela-artigo td:nth-child(2) {
        font-weight: 600;
    }

    .nota-metodologica {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
    }

    .nota-metodologica h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #1F2937;
        font-size: 1.05rem;
    }

    .nota-metodologica p {
        color: #374151;
        line-height: 1.65;
        margin-bottom: 0;
        text-align: justify;
    }

    .alerta-metodologico {
        background-color: #FFF7E6;
        border-left: 6px solid #E3A72F;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #374151;
        line-height: 1.6;
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


# ======================================================
# DICIONÁRIO DE RÓTULOS INTERPRETÁVEIS DAS VARIÁVEIS
# Apenas variáveis presentes nos modelos finais do simulador
# ======================================================

import textwrap

ROTULOS_VARIAVEIS = {
    # Indicadores educacionais gerais
    "taxa_distorcao_idade_serie": "Taxa de distorção idade-série",
    "percentual_docente_curso_superior": "Percentual de docentes com curso superior",
    "media_horas_aula": "Média de horas-aula diária",
    "media_alunos_turma": "Média de alunos por turma",
    "quantidade_de_matriculas": "Quantidade de matrículas",

    # Adequação da formação docente
    "grupo_1_adeq_form_docente": "Professores com formação superior adequada à área lecionada",
    "grupo_2_adeq_form_docente": "Professores com bacharelado na disciplina, sem complementação pedagógica",
    "grupo_3_adeq_form_docente": "Professores com formação superior em área diferente da que lecionam",
    "grupo_5_adeq_form_docente": "Professores sem curso superior completo",

    # Esforço docente
    "nivel_3_esforco_docente": "Docentes em nível intermediário de esforço docente",
    "nivel_4_esforco_docente": "Docentes em nível intermediário-alto de esforço docente",
    "nivel_5_esforco_docente": "Docentes em nível elevado de esforço docente",

    # Complexidade da gestão escolar
    "nivel_1_gestao_escola": "Escolas com menor complexidade de gestão",
    "nivel_2_gestao_escola": "Escolas com baixa complexidade de gestão",
    "nivel_3_gestao_escola": "Escolas com complexidade intermediária de gestão",
    "nivel_4_gestao_escola": "Escolas com complexidade intermediária-alta de gestão",
    "nivel_5_gestao_escola": "Escolas com alta complexidade de gestão",

    # Regularidade docente
    "media_baixa_regularidade": "Média de docentes com baixa regularidade na escola",
    "media_alta_regularidade": "Média de docentes com alta regularidade na escola",

    # Infraestrutura escolar
    "qt_salas_utiliza_climatizadas": "Quantidade de salas utilizadas climatizadas",
    "qt_desktop_aluno": "Computadores desktop disponíveis para alunos",
    "qt_escolas_com_agua_potavel": "Escolas com água potável",
    "qt_escolas_com_acessibilidade_rampas": "Escolas com rampas de acessibilidade",
    "qt_escolas_com_orgao_conselho_escolar": "Escolas com conselho escolar",

    # Profissionais escolares
    "qt_prof_pedagogia": "Profissionais de pedagogia",
    "qt_prof_secretario": "Profissionais com função de secretário escolar",

    # Variáveis econômicas
    "pib_per_capita": "PIB per capita municipal",
    "area_colhida_lavour": "Área colhida de lavouras",
    "valor_da_producao_na_extracao_vegetal": "Valor da produção na extração vegetal",
    "valor_da_producao_prod_origem_animal": "Valor da produção de origem animal",

    # Receitas e transferências municipais
    "iptu": "Arrecadação de IPTU",
    "cota_parte_icms": "Cota-parte do ICMS",
    "cota_parte_ipva": "Cota-parte do IPVA",
    "cota_parte_ipi_exp": "Cota-parte do IPI-Exportação",
    "pnate": "Transferências do PNATE",

    # Fundeb e despesas educacionais
    "valor_aplicado_em_mde": "Valor aplicado em manutenção e desenvolvimento do ensino",
    "receita_da_aplicacao_financeira_do_fundeb": "Receita de aplicação financeira do Fundeb",
    "receitas_destinadas_ao_fundeb_fundo_estadual": "Receitas destinadas ao Fundeb estadual",
    "contribuicao_na_formacao_do_fundef_fundeb_–_destinada": "Contribuição destinada à formação do Fundef/Fundeb",
    "creche": "Despesas associadas à creche",
    "pre_escola": "Despesas associadas à pré-escola",

    # Assistência social
    "valor_repassado_crianca_feliz": "Repasses do Programa Criança Feliz",
    "valor_repassado_protecao_social_basica": "Repasses à proteção social básica",
    "valor_repassado_gestao_suas": "Repasses para gestão do SUAS",
}


def obter_rotulo_variavel(nome_variavel):
    """
    Retorna o rótulo interpretável da variável.
    Caso a variável não esteja no dicionário, aplica uma formatação simples.
    """
    return ROTULOS_VARIAVEIS.get(
        nome_variavel,
        str(nome_variavel).replace("_", " ").strip().capitalize()
    )


def nome_variavel_texto(nome_variavel):
    """
    Função usada na interface do simulador para exibir nomes legíveis.
    Mantém o mesmo padrão de rótulos adotado no artigo.
    """
    return obter_rotulo_variavel(nome_variavel)


def quebrar_rotulo(texto, largura=48):
    """
    Quebra rótulos longos para melhorar a leitura em tabelas e gráficos.
    """
    return "\n".join(textwrap.wrap(str(texto), width=largura))


def obter_coluna_existente(df: pd.DataFrame, candidatas: list):
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


def localizar_artefato(caminhos_possiveis: list):
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


def valor_disponivel(valor) -> bool:
    """Verifica se um valor está disponível para exibição."""
    try:
        return valor is not None and not pd.isna(valor)
    except Exception:
        return False


def formatar_media_desvio(media, desvio=None, casas: int = 4) -> str:
    """Formata média e desvio-padrão no padrão de tabela acadêmica."""
    if not valor_disponivel(media):
        return "Não disponível"

    media_formatada = formatar_numero(media, casas)

    if valor_disponivel(desvio):
        desvio_formatado = formatar_numero(desvio, casas)
        return f"{media_formatada} ± {desvio_formatado}"

    return media_formatada


def obter_metrica_final(metricas_modelo_final, base: str, metrica: str):
    """Recupera MAE, RMSE ou R² da tabela final de treino/teste."""
    if not isinstance(metricas_modelo_final, pd.DataFrame):
        return np.nan

    if metricas_modelo_final.empty:
        return np.nan

    linha = metricas_modelo_final[
        metricas_modelo_final["base"].astype(str).str.lower() == base.lower()
    ]

    if linha.empty:
        return np.nan

    return linha.iloc[0].get(metrica, np.nan)


def criar_tabela_desempenho_modelo(artefato: dict) -> pd.DataFrame:
    """
    Cria tabela de desempenho no formato acadêmico:
    Modelo | Métrica | Treino | Teste | Validação Cruzada Inicial | RepeatedKFold.
    """
    modelo_final_escolhido = artefato.get("modelo_final_escolhido", {})
    metricas_modelo_final = artefato.get("metricas_modelo_final", None)
    resumo_repeated = artefato.get("resumo_repeated", None)

    if isinstance(modelo_final_escolhido, pd.Series):
        modelo_final_escolhido = modelo_final_escolhido.to_dict()

    nome_modelo = artefato.get("nome_modelo", "Não disponível")

    metricas = [
        {
            "codigo": "R2",
            "rotulo": "R²",
            "cv_media": "R2_cv",
            "cv_desvio": "R2_cv_desvio",
            "rep_media": "R2_repeated_medio",
            "rep_desvio": None,
        },
        {
            "codigo": "MAE",
            "rotulo": "MAE",
            "cv_media": "MAE_cv",
            "cv_desvio": "MAE_cv_desvio",
            "rep_media": "MAE_repeated_medio",
            "rep_desvio": None,
        },
        {
            "codigo": "RMSE",
            "rotulo": "RMSE",
            "cv_media": "RMSE_cv",
            "cv_desvio": "RMSE_cv_desvio",
            "rep_media": "RMSE_repeated_medio",
            "rep_desvio": "RMSE_repeated_desvio",
        },
    ]

    repeated_disponivel = False

    if isinstance(resumo_repeated, pd.DataFrame) and not resumo_repeated.empty:
        colunas_repeated = [
            "RMSE_repeated_medio",
            "MAE_repeated_medio",
            "R2_repeated_medio"
        ]

        for coluna in colunas_repeated:
            if coluna in resumo_repeated.columns and resumo_repeated[coluna].notna().any():
                repeated_disponivel = True

    linhas = []

    for i, item in enumerate(metricas):
        codigo = item["codigo"]

        treino = obter_metrica_final(metricas_modelo_final, "Treino", codigo)
        teste = obter_metrica_final(metricas_modelo_final, "Teste", codigo)

        cv_media = modelo_final_escolhido.get(item["cv_media"], np.nan)
        cv_desvio = modelo_final_escolhido.get(item["cv_desvio"], np.nan)

        linha = {
            "Modelo": nome_modelo if i == 0 else "",
            "Métrica": item["rotulo"],
            "Treino": formatar_numero(treino),
            "Teste": formatar_numero(teste),
            "Validação cruzada inicial": formatar_media_desvio(cv_media, cv_desvio),
        }

        if repeated_disponivel:
            rep_media = modelo_final_escolhido.get(item["rep_media"], np.nan)
            rep_desvio = modelo_final_escolhido.get(item["rep_desvio"], np.nan) if item["rep_desvio"] else np.nan
            linha["RepeatedKFold"] = formatar_media_desvio(rep_media, rep_desvio)

        linhas.append(linha)

    return pd.DataFrame(linhas)


def renderizar_tabela_artigo(tabela: pd.DataFrame):
    """Renderiza uma tabela com aparência mais próxima de tabela acadêmica."""
    html = tabela.to_html(index=False, escape=False)

    st.markdown(
        f"""
        <div class="tabela-artigo">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )

def renderizar_tabela_desempenho_artigo(tabela: pd.DataFrame):
    """
    Renderiza a tabela de desempenho com a coluna 'Modelo' mesclada
    nas linhas das métricas R², MAE e RMSE.
    """
    if tabela.empty:
        st.info("Não há métricas disponíveis para exibição.")
        return

    colunas = list(tabela.columns)
    nome_modelo = tabela["Modelo"].replace("", np.nan).dropna().iloc[0]
    numero_linhas = len(tabela)

    html = """
    <div class="tabela-artigo">
        <table>
            <thead>
                <tr>
    """

    for coluna in colunas:
        html += f"<th>{escape(str(coluna))}</th>"

    html += """
                </tr>
            </thead>
            <tbody>
    """

    for indice, (_, linha) in enumerate(tabela.iterrows()):
        html += "<tr>"

        if indice == 0:
            html += (
                f'<td rowspan="{numero_linhas}" class="celula-modelo">'
                f"{escape(str(nome_modelo))}"
                "</td>"
            )

        for coluna in colunas:
            if coluna == "Modelo":
                continue

            html += f"<td>{escape(str(linha[coluna]))}</td>"

        html += "</tr>"

    html += """
            </tbody>
        </table>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def renderizar_cartoes_modelo(artefato: dict):
    """Exibe cartões sintéticos com as principais informações do modelo."""
    modelo_final_escolhido = artefato.get("modelo_final_escolhido", {})

    if isinstance(modelo_final_escolhido, pd.Series):
        modelo_final_escolhido = modelo_final_escolhido.to_dict()

    nome_modelo = artefato.get("nome_modelo", "Não disponível")
    n_variaveis = len(artefato.get("variaveis_modelo", []))
    rmse_teste = modelo_final_escolhido.get("RMSE_teste", np.nan)
    r2_teste = modelo_final_escolhido.get("R2_teste", np.nan)

    st.markdown(
        f"""
        <div class="metric-card-grid">
            <div class="metric-card">
                <div class="metric-card-label">Modelo final</div>
                <div class="metric-card-value">{nome_modelo}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Número de variáveis</div>
                <div class="metric-card-value">{n_variaveis}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">RMSE no teste</div>
                <div class="metric-card-value">{formatar_numero(rmse_teste)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">R² no teste</div>
                <div class="metric-card-value">{formatar_numero(r2_teste)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ESTADO DA SESSÃO
# ============================================================

if "alteracoes_atuais" not in st.session_state:
    st.session_state.alteracoes_atuais = []

if "resultados_cenarios" not in st.session_state:
    st.session_state.resultados_cenarios = []

if "resultado_municipal" not in st.session_state:
    st.session_state.resultado_municipal = pd.DataFrame()

if "resultados_municipais_cenarios" not in st.session_state:
    st.session_state.resultados_municipais_cenarios = []


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
            st.session_state.resultados_municipais_cenarios.append(tabela_municipal.copy())
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
# ============================================================
# ABA 3: RESULTADOS MUNICIPAIS
# ============================================================

with aba_municipios:
    st.subheader("Resultados municipais dos cenários simulados")

    if "resultados_municipais_cenarios" not in st.session_state:
        st.session_state.resultados_municipais_cenarios = []

    if not st.session_state.resultados_municipais_cenarios:
        st.info("Nenhum resultado municipal foi gerado ainda.")
    else:
        tabela_municipal = pd.concat(
            st.session_state.resultados_municipais_cenarios,
            ignore_index=True
        )

        st.markdown("### Tabela municipal consolidada")

        col_filtro1, col_filtro2 = st.columns([1, 2])

        with col_filtro1:
            etapas_disponiveis_municipal = ["Todas"] + sorted(
                tabela_municipal["Etapa"].dropna().unique().tolist()
            )

            filtro_etapa_municipal = st.selectbox(
                "Filtrar por etapa de ensino",
                etapas_disponiveis_municipal,
                key="filtro_etapa_municipal"
            )

        with col_filtro2:
            cenarios_disponiveis_municipal = ["Todos"] + sorted(
                tabela_municipal["Cenário"].dropna().unique().tolist()
            )

            filtro_cenario_municipal = st.selectbox(
                "Filtrar por cenário",
                cenarios_disponiveis_municipal,
                key="filtro_cenario_municipal"
            )

        tabela_municipal_filtrada = tabela_municipal.copy()

        if filtro_etapa_municipal != "Todas":
            tabela_municipal_filtrada = tabela_municipal_filtrada[
                tabela_municipal_filtrada["Etapa"] == filtro_etapa_municipal
            ]

        if filtro_cenario_municipal != "Todos":
            tabela_municipal_filtrada = tabela_municipal_filtrada[
                tabela_municipal_filtrada["Cenário"] == filtro_cenario_municipal
            ]

        st.dataframe(
            tabela_municipal_filtrada,
            width="stretch",
            hide_index=True
        )

        st.markdown("### Municípios com maiores variações positivas")

        ranking_positivo = tabela_municipal_filtrada.sort_values(
            by="Diferença em relação ao previsto sem alteração",
            ascending=False
        ).head(15)

        fig_pos = px.bar(
            ranking_positivo,
            x="Diferença em relação ao previsto sem alteração",
            y="Município",
            color="Cenário",
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
            legend_title="Cenário",
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(fig_pos, width="stretch")

        st.markdown("### Municípios com maiores variações negativas")

        ranking_negativo = tabela_municipal_filtrada.sort_values(
            by="Diferença em relação ao previsto sem alteração",
            ascending=True
        ).head(15)

        fig_neg = px.bar(
            ranking_negativo,
            x="Diferença em relação ao previsto sem alteração",
            y="Município",
            color="Cenário",
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
            legend_title="Cenário",
            yaxis={"categoryorder": "total descending"}
        )

        st.plotly_chart(fig_neg, width="stretch")

        csv_municipal = tabela_municipal.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            label="Baixar resultados municipais consolidados em CSV",
            data=csv_municipal,
            file_name="resultados_municipais_cenarios_ideb.csv",
            mime="text/csv"
        )


# ============================================================
# ABA 4: MÉTRICAS DO MODELO
# ============================================================

with aba_metricas:
    st.subheader("Desempenho do modelo final")

    etapa_metricas = st.selectbox(
        "Selecione a etapa de ensino",
        list(ETAPAS_DISPONIVEIS.keys()),
        key="etapa_metricas"
    )

    artefato_metricas = carregar_artefato(
        ETAPAS_DISPONIVEIS[etapa_metricas]["artefato"]
    )

    renderizar_cartoes_modelo(artefato_metricas)

    st.markdown("#### Tabela de desempenho preditivo")

    tabela_desempenho = criar_tabela_desempenho_modelo(artefato_metricas)
    renderizar_tabela_desempenho_artigo(tabela_desempenho)
    st.markdown(
        """
        <div class="alerta-metodologico">
        As métricas apresentadas correspondem ao artefato final carregado pela aplicação.
        O conjunto de teste foi mantido separado durante o treinamento e a validação cruzada
        foi calculada apenas sobre os dados de treino, conforme o procedimento metodológico
        adotado no estudo.
        </div>
        """,
        unsafe_allow_html=True
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

    artefato_metodologia = carregar_artefato(
        ETAPAS_DISPONIVEIS[etapa_metodologia]["artefato"]
    )

    nome_modelo_metodologia = artefato_metodologia.get("nome_modelo", "Não disponível")
    variaveis_modelo_metodologia = artefato_metodologia["variaveis_modelo"]
    n_variaveis_metodologia = len(variaveis_modelo_metodologia)

    st.markdown(
        f"""
        <div class="nota-metodologica">
            <h4>Escopo da aplicação</h4>
            <p>
            Esta aplicação operacionaliza o artefato computacional descrito no estudo,
            permitindo a simulação exploratória de cenários associados ao IDEB municipal.
            Para a etapa selecionada, o sistema utiliza o modelo final <strong>{nome_modelo_metodologia}</strong>,
            treinado com <strong>{n_variaveis_metodologia}</strong> variáveis explicativas e
            armazenado em artefato próprio para uso na interface web.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="nota-metodologica">
            <h4>Procedimento de simulação</h4>
            <p>
            A simulação é realizada sobre os registros observados de 2023. O usuário seleciona
            uma ou mais variáveis explicativas, define a operação de aumento ou redução e informa
            o percentual de modificação. Em seguida, a aplicação reaplica o mesmo pré-processamento
            utilizado no treinamento, incluindo imputação por mediana, remoção de variáveis de
            variância zero e seleção das variáveis efetivamente usadas pelo modelo final.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="nota-metodologica">
            <h4>Interpretação dos resultados</h4>
            <p>
            A diferença entre o IDEB previsto sem alteração e o IDEB previsto no cenário simulado
            representa uma resposta condicionada do modelo às alterações informadas. Essa diferença
            deve ser interpretada como variação preditiva estimada, e não como efeito causal.
            A aplicação não identifica mecanismos causais, não controla confundimento por desenho
            experimental ou quase-experimental e não estima efeitos de tratamento.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="alerta-metodologico">
        A ferramenta deve ser utilizada como apoio à análise exploratória e à formulação de hipóteses.
        Os resultados não substituem avaliação educacional, análise institucional, leitura territorial
        ou interpretação substantiva das políticas públicas envolvidas.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Variáveis disponíveis para simulação")

    tabela_variaveis = pd.DataFrame(
        {
            "Variável exibida no simulador": [
                nome_variavel_texto(v) for v in variaveis_modelo_metodologia
            ],
            "Nome técnico no artefato": variaveis_modelo_metodologia
        }
    )

 
    tabela_variaveis.insert(
        0,
        "Nº",
        range(1, len(tabela_variaveis) + 1)
    )

    renderizar_tabela_artigo(tabela_variaveis)