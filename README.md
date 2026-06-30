# Simulador de Cenários Educacionais para o IDEB Municipal

Esta aplicação web permite simular cenários hipotéticos para o IDEB municipal a partir do artefato final do modelo selecionado no artigo. A ferramenta foi desenvolvida em Streamlit e utiliza o modelo final salvo pelo notebook metodológico, preservando o mesmo pré-processamento aplicado na modelagem, incluindo imputação, remoção de variáveis com variância zero e seleção final de variáveis.

A aplicação não realiza previsão futura nem inferência causal. Os resultados devem ser interpretados como simulações condicionais do modelo, úteis para análise exploratória e avaliação de sensibilidade.

## 1. Objetivo da aplicação

A aplicação foi construída para operacionalizar a simulação de cenários descrita no artigo. O usuário pode alterar percentualmente uma ou mais variáveis explicativas do modelo e observar a resposta estimada no IDEB médio municipal de 2023.

A diferença entre o IDEB previsto sem alteração e o IDEB previsto no cenário representa uma variação estimada pelo modelo. Essa diferença não deve ser interpretada como efeito causal direto das variáveis sobre o IDEB.

## 2. Modelo utilizado

O simulador utiliza o artefato final exportado pelo notebook metodológico do artigo. O artefato deve conter, no mínimo:

- modelo final treinado;
- nome do modelo selecionado;
- conjunto final de variáveis;
- lista de variáveis originais usadas no treinamento;
- imputador ajustado no conjunto de treino;
- seletor de variância ajustado no conjunto de treino;
- métricas do modelo final.

No artigo, o modelo final foi selecionado a partir de comparação entre diferentes algoritmos e subconjuntos de variáveis. A escolha considerou desempenho preditivo, validação cruzada, reavaliação dos finalistas com RepeatedKFold, teste de Friedman, comparações pareadas com Wilcoxon e correção de Holm, além de parcimônia.

## 3. Etapas de ensino contempladas

A aplicação pode contemplar:

1. Anos Iniciais do Ensino Fundamental;
2. Anos Finais do Ensino Fundamental.

Para que uma etapa apareça no sistema, a respectiva base de dados e o respectivo artefato final precisam estar disponíveis nas pastas esperadas. Caso apenas o artefato de Anos Iniciais esteja disponível, a aplicação exibirá apenas essa etapa.

## 4. Estrutura da pasta

```text
simulador_ideb/
│
├── app.py
├── treinar_modelos.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── logo_simulador_ideb.png
│
├── data/
│   ├── base_anos_iniciais.xlsx
│   └── base_anos_finais.xlsx
│
├── models/
│   ├── artefato_modelo_final_anos_iniciais.pkl
│   ├── artefato_modelo_final_anos_finais.pkl
│   └── artefato_modelo_final_ideb.pkl
│
└── outputs/
    └── resumo_artefatos_simulador.xlsx
```

Observação: para Anos Iniciais, a aplicação também aceita o nome genérico `artefato_modelo_final_ideb.pkl`, caso esse tenha sido o arquivo exportado pelo notebook.

## 5. Preparação dos arquivos

Antes de executar a aplicação, coloque as bases na pasta `data/` e os artefatos finais na pasta `models/`.

Exemplo mínimo para Anos Iniciais:

```text
data/base_anos_iniciais.xlsx
models/artefato_modelo_final_ideb.pkl
```

Exemplo com duas etapas:

```text
data/base_anos_iniciais.xlsx
models/artefato_modelo_final_anos_iniciais.pkl

data/base_anos_finais.xlsx
models/artefato_modelo_final_anos_finais.pkl
```

## 6. Instalação

Crie e ative um ambiente virtual, depois instale as dependências:

```bash
pip install -r requirements.txt
```

## 7. Validação dos artefatos

O arquivo `treinar_modelos.py` não treina novamente o modelo. Ele valida se os artefatos finais exportados pelo notebook metodológico estão completos e se podem ser usados pela aplicação.

Execute:

```bash
python treinar_modelos.py
```

Se os artefatos estiverem corretos, será criado o arquivo:

```text
outputs/resumo_artefatos_simulador.xlsx
```

## 8. Execução da aplicação

Execute:

```bash
streamlit run app.py
```

A aplicação abrirá no navegador. Na aba **Simulação**, selecione a etapa de ensino, escolha uma variável, defina aumento ou redução percentual e gere o cenário.

## 9. Variáveis disponíveis para simulação

As variáveis exibidas no simulador são carregadas automaticamente do artefato final, a partir da chave `variaveis_modelo`. Na interface, os nomes técnicos são apresentados em formato textual, sem sublinhados, para facilitar a leitura.

Exemplo:

```text
taxa_distorcao_idade_serie
```

aparece como:

```text
Taxa distorcao idade serie
```

## 10. Interpretação dos resultados

A aplicação apresenta:

- IDEB real médio em 2023;
- IDEB previsto sem alteração;
- IDEB previsto no cenário;
- diferença estimada em relação ao previsto sem alteração;
- diferença estimada em relação ao IDEB real;
- resultados municipais do último cenário gerado;
- gráficos com variações positivas e negativas por município.

Esses resultados devem ser lidos como respostas condicionais do modelo a alterações hipotéticas nas variáveis de entrada. Eles não indicam causalidade, nem substituem análise educacional, institucional e territorial.

## 11. Limitações

A aplicação depende da qualidade dos dados usados no treinamento, da agregação municipal das variáveis e das limitações próprias de modelos preditivos. O simulador permite explorar cenários, mas não estima efeitos causais nem garante que as alterações simuladas sejam operacionalmente viáveis em políticas públicas.
