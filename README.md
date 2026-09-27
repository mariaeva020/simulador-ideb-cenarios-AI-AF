# Simulador de Cenários Educacionais para o IDEB Municipal

Este repositório contém a aplicação web desenvolvida em Streamlit para exploração de cenários hipotéticos associados ao IDEB municipal. O simulador utiliza os modelos de implantação definidos a partir do protocolo metodológico do estudo e preserva o pré-processamento empregado na modelagem, incluindo imputação, remoção de variáveis com variância zero e seleção do subconjunto final de variáveis.

A aplicação não realiza previsão futura nem inferência causal. Os resultados representam respostas condicionais dos modelos de implantação a alterações hipotéticas aplicadas aos registros observados de 2023 e devem ser interpretados exclusivamente em caráter exploratório.

## 1. Objetivo da aplicação

O simulador foi desenvolvido para operacionalizar a etapa de exploração de cenários apresentada no estudo. O usuário pode selecionar uma etapa de ensino, escolher uma ou mais variáveis disponíveis no modelo, definir alterações percentuais e observar a resposta estimada do IDEB sob essas modificações.

Para cada cenário, a aplicação apresenta:

- IDEB real médio observado em 2023;
- IDEB previsto sem alteração;
- IDEB previsto após as alterações simuladas;
- diferença entre o cenário e a previsão sem alteração;
- diferença entre o cenário e o IDEB real observado;
- resultados municipais;
- identificação de possíveis extrapolações em relação ao suporte empírico das variáveis.

A diferença entre o IDEB previsto sem alteração e o IDEB previsto no cenário corresponde a uma variação preditiva estimada pelo modelo. Essa diferença não deve ser interpretada como efeito causal da variável modificada sobre o IDEB.

## 2. Separação entre avaliação científica e implantação

O projeto distingue explicitamente dois momentos metodológicos.

O **modelo de avaliação científica** é aquele submetido aos procedimentos de validação descritos no artigo. As métricas de MAE, RMSE e R² apresentadas na aplicação pertencem a esse estágio.

O **modelo de implantação** é reajustado somente após o encerramento da avaliação independente, utilizando todos os dados disponíveis de 2013 a 2023 e a especificação final definida no protocolo metodológico. Esse modelo é utilizado exclusivamente para gerar as simulações disponibilizadas na aplicação.

Consequentemente, as métricas apresentadas no simulador não devem ser interpretadas como métricas calculadas sobre o modelo reajustado para implantação.

## 3. Modelos utilizados

A aplicação contempla duas etapas do Ensino Fundamental:

### Anos Iniciais

- algoritmo: XGBoost;
- estratégia de seleção: SHAP Top-30;
- número de variáveis no modelo avaliado: 30;
- número de variáveis no modelo de implantação: 30;
- formato do modelo de implantação: UBJ nativo do XGBoost (`modelo_implantacao.ubj`).

O formato UBJ é utilizado para o modelo XGBoost de implantação de modo a evitar dependência da serialização do objeto completo por `joblib`.

### Anos Finais

- algoritmo: CatBoost;
- estratégia de seleção: frequência em dois ou mais métodos;
- número de variáveis no modelo avaliado: 21;
- número de variáveis no modelo de implantação: 23;
- formato do modelo de implantação: `joblib` (`modelo_implantacao.joblib`).

## 4. Estrutura do repositório

```text
simulador_ideb/
│
├── app.py
├── gerar_artefatos_modelos_finais_por_etapa.py
├── treinar_modelos.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── logo_simulador_ideb.png
│
├── data/
│   ├── base_referencia_2023_anos_iniciais.csv
│   └── base_referencia_2023_anos_finais.csv
│
├── models/
│   ├── anos_iniciais/
│   │   ├── modelo_implantacao.ubj
│   │   ├── imputador_implantacao.joblib
│   │   ├── seletor_variancia_implantacao.joblib
│   │   ├── variaveis_pos_variancia.joblib
│   │   ├── variaveis_modelo.joblib
│   │   ├── colunas_entrada_modelo.joblib
│   │   ├── metadados.json
│   │   ├── resumo_tecnico_modelo_final.csv
│   │   └── suporte_empirico_variaveis.csv
│   │
│   └── anos_finais/
│       ├── modelo_implantacao.joblib
│       ├── imputador_implantacao.joblib
│       ├── seletor_variancia_implantacao.joblib
│       ├── variaveis_pos_variancia.joblib
│       ├── variaveis_modelo.joblib
│       ├── colunas_entrada_modelo.joblib
│       ├── metadados.json
│       ├── resumo_tecnico_modelo_final.csv
│       └── suporte_empirico_variaveis.csv
│
└── outputs/
    ├── resumo_integracao_simulador.csv
    ├── manifesto_artefatos_simulador.csv
    └── resumo_validacao_artefatos_simulador.csv
```

As pastas temporárias `artefatos_anos_iniciais/` e `artefatos_anos_finais/` podem ser utilizadas localmente como origem para a integração dos artefatos produzidos pelos notebooks, mas não constituem a estrutura operacional final da aplicação.

## 5. Bases de referência de 2023

O diretório `data/` contém uma base de referência para cada etapa de ensino:

```text
data/base_referencia_2023_anos_iniciais.csv
data/base_referencia_2023_anos_finais.csv
```

Essas bases são exportadas pelos respectivos notebooks metodológicos. Além das variáveis utilizadas como entrada do modelo, elas contêm o IDEB observado e a coluna:

```text
ideb_predito_referencia
```

Essa coluna armazena as predições produzidas pelo pipeline de implantação no momento da exportação e é utilizada posteriormente para verificar a reprodutibilidade dos artefatos instalados.

## 6. Organização dos artefatos finais

O script:

```bash
python gerar_artefatos_modelos_finais_por_etapa.py
```

não treina novamente os modelos e não refaz a seleção de variáveis.

Sua função é:

1. localizar os artefatos exportados pelos notebooks finais;
2. validar a presença dos arquivos necessários;
3. conferir a coerência dos metadados;
4. reconstruir o pipeline de predição;
5. comparar as predições recalculadas com `ideb_predito_referencia`;
6. copiar os artefatos operacionais para `models/<etapa>/`;
7. copiar as bases de referência para `data/`;
8. gerar arquivos de auditoria em `outputs/`.

Para uso desse script, as pastas exportadas pelos notebooks devem estar temporariamente disponíveis na raiz do projeto:

```text
artefatos_anos_iniciais/
artefatos_anos_finais/
```

Nos Anos Iniciais, o script espera o modelo XGBoost em:

```text
artefatos_anos_iniciais/modelo_implantacao.ubj
```

Nos Anos Finais, espera:

```text
artefatos_anos_finais/modelo_implantacao.joblib
```

## 7. Validação independente dos artefatos instalados

Apesar do nome histórico, o arquivo:

```text
treinar_modelos.py
```

**não realiza treinamento de modelos**.

Ele executa uma validação independente dos artefatos já instalados no repositório. A verificação inclui:

- existência das bases e dos arquivos necessários;
- coerência dos metadados;
- número esperado de variáveis;
- reprodução do pré-processamento de implantação;
- geração das predições;
- verificação de valores finitos;
- comparação das predições recalculadas com `ideb_predito_referencia`.

A comparação é realizada com tolerância numérica compatível com pequenas diferenças de ponto flutuante:

```python
np.allclose(
    predicoes,
    referencia,
    rtol=1e-6,
    atol=1e-6,
    equal_nan=True
)
```

Se a reprodução não for compatível com a referência exportada pelo notebook, a validação é interrompida e o artefato não recebe status de aprovação.

Para executar:

```bash
python treinar_modelos.py
```

Quando a validação é concluída com sucesso, é criado:

```text
outputs/resumo_validacao_artefatos_simulador.csv
```

## 8. Ambiente computacional

O ambiente da aplicação é definido por:

```text
runtime.txt
requirements.txt
```

O runtime utilizado pelo repositório é:

```text
Python 3.12
```

As principais dependências estão fixadas no `requirements.txt`, incluindo:

```text
streamlit==1.51.0
pandas==2.3.3
numpy==2.0.2
scikit-learn==1.6.1
xgboost==3.4.1
catboost==1.2.8
plotly==6.5.0
joblib==1.5.2
```

A fixação das versões tem como objetivo tornar o ambiente de implantação explicitamente reproduzível.

## 9. Instalação

Recomenda-se utilizar um ambiente virtual limpo com Python 3.12.

Após criar e ativar o ambiente, instale as dependências:

```bash
pip install -r requirements.txt
```

Em seguida, valide os artefatos:

```bash
python treinar_modelos.py
```

Somente após a validação ser concluída com sucesso, execute a aplicação.

## 10. Execução da aplicação

Execute:

```bash
streamlit run app.py
```

A aplicação será disponibilizada no navegador.

Na aba **Simulação**, o usuário pode:

1. selecionar a etapa de ensino;
2. selecionar uma variável disponível para simulação;
3. definir aumento ou redução percentual;
4. combinar uma ou mais alterações;
5. atribuir um nome ao cenário;
6. gerar e analisar os resultados.

## 11. Procedimento de simulação

As alterações são aplicadas às variáveis em sua escala original na base de referência de 2023.

Depois da alteração, o sistema reaplica os componentes do pipeline de implantação na mesma ordem definida no processo metodológico:

1. seleção das colunas de entrada;
2. conversão numérica;
3. imputação dos valores ausentes;
4. aplicação do seletor de variância;
5. seleção do subconjunto final de variáveis;
6. predição pelo modelo de implantação.

A aplicação calcula separadamente a predição da base sem alteração e a predição do cenário modificado.

## 12. Suporte empírico e extrapolação

O arquivo:

```text
suporte_empirico_variaveis.csv
```

registra informações sobre o domínio empírico das variáveis utilizadas pelo modelo.

Durante a simulação, os valores modificados são comparados com:

- mínimo observado;
- percentil 1;
- percentil 99;
- máximo observado.

A aplicação sinaliza cenários que ultrapassem essas faixas. Esse mecanismo não elimina a possibilidade de extrapolação, mas fornece uma indicação explícita de que determinada alteração se afasta do suporte empírico observado nos dados utilizados na implantação.

## 13. Métricas apresentadas

A aba de métricas apresenta os resultados do protocolo de avaliação científica registrados nos artefatos finais, incluindo, quando disponíveis:

- MAE no teste territorial independente;
- RMSE no teste territorial independente;
- R² no teste territorial independente;
- MAE na validação temporal de 2023;
- RMSE na validação temporal de 2023;
- R² na validação temporal de 2023.

Essas métricas pertencem ao modelo submetido à avaliação científica e não ao modelo posteriormente reajustado com todos os dados para implantação.

## 14. Interpretação dos resultados

Os valores produzidos pelo simulador representam respostas condicionais dos modelos a alterações hipotéticas nas variáveis de entrada.

Os resultados:

- não representam previsão de valores futuros do IDEB;
- não representam efeitos causais;
- não estabelecem relações de causa e efeito entre as variáveis e o IDEB;
- não substituem análise educacional, institucional ou territorial;
- não garantem que os cenários simulados sejam operacionalmente viáveis em políticas públicas.

O simulador deve ser utilizado como ferramenta de exploração analítica e análise de sensibilidade, em conjunto com a interpretação substantiva dos indicadores considerados.

## 15. Reprodutibilidade

A reprodutibilidade operacional do simulador é apoiada por:

- versões explícitas das dependências;
- separação entre artefatos de avaliação e implantação;
- bases de referência de 2023;
- predições de referência exportadas pelos notebooks;
- validação automática das predições instaladas;
- arquivos de metadados;
- manifesto com hashes SHA-256 dos artefatos integrados;
- scripts independentes para organização e validação.

Os notebooks metodológicos permanecem como fonte de verdade para treinamento, seleção de variáveis, avaliação científica e geração dos artefatos finais. Os scripts deste repositório não substituem esse protocolo e não realizam nova seleção ou novo treinamento.
