\# Simulador de Cenários Educacionais para o IDEB Municipal



Esta aplicação web permite simular cenários hipotéticos para o IDEB municipal dos Anos Iniciais e dos Anos Finais do Ensino Fundamental, utilizando modelos XGBoost treinados com variáveis educacionais, socioeconômicas e territoriais.



\## 1. Objetivo da aplicação



A aplicação foi construída para operacionalizar a simulação de cenários descrita no artigo, permitindo que o usuário altere percentualmente uma ou mais variáveis explicativas e observe a resposta estimada do modelo sobre o IDEB médio municipal de 2023.



A ferramenta não realiza previsão futura nem inferência causal. Os resultados devem ser interpretados como simulações condicionais do modelo, úteis para análise exploratória e avaliação de sensibilidade.



\## 2. Etapas de ensino contempladas



A aplicação contempla apenas:



1\. Anos Iniciais do Ensino Fundamental

2\. Anos Finais do Ensino Fundamental



O Ensino Médio não faz parte do escopo desta versão.



\## 3. Estrutura da pasta



```text

simulador\_ideb/

│

├── app.py

├── treinar\_modelos.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── data/

│   ├── base\_anos\_iniciais.xlsx

│   └── base\_anos\_finais.xlsx

│

└── models/

&#x20;   ├── modelo\_xgb\_anos\_iniciais.pkl

&#x20;   ├── modelo\_xgb\_anos\_finais.pkl

&#x20;   ├── metricas\_anos\_iniciais.pkl

&#x20;   └── metricas\_anos\_finais.pkl

