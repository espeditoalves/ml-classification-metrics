- [1. 🎯 Titanic Classification - Evaluation Metrics](#1--titanic-classification---evaluation-metrics)
  - [1.1. 📌 Objetivo](#11--objetivo)
  - [1.2. ⚙️ O que você encontrará neste repositório](#12-️-o-que-você-encontrará-neste-repositório)
  - [1.3. 🧪 Tecnologias utilizadas](#13--tecnologias-utilizadas)
  - [1.4. ▶️ Como executar](#14-️-como-executar)
    - [1.4.1. 📂 Acessando o Jupyter Notebook](#141--acessando-o-jupyter-notebook)
    - [1.4.2. 💻 Usando no VS Code com kernel remoto](#142--usando-no-vs-code-com-kernel-remoto)
  - [1.5. 🧠 Motivação](#15--motivação)
  - [1.6. 📁 Estrutura do projeto](#16--estrutura-do-projeto)
  - [1.7. 📚 Referências](#17--referências)
  - [1.8. 📬 Contato para sugestões, dúvidas ou colaborações:\*\*](#18--contato-para-sugestões-dúvidas-ou-colaborações)

# 1. 🎯 Titanic Classification - Evaluation Metrics

Este repositório tem como objetivo aplicar e analisar **métricas de avaliação de performance** em modelos de classificação, utilizando como base o famoso conjunto de dados **Titanic** (disponível no [Kaggle](https://www.kaggle.com/competitions/titanic)).

---

## 1.1. 📌 Objetivo

Avaliar a capacidade preditiva de modelos de *machine learning* na tarefa de **classificar quais passageiros sobreviveram** ao naufrágio do Titanic, com ênfase na **interpretação e comparação de métricas de desempenho**.

---

## 1.2. ⚙️ O que você encontrará neste repositório

- 📂 Pré-processamento do dataset Titanic
  - Simples e objetivo
- 🤖 Treinamento de modelos de classificação
  - Random Forest
  - SVC
- 📊 Geração e análise de métricas como:
  - Acurácia
  - Precisão
  - Revocação (Recall)
  - F1-Score
  - Curva ROC e AUC
  - Matriz de Confusão com anotações interpretativas
- 📈 Visualizações com `matplotlib` e `seaborn` para facilitar a interpretação dos resultados
  - KS
  - Separação de classes

---

## 1.3. 🧪 Tecnologias utilizadas

- Python 3.11+
- scikit-learn
- pandas
- seaborn
- matplotlib
- docker

---

## 1.4. ▶️ Como executar

Clone o repositório, instale e inicie o ambiente em um container Docker.  
💡 *É necessário ter o [Docker](https://www.docker.com/) instalado na máquina.*

```bash
git clone https://github.com/espeditoalves/ml-classification-metrics.git
cd ml-classification-metrics
docker compose up       # Cria e inicia o container com Jupyter Notebook
```

---

### 1.4.1. 📂 Acessando o Jupyter Notebook

Após iniciar o container, abra o navegador e acesse:

📍 [http://localhost:8888](http://localhost:8888)

O terminal exibirá uma URL com um token de acesso. Exemplo:

```
http://127.0.0.1:8888/lab?token=abcdef123456...
```

---

### 1.4.2. 💻 Usando no VS Code com kernel remoto

Para editar os notebooks no VS Code:

1. Copie o link exibido no terminal (`http://127.0.0.1:8888/?token=...`)
2. No VS Code, abra o notebook (`.ipynb`)
3. Clique no canto superior direito em **"Select Kernel"**
4. Escolha **"Existing Jupyter Server"**
5. Cole a URL copiada (com token) e pressione Enter

✅ O VS Code estará conectado ao kernel do Jupyter rodando dentro do container.


## 1.5. 🧠 Motivação
Muitas vezes, a acurácia isolada pode ser enganosa — especialmente em conjuntos de dados desbalanceados. Este projeto visa demonstrar a importância de escolher e interpretar corretamente as métricas de avaliação ao treinar modelos preditivos.

## 1.6. 📁 Estrutura do projeto
```bash
├── config/                    # Arquivos de configuração do projeto
├── data/                      # Dados brutos e processados
│   ├── processed/             # Dados após limpeza e engenharia de atributos
│   └── titanic/               # Dataset original do Titanic
├── docs/                      # Documentação do projeto
├── image/                     # Imagens usadas em notebooks ou README
├── models/                    # Modelos treinados e serializados (pkl, joblib, etc.)
├── notebooks/                 # Análises e experimentos em Jupyter
├── tests/                     # Scripts de teste e validação
├── utils/                     # Funções utilitárias e módulos reutilizáveis
├── .pre-commit-config/        # Configuração de hooks de pré-commit
├── docker-compose.yml         # Ambiente com Jupyter Notebook via Docker
├── Makefile                   # Comandos automatizados para setup e execução
├── README.md                  # Documentação principal do projeto

```

## 1.7. 📚 Referências
- Kaggle Titanic Dataset
- Scikit-learn Documentation - Metrics
- Jupyter Docker Stacks

## 1.8. 📬 Contato para sugestões, dúvidas ou colaborações:**  

- ✉️ **E-mail:** [espedito.ferreira.alves@outlook.com](espedito.ferreira.alves@outlook.com) 
- 🔗 **LinkedIn:** [Espedito Ferreira Alves](https://www.linkedin.com/in/espedito-ferreira-alves/)  