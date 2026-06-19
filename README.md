# Previsão de Churn em Telecomunicações

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Status-Concluído-brightgreen" />
  <img src="https://img.shields.io/badge/Dataset-Telco%20Customer%20Churn-yellow" />
</p>

---

## Sobre o Projeto

> **Cenário:** Uma operadora de telecomunicações fictícia, a **TeleConnect Brasil**, enfrenta uma taxa de cancelamento (churn) de **26,5%** ao mês — bem acima da média setorial de 18%. Cada cliente perdido representa uma perda média de **R$ 65/mês** em receita recorrente. O time de Data Science foi acionado para identificar quais clientes estão prestes a cancelar, com antecedência suficiente para que o time comercial aja.

Este projeto apresenta um pipeline completo de Ciência de Dados — do tratamento de dados à explicabilidade do modelo — respondendo perguntas de negócio reais com evidências extraídas dos dados.

---

## Perguntas de Negócio

| # | Pergunta | Respondida em |
|---|----------|---------------|
| 1 | Qual é o **perfil de cliente** com maior risco de churn? | Etapa 2 — EDA |
| 2 | Quanto tempo de contrato **protege contra o cancelamento**? | Etapas 2 e 4 |
| 3 | Qual é o **impacto financeiro** estimado do churn atual? | Etapa 4 |

---

## Estrutura do Repositório

```
churn-telecom/
│
├── notebooks/
│   └── analise_churn_telecom.py      # código principal comentado (ETL → EDA → ML → Conclusão)
│
├── outputs/                          # Gráficos de análise
│   ├── 01_distribuicao_churn.png
│   ├── 02_churn_por_contrato.png
│   ├── 03_tenure_distribuicao.png
│   ├── 04_monthly_charges_kde.png
│   ├── 05_correlacao_churn.png
│   ├── 06_roc_confusion.png
│   ├── 07_feature_importance.png
│   └── 08_shap_values.png
│
├── src/
│   └── utils.py                      # Funções auxiliares
│
├── requirements.txt
└── README.md
```

---

## Dataset

| Campo | Valor |
|-------|-------|
| **Nome** | Telco Customer Churn |
| **Fonte** | [Kaggle — blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Origem** | IBM Sample Datasets |
| **Linhas** | 7.043 clientes |
| **Colunas** | 21 features |
| **Alvo** | `Churn` (Yes / No) |

### Principais colunas

| Coluna | Descrição |
|--------|-----------|
| `tenure` | Meses de permanência do cliente |
| `Contract` | Tipo de contrato (mensal / 1 ano / 2 anos) |
| `MonthlyCharges` | Cobrança mensal (R$) |
| `TotalCharges` | Total cobrado durante o contrato |
| `InternetService` | Tipo de serviço de internet |
| `OnlineSecurity` | Possui segurança online contratada |
| `Churn` | **Variável alvo** — se o cliente cancelou |

---

## Tecnologias Utilizadas

| Categoria | Ferramentas |
|-----------|-------------|
| Linguagem | Python 3.9+ |
| Manipulação | Pandas, NumPy |
| Visualização | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Balanceamento | imbalanced-learn (SMOTE) |
| Explicabilidade | SHAP |
| Ambiente | Jupyter / VS Code |

---

## Como Rodar o Projeto

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/churn-telecom.git
cd churn-telecom
```

### 2. Crie e ative um ambiente virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o script principal
```bash
python notebooks/analise_churn_telecom.py
```

> 💡 Os gráficos serão salvos automaticamente na pasta `outputs/`.

### 5. (Opcional) Baixe o dataset do Kaggle
```bash
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/
```

---

## Fluxo da Solução

```
Dados Brutos (CSV)
       │
       ▼
 ┌─────────────┐
 │  ETL / Load │  ← Carga, tipos, nulos
 └──────┬──────┘
        │
        ▼
 ┌──────────────┐
 │     EDA      │  ← 5 gráficos, correlações, insights
 └──────┬───────┘
        │
        ▼
 ┌──────────────────────┐
 │  Feature Engineering │  ← AvgRevenuePerMonth, TenureGroup, TotalAddOns
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────┐
 │  SMOTE + Scaling     │  ← Balanceamento de classes
 └──────┬───────────────┘
        │
        ▼
 ┌──────────────────────────────────────────┐
 │  Comparação de Modelos (CV 5-fold AUC)   │
 │  • Logistic Regression                   │
 │  • Random Forest                         │
 │  • Gradient Boosting  ← Vencedor      │
 └──────┬───────────────────────────────────┘
        │
        ▼
 ┌──────────────────────┐
 │  Avaliação + SHAP    │  ← ROC, Confusion Matrix, Explicabilidade
 └──────┬───────────────┘
        │
        ▼
 Respostas às Perguntas de Negócio
```

---

## Resultados do Modelo

| Modelo | AUC-ROC (CV) |
|--------|-------------|
| Logistic Regression | ~0.83 |
| Random Forest | ~0.88 |
| **Gradient Boosting** | **~0.91** |

### Métricas no conjunto de teste (Gradient Boosting):

| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Retained (0) | 0.90 | 0.93 | 0.91 |
| Churned (1) | 0.78 | 0.71 | 0.74 |

---

## Principais Insights

### 1. Tipo de Contrato é o Maior Predictor
> Clientes com contratos **mensais** apresentam taxa de churn **~4× maior** do que clientes com contratos de 2 anos. Incentivar a migração para planos anuais é a ação de retenção de maior impacto.

### 2. Os Primeiros 12 Meses São Críticos
> A maioria dos cancelamentos ocorre nos **primeiros 12 meses** de contrato. Uma estratégia de onboarding ativo (suporte, benefícios, check-ins) nesse período pode reduzir o churn estruturalmente.

### 3. Clientes Fiber Optic + Sem Segurança São Alto Risco
> Clientes com internet de fibra óptica e **sem** serviços como `OnlineSecurity` e `TechSupport` têm churn significativamente mais alto. Oferecer esses add-ons como trial gratuito pode aumentar a percepção de valor e a retenção.

### 4. Impacto Financeiro Quantificado
> O churn atual representa uma perda estimada de **R$ 120.000+/mês** em receita recorrente. Reduzir a taxa em apenas 20% geraria uma recuperação de **~R$ 24.000/mês** sem aquisição de novos clientes.

### 5. SHAP: O Modelo É Explicável
> O uso de SHAP values permite que o time comercial entenda **por que** um cliente específico foi classificado como risco alto — viabilizando ações personalizadas em vez de campanhas genéricas.

---

## Autor

**Paulo Magno**
- LinkedIn: [linkedin.com/in/seu-perfil](https://www.linkedin.com/in/paulomagnofilho/)
- GitHub: [github.com/seu-usuario](https://github.com/paulomagnofilho)
- E-mail: pmagno15@email.com

---

---

<p align="center">
  Feito com 🐍 Python e muito café
</p>
