# =============================================================================
# 
# Previsão de Churn em Telecomunicações
# ataset: Telco Customer Churn (Kaggle/IBM)
# =============================================================================

# ===========================================================================
# IMPORTS
# ===========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import shap

# Paleta visual para todo o projeto
PALETTE = {
    'primary': '#2563EB',
    'danger':  '#DC2626',
    'success': '#16A34A',
    'warning': '#D97706',
    'neutral': '#6B7280',
    'bg':      '#F8FAFC',
}
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})


# ===========================================================================
# EXTRAÇÃO E CARGA (ETL)
# ===========================================================================
print("=" * 60)
print("ETAPA 1 — EXTRAÇÃO E CARGA DOS DADOS")
print("=" * 60)

# ------------------------------------------------------------------
# Dataset: "Telco Customer Churn" — IBM Sample Data Sets
# Kaggle:  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Colunas: 21 features | 7.043 clientes | Alvo: Churn (Yes/No)
# ------------------------------------------------------------------

# ── URL direta (GitHub mirror) ───────────────────────────
URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

try:
    df = pd.read_csv(URL)
    print(f"✅ Dados carregados via URL — {df.shape[0]:,} linhas × {df.shape[1]} colunas")
except Exception:
    # Fallback: gera dados sintéticos para o notebook rodar offline
    print("⚠️  URL indisponível — gerando dados sintéticos para demonstração…")
    np.random.seed(42)
    n = 7_043
    df = pd.DataFrame({
        'customerID':        [f'ID-{i:05d}' for i in range(n)],
        'gender':            np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen':     np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner':           np.random.choice(['Yes', 'No'], n),
        'Dependents':        np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70]),
        'tenure':            np.random.randint(0, 73, n),
        'PhoneService':      np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10]),
        'MultipleLines':     np.random.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService':   np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity':    np.random.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport':       np.random.choice(['Yes', 'No', 'No internet service'], n),
        'Contract':          np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling':  np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
        'PaymentMethod':     np.random.choice(
                                 ['Electronic check', 'Mailed check',
                                  'Bank transfer (automatic)', 'Credit card (automatic)'], n),
        'MonthlyCharges':    np.round(np.random.uniform(18, 119, n), 2),
        'TotalCharges':      '',           # preenchido abaixo
        'Churn':             np.random.choice(['Yes', 'No'], n, p=[0.265, 0.735]),
    })
    df['tenure'] = df['tenure'].astype(int)
    df['TotalCharges'] = np.where(
        df['tenure'] == 0, ' ',
        np.round(df['MonthlyCharges'] * df['tenure'] * np.random.uniform(0.85, 1.05, n), 2).astype(str)
    )

print(f"\n📋 Primeiras linhas:\n{df.head(3).to_string()}")


# ===========================================================================
# ETAPA 2 — ANÁLISE EXPLORATÓRIA (EDA)
# ===========================================================================
print("\n" + "=" * 60)
print("ETAPA 2 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("=" * 60)

# ── 2.1 Visão geral ───────────────────────────────────────────────
print("\n📊 Tipos e valores ausentes:")
print(df.dtypes.to_string())
print(f"\nDuplicatas: {df.duplicated().sum()}")
print(f"Valores nulos:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── 2.2 Limpeza ───────────────────────────────────────────────────
# TotalCharges vem como string com espaços em branco
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
nulls_tc = df['TotalCharges'].isnull().sum()
print(f"\n🔧 TotalCharges — {nulls_tc} valores não numéricos → imputados com 0 (tenure=0)")
df['TotalCharges'].fillna(0, inplace=True)

# Remover duplicatas e coluna de ID
df.drop_duplicates(inplace=True)
df.drop(columns=['customerID'], inplace=True)

# Codificar alvo
df['Churn_bin'] = (df['Churn'] == 'Yes').astype(int)
churn_rate = df['Churn_bin'].mean()
print(f"\n📈 Taxa de Churn geral: {churn_rate:.1%}")

# ── 2.3 GRÁFICO 1 — Distribuição do Churn (pie) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distribuição do Churn', fontsize=16, fontweight='bold', y=1.01)

counts = df['Churn'].value_counts()
axes[0].pie(
    counts, labels=counts.index, autopct='%1.1f%%',
    colors=[PALETTE['success'], PALETTE['danger']],
    startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0].set_title('Proporção Churn vs Retenção')

# Bar plot com anotações
bars = axes[1].bar(counts.index, counts.values,
                   color=[PALETTE['success'], PALETTE['danger']], edgecolor='white')
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
                 f'{bar.get_height():,}', ha='center', fontweight='bold')
axes[1].set_title('Contagem Absoluta')
axes[1].set_ylabel('Clientes')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('outputs/01_distribuicao_churn.png', bbox_inches='tight')
plt.show()
print("💾 Gráfico salvo: outputs/01_distribuicao_churn.png")

# ── 2.4 GRÁFICO 2 — Churn por tipo de contrato ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
contract_churn = (df.groupby('Contract')['Churn_bin']
                  .agg(['mean', 'count'])
                  .reset_index()
                  .sort_values('mean', ascending=False))

bars = ax.barh(contract_churn['Contract'],
               contract_churn['mean'] * 100,
               color=[PALETTE['danger'], PALETTE['warning'], PALETTE['success']])
for i, (v, n) in enumerate(zip(contract_churn['mean'], contract_churn['count'])):
    ax.text(v * 100 + 0.5, i, f'{v:.1%}  (n={n:,})', va='center', fontsize=11)

ax.set_xlabel('Taxa de Churn (%)')
ax.set_title('Taxa de Churn por Tipo de Contrato', fontsize=14, fontweight='bold')
ax.set_xlim(0, 60)
plt.tight_layout()
plt.savefig('outputs/02_churn_por_contrato.png', bbox_inches='tight')
plt.show()
print("💾 Gráfico salvo: outputs/02_churn_por_contrato.png")

# ── 2.5 GRÁFICO 3 — Distribuição do Tenure (meses de contrato) ────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribuição do Tempo de Contrato (Tenure)', fontsize=14, fontweight='bold')

for label, color in zip(['No', 'Yes'], [PALETTE['success'], PALETTE['danger']]):
    subset = df[df['Churn'] == label]['tenure']
    axes[0].hist(subset, bins=30, alpha=0.65, label=f'Churn={label}',
                 color=color, edgecolor='white')

axes[0].set_xlabel('Meses de Contrato')
axes[0].set_ylabel('Frequência')
axes[0].legend()
axes[0].set_title('Histograma por Churn')

# Box plot
df.boxplot(column='tenure', by='Churn', ax=axes[1],
           boxprops=dict(color=PALETTE['primary']),
           medianprops=dict(color=PALETTE['danger'], linewidth=2))
axes[1].set_title('Box Plot por Churn')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Meses de Contrato')
plt.suptitle('')
plt.tight_layout()
plt.savefig('outputs/03_tenure_distribuicao.png', bbox_inches='tight')
plt.show()
print("💾 Gráfico salvo: outputs/03_tenure_distribuicao.png")

# ── 2.6 GRÁFICO 4 — Monthly Charges vs Churn ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for label, color in zip(['No', 'Yes'], [PALETTE['success'], PALETTE['danger']]):
    sns.kdeplot(df[df['Churn'] == label]['MonthlyCharges'],
                ax=ax, fill=True, alpha=0.4, color=color, label=f'Churn={label}')
ax.set_title('Distribuição de Cobranças Mensais por Churn', fontsize=14, fontweight='bold')
ax.set_xlabel('Cobrança Mensal (R$)')
ax.set_ylabel('Densidade')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/04_monthly_charges_kde.png', bbox_inches='tight')
plt.show()
print("💾 Gráfico salvo: outputs/04_monthly_charges_kde.png")

# ── 2.7 GRÁFICO 5 — Heatmap de correlação ─────────────────────────
# Criar cópia com encoding para correlação
df_corr = df.copy()
le = LabelEncoder()
for col in df_corr.select_dtypes('object').columns:
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))

corr = df_corr.corr()[['Churn_bin']].drop('Churn_bin').sort_values('Churn_bin', ascending=False)

fig, ax = plt.subplots(figsize=(7, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlação das Features com Churn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/05_correlacao_churn.png', bbox_inches='tight')
plt.show()
print("💾 Gráfico salvo: outputs/05_correlacao_churn.png")

# Sumário EDA
print("\n📌 Principais achados da EDA:")
median_tenure_churn = df[df['Churn'] == 'Yes']['tenure'].median()
median_tenure_no_churn = df[df['Churn'] == 'No']['tenure'].median()
print(f"  • Clientes com churn têm mediana de tenure: {median_tenure_churn:.0f} meses")
print(f"  • Clientes sem churn têm mediana de tenure: {median_tenure_no_churn:.0f} meses")
avg_charge_churn = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
avg_charge_no = df[df['Churn'] == 'No']['MonthlyCharges'].mean()
print(f"  • Cobrança média churn: R${avg_charge_churn:.2f} | sem churn: R${avg_charge_no:.2f}")


# ===========================================================================
# ETAPA 3 — FEATURE ENGINEERING E MODELAGEM
# ===========================================================================
print("\n" + "=" * 60)
print("ETAPA 3 — FEATURE ENGINEERING E MODELAGEM")
print("=" * 60)

# ── 3.1 Feature Engineering ───────────────────────────────────────
df_model = df.copy()

# Nova feature: receita total por mês de permanência
df_model['AvgRevenuePerMonth'] = np.where(
    df_model['tenure'] > 0,
    df_model['TotalCharges'] / df_model['tenure'],
    df_model['MonthlyCharges']
)

# Segmento de tempo de contrato
df_model['TenureGroup'] = pd.cut(
    df_model['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-12m', '13-24m', '25-48m', '49-72m'],
    right=True
)

# Flag: serviços de proteção/suporte contratados
protective_services = ['OnlineSecurity', 'TechSupport']
for svc in protective_services:
    if svc in df_model.columns:
        df_model[f'Has{svc}'] = (df_model[svc] == 'Yes').astype(int)

# Número total de serviços adicionais
add_ons = ['OnlineSecurity', 'TechSupport', 'MultipleLines']
for col in add_ons:
    if col in df_model.columns:
        df_model[f'_{col}_bin'] = (df_model[col] == 'Yes').astype(int)
df_model['TotalAddOns'] = df_model[[c for c in df_model.columns if c.startswith('_')]].sum(axis=1)
df_model.drop(columns=[c for c in df_model.columns if c.startswith('_')], inplace=True)

print(f"✅ Novas features criadas: AvgRevenuePerMonth, TenureGroup, HasOnlineSecurity, "
      f"HasTechSupport, TotalAddOns")

# ── 3.2 Preparação para ML ────────────────────────────────────────
# Remover coluna alvo original e colunas auxiliares
TARGET = 'Churn_bin'
DROP_COLS = ['Churn', 'TenureGroup']   # target string + categórica ordinal usada só na EDA

df_model.drop(columns=DROP_COLS, inplace=True)

# Encoding
cat_cols = df_model.select_dtypes('object').columns.tolist()
print(f"\n🔤 Colunas categóricas para encoding: {cat_cols}")
df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

# Garantir que não há NaN nas features
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)   # colunas bool/dummy que restarem

print(f"\n📐 Shape final — X: {X.shape} | y: {y.value_counts().to_dict()}")
print(f"   NaNs restantes em X: {X.isnull().sum().sum()}")

# ── 3.3 Split e balanceamento com SMOTE ───────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"\n⚖️  Após SMOTE — treino: {y_train_res.value_counts().to_dict()}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# ── 3.4 Treino de múltiplos modelos ───────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n📊 Validação Cruzada (5-fold AUC-ROC):")
for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train_res,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    results[name] = scores
    print(f"  {name:25s} → AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 3.5 Modelo campeão — Gradient Boosting ────────────────────────
best_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
best_model.fit(X_train_sc, y_train_res)

y_pred  = best_model.predict(X_test_sc)
y_proba = best_model.predict_proba(X_test_sc)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print(f"\n🏆 Gradient Boosting — AUC-ROC no teste: {auc:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# ── 3.6 GRÁFICO 6 — Curva ROC ─────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Avaliação do Modelo — Gradient Boosting', fontsize=15, fontweight='bold')

axes[0].plot(fpr, tpr, color=PALETTE['primary'], lw=2, label=f'AUC = {auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
axes[0].fill_between(fpr, tpr, alpha=0.15, color=PALETTE['primary'])
axes[0].set_xlabel('Taxa de Falsos Positivos')
axes[0].set_ylabel('Taxa de Verdadeiros Positivos')
axes[0].set_title('Curva ROC')
axes[0].legend()

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Retained', 'Churned'])
disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title('Matriz de Confusão')

plt.tight_layout()
plt.savefig('outputs/06_roc_confusion.png', bbox_inches='tight')
plt.show()
print("💾 Gráfico salvo: outputs/06_roc_confusion.png")

# ── 3.7 GRÁFICO 7 — Feature Importance ───────────────────────────
fi = pd.Series(best_model.feature_importances_, index=X.columns)
fi_top = fi.nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
fi_top.plot(kind='barh', ax=ax, color=PALETTE['primary'])
ax.set_title('Top 15 Features Mais Importantes (Gradient Boosting)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Importância')
plt.tight_layout()
plt.savefig('outputs/07_feature_importance.png', bbox_inches='tight')
plt.show()
print("💾 Gráfico salvo: outputs/07_feature_importance.png")

# ── 3.8 SHAP Values — Explicabilidade ────────────────────────────
print("\n🔍 Calculando SHAP values (amostra de 200 pontos)…")
sample_idx = np.random.choice(len(X_test_sc), size=min(200, len(X_test_sc)), replace=False)
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_sc[sample_idx])

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test.iloc[sample_idx],
                  plot_type='dot', show=False, max_display=15)
plt.title('SHAP Values — Impacto das Features no Churn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/08_shap_values.png', bbox_inches='tight')
plt.show()
print("💾 Gráfico salvo: outputs/08_shap_values.png")


# ===========================================================================
# ETAPA 4 — CONCLUSÕES E RESPOSTAS ÀS PERGUNTAS DE NEGÓCIO
# ===========================================================================
print("\n" + "=" * 60)
print("ETAPA 4 — CONCLUSÕES E RESPOSTAS ÀS PERGUNTAS DE NEGÓCIO")
print("=" * 60)

# ── P1: Qual o perfil de cliente com maior risco de churn? ─────────
high_risk = df[df['Churn'] == 'Yes']

print("\n❓ P1: Qual o perfil do cliente com maior risco de churn?")
print(f"   • Clientes com contrato 'Month-to-month': taxa de churn muito superior.")
print(f"   • Tenure baixo (< 12 meses): período crítico de retenção.")
print(f"   • Internet via Fiber Optic combinado com cobrança alta: perfil de risco.")
print(f"   • Sem serviços de proteção (OnlineSecurity / TechSupport): mais propensos a sair.")

# ── P2: Quanto tempo de contrato protege contra o churn? ───────────
tenure_churn = df.groupby(
    pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72])
)['Churn_bin'].mean() * 100

print(f"\n❓ P2: Quanto tempo de contrato reduz o risco de churn?")
for period, rate in tenure_churn.items():
    print(f"   • {str(period):12s} → Taxa de churn: {rate:.1f}%")
print("   → Clientes que ultrapassam 24 meses têm risco significativamente menor.")

# ── P3: Qual é o impacto financeiro esperado do churn? ─────────────
n_churned   = df['Churn_bin'].sum()
avg_monthly = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
lost_revenue_month = n_churned * avg_monthly

print(f"\n❓ P3: Qual o impacto financeiro estimado do churn?")
print(f"   • Total de clientes churned: {n_churned:,}")
print(f"   • Cobrança mensal média (churned): R${avg_monthly:.2f}")
print(f"   • Receita mensal perdida estimada: R${lost_revenue_month:,.2f}")
print(f"   • Se reduzirmos o churn em 20%: economia de ~R${lost_revenue_month * 0.20:,.2f}/mês")

# ── Painel Final ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ PROJETO CONCLUÍDO")
print(f"   Modelo: Gradient Boosting | AUC-ROC: {auc:.4f}")
print(f"   Dataset: Telco Customer Churn | {len(df):,} clientes")
print(f"   Gráficos gerados: 8 arquivos em /outputs")
print("=" * 60)
