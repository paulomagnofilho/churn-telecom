"""
utils.py — Funções auxiliares para o projeto Churn Telecom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report


PALETTE = {
    'primary': '#2563EB',
    'danger':  '#DC2626',
    'success': '#16A34A',
    'warning': '#D97706',
    'neutral': '#6B7280',
}


def resumo_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna um DataFrame com: tipo, nulos (%), únicos e amostra de valores.
    Útil para a primeira inspeção do dataset.
    """
    summary = pd.DataFrame({
        'Tipo':       df.dtypes,
        'Nulos':      df.isnull().sum(),
        'Nulos (%)':  (df.isnull().mean() * 100).round(2),
        'Únicos':     df.nunique(),
        'Exemplo':    df.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else None),
    })
    return summary.sort_values('Nulos (%)', ascending=False)


def plot_churn_por_categoria(df: pd.DataFrame, col: str,
                              target: str = 'Churn', ax=None) -> None:
    """
    Plota a taxa de churn por categoria de uma coluna.

    Args:
        df: DataFrame com os dados.
        col: Nome da coluna categórica.
        target: Nome da coluna alvo (deve ter valores 'Yes'/'No').
        ax: Eixo matplotlib opcional.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    churn_rate = (df.groupby(col)[target]
                  .apply(lambda x: (x == 'Yes').mean() * 100)
                  .reset_index()
                  .rename(columns={target: 'Churn Rate (%)'}))

    sns.barplot(data=churn_rate, x=col, y='Churn Rate (%)',
                palette='Reds_r', ax=ax)
    ax.set_title(f'Taxa de Churn por {col}', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Churn (%)')
    ax.tick_params(axis='x', rotation=30)

    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=9
        )


def calcular_impacto_financeiro(df: pd.DataFrame,
                                 churn_col: str = 'Churn',
                                 revenue_col: str = 'MonthlyCharges') -> dict:
    """
    Calcula métricas de impacto financeiro do churn.

    Returns:
        dict com: clientes_perdidos, receita_perdida_mes, ticket_medio_churn
    """
    churned      = df[df[churn_col] == 'Yes']
    n_churned    = len(churned)
    avg_revenue  = churned[revenue_col].mean()
    total_lost   = n_churned * avg_revenue

    return {
        'clientes_perdidos':     n_churned,
        'ticket_medio_churn':    round(avg_revenue, 2),
        'receita_perdida_mes':   round(total_lost, 2),
        'economia_20pct_reducao': round(total_lost * 0.20, 2),
    }


def avaliar_modelo(model, X_test, y_test, model_name: str = 'Modelo') -> dict:
    """
    Avalia um modelo de classificação e imprime um relatório formatado.

    Returns:
        dict com auc, y_pred, y_proba
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n{'=' * 50}")
    print(f"  {model_name} — AUC-ROC: {auc:.4f}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

    return {'auc': auc, 'y_pred': y_pred, 'y_proba': y_proba}
