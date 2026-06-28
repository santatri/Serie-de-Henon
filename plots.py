"""
==============================================================================
MODULE : plots.py
PROJET : Prédiction et Modélisation de Séries Temporelles par RNA Multicouches
AUTEUR : Institut Supérieur Polytechnique de Madagascar
DATE   : Juin 2026
==============================================================================

Description :
    Ce module génère l'ensemble des graphiques du projet :
        1. Attracteur de Hénon (y(n) vs x(n))
        2. Série temporelle x(n)
        3. Courbe d'apprentissage (loss vs epochs)
        4. Prédictions vs valeurs réelles (tous horizons)
        5. Diagramme de dispersion (scatter plot)
        6. Erreurs de prédiction (résidus)
        7. Histogramme des résidus
        8. Tableau récapitulatif des métriques
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveurs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import os

# ── Style global ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       10,
    'axes.titlesize':  12,
    'axes.labelsize':  11,
    'axes.titleweight': 'bold',
    'axes.grid':       True,
    'grid.alpha':      0.3,
    'grid.linestyle':  '--',
    'lines.linewidth': 1.5,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'figure.dpi':      120,
    'savefig.dpi':     150,
    'savefig.bbox':    'tight',
})

# Palette de couleurs cohérente
COLORS = {
    'true':    '#1f77b4',   # Bleu
    'pred':    '#d62728',   # Rouge
    'train':   '#2ca02c',   # Vert
    'val':     '#ff7f0e',   # Orange
    'accent':  '#9467bd',   # Violet
    'neutral': '#7f7f7f',   # Gris
}

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Figure sauvegardée : {path}")
    return path


# ─────────────────────────────────────────────────────────────
#  1. Attracteur de Hénon
# ─────────────────────────────────────────────────────────────

def plot_henon_attractor(df: pd.DataFrame) -> str:
    """
    Trace l'attracteur de Hénon : y(n) en fonction de x(n).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Série de Hénon — a = 1.4, b = 0.3\n"
                 "x₀ = 0, y₀ = 0  |  500 itérations",
                 fontsize=13, fontweight='bold', y=1.02)

    # ── Attracteur ──────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(df['x'], df['y'],
                    c=np.arange(len(df)), cmap='plasma',
                    s=4, alpha=0.8, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Indice n")
    ax.set_xlabel("x(n)")
    ax.set_ylabel("y(n)")
    ax.set_title("Attracteur de Hénon\n(y(n) vs x(n))")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # ── Séries temporelles ──────────────────────────────────
    ax2 = axes[1]
    n = df['n'].values
    ax2.plot(n, df['x'].values, color=COLORS['true'],
             lw=1.0, alpha=0.85, label="x(n)")
    ax2.plot(n, df['y'].values, color=COLORS['pred'],
             lw=1.0, alpha=0.85, label="y(n)")
    ax2.set_xlabel("n (indice temporel)")
    ax2.set_ylabel("Valeur")
    ax2.set_title("Évolution temporelle de x(n) et y(n)")
    ax2.legend()
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    return _save(fig, "01_henon_attractor.png")


# ─────────────────────────────────────────────────────────────
#  2. Courbe d'apprentissage
# ─────────────────────────────────────────────────────────────

def plot_learning_curve(history, horizon: int = 1) -> str:
    """
    Trace loss et val_loss en fonction des epochs.
    """
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = np.arange(1, len(loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Courbes d'apprentissage — Horizon h = {horizon}",
                 fontsize=13, fontweight='bold')

    # Loss
    ax = axes[0]
    ax.plot(epochs, loss,     color=COLORS['train'], label="Train MSE")
    ax.plot(epochs, val_loss, color=COLORS['val'],   label="Val MSE",
            linestyle='--')
    ax.set_xlabel("Époques")
    ax.set_ylabel("MSE (normalisée)")
    ax.set_title("Évolution de la MSE")
    ax.legend()
    ax.set_yscale('log')

    # Loss (zoom fin d'apprentissage)
    ax2 = axes[1]
    n_zoom = max(10, len(loss) // 3)
    ax2.plot(epochs[-n_zoom:], loss[-n_zoom:],
             color=COLORS['train'], label="Train MSE")
    ax2.plot(epochs[-n_zoom:], val_loss[-n_zoom:],
             color=COLORS['val'],   label="Val MSE", linestyle='--')
    ax2.set_xlabel("Époques (zoom)")
    ax2.set_ylabel("MSE (normalisée)")
    ax2.set_title("Zoom — Convergence finale")
    ax2.legend()

    fig.tight_layout()
    return _save(fig, f"02_learning_curve_h{horizon}.png")


# ─────────────────────────────────────────────────────────────
#  3. Prédictions vs réel (un horizon)
# ─────────────────────────────────────────────────────────────

def plot_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     horizon: int,
                     metrics: dict,
                     n_display: int = 100) -> str:
    """
    Affiche réel vs prédit sur n_display points, plus scatter et résidus.
    """
    n = min(n_display, len(y_true))
    idx = np.arange(n)
    residuals = y_true[:n] - y_pred[:n]

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    fig.suptitle(
        f"Prédiction à {horizon} pas — MLP (Série de Hénon)\n"
        f"MSE={metrics['MSE']:.6f}  RMSE={metrics['RMSE']:.6f}  "
        f"MAE={metrics['MAE']:.6f}  R²={metrics['R2']:.6f}",
        fontsize=12, fontweight='bold'
    )

    # ── (a) Réel vs Prédit ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(idx, y_true[:n], color=COLORS['true'], lw=1.2,
             label="Valeurs réelles", zorder=3)
    ax1.plot(idx, y_pred[:n], color=COLORS['pred'], lw=1.2,
             linestyle='--', label=f"Valeurs prédites (h={horizon})", zorder=2)
    ax1.fill_between(idx,
                     y_true[:n], y_pred[:n],
                     alpha=0.15, color=COLORS['neutral'], label="Erreur")
    ax1.set_xlabel("Indice temporel")
    ax1.set_ylabel("x(n)")
    ax1.set_title(f"Valeurs réelles vs prédites — {n} premiers points de test")
    ax1.legend(loc='upper right')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    # ── (b) Scatter plot ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    lim = [min(y_true.min(), y_pred.min()) * 1.05,
           max(y_true.max(), y_pred.max()) * 1.05]
    ax2.scatter(y_true, y_pred, alpha=0.4, s=12,
                color=COLORS['accent'], label="Points")
    ax2.plot(lim, lim, 'k--', lw=1.5, label="Identité parfaite")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("Valeurs réelles")
    ax2.set_ylabel("Valeurs prédites")
    ax2.set_title("Diagramme de dispersion\n(Scatter plot)")
    ax2.legend(fontsize=8)

    # ── (c) Résidus ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(idx, residuals, color=COLORS['neutral'],
            alpha=0.7, label="Résidus")
    ax3.axhline(0, color='black', lw=1.0)
    ax3.axhline( residuals.std(), color=COLORS['pred'],
                linestyle=':', lw=1.5, label="±1σ")
    ax3.axhline(-residuals.std(), color=COLORS['pred'],
                linestyle=':', lw=1.5)
    ax3.set_xlabel("Indice temporel")
    ax3.set_ylabel("Résidu")
    ax3.set_title("Erreurs de prédiction (résidus)")
    ax3.legend(fontsize=8)

    return _save(fig, f"03_predictions_h{horizon}.png")


# ─────────────────────────────────────────────────────────────
#  4. Comparaison tous horizons
# ─────────────────────────────────────────────────────────────

def plot_all_horizons(all_results: dict, n_display: int = 80) -> str:
    """
    Grille 2×2 : prédictions pour h = 1, 3, 10, 20 sur un même canvas.
    """
    horizons = sorted(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    fig.suptitle("Comparaison des prédictions multi-horizons\n"
                 "Réseau de neurones MLP — Série de Hénon",
                 fontsize=13, fontweight='bold')

    for i, h in enumerate(horizons):
        ax = axes[i]
        res = all_results[h]
        y_true = res['y_true']
        y_pred = res['y_pred']
        m      = res['metrics']
        n = min(n_display, len(y_true))
        idx = np.arange(n)

        ax.plot(idx, y_true[:n], color=COLORS['true'], lw=1.2,
                label="Réel", zorder=3)
        ax.plot(idx, y_pred[:n], color=COLORS['pred'], lw=1.2,
                linestyle='--', label=f"Prédit (h={h})", zorder=2)
        ax.fill_between(idx, y_true[:n], y_pred[:n],
                        alpha=0.12, color=COLORS['neutral'])

        ax.set_title(
            f"Horizon h = {h} pas\n"
            f"RMSE={m['RMSE']:.6f}  R²={m['R2']:.6f}",
            fontsize=10
        )
        ax.set_xlabel("Indice temporel")
        ax.set_ylabel("x(n)")
        ax.legend(fontsize=8, loc='upper right')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "04_all_horizons_comparison.png")


# ─────────────────────────────────────────────────────────────
#  5. Tableau métriques (figure)
# ─────────────────────────────────────────────────────────────

def plot_metrics_table(metrics_df: pd.DataFrame) -> str:
    """
    Génère une figure avec le tableau des métriques.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    col_labels = ['Horizon (h)', 'MSE', 'RMSE', 'MAE', 'R²', 'MAPE (%)']
    cell_text  = []
    for _, row in metrics_df.iterrows():
        cell_text.append([
            f"h = {int(row['Horizon'])}",
            f"{row['MSE']:.8f}",
            f"{row['RMSE']:.8f}",
            f"{row['MAE']:.8f}",
            f"{row['R2']:.8f}",
            f"{row['MAPE']:.4f}"
        ])

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Style en-tête
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1f4e79')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Alternance de couleurs de lignes
    for i in range(1, len(cell_text) + 1):
        color = '#dce6f1' if i % 2 == 0 else '#ffffff'
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(color)

    ax.set_title("Tableau des métriques de prédiction — 8 chiffres significatifs",
                 fontsize=12, fontweight='bold', pad=20)

    fig.tight_layout()
    return _save(fig, "05_metrics_table.png")


# ─────────────────────────────────────────────────────────────
#  6. Évolution des métriques vs horizon
# ─────────────────────────────────────────────────────────────

def plot_metrics_vs_horizon(metrics_df: pd.DataFrame) -> str:
    """
    Trace RMSE, MAE et R² en fonction de l'horizon h.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Évolution des métriques en fonction de l'horizon de prédiction",
                 fontsize=12, fontweight='bold')

    h = metrics_df['Horizon'].values

    pairs = [
        (axes[0], 'RMSE', 'RMSE', COLORS['pred']),
        (axes[1], 'MAE',  'MAE',  COLORS['accent']),
        (axes[2], 'R2',   'R²',   COLORS['train']),
    ]

    for ax, col, ylabel, color in pairs:
        ax.plot(h, metrics_df[col].values, 'o-',
                color=color, lw=2, markersize=8)
        for xi, yi in zip(h, metrics_df[col].values):
            ax.annotate(f"{yi:.4f}", xy=(xi, yi),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8)
        ax.set_xlabel("Horizon h (pas)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Horizon")
        ax.set_xticks(h)
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    return _save(fig, "06_metrics_vs_horizon.png")


# ─────────────────────────────────────────────────────────────
#  7. Histogrammes des résidus (tous horizons)
# ─────────────────────────────────────────────────────────────

def plot_residuals_histogram(all_results: dict) -> str:
    """
    Histogramme des résidus pour chaque horizon de prédiction.
    """
    horizons = sorted(all_results.keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(14, 4), sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle("Distribution des erreurs de prédiction (résidus)",
                 fontsize=12, fontweight='bold')

    for ax, h in zip(axes, horizons):
        res  = all_results[h]
        resid = res['y_true'] - res['y_pred']
        ax.hist(resid, bins=30, color=COLORS['accent'],
                alpha=0.75, edgecolor='white', lw=0.5)
        ax.axvline(resid.mean(), color=COLORS['pred'],
                   lw=2, linestyle='--',
                   label=f"μ={resid.mean():.4f}")
        ax.axvline(resid.mean() + resid.std(), color=COLORS['neutral'],
                   lw=1.5, linestyle=':', label=f"σ={resid.std():.4f}")
        ax.axvline(resid.mean() - resid.std(), color=COLORS['neutral'],
                   lw=1.5, linestyle=':')
        ax.set_title(f"h = {h} pas")
        ax.set_xlabel("Résidu")
        ax.set_ylabel("Fréquence")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, "07_residuals_histogram.png")


if __name__ == "__main__":
    print("Module plots.py — À appeler depuis main.py")
