"""
==============================================================================
MODULE : predict.py
PROJET : Prédiction et Modélisation de Séries Temporelles par RNA Multicouches
AUTEUR : Institut Supérieur Polytechnique de Madagascar
DATE   : Juin 2026
==============================================================================

Description :
    Ce module effectue les prédictions multi-horizons et calcule les métriques
    d'évaluation standard :
        — MSE  (Mean Squared Error)
        — RMSE (Root Mean Squared Error)
        — MAE  (Mean Absolute Error)
        — R²   (Coefficient de détermination)
        — MAPE (Mean Absolute Percentage Error)

    Toutes les valeurs numériques sont affichées avec 8 chiffres significatifs.
==============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─────────────────────────────────────────────────────────────
#  Métriques d'évaluation
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    label: str = "") -> dict:
    """
    Calcule et affiche les métriques d'évaluation de régression.

    Paramètres
    ----------
    y_true : np.ndarray
        Valeurs réelles (dénormalisées)
    y_pred : np.ndarray
        Valeurs prédites (dénormalisées)
    label : str
        Étiquette pour l'affichage

    Retourne
    --------
    dict avec clés : mse, rmse, mae, r2, mape
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # MAPE (éviter division par zéro)
    mask = np.abs(y_true) > 1e-10
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) /
                               y_true[mask])) * 100.0
    else:
        mape = np.nan

    metrics = {
        'MSE':  mse,
        'RMSE': rmse,
        'MAE':  mae,
        'R2':   r2,
        'MAPE': mape
    }

    # Affichage formaté
    title = f"MÉTRIQUES — {label}" if label else "MÉTRIQUES"
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")
    print(f"  MSE   : {mse:.8f}")
    print(f"  RMSE  : {rmse:.8f}")
    print(f"  MAE   : {mae:.8f}")
    print(f"  R²    : {r2:.8f}")
    print(f"  MAPE  : {mape:.4f} %")
    print(f"  {'─'*60}")

    return metrics


# ─────────────────────────────────────────────────────────────
#  Prédiction à 1 pas (step-by-step)
# ─────────────────────────────────────────────────────────────

def predict_one_step(model,
                     dataset: dict,
                     scaler_y) -> dict:
    """
    Prédiction à 1 pas : 10 valeurs prédites pour 10 valeurs existantes.
    (Correspond à la consigne : prédiction à a pas)

    Retourne
    --------
    dict avec y_true, y_pred (dénormalisés), métriques
    """
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    # Prédiction sur l'ensemble de test complet
    y_pred_norm = model.predict(X_test, verbose=0).ravel()

    # Dénormalisation
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_true, y_pred, label="PRÉDICTION À 1 PAS")

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': metrics
    }


# ─────────────────────────────────────────────────────────────
#  Prédiction multi-pas (itérative)
# ─────────────────────────────────────────────────────────────

def predict_multi_step(model,
                       dataset: dict,
                       scaler_X,
                       scaler_y,
                       horizon: int,
                       label: str = "") -> dict:
    """
    Prédiction à h pas via utilisation directe du modèle entraîné
    pour l'horizon h.

    (Modèle entraîné spécifiquement sur chaque horizon h)

    Retourne
    --------
    dict avec y_true, y_pred (dénormalisés), métriques
    """
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    # Prédiction directe avec le modèle entraîné pour cet horizon
    y_pred_norm = model.predict(X_test, verbose=0).ravel()

    # Dénormalisation
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()

    lbl = label if label else f"PRÉDICTION À {horizon} PAS"
    metrics = compute_metrics(y_true, y_pred, label=lbl)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': metrics
    }


# ─────────────────────────────────────────────────────────────
#  Tableau comparatif des métriques
# ─────────────────────────────────────────────────────────────

def print_metrics_table(all_results: dict):
    """
    Affiche un tableau comparatif des métriques pour tous les horizons.

    Paramètres
    ----------
    all_results : dict {horizon -> result_dict}
    """
    print("\n" + "=" * 80)
    print("  TABLEAU COMPARATIF DES MÉTRIQUES PAR HORIZON DE PRÉDICTION")
    print("=" * 80)
    print(f"  {'Horizon':>8} | {'MSE':>14} | {'RMSE':>14} | "
          f"{'MAE':>14} | {'R²':>10} | {'MAPE (%)':>10}")
    print("-" * 80)

    rows = []
    for h, res in all_results.items():
        m = res['metrics']
        rows.append({
            'Horizon': h,
            'MSE':  m['MSE'],
            'RMSE': m['RMSE'],
            'MAE':  m['MAE'],
            'R2':   m['R2'],
            'MAPE': m['MAPE']
        })
        print(f"  {h:>8d} | {m['MSE']:>14.8f} | {m['RMSE']:>14.8f} | "
              f"{m['MAE']:>14.8f} | {m['R2']:>10.8f} | {m['MAPE']:>10.4f}")

    print("=" * 80)

    df = pd.DataFrame(rows)
    df.to_csv("metrics_summary.csv", index=False, float_format="%.8f")
    print("  → Tableau sauvegardé dans 'metrics_summary.csv'")

    return df


if __name__ == "__main__":
    print("Module predict.py — Importer depuis main.py pour l'utilisation complète.")
