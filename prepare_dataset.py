"""
==============================================================================
MODULE : prepare_dataset.py
PROJET : Prédiction et Modélisation de Séries Temporelles par RNA Multicouches
AUTEUR : Institut Supérieur Polytechnique de Madagascar
DATE   : Juin 2026
==============================================================================

Description :
    Ce module prépare la base de données supervisée pour l'apprentissage du
    réseau de neurones. Il transforme la série temporelle x(n) en paires
    (entrées, cibles) via une fenêtre glissante.

    Stratégie de fenêtrage :
        Pour une fenêtre de taille W et un horizon h :
        Entrées  : [x(t-W+1), x(t-W+2), ..., x(t)]
        Cible    : x(t + h)
==============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def create_supervised_dataset(series: np.ndarray,
                               window_size: int = 10,
                               horizon: int = 1) -> tuple:
    """
    Construit un jeu de données supervisé par fenêtre glissante.

    Paramètres
    ----------
    series : np.ndarray
        Série temporelle univariée (ex : x(n) de Hénon)
    window_size : int
        Nombre de pas de temps utilisés comme entrée
    horizon : int
        Horizon de prédiction (nombre de pas en avant)

    Retourne
    --------
    X : np.ndarray, shape (N_samples, window_size)
    y : np.ndarray, shape (N_samples,)
    """
    X_list, y_list = [], []
    n = len(series)

    for i in range(n - window_size - horizon + 1):
        X_list.append(series[i: i + window_size])
        y_list.append(series[i + window_size + horizon - 1])

    return np.array(X_list), np.array(y_list)


def prepare_all_horizons(series: np.ndarray,
                          window_size: int = 10,
                          horizons: list = None,
                          test_size: float = 0.2,
                          random_state: int = 42) -> dict:
    """
    Prépare les jeux de données normalisés pour plusieurs horizons de
    prédiction.

    Paramètres
    ----------
    series : np.ndarray
        Série temporelle brute
    window_size : int
        Taille de la fenêtre d'entrée
    horizons : list
        Liste des horizons à préparer (ex : [1, 3, 10, 20])
    test_size : float
        Proportion des données de test (défaut : 0.20)
    random_state : int
        Graine aléatoire pour la reproductibilité

    Retourne
    --------
    dict : {horizon -> {'X_train', 'X_test', 'y_train', 'y_test',
                        'scaler_X', 'scaler_y'}}
    """
    if horizons is None:
        horizons = [1, 3, 10, 20]

    datasets = {}

    for h in horizons:
        X, y = create_supervised_dataset(series, window_size=window_size,
                                         horizon=h)

        # Normalisation MinMax sur [-1, 1] pour mieux capturer la dynamique chaotique
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Séparation temporelle (pas aléatoire pour les séries temporelles)
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

        datasets[h] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'X_raw': X,
            'y_raw': y,
            'window_size': window_size,
            'horizon': h,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        print(f"  Horizon h={h:2d} → Train: {len(X_train):4d} | "
              f"Test: {len(X_test):4d} | "
              f"Features: {X_train.shape[1]}")

    return datasets


def print_dataset_summary(datasets: dict):
    """Affiche un résumé tabulaire des jeux de données préparés."""
    print("\n" + "=" * 70)
    print("  RÉSUMÉ DES JEUX DE DONNÉES PAR HORIZON")
    print("=" * 70)
    print(f"  {'Horizon':>8} | {'N_train':>8} | {'N_test':>8} | "
          f"{'Features':>8} | {'Split %':>8}")
    print("-" * 70)
    for h, ds in datasets.items():
        total = ds['n_train'] + ds['n_test']
        pct = 100 * ds['n_train'] / total
        print(f"  {h:>8d} | {ds['n_train']:>8d} | {ds['n_test']:>8d} | "
              f"{ds['window_size']:>8d} | {pct:>7.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    from generate_henon import generate_henon

    df = generate_henon(500)
    series_x = df['x'].values

    print("\nPréparation des jeux de données supervisés...")
    datasets = prepare_all_horizons(series_x,
                                    window_size=10,
                                    horizons=[1, 3, 10, 20])
    print_dataset_summary(datasets)
