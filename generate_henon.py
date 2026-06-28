"""
==============================================================================
MODULE : generate_henon.py
PROJET : Prédiction et Modélisation de Séries Temporelles par RNA Multicouches
AUTEUR : Institut Supérieur Polytechnique de Madagascar
DATE   : Juin 2026
==============================================================================

Description :
    Ce module génère la série chaotique de Hénon définie par les récurrences :
        x(n+1) = y(n) + 1 - a * x(n)^2
        y(n+1) = b * x(n)
    avec les paramètres standards : a = 1.4, b = 0.3
==============================================================================
"""

import numpy as np
import pandas as pd


def generate_henon(n_points: int = 500,
                   a: float = 1.4,
                   b: float = 0.3,
                   x0: float = 0.0,
                   y0: float = 0.0) -> pd.DataFrame:
    """
    Génère la série de Hénon sur n_points itérations.

    Paramètres
    ----------
    n_points : int
        Nombre de points à générer (défaut : 500)
    a : float
        Paramètre de non-linéarité (défaut : 1.4)
    b : float
        Paramètre de contraction (défaut : 0.3)
    x0 : float
        Condition initiale x (défaut : 0.0)
    y0 : float
        Condition initiale y (défaut : 0.0)

    Retourne
    --------
    pd.DataFrame
        DataFrame avec colonnes ['n', 'x', 'y']
    """
    # Initialisation des tableaux
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Conditions initiales
    x[0] = x0
    y[0] = y0

    # Itération de la récurrence de Hénon
    for n in range(n_points - 1):
        x[n + 1] = y[n] + 1.0 - a * x[n] ** 2
        y[n + 1] = b * x[n]

    # Création du DataFrame avec index temporel
    df = pd.DataFrame({
        'n': np.arange(n_points),
        'x': x,
        'y': y
    })

    # Affichage des statistiques descriptives avec 8 chiffres significatifs
    print("=" * 70)
    print("  SÉRIE DE HÉNON — STATISTIQUES DESCRIPTIVES")
    print("=" * 70)
    print(f"  Paramètres : a = {a:.8f}, b = {b:.8f}")
    print(f"  Conditions initiales : x0 = {x0:.8f}, y0 = {y0:.8f}")
    print(f"  Nombre de points générés : {n_points}")
    print("-" * 70)
    print(f"  x(n) — min    : {x.min():.8f}")
    print(f"  x(n) — max    : {x.max():.8f}")
    print(f"  x(n) — moyenne: {x.mean():.8f}")
    print(f"  x(n) — std    : {x.std():.8f}")
    print("-" * 70)
    print(f"  y(n) — min    : {y.min():.8f}")
    print(f"  y(n) — max    : {y.max():.8f}")
    print(f"  y(n) — moyenne: {y.mean():.8f}")
    print(f"  y(n) — std    : {y.std():.8f}")
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = generate_henon(n_points=500, a=1.4, b=0.3, x0=0.0, y0=0.0)
    print("\nPremières valeurs :")
    print(df.head(10).to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    df.to_csv("henon_series.csv", index=False, float_format="%.8f")
    print("\nSérie sauvegardée dans 'henon_series.csv'")
