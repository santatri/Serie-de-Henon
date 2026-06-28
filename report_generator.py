"""
==============================================================================
MODULE : report_generator.py
PROJET : Prédiction et Modélisation de Séries Temporelles par RNA Multicouches
AUTEUR : Institut Supérieur Polytechnique de Madagascar
DATE   : Juin 2026
==============================================================================

Description :
    Génère un rapport scientifique en Markdown présentant la méthodologie,
    les résultats, les métriques, les figures et les conclusions du projet.

Usage :
    from report_generator import generate_scientific_report
    generate_scientific_report(...)

==============================================================================
"""

import os
from datetime import datetime
import pandas as pd


def _format_value(value, digits=8):
    if pd.isna(value):
        return "nan"
    if isinstance(value, (int, bool)):
        return str(value)
    return f"{value:.{digits}f}"


def _render_hyperparams(params):
    return (
        f"- Nombre de couches cachées : **{params['n_hidden_layers']}**\n"
        f"- Neurones par couche      : **{params['neurons_per_layer']}**\n"
        f"- Activation              : **{params['activation']}**\n"
        f"- Learning rate           : **{params['learning_rate']:.8f}**\n"
    )


def generate_scientific_report(df_henon: pd.DataFrame,
                               datasets: dict,
                               optimal_params: dict,
                               search_results: dict,
                               all_predictions: dict,
                               metrics_df: pd.DataFrame) -> str:
    """Génère et sauvegarde un rapport scientifique en Markdown."""
    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "scientific_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Rapport scientifique — Série de Hénon et MLP\n\n")
        f.write(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Contexte et objectifs\n\n")
        f.write(
            "Ce rapport présente la génération de la série chaotique de Hénon, "
            "la construction d'un jeu de données supervisé pour la série x(n), "
            "la recherche automatique d'une architecture de réseau de neurones "
            "multicouches, puis l'évaluation des performances pour des horizons "
            "de prédiction à 1, 3, 10 et 20 pas.\n\n"
        )

        f.write("## 2. Génération de la série de Hénon\n\n")
        f.write("Paramètres utilisés :\n\n")
        f.write("- a = 1.40000000\n")
        f.write("- b = 0.30000000\n")
        f.write("- Conditions initiales : x₀ = 0.00000000, y₀ = 0.00000000\n")
        f.write("- Nombre de points : 500\n\n")

        f.write("### 2.1 Statistiques descriptives\n\n")
        f.write("| Variable | Min | Max | Moyenne | Écart-type |\n")
        f.write("|----------|-----|-----|---------|------------|\n")
        for col in ["x", "y"]:
            values = df_henon[col]
            f.write(
                f"| {col} | {_format_value(values.min())} | "
                f"{_format_value(values.max())} | {_format_value(values.mean())} | "
                f"{_format_value(values.std())} |\n"
            )
        f.write("\n")

        f.write("## 3. Préparation des jeux de données supervisés\n\n")
        f.write(
            "La série x(n) a été transformée en un jeu supervisé via une fenêtre "
            "glissante de taille 10. Pour chaque horizon h, le modèle prédit x(t+h) "
            "à partir des 10 observations précédentes.\n\n"
        )

        f.write("### 3.1 Résumé des jeux de données par horizon\n\n")
        f.write("| Horizon | Train | Test | Features |\n")
        f.write("|--------:|------:|-----:|---------:|\n")
        for h, ds in datasets.items():
            f.write(
                f"| {h} | {ds['n_train']} | {ds['n_test']} | {ds['window_size']} |\n"
            )
        f.write("\n")

        f.write("## 4. Recherche automatique d'architecture\n\n")
        f.write(
            "Une recherche exhaustive a été menée sur une grille d'hyperparamètres : "
            "nombre de couches cachées (1–3), neurones par couche (32, 64, 128), "
            "fonctions d'activation (tanh, relu, elu) et taux d'apprentissage "
            "(0.0005, 0.001). Le modèle retenu minimise la MSE de validation.\n\n"
        )

        for h, params in optimal_params.items():
            f.write(f"### Horizon h = {h} pas\n\n")
            f.write(_render_hyperparams(params))
            f.write("\n")

        f.write("## 5. Entraînement final et modèles\n\n")
        f.write(
            "Chaque modèle optimal a été entraîné sur le jeu d'entraînement avec "
            "early stopping et réduction du learning rate. Les modèles sont sauvegardés "
            "dans le dossier `results/`.\n\n"
        )

        f.write("## 6. Évaluation des performances\n\n")
        f.write("Les métriques utilisées sont : MSE, RMSE, MAE, R² et MAPE.\n\n")

        f.write("| Horizon | MSE | RMSE | MAE | R² | MAPE (%) |\n")
        f.write("|--------:|----:|-----:|----:|----:|---------:|\n")
        for _, row in metrics_df.iterrows():
            f.write(
                f"| {int(row['Horizon'])} | {_format_value(row['MSE'])} | "
                f"{_format_value(row['RMSE'])} | {_format_value(row['MAE'])} | "
                f"{_format_value(row['R2'])} | {_format_value(row['MAPE'], digits=4)} |\n"
            )
        f.write("\n")

        f.write("## 7. Interprétation scientifique\n\n")
        f.write(
            "La dynamique chaotique de la série de Hénon limite naturellement la fluidité "
            "des prédictions à long terme. Les erreurs croissent avec l'horizon : la prédiction "
            "à court terme (h = 1 ou 3) reste acceptable, tandis que les horizons 10 et 20 pas "
            "montrent une perte de précision significative.\n\n"
        )

        f.write("### 7.1 Analyse par horizon\n\n")
        for h, res in all_predictions.items():
            m = res['metrics']
            f.write(
                f"- h = {h} pas : R² = {_format_value(m['R2'])}, "
                f"RMSE = {_format_value(m['RMSE'])}, MAE = {_format_value(m['MAE'])}.\n"
            )
        f.write("\n")

        f.write("### 7.2 Conclusion\n\n")
        f.write(
            "Le réseau de neurones multicouches démontre qu'il est possible d'approcher "
            "la série chaotique de Hénon pour des horizons courts. La dégradation des métriques "
            "en fonction de l'horizon confirme le caractère non prévisible à long terme du système.\n\n"
        )

        f.write("## 8. Fichiers produits\n\n")
        f.write("- `results/henon_series.csv` : série brute de Hénon.\n")
        for h in optimal_params.keys():
            f.write(f"- `results/model_horizon_{h}.keras` : modèle entraîné pour h={h}.\n")
            f.write(f"- `results/architecture_search_h{h}.csv` : résultats de recherche.\n")
        f.write("- `results/metrics_summary.csv` : tableau résumant les métriques.\n")
        f.write("- `results/scientific_report.md` : rapport scientifique généré.\n")
        f.write("- `figures/*.png` : graphiques de visualisation.\n")

        f.write("\n## 9. Perspectives\n\n")
        f.write(
            "- Tester des architectures récurrentes (LSTM / GRU) pour améliorer les horizons longs.\n"
            "- Étudier l'exposant de Lyapunov pour quantifier la durée de prévisibilité.\n"
            "- Ajouter une analyse de sensibilité aux conditions initiales et au bruit.\n"
        )

    return report_path
