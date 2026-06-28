"""
==============================================================================
MAIN.PY — PIPELINE COMPLET
PROJET : Prédiction et Modélisation de Séries Temporelles par RNA Multicouches
AUTEUR : Institut Supérieur Polytechnique de Madagascar — Juin 2026
==============================================================================

DESCRIPTION :
    Script principal exécutant le pipeline complet du projet:
    1. Génération de la série de Hénon (500 points)
    2. Préparation des jeux de données supervisés
    3. Recherche d'architecture optimale pour le MLP
    4. Entraînement des modèles pour chaque horizon (1, 3, 10, 20 pas)
    5. Prédictions multi-horizons
    6. Évaluation avec métriques complètes
    7. Génération de graphiques et rapport scientifique
    8. Sauvegarde des résultats

USAGE :
    python main.py

DURÉE ESTIMÉE : 5-15 minutes (selon machine)

==============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time as time_module

# Imports locaux
from generate_henon import generate_henon
from prepare_dataset import prepare_all_horizons, print_dataset_summary
from train_model import architecture_search, train_final_model
from predict import predict_multi_step, print_metrics_table
from plots import (plot_henon_attractor, plot_learning_curve,
                   plot_predictions, plot_all_horizons,
                   plot_metrics_table, plot_metrics_vs_horizon,
                   plot_residuals_histogram)
from report_generator import generate_scientific_report

# Suppression des avertissements non-critiques
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION GLOBALE
# ══════════════════════════════════════════════════════════════

CONFIG = {
    'n_points_henon': 500,
    'window_size': 10,
    'horizons': [1, 3, 10, 20],
    'test_size': 0.20,
    'max_epochs_search': 200,
    'max_epochs_train': 500,
    'patience': 30,
    'batch_size': 32,
    'random_state': 42
}

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════

def format_timestamp():
    """Retourne un timestamp formaté."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_header(text: str, char: str = "=", width: int = 80):
    """Affiche un en-tête formaté."""
    print("\n" + char * width)
    print(f"  {text.center(width - 4)}")
    print(char * width + "\n")


def print_section(text: str):
    """Affiche un titre de section."""
    print(f"\n{'─' * 80}")
    print(f"  ► {text}")
    print(f"{'─' * 80}\n")


# ══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    """Exécute le pipeline complet du projet."""

    print_header("PRÉDICTION ET MODÉLISATION DE SÉRIES TEMPORELLES")
    print_header("PAR RÉSEAUX DE NEURONES ARTIFICIELS MULTICOUCHES", "─")
    print_header(f"Démarrage : {format_timestamp()}", "─")

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 1 : Génération de la série de Hénon
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 1 : GÉNÉRATION DE LA SÉRIE DE HÉNON")
    t0 = time_module.time()

    df_henon = generate_henon(
        n_points=CONFIG['n_points_henon'],
        a=1.4,
        b=0.3,
        x0=0.0,
        y0=0.0
    )

    t1 = time_module.time()
    print(f"\n  ✓ Série générée en {t1 - t0:.4f} s")
    print(f"  Dimension : {len(df_henon)} points")
    print(f"  Sauvegarde : results/henon_series.csv")

    df_henon.to_csv(os.path.join(OUTPUT_DIR, "henon_series.csv"),
                    index=False, float_format="%.8f")

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 2 : Tracé de l'attracteur de Hénon
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 2 : VISUALISATION DE L'ATTRACTEUR DE HÉNON")
    t0 = time_module.time()

    plot_path = plot_henon_attractor(df_henon)

    t1 = time_module.time()
    print(f"\n  ✓ Graphique généré en {t1 - t0:.4f} s")

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 3 : Préparation des jeux de données supervisés
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 3 : PRÉPARATION DES JEUX DE DONNÉES SUPERVISÉS")
    t0 = time_module.time()

    series_x = df_henon['x'].values
    datasets = prepare_all_horizons(
        series_x,
        window_size=CONFIG['window_size'],
        horizons=CONFIG['horizons'],
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )

    print_dataset_summary(datasets)

    t1 = time_module.time()
    print(f"\n  ✓ Jeux de données préparés en {t1 - t0:.4f} s")

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 4 : Recherche d'architecture optimale pour chaque horizon
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 4 : RECHERCHE D'ARCHITECTURE OPTIMALE")
    print("  Cela peut prendre plusieurs minutes...\n")

    optimal_params = {}
    search_results = {}

    for h in CONFIG['horizons']:
        print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
        print(f"  ║  RECHERCHE POUR HORIZON h = {h} PAS")
        print(f"  ╚═══════════════════════════════════════════════════════════════╝")

        ds = datasets[h]
        t0 = time_module.time()

        search_result = architecture_search(
            ds['X_train'],
            ds['y_train'],
            ds['X_test'],
            ds['y_test'],
            input_dim=CONFIG['window_size'],
            max_epochs=CONFIG['max_epochs_search'],
            patience=20,
            verbose=False
        )

        optimal_params[h] = search_result['best_params']
        search_results[h] = search_result

        t1 = time_module.time()
        print(f"\n  ✓ Recherche complétée en {t1 - t0:.2f} s")
        print(f"    MSE minimal trouvée : {search_result['best_val_mse']:.8f}")

        # Sauvegarde des résultats de recherche
        results_df = search_result['results_df']
        results_df.to_csv(
            os.path.join(OUTPUT_DIR, f"architecture_search_h{h}.csv"),
            index=False,
            float_format="%.8f"
        )

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 5 : Entraînement des modèles optimaux
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 5 : ENTRAÎNEMENT DES MODÈLES OPTIMAUX")

    trained_models = {}
    histories = {}

    for h in CONFIG['horizons']:
        print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
        print(f"  ║  ENTRAÎNEMENT POUR HORIZON h = {h} PAS")
        print(f"  ╚═══════════════════════════════════════════════════════════════╝")

        ds = datasets[h]
        t0 = time_module.time()

        model, history = train_final_model(
            ds['X_train'],
            ds['y_train'],
            ds['X_test'],
            ds['y_test'],
            optimal_params[h],
            input_dim=CONFIG['window_size'],
            max_epochs=CONFIG['max_epochs_train'],
            patience=CONFIG['patience']
        )

        trained_models[h] = model
        histories[h] = history

        t1 = time_module.time()
        print(f"\n  ✓ Modèle entraîné en {t1 - t0:.2f} s")

        # Sauvegarde du modèle
        model_path = os.path.join(OUTPUT_DIR, f"model_horizon_{h}.keras")
        model.save(model_path)
        print(f"    Modèle sauvegardé : {model_path}")

        # Graphique de convergence
        plot_learning_curve(history, horizon=h)

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 6 : Prédictions et évaluations
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 6 : PRÉDICTIONS MULTI-HORIZONS ET ÉVALUATION")

    all_predictions = {}

    for h in CONFIG['horizons']:
        print(f"\n  ─ Horizon h = {h} pas")

        ds = datasets[h]
        model = trained_models[h]

        result = predict_multi_step(
            model,
            ds,
            ds['scaler_X'],
            ds['scaler_y'],
            horizon=h,
            label=f"PRÉDICTION À {h} PAS"
        )

        all_predictions[h] = result

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 7 : Tableau récapitulatif des métriques
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 7 : TABLEAU COMPARATIF DES MÉTRIQUES")

    metrics_df = print_metrics_table(all_predictions)

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 8 : Génération des graphiques de prédiction
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 8 : GÉNÉRATION DES GRAPHIQUES DE PRÉDICTION")

    print("  Génération des graphiques de prédiction vs réalité...")
    for h in CONFIG['horizons']:
        result = all_predictions[h]
        plot_predictions(
            result['y_true'],
            result['y_pred'],
            horizon=h,
            metrics=result['metrics']
        )

    print("  Génération des graphiques de comparaison multi-horizons...")
    plot_all_horizons(all_predictions)

    # Tableau des métriques et distribution des résidus
    plot_metrics_table(metrics_df)
    plot_metrics_vs_horizon(metrics_df)
    plot_residuals_histogram(all_predictions)

    print("\n  ✓ Tous les graphiques ont été générés")

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 9 : Génération du rapport scientifique
    # ─────────────────────────────────────────────────────────────
    print_section("ÉTAPE 9 : GÉNÉRATION DU RAPPORT SCIENTIFIQUE")

    report_path = generate_scientific_report(
        df_henon,
        datasets,
        optimal_params,
        search_results,
        all_predictions,
        metrics_df
    )

    print(f"  ✓ Rapport scientifique généré : {report_path}")

    # ─────────────────────────────────────────────────────────────
    # RÉSUMÉ FINAL
    # ─────────────────────────────────────────────────────────────
    print_header("RÉSUMÉ FINAL DU PROJET")

    print(f"  Série de Hénon générée           : {len(df_henon)} points")
    print(f"  Fenêtre glissante                : {CONFIG['window_size']} pas")
    print(f"  Horizons de prédiction évalués   : {CONFIG['horizons']}")
    print(f"  Configurations testées           : 3×3×3×2 = 54 architectures/horizon")
    print(f"  Total configurations             : 54 × 4 = 216 modèles")

    print(f"\n  Fichiers de résultats générés :")
    print(f"    • Série brute                  : results/henon_series.csv")
    print(f"    • Architectures optimales      : results/architecture_search_h*.csv")
    print(f"    • Métriques consolidées        : results/metrics_summary.csv")
    print(f"    • Graphiques                   : figures/*.png")
    print(f"    • Rapport scientifique         : {report_path}")

    print_header(f"Fin : {format_timestamp()}", "─")
    print_header("EXÉCUTION RÉUSSIE ✓", "═")


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⚠ Exécution interrompue par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  ✗ Erreur fatale : {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
