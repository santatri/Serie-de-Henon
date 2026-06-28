"""
==============================================================================
MODULE : train_model.py
PROJET : Prédiction et Modélisation de Séries Temporelles par RNA Multicouches
AUTEUR : Institut Supérieur Polytechnique de Madagascar
DATE   : Juin 2026
==============================================================================

Description :
    Ce module recherche automatiquement l'architecture optimale d'un MLP
    (Multi-Layer Perceptron) en explorant une grille de configurations :
        — Nombre de couches cachées : 1, 2, 3
        — Neurones par couche : 16, 32, 64, 128
        — Fonctions d'activation : tanh, relu, elu
        — Taux d'apprentissage : 0.001, 0.0005

    L'architecture minimisant la MSE de validation est retenue.
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from itertools import product
import time

# Supprimer les avertissements TensorFlow non critiques
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# ─────────────────────────────────────────────────────────────
#  Grille de recherche d'hyperparamètres
# ─────────────────────────────────────────────────────────────
SEARCH_GRID = {
    'n_hidden_layers': [1, 2, 3],
    'neurons_per_layer': [32, 64, 128],
    'activation': ['tanh', 'relu', 'elu'],
    'learning_rate': [0.001, 0.0005],
}


def build_mlp(input_dim: int,
              n_hidden_layers: int,
              neurons_per_layer: int,
              activation: str,
              learning_rate: float,
              dropout_rate: float = 0.1) -> keras.Model:
    """
    Construit un MLP avec régularisation (Dropout + L2).

    Paramètres
    ----------
    input_dim : int
        Dimension de l'entrée (= window_size)
    n_hidden_layers : int
        Nombre de couches cachées
    neurons_per_layer : int
        Nombre de neurones par couche cachée
    activation : str
        Fonction d'activation ('tanh', 'relu', 'elu')
    learning_rate : float
        Taux d'apprentissage Adam
    dropout_rate : float
        Taux de Dropout pour la régularisation

    Retourne
    --------
    keras.Model compilé
    """
    model = keras.Sequential(name="MLP_Henon")

    # Couche d'entrée
    model.add(layers.Input(shape=(input_dim,)))

    # Couches cachées
    for i in range(n_hidden_layers):
        model.add(layers.Dense(
            units=neurons_per_layer,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name=f"hidden_{i+1}"
        ))
        if dropout_rate > 0 and n_hidden_layers > 1:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    # Couche de sortie (régression → activation linéaire)
    model.add(layers.Dense(1, activation='linear', name="output"))

    # Compilation avec optimiseur Adam
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])

    return model


def architecture_search(X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray,
                         input_dim: int,
                         max_epochs: int = 200,
                         patience: int = 20,
                         verbose: bool = False) -> dict:
    """
    Recherche exhaustive de l'architecture optimale sur la grille SEARCH_GRID.

    Retourne
    --------
    dict avec :
        'best_params'   : hyperparamètres optimaux
        'best_val_mse'  : MSE de validation minimale
        'results_df'    : DataFrame de tous les essais
    """
    results = []
    configs = list(product(
        SEARCH_GRID['n_hidden_layers'],
        SEARCH_GRID['neurons_per_layer'],
        SEARCH_GRID['activation'],
        SEARCH_GRID['learning_rate']
    ))

    print("\n" + "=" * 70)
    print(f"  RECHERCHE D'ARCHITECTURE — {len(configs)} configurations à tester")
    print("=" * 70)

    best_val_mse = np.inf
    best_params = None

    for idx, (n_layers, neurons, act, lr) in enumerate(configs):
        t0 = time.time()

        model = build_mlp(
            input_dim=input_dim,
            n_hidden_layers=n_layers,
            neurons_per_layer=neurons,
            activation=act,
            learning_rate=lr
        )

        # Callbacks : arrêt précoce + réduction du LR
        cb_list = [
            callbacks.EarlyStopping(monitor='val_loss',
                                    patience=patience,
                                    restore_best_weights=True,
                                    verbose=0),
            callbacks.ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.5,
                                        patience=10,
                                        min_lr=1e-6,
                                        verbose=0)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=32,
            callbacks=cb_list,
            verbose=0
        )

        # Évaluation sur validation
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        elapsed = time.time() - t0
        epochs_done = len(history.history['loss'])

        # Mise à jour du meilleur modèle
        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_params = {
                'n_hidden_layers': n_layers,
                'neurons_per_layer': neurons,
                'activation': act,
                'learning_rate': lr
            }
            marker = " ◀ MEILLEUR"
        else:
            marker = ""

        results.append({
            'n_layers': n_layers,
            'neurons': neurons,
            'activation': act,
            'lr': lr,
            'val_mse': val_loss,
            'val_mae': val_mae,
            'epochs': epochs_done,
            'time_s': elapsed
        })

        # Affichage progressif
        print(f"  [{idx+1:3d}/{len(configs)}] "
              f"L={n_layers} N={neurons:3d} act={act:4s} "
              f"lr={lr:.4f} → val_MSE={val_loss:.8f}{marker}")

        keras.backend.clear_session()

    results_df = pd.DataFrame(results).sort_values('val_mse').reset_index(drop=True)

    print("\n" + "=" * 70)
    print("  TOP 5 ARCHITECTURES")
    print("=" * 70)
    cols = ['n_layers', 'neurons', 'activation', 'lr', 'val_mse', 'epochs']
    print(results_df[cols].head(5).to_string(
        index=False,
        float_format=lambda v: f"{v:.8f}"
    ))
    print("=" * 70)
    print(f"\n  ★ Architecture optimale sélectionnée :")
    for k, v in best_params.items():
        print(f"      {k:20s} : {v}")
    print(f"      {'val_MSE':20s} : {best_val_mse:.8f}")

    return {
        'best_params': best_params,
        'best_val_mse': best_val_mse,
        'results_df': results_df
    }


def train_final_model(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       best_params: dict,
                       input_dim: int,
                       max_epochs: int = 500,
                       patience: int = 30) -> tuple:
    """
    Entraîne le modèle final avec l'architecture optimale sur toutes
    les données d'entraînement.

    Retourne
    --------
    (model, history)
    """
    print("\n" + "=" * 70)
    print("  ENTRAÎNEMENT FINAL DU MODÈLE OPTIMAL")
    print("=" * 70)

    model = build_mlp(
        input_dim=input_dim,
        n_hidden_layers=best_params['n_hidden_layers'],
        neurons_per_layer=best_params['neurons_per_layer'],
        activation=best_params['activation'],
        learning_rate=best_params['learning_rate']
    )

    model.summary()

    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss',
                                patience=patience,
                                restore_best_weights=True,
                                verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.3,
                                    patience=15,
                                    min_lr=1e-7,
                                    verbose=0),
        callbacks.ModelCheckpoint('best_model.keras',
                                  save_best_only=True,
                                  monitor='val_loss',
                                  verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=32,
        callbacks=cb_list,
        verbose=1
    )

    final_val_loss, final_val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  Modèle final — val_MSE : {final_val_loss:.8f}")
    print(f"  Modèle final — val_MAE : {final_val_mae:.8f}")

    return model, history


if __name__ == "__main__":
    from generate_henon import generate_henon
    from prepare_dataset import prepare_all_horizons

    df = generate_henon(500)
    datasets = prepare_all_horizons(df['x'].values, window_size=10,
                                    horizons=[1])
    ds = datasets[1]
    X_train, y_train = ds['X_train'], ds['y_train']
    X_test, y_test = ds['X_test'], ds['y_test']

    search = architecture_search(X_train, y_train, X_test, y_test,
                                 input_dim=10, max_epochs=150, patience=15)
    model, hist = train_final_model(X_train, y_train, X_test, y_test,
                                    search['best_params'], input_dim=10)
