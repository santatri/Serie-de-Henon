"""
==============================================================================
app.py — Interface Streamlit
PROJET : Prédiction et Modélisation de Séries Temporelles par RNA Multicouches
AUTEUR : Institut Supérieur Polytechnique de Madagascar — Juin 2026
==============================================================================
Usage :
    streamlit run app.py
==============================================================================
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set TensorFlow log suppression and oneDNN flags before importing TensorFlow.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Import TensorFlow with a graceful fallback message if it's not installed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    tf.get_logger().setLevel('ERROR')
except Exception:
    st.error(
        "Missing dependency: TensorFlow is not installed in this Python environment.\n\n"
        "Install it with `py -m pip install -r requirements.txt` or `py -m pip install tensorflow`."
    )
    st.stop()
import time

warnings.filterwarnings('ignore')

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Série de Hénon — RNA Multicouches",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personnalisé ─────────────────────────────────────────
st.markdown("""
<style>
/* En-tête principal */
.main-header {
    background: linear-gradient(135deg, #1f4e79 0%, #2e75b6 50%, #1f4e79 100%);
    padding: 1.8rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.main-header h1 { color: #ffffff; font-size: 1.55rem; font-weight: 700; margin: 0; }
.main-header p  { color: #bdd7ee; font-size: 0.88rem; margin: 0.3rem 0 0; }

/* Carte métrique */
.metric-card {
    background: #f0f4fa;
    border-left: 5px solid #2e75b6;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
}
.metric-card .label { font-size: 0.78rem; color: #555; font-weight: 600; text-transform: uppercase; }
.metric-card .value { font-size: 1.3rem; color: #1f4e79; font-weight: 700; font-family: monospace; }

/* Section title */
.section-title {
    font-size: 1.1rem; font-weight: 700;
    color: #1f4e79; border-bottom: 2px solid #2e75b6;
    padding-bottom: 0.3rem; margin-bottom: 1rem;
}

/* Badge qualité */
.badge-excellent { background:#1a7a4a; color:white; border-radius:6px; padding:2px 10px; font-size:0.8rem; }
.badge-good      { background:#2e75b6; color:white; border-radius:6px; padding:2px 10px; font-size:0.8rem; }
.badge-acceptable{ background:#d07e00; color:white; border-radius:6px; padding:2px 10px; font-size:0.8rem; }
.badge-limited   { background:#c0392b; color:white; border-radius:6px; padding:2px 10px; font-size:0.8rem; }

/* Sidebar labels */
.sidebar-section { font-size: 0.82rem; color: #888; text-transform: uppercase;
                   font-weight: 700; margin: 1rem 0 0.3rem; letter-spacing: 0.06em; }

/* Info box */
.info-box {
    background:#e8f4fd; border-left:4px solid #2e75b6;
    border-radius:6px; padding:0.7rem 1rem;
    font-size:0.88rem; color:#1f4e79; margin:0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FONCTIONS INTÉGRÉES (sans import des autres modules)
# ══════════════════════════════════════════════════════════════

# ── Couleurs ──────────────────────────────────────────────────
COLORS = {
    'true':   '#1f77b4',
    'pred':   '#d62728',
    'train':  '#2ca02c',
    'val':    '#ff7f0e',
    'accent': '#9467bd',
    'neutral':'#7f7f7f',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.titleweight': 'bold', 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
    'lines.linewidth': 1.4, 'legend.fontsize': 8,
    'figure.dpi': 110,
})


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Génération Hénon ──────────────────────────────────────────
@st.cache_data
def generate_henon(n_points=500, a=1.4, b=0.3, x0=0.0, y0=0.0):
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    x[0], y[0] = x0, y0
    for n in range(n_points - 1):
        x[n+1] = y[n] + 1.0 - a * x[n]**2
        y[n+1] = b * x[n]
        # Détection de divergence : la série s'échappe vers l'infini
        if not (np.isfinite(x[n+1]) and np.isfinite(y[n+1])):
            return None  # série divergée
    df = pd.DataFrame({'n': np.arange(n_points), 'x': x, 'y': y})
    # Vérification globale (NaN via conditions initiales)
    if not (np.all(np.isfinite(df['x'])) and np.all(np.isfinite(df['y']))):
        return None
    return df


# ── Jeu de données supervisé ──────────────────────────────────
def create_supervised(series, window_size=10, horizon=1):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i: i + window_size])
        y.append(series[i + window_size + horizon - 1])
    return np.array(X), np.array(y)


def prepare_dataset(series, window_size=10, horizon=1, test_size=0.2):
    from sklearn.preprocessing import MinMaxScaler
    X, y = create_supervised(series, window_size, horizon)
    # Garde de sécurité : supprimer les lignes avec inf/NaN
    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[finite_mask], y[finite_mask]
    if len(X) == 0:
        raise ValueError("La série contient trop de valeurs non finies pour construire un dataset.")
    scX = MinMaxScaler((-1, 1)); scy = MinMaxScaler((-1, 1))
    Xs = scX.fit_transform(X)
    ys = scy.fit_transform(y.reshape(-1, 1)).ravel()
    split = int(len(Xs) * (1 - test_size))
    return (Xs[:split], Xs[split:], ys[:split], ys[split:],
            scX, scy, y)


# ── Construction MLP ──────────────────────────────────────────
def build_mlp(input_dim, n_layers, neurons, activation, lr):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for i in range(n_layers):
        model.add(layers.Dense(neurons, activation=activation,
                               kernel_regularizer=keras.regularizers.l2(1e-4)))
        if n_layers > 1:
            model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    return model


# ── Métriques ─────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 1e-10
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 if mask.sum()>0 else np.nan
    return {'MSE':mse,'RMSE':rmse,'MAE':mae,'R2':r2,'MAPE':mape}


# ── Figures ───────────────────────────────────────────────────
def fig_attractor(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Série de Hénon — Attracteur & Évolution temporelle",
                 fontweight='bold', fontsize=11)
    ax = axes[0]
    sc = ax.scatter(df['x'], df['y'], c=np.arange(len(df)),
                    cmap='plasma', s=5, alpha=0.8, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Indice n")
    ax.set_xlabel("x(n)"); ax.set_ylabel("y(n)")
    ax.set_title("Attracteur de Hénon (y(n) vs x(n))")

    ax2 = axes[1]
    ax2.plot(df['n'], df['x'], color=COLORS['true'], lw=0.9, label="x(n)")
    ax2.plot(df['n'], df['y'], color=COLORS['pred'], lw=0.9, label="y(n)", alpha=0.8)
    ax2.set_xlabel("n"); ax2.set_ylabel("Valeur")
    ax2.set_title("Évolution temporelle")
    ax2.legend()
    fig.tight_layout()
    return fig


def fig_learning(history, h):
    loss = history.history['loss']
    val  = history.history['val_loss']
    ep   = np.arange(1, len(loss)+1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    fig.suptitle(f"Courbes d'apprentissage — Horizon h={h}", fontweight='bold')
    axes[0].plot(ep, loss,  color=COLORS['train'], label="Train MSE")
    axes[0].plot(ep, val,   color=COLORS['val'],   label="Val MSE", ls='--')
    axes[0].set_xlabel("Époques"); axes[0].set_ylabel("MSE (log)")
    axes[0].set_title("MSE (échelle log)"); axes[0].legend()
    axes[0].set_yscale('log')
    n_zoom = max(10, len(loss)//3)
    axes[1].plot(ep[-n_zoom:], loss[-n_zoom:], color=COLORS['train'], label="Train")
    axes[1].plot(ep[-n_zoom:], val[-n_zoom:],  color=COLORS['val'],   label="Val", ls='--')
    axes[1].set_xlabel("Époques (zoom)"); axes[1].set_title("Convergence finale")
    axes[1].legend()
    fig.tight_layout()
    return fig


def fig_predictions(y_true, y_pred, h, metrics, n_show=100):
    n = min(n_show, len(y_true))
    idx = np.arange(n)
    resid = y_true[:n] - y_pred[:n]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    fig.suptitle(
        f"Prédiction à {h} pas — MLP  |  "
        f"MSE={metrics['MSE']:.6f}  RMSE={metrics['RMSE']:.6f}  "
        f"MAE={metrics['MAE']:.6f}  R²={metrics['R2']:.6f}",
        fontsize=10, fontweight='bold'
    )
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(idx, y_true[:n], color=COLORS['true'], lw=1.2, label="Valeurs réelles")
    ax1.plot(idx, y_pred[:n], color=COLORS['pred'], lw=1.2, ls='--',
             label=f"Valeurs prédites (h={h})")
    ax1.fill_between(idx, y_true[:n], y_pred[:n], alpha=0.13, color=COLORS['neutral'])
    ax1.set_xlabel("Indice temporel"); ax1.set_ylabel("x(n)")
    ax1.set_title(f"Réel vs Prédit — {n} premiers points de test")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    lim = [min(y_true.min(), y_pred.min())*1.05, max(y_true.max(), y_pred.max())*1.05]
    ax2.scatter(y_true, y_pred, alpha=0.35, s=10, color=COLORS['accent'])
    ax2.plot(lim, lim, 'k--', lw=1.5, label="Identité")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("Réelles"); ax2.set_ylabel("Prédites")
    ax2.set_title("Scatter plot"); ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(idx, resid, color=COLORS['neutral'], alpha=0.7)
    ax3.axhline(0, color='black', lw=1.0)
    ax3.axhline( resid.std(), color=COLORS['pred'], ls=':', lw=1.5, label="±1σ")
    ax3.axhline(-resid.std(), color=COLORS['pred'], ls=':', lw=1.5)
    ax3.set_xlabel("Indice"); ax3.set_ylabel("Résidu")
    ax3.set_title("Erreurs de prédiction"); ax3.legend(fontsize=7)
    return fig


def fig_all_horizons(all_results, n_show=80):
    horizons = sorted(all_results.keys())
    n_h = len(horizons)
    # Adapt grid: 1 row if ≤2 horizons, else 2×2
    if n_h <= 2:
        nrows, ncols = 1, max(n_h, 1)
    else:
        nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9), squeeze=False)
    axes = axes.ravel()
    fig.suptitle("Comparaison multi-horizons — MLP / Série de Hénon",
                 fontsize=12, fontweight='bold')
    for i, h in enumerate(horizons):
        ax = axes[i]
        r = all_results[h]
        n = min(n_show, len(r['y_true']))
        idx = np.arange(n)
        ax.plot(idx, r['y_true'][:n], color=COLORS['true'], lw=1.1, label="Réel")
        ax.plot(idx, r['y_pred'][:n], color=COLORS['pred'], lw=1.1, ls='--',
                label=f"Prédit (h={h})")
        ax.fill_between(idx, r['y_true'][:n], r['y_pred'][:n],
                        alpha=0.1, color=COLORS['neutral'])
        m = r['metrics']
        ax.set_title(f"h={h} pas  |  RMSE={m['RMSE']:.5f}  R²={m['R2']:.5f}", fontsize=9)
        ax.set_xlabel("Indice"); ax.set_ylabel("x(n)")
        ax.legend(fontsize=7)
    # Hide unused subplots
    for j in range(n_h, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout(rect=[0,0,1,0.95])
    return fig


def fig_metrics_bars(metrics_df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Métriques vs Horizon de prédiction", fontweight='bold', fontsize=11)
    h = metrics_df['Horizon'].values
    pairs = [('RMSE','RMSE',COLORS['pred']),('MAE','MAE',COLORS['accent']),('R2','R²',COLORS['train'])]
    for ax,(col,lbl,c) in zip(axes,pairs):
        bars = ax.bar(h.astype(str), metrics_df[col].values, color=c, alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, metrics_df[col].values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xlabel("Horizon h"); ax.set_ylabel(lbl); ax.set_title(f"{lbl} par horizon")
    fig.tight_layout()
    return fig


def fig_residuals_hist(all_results):
    horizons = sorted(all_results.keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(13, 4), squeeze=False)
    axes = axes.ravel()
    fig.suptitle("Distribution des résidus par horizon", fontweight='bold', fontsize=11)
    for ax, h in zip(axes, horizons):
        resid = all_results[h]['y_true'] - all_results[h]['y_pred']
        ax.hist(resid, bins=30, color=COLORS['accent'], alpha=0.75, edgecolor='white')
        ax.axvline(resid.mean(), color=COLORS['pred'], lw=2, ls='--',
                   label=f"μ={resid.mean():.4f}")
        ax.axvline(resid.mean()+resid.std(), color='gray', lw=1.5, ls=':',
                   label=f"σ={resid.std():.4f}")
        ax.axvline(resid.mean()-resid.std(), color='gray', lw=1.5, ls=':')
        ax.set_title(f"h={h} pas"); ax.set_xlabel("Résidu"); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
for key in ['henon_df','trained_models','all_results','metrics_df','histories']:
    if key not in st.session_state:
        st.session_state[key] = None


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem;background:#1f4e79;
    border-radius:10px;margin-bottom:1rem;'>
    <span style='font-size:2rem;'>🧠</span><br>
    <span style='color:#fff;font-weight:700;font-size:0.95rem;'>RNA Multicouches</span><br>
    <span style='color:#bdd7ee;font-size:0.75rem;'>Série de Hénon</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ Paramètres de Hénon</div>', unsafe_allow_html=True)
    n_points  = st.slider("Nombre de points",    100, 1000, 500, 50)
    a_param   = st.number_input("Paramètre a",   value=1.4, step=0.05, format="%.2f")
    b_param   = st.number_input("Paramètre b",   value=0.3, step=0.05, format="%.2f")

    st.markdown('<div class="sidebar-section">🪟 Fenêtre temporelle</div>', unsafe_allow_html=True)
    window_size = st.slider("Taille de fenêtre", 5, 30, 10)
    test_size   = st.slider("% données test",    10, 40, 20) / 100

    st.markdown('<div class="sidebar-section">🧱 Architecture MLP</div>', unsafe_allow_html=True)
    n_layers  = st.selectbox("Couches cachées",      [1, 2, 3], index=1)
    neurons   = st.selectbox("Neurones / couche",    [16, 32, 64, 128], index=2)
    activation= st.selectbox("Activation",           ['tanh', 'relu', 'elu'], index=0)
    lr        = st.select_slider("Learning rate",    [0.0005, 0.001, 0.002, 0.005], value=0.001)
    max_epochs= st.slider("Époques max",             50, 500, 200, 25)
    patience  = st.slider("Patience (early stop)",   10, 60, 25, 5)

    st.markdown('<div class="sidebar-section">🎯 Horizons de prédiction</div>', unsafe_allow_html=True)
    do_h1  = st.checkbox("h = 1  pas",  value=True)
    do_h3  = st.checkbox("h = 3  pas",  value=True)
    do_h10 = st.checkbox("h = 10 pas",  value=True)
    do_h20 = st.checkbox("h = 20 pas",  value=True)
    horizons = [h for h,v in [(1,do_h1),(3,do_h3),(10,do_h10),(20,do_h20)] if v]

    st.markdown("---")
    run_btn = st.button("▶  Lancer le pipeline", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  EN-TÊTE PRINCIPAL
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🧠 Prédiction de Séries Temporelles par Réseaux de Neurones Multicouches</h1>
  <p>Institut Supérieur Polytechnique de Madagascar — Mini Projet Juin 2026
  &nbsp;|&nbsp; Filières : ESIIA4 · IGGLIA4 · IMTICIA4 · ISAIA4</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════
if run_btn:
    if not horizons:
        st.error("⚠️ Veuillez sélectionner au moins un horizon de prédiction.")
        st.stop()

    # ── Génération de la série ────────────────────────────
    with st.spinner("Génération de la série de Hénon..."):
        df = generate_henon(n_points, a_param, b_param)

    if df is None:
        st.error(
            f"⚠️ **Série divergée** : avec a={a_param:.2f} et b={b_param:.2f}, "
            "la carte de Hénon s'échappe vers l'infini (bassin d'attraction dépassé).\n\n"
            "**Solutions** :\n"
            "- Utilisez **a = 1.4** et **b = 0.3** (paramètres standards)\n"
            "- Réduisez la valeur de **a** (la divergence survient typiquement pour a > 1.56)\n"
            "- Maintenez **b ∈ [0.1, 0.4]** pour rester dans le bassin chaotique"
        )
        st.stop()

    st.session_state['henon_df'] = df

    # ── Entraînement par horizon ──────────────────────────
    all_results  = {}
    all_histories= {}

    prog_bar = st.progress(0, text="Initialisation...")

    for step_i, h in enumerate(horizons):
        prog_bar.progress(
            int((step_i / len(horizons)) * 90),
            text=f"Entraînement horizon h={h} pas..."
        )
        keras.backend.clear_session()

        X_tr, X_te, y_tr, y_te, scX, scy, y_raw = prepare_dataset(
            df['x'].values, window_size, h, test_size
        )

        model = build_mlp(window_size, n_layers, neurons, activation, lr)

        cb_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                                        patience=12, min_lr=1e-7, verbose=0),
        ]

        history = model.fit(
            X_tr, y_tr, validation_data=(X_te, y_te),
            epochs=max_epochs, batch_size=32,
            callbacks=cb_list, verbose=0
        )

        y_pred_norm = model.predict(X_te, verbose=0).ravel()
        y_true = scy.inverse_transform(y_te.reshape(-1, 1)).ravel()
        y_pred = scy.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
        metrics = compute_metrics(y_true, y_pred)

        all_results[h]   = {'y_true': y_true, 'y_pred': y_pred, 'metrics': metrics}
        all_histories[h] = history

    prog_bar.progress(100, text="Pipeline terminé ✓")
    time.sleep(0.4)
    prog_bar.empty()

    st.session_state['all_results']  = all_results
    st.session_state['histories']    = all_histories

    # ── Tableau métriques ────────────────────────────────
    rows = []
    for h, res in all_results.items():
        m = res['metrics']
        rows.append({'Horizon': h, 'MSE': m['MSE'], 'RMSE': m['RMSE'],
                     'MAE': m['MAE'], 'R2': m['R2'], 'MAPE': m['MAPE']})
    st.session_state['metrics_df'] = pd.DataFrame(rows)

    st.success("✅ Pipeline terminé avec succès !")


# ══════════════════════════════════════════════════════════════
#  AFFICHAGE DES RÉSULTATS (onglets)
# ══════════════════════════════════════════════════════════════

if st.session_state['henon_df'] is not None:
    df          = st.session_state['henon_df']
    all_results = st.session_state['all_results']
    histories   = st.session_state['histories']
    metrics_df  = st.session_state['metrics_df']

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌀 Attracteur",
        "📈 Apprentissage",
        "🎯 Prédictions",
        "📊 Métriques",
        "🔬 Interprétation",
        "💾 Export"
    ])

    # ── Onglet 1 : Attracteur ─────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Attracteur de Hénon & Série temporelle</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Points générés", f"{len(df):,}")
        with col2:
            st.metric("x(n) — min / max",
                      f"{df['x'].min():.4f} / {df['x'].max():.4f}")
        with col3:
            st.metric("y(n) — min / max",
                      f"{df['y'].min():.4f} / {df['y'].max():.4f}")
        with col4:
            st.metric("Paramètres",f"a={a_param}, b={b_param}")

        fig = fig_attractor(df)
        st.pyplot(fig, use_container_width=True)

        with st.expander("📋 Premières et dernières valeurs (8 chiffres)"):
            show_df = pd.concat([df.head(10), df.tail(10)])
            st.dataframe(show_df.style.format({'x': '{:.8f}', 'y': '{:.8f}'}),
                         use_container_width=True)

    # ── Onglet 2 : Apprentissage ──────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Courbes d\'apprentissage par horizon</div>',
                    unsafe_allow_html=True)

        if all_results and histories:
            st.markdown(f"""
            <div class="info-box">
            Architecture : <b>{n_layers} couche(s) cachée(s)</b> · 
            <b>{neurons} neurones</b> · activation <b>{activation}</b> · 
            lr = <b>{lr}</b> · fenêtre = <b>{window_size}</b>
            </div>
            """, unsafe_allow_html=True)

            for h in sorted(histories.keys()):
                hist = histories[h]
                n_ep = len(hist.history['loss'])
                best_val = min(hist.history['val_loss'])
                st.markdown(f"**Horizon h = {h} pas** — {n_ep} époques | "
                            f"Meilleure val_MSE = `{best_val:.8f}`")
                fig = fig_learning(hist, h)
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Lancez le pipeline pour afficher les courbes d'apprentissage.")

    # ── Onglet 3 : Prédictions ────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Prédictions vs Valeurs réelles</div>',
                    unsafe_allow_html=True)

        if all_results:
            n_show = st.slider("Points affichés", 30, 200, 100, 10)

            if len(all_results) > 1:
                st.markdown("### Vue d'ensemble — Tous horizons")
                fig = fig_all_horizons(all_results, n_show)
                st.pyplot(fig, use_container_width=True)
                st.markdown("---")

            for h in sorted(all_results.keys()):
                res = all_results[h]
                m   = res['metrics']
                st.markdown(f"### Horizon h = {h} pas")
                fig = fig_predictions(res['y_true'], res['y_pred'],
                                      h, m, n_show)
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Lancez le pipeline pour afficher les prédictions.")

    # ── Onglet 4 : Métriques ──────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Tableau comparatif des métriques</div>',
                    unsafe_allow_html=True)

        if metrics_df is not None and all_results:
            # Tableau stylisé
            st.markdown("#### 📋 Métriques numériques (8 chiffres significatifs)")
            styled = metrics_df.style.format({
                'Horizon': '{:.0f}',
                'MSE':  '{:.8f}',
                'RMSE': '{:.8f}',
                'MAE':  '{:.8f}',
                'R2':   '{:.8f}',
                'MAPE': '{:.4f}'
            }).background_gradient(subset=['RMSE','MAE'], cmap='RdYlGn_r') \
              .background_gradient(subset=['R2'], cmap='RdYlGn') \
              .set_properties(**{'font-family':'monospace','font-size':'13px'})
            st.dataframe(styled, use_container_width=True)

            # Cartes métriques par horizon
            st.markdown("#### 📦 Détail par horizon")
            cols = st.columns(len(all_results))
            for col, h in zip(cols, sorted(all_results.keys())):
                m = all_results[h]['metrics']
                r2 = m['R2']
                if r2 > 0.95:
                    badge = "🟢 EXCELLENTE"
                elif r2 > 0.80:
                    badge = "🔵 BONNE"
                elif r2 > 0.50:
                    badge = "🟡 ACCEPTABLE"
                else:
                    badge = "🔴 LIMITÉE"
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Horizon h = {h} pas</div>
                        <div class="value">R² = {r2:.6f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">MSE</div>
                        <div class="value">{m['MSE']:.6f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">RMSE</div>
                        <div class="value">{m['RMSE']:.6f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">MAE</div>
                        <div class="value">{m['MAE']:.6f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">MAPE</div>
                        <div class="value">{m['MAPE']:.2f} %</div>
                    </div>
                    <div style="margin-top:0.5rem;font-size:0.82rem;font-weight:600;">
                        {badge}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 📊 Visualisation des métriques")
            fig = fig_metrics_bars(metrics_df)
            st.pyplot(fig, use_container_width=True)
            fig2 = fig_residuals_hist(all_results)
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("Lancez le pipeline pour afficher les métriques.")

    # ── Onglet 5 : Interprétation ─────────────────────────
    with tab5:
        st.markdown('<div class="section-title">🔬 Interprétation scientifique</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        ### 1. Attracteur de Hénon
        La série de Hénon (a=1.4, b=0.3) est un **système dynamique discret chaotique**
        introduit par Michel Hénon en 1976. Son attracteur présente une structure
        **fractale** caractéristique visible dans le plan de phase (x(n), y(n)).
        La dynamique chaotique se manifeste par une **forte sensibilité aux conditions
        initiales**, rendant la prédiction à long terme intrinsèquement limitée.
        """)

        if all_results and metrics_df is not None:
            st.markdown("### 2. Analyse des prédictions par horizon")

            for h in sorted(all_results.keys()):
                m  = all_results[h]['metrics']
                r2 = m['R2']
                rmse = m['RMSE']

                if r2 > 0.95:
                    qualite = "🟢 EXCELLENTE (R² > 0.95)"
                    comment = "Le modèle capture fidèlement la dynamique locale du système chaotique."
                elif r2 > 0.80:
                    qualite = "🔵 BONNE (0.80 < R² ≤ 0.95)"
                    comment = "La prédiction est fiable malgré la nature chaotique de la série."
                elif r2 > 0.50:
                    qualite = "🟡 ACCEPTABLE (0.50 < R² ≤ 0.80)"
                    comment = "La divergence des trajectoires chaotiques s'accentue progressivement."
                else:
                    qualite = "🔴 LIMITÉE (R² ≤ 0.50)"
                    comment = "L'horizon dépasse la limite prédictive du système chaotique."

                with st.expander(f"Horizon h = {h} pas — {qualite}"):
                    st.markdown(f"""
                    | Métrique | Valeur (8 chiffres) |
                    |----------|---------------------|
                    | MSE      | `{m['MSE']:.8f}`    |
                    | RMSE     | `{m['RMSE']:.8f}`   |
                    | MAE      | `{m['MAE']:.8f}`    |
                    | R²       | `{m['R2']:.8f}`     |
                    | MAPE     | `{m['MAPE']:.4f} %` |

                    **Commentaire :** {comment}
                    """)

        st.markdown("""
        ### 3. Remarques sur la stabilité — *Stabile ?*

        > La série de Hénon est un **attracteur étrange** : bien que déterministe,
        > elle est sensible aux perturbations infinitésimales.

        | Horizon | Comportement | Explication |
        |---------|-------------|-------------|
        | h = 1  | 🟢 Stable   | Dynamique locale bien approximée par le MLP |
        | h = 3  | 🔵 Fiable   | Légère accumulation des erreurs d'approximation |
        | h = 10 | 🟡 Dégradé  | Exposition exponentielle des erreurs — chaos |
        | h = 20 | 🔴 Diverge  | Convergence vers l'attracteur moyen, pas la trajectoire exacte |

        **Conclusion :** le MLP multicouche offre des prédictions **stables à court terme**
        (h ≤ 3), mais la prévisibilité diminue exponentiellement avec l'horizon,
        conformément à la théorie des systèmes chaotiques (exposant de Lyapunov positif).

        ### 4. Perspectives
        - Intégration de **réseaux récurrents** (LSTM, GRU) pour h ≥ 10
        - Analyse de l'**exposant de Lyapunov** pour borner l'horizon prédictif
        - Méthodes d'**ensemble** (bootstrap) pour quantifier l'incertitude prédictive
        """)

    # ── Onglet 6 : Export ────────────────────────────────
    with tab6:
        st.markdown('<div class="section-title">💾 Export des résultats</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Série de Hénon")
            csv_henon = df.to_csv(index=False, float_format="%.8f")
            st.download_button(
                "⬇️ Télécharger henon_series.csv",
                csv_henon, "henon_series.csv", "text/csv",
                use_container_width=True
            )

        with col_b:
            if metrics_df is not None:
                st.markdown("#### Tableau des métriques")
                csv_metrics = metrics_df.to_csv(index=False, float_format="%.8f")
                st.download_button(
                    "⬇️ Télécharger metrics_summary.csv",
                    csv_metrics, "metrics_summary.csv", "text/csv",
                    use_container_width=True
                )

        if all_results and metrics_df is not None:
            st.markdown("#### Figures")
            fig_choices = {
                "Attracteur de Hénon": fig_attractor(df),
                "Comparaison horizons": fig_all_horizons(all_results),
                "Métriques barres":    fig_metrics_bars(metrics_df),
                "Résidus histogrammes":fig_residuals_hist(all_results),
            }
            for name, fig in fig_choices.items():
                buf = fig_to_bytes(fig)
                filename = name.lower().replace(" ", "_") + ".png"
                st.download_button(
                    f"⬇️ {name}.png",
                    buf, filename, "image/png",
                    use_container_width=False
                )

        if all_results:
            st.markdown("#### Rapport complet (CSV)")
            all_preds = []
            for h, res in all_results.items():
                n = min(len(res['y_true']), len(res['y_pred']))
                sub = pd.DataFrame({
                    'horizon': h,
                    'index': np.arange(n),
                    'y_true': res['y_true'][:n],
                    'y_pred': res['y_pred'][:n],
                    'residual': res['y_true'][:n] - res['y_pred'][:n]
                })
                all_preds.append(sub)
            df_preds = pd.concat(all_preds, ignore_index=True)
            csv_preds = df_preds.to_csv(index=False, float_format="%.8f")
            st.download_button(
                "⬇️ Télécharger all_predictions.csv",
                csv_preds, "all_predictions.csv", "text/csv",
                use_container_width=True
            )

else:
    # ── État initial ──────────────────────────────────────
    st.markdown("""
    <div class="info-box">
    👈 <b>Configurez les paramètres</b> dans la barre latérale gauche,
    puis cliquez sur <b>"▶ Lancer le pipeline"</b> pour démarrer l'analyse.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🌀 Série de Hénon
        Système chaotique discret défini par :
        ```
        x(n+1) = y(n) + 1 - a·x(n)²
        y(n+1) = b·x(n)
        ```
        Paramètres : a=1.4, b=0.3
        """)
    with col2:
        st.markdown("""
        #### 🧠 Pipeline MLP
        - Génération série (500 pts)
        - Fenêtre glissante supervisée
        - Normalisation MinMax [-1, 1]
        - Entraînement par horizon
        - EarlyStopping + ReduceLR
        """)
    with col3:
        st.markdown("""
        #### 📊 Résultats produits
        - Attracteur de Hénon
        - Courbes d'apprentissage
        - Prédictions h=1,3,10,20
        - MSE/RMSE/MAE/R² (8 chiffres)
        - Interprétation scientifique
        """)
