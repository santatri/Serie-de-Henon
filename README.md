# 🧠 RNA Multicouches — Série de Hénon
## Interface Streamlit · ISP Madagascar · Juin 2026
### Filières : ESIIA4 | IGGLIA4 | IMTICIA4 | ISAIA4

---

## 📋 Description

Application web interactive pour la prédiction et modélisation de la
**série chaotique de Hénon** par un réseau de neurones MLP (Multi-Layer Perceptron).

---

## 📁 Structure du projet

```
henon_rna_streamlit/
│
├── app.py               ← Interface Streamlit principale (point d'entrée)
├── generate_henon.py    ← Génération de la série de Hénon
├── prepare_dataset.py   ← Préparation des jeux de données supervisés
├── train_model.py       ← Recherche d'architecture + entraînement MLP
├── predict.py           ← Prédictions multi-horizons + métriques
├── plots.py             ← Génération des figures (matplotlib)
├── main.py              ← Pipeline complet de génération, entraînement, et évaluation
├── report_generator.py  ← Génération du rapport scientifique Markdown
│
├── requirements.txt     ← Dépendances Python
└── README.md            ← Ce fichier
```

---

## ⚙️ Installation

```bash
# 1. Extraire le dossier zip
unzip henon_rna_streamlit.zip
cd henon_rna_streamlit/

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Lancement

Sur Windows, lancez le projet Streamlit avec :

```powershell
py -m streamlit run app.py
```

Pour exécuter le pipeline complet en ligne de commande (sans Streamlit) :

```powershell
py main.py
```


```bash
streamlit run app.py
```

Ouvrez votre navigateur sur **http://localhost:8501**

---

## 🖥️ Interface — 6 onglets

| Onglet | Contenu |
|--------|---------|
| 🌀 Attracteur | Attracteur de Hénon + séries temporelles x(n), y(n) |
| 📈 Apprentissage | Courbes loss/val_loss par horizon |
| 🎯 Prédictions | Réel vs prédit, scatter plot, résidus |
| 📊 Métriques | Tableau MSE/RMSE/MAE/R²/MAPE + visualisations |
| 🔬 Interprétation | Analyse scientifique + discussion stabilité |
| 💾 Export | Téléchargement CSV + figures PNG |

---

## ⚙️ Paramètres configurables (barre latérale)

- **Hénon** : nombre de points, a, b
- **Fenêtre** : taille temporelle, % test
- **Architecture MLP** : couches, neurones, activation, learning rate, époques
- **Horizons** : h = 1, 3, 10, 20 (sélection libre)

---

## 📐 Série de Hénon

```
x(n+1) = y(n) + 1 - 1.4 × x(n)²
y(n+1) = 0.3 × x(n)
x₀ = 0, y₀ = 0
```

---

## 📊 Métriques (8 chiffres significatifs)

| Métrique | Description |
|----------|-------------|
| MSE      | Erreur quadratique moyenne |
| RMSE     | Racine de la MSE |
| MAE      | Erreur absolue moyenne |
| R²       | Coefficient de détermination |
| MAPE     | Erreur relative moyenne (%) |

---

*Bon travail — Institut Supérieur Polytechnique de Madagascar, Juin 2026*
