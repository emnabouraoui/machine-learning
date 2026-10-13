# Lab Data Preparation — Adult Census Income

**Groupe 4CCE13** · Dataset : Adult (UCI, recensement US 1994)

Démarche CRISP-DM : Business Understanding → Data Understanding → Exploration → Preparation → Segmentation → Classification → Évaluation.

## Problématiques

- **Classification** : prédire si une personne gagne plus de 50 K$/an (`income`).
- **Segmentation** : identifier des profils socio-professionnels homogènes (clustering non supervisé).

## Contenu

| Fichier | Description |
|---|---|
| `lab_adult_data_preparation 2.ipynb` | Notebook complet, exécuté (les graphiques s'affichent directement sur GitHub) |
| `adult_clean.csv` | Dataset nettoyé après préparation (48 790 lignes) |
| `plots/` | Toutes les visualisations exportées en PNG + `metrics.csv` |

## Préparation des données

- Doublons supprimés (52), `?` → `NaN` puis modalité `Unknown` pour `workclass` / `occupation` / `native-country` (MNAR).
- `capital-gain` / `capital-loss` : transformation `log1p` + indicateurs binaires (variables très asymétriques).
- `age`, `hours-per-week` : winsorisation (percentiles 1/99).
- Feature engineering : `native_us`, `married`, `net_capital`, `age_group`.
- Suppressions : `education` (redondante avec `education-num`), `fnlwgt` (poids d'échantillonnage), `native-country` d'origine.
- Encodage One-Hot + StandardScaler **dans un Pipeline après le split** (pas de leakage).

## Segmentation (K-Means)

k = 3 (silhouette ≈ 0,41, max sur k = 2–10), visualisé en ACP 2D.

| Cluster | Effectif | Taux >50K | Profil |
|---|---|---|---|
| 0 | ~4 000 | 62 % | Détenteurs de capital, mariés, cadres/spécialistes |
| 1 | ~42 500 | 19 % | Masse sans capital, plus jeune, jamais mariés |
| 2 | ~2 300 | 50 % | Groupe avec pertes en capital, mariés, cadres |

## Classification (test 20 %, split stratifié)

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| HistGradBoost | 0,833 | 0,604 | 0,878 | 0,716 | **0,933** |
| RandomForest | 0,836 | 0,615 | 0,838 | 0,710 | 0,922 |
| LogisticReg (baseline) | 0,814 | 0,575 | 0,856 | 0,688 | 0,915 |

Variables les plus importantes : `capital-gain_log`, `education-num`, `age`, `hours-per-week`, statut marital, `sex`, professions cadres. Note : `sex` / `relationship` reflètent un biais historique du recensement 1994 — à auditer avant tout usage décisionnel.

## Exécuter le notebook

**VSCode** : ouvrir le dossier, extensions Python + Jupyter, interpreter `/opt/homebrew/bin/python3`, bouton *Run All*.
Dépendances : `scikit-learn pandas matplotlib seaborn ipykernel`.

**Google Colab** : *File > Upload notebook*, puis *Runtime > Run all* (dépendances préinstallées, données UCI téléchargées automatiquement).
