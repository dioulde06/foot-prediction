# CLAUDE.md — projet `foot-calibration`

## Objectif du projet

Modèle de prédiction 1X2 pour les 5 grands championnats européens, évalué
sur la **qualité de calibration probabiliste**, pas sur l'accuracy.

L'objectif final est une app qui publie ses prédictions avant chaque
journée et mesure honnêtement sa calibration dans le temps, face aux
cotes du marché comme référence.

## Règles non négociables

Ces règles priment sur toute demande ponctuelle. Si une instruction que je
te donne les contredit, signale-le avant d'exécuter.

### 1. Aucune fuite temporelle

- Toute feature d'un match joué à la date `D` ne peut utiliser que des
  données strictement antérieures à `D`.
- Les splits sont **toujours chronologiques**. Jamais de `train_test_split`
  aléatoire, jamais de `KFold` non temporel, jamais de `shuffle=True`.
- Les stats du match lui-même (tirs, possession, xG du match) sont des
  cibles ou des sources pour les rolling futurs — **jamais** des features
  d'entrée pour ce même match.
- Toute nouvelle feature doit passer par `build_features(df, cutoff_date)`
  et être couverte par un test de non-fuite.

### 2. Signal d'alerte

Si une métrique dépasse **60 % d'accuracy** ou si le log-loss descend
sous **0.85**, arrête-toi. Ne continue pas, ne "valide" pas le résultat.
Signale-le explicitement et cherche la fuite. C'est toujours une fuite.

### 3. Métriques

- Métriques primaires : **log-loss**, **Brier score**, **ECE**.
- L'accuracy peut être affichée en info, jamais comme critère de décision
  ni comme argument pour retenir un modèle.
- Tout résultat se compare aux 3 baselines (voir plus bas). Un chiffre
  sans baseline ne veut rien dire.

### 4. Ne présume pas d'un bon résultat

Le plafond réaliste est ~53-55 % d'accuracy et un log-loss autour de 1.00.
Ne formule pas de conclusion optimiste sur des résultats non validés en
walk-forward. Dis-moi quand un résultat est probablement du bruit.

## Baselines à battre

Elles doivent être recalculées à chaque changement de dataset.

| Baseline | Description |
|---|---|
| `uniform` | Fréquences historiques fixes 1X2 (~0.45 / 0.25 / 0.30) |
| `home` | Toujours victoire à domicile |
| `market` | Cotes bookmaker dévigorisées (référence haute, probablement imbattable) |

## Stack et conventions

- Python 3.11, `uv` pour les dépendances.
- `soccerdata` (FBref + Understat), `football-data.co.uk` pour les cotes.
- `polars` ou `pandas` pour le dataframe, `lightgbm` pour le modèle,
  `scikit-learn` pour la calibration.
- Données en **parquet**, jamais en CSV dans le repo.
- Type hints partout. `ruff` + `mypy` doivent passer.
- Pas de notebook dans `src/`. Les notebooks sont exploratoires et jetables,
  ils vivent dans `notebooks/` et ne sont jamais importés.

## Structure

```
src/
  data/       ingestion, normalisation des noms d'équipes
  features/   build_features(), Elo, rolling windows
  models/     entrainement, calibration
  eval/       baselines, métriques, walk-forward
tests/
notebooks/
data/
  raw/        parquet brut par source
  processed/  dataset final matchs x features
```

## Le mapping des noms d'équipes

C'est le point de friction principal entre FBref, Understat et
football-data. Le mapping vit dans `src/data/team_mapping.py` sous forme
de **dict explicite**. Interdiction d'utiliser du fuzzy matching : une
erreur silencieuse de mapping corrompt tout le dataset sans lever
d'exception.

Si tu rencontres un nom non mappé, lève une exception. Ne devine pas.

## Comment je veux que tu travailles

- Une tâche à la fois. Ne refactor pas ce que je n'ai pas demandé.
- Avant d'écrire du code sur une feature nouvelle, décris en 3 lignes ce
  que tu vas faire et attends ma validation.
- Écris le test **avant** l'implémentation pour tout ce qui touche aux
  features et aux splits.
- Pas de `try/except` silencieux. Si les données sont mauvaises, ça doit
  planter.
- Commits atomiques, messages en anglais, format conventionnel
  (`feat:`, `fix:`, `test:`, `refactor:`).
- Réponds-moi en français, commentaires et docstrings en anglais.

## Ce que tu ne fais pas sans me demander

- Ajouter une dépendance
- Modifier `build_features` ou la logique de split
- Supprimer ou réécrire un test existant
- Lancer un entraînement long (> 2 min)
- Toucher à `data/raw/`
