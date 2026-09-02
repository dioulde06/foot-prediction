# foot-prediction

Modèle de prédiction 1X2 pour les 5 grands championnats européens, évalué sur
la **qualité de calibration probabiliste**, pas sur l'accuracy. Un modèle qui
annonce 45 % doit avoir raison 45 fois sur 100.

L'objectif final : une app qui publie ses prédictions avant chaque journée et
mesure honnêtement sa calibration dans le temps, face aux cotes du marché.

## État d'avancement

| Phase | Sujet | État |
|---|---|---|
| 0 | Fondations (log-loss, Brier, ECE, dévigorisation) | exercices en cours dans `JOURNAL.md` |
| 1 | Squelette, ingestion, mapping des noms | football-data ingéré ; FBref/Understat en attente |
| 1bis | Audit de qualité des données | fait — `notebooks/01_audit_qualite.ipynb` |
| 2 | Baselines et métriques | fait — voir le tableau ci-dessous |
| 2bis | Exploration du signal (train uniquement) | à faire |
| 3 | Features (Elo, rolling, non-fuite) | à faire |
| 4 | Modèle LightGBM et calibration isotonique | à faire |
| 5 | Validation walk-forward | à faire |
| 6 | App de publication et suivi | à faire |

Le détail de chaque phase, avec ce qu'il faut comprendre et qui fait quoi, est
dans [`PLAN.md`](PLAN.md).

## Données

Source unique pour l'instant : **football-data.co.uk**, cotes de clôture et
statistiques de match.

- **10 734 matchs**, du 2020-08-21 au 2026-05-24
- 5 championnats : Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- 6 saisons : 2020-21 à 2025-26
- 25 colonnes → `data/raw/football_data.parquet` (gitignoré, régénérable)

Trois jeux de cotes de **clôture** sont conservés, jamais d'ouverture : moyenne
de marché (`avg`, 100 % de couverture), Bet365 (`b365`, 100 %) et Pinnacle
(`ps`, 92 % — mais seulement ~50 % sur la saison de test, donc inutilisable
seul comme baseline `market`).

Sources en attente d'arbitrage : Understat pour le xG (recommandé), FBref
(déconseillé, voir `PLAN.md`).

## Commandes

```bash
make install   # uv sync
make fetch     # télécharge les données (FORCE=1 pour ignorer le cache)
make test      # pytest
make lint      # ruff + mypy strict
make train     # phase 4, pas encore implémenté
make eval      # phase 5, pas encore implémenté

uv run python scripts/run_baselines.py            # tableau des 3 baselines
uv run python scripts/run_baselines.py --book ps  # contrôle croisé Pinnacle

uv run jupyter lab notebooks/01_audit_qualite.ipynb
```

## Structure

```
src/
  data/       ingestion (fetch.py), normalisation des noms d'équipes
  features/   build_features(), Elo, rolling windows
  models/     entraînement, calibration
  eval/       metrics.py, baselines.py, splits.py, walk-forward
scripts/      run_baselines.py
tests/
notebooks/    analyses exécutées, suivies dans git
reports/
  figures/    figures exportées en PNG
data/
  raw/        parquet brut par source (gitignoré)
  processed/   dataset final matchs x features (gitignoré)
```

## Les trois baselines

Aucun résultat ne veut rien dire sans elles. Recalculées à chaque changement de
dataset.

| Baseline | Description |
|---|---|
| `uniform` | Fréquences historiques 1X2, calculées sur le train uniquement |
| `home` | Toujours victoire à domicile |
| `market` | Cotes de clôture dévigorisées (`avg` par défaut) — référence haute, probablement imbattable |

`uniform` est ajustée sur le **train uniquement**. Calculée sur tout le dataset,
la baseline connaîtrait la saison de test et deviendrait artificiellement forte.

`home` est une prédiction dure `(1, 0, 0)`, volontairement non adoucie : sous
log-loss, une certitude fausse est catastrophique, et c'est précisément ce
qu'elle doit montrer. Son log-loss est un avertissement, jamais une cible.

La dévigorisation est multiplicative, `p_i = (1/o_i) / Σ(1/o_j)`. Elle suppose
la marge répartie proportionnellement, ce qui est empiriquement faux — les
bookmakers en chargent davantage sur les outsiders. Plafond connu et documenté.

### Résultats mesurés sur 2025-26 (1 751 matchs, cotes `avg`)

| Baseline | log-loss | Brier | ECE | accuracy* |
|---|---|---|---|---|
| `uniform` | 1.0721 | 0.6485 | 0.0078 | 44.0 % |
| `home` | **19.3307** | 1.1194 | 0.3731 | 44.0 % |
| `market` | **0.9778** | 0.5823 | 0.0118 | 53.5 % |

\* information seulement, ne décide de rien.

Ne rien savoir vaut `ln(3) = 1.0986`. **Le terrain de jeu fait 0.094 de large** :
tout modèle utile atterrira entre 1.0721 et 0.9778. L'entropie moyenne des
probabilités du marché vaut 0.9951, ce qui estime le plancher sous lequel un
log-loss n'est plus honnête.

Deux lectures à retenir de ce tableau :

- `home` et `uniform` ont **exactement la même accuracy** (44.0 %) pour un
  log-loss de 19.33 contre 1.07. C'est la démonstration que l'accuracy ne
  mesure pas ce qui nous intéresse.
- `uniform` est **mieux calibrée** que le marché (ECE 0.0078 contre 0.0118) tout
  en étant inutile : elle dit toujours la même chose. Un bon ECE sans pouvoir
  discriminant ne vaut rien — ne jamais optimiser l'ECE seul.

## Garde-fou

Si l'accuracy dépasse **60 %** ou si le log-loss descend sous **0.85**, on
s'arrête et on cherche la fuite temporelle. Ce n'est jamais une bonne nouvelle.

## Documents

| Fichier | Rôle |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Règles non négociables et conventions de travail |
| [`PLAN.md`](PLAN.md) | Plan phase par phase, avec le protocole d'apprentissage |
| [`PROMPTS.md`](PROMPTS.md) | Prompts de démarrage et de maintenance |
| `JOURNAL.md` | Prédictions écrites avant les résultats, et confrontation |
