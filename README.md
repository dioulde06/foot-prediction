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
| 2 | Baselines et métriques | à faire |
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

uv run jupyter lab notebooks/01_audit_qualite.ipynb
```

## Structure

```
src/
  data/       ingestion (fetch.py), normalisation des noms d'équipes
  features/   build_features(), Elo, rolling windows
  models/     entraînement, calibration
  eval/       baselines, métriques, walk-forward
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
| `market` | Cotes de clôture dévigorisées — référence haute, probablement imbattable |

Repère utile : ne rien savoir en 1X2 vaut `ln(3) = 1.0986` de log-loss. Un bon
modèle vise 0.98-1.02, le marché 0.95-0.98. Toute la partie se joue dans une
bande de 0.15.

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
