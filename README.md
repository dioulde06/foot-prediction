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
| 1 | Squelette, ingestion, mapping des noms | fait — 2 sources jointes à 100 %, 35 noms mappés |
| 1bis | Audit de qualité des données | fait — `notebooks/01_audit_qualite.ipynb` |
| 2 | Baselines et métriques | fait — voir le tableau ci-dessous |
| 2bis | Exploration du signal (train uniquement) | fait — `notebooks/02_analyse.ipynb` |
| 3 | Features (Elo, rolling, non-fuite) | fait — 6 features, 20 tests de non-fuite et d'invariance |
| 4 | Modèle LightGBM et calibration isotonique | à faire |
| 5 | Validation walk-forward | à faire |
| 6 | App de publication et suivi | à faire |

Le détail de chaque phase, avec ce qu'il faut comprendre et qui fait quoi, est
dans [`PLAN.md`](PLAN.md).

## Données

Deux sources, jointes dans `data/processed/matches.parquet` (33 colonnes).

- **10 734 matchs**, du 2020-08-21 au 2026-05-24
- 5 championnats : Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- 6 saisons : 2020-21 à 2025-26
- **football-data.co.uk** → résultats, statistiques de match, cotes de clôture
- **Understat** → xG, xG hors penalty, PPDA, passes profondes

Taux de jointure : **100.00 %**. La clé est `(saison, domicile, extérieur)` et
non la date : dans un championnat aller-retour cette triple est exactement
unique dans les deux sources, alors que les dates divergent d'un ou deux jours
sur 18 matchs à cause des coups d'envoi tardifs et des reports. Joindre sur la
date exigerait une fenêtre de tolérance, ce qui est l'équivalent temporel du
fuzzy matching et donc interdit ici.

Les noms d'équipes sont réconciliés par un dict explicite de **35 entrées** dans
`src/data/team_mapping.py`, les noms football-data servant de forme canonique.
Un nom inconnu lève `UnmappedTeamError`. `make audit-teams` liste ce qui manque.

Trois jeux de cotes de **clôture** sont conservés, jamais d'ouverture : moyenne
de marché (`avg`, 100 % de couverture), Bet365 (`b365`, 100 %) et Pinnacle
(`ps`, 92 % — mais seulement ~50 % sur la saison de test, donc inutilisable
seul comme baseline `market`).

FBref n'est pas utilisé : `soccerdata` 1.9.1 n'en expose plus que
schedule / keeper / shooting / misc, les tables de passes, possession et
défense qui justifiaient la source ont disparu, et son accès exige un pilote de
navigateur. Tout ce qu'il apporterait encore est déjà couvert par Understat.

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

make merge         # joint les sources
make audit-teams   # noms d'équipes non résolus

uv run jupyter lab notebooks/
```

## Structure

```
src/
  data/       ingestion (fetch.py), normalisation des noms d'équipes
  features/   build.py (build_features), elo.py
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

### Les 8 features de départ

Construites par `build_features(matches, cutoff_date)`, qui garantit deux
invariants vérifiés par `tests/test_no_leakage.py` : aucune ligne ne lit son
propre match, et rien après `cutoff_date` n'est lu du tout.

| Feature | Contenu |
|---|---|
| `elo_diff` | Elo domicile + avantage terrain − Elo extérieur |
| `form_points_diff_5` | points par match sur 5 matchs, différentiel |
| `goals_scored_diff_5` | buts marqués sur 5 matchs, différentiel |
| `goals_conceded_diff_5` | buts encaissés sur 5 matchs, différentiel |
| `shots_target_diff_5` | tirs cadrés sur 5 matchs, différentiel |
| `np_xg_created_diff_5` | xG hors penalty créé sur 5 matchs, différentiel |
| `np_xg_conceded_diff_5` | xG hors penalty concédé sur 5 matchs, différentiel |
| `rest_days_diff` | jours de repos, différentiel, **plafonné à 14 jours** |

Le plafond sur `rest_days` n'est pas cosmétique : le maximum brut était de 811
jours. Au-delà de deux semaines la feature ne mesure plus la fatigue, elle
encode « cette équipe était reléguée ou absente ».

Validation indépendante : `elo_diff` corrèle à **0.857** avec la probabilité de
victoire à domicile du marché. Aucun résultat n'entre dans ce calcul, seulement
deux estimations d'avant-match — c'est donc une vérification que le pipeline
produit du signal réel, pas une mesure de performance.

Attention à la redondance : les sept features de force corrèlent entre elles
entre 0.25 et 0.78. Elles mesurent largement la même chose. Seule
`rest_days_diff` est orthogonale (±0.01 partout). **Les 8 features valent en
pratique 2 dimensions indépendantes : la force et le repos.** Ajouter une
nouvelle variante de forme ne servira à rien.

Le xG n'apporte pas une dimension nouvelle — il corrèle à 0.78 avec les tirs
cadrés — mais c'est une **meilleure mesure de la même dimension** :
`np_xg_created_diff_5` prédit la marge de buts du match suivant à 0.332 contre
0.283 pour `goals_scored_diff_5`, soit **17 % de pouvoir prédictif en plus**.

### Décisions de conception

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

`market` utilise la dévigorisation par **puissance** (`p_i = π_i^k`, `k` résolu
pour que la somme fasse 1). La méthode multiplicative surestimait les outsiders
de 2.25 points et sous-estimait les favoris de 5.15 points, la distorsion étant
proportionnelle à la marge du bookmaker. La puissance ramène ces écarts à +0.34
et −0.60 et l'ECE de 0.0118 à 0.0086. Le log-loss, lui, ne bouge que de 0.0009 :
la mauvaise calibration était concentrée sur les tranches extrêmes, peu peuplées.

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

## Notebooks

Exécutés et suivis dans git, sorties et figures comprises. Les figures sont
aussi exportées dans `reports/figures/`.

| Notebook | Contenu |
|---|---|
| `01_audit_qualite.ipynb` | Qualité des données, tout le dataset : nulls, couverture des cotes, intégrité du calendrier, avantage domicile dans le temps, marges des bookmakers, anomalies |
| `02_analyse.ipynb` | Le terrain de jeu, le piège de l'accuracy, la fiabilité du marché, les corrélations entre features, le pouvoir prédictif de chacune, les trajectoires Elo, buts contre xG |
