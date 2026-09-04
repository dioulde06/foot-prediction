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
| 4 | Modèle LightGBM et calibration | fait — 1.0044 sur le test |
| 5 | Validation walk-forward | fait — stable sur 4 saisons ; test d'information : le modèle n'apporte rien au marché |
| 6 | App de publication et suivi | fait — `make publish` / `make track` |
| 7 | Site | fait — application à quatre onglets : matchs du mois, propositions, bilan des propositions, méthode |
| 8 | Publication automatique | fait — GitHub Actions deux fois par jour, GitHub Pages |

Le détail de chaque phase, avec ce qu'il faut comprendre et qui fait quoi, est
dans [`PLAN.md`](PLAN.md).

## Données

Deux sources, jointes dans `data/processed/matches.parquet` (33 colonnes).

- **10 734 matchs**, du 2020-08-21 au 2026-05-24
- 5 championnats : Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- 6 saisons : 2020-21 à 2025-26
- **football-data.co.uk** → résultats, statistiques de match, cotes de clôture
- **Understat** → xG, xG hors penalty, PPDA, passes profondes
- **ESPN** (scoreboard public) → scores en direct, appelé par le navigateur,
  jamais écrit dans le dataset

Taux de jointure : **100.00 %**. La clé est `(saison, domicile, extérieur)` et
non la date : dans un championnat aller-retour cette triple est exactement
unique dans les deux sources, alors que les dates divergent d'un ou deux jours
sur 18 matchs à cause des coups d'envoi tardifs et des reports. Joindre sur la
date exigerait une fenêtre de tolérance, ce qui est l'équivalent temporel du
fuzzy matching et donc interdit ici.

Les noms d'équipes sont réconciliés par des dicts explicites dans
`src/data/team_mapping.py`, les noms football-data servant de forme canonique :
**37 entrées** pour Understat, **43** pour ESPN. Un nom inconnu lève
`UnmappedTeamError`. `make audit-teams` liste ce qui manque, dataset et direct.

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
make fetch-current  # rafraîchit la saison en cours seulement, puis rejoint
make test      # pytest
make lint      # ruff + mypy strict
make train     # entraîne, calibre, écrit reports/model_report.md
make eval      # walk-forward + saturation de l'historique
make information  # le modèle sait-il quelque chose que le marché ignore ?
make publish   # prédit les matchs à venir, append-only ; capture cotes et buteurs
make site      # génère site/index.html depuis les parquets suivis dans git
make proposals-backtest  # rejoue le combinateur sur la dernière saison complète
make track     # calibration des prédictions publiées, une fois jouées

uv run python scripts/run_baselines.py            # tableau des 3 baselines
uv run python scripts/run_baselines.py --book ps  # contrôle croisé Pinnacle

make merge         # joint les sources
make audit-teams   # noms d'équipes non résolus (dataset et direct ESPN)

uv run jupyter lab notebooks/
```

## Structure

```
src/
  data/       fetch.py, merge.py, team_mapping.py, players.py (joueurs), schedule.py (calendrier)
  features/   build.py (build_features), elo.py
  models/     train.py, calibrate.py
  eval/       metrics.py, baselines.py, splits.py, report.py, walk_forward.py
  app/        publish.py — publication, capture des cotes, buteurs et propositions figés
              combos.py — le combinateur en Python, même algorithme que la page
              scorers.py — buts attendus par équipe, buteurs probables, marchés buts
              site.py — génération du site statique ; templates/index.html
configs/      lightgbm.yaml
scripts/      run_baselines, train_and_report, run_walk_forward, publish,
              track_calibration, audit_teams
tests/
notebooks/    analyses exécutées, suivies dans git
predictions/  predictions.parquet, historique append-only des prédictions publiées
              odds.parquet, cotes capturées à la publication, append-only aussi
              scorers.parquet, buteurs probables et buts attendus figés à la publication
              proposals.parquet, les trois propositions par objectif, figées chaque semaine
site/         index.html généré par `make site`, servi tel quel
reports/
  figures/    figures exportées en PNG
data/
  raw/        parquet brut par source ; understat_players.parquet et understat_schedule.parquet
              sont refaits à chaque publication, non suivis
  processed/  dataset joint
models/       modèle entraîné et métadonnées
.github/workflows/publish.yml   le robot de publication
```

`src/app/` et `configs/` ne figurent pas dans la structure décrite par
`CLAUDE.md` : la publication n'est ni de l'ingestion, ni des features, ni de
l'évaluation, et les hyperparamètres devaient sortir du code.

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

### Le modèle, mesuré sur 2025-26 (1 751 matchs)

| modèle | log-loss | Brier | ECE | accuracy* |
|---|---|---|---|---|
| modèle brut | 1.0049 | 0.5992 | 0.0145 | 51.5 % |
| modèle + isotonique | 1.0193 | 0.6006 | 0.0136 | 50.4 % |
| **modèle + température** | **1.0044** | **0.5989** | **0.0106** | 51.5 % |
| `market` | 0.9769 | 0.5818 | 0.0086 | 53.5 % |

Le modèle capture **71 %** de l'écart disponible entre `uniform` et `market`, et
reste 0.028 derrière le marché.

**La calibration isotonique dégrade le modèle** (1.0049 → 1.0193). Non
paramétrique, elle est assez souple pour apprendre le bruit d'une seule saison
de validation. La mise à l'échelle par température — un seul paramètre,
`p_i^(1/T)` avec `T = 1.036` — améliore les deux métriques à la fois. Un
paramètre ne peut pas surapprendre une saison.

### Stabilité en walk-forward

| saison de test | log-loss modèle | log-loss market | écart |
|---|---|---|---|
| 2022-23 | 1.0106 | 0.9753 | +0.0353 |
| 2023-24 | 0.9891 | 0.9521 | +0.0370 |
| 2024-25 | 0.9925 | 0.9600 | +0.0325 |
| 2025-26 | 1.0044 | 0.9769 | +0.0275 |

Amplitude du log-loss : **0.0215**. L'écart au marché reste entre +0.0275 et
+0.0370 sur quatre saisons, toujours du même signe et du même ordre. Cette
constance est l'argument le plus fort que le résultat est réel : une
fluctuation heureuse ne se reproduit pas quatre fois de suite.

Aucune moyenne n'est publiée volontairement — c'est la dispersion qui informe.

Le balayage de l'historique donne 1.0073 / 1.0050 / 1.0038 / 1.0044 pour 1, 2, 3
et 4 saisons d'entraînement. **Le gain sature à 3 saisons**, soit environ 5 400
matchs. Plus de données n'est pas le levier.

### Le test d'information : le modèle sait-il quelque chose que le marché ignore ?

Avant de chercher à réduire l'écart de 0.028 par des features ou des données,
une question décide de tout : une fois le marché connu, les résultats
donnent-ils encore du poids au modèle ? `src/eval/information.py` ajuste le
mélange géométrique `p_k ∝ modèle_k^a · marché_k^b` par maximum de
vraisemblance sur les prédictions hors échantillon des 4 saisons walk-forward
(7 081 matchs). Un modèle calibré mais sans information propre obtient
`a ≈ 0`, `b ≈ 1`.

| | valeur |
|---|---|
| `a` (poids du modèle) | **−0.126**, intervalle bootstrap 95 % [−0.240, −0.024] |
| `b` (poids du marché) | 1.138 |
| gain du mélange sur le marché, hors échantillon, par saison | −0.0008 / +0.0009 / +0.0008 / +0.0003 |

**Le poids du modèle est négatif, et l'intervalle exclut zéro.** Quand le
modèle s'écarte du marché, il a plus souvent tort que raison. Il ne contient
rien que le marché ne sache déjà : l'Elo, la forme, le xG et le repos sont
tous déjà dans les cotes, en mieux. Le mélange ne gagne rien de mesurable
(les quatre gains sont du bruit, de signe variable).

Conséquence : **ajouter des variantes des mêmes features ou plus d'historique ne
peut pas fermer l'écart.** Seule une source d'information que le marché
intègre et que le dataset n'a pas — compositions, absents, charge européenne —
peut y prétendre. Le `b > 1` dit au passage que le marché moyen reste un peu
sous-confiant après dévigorisation par puissance, mais le mélange hors
échantillon montre que ce n'est pas exploitable non plus.

### Les baselines mesurées sur 2025-26 (1 751 matchs, cotes `avg`)

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
`src/eval/report.py` vérifie les deux bornes à chaque entraînement et fait
échouer le script si l'une est franchie.

## Publier et vérifier

`make publish` prédit tous les matchs des **35 prochains jours**, lus dans le
calendrier de la saison Understat (`src/data/schedule.py`, coups d'envoi en
UTC) complété par le flux de fixtures football-data, et ajoute les lignes à
`predictions/predictions.parquet`. Deux propriétés rendent l'exercice honnête,
et toutes deux sont mécaniques :

- **Append-only.** Une ligne publiée n'est jamais modifiée ni supprimée. Quand
  de nouveaux résultats font bouger la prédiction d'un match (au-delà de
  0.0001 sur une probabilité), une nouvelle ligne horodatée s'ajoute ; la
  dernière avant le coup d'envoi est celle qui compte, les précédentes montrent
  le chemin. La réconciliation joint les résultats à la volée sur cette
  dernière ligne et n'écrit rien en retour. Des tests le vérifient.
- **Horodaté par ce qu'on ne contrôle pas.** Chaque ligne porte un
  `published_at` et le sha256 de sa charge, mais surtout le fichier est suivi
  dans git. Un sceptique n'a pas à croire l'horodatage du fichier :
  l'historique git dit quand la prédiction existait.

Les features d'un match à venir passent par `build_features` exactement comme
une ligne d'entraînement — pas de second chemin de code, donc pas d'écart
entraînement/production. Un test compare les features d'un match sans score à
celles du même match avec un 9-0 : elles sont identiques.

`make publish` capture aussi, dans `predictions/odds.parquet`, l'heure du coup
d'envoi et les cotes du flux au moment de la publication : la moyenne du
marché et six bookmakers (Bet365, Betfair, BetVictor, Bwin, Paddy Power, Sky
Bet). Fichier séparé, lui aussi en ajout seul : le schéma du registre des
prédictions est publié et ne bouge pas. La première capture est la référence,
le prix qui existait quand la prédiction est sortie ; chaque publication
suivante ajoute une photo quand la moyenne du marché a bougé, et la page montre
le mouvement (flèche sur la cote, ligne « depuis le … »). Un match que le
marché n'a pas encore coté n'est pas enregistré vide, il est capturé le jour où
il a un prix. Ce sont des cotes courantes, pas de clôture ; la clôture, elle,
arrive avec le résultat et sert à la « clôture battue » du carnet.

Les fenêtres glissantes de `build_features` lisent les cinq derniers matchs
**joués** d'une équipe et donnent à chaque ligne, jouée ou non, la fenêtre
telle qu'elle était strictement avant sa date. Un match dans trois semaines a
donc les mêmes features que s'il était demain, calculées avec ce qu'on sait
aujourd'hui ; les features d'entraînement sont identiques au bit près à
l'ancienne version, deux tests d'invariance couvrent le cas des matchs
intercalés.

`make track` mesure la calibration des prédictions publiées une fois leurs
matchs joués, face au marché.

## Le site

`make site` écrit `site/index.html`, une application d'une page, sans serveur,
fonction pure des fichiers suivis dans git : registre des prédictions (dernière
ligne par match), cotes capturées, buteurs figés, propositions figées,
calendrier, matchs joués, prédictions hors échantillon du walk-forward et
rejeu du combinateur. Tout ce qu'elle calcule tourne dans le navigateur, et la
page change quand on publie, jamais entre-temps.

Une barre fixe porte quatre onglets, le bookmaker dont les cotes s'affichent,
la mise, et le ticket. Le ticket est une colonne fixe sur grand écran, un
panneau coulissant sur mobile ; il vit dans l'adresse de la page, copier le
lien c'est partager le ticket.

- **Matchs** : deux listes séparées comme chez un bookmaker, **À venir**
  (à jouer et en cours) et **Terminés** (les résultats des 10 derniers jours,
  du plus récent au plus ancien). Les matchs sont regroupés par jour ou par
  championnat, repliés avec un résumé (favoris nets, matchs serrés), en vue
  compacte ou détaillée, filtrés par jour, championnat, équipe, matchs cotés,
  mes sélections, et par **probabilité du favori** (≥ 50, 60, 70, 80 %, sur
  notre probabilité, le nul n'étant la victoire de personne). Avec un seuil
  actif, l'en-tête de chaque groupe liste les équipes retenues avec leur
  probabilité. Heures dans le fuseau du visiteur, compte à rebours sous
  24 h, verrouillage au coup d'envoi, score en direct puis final. Un match pas
  encore coté se joue à notre prix juste (1 ÷ probabilité, sans marge), marqué
  ≈. La
  fiche d'un match donne forme, buteurs probables et marchés buts.
- **Propositions** : trois combinés sur les matchs filtrés, de 2 à 6
  sélections, quatre objectifs (cote cible, marge minimale, consensus
  modèle-marché, une sélection par journée), chacun avec sa raison en une
  ligne, sa probabilité selon le marché et selon nous, la marge composée, les
  gains si ça passe et le retour moyen sur la mise saisie. Des combinaisons,
  pas des conseils : toutes perdent en moyenne.
- **Bilan** : ce que les propositions ont donné, à la mise que le visiteur
  tape. Deux périodes : le réel depuis le lancement, et le rejeu de la dernière
  saison complète (`make proposals-backtest`, prédictions hors échantillon,
  cotes de clôture, vrais résultats). Pour chaque objectif : paris, gagnés face
  au taux annoncé, misé, récupéré, résultat face à l'attendu, la courbe
  cumulée réalisé contre attendu, et chaque semaine avec ses trois paris et
  leurs sélections cochées. Une semaine est figée le lundi mais **datée par ses
  matchs** : un lot figé le 31 août et joué du 4 au 7 septembre s'affiche en
  septembre. Puis le carnet du visiteur : ce qu'il a réellement
  misé, soldé avec les résultats, gardé dans son navigateur.
- **Méthode** : les chiffres du modèle, le diagramme de fiabilité, un lexique
  en huit entrées, la preuve par tranche et l'écart au marché saison par
  saison.

Le pari se place chez le bookmaker, jamais ici : un bouton ouvre son site, un
autre copie le ticket, on reporte la cote obtenue et « J'ai misé » alimente le
carnet. Le ticket se vide alors et laisse un reçu à sa place — mise, cote,
retour moyen, lien vers le carnet — qui disparaît à la sélection suivante. Rien
n'est reproposé : poser un pari n'est pas une raison d'en recevoir un autre.
Les issues déjà misées portent un ✓ vert dans la liste des matchs. Aucun bookmaker régulé n'offre d'API de placement, et automatiser un
compte viole leurs conditions.

### Les scores en direct

Le résultat officiel vient de football-data.co.uk, qui ne rafraîchit son
fichier de saison courante que quelques fois par semaine : un pari de la veille
restait « en cours » pendant des jours. La page appelle donc elle-même le
scoreboard public d'ESPN — pas de clé, `access-control-allow-origin: *`, aucun
serveur de notre côté — sur les 5 championnats et une fenêtre de 11 jours.

- Au chargement, à chaque retour sur l'onglet, puis toutes les 60 s tant qu'un
  match est en cours. Sinon un seul réveil au prochain coup d'envoi, sinon
  rien : pas de polling pour le plaisir.
- Un match en cours affiche son score et sa minute, marqués « direct », mais ne
  solde **jamais** un pari. Seul un match terminé écrit un résultat, marqué
  provisoire jusqu'à ce que le parquet le confirme.
- Un but redessine les matchs, les propositions, le ticket et le carnet, sans
  recharger la page ni perdre les sélections en cours.
- Le résultat officiel prime toujours sur le direct. Un désaccord, un nom
  d'équipe inconnu ou un flux injoignable est compté et affiché sur la page,
  jamais avalé en silence.
- Rien de tout cela n'entre dans `data/processed/matches.parquet`. Le classement
  de calibration, les tranches et le Bilan restent calculés en Python sur les
  résultats officiels : un score provisoire ne déplace aucune métrique.

### Ce qu'un bookmaker ne montre pas

Cinq calculs, tous dans le navigateur, à partir de ce qui est déjà sur la page :

- **Combiné ou simples.** Pour le ticket en cours, la même mise jouée en paris
  simples : probabilité de finir gagnant, gain maximum, retour moyen, marge
  payée. Le combiné multiplie la marge, c'est pour ça qu'on le vend ; les
  simples la paient une fois. Distribution exacte, 2ⁿ issues indépendantes.
- **Une saison à ce rythme.** Le même ticket joué 38 semaines, simulé 4 000
  fois : où finit le solde une saison sur vingt en dessous, en médiane, une sur
  vingt au-dessus, et la probabilité de finir gagnant. La variance avant de la
  subir.
- **Clôture battue.** Dans le carnet, la cote prise face à la cote de clôture
  du marché sur les sélections jouées : le test de compétence des parieurs
  professionnels, que les bookmakers ne montrent pas parce qu'un joueur qui bat
  la clôture se fait limiter. Le site fournit la clôture de chaque match joué.
- **Marge payée.** En euros, ce que les paris du carnet ont donné au bookmaker
  en marge, qu'ils aient gagné ou perdu.
- **Budget du mois.** Ce qu'on accepte de dépenser ; la page montre où on en
  est, alerte à 50 et 80 %, et dit la perte attendue sur les mises du mois. La
  mise optimale de Kelly est zéro quand l'espérance est négative ; le budget
  borne le plaisir.

### Les propositions figées

Le combinateur existe en deux exemplaires qui doivent rester identiques : le
JavaScript de la page, et `src/app/combos.py`. Le second sert à deux choses.
Chaque semaine, à la première publication qui voit au moins six matchs cotés
dans les sept jours, `make publish` fige dans `predictions/proposals.parquet`
les trois propositions de chaque objectif, telles que la page les montrait,
en ajout seul. Et `make proposals-backtest` rejoue le même algorithme semaine
par semaine sur la dernière saison complète dans
`reports/proposals_backtest.parquet`. Le site solde les deux registres avec
les matchs joués : une proposition est gagnée si toutes ses sélections
passent, perdue dès qu'une échoue, en cours sinon. Sur 2025-26, l'objectif
« cote cible » réussit 10,3 % de ses paris pour 10,2 % annoncés : les
probabilités disent vrai, et le résultat suit la marge du bookmaker.

## La publication automatique

Le site est en ligne : <https://dioulde06.github.io/foot-prediction/>.

Personne ne tape `make publish`. Le workflow `.github/workflows/publish.yml`
tourne deux fois par jour (7h et 16h UTC, ou à la main depuis l'onglet
Actions) : il rafraîchit la saison en cours, publie, régénère le site, commite
ce qui a changé dans `data/`, `predictions/` et `site/` sous l'identité du
robot, puis déploie `site/` sur GitHub Pages. Le dépôt est public : Pages
l'exige avec le plan gratuit, et l'antériorité prouvée par git n'a de valeur
que si n'importe qui peut la vérifier. Chaque passage est idempotent :
un match déjà publié n'est jamais republié, une cote ou un buteur déjà capturé
n'est jamais réécrit. Le commit du robot horodate l'antériorité mieux qu'un
commit fait à la main.

Pour que ça tourne, `data/`, `models/` et `predictions/` sont versionnés : le
robot ne reconstruit rien, il ne fait que l'incrément du jour. Côté réglages
du dépôt, GitHub Pages a « GitHub Actions » comme source.

### La saison glissante

La dernière saison ingérée se déduit du calendrier (coupure en juillet), plus
d'une constante. `make fetch-current` retélécharge uniquement la saison en
cours, cinq fichiers football-data et cinq pages Understat, remplace ses lignes
dans les parquets bruts et rejoint. Les saisons passées ne sont jamais
réécrites. Un match de la saison en cours dont Understat n'a pas encore publié
le xG est laissé de côté jusqu'au passage suivant, plutôt que d'entrer avec des
trous.

Le split d'entraînement est fixé par saisons nommées, donc le dataset qui
grossit ne change ni le modèle, ni le rapport, ni les baselines. Le
walk-forward, lui, ignore la saison incomplète : une demi-saison ne dit rien
sur la stabilité.

### Les buteurs probables

Le bouton « + » d'un match ouvre sa fiche : forme et chiffres des deux
équipes sur cinq matchs, les cinq buteurs les plus probables de chaque côté,
et les marchés buts. Tout vient de `predictions/scorers.parquet`, écrit par
`make publish` au même moment que la prédiction, en ajout seul lui aussi, pour
que l'estimation soit figée avant le coup d'envoi et vérifiable après.

Le calcul, dans `src/app/scorers.py`, a deux étages volontairement simples :

- **L'équipe.** Buts attendus hors penalty de chaque côté : moyenne
  géométrique de ce que l'attaque crée et de ce que la défense adverse
  concède, sur la fenêtre de cinq matchs que `build_features` calcule déjà,
  répartie par le facteur domicile du dataset (1.23). Les penalties reviennent
  au taux moyen mesuré (0.12 xG par équipe et par match). Buts en Poisson,
  d'où « les deux marquent » et « plus de 2,5 buts ».
- **Le joueur.** Son xG hors penalty par 90 minutes, cumulé sur la saison en
  cours et la précédente (Understat, `src/data/players.py`, mapping
  d'équipes explicite comme partout), rétréci vers le taux moyen de la ligue
  avec le poids de 8 matchs, puis multiplié par le rapport entre les buts
  attendus de ce match et la moyenne de son équipe. Probabilité de marquer au
  moins une fois : `1 − exp(−cela)`. Joueurs à plus de 900 minutes seulement,
  gardiens exclus. Un joueur transféré apparaît sous son nouveau club dès
  qu'il y a joué.

Ces probabilités **ne sont pas validées comme le 1X2** : elles supposent que le
joueur démarre et joue 90 minutes, ignorent blessures, suspensions et
rotations, et n'ont pas encore été mesurées face aux buteurs réels. Le site le
dit à côté des chiffres. Les mesurer est la suite logique, une fois assez de
matchs publiés joués.

## Rapports générés

| Fichier | Contenu |
|---|---|
| `reports/model_report.md` | résultats du test, garde-fou, calibration par tranche, biais par classe, importance des features |
| `reports/walk_forward.md` | stabilité saison par saison et saturation de l'historique |
| `reports/information.md` | test d'information : poids du modèle une fois le marché connu, mélange hors échantillon |
| `reports/oos_predictions.parquet` | prédictions hors échantillon des 4 saisons walk-forward, modèle et marché, réutilisables sans réentraîner |
| `reports/proposals_backtest.parquet` | le combinateur rejoué sur la dernière saison complète, une ligne par sélection |
| `reports/tracking.md` | calibration des prédictions publiées |

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
