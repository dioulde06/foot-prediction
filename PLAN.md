# PLAN.md — `foot-calibration`

Plan d'exécution **et** d'apprentissage. Complète `PROMPTS.md` (le quoi)
en fixant le protocole (le comment on travaille à deux).

## Règle du jeu

1. **Une phase = une session.** On ne passe pas à la suivante tant que le
   checkpoint n'est pas passé, même si le code marche.
2. **Pré-enregistrement.** Avant chaque expérience, tu écris dans
   `JOURNAL.md` le chiffre que tu attends. Puis on regarde. C'est ce qui
   calibre *toi*, pas le modèle.
3. **Checkpoint = tu m'expliques, sans regarder le code.** Si tu n'y
   arrives pas, on ne commite pas : je réexplique autrement.
4. **Je ne code jamais un truc critique sans que tu aies vu la signature
   ou le test d'abord.** Déjà dans CLAUDE.md, je le tiens.
5. **Ta part n'est pas décorative.** Sur chaque phase il y a un morceau
   que tu écris toi. Il est listé ci-dessous.

Répartition cible sur l'ensemble du projet : tu écris ~25 % du code, mais
**100 % des décisions** (features retenues, seuils, arbitrages).

---

## Phase 0 — Fondations (aucun code)

Le seul moyen de ne pas subir les phases suivantes.

**Ce que tu dois comprendre**
- Pourquoi la calibration ≠ l'accuracy. Un modèle qui dit "45 %" doit
  avoir raison 45 fois sur 100, pas plus, pas moins.
- Log-loss : ce que ça punit (la confiance à tort), et pourquoi une
  proba de 0 sur un match qui arrive = pénalité infinie.
- Brier : la différence avec le log-loss, et quand ils divergent.
- ECE : la traduction en "de combien je me trompe en moyenne, par tranche".
- La dévigorisation : pourquoi la somme des probas implicites du bookmaker
  fait 1.05 et pas 1.00, et où passent ces 5 %.

**Ta part (à la main, papier ou tableur)**
1. Trois matchs, trois jeux de probas inventés, tu calcules le log-loss.
2. Un jeu de cotes réel (2.10 / 3.40 / 3.60) : tu calcules la marge, tu
   dévigorises en multiplicatif, tu vérifies que ça somme à 1.
3. Tu réponds : "pourquoi la baseline `market` est probablement
   imbattable ?" — en une phrase, avec l'argument économique.

**Ma part** — je t'explique chaque point à la demande, je corrige tes
calculs, je ne code rien.

**Checkpoint** — tu m'énonces la différence log-loss / Brier / ECE et tu
me donnes un cas où deux modèles ont la même accuracy et un log-loss très
différent.

**Sortie** — `JOURNAL.md` créé, avec tes calculs et ta prédiction de
performance finale du projet (log-loss et écart au market). Datée. On la
relira en Phase 5.

Durée : 1 session.

---

## Phase 1 — Squelette, ingestion, mapping

Prompts 1.1 → 1.3 de `PROMPTS.md`.

**Ce que tu dois comprendre**
- Pourquoi le parquet et pas le CSV (typage, taille, pas de diff pourri).
- Pourquoi le mapping des noms est *le* risque silencieux du projet :
  une erreur ne lève rien, elle décale des lignes et pollue tout.
- Ce qu'est une clé de jointure et pourquoi `(date, home, away)` est
  fragile (matchs reportés, fuseaux, dates FBref vs football-data).

**Ta part**
- Tu remplis le dict de `team_mapping.py` **toi-même**, à partir de la
  sortie de `scripts/audit_teams.py`. Je ne devine aucun nom. C'est
  fastidieux et c'est exactement pour ça que c'est formateur : à la fin
  tu connais tes données.
- Tu décides du seuil de jointure acceptable (le 98 % de CLAUDE.md est
  un point de départ, pas une vérité).

**Ma part** — scaffolding, `fetch.py`, `merge.py`, le script d'audit,
les tests de jointure.

**Checkpoint** — tu m'expliques ce qui se passe si Nottingham Forest est
mappé en "Nottingham" chez FBref et "Nott'm Forest" chez football-data et
qu'on utilise du fuzzy matching. Et pourquoi ça ne plantera pas.

**Sortie** — `data/raw/*.parquet` + `data/processed/matches.parquet`,
taux de jointure loggé, mapping complet, tests verts.

Durée : 2-3 sessions (le mapping prend du temps, c'est normal).

---

## Phase 2 — Baselines et métriques

Prompt 2.1.

**Ce que tu dois comprendre**
- Un chiffre sans baseline ne veut rien dire. C'est la phase qui rend
  toutes les suivantes interprétables.
- Les bins inégaux : pourquoi on ne découpe pas en 10 tranches égales
  (il n'y a presque rien au-dessus de 65 %).
- La différence entre "mon modèle est bon" et "mon modèle bat quelque
  chose de stupide".

**Ta part**
- **Avant** de lancer `run_baselines.py`, tu écris dans `JOURNAL.md` les
  trois log-loss que tu attends (`uniform`, `home`, `market`). Trois
  chiffres. Puis on compare. L'écart entre ton intuition et le réel est
  l'information la plus utile de la phase.
- Tu écris toi-même **un** des tests de `metrics.py` (je propose lequel :
  le cas connu du log-loss sur une proba parfaite et sur une proba nulle).

**Ma part** — `metrics.py`, `baselines.py`, la dévigorisation documentée,
le script comparatif, le reste des tests.

**Checkpoint** — tu me dis, en regardant le tableau, laquelle des trois
baselines est la vraie cible et pourquoi les deux autres servent quand
même à quelque chose.

**Sortie** — tableau des 3 baselines sur 2025-26. C'est notre étalon
jusqu'à la fin. Il est recalculé à chaque changement de dataset.

Durée : 1 session.

---

## Phase 3 — Features (la phase la plus dangereuse)

Prompts 3.1 → 3.2.

**Ce que tu dois comprendre**
- La fuite temporelle : les 4 formes qu'elle prend ici (stats du match
  lui-même, rolling qui inclut le match courant, Elo mis à jour trop tôt,
  normalisation calculée sur tout le dataset).
- Pourquoi un test de non-fuite se construit sur un dataset **synthétique
  aberrant** et pas sur les vraies données.
- Elo : l'intuition (transfert de points à somme nulle), le rôle de K, et
  pourquoi il faut régresser vers la moyenne entre saisons.

**Ta part — la plus importante du projet**
- Tu écris `tests/test_no_leakage.py` toi-même. Je t'aide sur la syntaxe
  polars si besoin, mais la logique du test est de toi : quel dataset
  synthétique, quelle aberration, quelle assertion.
- Pour **chaque** feature de la liste, tu remplis un tableau à 3
  colonnes : `feature | donnée consommée | date de disponibilité`. Si tu
  n'arrives pas à remplir la 3e colonne, la feature ne rentre pas.
- Tu arbitres : on garde les 15 features de la liste, ou on démarre avec
  6 et on ajoute ? (mon avis : on démarre avec 6, je te dirai lesquelles.)

**Ma part** — `elo.py`, `build.py`, le test d'invariance temporelle de
l'Elo, et je relis ton test de fuite en essayant de le casser.

**Checkpoint** — tu m'expliques pourquoi `get_rating(team, date)` doit
retourner l'état *avant* la date, et ce qui se passerait sinon sur le
log-loss (indice : ça deviendrait très beau, et faux).

**Sortie** — `build_features()` + tests de non-fuite verts + ton tableau
de disponibilité des données. Ce tableau est un livrable, on le garde.

Durée : 3-4 sessions.

---

## Phase 4 — Modèle et calibration

Prompts 4.1 → 4.2.

**Ce que tu dois comprendre**
- Pourquoi les probas brutes d'un LightGBM sont mal calibrées (l'objectif
  optimise le log-loss mais l'early stopping et la profondeur poussent à
  la surconfiance).
- Régression isotonique : monotone, non paramétrique, et le piège du
  one-vs-rest qui ne somme plus à 1.
- **Le point crucial** : calibrer sur validation, mesurer sur test.
  Calibrer et mesurer sur le même set, c'est se mentir.
- `sample_weight` exponentiel : pourquoi une saison de 2020 pèse moins.

**Ta part**
- Tu prédis (JOURNAL.md) le log-loss test avant et après calibration.
- Tu lis le tableau des tranches de calibration et tu diagnostiques
  **toi-même** : le modèle est-il surconfiant, sous-confiant, ou biaisé
  sur une classe ? Je ne te donne pas la réponse avant la tienne.
- Tu fixes la demi-vie du `sample_weight` et tu justifies.

**Ma part** — `train.py`, `calibrate.py`, `report.py`, la config YAML,
la sauvegarde des métadonnées.

**Garde-fou** — si l'accuracy dépasse 60 % ou le log-loss descend sous
0.85, on s'arrête et on cherche la fuite. Ensemble. Ce n'est pas une
bonne nouvelle, c'est un bug. (Règle 2 de CLAUDE.md, je la respecte.)

**Checkpoint** — tu m'expliques ce que devient le rapport si on calibre
sur le test, et pourquoi le résultat serait excellent et inutilisable.

**Sortie** — un rapport markdown dans `reports/` : log-loss et Brier
avant/après calibration, les 3 baselines, les tranches, l'ECE, le biais.

Durée : 2-3 sessions.

---

## Phase 5 — Validation walk-forward

Prompt 5.1.

**Ce que tu dois comprendre**
- Pourquoi une seule mesure test ne prouve rien : la variance saison à
  saison est plus grande que l'écart entre deux modèles corrects.
- Lire la **stabilité**, pas la moyenne. Trois saisons à 1.01 / 1.00 /
  1.02 valent mieux que 0.95 / 1.08 / 0.99 à moyenne égale.
- La courbe de saturation : à partir de combien de saisons de train le
  gain devient du bruit.

**Ta part**
- Tu relis ta prédiction de Phase 0 et tu confrontes. Écart honnête,
  écrit dans le journal.
- **Tu décides** : est-ce que ce modèle est réel ou du bruit ? Je te
  donne les chiffres et mon avis, la décision est à toi. C'est la
  compétence qu'on cherche à construire depuis le début.
- Tu choisis la config finale (nb de saisons de train, demi-vie).

**Ma part** — `walk_forward.py`, l'expérience 2/4/6 saisons, le graphe.

**Checkpoint** — tu me montres, sur le tableau, un chiffre que tu refuses
d'interpréter parce qu'il est probablement du bruit — et tu me dis
pourquoi.

**Sortie** — tableau par saison (log-loss / Brier / ECE + écart au
market), graphe de saturation, et une décision écrite : go / no-go sur la
config retenue.

Durée : 2 sessions.

---

## Phase 6 — L'app : prédire avant, mesurer après

Pas encore dans `PROMPTS.md`. C'est l'objectif final de CLAUDE.md.

**Ce que tu dois comprendre**
- La différence entre backtest et production : en prod il n'y a pas de
  deuxième essai, et le dataset se met à jour tout seul (donc casse tout
  seul).
- Pourquoi on publie **avant** la journée et on horodate : sinon
  l'évaluation n'est plus honnête, et tu ne pourras plus le prouver.
- La dérive : un modèle calibré en 2025 ne l'est pas forcément en 2027.

**Ta part**
- Tu définis le contrat de publication : quoi, quand, sous quelle forme.
- Tu décides ce qui déclenche un ré-entraînement.

**Ma part** — job de prédiction hebdo (parquet horodaté + append-only),
job de réconciliation des résultats, page de suivi de calibration dans le
temps face au market.

**Checkpoint** — tu m'expliques comment quelqu'un de sceptique pourrait
vérifier que tu n'as pas triché sur tes prédictions passées.

**Sortie** — prédictions publiées automatiquement avant chaque journée,
historique append-only, courbe de calibration glissante.

Durée : 3-4 sessions.

---

## Prompts de maintenance

Ceux de `PROMPTS.md` (audit de fuite, revue avant commit, nouvelle
feature) sont à lancer **entre** les phases, pas seulement à la fin.
En particulier : audit de fuite obligatoire à la sortie de la Phase 3 et
de la Phase 4.

## Ce qui n'est pas dans ce plan (volontairement)

- Réseaux de neurones, embeddings d'équipes, xG maison : plus tard, si
  les baselines sont battues. Elles ne le seront probablement pas.
- Paris réels, gestion de bankroll, Kelly : hors périmètre, autre projet,
  autres risques.
- Autres championnats, autres marchés (over/under, handicap) : après une
  Phase 6 stable.
