# Prompts de démarrage — `foot-calibration`

À utiliser dans l'ordre. Un prompt = une session Claude Code idéalement.
Ne passe pas au suivant tant que le précédent n'est pas commité et testé.

---

## Phase 1 — Squelette et ingestion

### Prompt 1.1 — Scaffolding

```
Lis CLAUDE.md.

Crée la structure du projet décrite dans la section Structure :
arborescence, pyproject.toml avec uv, config ruff et mypy stricte,
un .gitignore qui exclut data/ et notebooks/.

Ajoute un Makefile avec les cibles : install, lint, test, fetch, train, eval.
Les cibles peuvent pointer vers des scripts encore vides.

Ne crée aucune logique métier à cette étape.
```

### Prompt 1.2 — Ingestion

```
Implémente src/data/fetch.py.

Objectif : récupérer 6 saisons (2020-21 à 2025-26) des 5 grands
championnats et écrire un parquet par source dans data/raw/.

Sources :
- FBref via soccerdata : résultats et stats de match
- Understat via soccerdata : xG par match
- football-data.co.uk : CSV par saison et par ligue, pour les cotes
  (codes E0, SP1, D1, I1, F1)

Contraintes :
- Cache local : si le parquet existe déjà, ne refetch pas sauf --force
- Rate limiting respectueux entre les requêtes
- Log du nombre de matchs récupérés par ligue et par saison

Écris d'abord la signature des fonctions et montre-la moi avant
d'implémenter.
```

### Prompt 1.3 — Le mapping des noms

```
Implémente src/data/team_mapping.py et src/data/merge.py.

merge.py doit joindre les 3 sources sur la clé (date, home_team, away_team)
après normalisation des noms via le mapping.

Exigences :
- Le mapping est un dict explicite source -> nom canonique
- Toute équipe non mappée lève UnmappedTeamError avec le nom et la source
- Écris un script scripts/audit_teams.py qui liste les noms distincts par
  source et signale ceux qui ne matchent pas, pour que je complète le dict
  manuellement
- Le merge doit logger le taux de matchs joints ; si < 98 %, lève une erreur

Commence par l'audit, je veux voir la liste avant qu'on écrive le mapping.
```

---

## Phase 2 — Baselines

### Prompt 2.1

```
Lis la section Baselines de CLAUDE.md.

Implémente src/eval/baselines.py et src/eval/metrics.py.

metrics.py : log_loss, brier_score_multiclass, expected_calibration_error,
calibration_bins (retourne les tranches inégales : 0-20, 20-30, 30-40,
40-50, 50-65, 65+ avec n, prédit moyen, observé).

baselines.py : les 3 baselines. Pour market, implémente la dévigorisation
des cotes (méthode multiplicative simple, documente la formule).

Puis un script scripts/run_baselines.py qui affiche un tableau comparatif
sur la saison 2025-26.

Tests unitaires obligatoires sur les métriques avec des cas connus.
```

---

## Phase 3 — Features

### Prompt 3.1 — Elo

```
Implémente src/features/elo.py.

Elo classique adapté au football : K=20, avantage domicile en points Elo
(paramètre, défaut 65), régression vers la moyenne de 25 % entre saisons
pour gérer promus/relégués.

L'API : une classe qui consomme les matchs dans l'ordre chronologique et
expose get_rating(team, date) qui ne retourne que l'état AVANT cette date.

Test critique : vérifier qu'un appel à get_rating pour un match donné ne
change pas selon que les matchs postérieurs ont été ingérés ou non.
```

### Prompt 3.2 — Rolling features

```
Implémente src/features/build.py avec la fonction centrale :

build_features(matches: pl.DataFrame, cutoff_date: date) -> pl.DataFrame

Pour chaque match, uniquement des données antérieures :
- Rolling 5 et 10 matchs : xG marqué/concédé, buts, tirs cadrés, points
- Versions séparées domicile-only et extérieur-only
- Différentiel Elo (domicile - extérieur)
- Jours de repos depuis le dernier match, par équipe
- Journée de championnat
- Force moyenne des adversaires rencontrés sur la fenêtre (Elo moyen)

Les colonnes finales doivent être des différentiels home - away quand ça
a du sens, en plus des valeurs brutes.

AVANT d'implémenter : écris tests/test_no_leakage.py qui construit un
dataset synthétique où le résultat du match N+1 est aberrant, et vérifie
que les features du match N sont identiques avec et sans ce match N+1
dans l'input. Montre-moi le test d'abord.
```

---

## Phase 4 — Modèle

### Prompt 4.1

```
Implémente src/models/train.py.

LightGBM multiclasse (3 classes), split chronologique strict :
- train : saisons 2020-21 à 2023-24
- valid : 2024-25
- test  : 2025-26

sample_weight exponentiel, demi-vie configurable, défaut 2 saisons.
Early stopping sur le log-loss de validation.
Hyperparamètres dans un fichier de config YAML, pas en dur.

Sauvegarde le modèle + les métadonnées (dates de split, features utilisées,
hash du dataset) dans models/.

Rappel CLAUDE.md : si l'accuracy dépasse 60 %, arrête et signale-le.
```

### Prompt 4.2 — Calibration

```
Implémente src/models/calibrate.py.

Calibration isotonique multiclasse (one-vs-rest puis renormalisation)
ajustée sur le set de validation uniquement.

Puis src/eval/report.py qui produit sur le TEST :
- log-loss et Brier avant / après calibration
- les 3 baselines
- le tableau des tranches de calibration
- l'ECE et le biais moyen

Le rapport sort en markdown dans reports/.

Attention : les bins de calibration se calculent sur le test, jamais sur
la validation qui a servi à calibrer.
```

---

## Phase 5 — Validation

### Prompt 5.1

```
Implémente src/eval/walk_forward.py.

Walk-forward saison par saison : pour chaque saison S de 2023-24 à 2025-26,
entraîner sur tout ce qui précède, calibrer sur S-1, tester sur S.

Sortie : un tableau log-loss / Brier / ECE par saison, plus l'écart au
market baseline. Je veux voir la stabilité, pas une moyenne.

Ajoute une expérience : refais tourner avec 2, 4, 6 saisons de train pour
voir où le gain sature. Sors un petit graphe matplotlib dans reports/.
```

---

## Prompts de maintenance

À garder sous la main.

### Audit de fuite

```
Passe en revue src/features/build.py et src/models/train.py.
Cherche uniquement les fuites temporelles potentielles. Pour chaque
feature, dis-moi quelle donnée elle consomme et à quelle date cette donnée
devient disponible. Ne corrige rien, liste d'abord.
```

### Revue avant commit

```
Diff en cours : vérifie la conformité à CLAUDE.md, en particulier les
règles de split et de métriques. Signale tout ce qui dévie. Ne modifie pas
le code.
```

### Nouvelle feature

```
Je veux ajouter la feature suivante : [DESCRIPTION].

Avant d'écrire quoi que ce soit :
1. Dis-moi si elle est calculable sans fuite, et à partir de quelle donnée
2. Dis-moi si elle est probablement redondante avec une feature existante
3. Propose le test de non-fuite correspondant

Attends ma validation avant d'implémenter.
```
