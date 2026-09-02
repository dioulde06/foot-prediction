# JOURNAL.md

Journal de bord. Prédictions écrites **avant** de voir les résultats,
puis confrontation. Rien ne s'efface, on ajoute.

---

## Phase 0 — Fondations (2026-09-02)

### Exercice 1 — Log-loss à la main

Trois matchs. Les probas du modèle, et l'issue réellement survenue.

| Match | p(1) domicile | p(X) nul | p(2) extérieur | Issue réelle |
|---|---|---|---|---|
| A | 0.50 | 0.25 | 0.25 | **1** |
| B | 0.30 | 0.30 | 0.40 | **X** |
| C | 0.60 | 0.25 | 0.15 | **2** |

À calculer :

1. La pénalité log-loss de chaque match : `-ln(p)` sur l'issue réalisée.
   - Match A : 
   - Match B : 
   - Match C : 
2. Le log-loss moyen des trois : 
3. Le log-loss de la baseline `uniform` naïve (1/3 partout) sur ces
   mêmes trois matchs : 
4. **La question qui compte** : ce modèle fait-il mieux ou moins bien
   que « ne rien savoir » ? Réponse et explication en une phrase :
   

### Exercice 2 — Dévigorisation

Cotes réelles : **1 → 2.10 | X → 3.40 | 2 → 3.60**

1. Probabilités implicites brutes (`1/cote`), à 4 décimales :
   - p(1) = 
   - p(X) = 
   - p(2) = 
2. Leur somme : 
3. L'overround (`somme - 1`), en % : 
4. La marge réelle du bookmaker (`1 - 1/somme`), en % : 
   Pourquoi ces deux chiffres diffèrent, en une phrase :
   
5. Probas dévigorisées en multiplicatif, et vérification qu'elles
   somment bien à 1.0000 :
   - p(1) = 
   - p(X) = 
   - p(2) = 
   - somme = 

### Exercice 3 — L'argument économique

Pourquoi la baseline `market` est-elle probablement imbattable ?
Une phrase, avec l'argument économique — pas « parce qu'ils ont plus de
données ».



### Prédiction de fin de projet

À écrire maintenant, avant d'avoir vu la moindre donnée. On relira cette
ligne en Phase 5 et on mesurera l'écart honnêtement.

- Log-loss test que j'espère atteindre : 
- Log-loss que j'attends pour la baseline `market` : 
- Écart au market que je pense obtenir (signé) : 
- Accuracy que j'attends (pour info seulement) : 

### Checkpoint Phase 0

À me répéter sans regarder ce fichier :

1. La différence entre log-loss, Brier et ECE.
2. Un cas concret où deux modèles ont la même accuracy et un log-loss
   très différent.
