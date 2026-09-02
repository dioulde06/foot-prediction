# Rapport modèle — saison de test 2025-26

Généré le 2026-09-02.

## Ce sur quoi le modèle a été entraîné

- Saisons d'entraînement : 2020-21, 2021-22, 2022-23, 2023-24 (7230 matchs, 2020-08-21 → 2024-06-02)
- Saison de validation : 2024-25 (1752 matchs) — sert à l'early stopping **et** à la calibration, jamais à mesurer
- Saison de test : 2025-26 (1751 matchs)
- Empreinte du dataset : `06819a758ab91c1e`
- Arrêt à l'itération 131, log-loss validation 0.9886
- Features (8) : `elo_diff`, `form_points_diff_5`, `goals_scored_diff_5`, `goals_conceded_diff_5`, `shots_target_diff_5`, `np_xg_created_diff_5`, `np_xg_conceded_diff_5`, `rest_days_diff`

## Résultats sur le test

| modele | log_loss | brier | ece | accuracy_info |
|---|---|---|---|---|
| modele brut | 1.0049 | 0.5992 | 0.0145 | 0.5151 |
| modele + isotonique | 1.0193 | 0.6006 | 0.0136 | 0.5037 |
| modele + temperature | 1.0044 | 0.5989 | 0.0106 | 0.5151 |
| uniform | 1.0721 | 0.6485 | 0.0078 | 0.4403 |
| home | 19.3307 | 1.1194 | 0.3731 | 0.4403 |
| market | 0.9769 | 0.5818 | 0.0086 | 0.5346 |

Repères : ne rien savoir vaut `ln(3) = 1.0986`. L'entropie moyenne des probabilités du marché vaut 0.9808, ce qui estime le plancher sous lequel un log-loss n'est plus honnête.

L'accuracy est affichée pour information et ne décide de rien.

## Garde-fou

Aucune alerte : accuracy ≤ 0.6 et log-loss ≥ 0.85.

## Calibration du modèle calibré, par tranche

Calculée sur le **test**. La validation a servi à ajuster le calibrateur, la lire ici décrirait l'ajustement et non le modèle.

| tranche | n | predit_moyen | observe | ecart |
|---|---|---|---|---|
| 0-20 | 910 | 0.1511 | 0.1637 | -0.0126 |
| 20-30 | 1958 | 0.2529 | 0.2605 | -0.0075 |
| 30-40 | 893 | 0.3399 | 0.3303 | 0.0096 |
| 40-50 | 632 | 0.4511 | 0.4256 | 0.0255 |
| 50-65 | 549 | 0.5653 | 0.5683 | -0.003 |
| 65+ | 311 | 0.7049 | 0.6945 | 0.0104 |

ECE = 0.0106, soit l'erreur moyenne en points de pourcentage.

## Biais signé par classe

Le biais global est identiquement nul en multiclasse normalisé : les lignes de probabilités et les issues réalisées somment toutes deux à 1. Seul le biais par classe informe.

| classe | predit_moyen | observe | biais |
|---|---|---|---|
| H | 0.4305 | 0.4403 | -0.0098 |
| D | 0.2542 | 0.2547 | -0.0005 |
| A | 0.3152 | 0.305 | 0.0103 |

## Importance des features

| feature | gain | splits |
|---|---|---|
| elo_diff | 7848.0 | 797.0 |
| np_xg_created_diff_5 | 2322.9 | 698.0 |
| np_xg_conceded_diff_5 | 1573.9 | 729.0 |
| shots_target_diff_5 | 1065.5 | 461.0 |
| goals_conceded_diff_5 | 804.6 | 465.0 |
| form_points_diff_5 | 553.0 | 354.0 |
| goals_scored_diff_5 | 502.3 | 319.0 |
| rest_days_diff | 269.8 | 175.0 |

