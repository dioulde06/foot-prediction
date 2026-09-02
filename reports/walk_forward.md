# Validation walk-forward

Pour chaque saison de test : entraînement sur tout ce qui précède la
saison de calibration, calibration sur la saison immédiatement
antérieure, test sur la saison elle-même.

## Stabilité

| test_season | n_train_seasons | n_train | n_test | log_loss | brier | ece | accuracy_info | market_log_loss | gap_to_market | temperature | best_iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-23 | 1 | 1826 | 1826 | 1.0106 | 0.6038 | 0.0088 | 0.5246 | 0.9753 | 0.0353 | 1.011 | 64 |
| 2023-24 | 2 | 3652 | 1752 | 0.9891 | 0.5887 | 0.0101 | 0.5331 | 0.9521 | 0.037 | 0.985 | 84 |
| 2024-25 | 3 | 5478 | 1752 | 0.9925 | 0.592 | 0.0111 | 0.5263 | 0.96 | 0.0325 | 0.931 | 91 |
| 2025-26 | 4 | 7230 | 1751 | 1.0044 | 0.5989 | 0.0106 | 0.5151 | 0.9769 | 0.0275 | 1.036 | 131 |

Amplitude du log-loss sur les saisons testées : **0.0215**.
Écart au marché : de +0.0275 à +0.0370, médiane +0.0339.

Une moyenne n'est pas donnée volontairement : c'est la dispersion qui
dit si le modèle est réel, et une moyenne la masque.

## Saturation de l'historique, testée sur 2025-26

| test_season | n_train_seasons | n_train | n_test | log_loss | brier | ece | accuracy_info | market_log_loss | gap_to_market | temperature | best_iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-26 | 1 | 1752 | 1751 | 1.0073 | 0.6009 | 0.0096 | 0.5128 | 0.9769 | 0.0304 | 0.995 | 76 |
| 2025-26 | 2 | 3578 | 1751 | 1.005 | 0.5992 | 0.0093 | 0.5191 | 0.9769 | 0.0282 | 0.995 | 85 |
| 2025-26 | 3 | 5404 | 1751 | 1.0038 | 0.5985 | 0.0071 | 0.514 | 0.9769 | 0.0269 | 0.984 | 87 |
| 2025-26 | 4 | 7230 | 1751 | 1.0044 | 0.5989 | 0.0106 | 0.5151 | 0.9769 | 0.0275 | 1.036 | 131 |

Prompt 5.1 demandait 2, 4 et 6 saisons d'entraînement. Avec six saisons
de données dont une pour la calibration et une pour le test, quatre est
le maximum disponible : le balayage va donc de 1 à 4.

