# Information test

Geometric blend `p_k ∝ model_k^a · market_k^b`, fitted by maximum likelihood on out-of-sample predictions (train before S-1, calibrate on S-1, predict S). `a` is the weight the outcomes give the model once the market is known.

Pooled over 7081 matches: **a = -0.1262**, b = 1.1383, 95 % bootstrap interval on a [-0.2397, -0.0237].

The interval sits below zero: given the market, the outcomes say to move *away* from the model. Where it disagrees with the market, it is wrong more often than right. Nothing in its inputs is missing from the market; only a new information source can close the gap.

Per season, the blend weights come from the other three seasons, so its log-loss is honest.

| test_season | n    | a_in_season | a_from_other_seasons | b_from_other_seasons | model_log_loss | market_log_loss | blend_log_loss | blend_gain_on_market |
|-------------|------|-------------|----------------------|----------------------|----------------|-----------------|----------------|----------------------|
| 2022-23     | 1826 | -0.0212     | -0.1756              | 1.1923               | 1.0106         | 0.9753          | 0.9761         | -0.0008              |
| 2023-24     | 1752 | -0.173      | -0.11                | 1.1081               | 0.9891         | 0.9521          | 0.9512         | 0.0009               |
| 2024-25     | 1752 | -0.2273     | -0.0951              | 1.1147               | 0.9925         | 0.96            | 0.9592         | 0.0008               |
| 2025-26     | 1751 | -0.1154     | -0.1286              | 1.1439               | 1.0044         | 0.9769          | 0.9765         | 0.0003               |
