# 09 — CORAL λ sensitivity

Deep validator PASS; schema `coral_lambda_sensitivity.v1`; analysis `b74d643edc359e62213f4b8fc26f128512de60ed3f2127c317722d6b2d27d17a`. Exact grid: `0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1`. No λ selection was performed. Bejís↔Muğla thermal CORAL behaviour was largely preserved across λ=0…0.1; the result is not solely an artefact of λ=1e-5.

| direction | model_family | metric | max_absolute_deviation_from_canonical | magnitude_token | numerical_instability_present | instability_token | n_numerical_failures | canonical_lambda_value | grid_minimum | grid_maximum | metric_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bejis_2022_to_mugla_2021 | thermal | roc_auc | 0.0015544577137051 | insensitive_over_grid | False |  | 0 | 0.5066377964680288 | 0.5051172496258455 | 0.5081922541817339 | 0.003075004555888383 |
| bejis_2022_to_mugla_2021 | thermal | pr_auc | 0.0002297141425953 | insensitive_over_grid | False |  | 0 | 0.0690723899914225 | 0.06884267584882711 | 0.06926691070479127 | 0.0004242348559641562 |
| bejis_2022_to_mugla_2021 | thermal | brier_score | 0.0005807028574216 | insensitive_over_grid | False |  | 0 | 0.094293529261196 | 0.09387838302753496 | 0.09487423211861769 | 0.0009958490910827317 |
| mugla_2021_to_bejis_2022 | thermal | roc_auc | 0.004729305116459 | insensitive_over_grid | False |  | 0 | 0.560332344022195 | 0.559871056197174 | 0.5650616491386541 | 0.0051905929414800545 |
| mugla_2021_to_bejis_2022 | thermal | pr_auc | 0.001550943671993 | insensitive_over_grid | False |  | 0 | 0.0779051864706781 | 0.0779051864706781 | 0.07945613014267114 | 0.0015509436719930436 |
| mugla_2021_to_bejis_2022 | thermal | brier_score | 0.0012584267164706 | modest_lambda_sensitivity | False |  | 0 | 0.10988088754051736 | 0.10935003832529734 | 0.11113931425698803 | 0.0017892759316906898 |
