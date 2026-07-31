# 06 — Few-shot recovery

Final validator PASS; schema `few_shot_recovery.v1`; analysis `7e4ca051c3e83074391652e28a163138129c1cb6610f8826248a90cd3d19409a`. Full exact ROC-AUC, PR-AUC and Brier records are in `tables/few_shot_recovery.csv`. Equal block effort is not equal target percentage; selection is label-aware, supervised and relatively optimistic. Selection intervals describe repeated block selection and are not confidence intervals.

| budget_blocks | directions_median_positive | n_directions | thermal_median_recovery_fraction |
| --- | --- | --- | --- |
| 4 | 5 | 6 | 0.14994732994273288 |
| 8 | 5 | 6 | 0.28487877616821755 |
| 16 | 6 | 6 | 0.4659658839037918 |
| 32 | 6 | 6 | 0.7066147883047805 |
