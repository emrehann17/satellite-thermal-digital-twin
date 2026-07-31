# 05 — Evia extended signed-AUC bootstrap

Evia is a high-prevalence sensitivity AOI, not a direct equal-prevalence comparison with the primary three regions.

## Positive-direction supported

| feature | auc_point_estimate | bootstrap_lower | bootstrap_upper | direction | interval_includes_0_5 | support_token |
| --- | --- | --- | --- | --- | --- | --- |
| ndvi_mean | 0.6391494562092995 | 0.5749285449525442 | 0.7006046421340476 | higher_values_rank_burned | False | bootstrap_supported_positive_direction |
| lst_anomaly_mean | 0.6400549435604515 | 0.5671047052827075 | 0.710175869871198 | higher_values_rank_burned | False | bootstrap_supported_positive_direction |

## Negative-direction supported

| feature | auc_point_estimate | bootstrap_lower | bootstrap_upper | direction | interval_includes_0_5 | support_token |
| --- | --- | --- | --- | --- | --- | --- |
| current_lst_mean | 0.37651907667348405 | 0.3012790089007279 | 0.45585820887246337 | lower_values_rank_burned | False | bootstrap_supported_negative_direction |
| current_tvdi_mean | 0.36188826587369816 | 0.2852365355103699 | 0.44217142834635265 | lower_values_rank_burned | False | bootstrap_supported_negative_direction |
| downscaled_lst_mean | 0.3765491129362415 | 0.29685914048213763 | 0.4593307010841324 | lower_values_rank_burned | False | bootstrap_supported_negative_direction |
| fused_lst_mean | 0.37566126384147186 | 0.30015191434549077 | 0.4556221037116367 | lower_values_rank_burned | False | bootstrap_supported_negative_direction |

## Interval includes 0.5

| feature | auc_point_estimate | bootstrap_lower | bootstrap_upper | direction | interval_includes_0_5 | support_token |
| --- | --- | --- | --- | --- | --- | --- |
| elevation_mean | 0.540559580910425 | 0.4483537317612215 | 0.626146085073554 | higher_values_rank_burned | True | interval_includes_zero |
| slope_mean | 0.486543098344048 | 0.41769297357779134 | 0.554042450008236 | lower_values_rank_burned | True | interval_includes_zero |
| tvdi_difference_mean | 0.5191031776643451 | 0.4444187250619392 | 0.5889301019447588 | higher_values_rank_burned | True | interval_includes_zero |
