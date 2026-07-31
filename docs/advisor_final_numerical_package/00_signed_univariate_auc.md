# 00 — Signed univariate AUC

The frozen rule requires opposite point directions and both ~5 km spatial-block intervals to exclude 0.5 for a **bootstrap-supported regional association reversal**. Opposite point estimates alone are insufficient.

| experiment_id | feature | signed_auc_point_estimate | bootstrap_lower | bootstrap_upper | direction | interval_includes_0_5 | reversal_pair | reversal_support_token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bejis_2022 | elevation_mean | 0.6432824053164721 | 0.5583150457861116 | 0.7290055219147464 | higher_values_rank_burned | False | manavgat_2021 | bootstrap_supported_reversal |
| bejis_2022 | current_lst_mean | 0.47674775069349284 | 0.4011532923534438 | 0.5474889652503846 | lower_values_rank_burned | True | manavgat_2021 | point_reversal_interval_uncertain |
| bejis_2022 | current_tvdi_mean | 0.5172782323463783 | 0.42900913810605534 | 0.595208856020078 | higher_values_rank_burned | True | mugla_2021 | point_reversal_interval_uncertain |
| bejis_2022 | downscaled_lst_mean | 0.4836430737466933 | 0.4003605903074035 | 0.5598325680030973 | lower_values_rank_burned | True | manavgat_2021 | point_reversal_interval_uncertain |
| bejis_2022 | fused_lst_mean | 0.4806129105103555 | 0.40394725127325576 | 0.5514025133468518 | lower_values_rank_burned | True | manavgat_2021 | point_reversal_interval_uncertain |
| manavgat_2021 | elevation_mean | 0.37410755666893913 | 0.2890796631009303 | 0.4711534398242251 | lower_values_rank_burned | False | bejis_2022;mugla_2021 | bootstrap_supported_reversal |
| manavgat_2021 | current_lst_mean | 0.5382779853437281 | 0.45176227230170396 | 0.620503038933485 | higher_values_rank_burned | True | bejis_2022;mugla_2021 | point_reversal_interval_uncertain |
| manavgat_2021 | current_tvdi_mean | 0.5520208992479892 | 0.4602480293883392 | 0.6410618349429631 | higher_values_rank_burned | True | mugla_2021 | point_reversal_interval_uncertain |
| manavgat_2021 | downscaled_lst_mean | 0.5520694210669516 | 0.4659633972389294 | 0.637150621852662 | higher_values_rank_burned | True | bejis_2022;mugla_2021 | point_reversal_interval_uncertain |
| manavgat_2021 | fused_lst_mean | 0.5400826188182982 | 0.4542805043929701 | 0.6218679874478806 | higher_values_rank_burned | True | bejis_2022;mugla_2021 | point_reversal_interval_uncertain |
| mugla_2021 | elevation_mean | 0.6114025048859928 | 0.5319021930545539 | 0.6904319830630276 | higher_values_rank_burned | False | manavgat_2021 | bootstrap_supported_reversal |
| mugla_2021 | current_lst_mean | 0.324818209196232 | 0.2713831954453114 | 0.3821312123411832 | lower_values_rank_burned | False | manavgat_2021 | point_reversal_interval_uncertain |
| mugla_2021 | current_tvdi_mean | 0.3357699119089857 | 0.27535685410747573 | 0.39757500369355586 | lower_values_rank_burned | False | bejis_2022;manavgat_2021 | point_reversal_interval_uncertain |
| mugla_2021 | downscaled_lst_mean | 0.30697725296436723 | 0.25320481926586585 | 0.36583706101459185 | lower_values_rank_burned | False | manavgat_2021 | point_reversal_interval_uncertain |
| mugla_2021 | fused_lst_mean | 0.32521434622074175 | 0.2720012810467721 | 0.3827633213333897 | lower_values_rank_burned | False | manavgat_2021 | point_reversal_interval_uncertain |
