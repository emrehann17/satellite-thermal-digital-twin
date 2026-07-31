# 07 — Transfer decomposition and negative recovery

Recovery is always `adapted − raw`; negative values are preserved, never clipped or absolutized. Summary: `{'total': 96, 'negative': 51, 'entirely_below_zero': 46, 'includes_zero': 9}`.

## Bejís ↔ Muğla

| direction | model_family | adaptation_method | metric | raw_value | adapted_value | recovery | interval_lower | interval_upper | chance_reference | recovery_support_status | relative_recovery | negative_recovery_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bejis_2022_to_mugla_2021 | baseline | regionwise_zscore | roc_auc | 0.5922389997163681 | 0.5649705351959404 | -0.027268464520427638 | -0.04551338378508227 | -0.00957357307824532 | 0.5 | negative_recovery | -0.1804890034983209 | True |
| bejis_2022_to_mugla_2021 | baseline | coral_after_regionwise_zscore | roc_auc | 0.5922389997163681 | 0.5701682656206 | -0.022070734095768096 | -0.0390344917815329 | -0.005626792130164209 | 0.5 | negative_recovery | -0.14608540940900497 | True |
| bejis_2022_to_mugla_2021 | baseline | regionwise_zscore | pr_auc | 0.0936314654400828 | 0.0819456319784706 | -0.0116858334616122 | -0.01963310650140617 | -0.004397601289336876 | 0.06975796788880902 | negative_recovery | -0.08899178471647673 | True |
| bejis_2022_to_mugla_2021 | baseline | coral_after_regionwise_zscore | pr_auc | 0.0936314654400828 | 0.0835876596838193 | -0.0100438057562635 | -0.017364543471531574 | -0.003386577689747988 | 0.06975796788880902 | negative_recovery | -0.07648715879202668 | True |
| bejis_2022_to_mugla_2021 | thermal | regionwise_zscore | roc_auc | 0.6184747489978262 | 0.5177332840752557 | -0.10074146492257052 | -0.11836157935932051 | -0.08324667950300661 | 0.5 | negative_recovery | -0.4188689917088648 | True |
| bejis_2022_to_mugla_2021 | thermal | coral_after_regionwise_zscore | roc_auc | 0.6184747489978262 | 0.5066377964680288 | -0.11183695252979742 | -0.1306395364622591 | -0.09269448092031152 | 0.5 | negative_recovery | -0.46500248510336156 | True |
| bejis_2022_to_mugla_2021 | thermal | regionwise_zscore | pr_auc | 0.0925385912286984 | 0.0688239034545231 | -0.02371468777417529 | -0.029689515441706216 | -0.01802637975116373 | 0.06975796788880902 | negative_recovery | -0.06695115380189626 | True |
| bejis_2022_to_mugla_2021 | thermal | coral_after_regionwise_zscore | pr_auc | 0.0925385912286984 | 0.0690723899914225 | -0.0234662012372759 | -0.029345943051765345 | -0.018091127440196687 | 0.06975796788880902 | negative_recovery | -0.06624962821117064 | True |
| mugla_2021_to_bejis_2022 | baseline | regionwise_zscore | roc_auc | 0.4507383379572875 | 0.573885605522937 | 0.12314726756564948 | 0.08916966127282683 | 0.15539397159419607 | 0.5 | supported_recovery_above_chance | 0.2996726328488025 | False |
| mugla_2021_to_bejis_2022 | baseline | coral_after_regionwise_zscore | roc_auc | 0.4507383379572875 | 0.6087499838699271 | 0.15801164591263955 | 0.12475241434251956 | 0.18942213852417158 | 0.5 | supported_recovery_above_chance | 0.38451333015708467 | False |
| mugla_2021_to_bejis_2022 | baseline | regionwise_zscore | pr_auc | 0.0641658658878618 | 0.0852407424254424 | 0.021074876537580597 | 0.012349970739148076 | 0.02959094510928136 | 0.07241606319947334 | supported_recovery_above_chance | 0.08832437405313777 | False |
| mugla_2021_to_bejis_2022 | baseline | coral_after_regionwise_zscore | pr_auc | 0.0641658658878618 | 0.0956030273675055 | 0.031437161479643705 | 0.021298286204564323 | 0.04148848260899585 | 0.07241606319947334 | supported_recovery_above_chance | 0.1317524970903439 | False |
| mugla_2021_to_bejis_2022 | thermal | regionwise_zscore | roc_auc | 0.5831912381443964 | 0.5352625975869411 | -0.04792864055745538 | -0.07644563470906497 | -0.02042224279991708 | 0.5 | negative_recovery | -0.14323310785992194 | True |
| mugla_2021_to_bejis_2022 | thermal | coral_after_regionwise_zscore | roc_auc | 0.5831912381443964 | 0.560332344022195 | -0.022858894122201434 | -0.05181970987924267 | 0.003598666666129796 | 0.5 | negative_recovery | -0.06831302555804525 | True |
| mugla_2021_to_bejis_2022 | thermal | regionwise_zscore | pr_auc | 0.0882719168819328 | 0.0721847069129346 | -0.016087209968998192 | -0.025636331499892136 | -0.008257304968063806 | 0.07241606319947334 | negative_recovery | -0.039281563313231085 | True |
| mugla_2021_to_bejis_2022 | thermal | coral_after_regionwise_zscore | pr_auc | 0.0882719168819328 | 0.0779051864706781 | -0.010366730411254702 | -0.020004869283630877 | -0.0022682709152963853 | 0.07241606319947334 | negative_recovery | -0.025313362465316215 | True |
