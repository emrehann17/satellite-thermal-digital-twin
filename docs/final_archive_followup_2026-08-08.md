## BurnDate DOY availability

The canonical Step8A parquet files for all five final AOIs already contain
cell-level MCD64A1 BurnDate information. Therefore, no new BurnDate export is
required.

Canonical fields:
- `burn_date`
- `burn_day_of_year`

Both fields were checked across all five AOIs and are identical row by row.

Semantics:
- `burn_date == 0` means the cell is unburned within the label window.
- `burn_date > 0` is MCD64A1 BurnDate expressed as day-of-year (DOY).
- `burned == 1` is exactly equivalent to `burn_date > 0` in all five datasets.
- `out_of_window_burndate == True` count is zero in all five AOIs.

Final audit results:

| AOI | Raw burned | Eligible + valid burned | TSG burned | TSG BurnDate DOYs |
| --- | ---: | ---: | ---: | --- |
| manavgat_2021 | 796 | 796 | 784 | 213–221, 241 |
| bejis_2022 | 1103 | 1103 | 1100 | 227–234, 237 |
| mugla_2021 | 3073 | 3026 | 2911 | 210–224, 231, 235 |
| evia_2021_extended | 2803 | 2788 | 2664 | 215–224, 226 |
| montiferru_2021 | 748 | 697 | 539 | 205–211, 213, 219 |

Eligibility notes:
- For AOIs with pre-label burn exclusion, downstream BurnDate analyses must
  preserve the existing `analysis_eligible` and `valid_for_modeling` filters.
- Bejís has no `analysis_eligible` column because no equivalent pre-label
  exclusion is applied there; `valid_for_modeling` is sufficient.
- For analyses restricted to the paper's primary natural-vegetation
  population, additionally apply:
  `burnable_tree_shrub_grass == True`.

This distinction is important because the raw number of positive BurnDate
cells is larger than the final analysis population in Muğla, Evia Extended,
and Montiferru due to excluded pre-label burns.

Advisor request status:
Cell-level BurnDate DOY data are already available in the five canonical
Step8A parquet files. No additional Earth Engine export is necessary.

## FIRMS cross-check status

The canonical five-AOI analyses use MCD64A1 BurnDate as the sole burned-area
target. FIRMS active-fire detections were not used as model labels or combined
with MCD64A1, which is intentional and consistent with the frozen target
definition.

A repository-level audit on 2026-08-08 found that the FIRMS MODIS+VIIRS
cross-check infrastructure exists, but no canonical FIRMS–MCD64A1 comparison
artifacts were produced for the five final AOIs:

- manavgat_2021
- bejis_2022
- mugla_2021
- evia_2021_extended
- montiferru_2021

Actual FIRMS cross-check artifacts were found only in the legacy Kozan
validation outputs.

Interpretation:
- This does NOT invalidate the within-region or cross-region analyses, whose
  estimand is explicitly defined against MCD64A1 BurnDate.
- This is NOT a leakage issue; FIRMS was deliberately kept outside the target
  and modeling pipeline.
- The missing five-AOI FIRMS comparison is instead an external-consistency /
  label-uncertainty limitation.
- FIRMS active-fire detections and MCD64A1 burned-area labels measure related
  but non-identical phenomena, so perfect agreement would not be expected.

Advisor request status:
No existing five-AOI FIRMS–MCD64A1 final artifact is available to archive.
This status should be reported explicitly rather than implying that the
cross-check was performed.

## Window-closure sensitivity

Window-closure sensitivity was completed for all five canonical AOIs before
the advisor's final follow-up request.

Regions:
- manavgat_2021
- bejis_2022
- mugla_2021
- evia_2021_extended
- montiferru_2021

For each AOI, the canonical predictor window was compared against predictor
windows closed 7 and 14 days earlier while preserving the frozen label window
and the corresponding regional analysis contract.

The final summary has been added to `ozet_sonuclar.xlsx`, including:
- canonical thermal contribution (delta ROC-AUC) and 95% CI,
- 7-day-earlier thermal contribution and 95% CI,
- 14-day-earlier thermal contribution and 95% CI,
- paired change in thermal ROC-AUC relative to canonical,
- PR-AUC change,
- Brier change,
- bootstrap support status.

Main result:
Thermal contribution remains positive and bootstrap-supported in all five
regions under both the 7-day and 14-day earlier closure variants.

However, the effect of moving the predictor window earlier is region-specific:
- Manavgat and Bejís improve under earlier closure.
- Muğla is weak/uncertain at 7 days but improves in ROC-AUC at 14 days.
- Evia Extended weakens under earlier closure.
- Montiferru shows little clear ROC-AUC sensitivity to the closure shift.

Therefore, the positive thermal contribution is not dependent on predictors
extending immediately up to the fire-label period. The five-region replication
provides substantially stronger evidence against a simple temporal-leakage
explanation than the original Manavgat-only sensitivity analysis.

Archive status:
Raw window-closure analysis artifacts for all five AOIs are already present in
the archive, and the consolidated results have now been added to the summary
workbook.

Implementation note:
Manavgat was produced under the earlier
`outputs/diagnostics/window_closure_sensitivity/` namespace, whereas the
subsequent regional runs use
`outputs/diagnostics/window_closure_region/`. This is a namespace/history
difference, not a missing analysis.

## Muğla 2022 temporal-transfer — initial gate attempt (provisional)

Initial Stage-1 temporal-transfer gate was run for `mugla_2022` using the
pre-specified one-calendar-year shift of the `mugla_2021` windows:

- predictor window: 2022-06-01 -> 2022-07-28
- label window: 2022-07-29 -> 2022-09-15
- baseline years: 2018-2021
- AOI / region: unchanged from `mugla_2021` (`mugla_aoi`)
- `exclude_pre_label_burns = True`

Observed gate result:

- decision: `insufficient_burned_positives`
- label-window burned cells: 12
- minimum positive threshold: 30
- predictor/pre-label burned cells excluded: 332
- first nonzero BurnDate in the exported combined span: 2022-06-21
- `downstream_authorized = False`

Interpretation:

The major 2022 burn signal is present in MCD64A1 but falls inside the
predictor/pre-label period under the exact +1-calendar-year design. Those
332 pre-label burned cells are therefore excluded from the analysis universe
rather than treated as unburned negatives. Only 12 burned cells remain in the
label window, so the gate fails.

No downstream predictor export, Step5/Step7/Step8, model fit, transfer, or
bootstrap was started after this gate failure.

Status: PROVISIONAL / AWAITING SUPERVISOR DECISION.

This entry does NOT freeze the final `mugla_2022` temporal contract. If the
supervisor explicitly requests an event-relative date design or another
pre-specified temporal contract, that decision must be recorded separately
before any rerun. The current gate result and artifacts must not be silently
overwritten.
