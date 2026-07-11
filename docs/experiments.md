# Step0: Experiment / Region Registry

This document explains the Step0 experiment/region selection layer added on
top of the existing Kozan 2023 pipeline (`core/regions.py`, `core/config.py`,
`scripts/main.py`).

## Region vs. experiment

- **Region** = geometry only (an AOI / Area Of Interest). Defined in
  `core/regions.py:build_regions()`.
- **Experiment** = region + year + predictor window + label window +
  baseline years + role + output namespace. Defined in
  `core/regions.py:EXPERIMENTS`.

One region can in principle be reused by multiple experiments (different
years); an experiment always resolves to exactly one region.

## Registered experiments

| experiment_id     | region_key          | role                  | enabled | notes |
|-------------------|---------------------|-----------------------|---------|-------|
| `kozan_2023`      | `kozan_aoi`          | `negative_control`    | yes     | Cropland/anız-burning dominated. Validated methodology (Step8A-8E). Kept as negative/control AOI. |
| `manavgat_2021`   | `manavgat_aoi`       | `anchor_wildfire`      | yes     | Next anchor natural-vegetation wildfire AOI. Registered only — pipeline not yet wired to run it. |
| `bejis_2022`      | `bejis_aoi`          | `mediterranean_transfer_wildfire` | yes | Second Mediterranean transfer wildfire case (Spain), comparable to Manavgat 2021. Registry + initial candidate AOI only — gate/predictors/Step7/Step8 not yet run. |
| `valencia_2022`   | `valencia_2022_aoi`  | `external_validation`  | no      | Placeholder for later Mediterranean external validation. |
| `zamora_2022`     | `zamora_2022_aoi`    | `hard_transfer_test`   | no      | Placeholder for a later, harder transfer test. |

`kozan_2023` is the default experiment (`core/regions.py:DEFAULT_EXPERIMENT_ID`)
for backward compatibility; nothing about today's pipeline behavior changes
unless a different `--experiment` is explicitly requested.

## bejis_2022 (compact entry)

- **role**: `mediterranean_transfer_wildfire`
- **region**: Bejís / Castellón / Valencian Community / Spain
- **predictor window**: `2022-06-15` -> `2022-08-14`
- **label window**: `2022-08-15` -> `2022-09-30` (fire start `2022-08-15`)
- **baseline years**: 2018–2021
- **status**: registry + initial AOI only — no gate, predictors, Step7,
  Step8, or transfer modeling has been run yet
- **note**: like Manavgat 2021, this is **not** a control region; the AOI
  must pass the same MCD64A1 burned-landcover gate before modeling. The
  registered `bejis_aoi` bounding box is an initial candidate, not a final
  burn perimeter, and should be refined after AOI preview and the gate
  (same iterative process documented for Manavgat in
  [`docs/aoi_refinement.md`](./aoi_refinement.md)).

```bash
python scripts/preview_experiment_aoi.py --experiment bejis_2022
python scripts/main.py --experiment bejis_2022 --dry-run
```

## Why Kozan 2023 is a control AOI

The supervisor's conclusion is that Kozan 2023 burned-area labels are
cropland/anız-burning dominated rather than natural-vegetation wildfire, so
Kozan 2023 is retained as a **negative/control** AOI rather than the primary
wildfire validation case. Manavgat/Antalya 2021 is the next **anchor
wildfire** AOI to validate the same methodology against.

## How to run

```bash
# Default (unchanged) behavior — same as before Step0 existed.
python scripts/main.py

# Explicit, equivalent to the default.
python scripts/main.py --experiment kozan_2023

# Preview Manavgat 2021's Step0 configuration without running anything.
python scripts/main.py --experiment manavgat_2021 --dry-run
```

### Current scope limitation (important)

In this first Step0 implementation, **only `kozan_2023` can actually be
executed end-to-end**. Step1–Step8E still read the legacy, hardcoded
constants in `core/config.py` (`REGION_NAME`, `PREDICTOR_START_DATE`, etc.),
which are wired to Kozan 2023's values. Running
`python scripts/main.py --experiment manavgat_2021` (without `--dry-run`)
fails fast with a clear error instead of silently running the Kozan pipeline
under a misleading experiment label. Full Step1-8E experiment-awareness is
left to a future, dedicated refactor.

`--dry-run` works for any registered experiment (including disabled ones, if
selected directly with the underlying `allow_disabled=True` API) and only
prints the Step0 banner:

```
[Step0] Active experiment: manavgat_2021
[Step0] Region: manavgat_aoi
[Step0] Role: anchor_wildfire
[Step0] Predictor window: 2021-06-01 -> 2021-07-27
[Step0] Label window: 2021-07-28 -> 2021-08-31
[Step0] Baseline years: 2017, 2018, 2019, 2020
[Step0] Output root: outputs/experiments/manavgat_2021
```

## Output namespacing

New, namespaced output roots are exposed for the next refactor phase:

```
outputs/experiments/<experiment_id>/<step_name>/
```

e.g. `outputs/experiments/kozan_2023/step8a/`,
`outputs/experiments/manavgat_2021/step8b/`, etc.

These are **not** used by Step1-Step8E yet. Legacy paths
(`outputs/step8a/`, `outputs/step8b/`, ...) keep working exactly as before;
nothing is deleted, moved, or silently redirected.

## Programmatic API (`core/regions.py`)

- `get_experiment(experiment_id) -> dict`
- `get_active_experiment(experiment_id=None, allow_disabled=False) -> dict`
  (defaults to `kozan_2023`)
- `list_experiments(include_disabled=False) -> dict`
- `get_region_for_experiment(experiment_id)` (resolves to an `ee.Geometry`,
  requires GEE auth like any other step)
- `get_experiment_output_root(experiment_id) -> Path`
- `get_step_output_dir(experiment_id, step_name, create=False) -> Path`

## Compatibility bridge (`core/config.py`)

New, additive names (do **not** replace or shadow the legacy constants):
`ACTIVE_EXPERIMENT_ID`, `ACTIVE_EXPERIMENT`, `EXPERIMENT_REGION_NAME`,
`EXPERIMENT_PREDICTOR_START_DATE`, `EXPERIMENT_PREDICTOR_END_DATE`,
`EXPERIMENT_LABEL_START_DATE`, `EXPERIMENT_LABEL_END_DATE`,
`EXPERIMENT_BASELINE_YEARS`, `EXPERIMENT_OUTPUT_ROOT`.

A fail-fast consistency check runs at import time: when
`ACTIVE_EXPERIMENT_ID == "kozan_2023"` (the default), the module asserts
these new values are byte-for-byte identical to the legacy `REGION_NAME` /
`PREDICTOR_START_DATE` / `PREDICTOR_END_DATE` / `LABEL_START_DATE` /
`LABEL_END_DATE` constants, so the two layers can never silently drift apart
for the default experiment.

## Validation

```bash
python -m py_compile core/regions.py
python -m py_compile core/config.py
python scripts/check_experiment_registry.py
python scripts/main.py --experiment kozan_2023 --help
python scripts/main.py --experiment kozan_2023 --dry-run
```

## Related: burned-landcover gate

Every new AOI/experiment must pass a burned-landcover diagnostic gate right
after Step6's label export, before Step7A/Step8A. See
[`docs/label_gate.md`](./label_gate.md) for details — in short: Kozan 2023 is
expected to land on `cropland_dominated_control` (its known negative/control
status), Manavgat 2021 is expected (but not yet verified) to land on
`wildfire_candidate_pass`.

## Related: AOI refinement & gate-only readiness

See [`docs/aoi_refinement.md`](./aoi_refinement.md) for the refined Manavgat
2021 AOI geometry (`core/regions.py:build_regions()`), the
`scripts/preview_experiment_aoi.py` metadata/AOI preview helper, and the
`scripts/run_label_gate_only.py` gate-only runner (currently `kozan_2023`
only — it never silently runs Kozan data under another experiment's label).

## Related: namespaced Manavgat gate-only run

See [`docs/manavgat_gate_only.md`](./manavgat_gate_only.md) for how
`scripts/run_label_gate_only.py --experiment manavgat_2021 --export-labels
--force` runs entirely under `outputs/experiments/manavgat_2021/` (gate
inputs, raw BurnDate, and gate outputs), how the gate-only 30 m reference
raster is prepared without running Step7's thermal predictors, and the
runtime safety check that stops any accidental write to Kozan's legacy
shared paths.

## Related: experiment-aware Manavgat predictor generation (Step1-Step5/5C)

`scripts/run_predictors_only.py` runs Step3 (Landsat LST/NDVI GEE
preparation) through Step5/Step5C (thermal anomaly + TVDI) for a given
experiment, entirely separate from Step7/Step8 (never run by this script).

```bash
python scripts/run_predictors_only.py --experiment kozan_2023 --dry-run
python scripts/run_predictors_only.py --experiment manavgat_2021 --dry-run
python scripts/run_predictors_only.py --experiment manavgat_2021 --export --force
python scripts/run_predictors_only.py --experiment manavgat_2021 --local-only --force
```

`kozan_2023` keeps its exact legacy behavior (`core/config.py` constants,
shared `data/`/`outputs/step5/`/`outputs/step5c/` paths). Every other
experiment (currently `manavgat_2021`) runs through
`core/experiment_context.py:build_experiment_context()`, which resolves the
AOI, predictor/baseline dates, and **all** input/output paths under
`outputs/experiments/<experiment_id>/` — verified at runtime by
`_assert_paths_are_safely_namespaced()`, which stops the run immediately if
any computed path would collide with Kozan's legacy shared directories.

For `manavgat_2021`, `current_period_days` is derived as a **plain date
difference** (`(predictor_end_date - predictor_start_date).days = 56`),
matching the exact convention `core/config.py`'s `CURRENT_PERIOD_DAYS`
already uses for Kozan (`60`, not `61`) — **not** an inclusive day count
(which would incorrectly be `57`). Both the exact dates and the derived
count are logged explicitly on every run.

`--export` for Manavgat exports current + baseline Landsat LST and NDVI
directly from Earth Engine to local, namespaced files (bypassing the
Drive-export/polling chain Step4/Step4B use for Kozan — the same
direct-to-local pattern already used by Step6/Step6A), then runs
Step5/Step5C. `--local-only` assumes those GeoTIFFs already exist and runs
only Step5/Step5C. Step5's and Step5C's own computation logic (QA masking,
NDVI physical-range validity, TVDI wet/dry-edge binning, reliability
filtering) is reused unchanged for both AOIs — no divergent implementation.

## Related: experiment-aware Step5B diagnostic report

`src/step5b_diagnostic_report.py` reads Step5/Step5C raster outputs and
writes a diagnostic summary (raster stats, seam-evidence, TVDI diagnostics)
— it never modifies raster values and its outputs are explicitly not "fire
risk" products.

```bash
python src/step5b_diagnostic_report.py                                    # legacy Kozan, unchanged
python src/step5b_diagnostic_report.py --experiment kozan_2023 --dry-run
python src/step5b_diagnostic_report.py --experiment manavgat_2021 --dry-run
python src/step5b_diagnostic_report.py --experiment manavgat_2021 --force
```

`kozan_2023` (or no `--experiment` at all) reads `outputs/step5/` /
`outputs/step5c/` and writes to `outputs/step5b_diagnostics/` — the exact
legacy paths, unchanged. `manavgat_2021` reads
`outputs/experiments/manavgat_2021/step5/` /
`outputs/experiments/manavgat_2021/step5c/` and writes to
`outputs/experiments/manavgat_2021/step5b/`, verified at runtime by the
same `_assert_paths_are_safely_namespaced()` pattern used elsewhere. The
generated `summary.md`/`diagnostic_stats.json` include the experiment
context (region, role, predictor/label windows, baseline years) and an
explicit warning whenever `tvdi_anomaly_zscore`'s
`low_tvdi_std_masked_ratio` exceeds `0.8` (heavily masked due to low
baseline TVDI std).

## Related: namespaced MODIS preparation for Step7

`scripts/prepare_modis_for_step7.py` exports the MODIS LST mean/std layers
Step7B/7D require, for a single experiment's **predictor window** (not a
multi-year baseline) — reusing `src/step2_modis_5year_mean.py`'s already
existing GEE aggregation logic and `run_predictors_only.py`'s direct-or-tiled
export fallback.

```bash
python scripts/prepare_modis_for_step7.py --experiment manavgat_2021 --dry-run
python scripts/prepare_modis_for_step7.py --experiment manavgat_2021 --export --force
```

Output (fully namespaced, never touches Kozan's legacy `data/modis/`):

```
outputs/experiments/manavgat_2021/data/modis/modis_lst_mean_celsius.tif
outputs/experiments/manavgat_2021/data/modis/modis_lst_std_celsius.tif
outputs/experiments/manavgat_2021/data/modis/modis_metadata.json
```

`run_step7_downscaling_only.py`'s own MODIS best-effort preparation now
delegates to this same function (no divergent implementation) — after
running this script successfully, `run_step7_downscaling_only.py
--experiment manavgat_2021 --dry-run` reports `MODIS: mean=[VAR] std=[VAR]`
and confirms the actual run will proceed instead of failing fast.

## Related: experiment-aware DEM/slope preparation for Step7

`scripts/prepare_dem_for_experiment.py` fixes a root cause of Step7B
producing zero training samples for Manavgat: the shared `data/dem/`
(elevation/slope) only ever covered Kozan's AOI and had **zero geographic
overlap** with Manavgat, so `elevation`/`slope` (both required core
features) silently zeroed out every sample. `core/experiment_context.py`'s
`ctx["dem_input_dir"]` is now **namespaced** for non-Kozan experiments
(`outputs/experiments/<experiment_id>/data/dem/`) instead of pointing at
the shared Kozan directory; Kozan itself is unaffected (`dem_input_dir`
still resolves to `data/dem/`).

```bash
python scripts/prepare_dem_for_experiment.py --experiment manavgat_2021 --dry-run
python scripts/prepare_dem_for_experiment.py --experiment manavgat_2021 --export --force
```

Reuses `src/step2b_dem.py`'s existing DEM source/slope logic (Copernicus
DEM GLO-30, fallback USGS SRTMGL1) and `run_predictors_only.py`'s
direct-or-tiled export fallback; exported rasters are then bilinear-aligned
to the experiment's own Step5 reference grid. Output:

```
outputs/experiments/manavgat_2021/data/dem/elevation.tif
outputs/experiments/manavgat_2021/data/dem/slope.tif
outputs/experiments/manavgat_2021/data/dem/dem_metadata.json
```

`run_step7_downscaling_only.py` now fails fast (before Step7B runs) if
namespaced DEM is missing for a non-Kozan experiment — *"Experiment-aware
DEM/slope missing. Run prepare_dem_for_experiment.py first."* — and its
dry-run reports `DEM required=True` / `elevation=[VAR]` / `slope=[VAR]`
accordingly, mirroring the existing MODIS-required reporting.

## Related: Step7D reads only Step7B's pre-aligned inputs (no silent resampling)

Step7B's `aligned_inputs/` now writes every feature raster (MODIS, NDVI,
elevation, slope, landcover, optional anomaly/TVDI) under **canonical
filenames matching the model's feature names exactly**
(`modis_lst_mean_celsius.tif`, `ndvi.tif`, `elevation.tif`, etc.) — even
when a feature already matched the reference grid and needed no
reprojection (previously such features were left at their original,
un-namespaced path, and reprojected ones were suffixed `_aligned.tif`,
which caused Step7D to look in the wrong place and fail with "Feature
raster(s) do not match reference grid").

For non-Kozan experiments, `src/step7d_predict_downscaled_lst.py` now
resolves **every** model feature exclusively from
`outputs/experiments/<experiment_id>/step7b/aligned_inputs/<feature>.tif`
— it never falls back to raw namespaced sources (`data/modis/`,
`dem_input_dir`, etc.) for inference. If an aligned file is missing, Step7D
fails immediately with: *"Aligned feature raster missing for Step7D:
&lt;feature&gt;. Re-run Step7B."* Kozan's legacy discovery
(`FEATURE_RASTER_CANDIDATES`) is unchanged, but now also opportunistically
prefers `outputs/step7b/aligned_inputs/<feature>.tif` when present.

Step7B's `downscaling_dataset_stats.json` now records `aligned_inputs_dir`
and `aligned_feature_paths`; Step7D's `downscaling_prediction_metadata.json`
now records `experiment_id`, `feature_list`, `feature_paths`,
`all_features_match_reference_grid=true`, and `no_silent_resampling=true`.

## Related: experiment-aware Step8A-E burned-area association modeling

`scripts/run_step8_modeling.py` runs Step8A (label-honest ~500 m
MCD64A1-cell dataset) through Step8E (compact final report) for a given
experiment, reusing Kozan's already-validated Step8A-D logic (block/tile
500 m-cell reconstruction, spatial-block CV, spatial-block bootstrap,
thermal feature ablation) completely unchanged — only input/output paths
become experiment-aware.

```bash
python scripts/run_step8_modeling.py --experiment manavgat_2021 --dry-run
python scripts/run_step8_modeling.py --experiment manavgat_2021 --force
python scripts/run_step8_modeling.py --experiment manavgat_2021 --force --allow-no-step7
```

`kozan_2023` keeps its exact legacy behavior (`outputs/step8a`..`step8e`).
`manavgat_2021` runs entirely under
`outputs/experiments/manavgat_2021/step8{a,b,c,d,e}/`, reading MCD64A1
labels from `validation/labels/mcd64a1_raw.tif` (never falls back to a
binary mask silently) and predictors from the experiment's namespaced
Step5/Step5C/Step7/DEM/landcover outputs — verified by the same
`_assert_paths_are_safely_namespaced()` pattern used elsewhere, plus a
runtime leakage guard that fails if any predictor name/path contains
`burndate`/`burned`/`mcd64a1`/`firms`/`post_fire`.

Labels are always ~500 m reconstructed MCD64A1 cells (`gate_level =
500m_reconstructed_mcd64a1_cell`) — 30 m pixels are never used as label
rows. Step7's fused LST is required by default; missing it fails fast
unless `--allow-no-step7` is passed (thermal feature set shrinks
accordingly). Step8E is a **new**, compact Manavgat-specific report
(`write_final_report()`, not a reuse of Kozan's multi-population
`step8e_final_report.py`) that pulls gate result, Step7 coverage gain,
Step8A cell counts, Step8B/C/D metrics, and the four required limitation
statements (500 m not 30 m; pre-fire association not real-time detection;
Step7 fused LST is predictor-window summary/fusion not daily operational
downscaling; burned labels used only for Step8 evaluation, never by Step7)
into a single JSON + Markdown report.

## Related: experiment-aware Step7A-E downscaling/fusion

`scripts/run_step7_downscaling_only.py` runs Step7A (tiling test) through
Step7E (observed+downscaled LST fusion) for a given experiment. It never
runs Step8 and never trains a burned-area/fire-risk model — Step7C trains a
**pure MODIS→Landsat LST downscaling model** only, with the same leakage
guard as Kozan (excludes `anomaly_zscore`, `current_tvdi`,
`tvdi_difference`, `modis_context_zscore` from training features).

```bash
python scripts/run_step7_downscaling_only.py --experiment kozan_2023 --dry-run
python scripts/run_step7_downscaling_only.py --experiment manavgat_2021 --dry-run
python scripts/run_step7_downscaling_only.py --experiment manavgat_2021 --force
python scripts/run_step7_downscaling_only.py --experiment manavgat_2021 \
    --from-step step7a --to-step step7e --force
```

`kozan_2023` keeps its exact legacy behavior (`outputs/step7a`..`step7e`,
`outputs/step5`, `outputs/step5c`, `data/...`). `manavgat_2021` runs
entirely under `outputs/experiments/manavgat_2021/step7{a,b,c,d,e}/`,
reading Step5/Step5C inputs from the same experiment's namespaced
directories — verified by the same `_assert_paths_are_safely_namespaced()`
pattern. Two inputs are deliberately **shared, read-only** for this patch
(Option B): DEM (`data/dem/elevation.tif`, `data/dem/slope.tif`) is the same
global asset both AOIs read from, and Manavgat's landcover reuses the
already-aligned raster Step6A produced during the gate-only stage
(`outputs/experiments/manavgat_2021/gate_inputs/...`) rather than a new
Step7-specific alignment. MODIS is attempted best-effort via a namespaced
export (reusing `run_predictors_only.py`'s tiled-export fallback) but is
never required — it's an optional core feature in Step7B/7D, so a failed or
skipped MODIS export logs a warning and the chain continues without it.