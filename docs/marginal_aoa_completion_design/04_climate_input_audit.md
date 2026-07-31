# 04. Climate Input Audit

Machine-readable companion: `candidate_climate_inputs.csv`.

**Verdict up front: the repository contains no climate-normal input for these
four AOIs. A new export is required — and it has since been AUTHORISED
(decision C-9).** This document records what was searched, what was found, and
why each candidate fails. The audit facts below are unchanged; only the
downstream status moved from "blocked" to "authorised".

The accepted export contract is in `05_climatic_distance_design.md`:
TerraClimate, 1991–2020, warm season months 6–9, **four** variables, on
TerraClimate's native valid-land support.

---

## 1. Search performed

```
rg -rn "ERA5|WORLDCLIM|WorldClim|TERRACLIMATE|TerraClimate|CHIRPS|CHELSA|
        IDAHO_EPSCOR|precipitation|aridity|koppen|Koppen|climate_zone"
    --glob '!venv/**' --glob '!old_codes/**' --glob '!docs/**' --glob '!*.log'
    .
→ no matches
```

Directory-level inspection of every per-experiment output tree:

```
outputs/experiments/<exp>/
    data/{_tiles, current_period, dem, landsat_timeseries, modis,
          ndvi_current_period, ndvi_timeseries}
    gate_inputs/  step0/  step5/  step5b/  step5c/
    step7a/ step7b/ step7c/ step7d/ step7e/
    step8a/ step8b/ step8c/ step8d/ step8e/
    validation/  robustness/  qa/
    predictor_export_metadata.json
```

Every raster and metadata file in these trees was accounted for. The complete set
of Earth Engine collections the project has ever exported is:

| Collection | Constant | Role |
|---|---|---|
| `MODIS/061/MOD11A1` | `MODIS_COLLECTION`, `core/config.py:9` | Daily 1 km LST, aggregated over the **event predictor window** |
| `LANDSAT/LC08/C02/T1_L2` | `LANDSAT_COLLECTION`, `core/config.py:10` | 30 m LST and NDVI, current period and baseline years |
| `COPERNICUS/DEM/GLO30` | `DEM_COLLECTION`, `core/config.py:61` | Static elevation / slope |
| `USGS/SRTMGL1_003` | `DEM_FALLBACK_DATASET`, `core/config.py:63` | DEM fallback |
| `ESA/WorldCover/v200` | `WORLDCOVER_COLLECTION`, `src/step6a_prepare_gate_inputs.py:91` | Static 10 m landcover |
| `MODIS/…/MCD64A1` | (label path) | Burned-area labels — forbidden here |

**No atmospheric reanalysis, no gridded climatology, no precipitation product, no
humidity product, no aridity or water-balance product, and no climate-zone
classification has ever been exported by this project.**

---

## 2. Candidates found, and why each fails

### 2.1 `data/modis/modis_lst_mean_celsius.tif` — **fails, and says so itself**

Present for all four AOIs, with `modis_lst_std_celsius.tif`.

The artifact's own metadata (`data/modis/modis_metadata.json`) states, verbatim
and identically in all four AOIs:

```json
"aggregation_window": "experiment predictor window (single-season mean/std over
                       daily MODIS scenes within the window),
                       NOT a multi-year baseline"
```

| Experiment | Window |
|---|---|
| `manavgat_2021` | 2021-06-01 → 2021-07-27 |
| `bejis_2022` | 2022-06-15 → 2022-08-14 |
| `mugla_2021` | 2021-06-01 → 2021-07-28 |
| `evia_2021_extended` | 2021-06-05 → 2021-08-02 |

This is a **single fire season**, in a **different year** for Bejís, over a
**different window** for each AOI. It is an event-period predictor. Relabelling it
as climate would be exactly the substitution the task forbids.

### 2.2 `step5/baseline_lst_mean_celsius.tif` — the closest artifact, still fails

This is the **only genuinely multi-year product in the repository**, and it is
worth stating precisely what it is before rejecting it.

Construction (`src/step3_landsat_lst.py:258` `get_landsat_baseline_window_median_collection`,
then `src/step5_preprocess_timeseries.py:844-855`): for each baseline year, a
QA-masked Landsat LST **median over a window of `window_days` ending on the
anniversary of the current-period end date**; then `np.nanmean` / `np.nanstd`
across the four yearly composites.

| Experiment | Baseline years | Anniversary date | Window (days) | Baseline mean (°C) | Baseline SD (°C) | Anomaly coverage |
|---|---|---|---:|---:|---:|---:|
| `manavgat_2021` | 2017–2020 | 07-27 | 56 | 35.861 | 1.579 | 69.6% |
| `bejis_2022` | **2018–2021** | **08-14** | 60 | 38.094 | 2.585 | 77.3% |
| `mugla_2021` | 2017–2020 | 07-28 | 57 | 33.580 | 1.289 | 54.2% |
| `evia_2021_extended` | 2017–2020 | 08-02 | 58 | 30.396 | 1.127 | 47.8% |

**Why it is not a climate normal, in order of severity:**

1. **Four years.** A WMO climate normal is 30 years. Four summer composites cannot
   separate a climatology from interannual variability; the 2017–2020 mean for a
   Mediterranean AOI is dominated by whichever of those four summers was extreme.
2. **The reference period is not shared.** Bejís uses **2018–2021**; the other
   three use 2017–2020. A "climatic distance" between Bejís and Manavgat computed
   from these would be partly a distance between two different four-year windows.
3. **The seasonal window is not shared.** Anniversary dates differ (07-27, 08-14,
   07-28, 08-02) and window widths differ (56, 60, 57, 58 days). Each AOI's
   baseline is symmetric to *its own event*, by design — which is correct for an
   anomaly predictor and disqualifying for a cross-AOI climate comparison.
4. **Coverage is cloud-driven and very uneven**: 47.8% for Evia against 77.3% for
   Bejís. The AOI mean is taken over a different, weather-selected subset of each
   AOI.
5. **It is a single variable** — daytime clear-sky land surface temperature.
   Climate distance in a fire context needs at minimum a moisture axis;
   temperature alone is not a climate vector.
6. **LST is not air temperature.** It is a clear-sky, sensor-specific, surface
   radiometric quantity that is strongly confounded with land cover and
   vegetation state — the very predictors the AoA is already measuring. A
   "climatic distance" built from it would be substantially collinear with the
   predictor-space dissimilarity it is supposed to complement.

Point 6 is the decisive one. Even if points 1–5 were repaired, a distance built
from baseline LST would not be an *independent* climatic axis.

### 2.3 `step5c/baseline_tvdi_mean.tif` — fails for the same reasons, plus one

Same four baseline years, same AOI-specific anniversary windows. TVDI is derived
**from** the baseline LST and baseline NDVI of the same AOI
(`step5c_metadata.json` `inputs` names `baseline_lst_mean_celsius.tif` and the
baseline NDVI directory). It inherits every defect in §2.2 and adds full
dependence on the AOI-internal NDVI–LST feature space, which is precisely what
`current_tvdi_mean` and `tvdi_difference_mean` already contribute to the
predictor-space dissimilarity.

### 2.4 Baseline NDVI GeoTIFFs — fail

Four per AOI (`data/ndvi_timeseries/ndvi_baseline_<year>-<mm-dd>.tif`), same
anniversary structure, same year mismatch for Bejís. Vegetation state, not
climate. Also already represented in the predictor set via `ndvi_mean`.

### 2.5 Event-period predictors — explicitly forbidden

`current_lst_mean`, `current_tvdi_mean`, `downscaled_lst_mean`,
`fused_lst_mean` and `lst_anomaly_mean` are all tied to the single fire season
and the AOI-specific predictor window. They are **model predictors already inside
the weighted dissimilarity**. Reusing any of them as a "climatic distance" would
double-count them and would be the exact mislabelling the task prohibits.

`lst_anomaly_mean` deserves a specific note because it is superficially tempting:
it is `(current_median − baseline_mean) / baseline_std`, so it *contains* a
multi-year term. But it is a standardised **anomaly of the event season**, not a
climatology. Its AOI mean measures how unusual that particular summer was, which
is a legitimate and interesting quantity — and it is **not** a climatic distance
between the AOIs.

### 2.6 DEM and landcover — static, not climate

`COPERNICUS/DEM/GLO30` and `ESA/WorldCover/v200` are static covariates already in
the predictor contract (`elevation_mean`, `slope_mean`, `landcover_dominant`).
Elevation is a strong climate *proxy* in Mediterranean terrain, but it is already
the top-weighted predictor in three of four AOIs. Recycling it as "climate" would
make the two components near-duplicates.

### 2.7 Domain-classifier audit — not a distance, and incomplete

`outputs/diagnostics/domain_classifier_audit/` reports source-vs-target
classifier AUC for **3 of the 6** unordered pairs (0.99987, 0.99979, 0.98224) and
the artifact itself states it does not claim to be a geographic distance. **Evia
is absent entirely.** It is also a *learned* separability measure over the same
predictors, not an independent climatic axis. Not usable.

---

## 3. Summary table

| Candidate | Multi-year? | Shared reference period? | Shared season window? | Independent of predictors? | Climate variable? | Verdict |
|---|---|---|---|---|---|---|
| `modis_lst_mean` | **No** (single season) | n/a | No | No | No | Reject |
| `baseline_lst_mean` (step5) | 4 years | **No** (Bejís differs) | **No** | No | LST only | Reject |
| `baseline_tvdi_mean` (step5c) | 4 years | **No** | **No** | No | No | Reject |
| baseline NDVI | 4 years | **No** | **No** | No | No | Reject |
| `lst_anomaly_mean` | partly | No | No | No | anomaly, not normal | Reject |
| DEM / slope | static | n/a | n/a | **No** — already a predictor | proxy only | Reject |
| landcover | static | n/a | n/a | **No** — already a predictor | No | Reject |
| domain-classifier AUC | n/a | n/a | n/a | No | No | Reject; Evia missing |

**No row survives.** A genuine climatic distance requires new data.

---

## 4. What a genuine climate input must satisfy

Derived from the failures above; these become the acceptance criteria for the
proposed export in `05_climatic_distance_design.md`:

1. **A single reference period shared by all four AOIs**, independent of each
   AOI's event year. Bejís burned in 2022 and the others in 2021; the climate
   vector must not encode that.
2. **≥ 30 years**, so it is a normal rather than a four-sample mean.
3. **A shared seasonal definition** — the same months for every AOI, not each
   AOI's own anniversary window.
4. **At least one moisture axis** in addition to temperature.
5. **Not derived from the Landsat/MODIS surface products already in the predictor
   set**, so the climatic distance carries information the weighted dissimilarity
   does not.
6. **Complete coverage over all four AOIs**, with no cloud-driven selection
   effect.
7. **A pinned, versioned collection ID**, so the artifact's provenance is
   reproducible.

---

## 5. Consequence for the artifact

Until the export in doc 05 is authorised and run, every climate field in the
completion artifact must be written as **explicitly absent**, never as a
substituted proxy:

```json
{
  "climate_distance": null,
  "climate_data_completeness": null,
  "climate_component_contributions": null,
  "climate_export_authorised": true,
  "climate_status": "authorised_pending_export"
}
```

The descriptive fields (`climate_distance_metric`, `climate_feature_count = 4`,
`climate_features`, `climate_reference_period`, `climate_season_months`,
`climate_scaling_contract`, `climate_source_version`, `climate_land_mask`) are
**known in advance** and may be populated from the preregistration before the
export runs; only the measured quantities stay null.

A validator check (doc 08, check 16) asserts that no field whose name matches
`climate_*` is ever populated from a path under `outputs/experiments/*/step5/`,
`step5c/`, `data/modis/` or `data/ndvi_timeseries/`. This is the enforcement of
"no event-period predictor is silently used as a climate normal", and it is
implemented as a path-provenance assertion rather than a naming convention, so it
cannot be bypassed by renaming a column.
