# Step6 Label Export Cleanup & Burned-Landcover Gate

This document explains two related additions to Step6:

1. **Label export cleanup** — Step6 now owns the canonical, honest export of
   the MCD64A1 BurnDate label.
2. **Burned-landcover gate (Step6B)** — a diagnostic that classifies an
   AOI/experiment as a wildfire candidate, a cropland/anız control, or
   insufficient/mixed, based on the landcover composition of its burned
   MCD64A1 cells.

## Why raw BurnDate DOY is required

Step8A needs actual MCD64A1 `BurnDate` day-of-year values (1..366) so it can
place each burned ~500 m cell into a monthly lead-time stratum (Aug / Sep /
Oct) and can tell "burned inside the label window" apart from "burned, but
outside the label window." A binary burned/unburned flag (`BurnDate > 0`)
throws that information away — every burned cell collapses to the same value
and cannot be dated at all.

## Why a binary `mcd64a1_raw.tif` was dangerous

Historically, the dedicated export step
(`scripts/export_mcd64a1_raw_burndate.py`) already wrote genuine BurnDate DOY
values to `outputs/validation/labels/mcd64a1_raw.tif` — but it was a
**separate script**, invoked as its own pipeline stage, with no structural
link to Step6. Nothing prevented it from being skipped, run out of order, or
simply forgotten if someone ran Step6 (or the pipeline) manually or partially.
If that ever happened, Step8A's fallback discovery logic
(`resolve_label_raster()`) could, in the worst case, fall back to a binary
mask misread as `BurnDate` — every burned pixel becomes `1`, which decodes to
DOY 1 (January 1st), outside any realistic label window. Step8A's own
fail-fast check (`inspect_label_raster()`) catches this today, but "catches
it after the fact, if the missing step is ever re-run" is a fragile
guarantee.

**Fix:** Step6 now owns this export directly
(`export_raw_mcd64a1_labels()` in `src/step6_validate_fire_relation.py`).
`scripts/export_mcd64a1_raw_burndate.py` is now a thin CLI wrapper around it
— there is exactly one implementation, not two divergent ones. Step6's own
internal association-test download (a genuinely binary GEE download, used
only for its own ROC/AUC computation) was also renamed internally so its
filename can never be mistaken for the canonical raw BurnDate file by
Step8A's `*mcd64*raw*.tif` fallback search.

Canonical output paths (unchanged from before):

```
outputs/validation/labels/mcd64a1_raw.tif      # raw BurnDate DOY (1..366)
outputs/validation/labels/mcd64a1_burned.tif   # binary mask (BurnDate > 0)
```

## What the burned-landcover gate checks

Supervisor feedback: Kozan 2023's burned MCD64A1 cells are cropland/anız-
burning dominated, not natural-vegetation wildfire. The Step8 methodology
itself is correct — but before modeling a new AOI, we need to know up front
whether its burned cells are actually wildfire-like.

`src/step6b_burned_landcover_gate.py` answers exactly this, at the same
`500m_reconstructed_mcd64a1_cell` level Step8A uses (same block size, same
tiling utility, same ESA WorldCover class mapping — reused directly from
`src/step8a_prepare_500m_modeling_dataset.py`, not reimplemented). It does
**not** read any continuous predictor (NDVI/DEM/thermal), so it can run
immediately after Step6's label export, before Step7A-7E and Step8A.

For each reconstructed ~500 m cell, the gate determines:
- burned (1) or unburned (0), from the raw BurnDate raster;
- the **dominant** landcover class within that cell (tree_cover, shrubland,
  grassland, cropland, built_up, water, bare_sparse_vegetation, ...).

### Decision rules

```
burned_count < STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES (30)
    -> "insufficient_burned_positives"

burned_tree_shrub_grass_count / burned_count >= NATURAL_THRESHOLD (0.50)
    -> "wildfire_candidate_pass"

burned_cropland_dominant_count / burned_count >= CROPLAND_THRESHOLD (0.50)
    -> "cropland_dominated_control"

otherwise
    -> "mixed_or_uncertain"
```

Thresholds live in `core/config.py`:
`STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES`,
`STEP6_BURNED_LANDCOVER_GATE_NATURAL_THRESHOLD`,
`STEP6_BURNED_LANDCOVER_GATE_CROPLAND_THRESHOLD`,
`STEP6_BURNED_LANDCOVER_GATE_LEVEL`.

### This is diagnostic only

A `cropland_dominated_control` result does **not** stop the pipeline — Kozan
2023 is expected to land there and stays a valid negative/control AOI. The
gate only raises (fails the run) if:
- the raw BurnDate raster is missing, or a binary mask was found instead of
  genuine DOY values (`inspect_label_raster()` fail-fast check, reused from
  Step8A);
- the reference 30 m grid or landcover raster cannot be resolved;
- the landcover class mapping cannot be resolved.

### Expected outcomes

- **Kozan 2023**: `cropland_dominated_control` (matches Step8A's own prior
  finding: 542 burned cells, 533 cropland-dominant, 9 tree+shrub+grass, 1
  tree+shrub — i.e. the vast majority of burned cells are cropland).
- **Manavgat 2021**: expected `wildfire_candidate_pass`, but this must be
  **verified by actually running the gate** once Manavgat's Step1-Step6
  outputs exist — it is not assumed or hardcoded.

### Output files

```
outputs/validation/labels/burned_landcover_gate.json
outputs/validation/labels/burned_landcover_gate.md
outputs/validation/labels/burned_landcover_gate.csv
```

## Pipeline position

`scripts/main.py` now runs, in order:

```
STEP 6                                    (error-tolerant, unchanged)
LABEL EXPORT CLEANUP (raw MCD64A1 BurnDate)   (required, NOT error-tolerant)
BURNED-LANDCOVER GATE                     (diagnostic; fails only on bad/missing inputs)
STEP 7A ...
STEP 8A ...
```

Label export cleanup and the gate are **not** executed in `--dry-run` mode
(dry-run only prints the Step0 experiment banner and exits before `main()`
runs at all).

## Known scope limitation

These paths (`outputs/validation/labels/...`) are **not** experiment/
namespace-aware yet — same limitation as the rest of Step1-Step8E (see
`docs/experiments.md`). Running the gate for Manavgat 2021 will read/write
the same Kozan-shaped paths unless a reference 30 m grid and landcover raster
for Manavgat exist under those same legacy paths. Full experiment-aware
output migration is left to a future refactor.
