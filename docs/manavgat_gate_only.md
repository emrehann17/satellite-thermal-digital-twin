# Manavgat Gate-Only Workflow (Step0C)

This document explains how the Manavgat 2021 burned-landcover gate runs
**fully namespaced**, without touching Kozan's legacy shared files and
without running Step7/Step8 or any thermal predictor pipeline.

## Kozan stays legacy/default

`kozan_2023` is unaffected by any of this. `scripts/run_label_gate_only.py
--experiment kozan_2023 ...` still uses the exact same shared paths as
before (`outputs/validation/labels/...`, `outputs/step5/...`,
`data/landcover/...`) with no namespacing.

## Manavgat (and any future non-Kozan experiment) is fully namespaced

Everything Manavgat's gate-only run touches lives under
`outputs/experiments/manavgat_2021/`:

```
outputs/experiments/manavgat_2021/gate_inputs/reference_30m.tif
outputs/experiments/manavgat_2021/gate_inputs/landcover_esa_worldcover_v200.tif
outputs/experiments/manavgat_2021/gate_inputs/landcover_esa_worldcover_v200_aligned_to_reference.tif
outputs/experiments/manavgat_2021/gate_inputs/gate_inputs_metadata.json
outputs/experiments/manavgat_2021/validation/labels/mcd64a1_raw.tif
outputs/experiments/manavgat_2021/validation/labels/mcd64a1_burned.tif
outputs/experiments/manavgat_2021/validation/labels/burned_landcover_gate.json
outputs/experiments/manavgat_2021/validation/labels/burned_landcover_gate.md
outputs/experiments/manavgat_2021/validation/labels/burned_landcover_gate.csv
```

`scripts/run_label_gate_only.py` asserts this at runtime for every non-Kozan
experiment (`_assert_paths_are_safely_namespaced`): every computed path must
resolve under `outputs/experiments/<experiment_id>/`, and none may resolve
into the legacy `outputs/validation/labels/` directory. A violation raises
immediately — before any export or gate computation runs.

## The 30 m reference raster is gate-only, not a thermal predictor

Step6B's gate needs a 30 m grid (width/height/CRS/transform) to reconstruct
approximate native ~500 m MCD64A1 cells — it never reads the grid's pixel
*values*. Running full Step3/Step5 just to get that grid would mean running
the entire thermal preprocessing pipeline merely to check landcover
composition, which defeats the purpose of a lightweight pre-modeling gate.

`src/step6a_prepare_gate_inputs.py` instead exports a **constant-valued**
30 m raster (`ee.Image.constant(1)`) directly from Earth Engine over the
experiment's AOI — fast, cheap, and carries no scientific/LST meaning
whatsoever. It then exports ESA WorldCover v200 for the same AOI and aligns
it to that reference grid with **nearest-neighbor** resampling (categorical
data is never resampled bilinear) — reusing Step8A's own
`prepare_aligned_landcover()` directly, so there is exactly one alignment
implementation, not two.

`gate_inputs_metadata.json` records the CRS, scale, AOI bounds, label
window, and an explicit note that this is a gate-only reference grid, not a
thermal predictor.

## Step6B still aggregates to reconstructed 500 m MCD64A1 cells

Nothing about the gate's aggregation logic changes for Manavgat.
`gate_level = "500m_reconstructed_mcd64a1_cell"` — the same block size /
tiling logic Step8A uses — is unchanged; only the *paths* it reads from and
writes to differ (see `--label-path` / `--reference-path` /
`--landcover-path` / `--output-dir` CLI args on
`src/step6b_burned_landcover_gate.py`).

## Running it

```bash
# See exactly what would happen, with every planned path, no execution:
python scripts/run_label_gate_only.py --experiment manavgat_2021 --dry-run

# Actually run it (requires a working, authenticated GEE environment):
python scripts/run_label_gate_only.py --experiment manavgat_2021 --export-labels --force
```

`--export-labels` triggers, in order:
1. Step6A gate-input preparation (reference grid + aligned landcover) if
   missing, or unconditionally if `--force` is also given.
2. Step6's canonical raw MCD64A1 BurnDate export, scoped to Manavgat's AOI
   and label window (`2021-07-28` → `2021-08-31`).
3. Step6B burned-landcover gate, with all four paths passed explicitly.

`--skip-export` (the default) skips step 2 and reuses whatever
`mcd64a1_raw.tif` already exists under Manavgat's namespaced label
directory — useful for re-running just the gate after inspecting/adjusting
thresholds, without spending GEE quota on a fresh export.

## Expected outcome

- **Expected: `wildfire_candidate_pass`** — but this must be verified by
  actually running the gate; it is not assumed.
- If the result is **`cropland_dominated_control`**: the AOI
  (`core/regions.py:manavgat_aoi_refined_bbox`) still includes too much of
  the coastal agricultural belt — tighten it (see `docs/aoi_refinement.md`).
- If the result is **`insufficient_burned_positives`**: check that the AOI
  bounds actually cover the 2021 fire extent and that the label window
  (`2021-07-28` → `2021-08-31`) matches the real fire dates.

## What this does *not* do

No Step7 (thermal downscaling/fusion), no Step8 (modeling), no full
pipeline run, no writes to any Kozan legacy path. This is exclusively the
minimum chain needed to answer one question: is Manavgat 2021 a genuine
wildfire candidate?