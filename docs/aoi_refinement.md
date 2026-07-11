# AOI Refinement & Gate-Only Readiness (Step0B)

This document covers two things: the refined Manavgat 2021 AOI geometry, and
the gate-only workflow used to validate a new AOI/experiment before modeling.

## Manavgat 2021: anchor wildfire AOI

`manavgat_2021` is registered as the **anchor natural-vegetation wildfire**
experiment (`role = anchor_wildfire`), in contrast with Kozan 2023's
`negative_control` role. Kozan remains unchanged and stays cropland/anız
control.

### Current Manavgat AOI is a refined *working* AOI, not a final fire scar

`core/regions.py:build_regions()` now defines three Manavgat geometries:

- **`manavgat_aoi_refined_bbox`** — the new default. A bounding box biased
  north/north-east of Manavgat town, deliberately **not** a symmetric buffer
  around the coast:
  - South edge (`lat_min=36.72`) sits just north of the coastline, keeping
    most of the dense coastal greenhouse/citrus agricultural belt out of the
    AOI (a thin strip is kept for unburned-neighbor controls).
  - North edge (`lat_max=37.35`) extends into the forested Taurus foothills
    (toward Akseki / Gündoğmuş), where the 2021 fire spread.
  - East/west edges (`lon 31.05–31.85`) cover the Manavgat valley and
    immediate neighboring forest without reaching into Antalya's urban area.
  - **This is not derived from an actual MCD64A1/FIRMS/fire-scar dataset.**
    It is a manually-drawn working AOI and must be checked against those
    outputs (via the burned-landcover gate — see below) before being trusted.
- **`manavgat_aoi_wide_buffer`** — the original symmetric point+50 km buffer,
  kept as a fallback/debug geometry. No experiment uses it by default.
- **`manavgat_aoi`** — the key `EXPERIMENTS["manavgat_2021"]["region_key"]`
  actually resolves to. Currently aliases `manavgat_aoi_refined_bbox`.

`kozan_aoi`, `dogu_akdeniz`, and the disabled Valencia/Zamora placeholders are
all unchanged.

### Before modeling Manavgat, we must run

1. Raw MCD64A1 BurnDate export (Step6's `export_raw_mcd64a1_labels()`).
2. The Step6B burned-landcover gate (`src/step6b_burned_landcover_gate.py`).

### Expected / possible gate outcomes for Manavgat

- **Expected: `wildfire_candidate_pass`** — but this is a prediction, not a
  guarantee, and **must be verified by actually running the gate** once
  Manavgat's Step1-Step6 outputs exist.
- If the gate returns **`cropland_dominated_control`**: the AOI still
  includes too much of the coastal agricultural belt — the bbox in
  `core/regions.py` should be tightened further (e.g. raise `lat_min`, or
  clip the southwest corner).
- If the gate returns **`insufficient_burned_positives`**: the AOI may be too
  small, mis-positioned, or the label window doesn't match the actual 2021
  fire dates — check `EXPERIMENTS["manavgat_2021"]` predictor/label windows
  and the AOI bounds together.

Either outcome means adjusting the geometry in **one place**
(`core/regions.py:build_regions()`, the `manavgat_aoi_refined_bbox` block) —
nothing else needs to change to iterate on the AOI.

## AOI preview helper

`scripts/preview_experiment_aoi.py` prints an experiment's Step0 metadata
(predictor/label windows, baseline years, output root) and, if GEE is
initialized, the AOI geometry type and approximate bounds — without running
any export, pipeline, or model. If GEE cannot be initialized (no
credentials, no network), it fails gracefully with a clear message and still
prints the Step0 metadata.

```bash
python scripts/preview_experiment_aoi.py --experiment kozan_2023
python scripts/preview_experiment_aoi.py --experiment manavgat_2021
```

When GEE is available, it also writes a small GeoJSON preview to:

```
outputs/experiments/<experiment_id>/step0/aoi_preview.geojson
```

## Gate-only runner

`scripts/run_label_gate_only.py` runs just the minimum sequence needed
before modeling: an optional raw BurnDate export, then the Step6B gate. It
does **not** run Step1-Step5, Step7, or Step8.

**Current scope**: `kozan_2023` uses the legacy, non-namespaced code path
unchanged. Every other experiment (currently `manavgat_2021`) runs through a
fully namespaced gate-only path — see
[`docs/manavgat_gate_only.md`](./manavgat_gate_only.md) for details on how
that works and what safety checks prevent it from ever touching Kozan's
legacy shared files.

```bash
python scripts/run_label_gate_only.py --experiment kozan_2023 --skip-export --force
python scripts/run_label_gate_only.py --experiment kozan_2023 --export-labels --force
python scripts/run_label_gate_only.py --experiment manavgat_2021 --dry-run
```

Flags: `--dry-run` (print Step0 summary only, run nothing), `--skip-export`
(default; gate uses whatever `mcd64a1_raw.tif` already exists),
`--export-labels` (run the GEE raw BurnDate export first — mutually
exclusive with `--skip-export`), `--force` (overwrite existing gate
outputs).