# Source-scene provenance and seam localization

The experiment pipeline includes two read-only QA stages:

```text
gate -> predictors -> scene-provenance -> step7 -> seam-audit -> seam-localization -> step8
```

Both stages are experiment-aware and AOI-agnostic. Provider selection follows
the experiment context layout; product resolution follows producer metadata,
exact context paths, and the shared Seam Audit V2 registry. Neither stage
changes a raster, trains a model, reruns Step8+, or submits an Earth Engine task.

## Configuration

The global defaults can be overridden in an experiment registry entry without
adding experiment-name branches:

```python
"source_scene_provenance": {
    "enabled": True,
    "mode": "metadata_only",
    "collection_roles": [
        "current_lst", "current_ndvi", "baseline_lst", "baseline_ndvi",
    ],
    "pixel_provenance_products": [
        "observation_count", "scene_count", "dominant_path_row",
        "dominant_scene_id", "dominant_scene_fraction",
        "median_acquisition_date", "acquisition_date_spread_days",
        "path_row_support_count",
    ],
},
"seam_localization": {
    "enabled": True,
    "boundary_sources": [
        "source_scene", "path_row", "observation_support", "export_tile",
    ],
    "artifact_families": [
        "current_lst", "current_ndvi", "baseline_lst_yearly",
        "baseline_ndvi_yearly", "baseline_lst_mean", "baseline_lst_std",
        "anomaly_zscore", "current_tvdi", "tvdi_difference",
        "downscaled_lst", "fused_lst",
    ],
    "audit_scales": ["native", "modeling_500m"],
    "local_control_offsets": [5, 10, 20],
    "minimum_valid_pairs": 100,
    "random_seed": 42,
},
```

Thresholds and conjunction/corroboration rules come from Seam Audit V2 and are
initial QA heuristics, not significance tests or causal attribution.

## Source-scene provenance

The versioned namespace is:

```text
outputs/experiments/<experiment_id>/qa/source_scene_provenance/v1/
```

Legacy experiments use the same QA namespace while their source artifacts are
resolved through the legacy layout provider. Metadata discovery is explicit;
there is no recursive glob fallback.

Outputs:

- `scene_manifest.parquet/json`: normalized scene metadata and deterministic,
  collision-free integer scene lookup.
- `scene_footprints.geojson`: deduplicated real source footprints.
- `scene_boundaries.geojson`: verified shared real-footprint edge segments,
  with stable boundary/lineage/geometry IDs and left/right support metadata.
- `artifact_scene_lineage.parquet`: scene-role to artifact relations.
- `artifact_lineage.json` and node/edge Parquets: producer-ordered artifact
  graph, transforms, aggregation, resampling, masking, semantic identity, grid
  signature, and missing-lineage state.
- `provenance_summary.json/md`: completeness and composite semantics.
- `pixel_provenance_export_plan.json`: explicit plan only; no task submission.
- `manifest.json`: output hashes and sizes.

For median, mean, and weighted composites, `selected_scene_id` is forbidden:
the reducer value may not equal any source pixel. `dominant_scene_id`, when
planned, means the scene with the largest valid-observation contribution count;
it is support metadata, not a selected value source. Selection IDs are valid
only for declared selecting composites such as single-scene, mosaic,
quality-mosaic, latest-valid, or earliest-valid.

Metadata-only mode consumes existing local scene lists. If IDs/footprints or
pixel support are unavailable, it writes an honest
`insufficient_boundary_metadata` or
`metadata_available_boundary_incomplete` status. Pixel provenance mode also
remains plan-only in this stage.

## Earliest-stage localization

The versioned namespace is:

```text
outputs/experiments/<experiment_id>/qa/seam_localization/v1/
```

The engine reprojects each LineString into each artifact CRS, samples matched
pixel pairs on both sides of horizontal, vertical, or oblique boundaries, and
builds deterministic parallel controls that avoid all known boundaries.
Canonical 500 m pairs use the georeferenced Step8A grid, never index division.

Outputs:

- `localization_summary.json/md`: overall completeness, earliest exact/bounded
  stage, upstream risk, 500 m propagation, blocker, and rerun recommendation.
- `artifact_boundary_metrics.parquet`: native and 500 m jump/control metrics.
- `boundary_stage_trace.parquet`: stable boundary identity through stages.
- `earliest_stage_candidates.parquet`: exact, bounded, and first-available
  detections.
- `visualization_checks.parquet`: fixed-physical and AOI-global robust stretch
  metadata; per-tile normalization is always false.
- `seam_profiles.parquet`: sampled pair values and signed jumps.
- `seam_hotspots.geojson`: localized candidate geometries.
- `matched_controls.parquet`, `artifact_resolution.parquet`, and
  `manifest.json`: audit support and reproducibility records.

If prior artifacts are missing, the result is
`bounded_but_not_exact`. If the first available legacy artifact already
contains the seam, it is
`present_at_first_available_artifact` and
`root_cause_upstream_of_available_artifacts=true`; this does not attribute
the cause to a source scene.

A scientific blocker requires verified source provenance, corroborated exact
detection, persistence on the same boundary into canonical 500 m, and an
actually used Step8 feature. Manual diagnostic boundaries never satisfy that
contract; they can set `potential_modeling_risk=true` only.

## Commands

```bash
venv/bin/python scripts/main.py experiment \
  --experiment mugla_2021 \
  --from-stage scene-provenance --to-stage scene-provenance \
  --predictor-mode local-only --dry-run

venv/bin/python scripts/main.py experiment \
  --experiment mugla_2021 \
  --from-stage seam-localization --to-stage seam-localization \
  --predictor-mode local-only --dry-run

venv/bin/python scripts/run_seam_localization.py \
  --experiment mugla_2021 \
  --manual-boundary seam_line.geojson --dry-run

venv/bin/python scripts/run_seam_localization.py \
  --experiment kozan_2023 \
  --manual-line '35.5,37.1;35.7,37.4' \
  --manual-crs EPSG:4326 --dry-run

venv/bin/python scripts/run_source_scene_provenance.py \
  --experiment mugla_2021 --mode pixel_provenance --export-plan-only
```

