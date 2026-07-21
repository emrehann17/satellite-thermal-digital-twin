# Seam Audit V2 (read-only QA)

`seam-audit` is the experiment-aware QA stage between Step7 and Step8. The
pipeline stage now runs V2 and writes only under:

```text
outputs/experiments/<output_namespace>/qa/seam_audit/v2/
```

V1 is retained unchanged in `qa/seam_audit/v1/`; its runner
`scripts/run_seam_audit.py`, implementation, schema, and historical outputs are
not migrated or overwritten. V2 also hashes the V1 directory before and after
each real run and aborts if any V1 file changes.

V2 never modifies, smooths, blends, resamples, or regenerates a source raster.
Audit-only canonical-cell means are held in memory and written only as QA
metrics.

## V2 architecture

- The registry in `core/seam_audit_v2_config.py` declares each product's exact
  `boundary_lineage`; there is no broad default boundary list.
- Artifacts resolve producer metadata first, then an ExperimentContext exact
  path, then a canonical filename confirmed by producer code. Missing paths
  stay missing; broad recursive globs and semantically different filename
  fallbacks are forbidden. Baseline NDVI/LST years expand from the
  experiment's `baseline_years`.
- Every product records `semantic_identity`, `artifact_kind`,
  `native_artifact_required`, `native_artifact_path`, `modeling_feature`,
  `derived_from`, `explicit_alias_group`, `resolution_status`, and
  `resolution_method`. Different semantic identities cannot resolve to one
  physical file unless both declare the same non-null alias group.
- Export boundaries come from actual `_tiles/<family>/*.tif` CRS, transform,
  bounds, and grid alignment. Overlap/gap is recorded.
- Processing boundaries come only from Step7D/E inference metadata and the
  same `core.utils.tiling.make_tile_grid` helper used by inference. Step7A's
  tiling test is never a fallback.
- Native geometry is mapped to the georeferenced Step8A canonical 500 m raster.
  Native indices are never divided by 17.
- Native and modeling evidence is joined only by the same stable
  `boundary_id`, `lineage_id`, and `geometry_hash`.
- Continuous seams use deterministic local parallel controls. Both an absolute
  jump and a local-control ratio must cross their heuristic thresholds.
- Nodata is a separate coverage audit. Its fraction is internal valid↔nodata
  adjacencies divided by all examined internal adjacencies; the outer raster
  perimeter is excluded. Its continuous status is `not_applicable`.
- Product WARN/FAIL requires configurable segment-count and boundary-fraction
  corroboration. A scientific blocker and rerun recommendation require a
  verified seam that propagates on the same boundary into the real modeling
  predictor.

The default thresholds are initial QA heuristics, not statistical significance
tests or causal attribution. Experiments may override them with a
`seam_audit_v2` mapping in `core.regions.EXPERIMENTS`.

## LST anomaly identity

Step8A currently resolves its legacy predictor key `lst_anomaly` to Step5's
`anomaly_zscore.tif` and aggregates that standardized raster into the existing
`lst_anomaly_mean` modeling column. Seam Audit does not change that feature or
any model output. It records the feature source explicitly as
`anomaly_zscore`, while keeping the intended native product identity
`absolute_lst_anomaly_celsius` separate.

Consequently, when no `lst_anomaly_celsius.tif` exists, the native artifact is
`missing_optional_native_artifact` and native status is
`insufficient_artifact`. If the Step8A feature is available, the 500 m audit
still runs from the existing dataset, the conclusion scope is
`modeling_scale_only`, and propagation is `insufficient_data`; a native PASS is
never inferred from `anomaly_zscore.tif`.

Optional configured products that the pipeline does not produce, such as
`baseline_ndvi_mean` and `baseline_ndvi_std`, are reported as
`not_produced_optional` and do not by themselves make overall assessment
incomplete. Summary reasons keep optional products, missing required artifacts,
missing boundary provenance, and artifact identity conflicts in separate
fields.

## Commands

Dry-run (path and provenance resolution only):

```bash
venv/bin/python scripts/main.py experiment \
  --experiment mugla_2021 \
  --from-stage seam-audit --to-stage seam-audit \
  --predictor-mode local-only --dry-run
```

Run V2:

```bash
venv/bin/python scripts/main.py experiment \
  --experiment mugla_2021 \
  --from-stage seam-audit --to-stage seam-audit \
  --predictor-mode local-only --force
```

The legacy V1 runner remains directly callable for regression inspection, but
the orchestrated `seam-audit` stage dispatches V2.

## V2 outputs

V2 writes `seam_audit_summary.json`, `seam_audit_summary.md`,
`product_metrics.parquet`, `boundary_segment_metrics.parquet`,
`control_metrics.parquet`, `boundary_registry.parquet`,
`artifact_resolution.parquet`, `seam_hotspots.geojson`, and `manifest.json`.
The manifest identifies schema `2.1`, the engine commit, and the V1
compatibility guarantee.

Source-scene auditing supports a future provenance interface under
`<output_root>/provenance/` (`scene_provenance.tif`, `source_scene_id.tif`,
`scene_boundary.geojson`, or `scene_manifest.parquet`). Until a compatible
pixel-level artifact is present, the result is
`insufficient_boundary_metadata`, `assessment_incomplete=true`, and never a
scientific blocker by itself.
