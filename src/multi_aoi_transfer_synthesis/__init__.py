"""multi_aoi_transfer_synthesis: generic, report-only cross-AOI synthesis.

This package NEVER runs modeling, adaptation, prediction, or bootstrap code.
It only reads already-frozen JSON/CSV/parquet outputs produced by earlier
pipeline stages (Step8, Step9B-G, Step10) for a user-supplied set of 2-5 AOI
experiment IDs, and synthesizes a generic cross-AOI report.

No module in this package may contain a hardcoded AOI name, a hardcoded
region count in an identifier, or a fixed pair/direction list. Everything is
derived generically from the AOI experiment IDs passed in by the caller.

Submodules:
    aoi_set             -- AOI-set validation, canonical ordering, canonical
                            set id, ordered-direction / unordered-pair
                            generation.
    resolvers           -- frozen-output resolution for the 6 input families.
    schema_adapters     -- per-schema-version parsing into a normalized
                            internal representation.
    status_derivation   -- pure functions deriving conservative status labels
                            from numeric values/confidence intervals.
    build               -- top-level orchestration (validate -> resolve ->
                            adapt -> derive statuses -> assemble -> render).
    render              -- JSON/Markdown/CSV rendering of the normalized
                            representation (no AOI-specific branching).
    manifest            -- manifest assembly + integrity/consistency checks.
"""
