"""Builds `multi_aoi_manifest.json`: the full list of resolved-input
records (path/schema_version/analysis_id/source-target ids/content
hash/primary_population/resolution_method) plus integrity/consistency
checks and, once the report files are written, their own output hashes.

This module performs no resolution/adaptation itself -- it only records
what `build.py` already resolved and hashes the files `render.py` wrote.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(synthesis: dict, output_paths: Optional[dict[str, Path]] = None) -> dict:
    """`output_paths`: {basename: Path} for the already-written report files
    (JSON/MD/3 CSVs); their content hashes are recorded once written. Pass
    None only for a dry-run manifest preview (no output files exist yet)."""
    resolved_inputs = synthesis.get("resolved_inputs", [])

    families_present = sorted({r["family"] for r in resolved_inputs})
    duplicate_check: dict[str, list[str]] = {}
    seen: dict[tuple, str] = {}
    for r in resolved_inputs:
        key = (r["family"], r.get("source_experiment_id"), r.get("target_experiment_id"))
        if key in seen and seen[key] != r["path"]:
            duplicate_check.setdefault(str(key), []).append(r["path"])
        else:
            seen[key] = r["path"]

    output_file_hashes = {}
    if output_paths:
        for basename, path in output_paths.items():
            output_file_hashes[basename] = _sha256_file(path)

    return {
        "canonical_set_id": synthesis["canonical_set_id"],
        "generated_at": synthesis["generated_at"],
        "primary_population": synthesis["primary_population"],
        "aois_display_order": synthesis["aois_display_order"],
        "aois_canonical_order": synthesis["aois_canonical_order"],
        "ordered_directions": synthesis["ordered_directions"],
        "unordered_pairs": synthesis["unordered_pairs"],
        "input_families_resolved": families_present,
        "resolved_inputs": resolved_inputs,
        "duplicate_input_paths_detected": duplicate_check,
        "output_file_hashes": output_file_hashes,
    }


def render_manifest(synthesis: dict, output_paths: dict[str, Path], manifest_path: Path) -> dict:
    import json

    manifest = build_manifest(synthesis, output_paths)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.write("\n")
    return manifest
