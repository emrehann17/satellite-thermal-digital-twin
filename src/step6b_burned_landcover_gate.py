"""
step6b_burned_landcover_gate.py

Burned-landcover diagnostic gate (Step6B).

WHY THIS EXISTS
---------------
Supervisor feedback: Kozan 2023 burned MCD64A1 cells are cropland/anız-burning
dominated, not natural-vegetation wildfire. The methodology (Step8A-8E) is
correct, but every NEW AOI/experiment must first pass a burned-landcover gate
BEFORE modeling, so we know up front whether an AOI is a genuine wildfire
candidate, a cropland/anız control, or has too few burned positives to say
anything useful.

This module answers exactly one question: "of the MCD64A1-burned ~500 m
cells, what landcover dominates them?" It does NOT train a model, does NOT
touch Step8 model science, and does NOT require Step7's thermal predictors
to exist -- only the raw MCD64A1 BurnDate raster (see Step6's
`export_raw_mcd64a1_labels()`) and a landcover raster aligned to the
predictor grid. This lets the gate run right after Step6's label export,
BEFORE Step7A-7E and Step8A.

REUSE, NOT REIMPLEMENTATION
----------------------------
This module deliberately imports its label/landcover/500m-cell-reconstruction
helpers directly from `src/step8a_prepare_500m_modeling_dataset.py`:
    resolve_reference_30m, resolve_label_raster, resolve_landcover,
    align_label_to_reference, inspect_label_raster, label_window_doy_bounds,
    doy_to_month_and_date, compute_block_size_pixels, mode_and_agreement,
    ESA_WORLDCOVER_CLASSES, LC_TREE_COVER, LC_SHRUBLAND, LC_GRASSLAND,
    LC_CROPLAND, LABEL_KIND_RAW, Step8AError.
This is the SAME 500 m block/tile reconstruction Step8A uses (same block
size, same tiling utility, same landcover class mapping), so gate numbers
are as comparable as possible to Step8A's own burned/landcover counts.
Step8A does NOT import this module, so there is no circular import.

Unlike Step8A's `build_dataset()`, this gate does NOT read any continuous
predictor raster (NDVI/DEM/thermal) -- only the label (BurnDate) and
landcover rasters -- so it can run before Step7/Step8A outputs exist.

GATE LEVEL
----------
    gate_level = "500m_reconstructed_mcd64a1_cell"
(an approximate native MCD64A1 grid reconstructed from the 30 m reference
grid, exactly like Step8A -- see STEP8A_MCD64A1_NATIVE_CELL_SIZE_M /
STEP8A_REFERENCE_PIXEL_SIZE_M in core/config.py).

DECISION RULES (supervisor-specified)
--------------------------------------
    burned_count < STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES
        -> "insufficient_burned_positives"
    burned_tree_shrub_grass_count / burned_count >= NATURAL_THRESHOLD (0.50)
        -> "wildfire_candidate_pass"
    burned_cropland_dominant_count / burned_count >= CROPLAND_THRESHOLD (0.50)
        -> "cropland_dominated_control"
    otherwise
        -> "mixed_or_uncertain"

This is DIAGNOSTIC ONLY. A "cropland_dominated_control" decision (the
expected Kozan 2023 outcome) does NOT stop the pipeline. This module only
raises (fails fast) if:
    - the raw BurnDate raster is missing or binary-looking (not real DOY),
    - required input rasters (reference grid / landcover) are missing,
    - the landcover class mapping cannot be resolved.

Inputs (read-only):
    outputs/validation/labels/mcd64a1_raw.tif   (preferred; see resolve_label_raster)
    outputs/step5/current_period_median_celsius.tif   (reference 30 m grid)
    data/landcover/landcover_esa_worldcover_v200_aligned_to_landsat.tif (preferred)
    data/landcover/landcover_esa_worldcover_v200.tif                    (fallback)

Outputs:
    outputs/validation/labels/burned_landcover_gate.json
    outputs/validation/labels/burned_landcover_gate.md
    outputs/validation/labels/burned_landcover_gate.csv

CLI:
    python src/step6b_burned_landcover_gate.py
    python src/step6b_burned_landcover_gate.py --force
    python src/step6b_burned_landcover_gate.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import rasterio
from rasterio.windows import Window

from core.config import (
    LABEL_END_DATE,
    LABEL_START_DATE,
    STEP6_BURNED_LANDCOVER_GATE_CROPLAND_THRESHOLD,
    STEP6_BURNED_LANDCOVER_GATE_LEVEL,
    STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES,
    STEP6_BURNED_LANDCOVER_GATE_NATURAL_THRESHOLD,
    STEP6_LABEL_OUTPUT_DIR,
)
from core.io_utils import setup_logger
from core.paths import PROJECT_ROOT
from core.utils.tiling import make_tile_grid

# --- Reuse Step8A's already-validated label/landcover/500m-cell helpers. ---
# Step8A does NOT import this module -> no circular import.
from src.step8a_prepare_500m_modeling_dataset import (
    ESA_WORLDCOVER_CLASSES,
    LC_CROPLAND,
    LC_GRASSLAND,
    LC_SHRUBLAND,
    LC_TREE_COVER,
    LABEL_KIND_RAW,
    Step8AError,
    _grid_matches,
    align_label_to_reference,
    compute_block_size_pixels,
    doy_to_month_and_date,
    inspect_label_raster,
    mode_and_agreement,
    prepare_aligned_landcover,
    resolve_label_raster,
    resolve_landcover,
    resolve_reference_30m,
)

BASE_DIR = PROJECT_ROOT
log, log_file = setup_logger("step6b_burned_landcover_gate")

GATE_LEVEL_500M = "500m_reconstructed_mcd64a1_cell"


class Step6BError(SystemExit):
    """Fail-fast error for Step6B (extends SystemExit like other steps)."""


def _dominant_landcover_name(code: int) -> str:
    return ESA_WORLDCOVER_CLASSES.get(code, f"unknown_{code}")


def _safe_fraction(numerator: int, denominator: int):
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _resolve_explicit_landcover(
    landcover_path_arg: str, reference_path: Path, out_dir: Path
) -> tuple[Path, dict]:
    """
    Acikca verilen (explicit) landcover rasterini cozer -- legacy
    data/landcover/ yollarina HIC dokunmadan.

    Verilen dosya referans gridle birebir uyusuyorsa dogrudan kullanilir.
    Uyusmuyorsa KAYNAK kabul edilir ve Step8A'nin prepare_aligned_landcover'i
    (nearest-neighbor; kategorik veri ASLA bilinear resample edilmez) ile
    out_dir altina (<stem>_aligned_to_reference.tif) hizalanir.
    """
    p = Path(landcover_path_arg)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        raise Step6BError(f"Belirtilen landcover rasteri bulunamadi: {p}")

    with rasterio.open(reference_path) as ref:
        ref_w, ref_h, ref_crs, ref_t = ref.width, ref.height, ref.crs, ref.transform

    if _grid_matches(p, ref_w, ref_h, ref_crs, ref_t):
        log.info("Explicit landcover referans gridle zaten hizali: %s", p)
        return p, {
            "original_landcover_path": str(p),
            "aligned_landcover_path": str(p),
            "landcover_alignment_method": "explicit_preexisting_aligned",
        }

    aligned_path = out_dir / f"{p.stem}_aligned_to_reference.tif"
    log.info(
        "Explicit landcover referans gridle uyusmuyor; nearest-neighbor ile "
        "hizalaniyor -> %s", aligned_path,
    )
    try:
        result = prepare_aligned_landcover(reference_path, p, aligned_path)
    except Step8AError as exc:
        raise Step6BError(str(exc)) from exc
    return result["path"], {
        "original_landcover_path": str(p),
        "aligned_landcover_path": str(result["path"]),
        "landcover_alignment_method": "nearest_neighbor_to_reference_grid",
        "landcover_alignment_created": result["created"],
    }


# =============================================================================
# Core gate computation (500 m-cell level)
# =============================================================================
def compute_gate(
    label_path: Path,
    label_kind: str,
    reference_path: Path,
    landcover_path: Path,
    label_start: str,
    label_end: str,
    output_dir: Path,
    min_positives: int,
    natural_threshold: float,
    cropland_threshold: float,
) -> dict:
    """
    Reconstructs approximate native ~500 m MCD64A1 cells from the 30 m
    reference grid (SAME block size / tiling logic as Step8A) and, for each
    cell, determines (a) burned/unburned from the raw BurnDate raster and
    (b) the dominant landcover class -- WITHOUT reading any continuous
    predictor raster.
    """
    with rasterio.open(reference_path) as ref:
        ref_w, ref_h = ref.width, ref.height
        ref_transform = ref.transform
        ref_crs = ref.crs
    ref_profile = {"width": ref_w, "height": ref_h, "crs": ref_crs, "transform": ref_transform}

    aligned_label_path = align_label_to_reference(label_path, ref_profile, output_dir)

    block_size = compute_block_size_pixels()
    tile_grid = make_tile_grid({"width": ref_w, "height": ref_h}, tile_size_pixels=block_size)
    tiles = tile_grid["tiles"]
    log.info(
        "500 m hucre rekonstruksiyonu (Step8A ile ayni block/tile mantigi): "
        "block_size_pixels=%d, toplam hucre (yaklasik)=%d",
        block_size, len(tiles),
    )

    label_src = rasterio.open(aligned_label_path)
    landcover_src = rasterio.open(landcover_path)
    landcover_nodata = landcover_src.nodata if landcover_src.nodata is not None else 0

    burn_month_available = (label_kind == LABEL_KIND_RAW)

    total_cells = 0
    burned_count = 0
    unburned_count = 0
    burned_dominant_counts: dict = {}
    unburned_dominant_counts: dict = {}
    burned_cells_without_valid_landcover = 0
    unburned_cells_without_valid_landcover = 0
    warnings_list: list[str] = []

    for tile in tiles:
        col_off, row_off, w, h = tile["write_window"]
        window = Window(col_off, row_off, w, h)
        total_cells += 1

        # --- Label (BurnDate) ---
        label_arr = label_src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        valid_label = label_arr[np.isfinite(label_arr)]
        positive_doy = valid_label[valid_label > 0]

        burned_flag = 0
        if positive_doy.size > 0:
            if burn_month_available:
                rep_doy, _rep_count, _n = mode_and_agreement(positive_doy)
                month, _iso = doy_to_month_and_date(rep_doy, label_start, label_end)
                burned_flag = 1 if month is not None else 0
            else:
                # Binary fallback label: no DOY/month info; any positive
                # value (i.e. 1) means burned.
                burned_flag = 1

        # --- Landcover (dominant class within the same 500 m block) ---
        lc_arr = landcover_src.read(1, window=window, masked=True)
        lc_filled = lc_arr.filled(landcover_nodata).astype("float64")
        lc_valid = ~np.ma.getmaskarray(lc_arr)
        lc_valid = lc_valid & np.isfinite(lc_filled) & (lc_filled != landcover_nodata)
        lc_valid_count = int(lc_valid.sum())

        dominant_name = None
        if lc_valid_count > 0:
            lc_vals = lc_filled[lc_valid].astype(int)
            uniq, counts = np.unique(lc_vals, return_counts=True)
            dominant_class = int(uniq[int(np.argmax(counts))])
            dominant_name = _dominant_landcover_name(dominant_class)

        if burned_flag == 1:
            burned_count += 1
            if dominant_name is not None:
                burned_dominant_counts[dominant_name] = burned_dominant_counts.get(dominant_name, 0) + 1
            else:
                burned_cells_without_valid_landcover += 1
        else:
            unburned_count += 1
            if dominant_name is not None:
                unburned_dominant_counts[dominant_name] = unburned_dominant_counts.get(dominant_name, 0) + 1
            else:
                unburned_cells_without_valid_landcover += 1

    label_src.close()
    landcover_src.close()

    if burned_cells_without_valid_landcover > 0:
        warnings_list.append(
            f"{burned_cells_without_valid_landcover} burned hucrede gecerli "
            "landcover pikseli yok; bu hucreler dominant-class sayimlarina "
            "DAHIL EDILMEDI (fraksiyonlarin paydasi yine de burned_count'tur, "
            "bu yuzden dominant-class sayimlarinin toplami burned_count'tan "
            "az olabilir)."
        )
    if not burn_month_available:
        warnings_list.append(
            "Binary burned mask (label_kind=binary_fallback_no_months) "
            "kullanildi; burned/unburned DOY penceresi kontrolu YAPILAMADI "
            "(her pozitif deger burned sayildi). Dogru sonuc icin gercek raw "
            "BurnDate rasterini kullanin."
        )

    burned_tree_cover_count = burned_dominant_counts.get("tree_cover", 0)
    burned_shrubland_count = burned_dominant_counts.get("shrubland", 0)
    burned_grassland_count = burned_dominant_counts.get("grassland", 0)
    burned_cropland_count = burned_dominant_counts.get("cropland", 0)
    burned_tree_shrub_grass_count = burned_tree_cover_count + burned_shrubland_count + burned_grassland_count
    burned_tree_shrub_count = burned_tree_cover_count + burned_shrubland_count
    burned_cropland_dominant_count = burned_cropland_count

    burned_natural_vegetation_fraction = _safe_fraction(burned_tree_shrub_grass_count, burned_count)
    burned_cropland_fraction = _safe_fraction(burned_cropland_dominant_count, burned_count)
    burned_tree_shrub_fraction = _safe_fraction(burned_tree_shrub_count, burned_count)

    # --- Decision rules (supervisor-specified) ---
    if burned_count < min_positives:
        decision = "insufficient_burned_positives"
        reason = f"burned_count={burned_count} < min_positives={min_positives}."
    elif burned_natural_vegetation_fraction is not None and burned_natural_vegetation_fraction >= natural_threshold:
        decision = "wildfire_candidate_pass"
        reason = (
            f"burned_tree_shrub_grass_count/burned_count="
            f"{burned_natural_vegetation_fraction:.3f} >= natural_threshold={natural_threshold}."
        )
    elif burned_cropland_fraction is not None and burned_cropland_fraction >= cropland_threshold:
        decision = "cropland_dominated_control"
        reason = (
            f"burned_cropland_dominant_count/burned_count="
            f"{burned_cropland_fraction:.3f} >= cropland_threshold={cropland_threshold}."
        )
    else:
        decision = "mixed_or_uncertain"
        reason = (
            "Ne natural (tree+shrub+grass dominant) ne de cropland-dominant "
            "esigi karsilanmadi (mixed/uncertain burned-landcover kompozisyonu)."
        )

    return {
        "gate_level": GATE_LEVEL_500M,
        "label_kind": label_kind,
        "burn_month_available": burn_month_available,
        "block_size_pixels": block_size,
        "total_valid_cells_or_pixels_considered": total_cells,
        "burned_count": burned_count,
        "unburned_count": unburned_count,
        "burned_landcover_dominant_counts": burned_dominant_counts,
        "unburned_landcover_dominant_counts": unburned_dominant_counts,
        "burned_tree_cover_count": burned_tree_cover_count,
        "burned_shrubland_count": burned_shrubland_count,
        "burned_grassland_count": burned_grassland_count,
        "burned_cropland_count": burned_cropland_count,
        "burned_tree_shrub_grass_count": burned_tree_shrub_grass_count,
        "burned_tree_shrub_count": burned_tree_shrub_count,
        "burned_cropland_dominant_count": burned_cropland_dominant_count,
        "burned_natural_vegetation_fraction": burned_natural_vegetation_fraction,
        "burned_cropland_fraction": burned_cropland_fraction,
        "burned_tree_shrub_fraction": burned_tree_shrub_fraction,
        "burned_cells_without_valid_landcover": burned_cells_without_valid_landcover,
        "unburned_cells_without_valid_landcover": unburned_cells_without_valid_landcover,
        "decision": decision,
        "reason": reason,
        "warnings": warnings_list,
        "thresholds": {
            "min_positives": min_positives,
            "natural_threshold": natural_threshold,
            "cropland_threshold": cropland_threshold,
        },
        "inputs": {
            "label_path": str(label_path),
            "aligned_label_path": str(aligned_label_path),
            "reference_path": str(reference_path),
            "landcover_path": str(landcover_path),
            "label_start": label_start,
            "label_end": label_end,
        },
    }


# =============================================================================
# Output writers
# =============================================================================
def write_json(gate: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gate, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    log.info("JSON yazildi: %s", out_path)
    return out_path


def write_csv(gate: dict, out_path: Path) -> Path:
    scalar_fields = [
        "gate_level", "label_kind", "burn_month_available", "block_size_pixels",
        "total_valid_cells_or_pixels_considered", "burned_count", "unburned_count",
        "burned_tree_cover_count", "burned_shrubland_count", "burned_grassland_count",
        "burned_cropland_count", "burned_tree_shrub_grass_count", "burned_tree_shrub_count",
        "burned_cropland_dominant_count", "burned_natural_vegetation_fraction",
        "burned_cropland_fraction", "burned_tree_shrub_fraction",
        "burned_cells_without_valid_landcover", "unburned_cells_without_valid_landcover",
        "decision", "reason",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for field in scalar_fields:
            writer.writerow([field, gate.get(field)])
        for cls_name, count in sorted(gate.get("burned_landcover_dominant_counts", {}).items()):
            writer.writerow([f"burned_landcover_dominant_counts__{cls_name}", count])
        for cls_name, count in sorted(gate.get("unburned_landcover_dominant_counts", {}).items()):
            writer.writerow([f"unburned_landcover_dominant_counts__{cls_name}", count])
    log.info("CSV yazildi: %s", out_path)
    return out_path


def write_markdown(gate: dict, out_path: Path) -> Path:
    lines = []
    lines.append("# Burned-Landcover Gate (Step6B)")
    lines.append("")
    lines.append(
        "Diagnostic gate: summarizes the landcover composition of MCD64A1-burned "
        f"~500 m cells (`gate_level = {gate['gate_level']}`). This does NOT train "
        "a model and does NOT stop the pipeline on a cropland-dominated result."
    )
    lines.append("")
    lines.append(f"**Decision: `{gate['decision']}`**")
    lines.append("")
    lines.append(f"Reason: {gate['reason']}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in [
        "total_valid_cells_or_pixels_considered", "burned_count", "unburned_count",
        "burned_tree_cover_count", "burned_shrubland_count", "burned_grassland_count",
        "burned_cropland_count", "burned_tree_shrub_grass_count", "burned_tree_shrub_count",
        "burned_cropland_dominant_count",
    ]:
        lines.append(f"| {key} | {gate[key]} |")
    lines.append("")
    lines.append("## Fractions (denominator = burned_count)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in ["burned_natural_vegetation_fraction", "burned_cropland_fraction", "burned_tree_shrub_fraction"]:
        val = gate[key]
        lines.append(f"| {key} | {val:.4f} |" if val is not None else f"| {key} | n/a |")
    lines.append("")
    lines.append("## Burned cells: dominant landcover breakdown")
    lines.append("")
    lines.append("| Dominant class | Burned cell count |")
    lines.append("|---|---|")
    for cls_name, count in sorted(gate["burned_landcover_dominant_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cls_name} | {count} |")
    lines.append("")
    if gate["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for w in gate["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Thresholds used")
    lines.append("")
    for k, v in gate["thresholds"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for k, v in gate["inputs"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown yazildi: %s", out_path)
    return out_path


# =============================================================================
# Orchestration
# =============================================================================
def main(
    output_dir_arg: str = STEP6_LABEL_OUTPUT_DIR,
    force: bool = False,
    label_raster_arg: str | None = None,
    reference_30m_arg: str | None = None,
    landcover_path_arg: str | None = None,
    label_start: str = LABEL_START_DATE,
    label_end: str = LABEL_END_DATE,
    min_positives: int = STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES,
    natural_threshold: float = STEP6_BURNED_LANDCOVER_GATE_NATURAL_THRESHOLD,
    cropland_threshold: float = STEP6_BURNED_LANDCOVER_GATE_CROPLAND_THRESHOLD,
) -> dict:
    """
    Path-aware davranis (Step0C):
        Hicbir path argumani verilmezse legacy Kozan davranisi BIREBIR
        korunur (outputs/validation/labels + outputs/step5 referansi +
        data/landcover kaynagi).
        Acik path'ler verilirse (or. Manavgat namespaced dosyalari) yalnizca
        onlar kullanilir -- legacy Kozan dosyalarina NE OKUMA NE YAZMA
        yapilir. landcover_path_arg verilirse referans gridle birebir
        uyusmasi zorunludur; uyusmuyorsa kaynak kabul edilip nearest-neighbor
        ile output_dir altina hizalanir (kategorik veri, ASLA bilinear degil;
        Step8A'nin prepare_aligned_landcover'i reuse edilir).
    """
    log.info("=" * 60)
    log.info("STEP 6B BASLIYOR (burned-landcover diagnostic gate)")
    log.info("=" * 60)

    out_dir = Path(output_dir_arg)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    required_outputs = [
        out_dir / "burned_landcover_gate.json",
        out_dir / "burned_landcover_gate.md",
        out_dir / "burned_landcover_gate.csv",
    ]
    if any(p.exists() for p in required_outputs) and not force:
        present = [p.name for p in required_outputs if p.exists()]
        raise Step6BError(
            "Step6B ciktilari zaten var (" + ", ".join(present)
            + "). Uzerine yazmak icin --force verin."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_path = resolve_reference_30m(reference_30m_arg)

    # Step6B needs GENUINE raw BurnDate (day-of-year), not a binary fallback --
    # otherwise the burned/unburned-in-window determination is not honest.
    label_path, label_kind = resolve_label_raster(label_raster_arg)
    if label_kind != LABEL_KIND_RAW:
        raise Step6BError(
            f"Step6B icin gercek raw BurnDate rasteri gerekli, ama bulunan "
            f"etiket dosyasi binary gorunuyor (label_kind={label_kind!r}, "
            f"path={label_path}). Once Step6'nin canonical export'unu "
            "calistirin: src.step6_validate_fire_relation.export_raw_mcd64a1_labels() "
            "(veya: python scripts/export_mcd64a1_raw_burndate.py)."
        )

    # Fail-fast: raises if the "raw" raster actually looks binary or has no
    # values inside the label DOY window (same check Step8A relies on).
    try:
        label_diag = inspect_label_raster(label_path, label_kind, label_start, label_end)
    except Step8AError as exc:
        raise Step6BError(str(exc)) from exc
    log.info(
        "Label raster diagnostics: kind=%s min=%s max=%s count_one=%s "
        "count_gt_one=%s in_DOY_range=%s",
        label_diag.get("label_kind"), label_diag.get("min"), label_diag.get("max"),
        label_diag.get("count_one"), label_diag.get("count_gt_one"),
        label_diag.get("count_in_label_doy_range"),
    )

    if landcover_path_arg:
        landcover_path, landcover_info = _resolve_explicit_landcover(
            landcover_path_arg, reference_path, out_dir
        )
    else:
        landcover_path, landcover_info = resolve_landcover(reference_path)
    log.info("Landcover rasteri (hizali): %s", landcover_path)

    gate = compute_gate(
        label_path=label_path,
        label_kind=label_kind,
        reference_path=reference_path,
        landcover_path=landcover_path,
        label_start=label_start,
        label_end=label_end,
        output_dir=out_dir,
        min_positives=min_positives,
        natural_threshold=natural_threshold,
        cropland_threshold=cropland_threshold,
    )
    gate["label_raster_diagnostics"] = label_diag
    gate["landcover_info"] = landcover_info

    json_path = write_json(gate, out_dir / "burned_landcover_gate.json")
    md_path = write_markdown(gate, out_dir / "burned_landcover_gate.md")
    csv_path = write_csv(gate, out_dir / "burned_landcover_gate.csv")

    log.info(
        "Karar: %s (burned_count=%d, natural_fraction=%s, cropland_fraction=%s)",
        gate["decision"], gate["burned_count"],
        gate["burned_natural_vegetation_fraction"], gate["burned_cropland_fraction"],
    )
    log.info("=" * 60)
    log.info("STEP 6B TAMAMLANDI (diagnostic only; pipeline durdurulmadi)")
    log.info("=" * 60)

    return {
        "decision": gate["decision"],
        "json_path": str(json_path),
        "md_path": str(md_path),
        "csv_path": str(csv_path),
        "burned_count": gate["burned_count"],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step6B: burned-landcover diagnostic gate. Summarizes the "
        "landcover composition of MCD64A1-burned ~500 m cells (same "
        "reconstruction level as Step8A) and classifies the AOI as a "
        "wildfire candidate, a cropland/anız control, or insufficient/mixed."
    )
    parser.add_argument("--output-dir", type=str, default=STEP6_LABEL_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--label-raster", "--label-path", dest="label_raster", type=str, default=None,
        help="Raw MCD64A1 BurnDate raster yolu (verilmezse legacy Kozan kesfi).",
    )
    parser.add_argument(
        "--reference-30m", "--reference-path", dest="reference_30m", type=str, default=None,
        help="30 m referans grid yolu (verilmezse legacy outputs/step5 dosyasi).",
    )
    parser.add_argument(
        "--landcover-path", type=str, default=None,
        help="Landcover raster yolu (verilmezse legacy data/landcover kesfi). "
        "Referans gridle uyusmuyorsa nearest-neighbor ile output-dir altina hizalanir.",
    )
    parser.add_argument("--label-start", type=str, default=LABEL_START_DATE)
    parser.add_argument("--label-end", type=str, default=LABEL_END_DATE)
    parser.add_argument("--min-positives", type=int, default=STEP6_BURNED_LANDCOVER_GATE_MIN_POSITIVES)
    parser.add_argument("--natural-threshold", type=float, default=STEP6_BURNED_LANDCOVER_GATE_NATURAL_THRESHOLD)
    parser.add_argument("--cropland-threshold", type=float, default=STEP6_BURNED_LANDCOVER_GATE_CROPLAND_THRESHOLD)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir_arg=args.output_dir,
        force=args.force,
        label_raster_arg=args.label_raster,
        reference_30m_arg=args.reference_30m,
        landcover_path_arg=args.landcover_path,
        label_start=args.label_start,
        label_end=args.label_end,
        min_positives=args.min_positives,
        natural_threshold=args.natural_threshold,
        cropland_threshold=args.cropland_threshold,
    )