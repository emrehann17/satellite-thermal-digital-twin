"""
preview_experiment_aoi.py

Step0B: hafif AOI onizleme / metadata yardimcisi.

Bir deneyin (experiment) Step0 metadata'sini (pencere, baseline yillari,
cikti koku vb.) ve -- mumkunse -- cozulmus AOI geometrisinin tipini/kaba
sinirlarini yazdirir. Hicbir export/pipeline/model calistirmaz.

GEE gerektiren kisim (region geometrisi cozumu) opsiyoneldir: GEE
initialize edilemezse (kimlik dogrulama yok, ag erisimi yok, vb.) script
CRASH ETMEZ -- yalnizca acik bir mesaj basar ve Step0 metadata'sini
gostermeye devam eder.

CLI:
    python scripts/preview_experiment_aoi.py --experiment kozan_2023
    python scripts/preview_experiment_aoi.py --experiment manavgat_2021

Cikti (mumkunse):
    outputs/experiments/<experiment_id>/step0/aoi_preview.geojson
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.io_utils import setup_logger
from core.regions import get_active_experiment, get_step_output_dir

log, log_file = setup_logger("preview_experiment_aoi")


def _print_experiment_summary(exp: dict, output_root: Path) -> None:
    log.info("experiment_id: %s", exp["experiment_id"])
    log.info("display_name: %s", exp["display_name"])
    log.info("region_key: %s", exp["region_key"])
    log.info("role: %s", exp["role"])
    log.info("predictor window: %s -> %s", exp["predictor_start_date"], exp["predictor_end_date"])
    log.info("label window: %s -> %s", exp["label_start_date"], exp["label_end_date"])
    log.info("baseline years: %s", ", ".join(str(y) for y in exp["baseline_years"]) or "(yok)")
    log.info("output root: %s", output_root)


def _try_resolve_geometry(experiment_id: str):
    """
    GEE'yi initialize etmeyi ve deneyin AOI geometrisini cozmeyi dener.

    Basarili olursa (geometry, type_str, bounds_info) dondurur.
    Basarisiz olursa (None, None, error_message) dondurur -- CRASH ETMEZ.
    """
    try:
        from core.gee_utils import init_gee
        from core.regions import get_region_for_experiment
    except Exception as exc:  # noqa: BLE001
        return None, None, f"GEE importlari basarisiz (ee/geemap eksik olabilir): {type(exc).__name__}: {exc}"

    try:
        init_gee()
    except Exception as exc:  # noqa: BLE001
        return None, None, (
            f"GEE initialize edilemedi ({type(exc).__name__}: {exc}). "
            "'earthengine authenticate' calistirin. AOI geometrisi "
            "gosterilemiyor, ancak Step0 metadata'si yukarida gecerlidir."
        )

    try:
        geometry = get_region_for_experiment(experiment_id)
        type_str = geometry.type().getInfo()
        bounds_info = geometry.bounds().getInfo()
    except Exception as exc:  # noqa: BLE001
        return None, None, f"AOI geometrisi cozulemedi: {type(exc).__name__}: {exc}"

    return geometry, type_str, bounds_info


def _write_geojson_preview(geometry, exp: dict, output_dir: Path) -> Path | None:
    try:
        geom_info = geometry.getInfo()
    except Exception as exc:  # noqa: BLE001
        log.warning("GeoJSON icin geometry.getInfo() basarisiz: %s", exc)
        return None

    feature = {
        "type": "Feature",
        "geometry": geom_info,
        "properties": {
            "experiment_id": exp["experiment_id"],
            "display_name": exp["display_name"],
            "region_key": exp["region_key"],
            "role": exp["role"],
            "notes": exp.get("notes"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "aoi_preview.geojson"
    out_path.write_text(json.dumps(feature, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("GeoJSON onizleme yazildi: %s", out_path)
    return out_path


def main(experiment_id: str = "kozan_2023") -> dict:
    exp = get_active_experiment(experiment_id)
    output_root = get_step_output_dir(experiment_id, "step0")
    _print_experiment_summary(exp, output_root)

    geometry, type_str, bounds_or_error = _try_resolve_geometry(experiment_id)

    geojson_path = None
    if geometry is not None:
        log.info("AOI geometry type: %s", type_str)
        log.info("AOI approximate bounds: %s", bounds_or_error)
        geojson_path = _write_geojson_preview(geometry, exp, output_root)
    else:
        log.warning("AOI geometrisi gosterilemedi: %s", bounds_or_error)

    return {
        "experiment_id": experiment_id,
        "region_key": exp["region_key"],
        "geometry_resolved": geometry is not None,
        "geojson_path": str(geojson_path) if geojson_path else None,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0B: bir deneyin Step0 metadata'sini ve (mumkunse) "
        "AOI geometri tipini/kaba sinirlarini yazdirir. Export/pipeline/model "
        "CALISTIRMAZ."
    )
    parser.add_argument("--experiment", type=str, default="kozan_2023")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(experiment_id=args.experiment)