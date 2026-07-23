"""
check_experiment_registry.py

Step0: pure, read-only deney (experiment) kayit defteri dogrulayicisi.

Bir experiment_id icin core/regions.py EXPERIMENTS kaydini okur, Step0
metadata'sini yazdirir ve saf kayit-defteri (registry) tutarlilik
kurallarini dogrular. Bu script:

    - HICBIR Earth Engine cagrisi YAPMAZ (ee.Geometry insa etmez; region_key
      cozunurlugu, core/regions.py'nin build_regions() donusunu STATIK
      olarak (AST ile) okuyarak dogrulanir -- build_regions() cagirmak
      ee.Initialize()/kimlik dogrulama gerektirir, bu yuzden hicbir zaman
      cagirilmaz).
    - run_label_gate_only / Step6A / Step6B / MCD64A1 export'unu CAGIRMAZ.
    - core/pipeline_orchestrator.py'yi import ETMEZ (import etmek modul
      seviyesinde bir logger/log-dizini yan etkisi tetikler); legacy
      uyumluluk (kozan_2023) da ayni sekilde STATIK olarak dogrulanir.
    - hicbir dizin/dosya OLUSTURMAZ; yalnizca stdout'a yazar (print).
    - raster/GeoTIFF dosyalarini KONTROL ETMEZ/GEREKTIRMEZ.

CLI:
    python scripts/check_experiment_registry.py --experiment kozan_2023
    python scripts/check_experiment_registry.py --experiment manavgat_2021
    python scripts/check_experiment_registry.py --experiment bejis_2022
    python scripts/check_experiment_registry.py --experiment mugla_2021
    python scripts/check_experiment_registry.py --experiment evia_2021
"""

from __future__ import annotations

import argparse
import ast
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.regions import get_experiment, get_experiment_output_root

_REGIONS_PY = _PROJECT_ROOT / "core" / "regions.py"
_ORCHESTRATOR_PY = _PROJECT_ROOT / "core" / "pipeline_orchestrator.py"
_LEGACY_LABELS_DIR = (_PROJECT_ROOT / "outputs" / "validation" / "labels").resolve()


class RegistryCheckError(SystemExit):
    """Fail-fast error for this read-only validator (diger step'lerle ayni konvansiyon)."""


# =============================================================================
# Statik (import/execute ETMEYEN) kaynak-kodu okuma yardimcilari
# =============================================================================
def _static_region_keys() -> set[str]:
    """`build_regions()` icindeki `return {...}` sozlugunun anahtarlarini,
    fonksiyonu HICBIR ZAMAN CAGIRMADAN (AST ile) doner.

    build_regions() cagrilirsa ee.Geometry.* insa edilir; bu, Earth Engine
    kutuphanesinin ilk kullanimda kimlik dogrulama/ag baglantisi denemesine
    (ee.Initialize gerektirir) yol acar -- bu yuzden BILEREK cagrilmaz.
    """
    tree = ast.parse(_REGIONS_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_regions":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    return {
                        key.value
                        for key in sub.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
    raise RegistryCheckError(
        "build_regions() icindeki return sozlugu statik olarak ayristirilamadi; "
        "region_key dogrulamasi yapilamiyor."
    )


def _static_legacy_experiment_id() -> str | None:
    """`core/pipeline_orchestrator.py`'deki `LEGACY_EXPERIMENT_ID = "..."`
    atamasini, modulu IMPORT ETMEDEN (AST ile) okur.

    core.pipeline_orchestrator'i import etmek modul seviyesinde bir
    setup_logger(...) yan etkisi (logs/ dizini + dosyasi) tetikler; bu
    read-only kayit defteri dogrulayicisi hicbir dizin/dosya OLUSTURMAMALIDIR,
    bu yuzden modul import edilmez, yalnizca kaynak metni okunur.
    """
    if not _ORCHESTRATOR_PY.exists():
        return None
    tree = ast.parse(_ORCHESTRATOR_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "LEGACY_EXPERIMENT_ID"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
    return None


# =============================================================================
# Yazdirma
# =============================================================================
def _print_registry_summary(exp: dict, output_root: Path) -> None:
    print(f"[Step0] experiment_id:  {exp['experiment_id']}")
    print(f"[Step0] display_name:   {exp['display_name']}")
    print(f"[Step0] region_key:     {exp['region_key']}")
    print(f"[Step0] role:           {exp['role']}")
    print(f"[Step0] predictor window: {exp['predictor_start_date']} -> {exp['predictor_end_date']}")
    print(f"[Step0] label window:     {exp['label_start_date']} -> {exp['label_end_date']}")
    print(f"[Step0] baseline years: {', '.join(str(y) for y in exp['baseline_years']) or '(yok)'}")
    print(f"[Step0] enabled:        {exp['enabled']}")
    print(f"[Step0] output_root (canonical): {output_root}")


# =============================================================================
# Dogrulamalar (hicbiri dosya/dizin OLUSTURMAZ, raster OKUMAZ)
# =============================================================================
def _validate_windows_and_baselines(exp: dict) -> None:
    predictor_end = datetime.strptime(exp["predictor_end_date"], "%Y-%m-%d")
    label_start = datetime.strptime(exp["label_start_date"], "%Y-%m-%d")
    if not predictor_end < label_start:
        raise RegistryCheckError(
            f"'{exp['experiment_id']}': predictor_end_date ({exp['predictor_end_date']}) "
            f"label_start_date'ten ({exp['label_start_date']}) ONCE degil."
        )

    event_year = label_start.year
    for year in exp["baseline_years"]:
        if year >= event_year:
            raise RegistryCheckError(
                f"'{exp['experiment_id']}': baseline yili {year}, olay yilindan "
                f"({event_year}) ONCE degil."
            )


def _validate_region_key(exp: dict, valid_region_keys: set[str]) -> None:
    if exp["region_key"] not in valid_region_keys:
        raise RegistryCheckError(
            f"'{exp['experiment_id']}': region_key ('{exp['region_key']}') "
            f"build_regions() ciktisinda bulunamadi. Gecerli anahtarlar: "
            f"{sorted(valid_region_keys)}."
        )


def _validate_namespacing(exp: dict, output_root: Path) -> None:
    """Kozan-disi deneyler icin output_root'un outputs/experiments/<experiment_id>/
    altinda kaldigini VE legacy paylasilan outputs/validation/labels/ dizini
    ile CAKISMADIGINI dogrular. Hicbir dizin/dosya OLUSTURULMAZ/OKUNMAZ --
    yalnizca Path string karsilastirmasi.
    """
    if exp["experiment_id"] == "kozan_2023":
        return

    experiments_root = (_PROJECT_ROOT / "outputs" / "experiments" / exp["experiment_id"]).resolve()
    resolved_root = output_root.resolve()

    if resolved_root != experiments_root and experiments_root not in resolved_root.parents:
        raise RegistryCheckError(
            f"'{exp['experiment_id']}': output_root ({resolved_root}) beklenen "
            f"outputs/experiments/{exp['experiment_id']}/ deseninde degil."
        )
    if resolved_root == _LEGACY_LABELS_DIR or _LEGACY_LABELS_DIR in resolved_root.parents:
        raise RegistryCheckError(
            f"'{exp['experiment_id']}': output_root ({resolved_root}) Kozan'in "
            "legacy paylasilan dizinine (outputs/validation/labels/) dusuyor."
        )


def _validate_legacy_compatibility() -> None:
    legacy_id = _static_legacy_experiment_id()
    if legacy_id != "kozan_2023":
        raise RegistryCheckError(
            "core/pipeline_orchestrator.py LEGACY_EXPERIMENT_ID artik 'kozan_2023' "
            f"degil (bulunan: {legacy_id!r}); legacy uyumluluk bozulmus."
        )


def main(experiment_id: str = "kozan_2023") -> dict:
    exp = get_experiment(experiment_id)
    output_root = get_experiment_output_root(experiment_id)

    _print_registry_summary(exp, output_root)

    _validate_windows_and_baselines(exp)
    _validate_region_key(exp, _static_region_keys())
    _validate_namespacing(exp, output_root)
    _validate_legacy_compatibility()

    print(
        f"[Step0] '{experiment_id}' registry dogrulamasi BASARILI "
        "(read-only; hicbir export/gate/predictor/model CALISTIRILMADI, "
        "hicbir dosya/dizin OLUSTURULMADI)."
    )

    return {
        "experiment_id": experiment_id,
        "region_key": exp["region_key"],
        "enabled": exp["enabled"],
        "output_root": str(output_root),
        "valid": True,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step0: saf, read-only deney (experiment) kayit defteri "
        "dogrulayicisi. Yalnizca core/regions.py EXPERIMENTS kaydini ve "
        "region_key/legacy-uyumluluk tutarliligini dogrular. Export/gate/"
        "predictor/model/bootstrap/report CALISTIRMAZ, hicbir dosya/dizin "
        "OLUSTURMAZ, hicbir Earth Engine cagrisi YAPMAZ."
    )
    parser.add_argument(
        "--experiment", type=str, default="kozan_2023",
        help="core/regions.py EXPERIMENTS kaydindaki experiment_id (orn. kozan_2023, "
        "manavgat_2021, bejis_2022, mugla_2021, evia_2021).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(experiment_id=args.experiment)
