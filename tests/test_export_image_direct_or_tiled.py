"""Regression tests for the shared direct/tiled GeoTIFF exporter
(scripts/run_predictors_only.py:export_image_direct_or_tiled).

Root cause covered here
-----------------------
`geemap.ee_export_image` REFUSES a filename whose extension is not `.tif`: it
prints "The filename must end with .tif" and returns WITHOUT raising and
WITHOUT producing a file. The direct-export temporary file used to be named
`.<name>.tif.direct.tmp`, whose extension is `.tmp`, so EVERY direct export
silently produced nothing and every export fell back to the (much slower)
tiled path -- exactly what the pre-label export log showed. The tiled path was
unaffected because its tile names already end with `.tif`.

Everything here is synthetic: a fake `geemap` module is injected into
`sys.modules`, no Earth Engine object is created and no network call is made.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.run_predictors_only as rpo


class FakeGeemap:
    """Mimics the geemap contract that matters here.

    * a filename not ending in `.tif` is REFUSED silently (message printed,
      no exception, no file);
    * otherwise a small file is written, unless the fake is configured to
      fail.
    """

    def __init__(self, *, mode: str = "success", payload: bytes = b"GEOTIFF-BYTES"):
        self.mode = mode
        self.payload = payload
        self.filenames: list[str] = []

    def ee_export_image(self, image, filename, scale=None, region=None, crs=None,
                        file_per_band=False, **kwargs):
        self.filenames.append(str(filename))
        if not str(filename).endswith(".tif"):
            # The real geemap prints this and returns None.
            print("The filename must end with .tif")
            return None
        if self.mode == "raise":
            raise RuntimeError("Total request size must be <= 50331648 bytes")
        if self.mode == "no_file":
            return None
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self.payload)
        return None


@pytest.fixture
def fake_geemap(monkeypatch):
    def _install(**kwargs):
        fake = FakeGeemap(**kwargs)
        module = types.ModuleType("geemap")
        module.ee_export_image = fake.ee_export_image
        monkeypatch.setitem(sys.modules, "geemap", module)
        return fake
    return _install


class _Region:
    """A region object whose bounds cannot be resolved offline.

    `_estimate_request_bytes` fails on it, which makes the exporter fall back
    to its documented behaviour of attempting the direct export first -- the
    path under test.
    """

    def bounds(self):  # pragma: no cover - defensive
        raise RuntimeError("no Earth Engine in tests")


def _export(out_path: Path, tiles_dir: Path, **kwargs):
    return rpo.export_image_direct_or_tiled(
        image=object(), out_path=out_path, region=_Region(), scale=30,
        crs="EPSG:4326", label="unit_test", force=True, tiles_dir=tiles_dir,
        run_alignment_qa=False, **kwargs,
    )


# =============================================================================
# 38, 39. The filename contract
# =============================================================================
def test_direct_export_temp_filename_ends_with_tif(tmp_path):
    out_path = tmp_path / "current_lst__scene_weighted_median.tif"
    temp = rpo.direct_export_tmp_path(out_path)

    assert temp.suffix == ".tif"
    assert temp.name.endswith(".tif")
    assert temp.parent == out_path.parent, "temp must stay on the same filesystem"
    assert temp.name.startswith("."), "temp must stay hidden"
    assert temp != out_path
    # The exact regression: the OLD name ended with `.tmp`.
    assert not temp.name.endswith(".tmp")


def test_filename_handed_to_geemap_ends_with_tif(tmp_path, fake_geemap):
    fake = fake_geemap(mode="success")
    out_path = tmp_path / "modis_lst_mean_celsius.tif"

    _export(out_path, tmp_path / "_tiles")

    assert fake.filenames, "geemap was never called"
    for filename in fake.filenames:
        assert filename.endswith(".tif"), f"geemap got a non-.tif filename: {filename}"


def test_geemap_refuses_a_non_tif_filename(tmp_path, fake_geemap, capsys):
    """Guards the assumption this whole fix rests on."""
    fake = fake_geemap(mode="success")
    rejected = tmp_path / ".artifact.tif.direct.tmp"

    fake.ee_export_image(object(), filename=str(rejected))

    assert "The filename must end with .tif" in capsys.readouterr().out
    assert not rejected.exists(), "a refused filename must not produce a file"


# =============================================================================
# 40, 43. Direct success
# =============================================================================
def test_direct_success_does_not_call_the_tiled_fallback(tmp_path, fake_geemap):
    fake = fake_geemap(mode="success")
    out_path = tmp_path / "current_ndvi__scene_valid_count.tif"

    with patch.object(rpo, "_export_tiled") as tiled:
        result = _export(out_path, tmp_path / "_tiles")

    tiled.assert_not_called()
    assert result["transport"] == "direct"
    assert result["tile_grid"] is None
    assert len(fake.filenames) == 1


def test_direct_success_moves_the_file_atomically_to_the_target(tmp_path, fake_geemap):
    fake = fake_geemap(mode="success", payload=b"PAYLOAD-42")
    out_path = tmp_path / "nested" / "baseline_lst_2019__scene_weighted_median.tif"

    result = _export(out_path, tmp_path / "_tiles")

    assert out_path.is_file()
    assert out_path.read_bytes() == b"PAYLOAD-42"
    assert Path(result["path"]) == out_path
    # The temporary file is gone; nothing half-written is left behind.
    assert not rpo.direct_export_tmp_path(out_path).exists()
    assert sorted(p.name for p in out_path.parent.iterdir()) == [out_path.name]


# =============================================================================
# 41, 42, 44. Fallback behaviour is preserved
# =============================================================================
@pytest.mark.parametrize("mode", ["no_file", "raise"])
def test_direct_failure_falls_back_to_tiled(tmp_path, fake_geemap, mode):
    fake_geemap(mode=mode)
    out_path = tmp_path / "current_lst__scene_weighted_median.tif"
    tiles_dir = tmp_path / "_tiles"

    def fake_tiled(image, target, region, scale, crs, label, force, **kwargs):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"TILED")
        return Path(target)

    with patch.object(rpo, "_export_tiled", side_effect=fake_tiled) as tiled:
        result = _export(out_path, tiles_dir)

    assert tiled.call_count == 1, "the tiled fallback must run exactly once"
    assert result["transport"] == "tiled_direct_fallback"
    assert result["tile_grid"] == (2, 2), "escalation must still start at 2x2"
    assert out_path.read_bytes() == b"TILED"


def test_tiled_fallback_receives_the_unchanged_target_and_tiles_dir(tmp_path, fake_geemap):
    fake_geemap(mode="no_file")
    out_path = tmp_path / "modis_valid_observation_count.tif"
    tiles_dir = tmp_path / "_tiles" / "modis"

    def fake_tiled(image, target, region, scale, crs, label, force, **kwargs):
        Path(target).write_bytes(b"TILED")
        return Path(target)

    with patch.object(rpo, "_export_tiled", side_effect=fake_tiled) as tiled:
        _export(out_path, tiles_dir)

    kwargs = tiled.call_args.kwargs
    args = tiled.call_args.args
    assert args[1] == out_path
    assert kwargs["tiles_dir"] == tiles_dir
    assert kwargs["tile_rows"] == 2 and kwargs["tile_cols"] == 2


def test_tile_filenames_already_end_with_tif(tmp_path):
    """Why the tiled path was never affected by the regression."""
    out_path = tmp_path / "current_lst__scene_weighted_median.tif"
    tile_name = f"{out_path.stem}_tile_r0_c0.tif"
    assert tile_name.endswith(".tif")


def test_existing_output_is_still_skipped_without_force(tmp_path, fake_geemap):
    """Unrelated existing behaviour must not change."""
    fake = fake_geemap(mode="success")
    out_path = tmp_path / "already_there.tif"
    out_path.write_bytes(b"OLD")

    result = rpo.export_image_direct_or_tiled(
        image=object(), out_path=out_path, region=_Region(), scale=30,
        crs="EPSG:4326", label="unit_test", force=False,
        tiles_dir=tmp_path / "_tiles", run_alignment_qa=False,
    )
    assert result["transport"] == "skipped_existing"
    assert out_path.read_bytes() == b"OLD"
    assert fake.filenames == [], "a skipped export must not call geemap"
