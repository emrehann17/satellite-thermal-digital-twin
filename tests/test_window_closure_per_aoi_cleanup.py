"""Architecture guards for independent per-AOI window-closure production."""

import pytest

from scripts.main import build_parser
from src.multi_region_window_closure.contract import assert_regional_aoi
from src.multi_region_window_closure.driver import run_regional_actual
from src.multi_region_window_closure.per_aoi import regional_resume_matches
from src.multi_region_window_closure.schema import REGIONAL_ARTIFACT_SPECS
from src.multi_region_window_closure.validation import evaluate_regional
from src.multi_region_window_closure.validators import REGIONAL_CHECKS


@pytest.mark.parametrize(
    "aoi",
    ("bejis_2022", "mugla_2021", "evia_2021_extended", "montiferru_2021"),
)
def test_regional_command_accepts_each_production_aoi(aoi):
    args = build_parser().parse_args(["window-closure-region", "--experiment", aoi])
    assert args.command == "window-closure-region"
    assert assert_regional_aoi(aoi) == aoi


@pytest.mark.parametrize(
    "removed_command",
    ("window-closure-multi", "window-closure-synthesis"),
)
def test_removed_multi_aoi_commands_are_not_registered(removed_command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([removed_command])


def test_regional_runtime_contracts_import_independently():
    assert callable(run_regional_actual)
    assert callable(evaluate_regional)
    assert REGIONAL_ARTIFACT_SPECS
    assert REGIONAL_CHECKS
    assert not regional_resume_matches(
        {}, aoi="bejis_2022", analysis_id="id", config_hash="c", input_hash="i"
    )
