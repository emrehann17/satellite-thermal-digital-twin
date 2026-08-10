"""Side-effect-free production wiring tests (all production calls mocked)."""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest

from src.multi_region_window_closure.contract import MultiRegionWindowClosureError
from src.multi_region_window_closure.contract import VARIANTS as _VARIANTS
from src.multi_region_window_closure.production import (
    PRODUCTION_STAGE_MAP, ProductionRegionalEngine,
    ProductionSynthesisInputAdapter,normalize_bootstrap_summary,derive_fit_accounting,
    resolve_production_stage_result,
)
from src.multi_region_window_closure.schema import build_manifest,write_manifest_with_digest,verify_manifest_digest,REGIONAL_ARTIFACT_SPECS


def test_production_regional_adapter_maps_exact_scientific_stages(tmp_path):
    calls=[]
    def runner(**kw):
        calls.append((kw["from_stage"],kw["to_stage"],kw["experiment_id"]))
        return {"status":"PASS","experiment_id":"bejis_2022","analysis_id":"prod","files_written":[]}
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=runner)
    root=tmp_path/"window_closure_region"/"bejis_2022"/"id";root.mkdir(parents=True)
    ctx={"aoi":"bejis_2022","analysis_id":"id"}
    for stage in ("plan","export","local-downstream","fit","compare"):
        engine.run_stage(stage,root,ctx)
    assert calls==[("plan","plan","bejis_2022"),("prelabel-export","predictor-export","bejis_2022"),("local-downstream","local-downstream","bejis_2022"),("model","model","bejis_2022"),("compare","compare","bejis_2022")]
    assert tuple(PRODUCTION_STAGE_MAP)==("plan","export","local-downstream","fit","compare")


def test_production_adapter_rejects_old_evia_and_cross_aoi_context(tmp_path):
    with pytest.raises(MultiRegionWindowClosureError): ProductionRegionalEngine(aoi="evia_2021")
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=Mock())
    with pytest.raises(MultiRegionWindowClosureError): engine.run_stage("plan",tmp_path,{"aoi":"mugla_2021"})


def test_production_adapter_failure_is_fail_closed(tmp_path):
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=lambda **kw:{"status":"FAIL","experiment_id":"bejis_2022"})
    root=tmp_path/"window_closure_region"/"bejis_2022"/"id";root.mkdir(parents=True)
    with pytest.raises(RuntimeError,match="resolved_status='fail'"): engine.run_stage("plan",root,{"aoi":"bejis_2022"})


def test_cli_actual_guards_and_no_injected_engine():
    from scripts.run_window_closure_region import main as regional_main
    with pytest.raises(MultiRegionWindowClosureError): regional_main(["--experiment","bejis_2022"])
    with pytest.raises(MultiRegionWindowClosureError): regional_main(["--experiment","bejis_2022","--dry-run","--execute-actual"])
    with pytest.raises(MultiRegionWindowClosureError): regional_main(["--experiment","bejis_2022","--force"])


def _j(path,payload): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload))


def test_manavgat_production_loader_uses_exact_pass_artifacts(tmp_path,monkeypatch):
    import hashlib
    from src.multi_region_window_closure.bootstrap import comparison_series
    from src.multi_region_window_closure import inputs
    from src.multi_region_window_closure.inputs import CANONICAL_STEP8A_SHA256
    root=tmp_path/"manavgat"; aid="a1"
    canonical=tmp_path/"canonical.parquet";canonical.write_bytes(b"canonical");digest=hashlib.sha256(canonical.read_bytes()).hexdigest();monkeypatch.setitem(CANONICAL_STEP8A_SHA256,"manavgat_2021",digest);monkeypatch.setattr(inputs,"canonical_step8a_path",lambda aoi:canonical)
    _j(root/"config/preregistration.json",{"analysis_id":aid,"experiment_id":"manavgat_2021","schema_version":"window_closure_sensitivity.v1","frozen_input_sha256":{"canonical_step8a":digest},"scientific_configuration":{"primary_population":"burnable_tree_shrub_grass"}})
    _j(root/"config/frozen_input_inventory.json",{"experiment_id":"manavgat_2021","read_only":True,"frozen_input_sha256":{"canonical_step8a":digest},"inventory":{"canonical_step8a":{"exists":True,"path":str(canonical),"sha256":digest}}})
    _j(root/"model/model_stage_metadata.json",{"analysis_id":aid,"experiment_id":"manavgat_2021","status":"pass","canonical_step8a_sha256":digest})
    _j(root/"compare/compare_stage_metadata.json",{"analysis_id":aid,"experiment_id":"manavgat_2021","status":"pass"})
    _j(root/"model/common_cohort/common_cohort_metadata.json",{"analysis_id":aid})
    _j(root/"model/shared_folds/shared_spatial_folds_metadata.json",{"analysis_id":aid})
    (root/"model/bootstrap").mkdir(parents=True)
    rows=[]
    for row in comparison_series("manavgat_2021"):
        rows.append({"comparison":row["comparison_family"],"variant_id":row["variant"],"model_family":row["model_a"],"metric":row["metric"],"point_delta":.1,"ci_low":.01,"ci_high":.2})
    pd.DataFrame(rows).to_csv(root/"model/bootstrap/paired_bootstrap_summary.csv",index=False)
    record=ProductionSynthesisInputAdapter._load_manavgat(root)
    assert record and record["validator_status"]=="PASS" and len(record["reference_file_hashes"])==7
    _j(root/"compare/compare_stage_metadata.json",{"analysis_id":"drift","experiment_id":"manavgat_2021","status":"pass"})
    assert ProductionSynthesisInputAdapter._load_manavgat(root)["validator_status"]=="FAIL"


def test_static_actual_paths_have_no_placeholder():
    paths=[Path("scripts/run_window_closure_region.py"),Path("src/multi_region_window_closure/production.py")]
    text="\n".join(path.read_text() for path in paths)
    assert "NotImplementedError" not in text
    assert "injected reviewed scientific engine" not in text

def _production_summary_shape():
    from src.multi_region_window_closure.bootstrap import comparison_series
    return pd.DataFrame([{"comparison":r["comparison_family"],"variant_id":r["variant"],"model_family":r["model_a"],"metric":r["metric"],"point_delta":.1,"ci_low":.01,"ci_high":.2} for r in comparison_series("x")])

def test_production_shape_has_27_unique_scientific_series():
    out=normalize_bootstrap_summary(_production_summary_shape(),aoi="bejis_2022",source_analysis_id="id")
    keys=["comparison_family","variant","model_a","model_b","metric"]
    assert len(out)==27 and not out.duplicated(keys).any()
    replicate=pd.DataFrame([(tuple(row),rep) for row in out[keys].itertuples(index=False,name=None) for rep in range(1000)],columns=["series","replicate_id"])
    assert not replicate.duplicated().any() and len(replicate)==27000

def test_duplicate_production_series_fails():
    frame=_production_summary_shape();frame.iloc[1]=frame.iloc[0]
    with pytest.raises(MultiRegionWindowClosureError,match="27 unique"):normalize_bootstrap_summary(frame,aoi="bejis_2022",source_analysis_id="id")

def test_manifest_digest_missing_malformed_and_drift(tmp_path):
    (tmp_path/"config.json").write_text("{}")
    manifest=build_manifest("id",tmp_path,"c","i",{"bejis_2022":"x"*64},"",schema_version="window_closure_region.v1",artifact_specs=REGIONAL_ARTIFACT_SPECS)
    write_manifest_with_digest(tmp_path/"manifest.json",manifest);assert verify_manifest_digest(tmp_path)[0]
    (tmp_path/"manifest.sha256").write_text("bad");assert not verify_manifest_digest(tmp_path)[0]
    write_manifest_with_digest(tmp_path/"manifest.json",manifest);(tmp_path/"manifest.json").write_text("{}");assert not verify_manifest_digest(tmp_path)[0]
    (tmp_path/"manifest.sha256").unlink();assert not verify_manifest_digest(tmp_path)[0]

def test_fit_accounting_comes_from_fold_ledger_and_aux_metadata(tmp_path):
    rows=[{"variant":v,"model":m,"fold_id":f} for v in ("canonical","close_7d_earlier","close_14d_earlier") for m in ("baseline","thermal") for f in range(5)]
    for v in ("close_7d_earlier","close_14d_earlier"):_j(tmp_path/"variants"/v/"local_downstream_metadata.json",{"status":"pass","downscaling_model_fit":True})
    result=derive_fit_accounting(pd.DataFrame(rows),tmp_path);assert result["completed_primary_estimator_fits"]==30 and result["completed_auxiliary_downscaling_fits"]==2 and result["completed_total_fits"]==32
    assert derive_fit_accounting(pd.DataFrame(rows[:-1]),tmp_path)["completed_primary_estimator_fits"]==29
    (tmp_path/"variants/close_14d_earlier/local_downstream_metadata.json").unlink();assert derive_fit_accounting(pd.DataFrame(rows),tmp_path)["completed_auxiliary_downscaling_fits"] is None

def test_existing_regional_namespace_fails_closed(tmp_path):
    from src.multi_region_window_closure.driver import run_regional_actual
    root=tmp_path/"bejis_2022"/"id";root.mkdir(parents=True)
    with pytest.raises(RuntimeError,match="REGIONAL_NAMESPACE_ALREADY_EXISTS"):run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=tmp_path,engine=Mock(),config_hash="c",input_hash="i",execute_actual=True)

@pytest.mark.parametrize("requested",[False,True])
def test_plan_inner_resume_is_always_false(tmp_path,requested):
    calls=[]
    def runner(**kw):calls.append(kw);return {"status":"PASS","experiment_id":"bejis_2022","analysis_id":"p","files_written":[]}
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=runner);root=tmp_path/"window_closure_region/bejis_2022/id";root.mkdir(parents=True)
    engine.run_stage("plan",root,{"aoi":"bejis_2022","resume_requested":requested})
    assert calls[0]["resume"] is False and calls[0]["force"] is False


def _pass_runner(calls):
    def runner(**kwargs):
        calls.append(kwargs)
        return {"status":"pass","experiment_id":"bejis_2022","analysis_id":"prod","files_written":[]}
    return runner


def _write_verified_inner_metadata(root, stage):
    import hashlib
    prod=root/"_production";aoi=prod/"bejis_2022"
    _j(aoi/"config/preregistration.json",{"analysis_id":"prod"})
    def record(path):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"verified")
        return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    if stage=="local-downstream":
        for variant in ("close_7d_earlier","close_14d_earlier"):
            artifact=aoi/"variants"/variant/"downstream"/"result.parquet"
            _j(aoi/"variants"/variant/"local_downstream_metadata.json",{
                "analysis_id":"prod","status":"pass","artifact_inventory":[record(artifact)]})
    elif stage=="fit":
        artifact=aoi/"model"/"oof_predictions.parquet"
        _j(aoi/"model/model_stage_metadata.json",{
            "analysis_id":"prod","status":"pass","artifact_inventory":[record(artifact)]})
    else:
        artifact=aoi/"compare"/"paired_bootstrap_summary.csv"
        _j(aoi/"compare/compare_stage_metadata.json",{
            "analysis_id":"prod","status":"pass","output_artifacts":[record(artifact)]})


@pytest.mark.parametrize("stage",["local-downstream","fit","compare"])
def test_missing_inner_stage_metadata_disables_resume(tmp_path,stage):
    calls=[];root=tmp_path/"window_closure_region/bejis_2022/id";root.mkdir(parents=True)
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=_pass_runner(calls))
    detail=engine.run_stage(stage,root,{"aoi":"bejis_2022","resume_requested":True})
    assert calls[0]["resume"] is False and detail["inner_resume"] is False


@pytest.mark.parametrize("stage",["local-downstream","fit","compare"])
def test_complete_verified_inner_stage_metadata_enables_recovery_resume(tmp_path,stage):
    calls=[];root=tmp_path/"window_closure_region/bejis_2022/id";root.mkdir(parents=True)
    _write_verified_inner_metadata(root,stage)
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=_pass_runner(calls))
    detail=engine.run_stage(stage,root,{"aoi":"bejis_2022","resume_requested":True})
    assert calls[0]["resume"] is True and detail["inner_resume"] is True


def test_outer_resume_reuses_pass_plan_export_before_missing_local_stage(tmp_path):
    from src.multi_region_window_closure.driver import run_regional_actual
    from src.multi_region_window_closure.synthetic import SyntheticRegionalEngine
    output=tmp_path/"regional"
    first=run_regional_actual(
        aoi="bejis_2022",analysis_id="id",output_root=output,
        engine=SyntheticRegionalEngine(),config_hash="c",input_hash="i",
        execute_actual=True,
    )
    root=Path(first["namespace"])
    for stage in ("local-downstream","fit","compare","summarize","validate"):
        (root/"stages"/f"{stage}.json").unlink(missing_ok=True)
    calls=[]
    class StopBeforeSideEffects(RuntimeError):pass
    def runner(**kwargs):
        calls.append(kwargs)
        raise StopBeforeSideEffects("mock production boundary")
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=runner)
    with pytest.raises(StopBeforeSideEffects):
        run_regional_actual(
            aoi="bejis_2022",analysis_id="id",output_root=output,
            engine=engine,config_hash="c",input_hash="i",resume=True,
            execute_actual=True,
        )
    assert [(call["from_stage"],call["to_stage"]) for call in calls]==[("local-downstream","local-downstream")]
    assert calls[0]["resume"] is False
    assert calls[0]["recover_partial_local_downstream"] is True

def test_stage_output_mutation_invalidates_resume(tmp_path):
    import hashlib
    from src.multi_region_window_closure.driver import _inventory_hash,_state_outputs_match
    path=tmp_path/"metrics.csv";path.write_text("before")
    digest=hashlib.sha256(path.read_bytes()).hexdigest();state={"stage_output_inventory":["metrics.csv"],"stage_output_hash":_inventory_hash({"metrics.csv":digest})}
    assert _state_outputs_match(tmp_path,state)
    path.write_text("after");assert not _state_outputs_match(tmp_path,state)

def test_required_null_summary_fails_normalization():
    frame=_production_summary_shape();frame.loc[0,"point_delta"]=None
    with pytest.raises((MultiRegionWindowClosureError,TypeError,ValueError)):normalize_bootstrap_summary(frame,aoi="bejis_2022",source_analysis_id="id")

def test_regional_exact_column_and_placeholder_export_fail(tmp_path):
    from src.multi_region_window_closure.synthetic import build_regional_fixture
    from src.multi_region_window_closure.validation import evaluate_regional
    root=build_regional_fixture(tmp_path/"region","bejis_2022","id")
    metrics=pd.read_csv(root/"metrics.csv").drop(columns=["estimate"]);metrics.to_csv(root/"metrics.csv",index=False)
    assert next(x for x in evaluate_regional(root,"bejis_2022")["checks"] if x["check_id"]=="REG-ARTEFACT-PRESENCE")["status"]=="FAIL"
    root=build_regional_fixture(tmp_path/"placeholder","bejis_2022","id2")
    pd.DataFrame([{"aoi":"bejis_2022","variant":v,"artifact_id":v} for v in ("canonical","close_7d_earlier","close_14d_earlier")]).to_csv(root/"export_plan.csv",index=False)
    assert next(x for x in evaluate_regional(root,"bejis_2022")["checks"] if x["check_id"]=="REG-ARTEFACT-PRESENCE")["status"]=="FAIL"

def test_stale_validator_pass_is_not_reusable_or_loadable(tmp_path):
    from src.multi_region_window_closure.synthetic import build_regional_fixture,finalize_regional_manifest
    from src.multi_region_window_closure.validation import evaluate_regional
    from src.multi_region_window_closure.driver import _validate_reusable
    from src.multi_region_window_closure.execution import _read_input_record
    root=build_regional_fixture(tmp_path/"region","bejis_2022","id");evaluate_regional(root,"bejis_2022",True)
    old=json.loads((root/"validator_summary.json").read_text())["validated_manifest_sha256"]
    report=root/"report.md";report.write_text(report.read_text()+"\nDescriptive update.")
    finalize_regional_manifest(root,"id","bejis_2022");current=(root/"manifest.sha256").read_text().strip();assert current!=old
    state={"analysis_id":"id","stage_output_inventory":[],"stage_output_hash":__import__("hashlib").sha256(b"{}").hexdigest()}
    assert not _validate_reusable(root,state)
    assert _read_input_record("bejis_2022",root)["validator_manifest_matches"] is False

def test_export_mutation_reruns_entire_downstream_chain(tmp_path):
    from src.multi_region_window_closure.driver import run_regional_actual
    from src.multi_region_window_closure.synthetic import SyntheticRegionalEngine
    class Engine(SyntheticRegionalEngine):
        def __init__(self):self.counts={}
        def run_stage(self,stage,root,context):
            result=super().run_stage(stage,root,context);self.counts[stage]=self.counts.get(stage,0)+1
            if stage!="summarize":(root/f"{stage}.marker").write_text(f"{stage}-{self.counts[stage]}")
            return result
    engine=Engine();out=tmp_path/"out"
    first=run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=out,engine=engine,config_hash="c",input_hash="i",execute_actual=True)
    root=Path(first["namespace"])
    import hashlib
    from src.multi_region_window_closure.driver import _inventory_hash
    upstream=None
    for stage in ("plan","export"):
        state_path=root/"stages"/f"{stage}.json";state=json.loads(state_path.read_text());rel=f"{stage}.marker";digest=hashlib.sha256((root/rel).read_bytes()).hexdigest();state.update(upstream_stage_output_hash=upstream,stage_output_inventory=[rel],stage_output_hash=_inventory_hash({rel:digest}));state_path.write_text(json.dumps(state));upstream=state["stage_output_hash"]
    (root/"export.marker").write_text("mutated")
    resumed=run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=out,engine=engine,config_hash="c",input_hash="i",resume=True,execute_actual=True)
    assert resumed["reused_stages"]==["plan"]
    assert resumed["ran_stages"]==["export","local-downstream","fit","compare","summarize","validate"]

def test_removed_review_package_builder_is_not_a_production_dependency():
    """The review ZIP builder went away with the multi-AOI/synthesis cleanup.

    Regional production must neither import it nor expect a review archive to
    be produced, so its absence is the contract now.
    """
    import importlib.util
    from src.multi_region_window_closure import production
    assert importlib.util.find_spec("scripts.build_window_closure_review_package") is None
    for module in sorted(Path(production.__file__).parent.glob("*.py")):
        assert "build_window_closure_review_package" not in module.read_text(encoding="utf-8")

def _real_plan_result(**updates):
    result={"ran":True,"dry_run":False,"status":"pass","experiment_id":"bejis_2022","analysis_id":"production-analysis","stages_run":["plan"],"prerequisites_ready":True,"missing_required_inputs":[],"files_written":["config/preregistration.json"],"files_written_count":1,"plan":{"files_written":["config/preregistration.json"],"files_written_count":1,"reused":False}}
    result.update(updates);return result

def test_real_production_pass_shape_is_accepted(tmp_path):
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=lambda **kwargs:_real_plan_result())
    root=tmp_path/"window_closure_region/bejis_2022/id";root.mkdir(parents=True)
    detail=engine.run_stage("plan",root,{"aoi":"bejis_2022","resume_requested":False})
    assert detail["resolved_status"]=="pass" and detail["resolved_experiment"]=="bejis_2022" and detail["production_analysis_id"]=="production-analysis"

def test_real_production_blocked_shape_preserves_diagnostics():
    result=_real_plan_result(status="blocked",blockers=["MISSING_FROZEN_INPUT"],warnings=["inventory incomplete"],failure_reason="prerequisite gate failed")
    with pytest.raises(RuntimeError) as caught:resolve_production_stage_result(result,stage="plan",requested_aoi="bejis_2022")
    text=str(caught.value);assert "resolved_status='blocked'" in text and "MISSING_FROZEN_INPUT" in text and "inventory incomplete" in text and "prerequisite gate failed" in text

def test_missing_status_reports_sorted_keys():
    result={"experiment_id":"bejis_2022","analysis_id":"a","files_written":[]}
    with pytest.raises(RuntimeError) as caught:resolve_production_stage_result(result,stage="plan",requested_aoi="bejis_2022")
    text=str(caught.value);assert "resolved_status=None" in text and "keys=['analysis_id', 'experiment_id', 'files_written']" in text

def test_production_result_aoi_mismatch_is_rejected():
    with pytest.raises(RuntimeError,match="AOI identity mismatch"):resolve_production_stage_result(_real_plan_result(experiment_id="mugla_2021"),stage="plan",requested_aoi="bejis_2022")

def test_plan_failure_reason_is_preserved_in_stage_state(tmp_path):
    from src.multi_region_window_closure.driver import run_regional_actual
    engine=ProductionRegionalEngine(aoi="bejis_2022",runner=lambda **kwargs:{"experiment_id":"bejis_2022","analysis_id":"a","blockers":["PLAN_BLOCKED"]})
    with pytest.raises(RuntimeError,match="PLAN_BLOCKED"):run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=tmp_path,engine=engine,config_hash="c",input_hash="i",execute_actual=True)
    state=json.loads((tmp_path/"bejis_2022/id/stages/plan.json").read_text())
    assert "resolved_status=None" in state["failure_reason"] and "PLAN_BLOCKED" in state["failure_reason"] and "keys=" in state["failure_reason"]


def _canonical_inventory_fixture(tmp_path, monkeypatch):
    import hashlib
    from src.multi_region_window_closure import inputs
    path = tmp_path / "step8a_500m_modeling_dataset.parquet"
    path.write_bytes(b"frozen canonical bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(inputs.CANONICAL_STEP8A_SHA256, "bejis_2022", digest)
    monkeypatch.setattr(inputs, "canonical_step8a_path", lambda aoi: path)
    flat = {"canonical_step8a": {"path": str(path), "sha256": digest}}
    persisted = {
        "experiment_id": "bejis_2022",
        "read_only": True,
        "frozen_input_sha256": {"canonical_step8a": digest},
        "inventory": {"canonical_step8a": {"exists": True, "path": str(path), "sha256": digest}},
    }
    return path, digest, flat, persisted


def test_canonical_inventory_resolver_supports_only_verified_planning_and_persisted_shapes(tmp_path, monkeypatch):
    from src.multi_region_window_closure.inputs import resolve_canonical_step8a_record
    path, digest, flat, persisted = _canonical_inventory_fixture(tmp_path, monkeypatch)
    planning = resolve_canonical_step8a_record(flat, aoi="bejis_2022")
    actual = resolve_canonical_step8a_record(persisted, aoi="bejis_2022")
    assert planning["source_schema"] == "planning_flat"
    assert actual["source_schema"] == "production_persisted"
    assert (planning["path"], planning["sha256"]) == (actual["path"], actual["sha256"]) == (str(path), digest)


@pytest.mark.parametrize("mutation,match", [
    (lambda flat, persisted: {}, "exactly one"),
    (lambda flat, persisted: {**persisted, **flat}, "exactly one"),
    (lambda flat, persisted: {"canonical_step8a": {"path": flat["canonical_step8a"]["path"]}}, "SHA-256 is missing"),
    (lambda flat, persisted: {"canonical_step8a": {"sha256": flat["canonical_step8a"]["sha256"]}}, "path is missing"),
    (lambda flat, persisted: {**persisted, "experiment_id": "mugla_2021"}, "AOI mismatch"),
])
def test_canonical_inventory_resolver_fails_closed_for_bad_shapes(tmp_path, monkeypatch, mutation, match):
    from src.multi_region_window_closure.inputs import resolve_canonical_step8a_record
    _, _, flat, persisted = _canonical_inventory_fixture(tmp_path, monkeypatch)
    with pytest.raises(MultiRegionWindowClosureError, match=match):
        resolve_canonical_step8a_record(mutation(flat, persisted), aoi="bejis_2022")


def test_canonical_inventory_resolver_rejects_anchor_and_byte_drift(tmp_path, monkeypatch):
    from src.multi_region_window_closure import inputs
    path, digest, flat, _ = _canonical_inventory_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(inputs.CANONICAL_STEP8A_SHA256, "bejis_2022", "0" * 64)
    with pytest.raises(MultiRegionWindowClosureError, match="central anchor mismatch"):
        inputs.resolve_canonical_step8a_record(flat, aoi="bejis_2022")
    monkeypatch.setitem(inputs.CANONICAL_STEP8A_SHA256, "bejis_2022", digest)
    path.write_bytes(b"drift")
    with pytest.raises(MultiRegionWindowClosureError, match="actual-byte hash drift"):
        inputs.resolve_canonical_step8a_record(flat, aoi="bejis_2022")


def test_persisted_canonical_inventory_rejects_mirror_mismatch_and_exists_false(tmp_path, monkeypatch):
    from src.multi_region_window_closure.inputs import resolve_canonical_step8a_record
    _, _, _, persisted = _canonical_inventory_fixture(tmp_path, monkeypatch)
    persisted["frozen_input_sha256"]["canonical_step8a"] = "0" * 64
    with pytest.raises(MultiRegionWindowClosureError, match="nested/mirror"):
        resolve_canonical_step8a_record(persisted, aoi="bejis_2022")
    _, _, _, persisted = _canonical_inventory_fixture(tmp_path, monkeypatch)
    persisted["inventory"]["canonical_step8a"]["exists"] = False
    with pytest.raises(MultiRegionWindowClosureError, match="exists must be true"):
        resolve_canonical_step8a_record(persisted, aoi="bejis_2022")


def test_canonical_config_representation_mismatch_fails_closed(tmp_path, monkeypatch):
    from src.multi_region_window_closure.production import _assert_canonical_hash_representations
    _, digest, _, _ = _canonical_inventory_fixture(tmp_path, monkeypatch)
    with pytest.raises(MultiRegionWindowClosureError, match="CONFIG_MISMATCH"):
        _assert_canonical_hash_representations(
            {"frozen_input_sha256": {"canonical_step8a": "0" * 64}},
            aoi="bejis_2022", resolved_sha256=digest, label="fixture",
        )


def test_outer_resume_reuses_all_expensive_stages_and_runs_only_summarize_validate(tmp_path):
    from src.multi_region_window_closure.driver import run_regional_actual
    from src.multi_region_window_closure.synthetic import SyntheticRegionalEngine
    output = tmp_path / "regional"
    first = run_regional_actual(
        aoi="bejis_2022", analysis_id="id", output_root=output,
        engine=SyntheticRegionalEngine(), config_hash="c", input_hash="i",
        execute_actual=True,
    )
    root = Path(first["namespace"])
    (root / "stages" / "summarize.json").unlink()
    (root / "stages" / "validate.json").unlink()

    class NoExpensiveCalls(SyntheticRegionalEngine):
        def __init__(self): self.calls = []
        def run_stage(self, stage, root, context):
            self.calls.append(stage)
            if stage != "summarize":
                raise AssertionError(f"expensive stage unexpectedly called: {stage}")
            return super().run_stage(stage, root, context)

    engine = NoExpensiveCalls()
    resumed = run_regional_actual(
        aoi="bejis_2022", analysis_id="id", output_root=output,
        engine=engine, config_hash="c", input_hash="i", resume=True,
        execute_actual=True,
    )
    assert resumed["reused_stages"] == ["plan", "export", "local-downstream", "fit", "compare"]
    assert resumed["ran_stages"] == ["summarize", "validate"]
    assert engine.calls == ["summarize"]


def test_summarize_accepts_actual_persisted_inventory_shape(tmp_path, monkeypatch):
    """Small persisted-output fixture; no runner, GEE, fit, or bootstrap call."""
    import hashlib
    import numpy as np
    from src.multi_region_window_closure import inputs, plan
    from src.multi_region_window_closure.production import normalize_production_regional_outputs

    root = tmp_path / "region"
    prod = root / "_production" / "bejis_2022"
    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"canonical")
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    monkeypatch.setitem(inputs.CANONICAL_STEP8A_SHA256, "bejis_2022", digest)
    monkeypatch.setattr(inputs, "canonical_step8a_path", lambda aoi: canonical)
    monkeypatch.setattr(plan, "export_plan_rows", lambda *args, **kwargs: [
        {"aoi": "bejis_2022", "variant": variant, "artifact_id": variant}
        for variant in ("canonical", "close_7d_earlier", "close_14d_earlier")
    ])
    _j(prod / "config/preregistration.json", {
        "experiment_id": "bejis_2022",
        "frozen_input_sha256": {"canonical_step8a": digest},
    })
    _j(prod / "config/frozen_input_inventory.json", {
        "experiment_id": "bejis_2022", "read_only": True,
        "frozen_input_sha256": {"canonical_step8a": digest},
        "inventory": {"canonical_step8a": {
            "exists": True, "path": str(canonical), "sha256": digest,
        }},
    })
    cohort = pd.DataFrame({
        "cell_id": ["a", "b"], "burned": [0, 1], "feature": [1.0, 2.0],
    })
    folds = pd.DataFrame({
        "cell_id": ["a", "b"], "spatial_block_id": [0, 1], "fold_id": [0, 1],
    })
    cohort_path = prod / "model/common_cohort/common_cohort.parquet"
    folds_path = prod / "model/shared_folds/shared_spatial_folds.parquet"
    cohort_path.parent.mkdir(parents=True, exist_ok=True); cohort.to_parquet(cohort_path, index=False)
    folds_path.parent.mkdir(parents=True, exist_ok=True); folds.to_parquet(folds_path, index=False)
    # The cohort accounting the production model stage records; summarize reads
    # it back instead of publishing placeholder zeros.
    _j(prod / "model/common_cohort/common_cohort_metadata.json", {
        "final_common_cohort_rows": len(cohort),
        "initial_rows_by_variant": {v: 5 for v in _VARIANTS},
        "removed_not_valid_for_modeling": {v: 1 for v in _VARIANTS},
        "removed_outside_primary_population": {v: 1 for v in _VARIANTS},
        "removed_prelabel_censor": {v: 1 for v in _VARIANTS},
        "removed_missing_required_feature_union": {v: 0 for v in _VARIANTS},
        "removed_variant_only_keys": {v: 0 for v in _VARIANTS},
        "removed_label_mismatch": 0,
        "removed_static_invariance_failure": 0,
    })
    # The fixed month-filter clipping each shifted variant actually exported.
    for shifted, clipped in (("close_7d_earlier", 7), ("close_14d_earlier", 14)):
        _j(prod / "variants" / shifted / "predictor_export_metadata.json", {
            "artifact_inventory": [
                {
                    "role": role,
                    "date_semantics": {
                        "duration_days": 60,
                        "calendar_month_filter_transparency": {
                            "calendar_month_filter": "6-9",
                            "clipped_day_count": clipped,
                            "effective_included_day_count": 60 - clipped,
                        },
                    },
                }
                for role in (
                    "modis_lst_mean", "modis_lst_std", "modis_valid_observation_count",
                )
            ],
        })
    metric_rows = []
    for variant in ("canonical", "close_7d_earlier", "close_14d_earlier"):
        for model in ("baseline", "thermal"):
            metric_rows.append({
                "variant_id": variant, "model_family": model, "row_count": 2,
                "positive_count": 1, "negative_count": 1,
                "roc_auc": .7, "pr_auc": .6, "brier": .2,
            })
            oof = folds.assign(variant_id=variant, model_family=model, y_true=[0, 1], y_score=[.2, .8])
            oof_path = prod / "model/variants" / variant / model / "oof_predictions.parquet"
            oof_path.parent.mkdir(parents=True, exist_ok=True); oof.to_parquet(oof_path, index=False)
    metrics_path = prod / "model/metrics/point_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    summary = _production_summary_shape()
    bootstrap = prod / "model/bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    summary.to_csv(bootstrap / "paired_bootstrap_summary.csv", index=False)
    replicate_columns = {
        f"{variant}__{model}_{metric}": np.array([.1, .2])
        for variant in ("canonical", "close_7d_earlier", "close_14d_earlier")
        for model in ("baseline", "thermal")
        for metric in ("roc_auc", "pr_auc", "brier")
    }
    pd.DataFrame(replicate_columns).to_parquet(
        bootstrap / "paired_bootstrap_replicates.parquet", index=False,
    )
    report = prod / "compare/report/window_closure_comparison.md"
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(
        "This is an observational predictive analysis. It does not establish a causal mechanism."
    )

    result = normalize_production_regional_outputs(
        root, {"aoi": "bejis_2022", "analysis_id": "id"},
    )
    assert result["normalized_from"] == str(prod)
    assert json.loads((root / "config.json").read_text())["canonical_hash"] == digest
    dates = pd.read_csv(root / "window_dates.csv")
    assert dates.analysis_id.tolist() == ["id", "id", "id"]
    assert "causal" not in (root / "report.md").read_text().lower()

    # The real removal accounting reaches cohort_inventory.csv unchanged...
    inventory = pd.read_csv(root / "cohort_inventory.csv").set_index("variant")
    assert inventory.loc["canonical", "initial_rows"] == 5
    assert inventory.loc["canonical", "removed_prelabel_censor"] == 1
    assert inventory.loc["canonical", "final_common_cohort_rows"] == len(cohort)
    # ...and the exported clipping reaches regional_summary.csv unchanged.
    regional = pd.read_csv(root / "regional_summary.csv")
    assert regional.loc[0, "modis_clipped_days_7d"] == 7
    assert regional.loc[0, "modis_clipped_days_14d"] == 14


def test_failed_validator_resume_reuses_upstream_and_repairs_only_summarize_validate(tmp_path):
    from src.multi_region_window_closure.driver import run_regional_actual
    from src.multi_region_window_closure.synthetic import build_regional_fixture, finalize_regional_manifest
    from src.multi_region_window_closure.validation import evaluate_regional

    class FixtureEngine:
        def __init__(self): self.calls=[];self.network_calls=0;self.gee_calls=0;self.export_calls=0;self.model_calls=0;self.bootstrap_calls=0
        def run_stage(self,stage,root,context):
            self.calls.append(stage)
            if stage=="summarize":build_regional_fixture(root,context["aoi"],context["analysis_id"],with_states=False)
            return {"stage":stage}
        def finalize_manifest(self,root,context):finalize_regional_manifest(root,context["analysis_id"],context["aoi"])

    output=tmp_path/"regional";first_engine=FixtureEngine()
    first=run_regional_actual(aoi="bejis_2022",analysis_id="runtime-id",output_root=output,engine=first_engine,config_hash="c",input_hash="i",execute_actual=True)
    root=Path(first["namespace"])
    dates=pd.read_csv(root/"window_dates.csv").drop(columns=["analysis_id"]);dates.to_csv(root/"window_dates.csv",index=False)
    (root/"report.md").write_text("This establishes a causal effect.",encoding="utf-8")
    failed=evaluate_regional(root,"bejis_2022",write_results=True)
    assert failed["overall_status"]=="FAIL"
    _j(root/"stages/validate.json",{"aoi":"bejis_2022","analysis_id":"runtime-id","config_hash":"c","input_hash":"i","stage":"validate","status":"FAIL","resume_eligible":False})

    recovery=FixtureEngine()
    resumed=run_regional_actual(aoi="bejis_2022",analysis_id="runtime-id",output_root=output,engine=recovery,config_hash="c",input_hash="i",resume=True,execute_actual=True)
    assert resumed["reused_stages"]==["plan","export","local-downstream","fit","compare"]
    assert resumed["ran_stages"]==["summarize","validate"] and recovery.calls==["summarize"]
    assert (recovery.network_calls,recovery.gee_calls,recovery.export_calls,recovery.model_calls,recovery.bootstrap_calls)==(0,0,0,0,0)
    final=evaluate_regional(root,"bejis_2022",False,require_final_status=True)
    assert final["overall_status"]=="PASS" and all(r["status"]=="PASS" for r in final["checks"])
