"""Synthetic regional actual execution, mutation, schema and manifest tests."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
from src.multi_region_window_closure.bootstrap import SERIES_PER_AOI
from src.multi_region_window_closure.contract import ACTUAL_AOIS,SYNTHESIS_AOIS
from src.multi_region_window_closure.driver import run_regional_actual
from src.multi_region_window_closure.schema import REGIONAL_ARTIFACT_SPECS,SYNTHESIS_ARTIFACT_SPECS
from src.multi_region_window_closure.synthetic import SyntheticRegionalEngine,build_regional_fixture
from src.multi_region_window_closure.validation import REGIONAL_EVALUATORS,SYNTHESIS_EVALUATORS,evaluate_regional
from src.multi_region_window_closure.validators import REGIONAL_CHECKS,SYNTHESIS_CHECKS

def _json_mutate(path,fn):
 d=json.loads(path.read_text());fn(d);path.write_text(json.dumps(d,indent=2),encoding="utf-8")
def _csv_mutate(path,fn):
 d=pd.read_csv(path);d=fn(d);d.to_csv(path,index=False)
def _pq_mutate(path,fn):
 d=pd.read_parquet(path);d=fn(d);d.to_parquet(path,index=False)

def _mutate(root,name):
 if name=="canonical_hash":_json_mutate(root/"config.json",lambda d:d.update(canonical_hash="0"*64))
 elif name=="window_day":_csv_mutate(root/"window_dates.csv",lambda d:d.assign(predictor_start=d.predictor_start.mask(d.index==1,"2022-06-23")))
 elif name=="label_day":_csv_mutate(root/"window_dates.csv",lambda d:d.assign(label_start=d.label_start.mask(d.index==1,"2022-07-02")))
 elif name=="cohort_missing":_csv_mutate(root/"cohort_inventory.csv",lambda d:d.iloc[:-1])
 elif name=="fold_variant":_csv_mutate(root/"cohort_inventory.csv",lambda d:d.assign(fold_mapping_hash=d.fold_mapping_hash.mask(d.index==1,"0"*64)))
 elif name=="fold_class":_pq_mutate(root/"fold_mapping.parquet",lambda d:d.assign(y_true=d.y_true.mask(d.fold_id==0,0)))
 elif name=="primary_fit":_json_mutate(root/"config.json",lambda d:d["fit_accounting"].update(completed_primary_estimator_fits=29))
 elif name=="aux_fit":_json_mutate(root/"config.json",lambda d:d["fit_accounting"].update(completed_auxiliary_downscaling_fits=1))
 elif name=="oof_duplicate":_pq_mutate(root/"oof_predictions.parquet",lambda d:pd.concat([d,d.iloc[[0]]],ignore_index=True))
 elif name=="oof_missing":_pq_mutate(root/"oof_predictions.parquet",lambda d:d.iloc[:-1])
 elif name=="metric":_csv_mutate(root/"metrics.csv",lambda d:d.assign(estimate=d.estimate.mask(d.index==0,d.estimate.iloc[0]-.1)))
 elif name=="brier_sign":_csv_mutate(root/"bootstrap_summary.csv",lambda d:d.assign(point_estimate_oriented=d.point_estimate_oriented.mask(d.metric=="brier",d.point_estimate_natural)))
 elif name=="brier_ci":_csv_mutate(root/"bootstrap_summary.csv",lambda d:d.assign(ci_low_oriented=d.ci_low_oriented.mask(d.metric=="brier",d.ci_low_natural)))
 elif name=="rep_duplicate":_pq_mutate(root/"bootstrap_replicates.parquet",lambda d:pd.concat([d,d.iloc[[0]]],ignore_index=True))
 elif name=="rep_accounting":_csv_mutate(root/"bootstrap_summary.csv",lambda d:d.assign(valid_replicates=d.valid_replicates.mask(d.index==0,999)))
 elif name=="wording":(root/"report.md").write_text((root/"report.md").read_text()+" statistically significant",encoding="utf-8")
 elif name=="evia_framing":(root/"report.md").write_text("Descriptive uncertainty remains for evia_2021_extended.",encoding="utf-8")
 elif name=="manifest_hash":_json_mutate(root/"manifest.json",lambda d:d["files"][0].update(sha256="0"*64))
 elif name=="stage_skip":_json_mutate(root/"stages/fit.json",lambda d:d.update(status="SKIP"))
 elif name=="cross_aoi":_csv_mutate(root/"metrics.csv",lambda d:pd.concat([d,pd.DataFrame([{**d.iloc[0].to_dict(),"aoi":"mugla_2021"}])],ignore_index=True))

@pytest.mark.parametrize("mutation,expected,aoi",[
 ("canonical_hash","REG-DATE-CANONICAL","bejis_2022"),("window_day","REG-DATE-SHIFTS","bejis_2022"),("label_day","REG-DATE-LABEL","bejis_2022"),("cohort_missing","REG-COHORT-COMMON","bejis_2022"),("fold_variant","REG-FOLD-COMMON","bejis_2022"),("fold_class","REG-FOLD-CLASSES","bejis_2022"),("primary_fit","REG-FIT-PRIMARY-COUNT","bejis_2022"),("aux_fit","REG-FIT-AUXILIARY-COUNT","bejis_2022"),("oof_duplicate","REG-OOF-COMPLETE","bejis_2022"),("oof_missing","REG-OOF-COMPLETE","bejis_2022"),("metric","REG-METRIC-ARITHMETIC","bejis_2022"),("brier_sign","REG-METRIC-BRIER-ORIENTATION","bejis_2022"),("brier_ci","REG-METRIC-BRIER-ORIENTATION","bejis_2022"),("rep_duplicate","REG-BOOT-ACCOUNTING","bejis_2022"),("rep_accounting","REG-BOOT-ACCOUNTING","bejis_2022"),("wording","REG-REPORT-WORDING","bejis_2022"),("evia_framing","REG-REPORT-EVIA","evia_2021_extended"),("manifest_hash","REG-MANIFEST-COMPLETE","bejis_2022"),("stage_skip","REG-STAGE-COMPLETE","bejis_2022"),("cross_aoi","REG-SCOPE-NO-CROSS-AOI","bejis_2022")])
def test_regional_mutations_fail_expected_check(tmp_path,mutation,expected,aoi):
 root=build_regional_fixture(tmp_path/aoi,aoi,"id");_mutate(root,mutation);result=evaluate_regional(root,aoi,False);records={r["check_id"]:r for r in result["checks"]};assert records[expected]["status"]=="FAIL";assert result["overall_status"]=="FAIL"

@pytest.mark.parametrize("aoi",ACTUAL_AOIS)
def test_synthetic_complete_regional_namespace_passes(tmp_path,aoi):
 root=build_regional_fixture(tmp_path/aoi,aoi,"id");result=evaluate_regional(root,aoi,True);assert result["overall_status"]=="PASS";assert result["required_skip"]==0

def test_synthesis_actual_driver_is_not_reachable():
 """The cross-AOI synthesis driver was removed with the per-AOI cleanup."""
 from src.multi_region_window_closure import driver
 assert not hasattr(driver,"run_synthesis_actual")

def test_every_registry_check_has_evaluator():
 assert set(REGIONAL_EVALUATORS)=={c.check_id for c in REGIONAL_CHECKS};assert set(SYNTHESIS_EVALUATORS)=={c.check_id for c in SYNTHESIS_CHECKS}

def test_regional_registry_contains_no_combined_count_language():
 text=" ".join(c.description.lower() for c in REGIONAL_CHECKS);assert not any(x in text for x in ("three actual aois","six shifted","four canonical","90 primary","6 auxiliary","96 total","81 bootstrap","81,000"))

def test_schema_row_formulas_are_execution_grain_specific():
 r={s.relative_path:s for s in REGIONAL_ARTIFACT_SPECS};s={x.relative_path:x for x in SYNTHESIS_ARTIFACT_SPECS};assert r["window_dates.csv"].expected_rows()==3;assert r["metrics.csv"].expected_rows()==18;assert r["bootstrap_summary.csv"].expected_rows()==27;assert r["oof_predictions.parquet"].expected_rows(10)==60;assert r["bootstrap_replicates.parquet"].expected_rows(1000)==27000;assert s["regional_result_index.csv"].expected_rows()==len(SYNTHESIS_AOIS);assert s["four_region_synthesis.csv"].expected_rows()==len(SYNTHESIS_AOIS)*SERIES_PER_AOI;assert "oof_predictions.parquet" not in s

def test_synthetic_regional_driver_end_to_end_and_resume_binding(tmp_path):
 out=tmp_path/"regional";result=run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=out,engine=SyntheticRegionalEngine(),config_hash="c",input_hash="i",execute_actual=True);assert result["technical_status"]=="PASS";root=Path(result["namespace"]);assert evaluate_regional(root,"bejis_2022",False)["overall_status"]=="PASS"
 reused=run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=out,engine=SyntheticRegionalEngine(),config_hash="c",input_hash="i",resume=True,execute_actual=True);assert set(reused["reused_stages"])==set(("plan","export","local-downstream","fit","compare","summarize","validate"))

def test_final_technical_statuses_are_manifest_bound(tmp_path):
 result=run_regional_actual(aoi="bejis_2022",analysis_id="status-id",output_root=tmp_path/"regional",engine=SyntheticRegionalEngine(),config_hash="c",input_hash="i",execute_actual=True);rroot=Path(result["namespace"])
 assert json.loads((rroot/"summary.json").read_text())["technical_status"]=="PASS"
 assert pd.read_csv(rroot/"regional_summary.csv").technical_status.tolist()==["PASS"]
 rv=json.loads((rroot/"validator_summary.json").read_text());assert rv["validated_manifest_sha256"]==(rroot/"manifest.sha256").read_text().strip()

def test_actual_drivers_require_explicit_guard(tmp_path):
 with pytest.raises(RuntimeError,match="execute-actual"):run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=tmp_path,engine=SyntheticRegionalEngine(),config_hash="c",input_hash="i")

@pytest.mark.parametrize("mutation",["wrong_schema","wrong_id","wrong_hash","missing_file","stray_file"])
def test_regional_manifest_fail_closed(tmp_path,mutation):
 root=build_regional_fixture(tmp_path/"r","bejis_2022","id")
 if mutation=="wrong_schema":_json_mutate(root/"manifest.json",lambda d:d.update(schema_version="legacy"))
 elif mutation=="wrong_id":_json_mutate(root/"manifest.json",lambda d:d.update(analysis_id="other"))
 elif mutation=="wrong_hash":_json_mutate(root/"manifest.json",lambda d:d["files"][0].update(sha256="0"*64))
 elif mutation=="missing_file":(root/"metrics.csv").unlink()
 else:(root/"stray.txt").write_text("x")
 result=evaluate_regional(root,"bejis_2022",False);assert result["overall_status"]=="FAIL";assert any(r["check_id"].startswith("REG-MANIFEST") and r["status"]!="PASS" for r in result["checks"])

def test_driver_resume_drift_and_force_are_aoi_isolated(tmp_path):
 out=tmp_path/"regional";engine=SyntheticRegionalEngine();run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=out,engine=engine,config_hash="c",input_hash="i",execute_actual=True)
 mugla=out/"mugla_2021"/"id";mugla.mkdir(parents=True);(mugla/"sentinel").write_text("m")
 synthesis=tmp_path/"synthesis/sentinel";synthesis.parent.mkdir();synthesis.write_text("s")
 drift=run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=out,engine=engine,config_hash="changed",input_hash="i",resume=True,execute_actual=True);assert "plan" in drift["ran_stages"]
 run_regional_actual(aoi="bejis_2022",analysis_id="id",output_root=out,engine=engine,config_hash="changed",input_hash="i",force=True,execute_actual=True);assert (mugla/"sentinel").read_text()=="m";assert synthesis.read_text()=="s";assert list((out/"bejis_2022/_quarantine").iterdir())

def test_report_wording_guard_rejects_affirmative_and_negating_forbidden_tokens_with_location(tmp_path):
 root=build_regional_fixture(tmp_path/"r","bejis_2022","runtime-id")
 report=root/"report.md";report.write_text("Allowed first line.\nThis establishes a causal effect.",encoding="utf-8")
 result=evaluate_regional(root,"bejis_2022",False);record=next(r for r in result["checks"] if r["check_id"]=="REG-REPORT-WORDING")
 assert record["status"]=="FAIL" and "report.md" in record["evidence"] and "line_number': 2" in record["evidence"] and "causal" in record["evidence"]
 report.write_text("This is non-causal.",encoding="utf-8")
 record=next(r for r in evaluate_regional(root,"bejis_2022",False)["checks"] if r["check_id"]=="REG-REPORT-WORDING")
 assert record["status"]=="FAIL"

@pytest.mark.parametrize("mutation",["missing","null","mismatch"])
def test_window_dates_analysis_identity_is_required(tmp_path,mutation):
 root=build_regional_fixture(tmp_path/"r","bejis_2022","runtime-id");path=root/"window_dates.csv";frame=pd.read_csv(path)
 if mutation=="missing":frame=frame.drop(columns=["analysis_id"])
 elif mutation=="null":frame.loc[1,"analysis_id"]=None
 else:frame.loc[1,"analysis_id"]="other-id"
 frame.to_csv(path,index=False)
 record=next(r for r in evaluate_regional(root,"bejis_2022",False)["checks"] if r["check_id"]=="REG-ARTEFACT-PRESENCE")
 assert record["status"]=="FAIL" and "window_dates.csv" in record["evidence"]

def test_window_dates_exact_runtime_identity_and_final_validator_32_of_32(tmp_path):
 root=build_regional_fixture(tmp_path/"r","bejis_2022","runtime-id")
 dates=pd.read_csv(root/"window_dates.csv")
 from src.multi_region_window_closure.dates import WINDOW_DATE_COLUMNS
 assert tuple(dates.columns)==WINDOW_DATE_COLUMNS and len(dates)==3
 assert dates.analysis_id.tolist()==["runtime-id"]*3 and not dates.duplicated(["analysis_id","aoi","variant"]).any()
 result=evaluate_regional(root,"bejis_2022",False,require_final_status=True)
 assert result["overall_status"]=="PASS" and all(r["status"]=="PASS" for r in result["checks"])

def test_pending_status_passes_prefinal_but_final_requires_pass(tmp_path):
 root=build_regional_fixture(tmp_path/"r","bejis_2022","runtime-id")
 _json_mutate(root/"summary.json",lambda d:d.update(technical_status="PENDING_VALIDATION"))
 _csv_mutate(root/"regional_summary.csv",lambda d:d.assign(technical_status="PENDING_VALIDATION"))
 from src.multi_region_window_closure.synthetic import finalize_regional_manifest
 finalize_regional_manifest(root,"runtime-id","bejis_2022")
 assert evaluate_regional(root,"bejis_2022",False,require_final_status=False)["overall_status"]=="PASS"
 final=evaluate_regional(root,"bejis_2022",False,require_final_status=True)
 assert final["overall_status"]=="FAIL"
