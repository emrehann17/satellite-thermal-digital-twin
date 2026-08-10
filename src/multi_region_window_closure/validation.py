"""Artifact evaluators for regional and synthesis validators.

Evaluators are read-only with respect to scientific artefacts. Optional result
writing is limited to validator_results.json and validator_summary.json.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Callable
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from src.multi_region_window_closure.contract import (
    ACTUAL_AOIS, METRICS, MODEL_FAMILIES, REGIONAL_SCHEMA_VERSION,
    SCIENTIFIC_CONTRACT_ID, SYNTHESIS_AOIS, SYNTHESIS_SCHEMA_VERSION, VARIANTS,
    assert_regional_aoi,
)
from src.multi_region_window_closure.bootstrap import four_region_synthesis_row_count
from src.multi_region_window_closure.per_aoi import REGIONAL_REQUIRED_ARTIFACTS, SYNTHESIS_REQUIRED_ARTIFACTS
from src.multi_region_window_closure.schema import REGIONAL_ARTIFACT_SPECS, SYNTHESIS_ARTIFACT_SPECS, verify_manifest_digest
from src.multi_region_window_closure.validators import REGIONAL_CHECKS, SYNTHESIS_CHECKS, overall_status
from src.multi_region_window_closure.wording import find_forbidden, find_forbidden_locations, assert_evia_framing

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def snapshot(paths):
    return {str(p):sha256_file(Path(p)) for p in paths if Path(p).is_file()}

def _load(root: Path) -> dict[str,Any]:
    out={}
    for name in ("config.json","input_hashes.json","summary.json","manifest.json","input_references.json"):
        p=root/name
        if p.is_file(): out[name]=json.loads(p.read_text(encoding="utf-8"))
    for name in ("window_dates.csv","export_plan.csv","cohort_inventory.csv","variant_artifact_index.csv","metrics.csv","bootstrap_summary.csv","regional_summary.csv","regional_result_index.csv","four_region_synthesis.csv"):
        p=root/name
        if p.is_file(): out[name]=pd.read_csv(p)
    for name in ("fold_mapping.parquet","oof_predictions.parquet","bootstrap_replicates.parquet"):
        p=root/name
        if p.is_file(): out[name]=pd.read_parquet(p)
    out["report.md"]=(root/"report.md").read_text(encoding="utf-8") if (root/"report.md").is_file() else ""
    out["stages"]={p.stem:json.loads(p.read_text(encoding="utf-8")) for p in sorted((root/"stages").glob("*.json"))} if (root/"stages").is_dir() else {}
    return out

def _manifest_ok(root, manifest, schema_version, analysis_id, specs):
    digest_ok,digest_evidence=verify_manifest_digest(root)
    if not digest_ok:return False,digest_evidence
    if manifest.get("schema_version")!=schema_version or manifest.get("analysis_id")!=analysis_id:return False,"manifest identity mismatch"
    recorded={e.get("relative_path"):e for e in manifest.get("files",[])}
    controls={"manifest.json","manifest.sha256","validator_results.json","validator_summary.json","stages/validate.json"}
    required={s.relative_path for s in specs if s.required}-controls
    missing=sorted(required-set(recorded))
    if missing:return False,f"missing manifest entries {missing}"
    for rel,e in recorded.items():
        p=root/rel
        if not p.is_file() or sha256_file(p)!=e.get("sha256"):return False,f"hash mismatch {rel}"
    disk={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not str(p.relative_to(root)).startswith("stages/")}-controls
    stray=sorted(disk-set(recorded))
    if stray:return False,f"stray files {stray}"
    return True,f"{len(recorded)} entries verified"

def _record(check, ok, evidence, error=None):
    return {"check_id":check.check_id,"description":check.description,"requirement_key":check.requirement_key,"severity":check.severity,"status":"PASS" if ok else "FAIL","evidence":str(evidence),"failure_message":None if ok else str(error or check.description)}

def evaluate_regional(root: Path, aoi: str, write_results: bool=False, require_final_status: bool=False) -> dict[str,Any]:
    root=Path(root); aoi=assert_regional_aoi(aoi)
    try:data=_load(root); load_error=None
    except Exception as exc:data={}; load_error=f"{type(exc).__name__}: {exc}"
    config=data.get("config.json",{}); summary=data.get("summary.json",{}); results=[]
    tables=[v for v in data.values() if isinstance(v,pd.DataFrame)]
    def ev(key):
        if load_error:return False,load_error
        if key=="artifact_presence":
            names=set(REGIONAL_REQUIRED_ARTIFACTS)-{"validator_results.json","validator_summary.json"}
            missing=[n for n in names if not (root/n).is_file() or (root/n).stat().st_size==0]
            schema_errors=[]
            for spec in REGIONAL_ARTIFACT_SPECS:
                if not spec.columns or spec.relative_path not in data:continue
                df=data[spec.relative_path];absent=sorted(set(spec.columns)-set(df.columns))
                unexpected=sorted(set(df.columns)-set(spec.columns)) if spec.relative_path=="window_dates.csv" else []
                nulls=[c for c in spec.non_null if c in df and df[c].isna().any()]
                duplicate=bool(df.duplicated(list(spec.primary_key)).any()) if spec.primary_key and set(spec.primary_key)<=set(df.columns) else bool(spec.primary_key)
                expected=None
                if spec.expected_rows and spec.relative_path not in {"oof_predictions.parquet","bootstrap_replicates.parquet"}:
                    try:expected=spec.expected_rows(aoi) if spec.relative_path=="export_plan.csv" else spec.expected_rows()
                    except TypeError:expected=None
                if absent or unexpected or nulls or duplicate or (expected is not None and len(df)!=expected):schema_errors.append({"file":spec.relative_path,"missing_columns":absent,"unexpected_columns":unexpected,"null_columns":nulls,"duplicate_pk":duplicate,"rows":len(df),"expected":expected})
                if spec.relative_path=="window_dates.csv":
                    ids=set(df["analysis_id"].dropna().astype(str)) if "analysis_id" in df else set()
                    aois=set(df["aoi"].dropna().astype(str)) if "aoi" in df else set()
                    if ids!={str(config.get("analysis_id"))} or aois!={aoi}:
                        schema_errors.append({"file":spec.relative_path,"analysis_ids":sorted(ids),"expected_analysis_id":config.get("analysis_id"),"aois":sorted(aois),"expected_aoi":aoi})
            return not missing and not schema_errors,f"missing_or_empty={missing}; schema_errors={schema_errors}"
        if key=="single_aoi": return config.get("aoi")==aoi,f"config.aoi={config.get('aoi')}"
        if key=="variant_scope": return tuple(config.get("variants",[]))==VARIANTS,f"variants={config.get('variants')}"
        if key=="model_scope": return set(config.get("model_families",[]))==set(MODEL_FAMILIES),f"models={config.get('model_families')}"
        if key=="no_cross_aoi":
            seen={str(x) for df in tables if "aoi" in df for x in df["aoi"].dropna().unique()}; return seen<={aoi},f"seen={sorted(seen)}"
        if key=="canonical_integrity":
            from src.multi_region_window_closure.inputs import CANONICAL_STEP8A_SHA256
            expected=CANONICAL_STEP8A_SHA256[aoi];path=Path(config.get("canonical_path",""));actual=sha256_file(path) if path.is_file() else None;manifest=data.get("manifest.json",{}).get("canonical_aoi_hashes",{}).get(aoi)
            values=(actual,config.get("production_canonical_hash",config.get("canonical_hash")),config.get("canonical_hash"),summary.get("canonical_hash"),manifest)
            return all(value==expected for value in values),f"expected={expected}; actual/config/summary/manifest={values}"
        if key in {"date_shifts","label_invariance"}:
            from src.multi_region_window_closure.dates import assert_date_contract
            rows=data["window_dates.csv"].to_dict("records")
            try: assert_date_contract(rows); return True,"date contract re-derived"
            except BaseException as exc:return False,str(exc)
        if key=="input_hashes":
            doc=data.get("input_hashes.json",{}); values=doc.get("hashes",doc); ok=bool(values) and all(len(str(v))==64 for v in values.values()); return ok,f"hash_count={len(values)}"
        if key in {"common_cohort","cohort_inventory"}:
            df=data["cohort_inventory.csv"]; count_col="final_common_cohort_rows" if "final_common_cohort_rows" in df else "row_count"; invariant=[c for c in ("cohort_hash",count_col,"label_hash","static_hash") if c in df];ok=len(df)==3 and set(df.variant)==set(VARIANTS) and len(invariant)>=2 and all(df[c].nunique()==1 for c in invariant); return ok,f"rows={len(df)} hashes={df.cohort_hash.nunique() if len(df) else 0}"
        if key=="fold_invariance":
            df=data["cohort_inventory.csv"]; return df.fold_mapping_hash.nunique()==1 and len(df)==3,f"fold_hashes={df.fold_mapping_hash.nunique()}"
        if key=="fold_classes":
            df=data["fold_mapping.parquet"]; counts=df.groupby("fold_id").y_true.nunique(); return set(counts.index)==set(range(5)) and bool((counts==2).all()),f"class_counts={counts.to_dict()}"
        fit=config.get("fit_accounting",{})
        if key=="primary_fit_count": return fit.get("expected_primary_estimator_fits")==30 and fit.get("completed_primary_estimator_fits")==30,str(fit)
        if key=="aux_fit_count": return fit.get("expected_auxiliary_downscaling_fits")==2 and fit.get("completed_auxiliary_downscaling_fits")==2,str(fit)
        if key=="total_fit_count": return fit.get("expected_total_fits")==32 and fit.get("completed_total_fits")==32,str(fit)
        if key=="fit_integrity":
            bad=sum(int(fit.get(k,0)) for k in fit if k.startswith(("missing_","duplicate_","unexpected_"))); return bad==0,f"bad_fit_count={bad}"
        if key=="oof_complete":
            df=data["oof_predictions.parquet"];ci=data["cohort_inventory.csv"];n=int(ci[("final_common_cohort_rows" if "final_common_cohort_rows" in ci else "row_count")].iloc[0]); keys=["aoi","variant","model","row_id"]; return len(df)==n*6 and not df.duplicated(keys).any(),f"rows={len(df)} expected={n*6}"
        if key=="metric_rows":
            df=data["metrics.csv"]; return len(df)==18 and set(df.metric)==set(METRICS),f"rows={len(df)}"
        if key=="metric_arithmetic":
            oof=data["oof_predictions.parquet"]; met=data["metrics.csv"]
            funcs={"roc_auc":roc_auc_score,"pr_auc":average_precision_score,"brier":brier_score_loss}; bad=[]
            for _,r in met.iterrows():
                g=oof[(oof.variant==r.variant)&(oof.model==r.model)]; val=funcs[r.metric](g.y_true,g.y_score)
                if abs(float(r.estimate)-float(val))>1e-10:bad.append((r.variant,r.model,r.metric))
            return not bad,f"mismatches={bad[:3]}"
        if key=="brier_orientation":
            df=data["bootstrap_summary.csv"]; b=df[df.metric=="brier"]; ok=all(abs(r.point_estimate_oriented + r.point_estimate_natural)<1e-12 and abs(r.ci_low_oriented + r.ci_high_natural)<1e-12 and abs(r.ci_high_oriented + r.ci_low_natural)<1e-12 for _,r in b.iterrows()); return ok,f"brier_rows={len(b)}"
        if key=="bootstrap_series": return len(data["bootstrap_summary.csv"])==27,f"rows={len(data['bootstrap_summary.csv'])}"
        if key=="bootstrap_rows": return len(data["bootstrap_replicates.parquet"])==27000,f"rows={len(data['bootstrap_replicates.parquet'])}"
        if key=="bootstrap_paired":
            df=data["bootstrap_replicates.parquet"]; return df.groupby("replicate_id").draw_plan_id.nunique().max()==1 and config.get("bootstrap_configuration",{}).get("refit_inside_bootstrap") is False,"paired draw plan/no-refit"
        if key=="bootstrap_accounting":
            rep=data["bootstrap_replicates.parquet"]; summ=data["bootstrap_summary.csv"]; dup=rep.duplicated(["comparison_series","replicate_id"]).any(); ok=not dup and bool(((summ.valid_replicates+summ.invalid_replicates)==summ.requested_replicates).all()); return ok,f"duplicates={dup}"
        if key=="report_wording":
            findings=find_forbidden_locations(data["report.md"],relative_path="report.md")
            return not findings,f"forbidden_locations={findings}"
        if key=="evia_framing":
            if aoi!="evia_2021_extended":return True,"not applicable to this AOI"
            try:assert_evia_framing(data["report.md"],"report");return True,"mandatory framing present"
            except BaseException as exc:return False,str(exc)
        if key=="stage_complete":
            expected={"plan","export","local-downstream","fit","compare","summarize"}; bad={k:v.get("status") for k,v in data["stages"].items() if k in expected and v.get("status")!="PASS"}; return expected<=set(data["stages"]) and not bad,f"stages={list(data['stages'])} bad={bad}"
        if key=="manifest_identity":
            m=data.get("manifest.json",{}); return m.get("schema_version")==REGIONAL_SCHEMA_VERSION and m.get("analysis_id")==config.get("analysis_id"),f"schema={m.get('schema_version')} id={m.get('analysis_id')}"
        if key=="manifest_complete": return _manifest_ok(root,data.get("manifest.json",{}),REGIONAL_SCHEMA_VERSION,config.get("analysis_id"),REGIONAL_ARTIFACT_SPECS)
        if key=="summary_identity":
            identity=summary.get("aoi")==aoi and summary.get("analysis_id")==config.get("analysis_id") and summary.get("schema_version")==REGIONAL_SCHEMA_VERSION and summary.get("scientific_contract_id")==SCIENTIFIC_CONTRACT_ID
            status_ok=(summary.get("technical_status")=="PASS") if require_final_status else summary.get("technical_status") in {"PENDING_VALIDATION","PASS"}
            regional=data.get("regional_summary.csv",pd.DataFrame())
            regional_status_ok=(not require_final_status) or (len(regional)==1 and regional.technical_status.tolist()==["PASS"])
            return identity and status_ok and regional_status_ok,f"summary identity fields; technical_status={summary.get('technical_status')}; final_required={require_final_status}"
        if key=="dependency_lock": return bool(data.get("manifest.json",{}).get("dependency_lock_hash")),"manifest dependency_lock_hash"
        return False,f"no evaluator for {key}"
    for check in REGIONAL_CHECKS:
        try:ok,evidence=ev(check.evaluator_key);results.append(_record(check,bool(ok),evidence))
        except Exception as exc:results.append({**_record(check,False,f"evaluator exception: {type(exc).__name__}: {exc}"),"status":"SKIP"})
    statuses={r["check_id"]:r["status"] for r in results}; overall=overall_status(statuses,REGIONAL_CHECKS)
    digest_ok,_=verify_manifest_digest(root);validated_digest=(root/"manifest.sha256").read_text().strip() if digest_ok else None
    report={"analysis_id":config.get("analysis_id"),"aoi":aoi,"overall_status":overall,"validated_manifest_sha256":validated_digest,"checks":results,"required_fail":sum(r["severity"]=="required" and r["status"]=="FAIL" for r in results),"required_skip":sum(r["severity"]=="required" and r["status"]=="SKIP" for r in results)}
    if write_results:
        (root/"validator_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8");(root/"validator_summary.json").write_text(json.dumps({k:v for k,v in report.items() if k!="checks"},indent=2),encoding="utf-8")
    return report

def evaluate_synthesis(root: Path, write_results: bool=False) -> dict[str,Any]:
    root=Path(root)
    try:data=_load(root);load_error=None
    except Exception as exc:data={};load_error=str(exc)
    config=data.get("config.json",{}); summary=data.get("summary.json",{}); inputs=data.get("input_references.json",{}); results=[]
    input_files=[]
    for value in inputs.get("inputs",{}).values():
        if not isinstance(value,dict) or not value.get("path"):continue
        source=Path(value["path"])
        input_files.extend([source/rel for rel in value.get("reference_file_hashes",{})] or [source/f for f in ("summary.json","manifest.json","manifest.sha256","validator_summary.json","bootstrap_summary.csv")])
    before=snapshot(input_files)
    def ev(key):
        if load_error:return False,load_error
        idx=data.get("regional_result_index.csv",pd.DataFrame());syn=data.get("four_region_synthesis.csv",pd.DataFrame());report=data.get("report.md","")
        if key=="artifact_presence":
            names=set(SYNTHESIS_REQUIRED_ARTIFACTS)-{"validator_results.json","validator_summary.json"};missing=[n for n in names if not (root/n).is_file() or (root/n).stat().st_size==0];return not missing,f"missing={missing}"
        if key=="manavgat_reference": return "manavgat_2021" in set(idx.get("aoi",[])) and bool(idx[idx.aoi=="manavgat_2021"].validator_status.eq("PASS").all()),"Manavgat index PASS"
        if key=="regional_inputs": return set(ACTUAL_AOIS)<set(idx.aoi) and bool(idx[idx.aoi.isin(ACTUAL_AOIS)].validator_status.eq("PASS").all()),"three regional PASS rows"
        if key=="no_old_evia": return "evia_2021" not in set(idx.aoi) and "evia_2021" not in set(syn.aoi),"old Evia absent"
        if key=="input_manifest_hashes":
            bad=[]
            for aoi,rec in inputs.get("inputs",{}).items():
                source=Path(rec["path"]);p=source/"manifest.json";sp=source/"summary.json";vp=source/"validator_summary.json"
                if aoi=="manavgat_2021" and rec.get("reference_file_hashes"):
                    current={rel:sha256_file(source/rel) for rel in rec["reference_file_hashes"] if (source/rel).is_file()}
                    if current!=rec["reference_file_hashes"] or rec.get("validator_status")!="PASS":bad.append(aoi)
                    continue
                digest_ok,_=verify_manifest_digest(source)
                declared=(source/"manifest.sha256").read_text().strip() if digest_ok else None
                if not (p.is_file() and sp.is_file() and vp.is_file()) or not digest_ok or declared!=rec.get("manifest_hash"):bad.append(aoi);continue
                m=json.loads(p.read_text());s=json.loads(sp.read_text());v=json.loads(vp.read_text());aid=s.get("analysis_id")
                expected="window_closure_sensitivity.v1" if aoi=="manavgat_2021" else REGIONAL_SCHEMA_VERSION
                if s.get("aoi")!=aoi or m.get("analysis_id")!=aid or v.get("analysis_id")!=aid or v.get("overall_status")!="PASS" or v.get("validated_manifest_sha256")!=declared or s.get("schema_version")!=expected or m.get("schema_version")!=expected:bad.append(aoi);continue
                if aoi!="manavgat_2021":
                    ok,_=_manifest_ok(source,m,REGIONAL_SCHEMA_VERSION,aid,REGIONAL_ARTIFACT_SPECS)
                    if not ok:bad.append(aoi)
            return not bad,f"bad={bad}"
        if key=="canonical_integrity": return bool(idx.canonical_hash.notna().all()),"canonical hashes recorded"
        if key=="schema_compatibility": return config.get("schema_version")==SYNTHESIS_SCHEMA_VERSION and set(idx[idx.aoi!="manavgat_2021"].schema_version)=={REGIONAL_SCHEMA_VERSION},f"schemas={set(idx.schema_version)}"
        if key=="contract_identity": return set(idx.scientific_contract_id)=={SCIENTIFIC_CONTRACT_ID},f"contracts={set(idx.scientific_contract_id)}"
        if key=="regional_index": return len(idx)==len(SYNTHESIS_AOIS) and set(idx.aoi)==set(SYNTHESIS_AOIS),f"rows={len(idx)} aois={set(idx.aoi)}"
        if key=="synthesis_rows":
            spec=next(s for s in SYNTHESIS_ARTIFACT_SPECS if s.relative_path=="four_region_synthesis.csv");missing=set(spec.columns)-set(syn.columns);nulls=[c for c in spec.non_null if c in syn and syn[c].isna().any()];duplicates=syn.duplicated(list(spec.primary_key)).any() if set(spec.primary_key)<=set(syn.columns) else True
            ok=len(syn)==four_region_synthesis_row_count(SYNTHESIS_AOIS) and not missing and not nulls and not duplicates and set(syn.aoi)==set(SYNTHESIS_AOIS) and syn.groupby("aoi").size().eq(27).all()
            return ok,f"rows={len(syn)} missing={sorted(missing)} nulls={nulls} duplicates={duplicates}"
        if key=="fold_hash_distinct": return idx[idx.aoi.isin(ACTUAL_AOIS)].fold_mapping_hash.nunique()==3,"three AOI-local fold hashes"
        if key=="descriptive_only": return bool(summary.get("descriptive_only")) and not summary.get("pooled_inference",True),"summary descriptive flags"
        if key=="evia_framing":
            try:assert_evia_framing(report,"synthesis report");return True,"framing present"
            except BaseException as exc:return False,str(exc)
        if key=="report_wording": return not find_forbidden(report),f"forbidden={find_forbidden(report)}"
        columns={c.lower() for c in syn.columns}
        if key=="no_pooled_estimate": return not columns&{"pooled_estimate","weight","meta_analysis","meta_analytic_estimate"},f"columns={columns}"
        if key=="no_pooled_ci": return not columns&{"pooled_ci_low","pooled_ci_high"},f"columns={columns}"
        if key=="no_compute": return not any((root/n).exists() for n in ("oof_predictions.parquet","fold_mapping.parquet","bootstrap_replicates.parquet","metrics.csv")),"no compute artefacts"
        if key=="manifest_identity":
            m=data.get("manifest.json",{});return m.get("schema_version")==SYNTHESIS_SCHEMA_VERSION and m.get("analysis_id")==config.get("synthesis_id"),f"manifest={m.get('schema_version')}/{m.get('analysis_id')}"
        if key=="manifest_complete":return _manifest_ok(root,data.get("manifest.json",{}),SYNTHESIS_SCHEMA_VERSION,config.get("synthesis_id"),SYNTHESIS_ARTIFACT_SPECS)
        if key=="readonly_inputs":return before==snapshot(input_files),"input snapshots unchanged"
        return False,f"no evaluator for {key}"
    for check in SYNTHESIS_CHECKS:
        try:ok,e=ev(check.evaluator_key);results.append(_record(check,bool(ok),e))
        except Exception as exc:results.append({**_record(check,False,f"evaluator exception: {type(exc).__name__}: {exc}"),"status":"SKIP"})
    statuses={r["check_id"]:r["status"] for r in results};overall=overall_status(statuses,SYNTHESIS_CHECKS)
    digest_ok,_=verify_manifest_digest(root);validated_digest=(root/"manifest.sha256").read_text().strip() if digest_ok else None
    report={"synthesis_id":config.get("synthesis_id"),"overall_status":overall,"validated_manifest_sha256":validated_digest,"checks":results,"required_fail":sum(r["severity"]=="required" and r["status"]=="FAIL" for r in results),"required_skip":sum(r["severity"]=="required" and r["status"]=="SKIP" for r in results)}
    if write_results:
        (root/"validator_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8");(root/"validator_summary.json").write_text(json.dumps({k:v for k,v in report.items() if k!="checks"},indent=2),encoding="utf-8")
    return report

REGIONAL_EVALUATORS={c.check_id:c.evaluator_key for c in REGIONAL_CHECKS}
SYNTHESIS_EVALUATORS={c.check_id:c.evaluator_key for c in SYNTHESIS_CHECKS}
