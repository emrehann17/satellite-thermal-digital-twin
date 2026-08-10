"""Actual orchestration bodies with explicit, injectable scientific engines.

The driver owns scope, identity, stages, resume, quarantine and validation.
Scientific work is delegated to the existing per-AOI implementation through
the engine interface; tests use a synthetic engine and never contact GEE.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any, Protocol
from src.multi_region_window_closure.contract import assert_regional_aoi
from src.multi_region_window_closure.per_aoi import quarantine_regional_namespace, regional_namespace, regional_resume_matches
from src.multi_region_window_closure.validation import evaluate_regional

REGIONAL_STAGES=("plan","export","local-downstream","fit","compare","summarize","validate")

def _inventory(root:Path)->dict[str,str]:
    controls={"manifest.json","manifest.sha256","validator_results.json","validator_summary.json","stages/validate.json"}
    out={}
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel=str(path.relative_to(root))
                if rel not in controls and not rel.startswith("stages/"):out[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
    return out

def _inventory_hash(inventory:dict[str,str])->str:
    return hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _state_outputs_match(root:Path,state:dict[str,Any])->bool:
    wanted=state.get("stage_output_inventory")
    if not isinstance(wanted,list):return False
    current={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in wanted if (root/rel).is_file()}
    return len(current)==len(wanted) and _inventory_hash(current)==state.get("stage_output_hash")

def _validate_reusable(root:Path,state:dict[str,Any])->bool:
    from src.multi_region_window_closure.schema import verify_manifest_digest
    if not _state_outputs_match(root,state):return False
    ok,_=verify_manifest_digest(root)
    try:summary=json.loads((root/"validator_summary.json").read_text())
    except (OSError,json.JSONDecodeError):return False
    current=(root/"manifest.sha256").read_text().strip() if ok else None
    return ok and summary.get("overall_status")=="PASS" and summary.get("analysis_id")==state.get("analysis_id") and summary.get("validated_manifest_sha256")==current

def _summarize_reusable(root:Path,state:dict[str,Any],*,aoi:str,analysis_id:str)->bool:
    """Read-only validator-contract preflight for summarize-stage reuse."""
    if not _state_outputs_match(root,state):return False
    from src.multi_region_window_closure.dates import WINDOW_DATE_COLUMNS
    from src.multi_region_window_closure.schema import REGIONAL_ARTIFACT_SPECS,verify_manifest_digest
    from src.multi_region_window_closure.wording import find_forbidden
    required={s.relative_path for s in REGIONAL_ARTIFACT_SPECS if s.required and s.stage=="summarize"}
    if any(not (root/rel).is_file() or (root/rel).stat().st_size==0 for rel in required):return False
    try:
        import pandas as pd
        dates=pd.read_csv(root/"window_dates.csv")
        report=(root/"report.md").read_text(encoding="utf-8")
    except (OSError,ValueError,UnicodeDecodeError):return False
    if tuple(dates.columns)!=WINDOW_DATE_COLUMNS or len(dates)!=3:return False
    if dates["analysis_id"].isna().any() or set(dates["analysis_id"].astype(str))!={analysis_id}:return False
    if dates["aoi"].isna().any() or set(dates["aoi"].astype(str))!={aoi}:return False
    if dates.duplicated(["analysis_id","aoi","variant"]).any():return False
    if find_forbidden(report):return False
    manifest_ok,_=verify_manifest_digest(root)
    return manifest_ok

class RegionalEngine(Protocol):
    def run_stage(self, stage:str, root:Path, context:dict[str,Any])->dict[str,Any]: ...

def run_regional_actual(*,aoi:str,analysis_id:str,output_root:Path,engine:RegionalEngine,config_hash:str,input_hash:str,resume:bool=False,force:bool=False,execute_actual:bool=False)->dict[str,Any]:
    if not execute_actual:raise RuntimeError("Actual execution requires --execute-actual.")
    if resume and force:raise RuntimeError("resume and force are mutually exclusive")
    aoi=assert_regional_aoi(aoi);root=regional_namespace(aoi,analysis_id,Path(output_root))
    if root.exists() and not resume and not force:raise RuntimeError("BLOCKER: REGIONAL_NAMESPACE_ALREADY_EXISTS")
    if force:quarantine_regional_namespace(aoi,analysis_id,"force",Path(output_root),__import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    root.mkdir(parents=True,exist_ok=True);(root/"stages").mkdir(exist_ok=True)
    context={"aoi":aoi,"analysis_id":analysis_id,"config_hash":config_hash,"input_hash":input_hash,"resume_requested":resume,"force_requested":force}
    ran=[];reused=[];upstream_hash=None;upstream_reran=False
    for stage in REGIONAL_STAGES:
        state_path=root/"stages"/f"{stage}.json"
        if resume and state_path.is_file():
            state=json.loads(state_path.read_text(encoding="utf-8"))
            identity=regional_resume_matches(state,aoi=aoi,analysis_id=analysis_id,config_hash=config_hash,input_hash=input_hash)
            if stage=="validate":reusable=_validate_reusable(root,state)
            elif stage=="summarize":reusable=_summarize_reusable(root,state,aoi=aoi,analysis_id=analysis_id)
            else:reusable=_state_outputs_match(root,state)
            dependency=state.get("upstream_stage_output_hash")==upstream_hash
            if identity and reusable and dependency and not upstream_reran:
                reused.append(stage);upstream_hash=state["stage_output_hash"];continue
        before=_inventory(root)
        try:
            if stage=="fit" and hasattr(engine,"assert_common_cohort"):engine.assert_common_cohort(root,context)
            if stage=="validate":
                validation=evaluate_regional(root,aoi,write_results=True,require_final_status=False)
                if validation["overall_status"]!="PASS":raise RuntimeError("regional validator did not PASS")
                summary_path=root/"summary.json"
                summary=json.loads(summary_path.read_text());summary["technical_status"]="PASS"
                summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
                import pandas as pd
                regional=pd.read_csv(root/"regional_summary.csv");regional["technical_status"]="PASS";regional.to_csv(root/"regional_summary.csv",index=False)
                if hasattr(engine,"finalize_manifest"):engine.finalize_manifest(root,context)
                validation=evaluate_regional(root,aoi,write_results=True,require_final_status=True)
                if validation["overall_status"]!="PASS":raise RuntimeError("regional validator did not PASS after technical-status binding")
                summarize_state_path=root/"stages"/"summarize.json"
                summarize_state=json.loads(summarize_state_path.read_text())
                owned=summarize_state.get("stage_output_inventory",[])
                refreshed={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in owned if (root/rel).is_file()}
                if len(refreshed)!=len(owned):raise RuntimeError("summarize outputs disappeared during final status binding")
                summarize_state["stage_output_hash"]=_inventory_hash(refreshed)
                summarize_state_path.write_text(json.dumps(summarize_state,indent=2),encoding="utf-8")
                upstream_hash=summarize_state["stage_output_hash"]
                detail={"validator_status":"PASS"}
            else:detail=engine.run_stage(stage,root,context) or {}
        except BaseException as exc:
            failed={**context,"stage":stage,"status":"FAIL","failure_reason":f"{type(exc).__name__}: {exc}","resume_eligible":False}
            state_path.write_text(json.dumps(failed,indent=2),encoding="utf-8")
            raise
        if stage=="summarize" and hasattr(engine,"finalize_manifest"):
            state_path.write_text(json.dumps({**context,"stage":stage,"status":"PASS"},indent=2),encoding="utf-8")
            engine.finalize_manifest(root,context)
        after=_inventory(root);changed={rel:digest for rel,digest in after.items() if before.get(rel)!=digest}
        state={**context,"stage":stage,"status":"PASS","detail":detail,"upstream_stage_output_hash":upstream_hash,"stage_output_inventory":sorted(changed),"stage_output_hash":_inventory_hash(changed),"resume_eligible":True}
        state_path.write_text(json.dumps(state,indent=2),encoding="utf-8");ran.append(stage);upstream_hash=state["stage_output_hash"];upstream_reran=True
    return {"aoi":aoi,"analysis_id":analysis_id,"namespace":str(root),"ran_stages":ran,"reused_stages":reused,"technical_status":"PASS"}
