"""Complete deterministic fixtures for validator and stage-driver tests only."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
from src.multi_region_window_closure.bootstrap import comparison_series
from src.multi_region_window_closure.contract import (
 ACTUAL_AOIS,METRICS,MODEL_FAMILIES,REGIONAL_SCHEMA_VERSION,SCIENTIFIC_CONTRACT_ID,SYNTHESIS_AOIS,SYNTHESIS_SCHEMA_VERSION,VARIANTS,
)
from src.multi_region_window_closure.dates import window_date_rows
from src.multi_region_window_closure.schema import REGIONAL_ARTIFACT_SPECS,SYNTHESIS_ARTIFACT_SPECS,build_manifest,write_manifest_with_digest
from src.multi_region_window_closure.validation import sha256_file

HASH="a"*64
def _json(path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
def _fit():return {"expected_primary_estimator_fits":30,"completed_primary_estimator_fits":30,"duplicate_primary_estimator_fits":0,"missing_primary_estimator_fits":0,"unexpected_primary_estimator_fits":0,"expected_auxiliary_downscaling_fits":2,"completed_auxiliary_downscaling_fits":2,"duplicate_auxiliary_downscaling_fits":0,"missing_auxiliary_downscaling_fits":0,"unexpected_auxiliary_downscaling_fits":0,"expected_total_fits":32,"completed_total_fits":32,"failed_fit_attempts":0,"retried_fit_attempts":0}

def build_regional_fixture(root:Path,aoi:str="bejis_2022",analysis_id:str="synthetic-regional",with_states:bool=True)->Path:
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 from src.multi_region_window_closure.inputs import CANONICAL_STEP8A_SHA256,canonical_step8a_path
 canonical_hash=CANONICAL_STEP8A_SHA256[aoi];canonical_path=canonical_step8a_path(aoi)
 config={"schema_version":REGIONAL_SCHEMA_VERSION,"scientific_contract_id":SCIENTIFIC_CONTRACT_ID,"analysis_id":analysis_id,"aoi":aoi,"variants":list(VARIANTS),"model_families":list(MODEL_FAMILIES),"canonical_hash":canonical_hash,"production_canonical_hash":canonical_hash,"canonical_path":str(canonical_path),"fit_accounting":_fit(),"bootstrap_configuration":{"refit_inside_bootstrap":False}}
 _json(root/"config.json",config);_json(root/"input_hashes.json",{"hashes":{"canonical":HASH,"labels":"b"*64}});_json(root/"repository_inventory.json",{"git_commit":"synthetic","dependency_lock_hash":HASH})
 dates=pd.DataFrame(window_date_rows((aoi,),(0,7,14)));dates.insert(0,"analysis_id",analysis_id);dates.to_csv(root/"window_dates.csv",index=False)
 from src.multi_region_window_closure.plan import export_plan_rows
 pd.DataFrame(export_plan_rows((aoi,))).assign(analysis_id=analysis_id).to_csv(root/"export_plan.csv",index=False)
 cohort_hash="c"*64;fold_hash=("d" if aoi=="bejis_2022" else "e" if aoi=="mugla_2021" else "f")*64
 pd.DataFrame([{"aoi":aoi,"variant":v,"cohort_hash":cohort_hash,"fold_mapping_hash":fold_hash,"row_count":10,"label_hash":"1"*64,"static_hash":"2"*64} for v in VARIANTS]).to_csv(root/"cohort_inventory.csv",index=False)
 fold=pd.DataFrame([{"aoi":aoi,"row_id":i,"grid_id":i,"fold_id":i//2,"y_true":i%2} for i in range(10)]);fold.to_parquet(root/"fold_mapping.parquet",index=False)
 pd.DataFrame([{"aoi":aoi,"variant":v,"relative_path":f"{v}/x.tif","sha256":HASH} for v in VARIANTS]).to_csv(root/"variant_artifact_index.csv",index=False)
 oof=[]
 for v_i,v in enumerate(VARIANTS):
  for m_i,m in enumerate(MODEL_FAMILIES):
   for i in range(10):
    y=i%2;score=(0.15+0.04*v_i+0.03*m_i) if y==0 else (0.75+0.03*v_i+0.04*m_i)
    oof.append({"aoi":aoi,"variant":v,"model":m,"row_id":i,"grid_id":i,"fold_id":i//2,"y_true":y,"y_score":score,"cohort_hash":cohort_hash,"fold_mapping_hash":fold_hash})
 oof_df=pd.DataFrame(oof);oof_df.to_parquet(root/"oof_predictions.parquet",index=False)
 metrics=[]
 funcs={"roc_auc":roc_auc_score,"pr_auc":average_precision_score,"brier":brier_score_loss}
 for (v,m),g in oof_df.groupby(["variant","model"]):
  for metric,fn in funcs.items():metrics.append({"aoi":aoi,"variant":v,"model":m,"metric":metric,"estimate":fn(g.y_true,g.y_score)})
 pd.DataFrame(metrics).to_csv(root/"metrics.csv",index=False)
 series=comparison_series(aoi); reps=[];summ=[]
 for s_i,s in enumerate(series):
  natural=0.01*(1+(s_i%3));metric=s["metric"];oriented=-natural if metric=="brier" else natural;lo=natural-.005;hi=natural+.005;olo,ohi=(-hi,-lo) if metric=="brier" else (lo,hi)
  key=f"series-{s_i:02d}"
  for rep in range(1000):reps.append({"aoi":aoi,"comparison_series":key,"replicate_id":rep,"draw_plan_id":f"draw-{rep}","difference_natural":natural,"difference_oriented":oriented,"valid":True,"invalid_reason":None})
  summ.append({**s,"comparison_series":key,"point_estimate_natural":natural,"ci_low_natural":lo,"ci_high_natural":hi,"point_estimate_oriented":oriented,"ci_low_oriented":olo,"ci_high_oriented":ohi,"requested_replicates":1000,"valid_replicates":1000,"invalid_replicates":0,"fold_mapping_hash":fold_hash})
 pd.DataFrame(reps).to_parquet(root/"bootstrap_replicates.parquet",index=False);pd.DataFrame(summ).to_csv(root/"bootstrap_summary.csv",index=False)
 pd.DataFrame([{"analysis_id":analysis_id,"aoi":aoi,"technical_status":"PASS","cohort_hash":cohort_hash,"fold_mapping_hash":fold_hash,"canonical_hash":HASH}]).to_csv(root/"regional_summary.csv",index=False)
 _json(root/"summary.json",{"schema_version":REGIONAL_SCHEMA_VERSION,"scientific_contract_id":SCIENTIFIC_CONTRACT_ID,"analysis_id":analysis_id,"aoi":aoi,"technical_status":"PASS","canonical_hash":canonical_hash,"cohort_hash":cohort_hash,"fold_mapping_hash":fold_hash})
 report="Descriptive point estimate; uncertainty remains under this frozen design."
 if aoi=="evia_2021_extended":
  from src.multi_region_window_closure.wording import evia_regime_note
  report+=" "+evia_regime_note()
 (root/"report.md").write_text(report,encoding="utf-8")
 # Conform the compact fixture to the authoritative production schemas.
 from src.multi_region_window_closure.schema import REGIONAL_ARTIFACT_SPECS
 for spec in REGIONAL_ARTIFACT_SPECS:
  path=root/spec.relative_path
  if not spec.columns or not path.is_file():continue
  frame=pd.read_parquet(path) if spec.kind=="parquet" else pd.read_csv(path)
  for column in spec.columns:
   if column not in frame:frame[column]="synthetic"
  if spec.relative_path=="cohort_inventory.csv":
   frame["final_common_cohort_rows"]=10;frame["cohort_hash"]=cohort_hash;frame["fold_mapping_hash"]=fold_hash
  if spec.relative_path=="fold_mapping.parquet":
   frame["cell_id"]=frame["row_id"];frame["block_id"]=frame["grid_id"];frame["cohort_hash"]=cohort_hash;frame["fold_mapping_hash"]=fold_hash
  if spec.relative_path=="bootstrap_replicates.parquet":
   for column in ("comparison_family","variant","model_a","model_b","metric"):
    mapping=pd.DataFrame(summ).set_index("comparison_series")[column];frame[column]=frame["comparison_series"].map(mapping)
  if spec.relative_path=="regional_summary.csv":
   frame=pd.DataFrame([{c:(aoi if c=="aoi" else analysis_id if c=="analysis_id" else canonical_hash if c=="canonical_step8a_sha256" else cohort_hash if c=="cohort_hash" else fold_hash if c=="fold_mapping_hash" else "PASS" if c=="technical_status" else "synthetic") for c in spec.columns}])
  frame.to_parquet(path,index=False) if spec.kind=="parquet" else frame.to_csv(path,index=False)
 if with_states:
  for stage in ("plan","export","local-downstream","fit","compare","summarize"):_json(root/"stages"/f"{stage}.json",{"aoi":aoi,"analysis_id":analysis_id,"stage":stage,"status":"PASS"})
 finalize_regional_manifest(root,analysis_id,aoi,canonical_hash)
 return root

def finalize_regional_manifest(root,analysis_id,aoi,canonical_hash=None):
 canonical_hash=canonical_hash or json.loads((Path(root)/"summary.json").read_text())["canonical_hash"]
 m=build_manifest(analysis_id,root,HASH,HASH,{aoi:canonical_hash},"",schema_version=REGIONAL_SCHEMA_VERSION,artifact_specs=REGIONAL_ARTIFACT_SPECS);write_manifest_with_digest(Path(root)/"manifest.json",m)

class SyntheticRegionalEngine:
 def run_stage(self,stage,root,context):
  if stage=="plan" and not (root/"config.json").exists():build_regional_fixture(root,context["aoi"],context["analysis_id"],with_states=False)
  return {"synthetic":True,"stage":stage}
 def assert_common_cohort(self,root,context):
  df=pd.read_csv(root/"cohort_inventory.csv");assert df.cohort_hash.nunique()==1
 def finalize_manifest(self,root,context):finalize_regional_manifest(root,context["analysis_id"],context["aoi"])

def build_reference_fixture(root:Path)->Path:
 root.mkdir(parents=True,exist_ok=True);_json(root/"summary.json",{"analysis_id":"manavgat-ref","aoi":"manavgat_2021","schema_version":"window_closure_sensitivity.v1","scientific_contract_id":SCIENTIFIC_CONTRACT_ID,"canonical_hash":HASH});digest=write_manifest_with_digest(root/"manifest.json",{"analysis_id":"manavgat-ref","schema_version":"window_closure_sensitivity.v1","files":[]});_json(root/"validator_summary.json",{"analysis_id":"manavgat-ref","overall_status":"PASS","validated_manifest_sha256":digest});return root

class SyntheticSynthesisEngine:
 def __init__(self,inputs):self.inputs=inputs
 def finalize_manifest(self,root,context):
  sid=context["synthesis_id"];m=build_manifest(sid,root,HASH,HASH,{},HASH,schema_version=SYNTHESIS_SCHEMA_VERSION,artifact_specs=SYNTHESIS_ARTIFACT_SPECS);write_manifest_with_digest(root/"manifest.json",m)
 def build_read_only(self,root,context):
  sid=context["synthesis_id"];_json(root/"config.json",{"schema_version":SYNTHESIS_SCHEMA_VERSION,"synthesis_id":sid,"scientific_contract_id":SCIENTIFIC_CONTRACT_ID});refs={};idx=[];rows=[]
  for aoi,path in self.inputs.items():
   summary=json.loads((path/"summary.json").read_text());validator=json.loads((path/"validator_summary.json").read_text());mh=(path/"manifest.sha256").read_text().strip();refs[aoi]={"path":str(path),"manifest_hash":mh};fold=summary.get("fold_mapping_hash",("9" if aoi=="manavgat_2021" else "8")*64);idx.append({"aoi":aoi,"analysis_id":summary["analysis_id"],"schema_version":summary["schema_version"],"scientific_contract_id":SCIENTIFIC_CONTRACT_ID,"validator_status":validator["overall_status"],"manifest_hash":mh,"canonical_hash":summary.get("canonical_hash",HASH),"fold_mapping_hash":fold})
   if aoi=="manavgat_2021":source=comparison_series(aoi)
   else:source=pd.read_csv(path/"bootstrap_summary.csv").to_dict("records")
   for j,r in enumerate(source):
    from src.multi_region_window_closure.contract import METRIC_DIRECTION,orient,orient_interval,orientations_equal,classify_interval
    metric=r["metric"];natural=r.get("point_estimate_natural",0.01);lo=r.get("ci_low_natural",0.0);hi=r.get("ci_high_natural",0.02);olo,ohi=orient_interval(metric,lo,hi)
    rows.append({"aoi":aoi,"source_analysis_id":summary["analysis_id"],"comparison_family":r["comparison_family"],"variant":r["variant"],"model_a":r["model_a"],"model_b":r["model_b"],"metric":metric,"metric_direction":METRIC_DIRECTION[metric],"point_estimate_natural":natural,"ci_low_natural":lo,"ci_high_natural":hi,"point_estimate_oriented":orient(metric,natural),"ci_low_oriented":olo,"ci_high_oriented":ohi,"orientations_equal":orientations_equal(metric),"interval_status":classify_interval(olo,ohi),"descriptive_only":True})
  from src.multi_region_window_closure.wording import evia_regime_note
  _json(root/"input_references.json",{"inputs":refs});_json(root/"input_hashes.json",{"hashes":{a:r["manifest_hash"] for a,r in refs.items()}});pd.DataFrame(idx).to_csv(root/"regional_result_index.csv",index=False);pd.DataFrame(rows).to_csv(root/"four_region_synthesis.csv",index=False);_json(root/"summary.json",{"schema_version":SYNTHESIS_SCHEMA_VERSION,"synthesis_id":sid,"technical_status":"PENDING_VALIDATION","descriptive_only":True,"pooled_inference":False});(root/"report.md").write_text("Descriptive point estimates; uncertainty remains. "+evia_regime_note(),encoding="utf-8");self.finalize_manifest(root,context)
