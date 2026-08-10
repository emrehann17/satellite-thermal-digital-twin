"""Validator registry for one independent regional execution."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class ValidatorCheck:
    check_id: str
    description: str
    requirement_key: str
    evaluator_key: str
    severity: str = "required"

def R(cid, desc, key, evaluator=None, severity="required"):
    return ValidatorCheck(cid, desc, key, evaluator or key, severity)

REGIONAL_CHECKS = (
 R("REG-SCOPE-ONE-AOI","Exactly one allow-listed AOI","single_aoi"),
 R("REG-SCOPE-VARIANTS","Exactly three variants and two shifted scenarios","variant_scope"),
 R("REG-SCOPE-MODELS","Exactly baseline and thermal models","model_scope"),
 R("REG-SCOPE-NO-CROSS-AOI","No artefact row belongs to another AOI","no_cross_aoi"),
 R("REG-DATE-CANONICAL","Canonical dates and hash match frozen inputs","canonical_integrity"),
 R("REG-DATE-SHIFTS","Exact 7/14-day shifts and constant duration","date_shifts"),
 R("REG-DATE-LABEL","Label/event/gate dates and semantics invariant","label_invariance"),
 R("REG-HASH-INPUTS","Recorded input hashes are complete","input_hashes"),
 R("REG-COHORT-COMMON","Three variants have identical cohort IDs/labels","common_cohort"),
 R("REG-COHORT-NODUP","Cohort inventory has no duplicate or missing variant","cohort_inventory"),
 R("REG-FOLD-COMMON","One fold mapping/hash is used across variants","fold_invariance"),
 R("REG-FOLD-CLASSES","Exactly five folds each contain both classes","fold_classes"),
 R("REG-FIT-PRIMARY-COUNT","expected/completed primary estimator fits equal 30","primary_fit_count"),
 R("REG-FIT-AUXILIARY-COUNT","expected/completed auxiliary fits equal 2","aux_fit_count"),
 R("REG-FIT-TOTAL-COUNT","expected/completed total fits equal 32","total_fit_count"),
 R("REG-FIT-NODUP","No missing, duplicate, or unexpected logical fit","fit_integrity"),
 R("REG-OOF-COMPLETE","OOF coverage is exact and duplicate-free","oof_complete"),
 R("REG-METRIC-ROWS","Exactly 18 metric rows","metric_rows"),
 R("REG-METRIC-ARITHMETIC","ROC-AUC, PR-AUC and Brier recompute from OOF","metric_arithmetic"),
 R("REG-METRIC-BRIER-ORIENTATION","Brier natural/oriented signs and CI endpoints are correct","brier_orientation"),
 R("REG-BOOT-SERIES","Exactly 27 comparison series and summary rows","bootstrap_series"),
 R("REG-BOOT-ROWS","Exactly 27,000 requested replicate rows","bootstrap_rows"),
 R("REG-BOOT-PAIRED","Draw plan is paired and no-refit","bootstrap_paired"),
 R("REG-BOOT-ACCOUNTING","Replicate IDs and valid/invalid accounting reconcile","bootstrap_accounting"),
 R("REG-REPORT-WORDING","No pooled/significance/causal/best-window language","report_wording"),
 R("REG-REPORT-EVIA","Extended Evia mandatory framing is present when applicable","evia_framing"),
 R("REG-STAGE-COMPLETE","Scientific stages plan/export/local-downstream/fit/compare/summarize explicitly PASS","stage_complete"),
 R("REG-ARTEFACT-PRESENCE","All regional artefacts are present and non-empty","artifact_presence"),
 R("REG-MANIFEST-SCHEMA","Manifest has regional schema and matching analysis ID","manifest_identity"),
 R("REG-MANIFEST-COMPLETE","Manifest entries and output hashes are complete","manifest_complete"),
 R("REG-SUMMARY-IDENTITY","Summary AOI/analysis/schema/contract identities match","summary_identity"),
 R("REG-LOCK-HASH","Dependency lock hash is recorded","dependency_lock",severity="advisory"),
)

def check_counts(checks):
    ids=[c.check_id for c in checks]
    if len(ids)!=len(set(ids)): raise ValueError("duplicate validator check ID")
    return {"total":len(checks),"required":sum(c.severity=="required" for c in checks),"advisory":sum(c.severity=="advisory" for c in checks)}

REGIONAL_CHECK_COUNTS=check_counts(REGIONAL_CHECKS)

# Kept empty while shared modules are being consumed by the regional path;
# there is no synthesis validator registration or executable check set.
SYNTHESIS_CHECKS = ()
SYNTHESIS_CHECK_COUNTS = check_counts(SYNTHESIS_CHECKS)
def overall_status(results: Mapping[str,str], checks):
    return "PASS" if all(c.severity!="required" or results.get(c.check_id)=="PASS" for c in checks) else "FAIL"
