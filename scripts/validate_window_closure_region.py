#!/usr/bin/env python3
"""Execute every artifact-level check for one regional namespace."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.multi_region_window_closure.contract import ACTUAL_AOIS
from src.multi_region_window_closure.per_aoi import regional_namespace
from src.multi_region_window_closure.validation import evaluate_regional
from src.multi_region_window_closure.validators import REGIONAL_CHECKS,REGIONAL_CHECK_COUNTS
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--experiment",required=True,choices=list(ACTUAL_AOIS));p.add_argument("--analysis-id");p.add_argument("--inventory",action="store_true");p.add_argument("--output-root",type=Path);p.add_argument("--no-write",action="store_true");args=p.parse_args(argv)
 if args.inventory:print(json.dumps({"counts":REGIONAL_CHECK_COUNTS,"checks":[c.__dict__ for c in REGIONAL_CHECKS]},indent=2));return 0
 if not args.analysis_id:p.error("--analysis-id is required outside --inventory")
 result=evaluate_regional(regional_namespace(args.experiment,args.analysis_id,args.output_root),args.experiment,write_results=not args.no_write);print(json.dumps(result,indent=2));return 0 if result["overall_status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
