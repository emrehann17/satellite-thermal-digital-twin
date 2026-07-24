"""Generic, REPORT-ONLY multi-experiment synthesis of existing canonical
Step9G univariate feature-AUC direction-reversal pair reports into one
side-by-side comparison. Never reruns concept-shift, never recomputes an
AUC/CI/bootstrap draw/reversal classification. No experiment ID is
hard-coded anywhere in this package.

Layering (mirrors src/multi_aoi_transfer_synthesis/):
    discovery   -- resolve canonical pair-report paths for a caller-supplied
                   experiment-ID set (both unordered directory orderings).
    parse       -- read one pair report (+ sibling preregistration) verbatim
                   into normalized region-feature / pair-feature records,
                   validating the fixed scientific contract.
    consistency -- cross-report deduplication/agreement validation with a
                   strict numerical tolerance.
    build       -- top-level orchestration: resolve -> discover -> parse ->
                   validate -> assemble tables -> manifest/analysis_id.
    render      -- write JSON/CSV/Markdown outputs.
"""
