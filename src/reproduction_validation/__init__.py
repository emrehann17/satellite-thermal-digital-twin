"""reproduction_validation: independent, read-only reproduction checkers.

WHAT THIS IS
------------
A validation-only namespace whose sole job is to RE-EXECUTE already-frozen
canonical analyses through the SAME canonical functions and compare the
freshly computed numbers against the frozen artefacts on disk.

WHAT THIS IS NOT
----------------
It defines NO scientific method of its own. Every model, feature list,
population rule, spatial-block rule, seed, adaptation transform, bootstrap
and metric is imported from the canonical pipeline modules:

    src.step8b_train_baseline_vs_thermal_model   (within-region OOF CV)
    src.step9b_run_cross_region_transfer         (raw transfer)
    src.step9c_cross_region_block_bootstrap      (target block bootstrap)
    src.step10b_label_blind_adaptation           (z-score / CORAL adaptation)
    src.step10c_paired_evaluation_bootstrap      (aligned evaluation)
    core.step10_shared                           (CORAL, metrics, constants)

Nothing in this package writes into `outputs/experiments/`,
`outputs/cross_region/` or any other canonical namespace. All output goes
under `outputs/diagnostics/<validation namespace>/`.

Submodules:
    common        -- environment/provenance capture, hashing, cohort resolution
    five_region   -- Task 1: frozen five-region within-region + CORAL check
    mugla_bidirectional -- Task 2: isolated Muglas 2021 <-> 2022 Step9B/9C replay
"""
