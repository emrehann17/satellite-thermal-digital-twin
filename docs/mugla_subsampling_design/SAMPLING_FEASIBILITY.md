# Sampling Feasibility — read-only computation

Every number in this document was computed on **2026-08-03** from the frozen
Step8A parquet at sha256 `c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e`,
using the canonical `assign_large_blocks(df, 10)` utility. No output was
written to any production path; no model was fitted except in the isolated
timing probe of §7, whose predictions were discarded.

**Verdict: the design is feasible with no blockers.**

---

## 1. Populations

| Experiment | Total rows | Valid | Primary population | Positive | Negative | Prevalence |
|---|---:|---:|---:|---:|---:|---:|
| `mugla_2021` | 73,098 | 73,045 | **41,730** | 2,911 | 38,819 | 0.06975797 |
| `manavgat_2021` | 24,150 | 24,087 | **20,511** | 784 | 19,727 | 0.03822339 |
| `bejis_2022` | 15,759 | 15,759 | **15,190** | 1,100 | 14,090 | 0.07241606 |

Population = `valid_for_modeling == True` ∧ `burnable_tree_shrub_grass == True`,
i.e. `step9b.population_subset`. The two counts named in the contract —
41,730 and 20,511 — reproduce exactly.

```
sampling fraction = 20511 / 41730 = 0.4915288759...
```

## 2. Spatial strata at 10 cells (≈ 5 km)

`assign_large_blocks(df, 10)` on the full frame, then the population filter:

| Quantity | Value |
|---|---:|
| Blocks touching any row of the full frame | 760 |
| Blocks containing ≥ 1 primary-population row | **576** |
| Non-empty strata (block × label) | **636** |
| Label-0 strata | 566 |
| Label-1 strata | **70** |
| Blocks with both labels | 60 |
| Blocks negative-only | 506 |
| Blocks positive-only | 10 |

This matches the frozen `block_manifest.json` exactly
(`unique_spatial_blocks: 576`, `positive_containing_blocks: 70`,
`negative_containing_blocks: 566`, `mixed_class_blocks: 60`).

Capacity distribution across the 636 strata: min 1, median 80.5, mean 65.6,
max 100 (a full 10 × 10 block), sd 35.8. Twelve strata have capacity 1.

| Capacity bucket | Label-0 strata | cap → alloc | Label-1 strata | cap → alloc |
|---|---:|---:|---:|---:|
| 1 | 7 | 7 → 7 | 5 | 5 → 5 |
| 2–5 | 28 | 104 → 53 | 10 | 34 → 20 |
| 6–10 | 22 | 177 → 83 | 5 | 44 → 21 |
| 11–25 | 50 | 862 → 418 | 12 | 186 → 91 |
| 26–50 | 66 | 2,393 → 1,182 | 11 | 425 → 210 |
| 51–75 | 64 | 4,175 → 2,051 | 10 | 637 → 314 |
| 76–100 | 329 | 31,101 → 15,279 | 17 | 1,580 → 777 |
| **total** | **566** | **38,819 → 19,073** | **70** | **2,911 → 1,438** |

## 3. Hamilton allocation — computed, not assumed

Integer-exact largest-remainder at `N = 20511`, `N_total = 41730`:

| Quantity | Value |
|---|---:|
| `Σ floor_s` | 20,211 |
| `shortfall` | **300** |
| Strata strictly above the remainder cut | 295 |
| Strata tied *at* the cut | **12** (all of capacity 5, remainder numerator 19,095) |
| Of those, awarded by the tie-break | **5** |
| `Σ alloc_s` | **20,511** ✅ |
| Strata with `alloc_s > c_s` | **0** ✅ |
| Max `alloc_s − c_s` | 0 |
| Strata with `alloc_s == c_s` | 12 (exactly the capacity-1 strata) |
| Strata with `alloc_s == 0` | **0** ✅ |
| Min / max `alloc_s` | 1 / 49 |
| Min / max `alloc_s / c_s` | 0.400 / 1.000 |
| Blocks represented in every subsample | **576 / 576** ✅ |

The integer formulation (`(c·N) // N_total`, `(c·N) % N_total`) and the float
formulation (`floor(c·N/N_total)`, fractional part) were both computed and give
**identical allocations**. Integer is the frozen rule so the tie set is exact
by construction.

**The tie-break is materially exercised.** 295 strata clear the cut outright;
the 300th unit must be chosen from 12 equally-entitled capacity-5 strata, and
exactly 5 of them receive it. Ordering by `stratum_id` ascending is what makes
that choice reproducible. A different tie-break rule would change 5 of the
20,511 selected rows in every repeat.

## 4. Prevalence

| Quantity | Value |
|---|---:|
| Full-Muğla prevalence | 0.069757968 (2,911 / 41,730) |
| Exact proportional positive count | 1,430.8057 |
| Allocated positives | **1,438** |
| Allocated negatives | **19,073** |
| Subsample prevalence | **0.070108722** (1,438 / 20,511) |
| Absolute drift | +0.00035075 |
| Relative drift | **+0.5028 %** |
| Rounding bound `70 / 20511` | 0.003413 |
| Drift within bound? | **Yes**, by a factor of 9.7 |

The +7.19 positive surplus is the accumulated largest-remainder rounding across
the 70 label-1 strata. Muğla's prevalence is preserved in exactly the "within
integer rounding limits" sense, and the bound is stated so the claim is
checkable rather than rhetorical.

**Muğla's positive count is not equalised to Manavgat's.** 1,438 vs 784 — the
subsample carries 1.83 × Manavgat's positives, by design.

## 5. Fold composition — deterministic and repeat-invariant

The allocation depends only on stratum capacities, and each 10-cell block
belongs to exactly one fold in the frozen mapping. Therefore **every repeat has
the same per-fold row and class counts**; only the identity of the cells
varies.

| Fold | Blocks | Full rows | Full pos | Subsample rows | Subsample pos | Subsample neg |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 109 | 8,374 | 594 | 4,111 | 293 | 3,818 |
| 1 | 117 | 8,325 | 569 | 4,096 | 280 | 3,816 |
| 2 | 109 | 8,360 | 597 | 4,107 | 295 | 3,812 |
| 3 | 123 | 8,331 | 566 | 4,096 | 281 | 3,815 |
| 4 | 118 | 8,340 | 585 | 4,101 | 289 | 3,812 |
| **total** | **576** | **41,730** | **2,911** | **20,511** | **1,438** | **19,073** |

**Fold guarantees, all satisfied:**

- Every fold is non-empty on both sides; the smallest evaluation side is 4,096
  rows with 280 positives.
- Training side of each fold: ≥ 16,404 rows, ≥ 1,143 positives. Both classes
  present on both sides of every fold, in every repeat.
- `STEP8B_MIN_POSITIVES_PER_POPULATION = 30` is cleared by 9.3 × at the
  tightest point.
- 0 of 576 blocks span more than one fold (frozen artifact property, inherited).
- OOF coverage is exactly-once by construction: fold ids come from a per-`cell_id`
  join, and `Σ` fold sizes = 20,511.

Because these counts are fixed, a validator can assert them as literals rather
than recomputing tolerances.

## 6. Fit accounting

| Arm | Contract upper bound | Design | Why |
|---|---:|---:|---|
| Within-Muğla | 20 × 2 × 5 = **200** | **200** | no reduction available; every (repeat, family, fold) is a distinct training frame |
| Muğla as source | 20 × 2 targets × 2 = **80** | **40** | the source model is fitted on the sampled Muğla frame only and never sees the target, so one fit per (repeat, family) serves both targets |
| Muğla as target | 2 × 2 = **4** | **0** | frozen per-cell raw-transfer predictions are exact for any subset (`REPOSITORY_INVENTORY.md` §5) |
| **Total** | **284** | **240** | |

**Fit registry.** A `FitRegistry` keyed on a `fit_identity` string, following
`few_shot_recovery.FitRegistry` (line 740). Identities:

```
within  |{repeat_id}|{family}|{fold_id}      -> 20 × 2 × 5 = 200
source  |{repeat_id}|{family}                -> 20 × 2     =  40
target  (no identity — no fit)               ->              0
```

The registry memoises fit **results** (probability vectors), not fitted
estimators, so memory stays flat. `accounting()` must report
`unique_fits == 240`, `within_fits == 200`, `source_fits == 40`,
`reuse_events == 40` (each source fit referenced twice, once per target). The
run asserts this against `expected_unique_fit_count()` and fails closed on
mismatch.

**Both reductions are exactness-preserving, and both are audited.** The source
reduction is sound because `build_pipeline(...).fit(X_source, y_source)` with
`random_state=42` is a deterministic function of the source frame alone; the
canonical `step9b.run_one_direction_population` fits it the same way per
direction and would produce a bit-identical estimator. The validator's deep
mode re-fits one `(repeat_id, family)` pair independently and asserts the
predictions are bit-identical; those 2 audit fits are accounted separately and
are not part of the 240.

## 7. Runtime, memory and disk — measured, not guessed

Timing probe on the real canonical frame, single process, `n_jobs=-1` RF,
this machine:

| Operation | Rows | Measured |
|---|---:|---:|
| baseline fit (Arm A fold) | 16,400 | 1.22 s |
| thermal fit (Arm A fold) | 16,400 | 1.25 s |
| `predict_proba` (Arm A fold) | 4,111 | 0.16 s |
| thermal fit (Arm B source) | 20,511 | 2.14 s |

**Runtime estimate**

| Stage | Work | Estimate |
|---|---|---:|
| `plan` | load 3 frames, hash 3 + 6 artifacts, block assign, allocate, 20 × selection, write | ≈ 60 s |
| `fit` — Arm A | 200 fits × ~1.25 s + 200 predicts × 0.16 s | ≈ 285 s |
| `fit` — Arm B | 40 fits × ~2.14 s + 80 predicts × ~0.2 s | ≈ 100 s |
| `fit` — Arm C | 20 × 2 subset-joins + metric recomputation, no fits | ≈ 10 s |
| `summarize` | 600 repeat rows → 30 summary rows, report | ≈ 5 s |
| **total** | | **≈ 7–10 min wall clock** |

**Memory.** The Muğla Step8A frame is 46.2 MB in memory after block
assignment; Manavgat and Bejís add ≈ 25 MB together. A 300-tree
`RandomForestClassifier` on 16.4 k × 14 encoded features adds a few hundred MB
during `fit`. Only one estimator is alive at a time (the registry stores
vectors, not models). **Peak resident set ≈ 1.0–1.5 GB.** No streaming or
chunking is required.

**Disk.**

| Output | Rows | Estimate |
|---|---:|---:|
| `selected_cells.parquet` | 20 × 20,511 = 410,220 | ≈ 6–10 MB (dictionary-encoded ids) |
| `oof_predictions/part-within.parquet` | 410,220 | ≈ 8 MB |
| `oof_predictions/part-source.parquet` | 20 × (20,511 + 15,190) = 714,020 | ≈ 14 MB |
| `oof_predictions/part-target.parquet` | 20 × 2 × 20,511 = 820,440 | ≈ 16 MB |
| `fold_mapping.parquet` | 41,730 | < 1 MB |
| `stratum_allocation.csv` | 636 | ≈ 50 KB |
| `sampling_inventory.csv` | ~30 | < 10 KB |
| `reference_metrics.csv` | 30 | < 10 KB |
| `repeat_metrics.csv` | 600 | ≈ 100 KB |
| `subsampling_summary.csv` | 30 | < 10 KB |
| JSON + report | — | < 1 MB |
| **total** | | **≈ 45–55 MB**, headroom to 100 MB |

## 8. Blockers

**None.** Every precondition the design depends on was checked and holds:

| Precondition | Status |
|---|---|
| Three canonical Step8A hashes match | ✅ |
| Muğla primary = 41,730, Manavgat primary = 20,511 | ✅ |
| `assign_large_blocks(df, 10)` passes `validate_canonical_grid` on Muğla | ✅ |
| Hamilton allocation sums to 20,511, never over-draws, never drops a stratum | ✅ |
| Prevalence drift inside the stated bound | ✅ (0.00035 ≤ 0.00341) |
| Persisted full-Muğla 10-cell fold mapping exists and covers all 41,730 rows | ✅ |
| Block ↔ fold mapping consistent (0 blocks span folds) | ✅ |
| Both classes on both sides of every fold in every repeat | ✅ |
| Arm A reference recomputes from its own OOF vectors | ✅ exact |
| Arm C raw-transfer predictions cover the full Muğla primary cell set | ✅ exact set equality |
| All four transfer references recompute from persisted probabilities | ✅ exact |
| Transfer references provenance-bound to the same Step8A hashes | ✅ |

**Two conditions to re-verify at implementation time**, both cheap and both
already covered by the fail-closed list:

1. If `outputs/experiments/mugla_2021/robustness/step8_big_blocks/block_10_cells/oof_predictions.parquet`
   is ever regenerated, its sha256 changes and the fold mapping must be
   re-pinned. The run records the digest and the validator compares it.
2. `manavgat_2021__mugla_2021/…predictions.parquet` and
   `mugla_2021__manavgat_2021/…predictions.parquet` both contain both
   directions but are byte-different (separate runs, different row order).
   Their metrics agree exactly. The frozen resolution rule — read direction
   `S → T` from `outputs/cross_region/{S}__{T}/step9b/` — removes the ambiguity;
   the validator asserts the rule was followed by comparing the recorded digest.
