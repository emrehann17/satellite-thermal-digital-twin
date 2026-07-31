# Reference Artifacts — resolved Step10 references, hashes, duplicate rule

Read-only inventory taken **2026-08-03**. Every digest below was recomputed
from disk. Nothing was written.

---

## 1. Canonical Step8A inputs — hash gate

All three recomputed digests **match the contract exactly**:

| Experiment | Path | sha256 | Primary rows | Pos | Neg | Prevalence |
|---|---|---|---:|---:|---:|---:|
| `manavgat_2021` | `outputs/experiments/manavgat_2021/step8a/step8a_500m_modeling_dataset.parquet` | `054a1961…787f3439` ✅ | 20,511 | 784 | 19,727 | 0.03822339 |
| `bejis_2022` | `outputs/experiments/bejis_2022/step8a/step8a_500m_modeling_dataset.parquet` | `3dec785a…c24d9e393` ✅ | 15,190 | 1,100 | 14,090 | 0.07241606 |
| `mugla_2021` | `outputs/experiments/mugla_2021/step8a/step8a_500m_modeling_dataset.parquet` | `c4ab107d…0be7db8e` ✅ | 41,730 | 2,911 | 38,819 | 0.06975797 |

Population = `valid_for_modeling == True` ∧ `burnable_tree_shrub_grass == True`,
i.e. `step9b.population_subset(df, PRIMARY_POPULATION)`.

## 2. The duplicate-artifact problem, and its resolution

### 2.1 Why duplicates exist

`core/step10_shared.step10_output_dir(source_id, target_id)` resolves to
`cross_region_output_root(source_id, target_id)/step10`, and
`cross_region_output_root` (`src/step9a_audit_cross_region_inputs.py:136`)
does **not** normalise the pair order:

```python
return BASE_DIR / "outputs" / "cross_region" / f"{source_id}__{target_id}"
```

But `run_step10b(source_id, target_id)` (`step10b:193`) builds
`directions = [(source, target, …), (target, source, …)]` and writes **both**
directions into the **invoking** pair directory. Running the pair from each
end therefore produces two complete artifacts, each containing both
directions.

Both orderings were run for both of our pairs, so every one of the four
directions is present in **two** artifacts:

| Pair | Directory A | Directory B |
|---|---|---|
| {bejís, muğla} | `bejis_2022__mugla_2021/step10/` | `mugla_2021__bejis_2022/step10/` |
| {manavgat, muğla} | `manavgat_2021__mugla_2021/step10/` | `mugla_2021__manavgat_2021/step10/` |

### 2.2 The duplicates are NOT interchangeable — measured

| Comparison | Cell coverage | Probability vectors | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| `bejis_2022__mugla_2021` vs `mugla_2021__bejis_2022` | identical, exact | **not** bit-identical; max abs diff **4.44e-16** | 5 of 16 rows differ, max **4.867e-08** | 4 of 16 differ, max **1.618e-08** |
| `manavgat_2021__mugla_2021` vs `mugla_2021__manavgat_2021` | identical, exact | max abs diff **2.22e-16** | 3 of 16 differ, max **2.212e-08** | 1 of 16 differs, max **1.881e-09** |

The `analysis_id` also differs between the two copies of each pair, because the
preregistration hashes the invoking (source, target) order.

**Cause.** `RandomForestClassifier(n_jobs=-1)` accumulates per-tree
probabilities into a shared array under a lock; the summation order depends on
thread scheduling, so float non-associativity produces ~1 ULP differences.
Those ULP differences flip a handful of near-tied ranks, and each rank flip
moves ROC-AUC by exactly `1/(n_pos·n_neg)`:

| Target | `1/(n_pos·n_neg)` | observed max ROC-AUC drift | ≈ rank flips |
|---|---:|---:|---:|
| `mugla_2021` | 8.849e-09 | 4.867e-08 | ~5.5 |
| `bejis_2022` | 6.452e-08 | — | — |
| `manavgat_2021` | 6.466e-08 | 2.212e-08 | <1 |

This is entirely consistent and is **not** evidence of a defect in either
artifact. It is, however, decisive for the reproduction gate — see
`NUMERICAL_FEASIBILITY.md` §4.

### 2.3 Frozen resolution rule

> **For direction `S → T`, the reference artifact is
> `outputs/cross_region/{S}__{T}/step10/`.**

Rationale, in order of weight:

1. It is exactly what the repository's own resolver returns:
   `step10_output_dir(S, T)`. No new convention is invented.
2. It is the same source-first rule already frozen for the Step9B references in
   `docs/mugla_subsampling_design/`, so the two analyses resolve directions
   identically.
3. All four required directories exist, so the rule is total.
4. It is deterministic and independent of file mtimes or of which run happened
   last.

**Consequence, stated explicitly:** the four directions are drawn from four
*different* artifacts with four different `analysis_id`s. Each of those
artifacts also contains the reverse direction; **that half is ignored**. Anyone
comparing against the other copy will see ROC-AUC differences up to ~4.9e-08
for the reasons in §2.2. The chosen file's digest is frozen into
`input_hashes.json`, so the choice is auditable after the fact.

## 3. Resolved artifacts, per direction

Schema of every file below is the Step10 layout produced by
`src/step10{a,b,c,d}_*.py`; there is no `schema_version` string inside them, so
the schema is identified by producer module + file name.

### 3.1 `bejis_2022_to_mugla_2021` → `outputs/cross_region/bejis_2022__mugla_2021/step10/`

| File | sha256 (first 16) | Bytes |
|---|---|---:|
| `step10_preregistration.json` | `2c37b4f0…` *(recompute at run time)* | 6,676 |
| `step10_input_audit.json` | — | 2,788 |
| `step10_adaptation_statistics.json` | — | 15,717 |
| `step10_predictions.parquet` | `61f3da0c6bff431c` | 4,261,144 |
| `step10_metrics.csv` | `f7f54641df97fd42` | 2,742 |
| `step10_metrics.json` | — | 10,151 |
| `step10_bootstrap_replicates.parquet` | — | 712,575 |
| `step10_bootstrap_summary.json` | — | 16,669 |
| `step10_final_report.json` | — | 73,676 |

`analysis_id` = `af4e071bdebe4d47…` · direction coverage = both
`bejis_2022_to_mugla_2021` and `mugla_2021_to_bejis_2022` · Step8A digests
pinned in `step10_input_audit.json`: `3dec785a…` (bejís) and `c4ab107d…`
(muğla) · canonical λ = `1e-05` at
`step10_preregistration.json → /scientific_config/adaptation_methods/coral_after_regionwise_zscore/lambda`
· predictions rows = 341,520 · **target rows for this direction = 41,730**
(unique cells 41,730).

### 3.2 `mugla_2021_to_bejis_2022` → `outputs/cross_region/mugla_2021__bejis_2022/step10/`

`analysis_id` = `c84943adc568f96c…` · `step10_predictions.parquet` sha256
`1b296c1536ca8587…`, 4,261,144 B · `step10_metrics.csv` sha256
`94e34c87f7d63ba6…` · Step8A digests `3dec785a…`, `c4ab107d…` · λ = 1e-05 ·
predictions rows 341,520 · **target rows = 15,190** (unique cells 15,190).

### 3.3 `manavgat_2021_to_mugla_2021` → `outputs/cross_region/manavgat_2021__mugla_2021/step10/`

`analysis_id` = `a382046d5453412e…` · `step10_predictions.parquet` sha256
`a35e2e9ce9be221b…`, 4,826,274 B · `step10_metrics.csv` sha256
`aac85958e0cd18d0…` · Step8A digests `054a1961…`, `c4ab107d…` · λ = 1e-05 ·
predictions rows 373,446 · **target rows = 41,730**.

### 3.4 `mugla_2021_to_manavgat_2021` → `outputs/cross_region/mugla_2021__manavgat_2021/step10/`

`analysis_id` = `2843b998c142c98d…` · `step10_predictions.parquet` sha256
`40e28a418635c556…`, 4,866,814 B · `step10_metrics.csv` sha256
`2f22e928b42e32bf…` · Step8A digests `054a1961…`, `c4ab107d…` · λ = 1e-05 ·
predictions rows 373,446 · **target rows = 20,511**.

> The run must recompute and freeze **all** digests at plan time; the ones
> quoted here are the audit-time values and exist so that drift is detectable.

## 4. Artifact schemas actually used

`step10_predictions.parquet` — 10 columns:

```
analysis_id, direction, source_experiment, target_experiment, population,
target_cell_id, target_spatial_block_id, model_family, adaptation_method,
prediction_probability
```

`adaptation_method ∈ {raw_source_only, regionwise_zscore, coral_after_regionwise_zscore}`;
`population` is always `burnable_tree_shrub_grass`.

`step10_metrics.csv` — 8 columns:

```
analysis_id, direction, method, model_family, roc_auc, pr_auc,
positive_count, negative_count
```

`step10_bootstrap_replicates.parquet` — 2,000 rows (2 directions × 1,000),
wide, columns of the form `{metric}__{series}` and
`delta_{metric}__{contrast}__{family}`, with
`metric ∈ {roc_auc, pr_auc}` and
`series ∈ {within, raw_source_only, regionwise_zscore, coral_after_regionwise_zscore} × {baseline, thermal}`.

### 4.1 **Step10 has no Brier score — scope gap**

Neither `step10_metrics.csv` nor `step10_bootstrap_replicates.parquet` contains
a Brier column. `core/step10_shared.compute_threshold_free_metrics` (line 227)
returns **only** `roc_auc`, `pr_auc`, `positive_count`, `negative_count`.

The only `brier_score` values anywhere in the Step10 outputs are inside
`step10_metrics.json` under `/raw_reproduction/**/step9_reference/**` and
`/step9_raw_metric_provenance/**` — i.e. **Step9B values for the raw method
only**, carried in as provenance. There is no Brier for `regionwise_zscore`
and none for `coral_after_regionwise_zscore`.

Consequences for this design are set out in `SCIENTIFIC_CONTRACT.md` §7.2 and
in the blocker list of `README.md`.

## 5. Frozen reference values under the §2.3 resolution rule

ROC-AUC and PR-AUC are read from the resolved `step10_metrics.csv`. **Brier is
recomputed** from the resolved `step10_predictions.parquet` joined to the
canonical Step8A labels, because Step10 stores none (§4.1).

| Direction | Family | Method | ROC-AUC | PR-AUC | Brier (recomputed) |
|---|---|---|---:|---:|---:|
| bejís→muğla | baseline | raw | 0.592239000 | 0.093631465 | 0.089877747 |
| bejís→muğla | baseline | z-score | 0.564970535 | 0.081945632 | 0.107130955 |
| bejís→muğla | baseline | **CORAL λ=1e-5** | **0.570168266** | **0.083587660** | **0.102774772** |
| bejís→muğla | thermal | raw | 0.618474749 | 0.092538591 | 0.078925904 |
| bejís→muğla | thermal | z-score | 0.517733284 | 0.068823903 | 0.095424188 |
| bejís→muğla | thermal | **CORAL λ=1e-5** | **0.506637796** | **0.069072390** | **0.094293529** |
| muğla→bejís | baseline | raw | 0.450738338 | 0.064165866 | 0.130966651 |
| muğla→bejís | baseline | z-score | 0.573885606 | 0.085240742 | 0.144595300 |
| muğla→bejís | baseline | **CORAL λ=1e-5** | **0.608749984** | **0.095603027** | **0.144813780** |
| muğla→bejís | thermal | raw | 0.583191238 | 0.088271917 | 0.110793034 |
| muğla→bejís | thermal | z-score | 0.535262598 | 0.072184707 | 0.109941306 |
| muğla→bejís | thermal | **CORAL λ=1e-5** | **0.560332344** | **0.077905186** | **0.109880888** |
| manavgat→muğla | baseline | raw | 0.507929432 | 0.074905511 | 0.115699397 |
| manavgat→muğla | baseline | z-score | 0.471436657 | 0.065159039 | 0.122782215 |
| manavgat→muğla | baseline | **CORAL λ=1e-5** | **0.485676776** | **0.068475239** | **0.123480200** |
| manavgat→muğla | thermal | raw | 0.470159871 | 0.062646470 | 0.104681521 |
| manavgat→muğla | thermal | z-score | 0.430504328 | 0.061034347 | 0.096706559 |
| manavgat→muğla | thermal | **CORAL λ=1e-5** | **0.442962246** | **0.063727556** | **0.105905625** |
| muğla→manavgat | baseline | raw | 0.521519830 | 0.037836864 | 0.196262879 |
| muğla→manavgat | baseline | z-score | 0.576235028 | 0.048126968 | 0.102107906 |
| muğla→manavgat | baseline | **CORAL λ=1e-5** | **0.601340698** | **0.047935831** | **0.107190364** |
| muğla→manavgat | thermal | raw | 0.400999666 | 0.028738774 | 0.177117443 |
| muğla→manavgat | thermal | z-score | 0.558937404 | 0.043194286 | 0.061046721 |
| muğla→manavgat | thermal | **CORAL λ=1e-5** | **0.560404173** | **0.043941261** | **0.066256598** |

The four `raw_source_only` ROC-AUC values reproduce the canonical Step9B
transfer metrics exactly (`0.592239`, `0.450738`, `0.507929`, `0.400999`),
which independently confirms that the resolved artifacts sit on the same
canonical chain.

## 6. Manavgat ↔ Bejís — contextual reference only

`outputs/cross_region/manavgat_2021__bejis_2022/step10/` exists (13 files) and
is the historical primary Step10 pair. Per the scope, it is **inventoried for
context and never re-run**, and it contributes **no row** to any sensitivity
output. It is recorded in `repository_inventory.json` with its digests so the
provenance chain is complete, and is excluded from `metrics.csv`,
`bootstrap_summary.csv` and `sensitivity_summary.csv` by an explicit direction
allow-list.

## 7. Read-only guarantee

Every path in this document is opened **read-only** and bound by sha256. No
file under `outputs/experiments/`, `outputs/cross_region/` or
`outputs/robustness/` is written, moved or re-run by this analysis, and no
Step10 artifact is copied into the sensitivity namespace — only its digest is.
