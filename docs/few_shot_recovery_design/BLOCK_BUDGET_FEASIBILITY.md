# Block Inventory and Budget Feasibility

All numbers were measured in this repository on 2026-08-02 from the three
hash-verified canonical Step8A datasets, on population
`valid_for_modeling == True ∧ burnable_tree_shrub_grass == True`, using
`assign_large_blocks(df, 10)` and
`make_spatial_folds(y, large_block_id, 5, 42, strict=True)`.

No model was fitted to produce this file (only a single timing probe, reported
in `IMPLEMENTATION_PLAN.md`, which wrote nothing).

---

## 1. Population totals

| AOI | valid rows | population rows | positives | negatives | prevalence |
|---|---:|---:|---:|---:|---:|
| manavgat_2021 | 24 087 | 20 511 | 784 | 19 727 | 0.0382 |
| bejis_2022 | 15 759 | 15 190 | 1 100 | 14 090 | 0.0724 |
| mugla_2021 | 73 045 | 41 730 | 2 911 | 38 819 | 0.0698 |

All three exceed `STEP8B_MIN_POSITIVES_PER_POPULATION = 30` on both classes by
a wide margin, so no direction is skipped.

## 2. Block inventory at both scales

### 2.1 Canonical 2-cell (~1 km) blocks — for comparison only

| AOI | total blocks | with burned | unburned-only | both classes | burned-only | median rows/block |
|---|---:|---:|---:|---:|---:|---:|
| manavgat_2021 | 5 439 | 235 | 5 204 | 72 | 163 | 4 |
| bejis_2022 | 3 967 | 302 | 3 665 | 50 | 252 | 4 |
| mugla_2021 | 11 316 | 843 | 10 473 | 168 | 675 | 4 |

**A median of 4 cells per block.** This is the measurement behind forced
decision 1: at this scale a "labeled block" is not a unit of labeling effort,
and a 32-block budget would be ~128 cells drawn from terrain immediately
adjacent to the evaluation blocks.

### 2.2 Adopted 10-cell (~5 km) blocks

| AOI | total blocks | with burned | unburned-only | both classes | burned-only | median rows/block |
|---|---:|---:|---:|---:|---:|---:|
| manavgat_2021 | 237 | 28 | 209 | 26 | 2 | 99 |
| bejis_2022 | 176 | 19 | 157 | 15 | 4 | 100 |
| mugla_2021 | 576 | 70 | 506 | 60 | 10 | 90 |

A 10-cell block holds ~90–100 population cells — a plausible unit of labeling
effort, and coarse enough that adaptation and evaluation blocks are not
immediate neighbours.

**Answer to "target block counts":** 237 / 176 / 576 total blocks; 28 / 19 / 70
contain burned cells; 209 / 157 / 506 are unburned-only.

## 3. Outer fold structure

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on
`large_block_id`. Verified for all three targets: `strict=True` succeeded with
`n_splits_used = 5`, **zero** train/eval block overlap in every fold, both
classes present on both sides of every fold, and OOF coverage exactly 1 for
every row.

`pool` = target training-pool blocks for that fold (all target blocks outside
the evaluation fold). Tiers are as defined in `SCIENTIFIC_CONTRACT.md` §7.3.

| target | fold | pool | TIER_A both-class | TIER_B pos-only | TIER_C neg-only | eval blocks | eval rows | eval positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| manavgat_2021 | 0 | 188 | 25 | 1 | 162 | 49 | 4 107 | 140 |
| manavgat_2021 | 1 | 189 | 19 | 2 | 168 | 48 | 4 123 | 165 |
| manavgat_2021 | 2 | 189 | 24 | 2 | 163 | 48 | 4 069 | 157 |
| manavgat_2021 | 3 | 189 | 25 | 1 | 163 | 48 | 4 072 | 160 |
| manavgat_2021 | 4 | 193 | 11 | 2 | 180 | 44 | 4 140 | 162 |
| bejis_2022 | 0 | 141 | 13 | 3 | 125 | 35 | 3 041 | 236 |
| bejis_2022 | 1 | 142 | 11 | 3 | 128 | 34 | 3 087 | 226 |
| bejis_2022 | 2 | 139 | 13 | 3 | 123 | 37 | 3 073 | 235 |
| bejis_2022 | 3 | 140 | 14 | 3 | 123 | 36 | 2 975 | 193 |
| bejis_2022 | 4 | 142 | 9 | 4 | 129 | 34 | 3 014 | 210 |
| mugla_2021 | 0 | 467 | 49 | 8 | 410 | 109 | 8 374 | 594 |
| mugla_2021 | 1 | 459 | 49 | 8 | 402 | 117 | 8 325 | 569 |
| mugla_2021 | 2 | 467 | 50 | 8 | 409 | 109 | 8 360 | 597 |
| mugla_2021 | 3 | 453 | 45 | 7 | 401 | 123 | 8 331 | 566 |
| mugla_2021 | 4 | 458 | 47 | 9 | 402 | 118 | 8 340 | 585 |

Binding minima across all 15 (target × fold) combinations:

```
min |pool|                       = 139   (bejis_2022 fold 2)
min |TIER_A|                     =   9   (bejis_2022 fold 4)
min |TIER_A| + |TIER_B|          =  13   (bejis_2022 fold 4)
```

## 4. Budget feasibility

Candidate set: `0, 1, 2, 4, 8, 16, 32`.

### 4.1 Result

```
COMMON FEASIBLE BUDGET SET = {0, 1, 2, 4, 8, 16, 32}
```

**No budget is dropped.** All seven are applicable to every direction and every
outer fold. Nothing was silently reduced.

### 4.2 Why

The only hard constraint is that a budget cannot exceed the fold's training
pool. `min |pool| = 139 ≫ 32`, so every budget clears it with a factor of more
than four in the tightest case.

### 4.3 The composition transition at k=16 — documented, not an infeasibility

The soft preference "adaptation blocks should contain both classes as far as
available" cannot be satisfied entirely at the top two budgets.

| budget | every fold fills entirely from TIER_A (both-class)? | every fold fills entirely from TIER_A ∪ TIER_B (positive-containing)? | folds needing TIER_C |
|---:|---|---|---:|
| 1 | yes (min TIER_A = 9) | yes | 0 / 15 |
| 2 | yes | yes | 0 / 15 |
| 4 | yes | yes | 0 / 15 |
| 8 | yes (8 ≤ min TIER_A = 9) | yes | 0 / 15 |
| 16 | no | no (min = 13 < 16) | 3 / 15 |
| 32 | no | no | 10 / 15 |

Precisely:

- **k ≤ 8:** every fold draws exclusively from **both-class** blocks, since
  `min |TIER_A| = 9 ≥ 8`. No unburned-only block is ever selected at these
  budgets, in any fold, repeat or direction.
- **k = 16:** the three folds with `|TIER_A| + |TIER_B| < 16` must fill the
  remainder from TIER_C — manavgat fold 4 (13), bejis fold 1 (14) and bejis
  fold 4 (13). The other 12 folds fill entirely from positive-containing
  blocks. mugla (57, 57, 58, 52, 56) is unaffected at every fold.
- **k = 32:** all ten manavgat and bejis folds draw some TIER_C blocks
  (`|TIER_A| + |TIER_B|` is 13–26 across them). All five mugla folds still
  fill entirely from positive-containing blocks (52–58 available).

This is expected and scientifically informative rather than a defect: with only
13–70 positive-containing blocks in a pool, a realistic 32-block labeling
budget necessarily includes unburned terrain, exactly as a field campaign
would. The per-row columns `n_blocks_tier_a`, `n_blocks_tier_b`,
`n_blocks_tier_c`, `adaptation_positive_count` and `adaptation_row_count` make
the transition visible in the curve.

### 4.4 The guarantee that survives at every budget

Because TIER_A is exhausted before TIER_B and TIER_B before TIER_C, and
`min |TIER_A| = 9 > 0`:

> **Every adaptation set with `k ≥ 1`, in every fold, every repeat and every
> direction, contains at least one burned cell.**

This is asserted by the validator, not assumed. It is what makes even `k = 1`
a meaningful supervised adaptation step rather than a degenerate
negatives-only augmentation.

### 4.5 Nesting

One ordering per `(direction, fold, repeat)`; budget `k` takes its first `k`
entries. Nesting `{1} ⊂ {2} ⊂ {4} ⊂ {8} ⊂ {16} ⊂ {32}` therefore holds by
construction, not by post-hoc check — though the validator checks it anyway.

## 5. Direction-level feasibility

Because outer folds depend only on the target, feasibility is a property of
the target and transfers to both directions that share it.

| direction | source rows (train) | target pool blocks/fold (min) | all budgets feasible |
|---|---:|---:|---|
| manavgat_2021 → bejis_2022 | 20 511 | 139 | yes |
| manavgat_2021 → mugla_2021 | 20 511 | 453 | yes |
| bejis_2022 → manavgat_2021 | 15 190 | 188 | yes |
| bejis_2022 → mugla_2021 | 15 190 | 453 | yes |
| mugla_2021 → manavgat_2021 | 41 730 | 188 | yes |
| mugla_2021 → bejis_2022 | 41 730 | 139 | yes |

`direction_budget_feasibility.csv` reproduces this table at
(direction × fold × budget) granularity with a `feasible` column that is `true`
in all 6 × 5 × 7 = 210 rows, together with the tier-composition columns, so
that the feasibility claim is auditable rather than asserted.
