# Window and Date Audit

All dates below are **derived**, not transcribed: the canonical values come from the
`EXPERIMENTS` registry and the shifted values from the frozen arithmetic in
`build_window_variants`. The reproduction command is in §6.

---

## 1. Canonical date sources

| Field | Authoritative source | Notes |
|---|---|---|
| `predictor_start` | `core/regions.py` → `EXPERIMENTS[aoi]["predictor_start_date"]` | Single source of truth |
| `predictor_end` | `EXPERIMENTS[aoi]["predictor_end_date"]` | |
| `label_start` | `EXPERIMENTS[aoi]["label_start_date"]` | |
| `label_end` | `EXPERIMENTS[aoi]["label_end_date"]` | |
| `baseline_years` | `EXPERIMENTS[aoi]["baseline_years"]` | 4 years per AOI |
| `current_period_days` | `core/experiment_context._current_period_days` (line 38) = `(predictor_end − predictor_start).days` | **Not** `core/config.CURRENT_PERIOD_DAYS`, which is the legacy default |
| `pre_label_burn_window` | `EXPERIMENTS[aoi]["pre_label_burn_window"]` | Present only for `mugla_2021`, `evia_2021`, `evia_2021_extended` |

The registry is read through `core.experiment_context.build_experiment_context`, and
`canonical_window` (line 287) raises `WindowClosureError` if any of the four date keys is
missing. **No date is ever hard-coded in `src/window_closure_sensitivity.py`.**

### 1.1 `event_*` and `gate_*` — resolved, with an explicit caveat

The task schema asks for `event_start/end` and `gate_start/end`. **The registry and the
experiment context contain no such fields.** This was checked directly: the only date-bearing
context keys are `predictor_*`, `label_*`, `baseline_*`, `current_period_*` and
`exclude_pre_label_burns` / `pre_label_burn_window`. `gate_labels_dir` is a *path*, not a date
range.

The honest, non-inventing mapping — which the schema must record with explicit provenance:

| Schema column | Resolves to | `source_field` recorded in `window_dates.csv` |
|---|---|---|
| `event_start`, `event_end` | The **label window** — the interval over which the burn event is observed | `EXPERIMENTS[aoi].label_start_date` / `label_end_date` |
| `gate_start`, `gate_end` | The **label window** — the MCD64A1 burned-landcover gate reads the DOY-masked label raster over exactly this interval | `EXPERIMENTS[aoi].label_start_date` / `label_end_date` |

Named fire-start dates (e.g. Bejís "fire start 2022-08-15", the Bördübet/Marmaris fire
"~2021-06-21..25") exist **only in free-text `notes`** and are *not* machine-readable
contract fields. They must never be parsed out of prose. They are reported in `report.md` as
descriptive context only.

Because `event_*` and `gate_*` are aliases of the label window, the requirement that they be
invariant across variants is **satisfied by construction** — but it is still asserted
explicitly (checks `D04`–`D06`) rather than assumed.

---

## 2. Inclusivity and exclusivity semantics

This is the single most error-prone area, and the existing implementation already resolves it
explicitly. Three different conventions coexist, and each must be recorded separately.

### 2.1 Registry dates — inclusive, human-readable

`predictor_start_date` and `predictor_end_date` are inclusive calendar endpoints as written.

### 2.2 Duration arithmetic — **exclusive difference**

```python
duration_days = (predictor_end − predictor_start).days     # canonical_window, line 300
```

This is the **difference**, not the inclusive day count. For Manavgat,
`2021-07-27 − 2021-06-01 = 56`, while the window spans **57** calendar days inclusive.

Both numbers must appear in `window_dates.csv`, under distinct names, so the ±1 can never be
confused:

| Column | Definition | Manavgat canonical |
|---|---|---|
| `calendar_duration_days` | `(end − start).days` — the frozen `duration_days` | 56 |
| `calendar_days_inclusive` | `(end − start).days + 1` | 57 |

### 2.3 Earth Engine `filterDate` — **end-EXCLUSIVE**

`window_closure_date_window_semantics` (line 524) and `landsat_job_date_semantics`
(line 2856) state it explicitly:

> "Production calls `filterDate(start_date, end_date)`, whose END is EXCLUSIVE. That
> behaviour is preserved verbatim — this stage never adds a silent +1 day — and the effective
> last included date is recorded so the off-by-one is visible rather than implicit."

So `earth_engine_filter_end == predictor_end`, `earth_engine_end_exclusive == true`, and the
**effective last included date is `predictor_end − 1 day`**.

**This is deliberate, pre-existing, frozen production behaviour. It must be preserved, not
"fixed".** Changing it would break comparability with the Manavgat PASS and with every
canonical Step3/Step8A artefact.

### 2.4 Python range semantics

`modis_month_filter_transparency` (line 2879) iterates
`range((end_dt − start_dt).days)` — i.e. **excluding** `end_dt` — precisely to mirror the
end-exclusive `filterDate`. Consistent.

### 2.5 The Landsat calendar-month filter is derived, and never clips

`_current_window` (`src/landsat_composite_counterfactual_audit.py:829`) derives
`months_filter` **from the window itself**, using an inclusive `+1` day range. For every AOI
and variant here the window spans enough of the calendar that the derived filter widens to
`1-12`, which is **redundant** next to the exact `filterDate` range. This is recorded as
`calendar_month_filter_redundant = true`. It does **not** mean whole-year data is used —
`filterDate` remains the binding date contract.

Baseline-year windows use **no** calendar-month filter at all
(`_baseline_year_window`, line 851).

---

## 3. Exact ISO dates — all 4 AOIs × 3 variants

Shift rule (`build_window_variants`, line 311):
`start' = start − shift`, `end' = end − shift`. Both ends move by the same amount, so the
duration is preserved exactly; the duration equality is **asserted**, not assumed, and
`end' >= label_start` is a hard failure.

### 3.1 `manavgat_2021` — READ-ONLY REFERENCE (not recomputed)

Canonical: predictor `2021-06-01 .. 2021-07-27` (56 d), label `2021-07-28 .. 2021-08-31`,
baseline years 2017–2020. Shared pre-label censor interval: `2021-05-18 .. 2021-07-27`.

| variant | predictor_start | predictor_end | dur | lead | EE last incl. | cal. days incl. | MODIS eff. days | MODIS clipped |
|---|---|---|---|---|---|---|---|---|
| `canonical` | 2021-06-01 | 2021-07-27 | 56 | 1 | 2021-07-26 | 57 | 56 | 0 |
| `close_7d_earlier` | 2021-05-25 | 2021-07-20 | 56 | 8 | 2021-07-19 | 57 | 49 | **7** |
| `close_14d_earlier` | 2021-05-18 | 2021-07-13 | 56 | 15 | 2021-07-12 | 57 | 42 | **14** |

These values match the frozen `config/preregistration.json`
(`canonical_window.duration_days = 56`, `common_prelabel_start = 2021-05-18`,
`common_prelabel_end = 2021-07-27`) — an independent confirmation that the arithmetic in this
document reproduces the already-PASSed analysis.

### 3.2 `bejis_2022` — NEW ACTUAL AOI

Canonical: predictor `2022-06-15 .. 2022-08-14` (60 d), label `2022-08-15 .. 2022-09-30`,
baseline years 2018–2021. Shared pre-label censor interval: `2022-06-01 .. 2022-08-14`.

| variant | predictor_start | predictor_end | dur | lead | EE last incl. | cal. days incl. | MODIS eff. days | MODIS clipped |
|---|---|---|---|---|---|---|---|---|
| `canonical` | 2022-06-15 | 2022-08-14 | 60 | 1 | 2022-08-13 | 61 | 60 | **0** |
| `close_7d_earlier` | 2022-06-08 | 2022-08-07 | 60 | 8 | 2022-08-06 | 61 | 60 | **0** |
| `close_14d_earlier` | 2022-06-01 | 2022-07-31 | 60 | 15 | 2022-07-30 | 61 | 60 | **0** |

Baseline-year windows (shifted variants):

| year | `close_7d_earlier` | `close_14d_earlier` |
|---|---|---|
| 2018 | 2018-06-08 .. 2018-08-07 | 2018-06-01 .. 2018-07-31 |
| 2019 | 2019-06-08 .. 2019-08-07 | 2019-06-01 .. 2019-07-31 |
| 2020 | 2020-06-08 .. 2020-08-07 | 2020-06-01 .. 2020-07-31 |
| 2021 | 2021-06-08 .. 2021-08-07 | 2021-06-01 .. 2021-07-31 |

**Bejís is the only AOI with zero MODIS clipping in every variant** — even the 14-day shift
starts exactly on 1 June. Warning `W2`.

### 3.3 `mugla_2021` — NEW ACTUAL AOI

Canonical: predictor `2021-06-01 .. 2021-07-28` (57 d), label `2021-07-29 .. 2021-09-15`,
baseline years 2017–2020. Shared pre-label censor interval: `2021-05-18 .. 2021-07-28`.

| variant | predictor_start | predictor_end | dur | lead | EE last incl. | cal. days incl. | MODIS eff. days | MODIS clipped |
|---|---|---|---|---|---|---|---|---|
| `canonical` | 2021-06-01 | 2021-07-28 | 57 | 1 | 2021-07-27 | 58 | 57 | 0 |
| `close_7d_earlier` | 2021-05-25 | 2021-07-21 | 57 | 8 | 2021-07-20 | 58 | 50 | **7** |
| `close_14d_earlier` | 2021-05-18 | 2021-07-14 | 57 | 15 | 2021-07-13 | 58 | 43 | **14** |

Baseline-year windows (shifted variants):

| year | `close_7d_earlier` | `close_14d_earlier` |
|---|---|---|
| 2017 | 2017-05-25 .. 2017-07-21 | 2017-05-18 .. 2017-07-14 |
| 2018 | 2018-05-25 .. 2018-07-21 | 2018-05-18 .. 2018-07-14 |
| 2019 | 2019-05-25 .. 2019-07-21 | 2019-05-18 .. 2019-07-14 |
| 2020 | 2020-05-25 .. 2020-07-21 | 2020-05-18 .. 2020-07-14 |

**Muğla interaction to record:** the registry's own `pre_label_burn_window` is
`2021-06-01 .. 2021-07-28`, but this analysis's shared censor interval starts **14 days
earlier**, at `2021-05-18`. The window-closure censor is therefore a strict **superset** of
the canonical Step8A pre-label exclusion. Any cell that burned in `2021-05-18 .. 2021-05-31`
is censored here but was not censored in canonical Step8A. That is the intended, generic
behaviour of `common_prelabel_interval` and is exactly why the interval is derived from
`min(variant predictor_start)`. It must be **counted and reported** per AOI
(`removed_prelabel_censor`), not silently absorbed.

### 3.4 `evia_2021_extended` — NEW ACTUAL AOI / DIFFERENT-REGIME CONTROL

Canonical: predictor `2021-06-05 .. 2021-08-02` (58 d), label `2021-08-03 .. 2021-09-30`,
baseline years 2017–2020. Shared pre-label censor interval: `2021-05-22 .. 2021-08-02`.

| variant | predictor_start | predictor_end | dur | lead | EE last incl. | cal. days incl. | MODIS eff. days | MODIS clipped |
|---|---|---|---|---|---|---|---|---|
| `canonical` | 2021-06-05 | 2021-08-02 | 58 | 1 | 2021-08-01 | 59 | 58 | 0 |
| `close_7d_earlier` | 2021-05-29 | 2021-07-26 | 58 | 8 | 2021-07-25 | 59 | 55 | **3** |
| `close_14d_earlier` | 2021-05-22 | 2021-07-19 | 58 | 15 | 2021-07-18 | 59 | 48 | **10** |

Baseline-year windows (shifted variants):

| year | `close_7d_earlier` | `close_14d_earlier` |
|---|---|---|
| 2017 | 2017-05-29 .. 2017-07-26 | 2017-05-22 .. 2017-07-19 |
| 2018 | 2018-05-29 .. 2018-07-26 | 2018-05-22 .. 2018-07-19 |
| 2019 | 2019-05-29 .. 2019-07-26 | 2019-05-22 .. 2019-07-19 |
| 2020 | 2020-05-29 .. 2020-07-26 | 2020-05-22 .. 2020-07-19 |

**Note:** `evia_2021` (excluded) carries byte-identical predictor and label dates. The dates
alone therefore do **not** distinguish the two Evia experiments — only `region_key`
(`north_evia_extended` vs `north_evia`) and `output_namespace` do. The exclusion check must
key on **`experiment_id` and the resolved `output_namespace` path**, never on dates.

### 3.5 Label / event / gate invariance — all four AOIs

| AOI | label & event & gate window | Identical in all 3 variants |
|---|---|---|
| `manavgat_2021` | 2021-07-28 .. 2021-08-31 | ✅ |
| `bejis_2022` | 2022-08-15 .. 2022-09-30 | ✅ |
| `mugla_2021` | 2021-07-29 .. 2021-09-15 | ✅ |
| `evia_2021_extended` | 2021-08-03 .. 2021-09-30 | ✅ |

Enforced structurally: `build_window_variants` copies `label_start_date` / `label_end_date`
from the canonical window into every variant and sets `label_window_unchanged = True`;
`common_prelabel_interval` additionally raises if any two variants disagree on
`label_start_date`.

---

## 4. MODIS season policy

| Question | Answer |
|---|---|
| Where defined? | `core/config.py:107-108` — `SUMMER_MONTH_START = 6`, `SUMMER_MONTH_END = 9` |
| Where applied? | `scripts/prepare_modis_for_step7.py:_build_qc_masked_modis_stack`, as a fixed `ee.Filter.calendarRange(6, 9)` **on top of** `filterDate` |
| Which products? | The three MODIS current-window roles only: `modis_lst_mean`, `modis_lst_std`, `modis_valid_observation_count` |
| Landsat affected? | **No.** The Landsat month filter is *derived from the window* and widens to `1-12`, so it never clips. Baseline-year Landsat windows carry no month filter. |
| Fixed or derived? | **Fixed.** This is the crucial asymmetry, called out verbatim in `modis_month_filter_transparency` (line 2879): "Unlike the Landsat month filter — which production DERIVES from the window and which therefore never clips it — this one is a constant, so an earlier-closing window can legitimately lose its earliest days." |
| Interaction with the shift | Days of a shifted window that fall in **May** are silently removed by production. Clipping is therefore AOI-specific and shift-specific. |
| Reusable unchanged for the new AOIs? | **Yes.** It is a global config constant with no AOI-specific branch. |
| Any AOI-specific hard-coded condition? | **None found.** Searched `src/window_closure_sensitivity.py` and `scripts/prepare_modis_for_step7.py`: no AOI name participates in any date or policy branch. |
| Drift risk | Low but real: `SUMMER_MONTH_*` are shared global constants. If either changed between the Manavgat run and this one, MODIS support would move for reasons unrelated to the closure date. **Mitigation:** record both values in `config.json` and assert them (check `D08`). |

### 4.1 MODIS clipping summary — the headline cross-AOI asymmetry

| AOI | canonical | `close_7d_earlier` | `close_14d_earlier` |
|---|---|---|---|
| `manavgat_2021` (ref) | 0 d | 7 d | 14 d |
| `bejis_2022` | 0 d | **0 d** | **0 d** |
| `mugla_2021` | 0 d | 7 d | 14 d |
| `evia_2021_extended` | 0 d | **3 d** | **10 d** |

Because Bejís's canonical window opens on 15 June, a 14-day shift only reaches 1 June, so it
never crosses into May. Evia-extended opens on 5 June, so only the first 3 (resp. 10) days
are lost.

**Interpretive consequence, which must be stated explicitly in `report.md`:** the shifted
variants are *not* mechanistically identical across AOIs. For Manavgat and Muğla the shift
moves the window **and** reduces effective MODIS support; for Bejís it moves the window with
**no** MODIS support loss. A cross-AOI difference in the result is therefore
**direction-dependent and confounded with support loss** — it may not be attributed to the
closure date alone. This is a strengthening of the existing Manavgat limitation
(`COMPARE_ANALYSIS_CONTRACT`, line 8888), not a new caveat.

Bejís consequently functions as a useful near-control for the support-loss mechanism.

---

## 5. Off-by-one, leap-day and boundary analysis

| Risk | Assessment | Verdict |
|---|---|---|
| `duration_days` vs inclusive day count | Differ by exactly 1 (56 vs 57 etc.). Both are emitted under distinct column names. | **Controlled** |
| `filterDate` end-exclusivity | Explicit, recorded per job as `ee_filter_end_semantics = "exclusive"` and `effective_last_included_date = end − 1 day`. Never silently `+1`-ed. | **Controlled** |
| Duration change under shift | `build_window_variants` recomputes and **raises** if `duration != canonical duration`. Verified 56/56/56, 60/60/60, 57/57/57, 58/58/58. | **Impossible by construction** |
| `predictor_end >= label_start` | Hard failure in both `canonical_window` and `build_window_variants`. Minimum lead is 1 day (canonical); shifting only increases it to 8 and 15. | **Impossible by construction** |
| Leap day (29 Feb) | `_baseline_year_window` uses `datetime.replace(year=…)`, which raises `ValueError` on 29 Feb → non-leap year. **All windows here lie in May–August**, so no date can be 29 Feb. Baseline years 2017–2021 include leap year 2020, but 2020-05-xx/07-xx are all valid. | **Not applicable — no exposure** |
| Year boundary crossing | No window crosses 31 Dec. Widest span: `2021-05-18 .. 2021-07-14`. | **Not applicable** |
| Dates escaping the event year | All predictor and label dates stay inside the event year for every AOI and variant. Baseline-year windows are *intentionally* in 2017–2021 via `replace(year=…)`. | **Controlled** |
| Month-boundary arithmetic | Handled by `datetime` arithmetic throughout; no manual day-of-month math anywhere. | **Controlled** |
| MODIS month filter clipping | Real and AOI-specific. **Measured, reported, never corrected.** | **Controlled → reported as `W1`/`W2`** |
| Two Evia experiments with identical dates | Dates cannot disambiguate them. | **Controlled** — exclusion keys on `experiment_id`/`output_namespace` (check `S05`) |

**Conclusion: no off-by-one defect found.** Every boundary convention is explicit in code and
recorded in the emitted artefacts.

---

## 6. Reproduction commands (read-only)

Recompute the four canonical hashes:

```bash
cd /home/emrehan-metin/satellite-thermal-digital-twin
for a in manavgat_2021 bejis_2022 mugla_2021 evia_2021_extended; do
  sha256sum "outputs/experiments/$a/step8a/step8a_500m_modeling_dataset.parquet"
done
```

Expected:

```
054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439  .../manavgat_2021/...
3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393  .../bejis_2022/...
c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e  .../mugla_2021/...
bdce859cf482f575d0f273174b157f47efd61779953fdd23d9486c5face5e553  .../evia_2021_extended/...
```

Regenerate every date in §3 straight from the frozen implementation (no Earth Engine, no
writes):

```bash
cd /home/emrehan-metin/satellite-thermal-digital-twin
python - <<'PY'
from core.experiment_context import build_experiment_context
from src.window_closure_sensitivity import (
    build_window_variants, canonical_window, common_prelabel_interval,
    modis_month_filter_transparency,
)
for aoi in ("manavgat_2021", "bejis_2022", "mugla_2021", "evia_2021_extended"):
    ctx = build_experiment_context(aoi)
    can = canonical_window(ctx)
    variants = build_window_variants(ctx, (0, 7, 14))
    print(aoi, can["duration_days"], common_prelabel_interval(variants))
    for v in variants:
        m = modis_month_filter_transparency(
            v["predictor_start_date"], v["predictor_end_date"])
        print("  ", v["variant_id"], v["predictor_start_date"], v["predictor_end_date"],
              v["duration_days"], v["lead_days"], m["clipped_day_count"])
PY
```

Confirm the Manavgat frozen values this document reproduces:

```bash
python -c "import json;d=json.load(open('outputs/diagnostics/window_closure_sensitivity/manavgat_2021/config/preregistration.json'));print(d['canonical_window'], d['common_censor_interval'])"
```

---

## 7. Blockers

**None.**

`CANONICAL_DATE_SOURCE_UNRESOLVED` does **not** apply: every canonical date for all four AOIs
resolves deterministically from `core/regions.py:EXPERIMENTS`, and every shifted date is
computed by the same frozen function that produced the already-PASSed Manavgat analysis.

`CANONICAL_HASH_DRIFT` does **not** apply: all four Step8A digests reproduced exactly.

The one genuine ambiguity — `event_*` / `gate_*` having no dedicated registry fields — is
**resolved by explicit aliasing to the label window with recorded `source_field` provenance**
(§1.1), not by assumption, and not by parsing dates out of prose notes.
