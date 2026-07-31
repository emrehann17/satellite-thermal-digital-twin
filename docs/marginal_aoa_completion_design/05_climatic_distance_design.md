# 05. Climatic Distance — Exact Design

> **Decisions C-8 and C-9 are ACCEPTED. The new climate export is AUTHORISED.**
> Neither is an implementation blocker any longer.

`04_climate_input_audit.md` establishes that no climate-normal input exists in
the repository, so this component is built on a new Earth Engine export. Nothing
below was executed: no GEE query, no export, no distance computation.

---

## 1. Authorised data source

```
collection        IDAHO_EPSCOR/TERRACLIMATE
version           TerraClimate (Abatzoglou et al. 2018), Earth Engine asset as published;
                  the resolved asset version string is captured at export time
temporal coverage 1958-01 → present, monthly
reference period  1991-01-01 .. 2020-12-31        (30 years, WMO normal period)
month count       360
year count        30
shared warm season  months 6, 7, 8, 9
native resolution ~4638 m (1/24°)
CRS               EPSG:4326
```

**Season provenance.** Months 6–9 are the project's existing fire-season
definition — `SUMMER_MONTH_START = 6`, `SUMMER_MONTH_END = 9`
(`core/config.py:107-108`). The window is reused, not invented, and it is **one
shared window for all four AOIs**, unlike the AOI-specific anniversary windows
that disqualified the Step5 baseline (doc 04 §2.2).

### Why TerraClimate

| Candidate | Resolution | Water-balance variables | Status |
|---|---|---|---|
| **`IDAHO_EPSCOR/TERRACLIMATE`** | ~4.6 km | `pdsi`, `aet`, `def`, `soil`, `vpd`, `pr`, `tmmx`, `tmmn` — all native | **ADOPTED** |
| `ECMWF/ERA5_LAND/MONTHLY_AGGR` | ~11 km | temperature, dewpoint, precipitation; VPD must be derived; no climatic water deficit | **Not part of this run.** Recorded as a possible later sensitivity only. |
| `WORLDCLIM/V1/BIO` | ~1 km | temperature/precipitation only; fixed 1960–1990 period | Rejected — stale reference period, no moisture-deficit axis |
| `UCSB-CHG/CHIRPS/DAILY` | ~5.5 km | precipitation only | Rejected — single axis |

Three reasons:

1. **Climatic water deficit (`def`) is native.** Water deficit is the canonical
   fire-climate covariate; it integrates temperature and moisture supply into the
   quantity that governs fuel dryness.
2. **~4.6 km is adequate for these AOIs.** The smallest AOI (`bejis_2022`,
   0.70° × 0.47°) spans roughly 60 × 52 km, so ~13 × 11 TerraClimate cells.
   ERA5-Land at ~11 km would give roughly 5 × 5 cells — too coarse to resolve the
   coastal/inland contrast that drives the Bejís result.
3. **Independent of the Landsat/MODIS surface products** already in the predictor
   set, satisfying criterion 5 of doc 04 §4.

**No ERA5-Land cross-check is performed in the initial completion work.** An
earlier draft proposed one; it is removed from the primary contract and recorded
as a deferred sensitivity that would need its own preregistration.

---

## 2. Primary climate vector — four variables

Four axes, chosen so that no two measure the same thing: one temperature level,
one moisture supply, one moisture deficit, one atmospheric demand.

| # | Field name | TerraClimate band | Aggregation over 1991–2020 | Units |
|---|---|---|---|---|
| 1 | `annual_mean_temperature_c` | `(tmmx + tmmn) / 2` | mean over **all 360 months** | °C |
| 2 | `annual_precipitation_mm` | `pr` | yearly sum, then mean of the **30 yearly sums** | mm |
| 3 | `warm_season_climatic_water_deficit_mm` | `def` | Jun–Sep yearly sum, then mean of the **30 yearly sums** | mm |
| 4 | `warm_season_vpd_kpa` | `vpd` | mean of the Jun–Sep months over 1991–2020 | kPa |

```
climate_feature_count = 4
```

### Two axes deliberately removed

An earlier draft proposed six variables, adding *warm-season mean temperature*
and *warm-season precipitation*. **Both are removed.** They are near-duplicates
of the annual axes they sit beside — in a Mediterranean climate the warm-season
and annual temperature signals are strongly collinear, as are the two
precipitation measures — and including them would have weighted the temperature
and precipitation directions twice each in an equally-weighted metric, quietly
downweighting the deficit and VPD axes that carry the fire-relevant information.

The retained four keep one axis per physical mechanism.

### Band scale factors

TerraClimate stores `tmmx`/`tmmn` at 0.1 °C and `vpd` at 0.01 kPa. **The scale
factors used must come from the pinned export contract, be read from the
collection's band metadata at export time, and be recorded truthfully in
`climate_export_metadata.json`.** They must not be hard-coded from memory, and
the recorded values must match what was actually applied.

---

## 3. Spatial mask and summary

**Mask: the native valid-land support of TerraClimate**, applied consistently to
both the AOI summaries and the reference window. TerraClimate is a terrestrial
product and is undefined over water; its own valid-data mask is therefore the
land mask, and using it avoids introducing a second, independently-defined
landmask that could disagree with it.

An earlier draft proposed reusing the per-AOI ESA WorldCover raster with class 80
excluded. **That is replaced by the native TerraClimate support.** One mask, one
definition, applied identically everywhere.

```
climate_value(aoi, v)
    = arithmetic mean of variable v
      over valid TerraClimate land pixels
      inside the canonical AOI bbox
```

The mask matters: `mugla_2021`'s bbox is 38.9% permanent water and
`evia_2021_extended`'s is 57.7% (doc 01 §6). Without a land restriction those two
AOIs would be summarised over largely undefined or marine cells.

**Forbidden inputs, restated as a contract:**

- no event-period rasters,
- no Step5 or Step5C proxies (`baseline_lst_mean`, `baseline_tvdi_mean`, …),
- no current LST/TVDI predictors,
- no target labels.

Enforced by validator check 16 as a path-provenance assertion (doc 08).

---

## 4. Reference window and standardisation

The scaling problem: standardising by the SD **across the four AOIs** would let a
single AOI define the metric and would change every distance if a fifth AOI were
added. The scale is therefore taken from a fixed, broad reference region.

```
Mediterranean scaling window (fixed, never tuned):
    lon_min = -10
    lat_min =  30
    lon_max =  42
    lat_max =  47
```

All four AOIs lie strictly inside it (doc 06 §2).

```
For each climate variable v, over valid TerraClimate land pixels
of the reference window, on the same 1991-2020 climatology:

    ref_mean_v
    ref_sd_v          (population SD, ddof = 0)

Standardise each AOI value:

    s_v(aoi) = ( climate_value(aoi, v) - ref_mean_v ) / ref_sd_v
```

The reference window uses the **same native TerraClimate land support** as the
AOI summaries — one mask definition throughout.

---

## 5. Climatic distance

```
climate_distance(A, B)
    = sqrt(  mean over the four climate variables of
             ( s_v(A) - s_v(B) )^2  )

    = sqrt( (1/4) * Σ_v ( s_v(A) - s_v(B) )^2 )
```

| Option | Status |
|---|---|
| Standardised Euclidean over a broad reference distribution | **ADOPTED** |
| Mahalanobis | **Rejected.** The covariance would have to be estimated from 4 points in 4 dimensions; the sample covariance has rank at most 3 and is non-invertible. Regularising it would make the metric depend entirely on an arbitrary ridge parameter. |
| Gower | Rejected — all four variables are continuous, so Gower reduces to a range-normalised Manhattan distance with the range again estimated from 4 points. |

Properties:

- **Symmetric.** `climate_distance(A,B) == climate_distance(B,A)` exactly, bit
  for bit, since the expression depends only on `(s(A) − s(B))²`. Asserted by a
  validator check and a test.
- **Dimensionless**, all four axes on a common Mediterranean-relative scale.
- **Reproducible** from the collection ID, the period, the band list, the season
  months, the reference window and the land-support rule — all pinned.
- **Stable under adding a fifth AOI**, because the scale comes from the reference
  window and not from the AOI set.
- The `1/4` factor makes the value a root-mean-square per-variable difference in
  reference-SD units: a distance of 1.0 means the two AOIs differ by about one
  Mediterranean SD on a typical climate axis.

**Equal weighting of the four variables is a preregistered choice, not a
derivation.** No defensible non-arbitrary weighting is available from four AOIs,
and an importance-derived weighting would reintroduce the source model into a
quantity that should be independent of it. Removing the two duplicate axes (§2)
is what makes equal weighting defensible: each of the four now carries a distinct
mechanism.

### Component contributions

```
climate_component_contributions[v] = ( s_v(A) - s_v(B) )^2 / 4

Σ_v climate_component_contributions[v] == climate_distance^2      exactly
```

A validator check asserts the identity to 1e-12. This is what lets the advisor
see *which* climate axis separates a given pair.

---

## 6. Output fields

Per unordered AOI pair (6 rows in `pairwise_climate_distance.csv`), echoed onto
all 12 directed rows in the main table:

| Field | Content |
|---|---|
| `climate_distance` | the scalar above, full float precision |
| `climate_distance_metric` | `"standardised_euclidean_mediterranean_reference_v1"` |
| `climate_feature_count` | `4` |
| `climate_features` | `["annual_mean_temperature_c", "annual_precipitation_mm", "warm_season_climatic_water_deficit_mm", "warm_season_vpd_kpa"]` |
| `climate_reference_period` | `"1991-01-01/2020-12-31"` |
| `climate_season_months` | `[6, 7, 8, 9]` |
| `climate_scaling_contract` | `"z-score against valid TerraClimate land pixels of lon[-10,42] lat[30,47] on the same 1991-2020 climatology, population SD ddof=0"` |
| `climate_source_version` | `"IDAHO_EPSCOR/TERRACLIMATE"` plus the resolved asset version captured at export time |
| `climate_land_mask` | `"terraclimate_native_valid_land_support"` |
| `climate_band_scale_factors` | the factors actually applied, per band, as read from band metadata |
| `climate_data_completeness` | per-AOI fraction of valid land pixels carrying a value for all four variables |
| `climate_component_contributions` | per-variable, summing to `climate_distance²` |
| `climate_status` | `"authorised_pending_export"` before the export runs; `"available"` after |
| `climate_export_authorised` | `true` |
| `climate_uncertainty` | `"deterministic_aoi_level_value_no_interval"` |

---

## 7. Export contract

**Authorised. Not yet performed.**

```
task                 climate_normals_export
collection           IDAHO_EPSCOR/TERRACLIMATE
period               1991-01-01 .. 2020-12-31
bands                tmmx, tmmn, pr, def, vpd
season_filter        calendar months 6,7,8,9   (deficit and VPD aggregates)
annual_aggregates    all 12 months             (temperature and precipitation)
land_support         TerraClimate native valid-data mask
regions              4 AOI bboxes  +  the Mediterranean reference window
scale                4638 m  (TerraClimate native)
crs                  EPSG:4326
export_transport     reuse export_image_direct_or_tiled(), as every other export does
destination          outputs/experiments/<exp>/data/climate/terraclimate_normals_1991_2020.tif
                     outputs/diagnostics/climate_reference/mediterranean_reference_1991_2020.tif
metadata             climate_export_metadata.json, mirroring predictor_export_metadata.json:
                       collection, resolved asset version, band scale factors as applied,
                       period, season months, land-support rule, scale, crs, transport,
                       tile grid, created_at, per-band valid-pixel counts,
                       sha256 of each tif
```

Volume: four small AOIs at ~4.6 km is a few hundred kilobytes each; the reference
window is roughly 1250 × 410 pixels per band, a few megabytes. Small by this
project's standards.

**The export runs in a separate script** (`scripts/export_climate_normals.py`)
which carries `gee_query_issued = true`. The completion module reads the exported
rasters only and carries `gee_query_issued = false`. The two must never share a
module — validator check 24.

---

## 8. Limitations to carry

- A 1991–2020 normal describes the climate the AOIs sit in, not the weather of
  the fire seasons. Bejís 2022 and the three 2021 events differ in event-year
  conditions in ways this component deliberately does not capture — that is what
  `lst_anomaly_mean` does, inside the predictor space.
- TerraClimate is an interpolated/downscaled product with known biases in complex
  coastal terrain, which is the terrain of all four AOIs.
- Four AOIs give six unordered climate distances. Six points can describe a
  pattern; they cannot establish a relationship between climatic distance and
  transfer performance.
- The four-variable equal weighting is preregistered, not derived.
- The AOI summary is a spatial mean over a bbox. All four AOIs have
  coastal-to-montane relief, and the compression to a single point is larger for
  the larger AOIs — Muğla's bbox is 1.80° × 0.85°, more than five times the area
  of Bejís's 0.70° × 0.47°.
- No ERA5-Land cross-check was run, so the sensitivity of the ordering to the
  choice of reanalysis product is untested in this run.
