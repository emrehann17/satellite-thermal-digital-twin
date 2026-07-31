# 06. Geographic Distance — Exact Design

Machine-readable companion: `canonical_geometry_inventory.csv`.

**No distance was computed.** This document specifies the method; the numbers in
§2 are the geometry constants themselves and their arithmetic centres, which are
properties of the frozen geometry contract, not analysis results.

---

## 1. Source of truth

```
file        core/regions.py
sha256      980eb5d4cf459ee52bf065f3b2fb2d644fb72449d62c0f8dfba1c58c93396275
tracked     yes
clean       yes  (git status --short reports nothing for this path)
last commit 0a3c5fe85926a3ec3204c810fc72d2fe93afe923
            "feat(regions): register extended North Evia experiment"
resolvable at HEAD 19d825bb7dc21459aebe0870828cb25d8fc2a892:  yes
```

Two layers, both in the same file:

1. **Module-level bbox constants**, deliberately defined as plain Python tuples so
   that "tests (and provenance/hash logic) can verify AOI coordinates **without**
   GEE auth" — the file says so explicitly at `core/regions.py:33-40`. This is the
   layer the completion module must read.
2. `build_regions()`, which constructs `ee.Geometry.BBox(*CONSTANT)` from those
   same constants. Reading this layer would require an Earth Engine session and a
   `.getInfo()` server call. **The completion module must not use it.**

The experiment → region key mapping comes from the `EXPERIMENTS` registry in the
same file, resolved through `core.regions.get_experiment`.

| Experiment | `region_key` | Geometry constant |
|---|---|---|
| `manavgat_2021` | `manavgat_aoi` | `manavgat_aoi_refined_bbox`, inline literal at `core/regions.py:183` |
| `bejis_2022` | `bejis_aoi` | inline literal at `core/regions.py:216` |
| `mugla_2021` | `mugla_aoi` | `MUGLA_AOI_BBOX` |
| `evia_2021_extended` | `north_evia_extended` | `NORTH_EVIA_EXTENDED_AOI_BBOX` |

**Provenance asymmetry to record.** Muğla and extended-Evia are defined by named
module-level constants; Manavgat and Bejís are defined by inline
`ee.Geometry.BBox(...)` literals inside `build_regions()`. Reading the latter two
without a GEE session requires either parsing the literal or promoting it to a
module-level constant. **Recommendation: read all four from a single explicit
mapping declared in the completion module, hard-pinned to the values below, with
a test that asserts the mapping matches `core/regions.py`.** This avoids both a
GEE dependency and a production-code change. Promoting the two inline literals to
module constants would be cleaner but is a production edit and is therefore out
of scope for this task; it is recorded as decision A-10b.

---

## 2. AOI geometry inventory

All four are **axis-aligned rectangles in EPSG:4326**, given in
`ee.Geometry.BBox` argument order `(lon_min, lat_min, lon_max, lat_max)`.

| Experiment | lon_min | lat_min | lon_max | lat_max | Width (°) | Height (°) | bbox centre lon | bbox centre lat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bejis_2022` | −1.05 | 39.68 | −0.35 | 40.15 | 0.70 | 0.47 | −0.700000 | 39.915000 |
| `evia_2021_extended` | 23.05 | 38.55 | 23.85 | 39.15 | 0.80 | 0.60 | 23.450000 | 38.850000 |
| `manavgat_2021` | 31.05 | 36.72 | 31.85 | 37.35 | 0.80 | 0.63 | 31.450000 | 37.035000 |
| `mugla_2021` | 27.10 | 36.60 | 28.90 | 37.45 | 1.80 | 0.85 | 28.000000 | 37.025000 |

All four lie strictly inside the Mediterranean reference window proposed in doc
05 §3 (`lon [−10, 42]`, `lat [30, 47]`).

None of the four is `evia_2021` — the narrower bbox `(23.12, 38.68, 23.52, 39.08)`
belongs to the separate `evia_2021` experiment, which is **not** in this analysis
set. The extended AOI geometrically contains it on all four sides. A validator
check must assert that the resolved geometry for `evia_2021_extended` is the
extended bbox and not the narrow one, because the two experiment IDs differ by a
suffix and are easy to confuse.

### Deterministic geometry representation

For hashing and provenance:

```
geometry_contract_string = "{experiment_id}|EPSG:4326|bbox|{lon_min!r},{lat_min!r},{lon_max!r},{lat_max!r}"
geometry_contract_hash   = sha256(geometry_contract_string.encode("utf-8")).hexdigest()
```

Using `repr()` on the floats guarantees a round-trippable decimal representation,
so the hash is stable across platforms. The per-AOI hashes must be computed at
implementation time and recorded; they are not precomputed here because the exact
string format is still decision A-10a.

---

## 3. Centroid definition

| Option | Assessment |
|---|---|
| **Bounding-box centre** | **ADOPTED — the only centroid in this design.** |
| Geometry centroid | For an axis-aligned rectangle these are the same point in the lon/lat plane — see below. |
| Population-weighted centroid | **Excluded from the completion run entirely.** See below. |
| Valid-cell centroid | Excluded, same reasoning. |

### Geometry centroid vs bbox centre

For an axis-aligned rectangle in the EPSG:4326 lon/lat plane, the planar geometry
centroid **is** the bbox centre, exactly:
`((lon_min+lon_max)/2, (lat_min+lat_max)/2)`.

The two would differ only under a spherical/geodesic area-weighted definition,
where the `cos(lat)` area weighting pulls the centroid slightly toward the
equator. For these AOIs the latitudinal extent is 0.47°–0.85°, and the resulting
displacement is well under 100 m — negligible against inter-AOI distances of
hundreds to thousands of kilometres, but non-zero.

**Adopted:** the planar bbox centre, declared explicitly as
`centroid_definition = "bbox_centre_planar_epsg4326"` so that the choice is
recorded rather than left ambiguous. The alternative spherical definition is not
computed.

### Why the population centroid is excluded entirely

An earlier draft proposed reporting a Step8A population-weighted centroid as a
secondary column. **It is removed from the design.** The geographic component
reads no Step8A data at all.

The argument for it was real: for Muğla and Evia the population centroid differs
materially from the bbox centre, because those bboxes are 38.9% and 57.7%
permanent water respectively (doc 01 §6), and Muğla's burnable population sits in
separated peninsular clusters. But three reasons decide against it, and the third
is the one that settles it:

1. **It depends on analysis products.** The valid mask, the landcover raster and
   the `analysis_eligible` flag all feed it, so it would change if Step8A were
   ever regenerated — silently changing a "geographic" distance that ought to be
   a fixed property of where the study areas are.
2. **It is not exactly recomputable from a committed constant.** The bbox centre
   is reproducible from four numbers in a tracked file; the population centroid
   requires reading a 41 731-row git-ignored parquet.
3. **It would make the geographic component read target predictor values.**
   `lon`/`lat` are in `FORBIDDEN_MODEL_COLUMNS`, and computing a target-side
   population centroid means opening the target Step8A frame. Keeping the
   component to pure geometry means the firewall statement for it is
   unconditional — *the geographic distance reads no target data of any kind* —
   rather than carrying an exemption clause.

Consequence to disclose: for Muğla and Evia the bbox centre is not where the
modelled cells are. That is stated in the limitations (§8) rather than patched
with a second centroid.

```
geographic_component_reads_step8a            = false
geographic_component_reads_target_predictors = false
population_centroid_reported                 = false
```

---

## 4. Distance method

```
geographic_distance_method = "wgs84_geodesic_inverse_km"
ellipsoid                  = WGS84  (a = 6378137.0 m, f = 1/298.257223563)
output                     = kilometres, full float64 precision, never rounded in storage
```

Primary metric:

```
centroid_geodesic_distance_km(A, B)
    = geodesic_inverse( bbox_centre(A), bbox_centre(B) ).distance / 1000.0
```

Symmetric by construction. A validator check and a test assert
`d(A,B) == d(B,A)` exactly.

### Dependency decision — RESOLVED

> **Decision C-11 is ACCEPTED: use the pinned Python package `geographiclib`.**
> It is no longer an implementation blocker.

```
geodesic_implementation = "geographiclib_wgs84"
api                     = geographiclib.geodesic.Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
distance_field          = result["s12"]   (metres) -> / 1000.0 for kilometres
```

`geographiclib` is the reference implementation of Karney's algorithm: exact to
round-off, always convergent, pure Python, no compiled extension, no PROJ. It is
the correct tool and it removes the need for any hand-rolled geodesy.

**Rejected alternatives:**

| Option | Why not |
|---|---|
| Custom Vincenty inverse in-repo | An earlier draft recommended this. **Rejected.** Vincenty fails to converge for near-antipodal pairs and requires a hand-written iteration cap, a fail-closed branch and its own reference-pair test suite — a meaningful amount of numerical code to own and maintain in exchange for avoiding one small pure-Python dependency. Karney's algorithm supersedes Vincenty precisely on this point. |
| Haversine on a sphere | **Rejected as primary.** Spherical-earth error reaches ~0.5%, several kilometres over these distances. |
| `pyproj` | Also uses Karney, but pulls in compiled PROJ. `geographiclib` gives the same algorithm with no build dependency. |
| `rasterio`'s bundled PROJ | Rejected — `rasterio.warp` does coordinate transformation, not geodesic inverse. |

**Installation state and process.** Neither `geographiclib` nor `pyproj`,
`geopy` or `shapely` is currently installed or listed in `requirements.txt` /
`requirements-lock.txt` — verified by inspecting
`venv/lib/python*/site-packages/` and grepping the lock file. Available and
relevant today: `numpy==2.5.1`, `scipy==1.18.0`, `rasterio==1.5.0`,
`scikit-learn==1.9.0`, `pyarrow==24.0.0`.

Therefore:

- The implementation stage **may** add `geographiclib` to `requirements.txt` and
  `requirements-lock.txt`.
- **Installation is a user-run step and must be separately reviewed.** No
  auto-install, no `pip install` invoked from any script, no silent dependency
  resolution at import time.
- The completion module must fail closed with a clear message if the package is
  absent, rather than falling back to haversine or to a local approximation. A
  missing dependency must never silently degrade the metric.

Required tests remain:

- at least three published WGS84 reference pairs agree to ≤ 1 mm,
- `d(A,B) == d(B,A)` exactly,
- `d(A,A) == 0.0` exactly,
- absence of `geographiclib` raises rather than falling back.

---

## 5. Secondary metric: minimum boundary-to-boundary distance

Proposed as a **secondary, advisor-interpretation column only**, never as the
primary.

```
optional_minimum_boundary_distance_km(A, B)
    = min over  p ∈ ∂A,  q ∈ ∂B   of   geodesic(p, q)
```

For two axis-aligned lon/lat rectangles this reduces to a small closed-form case
analysis on the lon and lat interval gaps, evaluated at the closest latitude.

**Why it is worth reporting here specifically:** Muğla's bbox is 1.80° wide and
Manavgat's is 0.80° wide, and their bbox centres are 2.55° of longitude apart —
but their eastern and western edges are only 2.15° apart. Centre-to-centre
distance systematically overstates the separation of large AOIs. Reporting both
lets the advisor see when AOI extent, rather than AOI location, is driving the
number.

**Why it is not primary:** it is not a distance between the AOIs as objects — it
is a distance between their closest corners, and it goes to zero for any two
overlapping or adjacent AOIs regardless of how different they are. It is also
more sensitive to the arbitrary generosity of a hand-drawn bbox than the centre
is. The four bboxes here are all explicitly documented as working AOIs that were
"not an exact fire perimeter", so their edges carry less meaning than their
locations.

---

## 6. Proposed output fields

Per unordered AOI pair (6 rows in `pairwise_geographic_distance.csv`), echoed onto
all 12 directed rows:

| Field | Content |
|---|---|
| `source_centroid_lon` | bbox centre longitude of the source AOI |
| `source_centroid_lat` | bbox centre latitude of the source AOI |
| `target_centroid_lon` | bbox centre longitude of the target AOI |
| `target_centroid_lat` | bbox centre latitude of the target AOI |
| `centroid_geodesic_distance_km` | primary metric, full precision |
| `optional_minimum_boundary_distance_km` | secondary metric |
| `geographic_distance_method` | `"wgs84_geodesic_inverse_km"` |
| `centroid_definition` | `"bbox_centre_planar_epsg4326"` |
| `geodesic_implementation` | `"geographiclib_wgs84"` |
| `geodesic_package_version` | the resolved `geographiclib` version, recorded at run time |
| `geometry_source_path` | `"core/regions.py"` |
| `geometry_source_sha256` | `980eb5d4cf459ee52bf065f3b2fb2d644fb72449d62c0f8dfba1c58c93396275` |
| `geometry_source_commit` | `0a3c5fe85926a3ec3204c810fc72d2fe93afe923` |
| `source_geometry_contract_hash` | per-AOI hash from §2 |
| `target_geometry_contract_hash` | per-AOI hash from §2 |
| `source_bbox` / `target_bbox` | the four coordinates, stored explicitly so the value is recomputable without the repository |
| `geographic_component_reads_step8a` | `false` |
| `population_centroid_reported` | `false` |
| `geographic_distance_uncertainty` | `"deterministic_aoi_level_value_no_interval"` |

No population-centroid column appears in this artifact.

Storing the raw bboxes on every row is deliberate: it makes
`centroid_geodesic_distance_km` independently recomputable by the advisor from the
artifact alone, which is validator check 18.

---

## 7. Firewall properties

| Property | Status |
|---|---|
| Symmetric | Yes, by construction; asserted by test |
| Target label used | No |
| Target predictor values used | **No — unconditionally.** The component reads only `core/regions.py` geometry constants. It opens no Step8A frame at all. |
| Step8A read | **No** |
| Transfer metric used | No |
| Stored precision | full float64 km, never rounded in storage; rounding is a presentation choice only |
| Deterministic | Yes; no seed, no sampling, no fitting |

---

## 8. Limitations

- Six unordered distances from four AOIs. This can describe a pattern; it cannot
  establish a relationship between geographic separation and transfer
  performance.
- The AOIs are hand-drawn working rectangles, not fire perimeters, and their
  extents differ by more than a factor of five in area. A centroid distance
  compresses a 1.80° × 0.85° AOI and a 0.70° × 0.47° AOI to single points that are
  not equally representative of their AOIs.
- Geographic distance in the Mediterranean is a poor proxy for ecological
  similarity: Bejís (Spain) and Manavgat (Turkey) are far apart and both are
  Mediterranean pine systems, while Manavgat's coastal and montane portions differ
  sharply within a few tens of kilometres.
- **The bbox centre of a mostly-marine AOI is not where the modelled cells are.**
  Muğla's bbox is 38.9% permanent water and Evia's is 57.7%, so for those two the
  reported centroid sits away from the burnable population — in Muğla's case
  possibly in the sea between two peninsular clusters. No population centroid is
  reported to correct for this, by design (§3): the component is kept to pure
  geometry so that it reads no Step8A data. The displacement is a known and
  disclosed property of the metric, not a hidden one.
- The minimum boundary distance (§5) partly compensates, since it is driven by
  the nearest edges rather than by the centres, and it should be read alongside
  the primary for the two large AOIs.
