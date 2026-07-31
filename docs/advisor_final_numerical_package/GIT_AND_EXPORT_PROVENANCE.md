# Git and export provenance

## Git before documentation

- Old advisor commit: `c648486...`
- Final HEAD: `483027a38148319099b20b97f2307d5457c51260`
- Branch: `main`
- Remote: `origin https://github.com/emrehann17/satellite-thermal-digital-twin` (fetch/push)
- Tracking upstream: not configured for `main`
- Remote-tracking comparison: `HEAD` equals `refs/remotes/origin/main` (ahead 0, behind 0): **local HEAD matches upstream**
- Source working tree before documentation: clean.

## Canonical inputs

| experiment | dataset_path | sha256 | row_count | positive_count | negative_count | prevalence | prevalence_percent | schema_version | manifest_path | manifest_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| manavgat_2021 | outputs/experiments/manavgat_2021/step8a/step8a_500m_modeling_dataset.parquet | 054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439 | 20511 | 784 | 19727 | 0.03822339232606894 | 3.8223392326068937 | step8a.modeling_dataset; primary_population=burnable_tree_shrub_grass | outputs/experiments/manavgat_2021/step8a/step8a_dataset_stats.json | 18170d2a67aa17e2d515dbd782c69bc7e43f08062d3d5b15de410795d2898cd5 |
| bejis_2022 | outputs/experiments/bejis_2022/step8a/step8a_500m_modeling_dataset.parquet | 3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393 | 15190 | 1100 | 14090 | 0.07241606319947334 | 7.2416063199473335 | step8a.modeling_dataset; primary_population=burnable_tree_shrub_grass | outputs/experiments/bejis_2022/step8a/step8a_dataset_stats.json | af8f7ea2db814f065ca8632ba7de486a5dfbaed0eaaad8a9cf6296ca4ea48e23 |
| mugla_2021 | outputs/experiments/mugla_2021/step8a/step8a_500m_modeling_dataset.parquet | c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e | 41730 | 2911 | 38819 | 0.06975796788880902 | 6.975796788880901 | step8a.modeling_dataset; primary_population=burnable_tree_shrub_grass | outputs/experiments/mugla_2021/step8a/step8a_dataset_stats.json | c9bebf854270393de2e4a75e1bb1d6ce21bdddc00017498ee54ea6802b62ebd8 |
| evia_2021_extended | outputs/experiments/evia_2021_extended/step8a/step8a_500m_modeling_dataset.parquet | bdce859cf482f575d0f273174b157f47efd61779953fdd23d9486c5face5e553 | 9298 | 2664 | 6634 | 0.28651322865132284 | 28.651322865132286 | step8a.modeling_dataset; primary_population=burnable_tree_shrub_grass | outputs/experiments/evia_2021_extended/step8a/step8a_dataset_stats.json | 8ba510dacff226db389a406f736372fedf511e83edc95a8f4cec332a72aef74f |

The requested “manifest” field uses each canonical Step8A `step8a_dataset_stats.json`, because no common root `manifest.json` exists for all four experiments; its SHA256 is reported explicitly.

## TerraClimate/export contract

- Collection ID: `IDAHO_EPSCOR/TERRACLIMATE`
- Climatology: `1991-01-01/2020-12-31`; expected months `360`
- Bands: `tmmn, tmmx, def, vpd, pr`
- Projection/CRS resolution method: `ee_single_band_projection_with_wkt_fallback_v1`
- CRS: WKT geographic CRS (`export_crs_representation=wkt`; semantic-equivalence QA passed)
- WKT fallback used: yes; the recorded method is `ee_single_band_projection_with_wkt_fallback_v1` and the exported CRS representation is WKT
- Native source transform: `[0.041666666666666664, 0.0, -180.0, 0.0, -0.041666666666666664, 90.0]`; audited output transform: `[0.0416638628774634, 0.0, -10.04099095346868, 0.0, -0.0416638628774634, 47.038501188656184]`
- Nominal scale: `4638.312116386398` m
- Provenance method token: `ee_single_band_projection_with_wkt_fallback_v1`
- Export timestamp: `2026-08-02T17:53:03.244633+00:00`

## Software

- Python version recorded for documentation environment: `3.12.3` (last analysis runtime version: `NOT_VERIFIED_FROM_FINAL_ARTIFACT` unless stated in a source manifest)
- `requirements-lock.txt` SHA256: `7d735558594ccf3cf824beb0aeac42789349eb7f741b187bbd5730a2d382f84a`
- Few-shot package versions: `{'numpy': '2.4.4', 'pandas': '3.0.2', 'scikit-learn': '1.9.0'}`
- Muğla-subsampling package versions: `{'numpy': '2.4.4', 'pandas': '3.0.2', 'scikit-learn': '1.9.0'}`
- Environment/provenance manifest SHA256: few-shot `3aa6634c69cddf6521b78a828de450a79c9b1cc420946416b824ad3819bd9038`; Muğla subsampling `915ede272bc0550a746f3b5d29fe009fcc0a4ead2263c06b0625771d88b341d1`; CORAL `4addf0b81bae48980a5ac1b432b7b9d1119909de720e6f7e76e46c6dc728f544`
