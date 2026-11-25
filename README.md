# Open-data coastal bathymetry and sediment-mobility assessment at Anza–Agadir, Morocco

This repository contains the reproducible analysis pipeline for assessing what open satellite, bathymetric, and regional hydrodynamic data can—and cannot—support at the Anza–Agadir coast of Morocco.

The study combines ICESat-2 ATL24 coastal bathymetric photons, Sentinel-2 L2A imagery, EMODnet bathymetry, and Copernicus Marine IBI sea-level, wave, and current products. It evaluates photon-to-image matchups, track-held-out bathymetry predictions, sensitivity to relative sea-level adjustment, and scenario-based sediment-mobility screening.

## Scientific scope

The workflow is designed to make data support and uncertainty visible before interpreting coastal bathymetry or sediment mobility. It is an assessment and screening pipeline, not a validated wall-to-wall bathymetric product or a sediment-transport model.

Key interpretation limits:

- ATL24 provides along-track photon observations, not continuous ground-truth bathymetry. Validation holds out complete ATL24 granules where supported.
- Sentinel-2 and ATL24 acquisition times may differ. The 0–6 h matchup is treated as the primary near-synchronous window; wider windows are exploratory.
- IBI sea-surface height is used for relative temporal sensitivity only. It is not a local tide-gauge record or a vertical datum tie.
- EMODnet is regional bathymetric context. Its vertical reference is not assumed to match the local hydrographic zero.
- IBI wave and current products are regional and do not resolve surf-zone processes.
- Grain sizes are scenarios in the absence of representative local sediment samples. Threshold exceedance is not a measured sediment flux or a probability of transport.
- Weak or negative out-of-sample skill does not justify presenting a continuous satellite-derived bathymetry map as validated.

## Data sources

| Data | Role | Official source |
|---|---|---|
| ICESat-2 ATL24 v003 | Coastal bathymetric photons and surface observations | [NSIDC ATL24 v003](https://nsidc.org/data/atl24/versions/3) |
| Sentinel-2 L2A | Optical reflectance and scene classification | [Earth Search STAC](https://earth-search.aws.element84.com/v1/) |
| EMODnet Bathymetry | Regional bathymetric context and comparison | [EMODnet web services](https://emodnet.ec.europa.eu/en/emodnet-web-service-documentation) |
| Copernicus Marine IBI | Regional sea-level, wave, and current forcing | [Copernicus Marine Data Store](https://data.marine.copernicus.eu/) |
| GOT tide model | Harmonic tide-model context and anomaly checks | [pyTMD](https://pytmd.readthedocs.io/) |

Large source files may be excluded from Git. Check `Data/Manifests/` and the acquisition scripts to see what is inventoried or must be downloaded locally. Data providers retain their own terms of use and attribution requirements.

## Repository layout

- `Scripts/` — numbered inventory, download, processing, audit, analysis, and figure-generation scripts.
- `Data/Manifests/` — scene and granule inventories and selection tables.
- `Data/Raw/` — locally acquired source data; some large files are intentionally not versioned.
- `Data/Processed/` — derived raster and tabular inputs where included.
- `Results/` — analysis tables, diagnostics, and figures.
- `config.json` — AOI, date range, product choices, and scenario parameters.
- `requirements.txt` — Python dependencies.
- `PIPELINE_POWERSHELL_FR.md` — detailed command-by-command workflow in French.
- `README_FR.md` — French project description and research-design context.

## Setup

The documented workflow targets Windows PowerShell and Python 3.11. From the repository root:

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
& ".\.venv\Scripts\python.exe" .\Scripts\00_preflight.py
```

Read `PIPELINE_POWERSHELL_FR.md` before running the analysis. The pipeline starts with inventories and read-only audits, then downloads only after the relevant selections are reviewed. Earthdata and Copernicus Marine downloads require the user's own accounts. Do not place passwords, tokens, or credential files in this repository; local credential paths are ignored by Git.

## Analysis stages

1. Check the environment and inventory ATL24 granules, Sentinel-2 scenes, and regional bathymetric/forcing coverage.
2. Download the selected open products and audit completeness, quality flags, timestamps, grid support, and vertical-reference metadata.
3. Extract ATL24 photons and process Sentinel-2 reflectance and clear-water masks.
4. Match optical and photon observations under explicit time and distance windows; compare raw depth with a same-cell relative IBI sea-level sensitivity target.
5. Evaluate predictions with granule-held-out and buffered spatial-block diagnostics; report support counts and baseline comparisons.
6. Run depth-stratified and paired-target sensitivity analyses, then calculate conditional wave–current mobility scenarios.
7. Generate figures and interpretation notes from the saved analysis outputs.

Scripts are numbered by stage. Use the pipeline guide for exact command order and options; do not infer that later-numbered stages can run before their required inputs exist.

## Reproducibility and credentials

The scripts record inventories, analysis tables, and diagnostics under `Data/Manifests/` and `Results/`. Keep downloaded source files unchanged and retain the manifests alongside derived outputs. The `.gitignore` excludes local virtual environments, credentials, references, and manuscript material. Never commit `.dodsrc`, Earthdata credentials, Copernicus credentials, or access tokens.

## Current status

This is an active research workflow. Its conclusions are conditional on the sampled ATL24 tracks, selected cloud-free Sentinel-2 observations, regional IBI forcing, and stated bathymetric and grain-size scenarios. It does not claim local tide-gauge validation, an absolute vertical datum connection, surf-zone-resolved hydrodynamics, measured sediment transport, or guaranteed journal acceptance.
