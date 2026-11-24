from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_HOURS = (1, 3, 6, 12, 24)


def utc_nanoseconds(values: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(values, utc=True, format="mixed", errors="coerce")
    return parsed.to_numpy(dtype="datetime64[ns]").astype("int64")


def main() -> None:
    atl_path = ROOT / "Data/Processed/ATL24/atl24_bathy_points.csv.gz"
    scene_path = ROOT / "Data/Manifests/sentinel2_selection.csv"
    if not atl_path.is_file():
        raise SystemExit("Run Scripts/06_extract_atl24_points.py first.")
    if not scene_path.is_file():
        raise SystemExit("Run Scripts/04_catalog_sentinel2.py first.")

    photons = pd.read_csv(atl_path, usecols=["granule_id", "utc_time_source_epoch"])
    scenes = pd.read_csv(
        scene_path,
        usecols=[
            "item_id",
            "acquisition_utc",
            "acquisition_date",
            "platform",
            "tile_cloud_percent",
            "selected",
        ],
        encoding="utf-8-sig",
    )
    photons = photons.dropna(subset=["granule_id", "utc_time_source_epoch"]).copy()
    scenes = scenes.dropna(subset=["item_id", "acquisition_utc"]).copy()
    if photons.empty or scenes.empty:
        raise SystemExit("The ATL24 point table or Sentinel-2 catalogue has no usable timestamps.")

    atl_ns = utc_nanoseconds(photons["utc_time_source_epoch"])
    s2_ns = utc_nanoseconds(scenes["acquisition_utc"])
    atl_ok = atl_ns != np.iinfo("int64").min
    s2_ok = s2_ns != np.iinfo("int64").min
    photons = photons.loc[atl_ok].reset_index(drop=True)
    scenes = scenes.loc[s2_ok].reset_index(drop=True)
    atl_ns = atl_ns[atl_ok]
    s2_ns = s2_ns[s2_ok]
    order = np.argsort(atl_ns, kind="stable")
    atl_ns = atl_ns[order]
    photons = photons.iloc[order].reset_index(drop=True)

    nearest_right = np.searchsorted(atl_ns, s2_ns, side="left")
    right = np.clip(nearest_right, 0, len(atl_ns) - 1)
    left = np.clip(nearest_right - 1, 0, len(atl_ns) - 1)
    use_left = np.abs(atl_ns[left] - s2_ns) <= np.abs(atl_ns[right] - s2_ns)
    nearest = np.where(use_left, left, right)
    scenes["nearest_photon_gap_min"] = np.abs(atl_ns[nearest] - s2_ns) / 60_000_000_000
    scenes["nearest_photon_utc"] = pd.to_datetime(atl_ns[nearest], utc=True).astype(str)
    granule_values = photons["granule_id"].to_numpy()
    scenes["nearest_atl24_granule"] = granule_values[nearest]

    granules_represented: dict[int, int] = {}
    for hours in WINDOWS_HOURS:
        half_window_ns = int(hours * 60 * 60 * 1_000_000_000)
        lower = np.searchsorted(atl_ns, s2_ns - half_window_ns, side="left")
        upper = np.searchsorted(atl_ns, s2_ns + half_window_ns, side="right")
        point_counts = upper - lower
        unique_granules = np.zeros(len(scenes), dtype="int32")
        represented: set[str] = set()
        for i in np.flatnonzero(point_counts):
            nearby = np.unique(granule_values[lower[i] : upper[i]])
            unique_granules[i] = len(nearby)
            represented.update(str(value) for value in nearby)
        scenes[f"atl24_points_pm_{hours}h"] = point_counts.astype("int32")
        scenes[f"atl24_granules_pm_{hours}h"] = unique_granules
        scenes[f"temporal_candidate_pm_{hours}h"] = point_counts > 0
        granules_represented[hours] = len(represented)

    out_dir = ROOT / "Results/Matchup_Audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "sentinel2_atl24_temporal_matchup_audit.csv"
    scenes.sort_values(["nearest_photon_gap_min", "acquisition_utc", "item_id"]).to_csv(
        table_path, index=False
    )

    summary = {
        "catalog_scene_items": int(len(scenes)),
        "catalog_distinct_dates": int(scenes["acquisition_date"].nunique()),
        "atl24_retained_points": int(len(photons)),
        "atl24_granules_with_retained_points": int(photons["granule_id"].nunique()),
        "windows_are_symmetric_about_sentinel_acquisition": True,
        "window_summaries": {},
        "limitations": [
            "Temporal matching only; no per-pixel cloud, water, or image-footprint QA was performed.",
            "Scene counts can include duplicate products or multiple items on the same date.",
            "Wider windows are not exact co-temporal bathymetric matchups; tide and sea-level change must be assessed.",
            "The selected column in the original Sentinel-2 catalogue was not changed.",
        ],
    }
    for hours in WINDOWS_HOURS:
        point_col = f"atl24_points_pm_{hours}h"
        granule_col = f"atl24_granules_pm_{hours}h"
        candidates = scenes.loc[scenes[point_col] > 0]
        summary["window_summaries"][f"{hours}h"] = {
            "candidate_scene_items": int(len(candidates)),
            "candidate_distinct_dates": int(candidates["acquisition_date"].nunique()),
            "unique_atl24_granules_represented": granules_represented[hours],
            "scene_photon_pairs_sum_including_duplicate_items": int(candidates[point_col].sum()),
            "maximum_granules_near_one_scene": int(candidates[granule_col].max()) if len(candidates) else 0,
        }

    summary_path = out_dir / "sentinel2_atl24_temporal_matchup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Read-only Sentinel-2/ATL24 temporal matchup audit complete.")
    print(f"Catalogue items: {len(scenes):,}; retained ATL24 points: {len(photons):,}")
    for hours in WINDOWS_HOURS:
        info = summary["window_summaries"][f"{hours}h"]
        print(
            f"  ±{hours} h: {info['candidate_scene_items']} scene items / "
            f"{info['candidate_distinct_dates']} dates; "
            f"{info['unique_atl24_granules_represented']} ATL24 granules represented"
        )
    print(f"Audit table: {table_path}")
    print(f"Summary: {summary_path}")
    print("No scene was selected or downloaded; wider windows require explicit tide/sea-level sensitivity analysis.")


if __name__ == "__main__":
    main()
