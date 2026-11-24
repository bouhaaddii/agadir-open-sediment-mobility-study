from __future__ import annotations

import csv
import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
TIDE_MODEL_DIR = ROOT / "Data/Raw/Tide_Models"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def reflectance_scale(src: rasterio.io.DatasetReader, arr: np.ndarray) -> float:
    scale = float(src.scales[0]) if src.scales else 1.0
    if 0 < scale < 0.01:
        return scale
    tags = src.tags(1)
    for key in ("scale", "reflectance_scale", "scale_factor"):
        try:
            value = float(tags.get(key, ""))
            if 0 < value < 0.01:
                return value
        except (TypeError, ValueError):
            pass
    return 1e-4 if np.issubdtype(arr.dtype, np.integer) and np.nanmax(arr) > 2 else 1.0


def predict_got56_no_extrapolation(lon: np.ndarray, lat: np.ndarray, time_utc) -> np.ndarray:
    """Predict GOT5.6 elevations at point/time pairs; unsupported cells stay NaN."""
    import pyTMD

    times = pd.to_datetime(pd.Series(time_utc), utc=True, errors="coerce")
    if times.isna().any():
        raise ValueError("Invalid UTC timestamp in ATL24/Sentinel-2 matchup.")
    delta_time = times.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    result = pyTMD.compute.tide_elevations(
        x=np.asarray(lon, dtype="float64"),
        y=np.asarray(lat, dtype="float64"),
        delta_time=delta_time,
        directory=TIDE_MODEL_DIR,
        model="GOT5.6",
        crs="EPSG:4326",
        type="trajectory",
        standard="datetime",
        method="linear",
        extrapolate=False,
        infer_minor=True,
    )
    return np.asarray(result.values, dtype="float64").reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate Sentinel-2 SDB models against tide-normalized ATL24 points.")
    parser.add_argument("--selection-file", default=None, help="Optional selection CSV relative to project root or an absolute path.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory relative to project root or an absolute path.")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    selection = Path(args.selection_file) if args.selection_file else ROOT / "Data/Manifests/sentinel2_selection.csv"
    if not selection.is_absolute():
        selection = ROOT / selection
    outdir = Path(args.output_dir) if args.output_dir else ROOT / "Results/Bathymetry"
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    atl_path = ROOT / "Data/Processed/ATL24/atl24_bathy_points.csv.gz"
    if not selection.exists() or not atl_path.exists():
        raise SystemExit("Run Scripts/04, 05, 06 and 07 before fitting SDB models.")
    if not TIDE_MODEL_DIR.is_dir():
        raise SystemExit("GOT5.6 model files are missing. Run Scripts/07_download_got56.py --download first.")
    try:
        import pyTMD  # noqa: F401
    except ImportError as exc:
        raise SystemExit("pyTMD is missing. Install the project requirements before tide normalization.") from exc
    with selection.open(newline="", encoding="utf-8-sig") as stream:
        scenes = [r for r in csv.DictReader(stream) if r.get("selected", "").upper() == "YES"]
    atl = pd.read_csv(atl_path)
    atl["utc_time_source_epoch"] = pd.to_datetime(atl["utc_time_source_epoch"], utc=True, errors="coerce")
    rows = []
    matchup_audit = []
    max_gap = pd.Timedelta(hours=float(cfg["s2_atl24_max_time_gap_hours"]))
    for si, scene in enumerate(scenes, 1):
        scene_time = pd.to_datetime(scene["acquisition_utc"], utc=True)
        pts = atl.loc[(atl["utc_time_source_epoch"] - scene_time).abs() <= max_gap].copy()
        n_time_candidates = int(len(pts))
        if pts.empty:
            print(f"{si}/{len(scenes)} {scene['acquisition_date']}: no ATL24 photons within {max_gap}.")
            matchup_audit.append({"scene_id": scene["item_id"], "date": scene["acquisition_date"],
                                  "time_candidates": 0, "clear_water_candidates": 0,
                                  "GOT5.6_supported_candidates": 0, "retained_training_pairs": 0})
            continue
        date = scene["acquisition_date"].replace("-", "")
        prefix = ROOT / "Data/Processed/Sentinel2" / f"{date}_{safe_name(scene['item_id'])}"
        raster_path = Path(str(prefix) + "_watermask.tif")
        ndwi_path = Path(str(prefix) + "_NDWI.tif")
        ndti_path = Path(str(prefix) + "_NDTI.tif")
        if not raster_path.is_file() or not ndwi_path.is_file() or not ndti_path.is_file():
            raise SystemExit(f"Processed water mask/indices missing for {scene['item_id']}. Run Scripts/07_process_sentinel2.py first.")
        with rasterio.open(raster_path) as mask_src, rasterio.open(ndwi_path) as ndwi_src, rasterio.open(ndti_path) as red_src:
            # Read the original band clips to keep model predictors in physically scaled reflectance.
            raw_prefix = ROOT / "Data/Raw/Sentinel2" / f"{date}_{safe_name(scene['item_id'])}"
            band = {}
            for name in ("blue", "green", "red", "nir"):
                with rasterio.open(Path(str(raw_prefix) + f"_{name}.tif")) as src:
                    raw_values = src.read(1)
                    values = raw_values.astype("float32") * reflectance_scale(src, raw_values)
                    if src.transform != mask_src.transform or src.crs != mask_src.crs or values.shape != (mask_src.height, mask_src.width):
                        aligned = np.full((mask_src.height, mask_src.width), np.nan, dtype="float32")
                        reproject(values, aligned, src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=mask_src.transform, dst_crs=mask_src.crs,
                                  resampling=Resampling.bilinear)
                        values = aligned
                    band[name] = values
            xy = list(zip(pts["easting_m"].to_numpy(), pts["northing_m"].to_numpy()))
            water = np.array([v[0] for v in mask_src.sample(xy)], dtype="uint8") == 1
            indices = np.array([v[0] for v in ndwi_src.sample(xy)], dtype="float32")
            ratio_index = np.array([v[0] for v in red_src.sample(xy)], dtype="float32")
            sampled = {}
            for name, values in band.items():
                # Sample aligned arrays using the raster's row/column transform.
                rc = [rasterio.transform.rowcol(mask_src.transform, x, y) for x, y in xy]
                sampled[name] = np.array([values[r, c] if 0 <= r < values.shape[0] and 0 <= c < values.shape[1] else np.nan for r, c in rc], dtype="float32")
        keep = water & np.isfinite(indices) & np.isfinite(ratio_index)
        for name in sampled:
            keep &= np.isfinite(sampled[name])
        pts = pts.loc[keep].copy()
        if pts.empty:
            print(f"{si}/{len(scenes)} {scene['acquisition_date']}: zero clear-water ATL24 matches.")
            matchup_audit.append({"scene_id": scene["item_id"], "date": scene["acquisition_date"],
                                  "time_candidates": n_time_candidates, "clear_water_candidates": 0,
                                  "GOT5.6_supported_candidates": 0, "retained_training_pairs": 0})
            continue
        pts["scene_id"] = scene["item_id"]
        pts["scene_utc"] = scene["acquisition_utc"]
        pts["time_gap_minutes"] = (pts["utc_time_source_epoch"] - scene_time).abs().dt.total_seconds() / 60
        pts["NDWI"] = indices[keep]
        pts["NDTI_relative"] = ratio_index[keep]
        for name, values in sampled.items():
            pts[f"{name}_refl"] = values[keep]
        # ATL24 depth is referenced to the sea surface at the photon time.
        # Normalize it to the Sentinel-2 overpass using only the modeled
        # relative astronomical-tide change; no spatial extrapolation is used.
        eta_atl24 = predict_got56_no_extrapolation(
            pts["longitude_deg"].to_numpy(), pts["latitude_deg"].to_numpy(), pts["utc_time_source_epoch"]
        )
        eta_s2 = predict_got56_no_extrapolation(
            pts["longitude_deg"].to_numpy(), pts["latitude_deg"].to_numpy(), [scene_time] * len(pts)
        )
        if len(eta_atl24) != len(pts) or len(eta_s2) != len(pts):
            raise RuntimeError(f"Unexpected GOT5.6 prediction count for {scene['item_id']}.")
        tide_delta = eta_s2 - eta_atl24
        got_supported = np.isfinite(eta_atl24) & np.isfinite(eta_s2) & np.isfinite(tide_delta)
        pts["got56_tide_at_atl24_m"] = eta_atl24
        pts["got56_tide_at_sentinel2_m"] = eta_s2
        pts["got56_relative_tide_change_s2_minus_atl24_m"] = tide_delta
        pts["depth_at_sentinel2_time_m"] = pts["instantaneous_depth_m"].to_numpy(dtype="float64") + tide_delta
        n_clear = int(len(pts))
        pts = pts.loc[got_supported & np.isfinite(pts["depth_at_sentinel2_time_m"]) &
                      (pts["depth_at_sentinel2_time_m"] > 0)].copy()
        matchup_audit.append({
            "scene_id": scene["item_id"], "date": scene["acquisition_date"],
            "time_candidates": n_time_candidates,
            "clear_water_candidates": n_clear,
            "GOT5.6_supported_candidates": int(got_supported.sum()),
            "retained_training_pairs": int(len(pts)),
            "median_absolute_time_gap_min": float(pts["time_gap_minutes"].median()) if len(pts) else float("nan"),
            "median_relative_tide_change_m": float(pts["got56_relative_tide_change_s2_minus_atl24_m"].median()) if len(pts) else float("nan"),
        })
        if pts.empty:
            print(f"{si}/{len(scenes)} {scene['acquisition_date']}: no clear-water points with GOT5.6 support at both times.")
            continue
        eps = 1e-6
        pts["log_blue_green"] = np.log(np.maximum(pts["blue_refl"], eps) / np.maximum(pts["green_refl"], eps))
        primary_gap_min = 60.0 * float(cfg.get("s2_atl24_primary_time_gap_hours", 6.0))
        pts["matchup_window_class"] = np.where(
            pts["time_gap_minutes"] <= primary_gap_min, "primary_0_6h", "extended_6_24h"
        )
        rows.append(pts)
        print(f"{si}/{len(scenes)} {scene['acquisition_date']}: tide-supported training pairs={len(pts):,}; median time gap={pts['time_gap_minutes'].median():.1f} min; median GOT5.6 correction={pts['got56_relative_tide_change_s2_minus_atl24_m'].median():+.3f} m")
    if not rows:
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(matchup_audit).to_csv(outdir / "atl24_sentinel2_tide_matchup_support.csv", index=False)
        raise SystemExit("No clear-water near-time ATL24/Sentinel-2 sample pairs. Expand the analysis period/AOI or use the workflow diagnostics; do not relax the time gap silently.")
    data = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    outdir.mkdir(parents=True, exist_ok=True)
    audit_path = outdir / "atl24_sentinel2_tide_matchup_support.csv"
    pd.DataFrame(matchup_audit).to_csv(audit_path, index=False)
    pairs_path = outdir / "atl24_sentinel2_training_pairs.csv.gz"
    data.to_csv(pairs_path, index=False, compression="gzip")
    feature_sets = {
        "Log_blue_green_ratio_OLS": ["log_blue_green"],
        "Multiband_RandomForest": ["blue_refl", "green_refl", "red_refl", "nir_refl", "NDWI"],
        "Multiband_HistGradientBoosting": ["blue_refl", "green_refl", "red_refl", "nir_refl", "NDWI"],
    }
    models = {
        "Log_blue_green_ratio_OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "Multiband_RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.8, n_jobs=-1, random_state=42),
        "Multiband_HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=42),
    }
    summaries, predictions = [], []
    target_scenarios = {
        "uncorrected_instantaneous_depth": "instantaneous_depth_m",
        "GOT56_relative_tide_adjusted_depth": "depth_at_sentinel2_time_m",
    }
    window_sets = {
        "primary_0_6h": data.loc[data["matchup_window_class"] == "primary_0_6h"].copy(),
        "extended_6_24h": data.loc[data["matchup_window_class"] == "extended_6_24h"].copy(),
        "combined_0_24h": data.copy(),
    }
    for window_name, subset in window_sets.items():
        groups = subset["granule_id"].astype(str)
        n_groups = int(groups.nunique())
        n_splits = min(5, n_groups) if n_groups >= 3 else 0
        for scenario, target_col in target_scenarios.items():
            y = subset[target_col].to_numpy(dtype="float64")
            for name, features in feature_sets.items():
                if n_groups < 3 or len(subset) < 50:
                    summaries.append({"window_analysis": window_name, "target_scenario": scenario, "model": name,
                                      "n": int(len(subset)), "independent_atl24_granules": n_groups,
                                      "group_folds": n_splits, "status": "insufficient_independent_granules_or_pairs",
                                      "RMSE_m": np.nan, "MAE_m": np.nan, "bias_m": np.nan, "R2": np.nan})
                    continue
                cv = GroupKFold(n_splits=n_splits)
                pred = np.full(len(subset), np.nan)
                for tr, te in cv.split(subset[features], y, groups=groups):
                    model = clone(models[name])
                    model.fit(subset.iloc[tr][features], y[tr])
                    pred[te] = model.predict(subset.iloc[te][features])
                ok = np.isfinite(pred) & np.isfinite(y)
                summaries.append({"window_analysis": window_name, "target_scenario": scenario, "model": name,
                                  "n": int(ok.sum()), "independent_atl24_granules": n_groups,
                                  "group_folds": n_splits, "status": "grouped_cross_validation",
                                  "RMSE_m": float(mean_squared_error(y[ok], pred[ok]) ** 0.5),
                                  "MAE_m": float(mean_absolute_error(y[ok], pred[ok])),
                                  "bias_m": float(np.mean(pred[ok] - y[ok])), "R2": float(r2_score(y[ok], pred[ok]))})
                block = subset[["scene_id", "granule_id", "beam", "utc_time_source_epoch", "time_gap_minutes",
                                "matchup_window_class", "easting_m", "northing_m", "instantaneous_depth_m",
                                "got56_relative_tide_change_s2_minus_atl24_m", "depth_at_sentinel2_time_m"]].copy()
                block["window_analysis"] = window_name
                block["target_scenario"] = scenario
                block["observed_depth_m"] = y
                block["model"] = name
                block["prediction_depth_m"] = pred
                predictions.append(block)
    metrics_path = outdir / "grouped_sdb_model_metrics.csv"
    prediction_path = outdir / "grouped_sdb_oof_predictions.csv.gz"
    pd.DataFrame(summaries).to_csv(metrics_path, index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False, compression="gzip")
    else:
        pd.DataFrame(columns=["window_analysis", "target_scenario", "model", "observed_depth_m", "prediction_depth_m"]).to_csv(prediction_path, index=False, compression="gzip")
    metadata = {
        "tide_model": "GOT5.6 standard ocean-tide constituents",
        "spatial_extrapolation": False,
        "primary_time_gap_hours": float(cfg.get("s2_atl24_primary_time_gap_hours", 6.0)),
        "extended_time_gap_hours": float(cfg["s2_atl24_max_time_gap_hours"]),
        "depth_normalization": "depth_S2 = depth_ATL24 + eta_GOT56(t_S2) - eta_GOT56(t_ATL24)",
        "model_validation": "GroupKFold grouped by full ATL24 granule; only model-supported clear-water pairs retained",
        "n_training_pairs": int(len(data)),
        "n_independent_atl24_granules": int(data["granule_id"].astype(str).nunique()),
        "caveat": "Relative astronomical-tide normalization only; no surge/runup correction or absolute vertical-datum tie. ATL24 tracks do not validate unsampled pixels.",
    }
    (outdir / "sdb_tide_normalization_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Pair table: {pairs_path}")
    print(f"Tide-support audit: {audit_path}")
    print(f"Grouped validation metrics (raw and tide-adjusted): {metrics_path}")
    print(f"Out-of-fold predictions: {prediction_path}")
    print(f"Run metadata: {outdir / 'sdb_tide_normalization_run_metadata.json'}")
    print("Depth target corrected with the relative GOT5.6 tide change at each ATL24 point; unsupported predictions were excluded (no extrapolation).")
    print("Groups are full ATL24 granules; intervals do not include GOT5.6, surge/runup, optical-classification or vertical-datum uncertainty. Metrics apply to sampled tracks only, not unsampled pixels or a datum-tied bathymetric surface.")


if __name__ == "__main__":
    main()
