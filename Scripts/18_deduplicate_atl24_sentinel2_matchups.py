from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "Results/Bathymetry/ATL24_Sentinel2_IBI_SSH_24h"
KEYS = ["granule_id", "beam", "source_photon_index"]
FEATURES = {
    "Log_blue_green_ratio_OLS": ["log_blue_green"],
    "Multiband_RandomForest": ["blue_refl", "green_refl", "red_refl", "nir_refl", "NDWI"],
    "Multiband_HistGradientBoosting": ["blue_refl", "green_refl", "red_refl", "nir_refl", "NDWI"],
}
MODELS = {
    "Log_blue_green_ratio_OLS": make_pipeline(StandardScaler(), LinearRegression()),
    "Multiband_RandomForest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.8, n_jobs=-1, random_state=42
    ),
    "Multiband_HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
    ),
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def effective_group_count(groups: pd.Series) -> float:
    shares = groups.value_counts(normalize=True).to_numpy(dtype=float)
    return float(1.0 / np.sum(shares ** 2)) if len(shares) else float("nan")


def metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return {"RMSE_m": np.nan, "MAE_m": np.nan, "bias_m": np.nan, "R2": np.nan}
    return {
        "RMSE_m": float(mean_squared_error(y[valid], pred[valid]) ** 0.5),
        "MAE_m": float(mean_absolute_error(y[valid], pred[valid])),
        "bias_m": float(np.mean(pred[valid] - y[valid])),
        "R2": float(r2_score(y[valid], pred[valid])) if valid.sum() > 1 else np.nan,
    }


def unique_nearest_scene(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in KEYS + ["scene_id", "scene_utc", "time_gap_minutes"] if c not in frame.columns]
    if missing:
        raise SystemExit(f"Required matchup columns are missing: {missing}")
    ordered = frame.sort_values(
        ["time_gap_minutes", "scene_utc", "scene_id"], kind="mergesort"
    )
    return ordered.drop_duplicates(subset=KEYS, keep="first").copy()


def evaluate(frame: pd.DataFrame, radius: float, window: str, target_name: str,
             target_column: str) -> tuple[list[dict], list[dict], list[dict]]:
    subset = frame.loc[frame["maximum_grid_distance_km"].eq(radius)].copy()
    if window != "combined_0_24h":
        subset = subset.loc[subset["matchup_window_class"].eq(window)].copy()
    subset[target_column] = pd.to_numeric(subset[target_column], errors="coerce")
    subset = subset.loc[np.isfinite(subset[target_column]) & (subset[target_column] > 0)].copy()
    if subset.empty:
        return [], [], []

    y = subset[target_column].to_numpy(dtype=float)
    groups = subset["granule_id"].astype(str)
    n_groups = int(groups.nunique())
    folds = min(5, n_groups)
    status = (
        "grouped_cross_validation_primary_support"
        if n_groups >= 5 and len(subset) >= 50
        else "grouped_cross_validation_exploratory"
        if n_groups >= 3 and len(subset) >= 30
        else "insufficient_independent_granules"
    )
    common = {
        "maximum_grid_distance_km": radius,
        "window_analysis": window,
        "target_scenario": target_name,
        "n_unique_photons": int(len(subset)),
        "n_independent_granules": n_groups,
        "effective_granule_count_by_photon_weight": effective_group_count(groups),
        "smallest_granule_n": int(groups.value_counts().min()),
        "largest_granule_n": int(groups.value_counts().max()),
        "group_folds": folds if n_groups >= 3 else 0,
        "status": status,
    }
    if status == "insufficient_independent_granules":
        return [{**common, "model": name, "RMSE_m": np.nan, "MAE_m": np.nan, "bias_m": np.nan,
                 "R2": np.nan, "median_baseline_RMSE_m": np.nan, "median_baseline_MAE_m": np.nan,
                 "RMSE_skill_vs_grouped_median_baseline": np.nan} for name in FEATURES], [], []

    oof_rows = []
    group_rows = []
    metric_rows = []

    for model_name, columns in FEATURES.items():
        valid_features = np.isfinite(subset[columns].to_numpy(dtype=float)).all(axis=1)
        work = subset.loc[valid_features].copy()
        n_work_groups = int(work["granule_id"].nunique())
        if len(work) < 30 or n_work_groups < 3:
            metric_rows.append({**common, "model": model_name, "RMSE_m": np.nan, "MAE_m": np.nan,
                                "bias_m": np.nan, "R2": np.nan, "median_baseline_RMSE_m": np.nan,
                                "median_baseline_MAE_m": np.nan,
                                "RMSE_skill_vs_grouped_median_baseline": np.nan,
                                "status": "insufficient_complete_feature_support"})
            continue

        yy = work[target_column].to_numpy(dtype=float)
        gg = work["granule_id"].astype(str)
        cv = GroupKFold(n_splits=min(5, int(gg.nunique())))
        predictions = np.full(len(work), np.nan)
        baseline = np.full(len(work), np.nan)
        fold_id = np.full(len(work), -1, dtype=int)
        for fold, (train_index, test_index) in enumerate(cv.split(work, yy, groups=gg)):
            estimator = clone(MODELS[model_name])
            estimator.fit(work.iloc[train_index][columns], yy[train_index])
            predictions[test_index] = estimator.predict(work.iloc[test_index][columns])
            baseline[test_index] = float(np.median(yy[train_index]))
            fold_id[test_index] = fold

        model_metrics = metric_values(yy, predictions)
        baseline_metrics = metric_values(yy, baseline)
        skill = (
            1.0 - model_metrics["RMSE_m"] / baseline_metrics["RMSE_m"]
            if np.isfinite(model_metrics["RMSE_m"]) and baseline_metrics["RMSE_m"] > 0
            else np.nan
        )
        metric_rows.append({
            **common, "model": model_name, **model_metrics,
            "median_baseline_RMSE_m": baseline_metrics["RMSE_m"],
            "median_baseline_MAE_m": baseline_metrics["MAE_m"],
            "RMSE_skill_vs_grouped_median_baseline": float(skill),
            "complete_feature_photons": int(len(work)),
            "complete_feature_granules": int(gg.nunique()),
        })
        pred = work[[
            "scene_id", "acquisition_date", "granule_id", "beam", "source_photon_index",
            "utc_time_source_epoch",
            "time_gap_minutes", "ibi_grid_distance_km",
            "ibi_relative_ssh_change_s2_minus_atl24_m", "instantaneous_depth_m",
            "depth_ibi_corrected_at_s2_m", "maximum_grid_distance_km",
            "matchup_window_class"
        ]].copy()
        pred["window_analysis"] = window
        pred["target_scenario"] = target_name
        pred["model"] = model_name
        pred["observed_depth_m"] = yy
        pred["prediction_depth_m"] = predictions
        pred["fold_median_baseline_depth_m"] = baseline
        pred["cv_fold"] = fold_id
        oof_rows.extend(pred.to_dict("records"))

        residual = predictions - yy
        per_granule = pd.DataFrame({
            "granule_id": gg.to_numpy(),
            "observed": yy,
            "prediction": predictions,
            "baseline": baseline,
            "residual": residual,
        }).groupby("granule_id", sort=True)
        for granule, part in per_granule:
            gm = metric_values(part["observed"].to_numpy(), part["prediction"].to_numpy())
            bm = metric_values(part["observed"].to_numpy(), part["baseline"].to_numpy())
            group_rows.append({
                "maximum_grid_distance_km": radius,
                "window_analysis": window,
                "target_scenario": target_name,
                "model": model_name,
                "granule_id": granule,
                "n_unique_photons": int(len(part)),
                "model_RMSE_m": gm["RMSE_m"],
                "model_MAE_m": gm["MAE_m"],
                "model_bias_m": gm["bias_m"],
                "fold_median_baseline_RMSE_m": bm["RMSE_m"],
            })
    return metric_rows, oof_rows, group_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate ATL24 photons to their nearest Sentinel-2 scene and rerun grouped SDB validation."
    )
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.is_absolute():
        analysis_dir = ROOT / analysis_dir
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "Unique_Photon_Audit"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    pair_path = analysis_dir / "s2_atl24_ibi_ssh_training_pairs_by_distance.csv.gz"
    if not pair_path.is_file():
        raise SystemExit(f"Missing Script 16 matchup file: {pair_path}")

    pairs = pd.read_csv(pair_path, low_memory=False)
    if "source_photon_index" not in pairs.columns:
        raise SystemExit("Stable source_photon_index is absent from matchup pairs. Rerun Scripts/06_extract_atl24_points.py and Scripts/16_fit_sdb_ibi_ssh_sensitivity.py first.")
    for col in ("time_gap_minutes", "maximum_grid_distance_km", "ibi_grid_distance_km",
                "instantaneous_depth_m", "depth_ibi_corrected_at_s2_m"):
        pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
    original_counts = pairs.groupby("maximum_grid_distance_km").size().to_dict()
    deduped = pd.concat(
        [unique_nearest_scene(group) for _, group in pairs.groupby("maximum_grid_distance_km", sort=True)],
        ignore_index=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(output_dir / "nearest_scene_unique_photon_pairs.csv.gz",
                   index=False, compression="gzip")

    support_rows = []
    for radius, group in deduped.groupby("maximum_grid_distance_km", sort=True):
        original_n = int(original_counts.get(radius, 0))
        for window in ("primary_0_6h", "extended_6_24h", "combined_0_24h"):
            sub = group if window == "combined_0_24h" else group.loc[group["matchup_window_class"].eq(window)]
            support_rows.append({
                "maximum_grid_distance_km": float(radius),
                "window_analysis": window,
                "unique_photons": int(len(sub)),
                "granules": int(sub["granule_id"].nunique()),
                "scenes": int(sub["scene_id"].nunique()),
                "median_time_gap_min": float(sub["time_gap_minutes"].median()) if len(sub) else np.nan,
                "p90_time_gap_min": float(sub["time_gap_minutes"].quantile(0.90)) if len(sub) else np.nan,
                "median_grid_distance_km": float(sub["ibi_grid_distance_km"].median()) if len(sub) else np.nan,
                "median_abs_ibi_adjustment_m": float(
                    sub["ibi_relative_ssh_change_s2_minus_atl24_m"].abs().median()
                ) if len(sub) else np.nan,
                "original_pair_rows_at_radius": original_n if window == "combined_0_24h" else np.nan,
                "duplicate_rows_removed_at_radius": original_n - int(len(group)) if window == "combined_0_24h" else np.nan,
            })
    pd.DataFrame(support_rows).to_csv(output_dir / "nearest_scene_unique_photon_support.csv", index=False)

    metric_rows, oof_rows, granule_rows = [], [], []
    targets = {
        "raw_instantaneous_depth": "instantaneous_depth_m",
        "IBI_zos_relative_adjustment": "depth_ibi_corrected_at_s2_m",
    }
    for radius in sorted(deduped["maximum_grid_distance_km"].dropna().unique()):
        for window in ("primary_0_6h", "extended_6_24h", "combined_0_24h"):
            for target_name, target_column in targets.items():
                m, p, g = evaluate(deduped, float(radius), window, target_name, target_column)
                metric_rows.extend(m)
                oof_rows.extend(p)
                granule_rows.extend(g)
    pd.DataFrame(metric_rows).to_csv(output_dir / "nearest_scene_unique_photon_grouped_metrics.csv", index=False)
    if oof_rows:
        pd.DataFrame(oof_rows).to_csv(output_dir / "nearest_scene_unique_photon_oof_predictions.csv.gz",
                                      index=False, compression="gzip")
    if granule_rows:
        pd.DataFrame(granule_rows).to_csv(output_dir / "nearest_scene_unique_photon_by_granule.csv", index=False)

    metadata = {
        "source_pairs": str(pair_path),
        "deduplication_key": KEYS,
        "source_photon_identifier": "ATL24 source_photon_index = ATL24 index_ph, the photon index inherited from ATL03; combined with granule_id and beam.",
        "selection_rule": "For every ATL24 photon identity and distance threshold, retain the Sentinel-2 matchup with the smallest absolute time gap; ties are resolved by scene UTC and scene ID.",
        "validation": "GroupKFold by full ATL24 granule, with fold-specific training median depth as a no-optical-predictor baseline.",
        "models": list(FEATURES),
        "target_scenarios": list(targets),
        "interpretation": "This audit tests the sensitivity of reported skill to repeated use of the same ATL24 photon across multiple Sentinel-2 scenes. It does not validate a continuous bathymetry map or remove datum, IBI, runup, photon, or spatial sampling uncertainty.",
        "original_rows_by_radius": {str(k): int(v) for k, v in original_counts.items()},
        "unique_nearest_rows_by_radius": {
            str(k): int(v) for k, v in deduped.groupby("maximum_grid_distance_km").size().to_dict().items()
        },
    }
    (output_dir / "nearest_scene_unique_photon_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Nearest-scene, unique-photon grouped validation complete; source imagery, ATL24 and IBI inputs were read only.")
    print(pd.DataFrame(support_rows).query("window_analysis == 'combined_0_24h'").to_string(index=False))
    print(f"Outputs: {output_dir}")
    print("Compare these scores with Script 17. Grouped predictions are track-level diagnostics, not continuous-map validation.")


if __name__ == "__main__":
    main()




