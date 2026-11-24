from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "Results/Bathymetry/ATL24_Sentinel2_IBI_SSH_24h/Unique_Photon_Audit"
    / "nearest_scene_unique_photon_pairs.csv.gz"
)
FEATURES = {
    "Log_blue_green_ratio_OLS": ["log_blue_green"],
    "Multiband_RandomForest": ["blue_refl", "green_refl", "red_refl", "nir_refl", "NDWI"],
}
MODELS = {
    "Log_blue_green_ratio_OLS": make_pipeline(StandardScaler(), LinearRegression()),
    "Multiband_RandomForest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.8,
        n_jobs=1, random_state=42,
    ),
}
TARGETS = {
    "raw_instantaneous_depth": "instantaneous_depth_m",
    "IBI_zos_relative_adjustment": "depth_ibi_corrected_at_s2_m",
}
WINDOWS = ["primary_0_6h", "combined_0_24h"]


def metric(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    residual = pred - y
    return (
        float(np.sqrt(np.mean(residual**2))),
        float(np.mean(np.abs(residual))),
        float(np.mean(residual)),
    )


def rectangle_distance(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float,
                       ymin: float, ymax: float) -> np.ndarray:
    dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
    dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
    return np.hypot(dx, dy)


def block_group_metrics(predictions: pd.DataFrame, replicates: int,
                        rng: np.random.Generator,
                        stratify_by_depth: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_cols = [
        "maximum_grid_distance_km", "window_analysis", "target_scenario", "model",
        "block_size_m", "buffer_m",
    ]
    work_data = predictions.copy()
    if stratify_by_depth:
        work_data["depth_stratum_raw_ATL24"] = pd.cut(
            work_data["raw_depth_m"], bins=[0, 2, 5, 10, 15, 20, 40],
            labels=["0-2 m", "2-5 m", "5-10 m", "10-15 m", "15-20 m", "20-40 m"],
            right=False, include_lowest=True,
        )
        work_data = work_data.loc[work_data["depth_stratum_raw_ATL24"].notna()].copy()
        group_cols.append("depth_stratum_raw_ATL24")
    for keys, frame in work_data.groupby(group_cols, sort=True, observed=True):
        if stratify_by_depth:
            radius, window, target, model_name, block_size, buffer_m, depth_stratum = keys
        else:
            radius, window, target, model_name, block_size, buffer_m = keys
            depth_stratum = "all_depths"
        per_block = frame.groupby("holdout_block", sort=True).agg(
            model_mse=("error_sq", "mean"),
            baseline_mse=("baseline_error_sq", "mean"),
            n_photons=("observed_depth_m", "size"),
            n_granules=("granule_id", "nunique"),
        )
        if per_block.empty:
            continue
        model_mse = per_block["model_mse"].to_numpy(dtype=float)
        baseline_mse = per_block["baseline_mse"].to_numpy(dtype=float)
        block_count = len(per_block)
        equal_model_rmse = float(np.sqrt(np.mean(model_mse)))
        equal_baseline_rmse = float(np.sqrt(np.mean(baseline_mse)))
        equal_skill = (
            1.0 - equal_model_rmse / equal_baseline_rmse
            if equal_baseline_rmse > 0 else np.nan
        )
        pooled_y = frame["observed_depth_m"].to_numpy(dtype=float)
        pooled_pred = frame["prediction_depth_m"].to_numpy(dtype=float)
        pooled_base = frame["fold_median_baseline_depth_m"].to_numpy(dtype=float)
        pooled_rmse, pooled_mae, pooled_bias = metric(pooled_y, pooled_pred)
        pooled_baseline_rmse = metric(pooled_y, pooled_base)[0]
        pooled_skill = (
            1.0 - pooled_rmse / pooled_baseline_rmse
            if pooled_baseline_rmse > 0 else np.nan
        )

        draw = rng.integers(0, block_count, size=(replicates, block_count))
        boot_model_rmse = np.sqrt(np.mean(model_mse[draw], axis=1))
        boot_base_rmse = np.sqrt(np.mean(baseline_mse[draw], axis=1))
        boot_skill = np.divide(
            boot_base_rmse - boot_model_rmse, boot_base_rmse,
            out=np.full(replicates, np.nan), where=boot_base_rmse > 0,
        )
        rmse_lo, rmse_hi = np.nanquantile(boot_model_rmse, [0.025, 0.975])
        skill_lo, skill_hi = np.nanquantile(boot_skill, [0.025, 0.975])
        n_photons = int(len(frame))
        n_granules = int(frame["granule_id"].nunique())
        status = (
            "replicated_spatial_support"
            if block_count >= 10 and n_photons >= 100 and n_granules >= 5
            else "limited_spatial_support"
        )
        rows.append({
            "maximum_grid_distance_km": float(radius),
            "window_analysis": str(window),
            "target_scenario": str(target),
            "model": str(model_name),
            "block_size_m": int(block_size),
            "buffer_m": int(buffer_m),
            "depth_stratum_raw_ATL24": str(depth_stratum),
            "held_out_spatial_blocks": block_count,
            "held_out_photons": n_photons,
            "held_out_granules": n_granules,
            "equal_block_RMSE_m": equal_model_rmse,
            "equal_block_RMSE_CI95_low_m": float(rmse_lo),
            "equal_block_RMSE_CI95_high_m": float(rmse_hi),
            "equal_block_baseline_RMSE_m": equal_baseline_rmse,
            "equal_block_skill_vs_training_median": float(equal_skill),
            "equal_block_skill_CI95_low": float(skill_lo),
            "equal_block_skill_CI95_high": float(skill_hi),
            "pooled_RMSE_m": pooled_rmse,
            "pooled_MAE_m": pooled_mae,
            "pooled_bias_m": pooled_bias,
            "pooled_baseline_RMSE_m": pooled_baseline_rmse,
            "pooled_skill_vs_training_median": float(pooled_skill),
            "bootstrap_replicates": replicates,
            "bootstrap_unit": "held-out spatial blocks; conditional on fitted block splits",
            "support_status": status,
            "scope": (
                "spatial transfer within this sampled coastal footprint; not external-site validation "
                "or continuous-map accuracy"
            ),
        })
    return rows


def make_figure(summary: pd.DataFrame, out: Path) -> None:
    frame = summary.loc[summary["model"].eq("Multiband_RandomForest")].copy()
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, constrained_layout=True)
    colors = {"raw_instantaneous_depth": "#19647E", "IBI_zos_relative_adjustment": "#E07A32"}
    for ax, window in zip(axes, WINDOWS):
        part = frame.loc[
            frame["window_analysis"].eq(window)
            & frame["maximum_grid_distance_km"].eq(5.0)
        ]
        for target, target_frame in part.groupby("target_scenario", sort=True):
            target_frame = target_frame.sort_values("block_size_m")
            x = target_frame["block_size_m"].to_numpy(dtype=float)
            y = target_frame["equal_block_skill_vs_training_median"].to_numpy(dtype=float)
            lo = target_frame["equal_block_skill_CI95_low"].to_numpy(dtype=float)
            hi = target_frame["equal_block_skill_CI95_high"].to_numpy(dtype=float)
            valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            ax.errorbar(
                x[valid], y[valid],
                yerr=np.vstack([
                    np.maximum(0.0, y[valid] - lo[valid]),
                    np.maximum(0.0, hi[valid] - y[valid]),
                ]),
                marker="o", linewidth=1.4, capsize=3, color=colors[target],
                label={"raw_instantaneous_depth": "Raw ATL24 depth", "IBI_zos_relative_adjustment": "Relative IBI-adjusted target"}[target],
            )
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9)
        ax.set_title("Near-synchronous (0–6 h)" if window == "primary_0_6h" else "Combined (0–24 h)")
        ax.set_xlabel("Spatial block width (m); buffer = 50% of block width")
        ax.set_xticks(sorted(frame["block_size_m"].unique()))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Equal-block RMSE skill vs training median")
    fig.suptitle("Spatially blocked SDB transfer within the Anza footprint (≤5 km)", fontsize=12)
    fig.savefig(out / "Figure_spatial_blocked_SDB_transfer.png", dpi=350)
    fig.savefig(out / "Figure_spatial_blocked_SDB_transfer.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buffered spatial-block validation for unique ATL24–Sentinel-2 photon matchups."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--maximum-distances-km", nargs="+", type=float, default=[5.0])
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=WINDOWS)
    parser.add_argument("--block-sizes-m", nargs="+", type=int, default=[1000, 2000])
    parser.add_argument("--minimum-buffer-m", type=int, default=0,
                        help="If 0, use half of each block width.")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    if args.replicates < 200:
        raise SystemExit("Use at least 200 spatial-block bootstrap replicates.")
    if any(size <= 0 for size in args.block_sizes_m):
        raise SystemExit("Block sizes must be positive.")
    if args.minimum_buffer_m < 0:
        raise SystemExit("Buffer must be zero or positive.")
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    out = Path(args.output_dir) if args.output_dir else input_path.parent / "Spatial_Block_Validation"
    if not out.is_absolute():
        out = ROOT / out
    if not input_path.is_file():
        raise SystemExit(f"Missing unique-photon matchup table: {input_path}")

    data = pd.read_csv(input_path, low_memory=False)
    required = [
        "granule_id", "beam", "source_photon_index", "easting_m", "northing_m",
        "maximum_grid_distance_km", "matchup_window_class", "instantaneous_depth_m",
        "depth_ibi_corrected_at_s2_m", *{c for cols in FEATURES.values() for c in cols},
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit(f"Unique-photon input is missing required columns: {missing}")
    for column in [
        "maximum_grid_distance_km", "easting_m", "northing_m",
        "instantaneous_depth_m", "depth_ibi_corrected_at_s2_m",
        *{c for cols in FEATURES.values() for c in cols},
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.duplicated(["granule_id", "beam", "source_photon_index", "maximum_grid_distance_km"]).any():
        raise SystemExit("Input repeats source photons within a radius; run Script 18 first.")
    data = data.loc[
        np.isfinite(data["easting_m"])
        & np.isfinite(data["northing_m"])
    ].copy()
    data["source_photon_index"] = data["source_photon_index"].astype(str)
    pred_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    x_all = data["easting_m"].to_numpy(dtype=float)
    y_all = data["northing_m"].to_numpy(dtype=float)

    for radius in args.maximum_distances_km:
        radius_data = data.loc[np.isclose(data["maximum_grid_distance_km"], radius)].copy()
        for window in args.windows:
            if window == "combined_0_24h":
                time_data = radius_data
            else:
                time_data = radius_data.loc[radius_data["matchup_window_class"].eq(window)].copy()
            for block_size in args.block_sizes_m:
                buffer_m = args.minimum_buffer_m or int(round(block_size * 0.5))
                for target_name, target_column in TARGETS.items():
                    for model_name, feature_cols in FEATURES.items():
                        cols = [target_column, *feature_cols, "easting_m", "northing_m"]
                        work = time_data.dropna(subset=cols).copy()
                        work = work.loc[
                            work[target_column].gt(0)
                            & work[target_column].lt(80)
                            & np.isfinite(work[feature_cols].to_numpy(dtype=float)).all(axis=1)
                        ].copy()
                        if len(work) < 100 or work["granule_id"].nunique() < 3:
                            skipped_rows.append({
                                "maximum_grid_distance_km": radius,
                                "window_analysis": window,
                                "target_scenario": target_name,
                                "model": model_name,
                                "block_size_m": block_size,
                                "buffer_m": buffer_m,
                                "reason": "fewer than 100 photons or fewer than 3 ATL24 granules",
                                "photons_available": int(len(work)),
                                "granules_available": int(work["granule_id"].nunique()),
                            })
                            continue
                        x = work["easting_m"].to_numpy(dtype=float)
                        y = work["northing_m"].to_numpy(dtype=float)
                        block_x = np.floor(x / block_size).astype(np.int64)
                        block_y = np.floor(y / block_size).astype(np.int64)
                        block_ids = pd.Series(
                            [f"{bx}_{by}" for bx, by in zip(block_x, block_y)], index=work.index
                        )
                        unique_x = np.array(sorted(np.unique(block_x)), dtype=np.int64)
                        x_folds = [part for part in np.array_split(unique_x, min(5, len(unique_x))) if len(part)]
                        valid_folds = 0
                        for fold_id, fold_x in enumerate(x_folds):
                            mask_test = np.isin(block_x, fold_x)
                            test_ids = sorted(block_ids.iloc[np.flatnonzero(mask_test)].unique())
                            distance_to_test = np.full(len(work), np.inf, dtype=float)
                            for holdout in test_ids:
                                bx, by = (int(value) for value in holdout.split("_"))
                                xmin, xmax = bx * block_size, (bx + 1) * block_size
                                ymin, ymax = by * block_size, (by + 1) * block_size
                                distance_to_test = np.minimum(
                                    distance_to_test,
                                    rectangle_distance(x, y, xmin, xmax, ymin, ymax),
                                )
                            held_out_granules = set(work.loc[mask_test, "granule_id"].astype(str))
                            same_granule = work["granule_id"].astype(str).isin(held_out_granules).to_numpy()
                            mask_train = (~mask_test) & (distance_to_test > buffer_m) & (~same_granule)
                            if mask_test.sum() < 10 or mask_train.sum() < 100:
                                skipped_rows.append({
                                    "maximum_grid_distance_km": radius,
                                    "window_analysis": window,
                                    "target_scenario": target_name,
                                    "model": model_name,
                                    "block_size_m": block_size,
                                    "buffer_m": buffer_m,
                                    "spatial_fold": fold_id,
                                    "reason": "spatial fold has <10 test photons or buffered training set has <100",
                                    "test_photons": int(mask_test.sum()),
                                    "training_photons": int(mask_train.sum()),
                                    "test_blocks": len(test_ids),
                                })
                                continue
                            train = work.loc[mask_train]
                            test = work.loc[mask_test]
                            if train["granule_id"].nunique() < 3:
                                continue
                            estimator = clone(MODELS[model_name])
                            estimator.fit(train[feature_cols], train[target_column].to_numpy(dtype=float))
                            prediction = estimator.predict(test[feature_cols])
                            baseline = float(np.median(train[target_column].to_numpy(dtype=float)))
                            observed = test[target_column].to_numpy(dtype=float)
                            residual = prediction - observed
                            test_blocks = block_ids.loc[test.index]
                            for holdout in test_ids:
                                block_mask = test_blocks.eq(holdout).to_numpy()
                                block_y = observed[block_mask]
                                block_pred = prediction[block_mask]
                                block_rmse, block_mae, block_bias = metric(block_y, block_pred)
                                baseline_rmse = metric(block_y, np.full(len(block_y), baseline))[0]
                                block_test = test.loc[block_mask]
                                block_rows.append({
                                    "maximum_grid_distance_km": radius,
                                    "window_analysis": window,
                                    "target_scenario": target_name,
                                    "model": model_name,
                                    "block_size_m": block_size,
                                    "buffer_m": buffer_m,
                                    "spatial_fold": fold_id,
                                    "holdout_block": holdout,
                                    "test_photons": int(len(block_test)),
                                    "test_granules": int(block_test["granule_id"].nunique()),
                                    "training_photons": int(len(train)),
                                    "training_granules": int(train["granule_id"].nunique()),
                                    "RMSE_m": block_rmse,
                                    "MAE_m": block_mae,
                                    "bias_m": block_bias,
                                    "training_median_baseline_RMSE_m": baseline_rmse,
                                })
                            for position, (_, row) in enumerate(test.iterrows()):
                                holdout = test_blocks.iloc[position]
                                pred_rows.append({
                                    "granule_id": row["granule_id"],
                                    "beam": row["beam"],
                                    "source_photon_index": row["source_photon_index"],
                                    "scene_id": row.get("scene_id", ""),
                                    "acquisition_date": row.get("acquisition_date", ""),
                                    "maximum_grid_distance_km": radius,
                                    "window_analysis": window,
                                    "target_scenario": target_name,
                                    "model": model_name,
                                    "block_size_m": block_size,
                                    "buffer_m": buffer_m,
                                    "spatial_fold": fold_id,
                                    "holdout_block": holdout,
                                    "easting_m": float(row["easting_m"]),
                                    "northing_m": float(row["northing_m"]),
                                    "raw_depth_m": float(row["instantaneous_depth_m"]),
                                    "observed_depth_m": float(observed[position]),
                                    "prediction_depth_m": float(prediction[position]),
                                    "fold_median_baseline_depth_m": baseline,
                                    "residual_m": float(residual[position]),
                                    "error_sq": float(residual[position] ** 2),
                                    "baseline_error_sq": float((baseline - observed[position]) ** 2),
                                })
                            valid_folds += 1
                        print(
                            f"r={radius:g} km | {window} | {target_name} | {model_name} | "
                            f"block={block_size} m buffer={buffer_m} m: {valid_folds} valid spatial folds"
                        )

    out.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(pred_rows)
    blocks = pd.DataFrame(block_rows)
    pred.to_csv(out / "spatial_block_oof_predictions.csv.gz", index=False, compression="gzip")
    blocks.to_csv(out / "spatial_block_fold_metrics.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(out / "spatial_block_skipped_folds.csv", index=False)
    if pred.empty:
        raise SystemExit("No spatial holdout block had sufficient support; inspect skipped-fold audit.")
    rng = np.random.default_rng(args.seed)
    summary = pd.DataFrame(block_group_metrics(pred, args.replicates, rng))
    summary.to_csv(out / "spatial_block_transfer_summary.csv", index=False)
    depth_summary = pd.DataFrame(
        block_group_metrics(pred, args.replicates, rng, stratify_by_depth=True)
    )
    depth_summary.to_csv(out / "spatial_block_depth_transfer_summary.csv", index=False)
    make_figure(summary, out)
    metadata = {
        "input": str(input_path),
        "unique_photon_key": ["granule_id", "beam", "source_photon_index"],
        "maximum_distances_km": args.maximum_distances_km,
        "windows": args.windows,
        "block_sizes_m": args.block_sizes_m,
        "buffer_rule": "half of block width unless an explicit minimum buffer is supplied",
        "fold_assignment": "up to five contiguous bands of projected-easting block columns",
        "additional_independence_rule": "exclude all training photons from any ATL24 granule represented in the held-out spatial fold",
        "models": list(FEATURES),
        "targets": TARGETS,
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "interpretation": (
            "Buffered within-site spatial transfer diagnostic. It is not external validation, "
            "a datum correction, or accuracy evidence for unsampled seafloor."
        ),
    }
    (out / "spatial_block_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    caption = (
        "Figure. Buffered spatial-block out-of-fold skill of the multiband random forest relative to "
        "a training-set median baseline at a maximum IBI-grid distance of 5 km. Up to five "
        "contiguous easting-band folds are withheld in turn; training photons within half a block "
        "width of any held-out rectangle and all other photons from held-out ATL24 granules are "
        "excluded. Error bars are "
        "percentile intervals from resampling held-out blocks, "
        "conditional on the fitted spatial splits. Positive skill indicates lower equal-block RMSE "
        "than the training-median baseline. The 0–6 h subset is the near-synchronous primary window; "
        "0–24 h includes time-mismatched secondary observations. The relative IBI target is a regional "
        "model sensitivity, not independent bathymetric truth."
    )
    (out / "figure_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")
    main_rf = summary.loc[
        summary["maximum_grid_distance_km"].eq(5.0)
        & summary["model"].eq("Multiband_RandomForest")
    ].sort_values(["window_analysis", "target_scenario", "block_size_m"])
    result_lines = []
    for row in main_rf.itertuples(index=False):
        result_lines.append(
            f"{row.window_analysis}; {row.target_scenario}; {row.block_size_m} m: "
            f"equal-block RMSE={row.equal_block_RMSE_m:.2f} m "
            f"(95% block-bootstrap interval {row.equal_block_RMSE_CI95_low_m:.2f}–{row.equal_block_RMSE_CI95_high_m:.2f}); "
            f"equal-block skill={row.equal_block_skill_vs_training_median:.3f} "
            f"[{row.equal_block_skill_CI95_low:.3f}, {row.equal_block_skill_CI95_high:.3f}]; "
            f"pooled skill={row.pooled_skill_vs_training_median:.3f}; "
            f"held-out blocks={row.held_out_spatial_blocks}; granules={row.held_out_granules}."
        )
    (out / "spatial_block_key_results_EN.txt").write_text(
        "Buffered spatial-block SDB results (5 km IBI-distance scenario; multiband random forest).\n"
        "Skill is relative to the training-set median. Block intervals resample held-out blocks and are "
        "conditional on the spatial folds and fitted predictions.\n\n"
        + "\n".join(result_lines)
        + "\n\nThe primary 0–6 h estimates have limited independent ATL24 support (six granules). "
        "Pooled and equal-block skill can differ in sign because they weight photon-dense and sparse "
        "areas differently. The 0–24 h results are exploratory because this window includes 6–24 h "
        "temporal mismatches. Positive skill against a median baseline does not establish bathymetric "
        "accuracy; equal-block RMSE remains large and the study does not validate a continuous map.\n",
        encoding="utf-8",
    )
    main_depth = depth_summary.loc[
        depth_summary["maximum_grid_distance_km"].eq(5.0)
        & depth_summary["model"].eq("Multiband_RandomForest")
    ].sort_values([
        "window_analysis", "target_scenario", "block_size_m", "depth_stratum_raw_ATL24"
    ])
    depth_lines = []
    for row in main_depth.itertuples(index=False):
        depth_lines.append(
            f"{row.window_analysis}; {row.target_scenario}; {row.block_size_m} m; "
            f"{row.depth_stratum_raw_ATL24}: photons={row.held_out_photons}; "
            f"blocks={row.held_out_spatial_blocks}; equal-block RMSE={row.equal_block_RMSE_m:.2f} m; "
            f"skill={row.equal_block_skill_vs_training_median:.3f} "
            f"[{row.equal_block_skill_CI95_low:.3f}, {row.equal_block_skill_CI95_high:.3f}]; "
            f"status={row.support_status}."
        )
    (out / "spatial_block_depth_results_EN.txt").write_text(
        "Depth-stratified buffered spatial-block results; strata defined by raw ATL24 depth.\n"
        "Intervals are conditional on fitted spatial splits; sparse strata are descriptive only.\n\n"
        + "\n".join(depth_lines)
        + "\n",
        encoding="utf-8",
    )
    notes = (
        "Validation spatiale par blocs avec zone tampon. Chaque pli teste une bande spatiale contiguë; "
        "les photons proches des blocs de test et tous les autres photons des granules testés sont retirés "
        "de l’apprentissage. Cette procédure mesure un transfert spatial interne au site; elle ne remplace "
        "pas une validation sur un second site.\n\n"
        "Les métriques à pondération égale par bloc et les métriques groupées par photon sont toutes deux "
        "fournies, car elles peuvent diverger lorsque les photons sont concentrés dans certaines zones. "
        "La fenêtre 0–6 h est principale mais ne comprend que six granules indépendants. La fenêtre 0–24 h "
        "est exploratoire et inclut des coïncidences décalées de 6–24 h. Les intervalles bootstrap sont "
        "conditionnels aux plis spatiaux; ils ne propagent pas les incertitudes de datum, de marée, de "
        "qualité optique ou de modèle régional. Une compétence positive par rapport à la médiane ne suffit "
        "pas à établir la précision bathymétrique ni à valider une carte continue.\n\n"
        + "Résultats RF principaux :\n"
        + "\n".join(result_lines)
        + "\n"
    )
    (out / "interpretation_notes_FR.txt").write_text(notes, encoding="utf-8")
    print("Buffered spatial-block SDB transfer audit complete; source matchups were read only.")
    print(f"Valid held-out block predictions: {len(pred):,} photon-model rows")
    print(f"Summary: {out / 'spatial_block_transfer_summary.csv'}")
    print(f"Depth-stratified summary: {out / 'spatial_block_depth_transfer_summary.csv'}")
    print(f"Fold metrics and skipped-block audit: {out / 'spatial_block_fold_metrics.csv'}; {out / 'spatial_block_skipped_folds.csv'}")
    print(f"Figures: {out / 'Figure_spatial_blocked_SDB_transfer.png'} and PDF")
    print(f"Metadata and interpretation: {out / 'spatial_block_validation_metadata.json'}; {out / 'interpretation_notes_FR.txt'}")
    print(f"Key results (EN): {out / 'spatial_block_key_results_EN.txt'}")
    print(f"Depth results (EN): {out / 'spatial_block_depth_results_EN.txt'}")
    print("Interpret as within-site spatial transfer only; no continuous bathymetry map is validated.")


if __name__ == "__main__":
    main()
