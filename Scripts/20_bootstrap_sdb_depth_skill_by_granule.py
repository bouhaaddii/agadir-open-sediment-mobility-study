from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "Results/Bathymetry/ATL24_Sentinel2_IBI_SSH_24h"
    / "Unique_Photon_Audit/nearest_scene_unique_photon_oof_predictions.csv.gz"
)
BIN_EDGES = [0, 2, 5, 10, 15, 20, 40]
BIN_LABELS = ["0–2 m", "2–5 m", "5–10 m", "10–15 m", "15–20 m", "20–40 m"]
MODEL_ORDER = [
    "Log_blue_green_ratio_OLS",
    "Multiband_RandomForest",
    "Multiband_HistGradientBoosting",
]
MODEL_LABELS = {
    "Log_blue_green_ratio_OLS": "Blue–green log ratio",
    "Multiband_RandomForest": "Random Forest",
    "Multiband_HistGradientBoosting": "HistGradientBoosting",
}
MODEL_COLORS = {
    "Log_blue_green_ratio_OLS": "#667085",
    "Multiband_RandomForest": "#17807E",
    "Multiband_HistGradientBoosting": "#D97732",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster-bootstrap depth-specific SDB skill across independent ATL24 granules."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    if args.replicates < 200:
        raise SystemExit("--replicates must be at least 200 for percentile intervals.")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "Granule_Bootstrap"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not input_path.is_file():
        raise SystemExit(f"Missing Script 18 OOF predictions: {input_path}")

    data = pd.read_csv(input_path, low_memory=False)
    required = [
        "maximum_grid_distance_km", "window_analysis", "target_scenario", "model",
        "observed_depth_m", "prediction_depth_m", "fold_median_baseline_depth_m",
        "granule_id", "scene_id",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit(f"Required OOF columns are missing: {missing}")
    for column in (
        "maximum_grid_distance_km", "observed_depth_m",
        "prediction_depth_m", "fold_median_baseline_depth_m",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[
        np.isfinite(data["observed_depth_m"])
        & np.isfinite(data["prediction_depth_m"])
        & np.isfinite(data["fold_median_baseline_depth_m"])
        & data["observed_depth_m"].between(BIN_EDGES[0], BIN_EDGES[-1], inclusive="left")
    ].copy()
    data["depth_bin"] = pd.cut(
        data["observed_depth_m"], bins=BIN_EDGES, labels=BIN_LABELS,
        right=False, include_lowest=True
    )

    rng = np.random.default_rng(args.seed)
    grouping = [
        "maximum_grid_distance_km", "window_analysis", "target_scenario", "model", "depth_bin"
    ]
    rows = []
    for keys, frame in data.groupby(grouping, observed=True, sort=True):
        radius, window, target, model, depth_bin = keys
        work = frame.copy()
        work["error_sq"] = (work["prediction_depth_m"] - work["observed_depth_m"]) ** 2
        work["baseline_error_sq"] = (
            work["fold_median_baseline_depth_m"] - work["observed_depth_m"]
        ) ** 2
        group_stats = work.groupby("granule_id", sort=True).agg(
            model_mse=("error_sq", "mean"),
            baseline_mse=("baseline_error_sq", "mean"),
            n_photons=("observed_depth_m", "size"),
        )
        n_groups = len(group_stats)
        n_photons = int(len(work))
        model_mse = group_stats["model_mse"].to_numpy(dtype=float)
        baseline_mse = group_stats["baseline_mse"].to_numpy(dtype=float)
        if n_groups == 0:
            continue

        point_model_rmse = float(np.sqrt(np.mean(model_mse)))
        point_baseline_rmse = float(np.sqrt(np.mean(baseline_mse)))
        point_skill = (
            1.0 - point_model_rmse / point_baseline_rmse
            if point_baseline_rmse > 0 else np.nan
        )
        draw_idx = rng.integers(0, n_groups, size=(args.replicates, n_groups))
        boot_model_rmse = np.sqrt(np.mean(model_mse[draw_idx], axis=1))
        boot_baseline_rmse = np.sqrt(np.mean(baseline_mse[draw_idx], axis=1))
        boot_skill = np.divide(
            boot_baseline_rmse - boot_model_rmse,
            boot_baseline_rmse,
            out=np.full(args.replicates, np.nan),
            where=boot_baseline_rmse > 0,
        )
        model_lo, model_hi = np.nanquantile(boot_model_rmse, [0.025, 0.975])
        skill_lo, skill_hi = np.nanquantile(boot_skill, [0.025, 0.975])
        group_shares = group_stats["n_photons"] / n_photons
        effective_groups = float(1.0 / np.sum(group_shares.to_numpy() ** 2))
        rows.append({
            "maximum_grid_distance_km": float(radius),
            "window_analysis": window,
            "target_scenario": target,
            "model": model,
            "depth_bin": str(depth_bin),
            "n_photons": n_photons,
            "n_granules": int(n_groups),
            "effective_granule_count_by_photon_weight": effective_groups,
            "largest_granule_share": float(group_shares.max()),
            "granule_equal_RMSE_m": point_model_rmse,
            "granule_equal_RMSE_CI95_low_m": float(model_lo),
            "granule_equal_RMSE_CI95_high_m": float(model_hi),
            "granule_equal_baseline_RMSE_m": point_baseline_rmse,
            "granule_equal_skill_vs_baseline": float(point_skill),
            "granule_equal_skill_CI95_low": float(skill_lo),
            "granule_equal_skill_CI95_high": float(skill_hi),
            "bootstrap_replicates": int(args.replicates),
            "bootstrap_unit": "ATL24 granule; resampled with all its held-out photon residuals",
            "interval_scope": "conditional on saved out-of-fold predictions; model fitting uncertainty is not re-estimated",
            "support_status": (
                "replicated_and_balanced"
                if n_groups >= 5 and n_photons >= 50 and effective_groups >= 3.0 and group_shares.max() <= 0.5
                else "sparse_or_unbalanced"
            ),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("No finite out-of-fold predictions were available for bootstrap.")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sdb_depth_skill_granule_bootstrap_CI.csv"
    summary.to_csv(csv_path, index=False)

    # Display the primary distance sensitivities with uncertainty bars from
    # whole-granule resampling. Intervals are conditional on the saved OOF predictions.
    fig_data = summary.loc[
        summary["window_analysis"].eq("combined_0_24h")
        & summary["maximum_grid_distance_km"].isin([2.5, 5.0])
    ]
    panels = [
        (2.5, "raw_instantaneous_depth", "A", "≤2.5 km | instantaneous depth"),
        (2.5, "IBI_zos_relative_adjustment", "B", "≤2.5 km | IBI-relative adjustment"),
        (5.0, "raw_instantaneous_depth", "C", "≤5 km | instantaneous depth"),
        (5.0, "IBI_zos_relative_adjustment", "D", "≤5 km | IBI-relative adjustment"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=True, sharey=True)
    x = np.arange(len(BIN_LABELS))
    offsets = {
        MODEL_ORDER[0]: -0.18,
        MODEL_ORDER[1]: 0.0,
        MODEL_ORDER[2]: 0.18,
    }
    for ax, (radius, target, panel, title) in zip(axes.flat, panels):
        part = fig_data.loc[
            fig_data["maximum_grid_distance_km"].eq(radius)
            & fig_data["target_scenario"].eq(target)
        ]
        for model in MODEL_ORDER:
            m = (
                part.loc[part["model"].eq(model)]
                .set_index("depth_bin")
                .reindex(BIN_LABELS)
            )
            point = m["granule_equal_skill_vs_baseline"].to_numpy(dtype=float)
            lo = m["granule_equal_skill_CI95_low"].to_numpy(dtype=float)
            hi = m["granule_equal_skill_CI95_high"].to_numpy(dtype=float)
            lower_error = np.maximum(0.0, point - lo)
            upper_error = np.maximum(0.0, hi - point)
            ax.errorbar(
                x + offsets[model], point, yerr=np.vstack([lower_error, upper_error]),
                color=MODEL_COLORS[model], marker="o", linestyle="none",
                capsize=2.5, markersize=4.5, linewidth=1.1, label=MODEL_LABELS[model],
            )
        ax.axhline(0.0, color="#252B33", linestyle="--", linewidth=1.2)
        ax.set_title(f"({panel}) {title}", loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", color="#DDE2E7", linewidth=0.7)
        ax.set_xticks(x, BIN_LABELS)
        ax.set_ylabel("Granule-equal RMSE skill vs median baseline")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=8.5)
    fig.supxlabel("Observed depth stratum", y=0.075, fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 1), h_pad=1.8, w_pad=1.6)
    png_path = output_dir / "Figure_SDB_depth_skill_granule_bootstrap.png"
    pdf_path = output_dir / "Figure_SDB_depth_skill_granule_bootstrap.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Depth-specific out-of-fold skill relative to a fold-specific median-depth baseline. "
        "Error bars show 95% percentile intervals from resampling complete ATL24 granules, "
        "not individual photons. Panels compare two maximum distances and two target scenarios. "
        "Intervals are conditional on the saved grouped out-of-fold predictions and do not "
        "include model-refitting, vertical datum, IBI, optical classification, or unsampled-area uncertainty."
    )
    (output_dir / "figure_caption_EN.txt").write_text(caption + chr(10), encoding="utf-8")
    notes = chr(10).join([
        "Intervalles d'incertitude par rééchantillonnage des granules ATL24",
        "",
        "Les unités bootstrap sont les granules ATL24 entiers, afin de ne pas traiter les photons d'un même passage comme indépendants. Les intervalles portent sur les prédictions hors-échantillon déjà enregistrées; les modèles ne sont pas réentraînés dans chaque réplication. Ils quantifient donc la variabilité entre granules conditionnelle à ces prédictions, sans propager l'incertitude de recalibrage du modèle, de datum, d'IBI, de classification optique ou des zones non échantillonnées.",
        "",
        "Les intervalles doivent être considérés comme fragiles lorsque le nombre effectif de granules est faible ou lorsque quelques granules concentrent l'essentiel des photons. Ils ne constituent pas une validation de la bathymétrie continue."
    ])
    (output_dir / "interpretation_notes_FR.txt").write_text(notes + chr(10), encoding="utf-8")
    print("Granule-cluster bootstrap of depth-stratified out-of-fold skill complete; models were not refitted.")
    print(f"Replicates per stratum: {args.replicates}; seed: {args.seed}")
    print(f"Table: {csv_path}")
    print(f"Figure: {png_path}")
    print(f"Vector PDF: {pdf_path}")
    print(f"Caption and interpretation: {output_dir / 'figure_caption_EN.txt'}; {output_dir / 'interpretation_notes_FR.txt'}")
    print("Intervals are conditional on saved out-of-fold predictions; do not present them as total model or datum uncertainty.")


if __name__ == "__main__":
    main()
