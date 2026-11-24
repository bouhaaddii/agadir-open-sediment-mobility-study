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


def score_group(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["observed_depth_m"].to_numpy(dtype=float)
    pred = frame["prediction_depth_m"].to_numpy(dtype=float)
    baseline = frame["fold_median_baseline_depth_m"].to_numpy(dtype=float)
    error = pred - y
    baseline_error = baseline - y
    group = pd.DataFrame({
        "granule_id": frame["granule_id"].astype(str).to_numpy(),
        "model_sq_error": error ** 2,
        "baseline_sq_error": baseline_error ** 2,
    }).groupby("granule_id", sort=False).mean()
    model_granule_rmse = np.sqrt(group["model_sq_error"].to_numpy())
    baseline_granule_rmse = np.sqrt(group["baseline_sq_error"].to_numpy())
    pooled_rmse = float(np.sqrt(np.mean(error ** 2)))
    pooled_baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
    equal_granule_rmse = float(np.sqrt(np.mean(group["model_sq_error"])))
    equal_granule_baseline_rmse = float(np.sqrt(np.mean(group["baseline_sq_error"])))
    return {
        "pooled_RMSE_m": pooled_rmse,
        "pooled_baseline_RMSE_m": pooled_baseline_rmse,
        "pooled_RMSE_skill_vs_baseline": (
            1.0 - pooled_rmse / pooled_baseline_rmse if pooled_baseline_rmse > 0 else np.nan
        ),
        "equal_granule_RMSE_m": equal_granule_rmse,
        "equal_granule_baseline_RMSE_m": equal_granule_baseline_rmse,
        "equal_granule_RMSE_skill_vs_baseline": (
            1.0 - equal_granule_rmse / equal_granule_baseline_rmse
            if equal_granule_baseline_rmse > 0 else np.nan
        ),
        "macro_mean_granule_RMSE_m": float(np.mean(model_granule_rmse)),
        "MAE_m": float(np.mean(np.abs(error))),
        "bias_m": float(np.mean(error)),
        "median_absolute_error_m": float(np.median(np.abs(error))),
        "within_1m_fraction": float(np.mean(np.abs(error) <= 1.0)),
        "within_2m_fraction": float(np.mean(np.abs(error) <= 2.0)),
        "within_5m_fraction": float(np.mean(np.abs(error) <= 5.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit out-of-fold SDB skill by observed depth using the stable-photon Script 18 output."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "Depth_Support"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not input_path.is_file():
        raise SystemExit(
            f"Missing Script 18 OOF predictions: {input_path}. "
            "Run Scripts/18_deduplicate_atl24_sentinel2_matchups.py first."
        )

    data = pd.read_csv(input_path, low_memory=False)
    required = [
        "maximum_grid_distance_km", "window_analysis", "target_scenario", "model",
        "observed_depth_m", "prediction_depth_m", "fold_median_baseline_depth_m",
        "granule_id", "scene_id", "time_gap_minutes",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit(f"Required OOF columns are missing: {missing}")
    numeric = [
        "maximum_grid_distance_km", "observed_depth_m", "prediction_depth_m",
        "fold_median_baseline_depth_m", "time_gap_minutes",
    ]
    for column in numeric:
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

    rows = []
    grouping = [
        "maximum_grid_distance_km", "window_analysis", "target_scenario", "model", "depth_bin"
    ]
    for keys, frame in data.groupby(grouping, observed=True, sort=True):
        radius, window, target, model, depth_bin = keys
        group_sizes = frame.groupby("granule_id").size()
        metrics = score_group(frame)
        n_groups = int(frame["granule_id"].nunique())
        effective_granules = float(1.0 / np.sum((group_sizes / group_sizes.sum()) ** 2))
        largest_share = float(group_sizes.max() / group_sizes.sum())
        rows.append({
            "maximum_grid_distance_km": float(radius),
            "window_analysis": window,
            "target_scenario": target,
            "model": model,
            "depth_bin": str(depth_bin),
            "n_photons": int(len(frame)),
            "n_granules": n_groups,
            "n_scenes": int(frame["scene_id"].nunique()),
            "effective_granule_count_by_photon_weight": float(
                1.0 / np.sum((group_sizes / group_sizes.sum()) ** 2)
            ),
            "smallest_granule_n": int(group_sizes.min()),
            "largest_granule_n": int(group_sizes.max()),
            "largest_granule_share": largest_share,
            "median_observed_depth_m": float(frame["observed_depth_m"].median()),
            "median_time_gap_min": float(frame["time_gap_minutes"].median()),
            **metrics,
            "support_status": (
                "replicated_and_balanced_depth_stratum"
                if n_groups >= 5 and len(frame) >= 50 and effective_granules >= 3.0 and largest_share <= 0.5
                else "descriptive_sparse_or_unbalanced_depth_stratum"
            ),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("No finite out-of-fold predictions fell inside the declared 0–40 m depth range.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sdb_oof_skill_by_depth_bin.csv"
    summary.to_csv(summary_path, index=False)

    # Four panels compare depth-stratified, held-out errors for both distance
    # sensitivities and both vertical-target scenarios. All captions and axes are in English.
    figure_data = summary.loc[summary["window_analysis"].eq("combined_0_24h")]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=True, sharey=True)
    panels = [
        (2.5, "raw_instantaneous_depth", "A", "≤2.5 km | instantaneous depth"),
        (2.5, "IBI_zos_relative_adjustment", "B", "≤2.5 km | IBI-relative adjustment"),
        (5.0, "raw_instantaneous_depth", "C", "≤5 km | instantaneous depth"),
        (5.0, "IBI_zos_relative_adjustment", "D", "≤5 km | IBI-relative adjustment"),
    ]
    x = np.arange(len(BIN_LABELS))
    for ax, (radius, target, panel, title) in zip(axes.flat, panels):
        part = figure_data.loc[
            figure_data["maximum_grid_distance_km"].eq(radius)
            & figure_data["target_scenario"].eq(target)
        ]
        baseline = (
            part[["depth_bin", "equal_granule_baseline_RMSE_m"]]
            .drop_duplicates("depth_bin")
            .set_index("depth_bin")
            .reindex(BIN_LABELS)["equal_granule_baseline_RMSE_m"]
        )
        ax.plot(x, baseline.to_numpy(dtype=float), color="#252B33", linestyle="--",
                marker="o", linewidth=1.6, markersize=4.5, label="Fold-specific median baseline")
        for model in MODEL_ORDER:
            model_part = (
                part.loc[part["model"].eq(model)]
                .set_index("depth_bin")
                .reindex(BIN_LABELS)
            )
            y = model_part["equal_granule_RMSE_m"].to_numpy(dtype=float)
            ax.plot(x, y, color=MODEL_COLORS[model], marker="o", linewidth=1.7,
                    markersize=4.5, label=MODEL_LABELS[model])
        for j, label in enumerate(BIN_LABELS):
            count_rows = part.loc[part["depth_bin"].eq(label)]
            n = int(count_rows["n_photons"].max()) if not count_rows.empty else 0
            g = int(count_rows["n_granules"].max()) if not count_rows.empty else 0
            ax.text(j, 0.02, f"n={n}; g={g}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color="#555555")
        ax.set_title(f"({panel}) {title}", loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", color="#DDE2E7", linewidth=0.7)
        ax.set_xticks(x, BIN_LABELS, rotation=0)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Granule-equal out-of-fold RMSE (m)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=8.5)
    fig.supxlabel("Observed depth stratum", y=0.075, fontsize=10)
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0.10, 1, 1), h_pad=1.8, w_pad=1.6)
    png_path = output_dir / "Figure_SDB_depth_stratified_OOF_RMSE.png"
    pdf_path = output_dir / "Figure_SDB_depth_stratified_OOF_RMSE.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Depth-stratified out-of-fold performance of optical satellite-derived bathymetry "
        "candidates using the nearest Sentinel-2 acquisition per ATL24 source photon. "
        "Predictions were generated with ATL24 granules held out as groups. Panels compare "
        "the instantaneous ATL24 depth target and the IBI-relative sea-level-adjusted target "
        "at two maximum distances to the nearest wet IBI grid cell. The dashed line is the "
        "fold-specific training-set median-depth baseline. "
        "Curves show equal-granule RMSE, and labels give photon and granule counts in each depth stratum. Sparse or imbalanced strata should not be interpreted as "
        "validated depth limits; ATL24 uncertainty, IBI model error, datum uncertainty and "
        "spatial sampling uncertainty are not propagated."
    )
    (output_dir / "figure_caption_EN.txt").write_text(caption + chr(10), encoding="utf-8")
    notes = chr(10).join([
        "Audit des performances hors-échantillon par classe de profondeur",
        "",
        "Les prédictions sont celles de la validation groupée par granule ATL24 et proviennent de l'audit à identifiant photon source. Les classes de profondeur sont fixées avant l'inspection des scores : 0–2, 2–5, 5–10, 10–15, 15–20 et 20–40 m. Les métriques par classe décrivent les erreurs sur les photons observés dans cette classe; elles ne constituent pas une validation de pixels bathymétriques non échantillonnés.",
        "",
        "La RMSE à poids égaux entre granules est fournie pour éviter qu'un granule très dense domine seul le résumé. Un appui répliqué et équilibré exige au moins 50 photons, cinq granules, un effectif effectif de granules supérieur ou égal à 3 et aucune contribution individuelle supérieure à 50 %. Les autres classes sont descriptives. Aucune classe n'est déclarée comme limite de détection sans support indépendant et stable. Les erreurs de datum, d'IBI, de classification optique, de pénétration lumineuse et les incertitudes ATL24 ne sont pas propagées."
    ])
    (output_dir / "interpretation_notes_FR.txt").write_text(notes + chr(10), encoding="utf-8")
    print("Depth-stratified out-of-fold SDB audit complete; no model was refitted.")
    print("The table includes pooled and equal-granule error, support counts, and baseline skill by depth stratum.")
    print(f"Table: {summary_path}")
    print(f"Figure: {png_path}")
    print(f"Vector PDF: {pdf_path}")
    print(f"Caption and interpretation: {output_dir / 'figure_caption_EN.txt'}; {output_dir / 'interpretation_notes_FR.txt'}")
    print("Do not treat a sparse or high-skill depth bin as a validated continuous bathymetry range.")


if __name__ == "__main__":
    main()



