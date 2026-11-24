from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results/Bathymetry/Transferability_Audit"


def scene_label(value: str) -> str:
    match = re.search(r"_(\d{8})_", str(value))
    if match:
        token = match.group(1)
        return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    return str(value)


def main() -> None:
    pair_path = ROOT / "Results/Bathymetry/atl24_sentinel2_training_pairs.csv.gz"
    prediction_path = ROOT / "Results/Bathymetry/grouped_sdb_oof_predictions.csv.gz"
    for path in (pair_path, prediction_path):
        if not path.is_file():
            raise SystemExit(f"Required SDB output missing: {path}. Run Scripts/08_fit_sdb_models.py first.")

    pairs = pd.read_csv(pair_path)
    pred = pd.read_csv(prediction_path)
    depth_col = "depth_at_sentinel2_time_m"
    required_pairs = {"granule_id", "scene_id", depth_col}
    required_pred = {
        "window_analysis", "target_scenario", "model", "granule_id", "scene_id",
        "observed_depth_m", "prediction_depth_m",
    }
    if required_pairs - set(pairs.columns):
        raise SystemExit(f"Pair table missing columns: {sorted(required_pairs - set(pairs.columns))}")
    if required_pred - set(pred.columns):
        raise SystemExit(f"Prediction table missing columns: {sorted(required_pred - set(pred.columns))}")

    OUT.mkdir(parents=True, exist_ok=True)
    support_rows = []
    for granule, group in pairs.groupby("granule_id", sort=True):
        values = pd.to_numeric(group[depth_col], errors="coerce").dropna()
        support_rows.append({
            "granule_id": granule,
            "scene_id": group["scene_id"].iloc[0],
            "scene_date": scene_label(group["scene_id"].iloc[0]),
            "n_pairs": int(len(values)),
            "depth_min_m": float(values.min()),
            "depth_q05_m": float(values.quantile(0.05)),
            "depth_median_m": float(values.median()),
            "depth_q95_m": float(values.quantile(0.95)),
            "depth_max_m": float(values.max()),
        })
    support = pd.DataFrame(support_rows).sort_values("scene_date")
    central_low = float(support["depth_q05_m"].max())
    central_high = float(support["depth_q95_m"].min())
    has_common = central_low < central_high
    support["n_pairs_in_all_granule_common_q05_q95_m"] = 0
    if has_common:
        for idx, row in support.iterrows():
            g = pairs.loc[pairs["granule_id"] == row["granule_id"], depth_col]
            support.loc[idx, "n_pairs_in_all_granule_common_q05_q95_m"] = int(g.between(central_low, central_high).sum())
    support_path = OUT / "depth_support_by_atl24_granule.csv"
    support.to_csv(support_path, index=False)

    valid = pred.loc[
        pred["window_analysis"].eq("combined_0_24h")
        & pred["target_scenario"].eq("GOT56_relative_tide_adjusted_depth")
    ].copy()
    valid = valid.dropna(subset=["observed_depth_m", "prediction_depth_m"])
    error_rows = []
    for (model, granule), group in valid.groupby(["model", "granule_id"], sort=True):
        obs = group["observed_depth_m"].to_numpy(dtype="float64")
        estimate = group["prediction_depth_m"].to_numpy(dtype="float64")
        error_rows.append({
            "model": model,
            "granule_id": granule,
            "scene_id": group["scene_id"].iloc[0],
            "scene_date": scene_label(group["scene_id"].iloc[0]),
            "n_pairs": int(len(group)),
            "bias_predicted_minus_observed_m": float(np.mean(estimate - obs)),
            "MAE_m": float(mean_absolute_error(obs, estimate)),
            "RMSE_m": float(np.sqrt(mean_squared_error(obs, estimate))),
            "R2_within_granule": float(r2_score(obs, estimate)) if len(obs) > 1 and np.var(obs) > 0 else float("nan"),
        })
    errors = pd.DataFrame(error_rows)
    error_path = OUT / "tide_adjusted_oof_error_by_granule.csv"
    errors.to_csv(error_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    ordered = support.sort_values("scene_date")
    box_data = [pairs.loc[pairs["granule_id"] == g, depth_col].dropna().to_numpy() for g in ordered["granule_id"]]
    ax.boxplot(box_data, tick_labels=ordered["scene_date"].tolist(), showfliers=False, patch_artist=True,
               medianprops={"color": "#142D4E", "linewidth": 1.5},
               boxprops={"facecolor": "#A9D3E2", "edgecolor": "#30758D"},
               whiskerprops={"color": "#30758D"}, capprops={"color": "#30758D"})
    ax.set_title("A. Tide-adjusted depth support by ATL24 pass", loc="left", weight="bold")
    ax.set_ylabel("Depth at Sentinel-2 acquisition time (m)")
    ax.set_xlabel("Sentinel-2 scene date")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", color="#D9E0E4", linewidth=0.6)
    if not has_common:
        ax.text(0.02, 0.98,
                f"No shared central 90% depth interval across all {len(support)} passes\n"
                f"max Q05 = {central_low:.2f} m; min Q95 = {central_high:.2f} m",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#AAB7C4", "alpha": 0.94})
    else:
        ax.axhspan(central_low, central_high, color="#F2B66D", alpha=0.22,
                   label=f"Shared central range: {central_low:.1f}–{central_high:.1f} m")
        ax.legend(frameon=False, fontsize=8)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(ordered))))
    color_by_scene = {r.scene_id: colors[i] for i, r in enumerate(ordered.itertuples())}
    model_order = ["Log_blue_green_ratio_OLS", "Multiband_RandomForest", "Multiband_HistGradientBoosting"]
    for ax, model, letter in zip(axes.flat[1:], model_order, ["B", "C", "D"]):
        gmodel = valid.loc[valid["model"].eq(model)]
        if gmodel.empty:
            ax.text(0.5, 0.5, "No out-of-fold predictions available", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        for scene_id, group in gmodel.groupby("scene_id", sort=True):
            ax.scatter(group["observed_depth_m"], group["prediction_depth_m"], s=18, alpha=0.7,
                       color=color_by_scene.get(scene_id, "#666666"), label=scene_label(scene_id), edgecolors="none")
        low = float(np.nanmin(np.r_[gmodel["observed_depth_m"].to_numpy(), gmodel["prediction_depth_m"].to_numpy()]))
        high = float(np.nanmax(np.r_[gmodel["observed_depth_m"].to_numpy(), gmodel["prediction_depth_m"].to_numpy()]))
        pad = max(0.5, 0.04 * (high - low))
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1, label="1:1 line")
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_title(f"{letter}. {model.replace('_', ' ')}", loc="left", weight="bold")
        ax.set_xlabel("Observed tide-adjusted depth (m)")
        ax.set_ylabel("Out-of-fold predicted depth (m)")
        ax.grid(color="#D9E0E4", linewidth=0.6)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Depth-domain support and cross-granule SDB transferability", fontsize=14, weight="bold")
    fig_path = OUT / "Figure_sdb_transferability_diagnostic.png"
    pdf_path = OUT / "Figure_sdb_transferability_diagnostic.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_atl24_granules": int(len(support)),
        "all_granule_central_90pct_depth_overlap_exists": bool(has_common),
        "max_granule_q05_m": central_low,
        "min_granule_q95_m": central_high,
        "interpretation": (
            "No shared central depth domain across all passes means pooled transfer metrics combine different depth populations. "
            "Negative pooled R2 and high cross-granule errors indicate poor transferability in this open-data sample; "
            "do not use these estimates to claim a validated bathymetric map."
            if not has_common else
            "A central common-support interval exists, but transferability must still be judged using the saved held-out predictions."
        ),
    }
    summary_path = OUT / "transferability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    note_path = OUT / "interpretation_notes_FR.txt"
    note_path.write_text(
        f"Audit de transférabilité SDB sur {len(support)} granules ATL24 indépendants. "
        f"Intervalle central commun Q05–Q95 : {'aucun' if not has_common else f'{central_low:.2f}–{central_high:.2f} m'}. "
        "Les erreurs par granule documentent l'hétérogénéité du transfert inter-scènes. "
        "Les résultats ne valident ni une bathymétrie continue, ni les pixels non échantillonnés; "
        "les corrections GOT5.6 restent relatives et sans contrôle local par marégraphe.\n",
        encoding="utf-8",
    )
    print("Read-only SDB transferability audit complete; fitted models and source observations were not changed.")
    print(f"Granules: {len(support)}; common central depth overlap: {'yes' if has_common else 'no'}")
    print(f"Depth support: {support_path}")
    print(f"Granule errors: {error_path}")
    print(f"Figures: {fig_path}; {pdf_path}")
    print(f"Summary and interpretation: {summary_path}; {note_path}")
    print("Use the audit to characterize domain shift; it does not create or validate a continuous bathymetric map.")


if __name__ == "__main__":
    main()
