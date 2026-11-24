from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Figures"


def save_mobility_figure() -> bool:
    directional_table = ROOT / "Results" / "Sediment_Mobility" / "mobility_scenario_frequency_by_grid_d50_directional.csv"
    baseline_table = ROOT / "Results" / "Sediment_Mobility" / "mobility_scenario_frequency_by_grid_d50.csv"
    table_path = directional_table if directional_table.exists() else baseline_table
    if not table_path.exists():
        return False

    df = pd.read_csv(table_path)
    required = {
        "longitude_deg", "latitude_deg", "d50_scenario_mm",
        "separate_stress_scenario_frequency_mean",
        "aligned_envelope_scenario_frequency_mean",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Mobility table is missing required fields: {sorted(missing)}")

    d50_values = sorted(df["d50_scenario_mm"].dropna().unique())
    if len(d50_values) != 4:
        print(f"Warning: expected four D50 scenarios; found {len(d50_values)}.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    row_specs = [
        ("separate_stress_scenario_frequency_mean", "Separate-stress screen"),
        ("aligned_envelope_scenario_frequency_mean", "Aligned envelope"),
    ]
    if "soulsby_combined_max_scenario_frequency_mean" in df.columns:
        row_specs.append(("soulsby_combined_max_scenario_frequency_mean", "Soulsby combined maximum"))
    nrows = len(row_specs)
    fig, axes = plt.subplots(nrows, max(1, len(d50_values)), figsize=(13.2, 3.0 * nrows + 0.7), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(nrows, -1)
    last_scatter = None
    lat_mid = float(df["latitude_deg"].median())
    for row, (col, row_title) in enumerate(row_specs):
        for column, d50 in enumerate(d50_values):
            ax = axes[row, column]
            subset = df.loc[np.isclose(df["d50_scenario_mm"], d50)]
            last_scatter = ax.scatter(
                subset["longitude_deg"], subset["latitude_deg"],
                c=subset[col], cmap="viridis", vmin=0, vmax=1,
                s=42, marker="s", edgecolors="white", linewidths=0.35,
                rasterized=True,
            )
            ax.set_title(f"D₅₀ = {d50:g} mm\n(n = {len(subset)} forcing nodes)", pad=7)
            ax.grid(True, color="#d9dee3", linewidth=0.55, alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_aspect(np.cos(np.deg2rad(lat_mid)))
            ax.tick_params(length=3, width=0.7, colors="#333333")
            for spine in ax.spines.values():
                spine.set_color("#7a858e")
                spine.set_linewidth(0.7)
            if row == nrows - 1:
                ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)" if column == 0 else "")

    top, bottom = 0.96, 0.13
    fig.subplots_adjust(left=0.12, right=0.89, bottom=bottom, top=top, wspace=0.13, hspace=0.28)
    for row, (_, row_title) in enumerate(row_specs):
        y = top - (row + 0.5) * (top - bottom) / nrows
        fig.text(0.04, y, row_title, rotation=90, ha="center", va="center", fontsize=8.5, weight="bold")
    cax = fig.add_axes([0.92, 0.20, 0.018, 0.62])
    cb = fig.colorbar(last_scatter, cax=cax)
    cb.set_label("Mean scenario exceedance frequency", labelpad=8)
    fig.text(
        0.5, 0.035,
        "Conditional screening across prescribed depth, wave-height, and current-speed scenarios; not a probability or observed sediment transport.",
        ha="center", va="bottom", fontsize=8, color="#414a52",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"Figure_mobility_scenario_envelopes.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return True


def save_sdb_figure() -> bool:
    table_path = ROOT / "Results" / "Bathymetry" / "grouped_sdb_model_metrics.csv"
    if not table_path.exists():
        return False
    df = pd.read_csv(table_path)
    required = {"window_analysis", "target_scenario", "model", "n", "independent_atl24_granules", "status", "RMSE_m", "R2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SDB metrics table is missing required fields: {sorted(missing)}")

    valid = df.loc[df["status"].eq("grouped_cross_validation")].copy()
    if valid.empty:
        return False
    windows = [
        ("extended_6_24h", "Extended matchup (6–24 h)"),
        ("combined_0_24h", "Combined matchup (0–24 h)"),
    ]
    model_order = [
        "Log_blue_green_ratio_OLS",
        "Multiband_RandomForest",
        "Multiband_HistGradientBoosting",
    ]
    model_labels = ["Blue–green OLS", "Random forest", "Histogram gradient boosting"]
    scenarios = [
        ("uncorrected_instantaneous_depth", "Uncorrected", "o", "#2878A5"),
        ("GOT56_relative_tide_adjusted_depth", "Relative-tide adjusted", "D", "#D17A32"),
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
    legend_handles = []
    for row, (window, window_label) in enumerate(windows):
        sub = valid.loc[valid["window_analysis"].eq(window)]
        n = int(sub["n"].max()) if not sub.empty else 0
        granules = int(sub["independent_atl24_granules"].max()) if not sub.empty else 0
        axes[row, 0].set_title(f"{window_label} | {n} pairs, {granules} tracks")
        axes[row, 1].set_title(f"{window_label} | {n} pairs, {granules} tracks")
        for col, metric, ylabel in [(0, "RMSE_m", "Grouped out-of-track RMSE (m)"), (1, "R2", "Grouped out-of-track R²")]:
            ax = axes[row, col]
            for model_idx, (model, label) in enumerate(zip(model_order, model_labels)):
                model_rows = sub.loc[sub["model"].eq(model)]
                points = []
                for x_offset, (scenario, scenario_label, marker, color) in zip((-0.12, 0.12), scenarios):
                    hit = model_rows.loc[model_rows["target_scenario"].eq(scenario)]
                    if hit.empty:
                        continue
                    value = float(hit.iloc[0][metric])
                    points.append((model_idx + x_offset, value, color, marker, scenario_label))
                    handle = ax.scatter(
                        model_idx + x_offset, value, s=38, color=color, marker=marker,
                        edgecolor="white", linewidth=0.45, zorder=3,
                        label=scenario_label if (row == 0 and col == 0) else None,
                    )
                    if row == 0 and col == 0:
                        legend_handles.append(handle)
                if len(points) == 2:
                    ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], color="#89939b", linewidth=0.8, zorder=1)
            ax.set_xticks(range(len(model_labels)), model_labels, rotation=12, ha="right")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="#d9dee3", linewidth=0.55)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#7a858e")
                spine.set_linewidth(0.7)
            if metric == "RMSE_m":
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(0, color="#343a40", linewidth=0.8, linestyle="--")
    # Keep a single shared legend and make the core transferability result explicit.
    unique_handles = []
    unique_labels = []
    for handle in legend_handles:
        label = handle.get_label()
        if label and label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    fig.legend(unique_handles, unique_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.text(
        0.5, 0.025,
        "The 0–6 h subset had only two independent ATL24 tracks and is not scored. All evaluable grouped-CV R² values are negative; transfer skill is not demonstrated.",
        ha="center", va="bottom", fontsize=8, color="#414a52",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.94), h_pad=1.5, w_pad=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"Figure_sdb_grouped_validation.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mobility = save_mobility_figure()
    sdb = save_sdb_figure()
    print(f"Mobility figure: {'updated' if mobility else 'skipped (input table absent)'}")
    print(f"SDB transferability figure: {'updated' if sdb else 'skipped (no grouped-CV rows)'}")
    print(f"Output directory: {OUT}")
    captions = [
        "Figure 1. Spatial patterns of the mean scenario exceedance frequency for four assumed median grain diameters (D50). Rows show (top to bottom) the separate-stress screen, the aligned-stress envelope, and, when the wave-direction input is available, the Soulsby DATA2 directional combined-stress estimate. Frequencies summarize prescribed depth, wave-height and current-speed scenarios under regional forcing; they are not calibrated probabilities, observed sediment transport, or a surf-zone-resolving model.",
        "Figure 2. Grouped out-of-track satellite-derived bathymetry performance for uncorrected instantaneous depth and relative-tide-adjusted depth, shown for the 6–24 h and 0–24 h matchup windows. The 0–6 h subset was not scored because it contained only two independent ATL24 tracks. Negative grouped cross-validation R2 values indicate that transferable predictive skill was not demonstrated. The absence of common central depth support among tracks further limits interpretation.",
    ]
    (OUT / "figure_captions_EN.txt").write_text("\n\n".join(captions) + "\n", encoding="utf-8")
    (OUT / "figure_interpretation_FR.txt").write_text(
        "Les cartes de mobilité représentent une fréquence moyenne de dépassement calculée sur des scénarios prescrits de profondeur, de hauteur de vague et de vitesse du courant. La nouvelle estimation directionnelle combinée emploie la paramétrisation empirique DATA2 de Soulsby et la direction de propagation dérivée de VMDR; elle n'est pas une simulation locale de couche limite ou de déferlement. Les quantiles P10–P90 décrivent la dispersion parmi les 27 combinaisons de paramètres prescrites, pas une incertitude probabiliste. Aucune de ces fréquences n'est un flux sédimentaire observé. Les diamètres D50 sont hypothétiques faute de mesures granulométriques locales.\n\n"
        "L'évaluation SDB par validation croisée groupée selon les traces ATL24 ne montre pas de compétence prédictive transférable : les R² sont négatifs pour toutes les configurations évaluables. La fenêtre primaire 0–6 h n'est pas notée car elle ne contient que deux traces indépendantes. L'absence de recouvrement central des profondeurs entre toutes les traces indique un décalage de domaine; les résultats ne justifient donc pas la production ou la présentation d'une carte bathymétrique validée.\n",
        encoding="utf-8",
    )
    print(f"Captions: {OUT / 'figure_captions_EN.txt'}")
    print(f"Interpretation notes: {OUT / 'figure_interpretation_FR.txt'}")


if __name__ == "__main__":
    main()
