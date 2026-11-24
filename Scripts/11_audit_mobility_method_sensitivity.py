from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Results" / "Sediment_Mobility" / "mobility_scenario_frequency_by_grid_d50_directional.csv"
OUT = ROOT / "Results" / "Sediment_Mobility" / "Method_Sensitivity"

METHODS = {
    "Separate-stress screen": "separate_stress_scenario_frequency_mean",
    "Soulsby directional": "soulsby_combined_max_scenario_frequency_mean",
    "Aligned envelope": "aligned_envelope_scenario_frequency_mean",
}


def rank_summary(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan"), float("nan")
    rho = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
    k = max(1, int(np.ceil(0.25 * len(pair))))
    top_a = set(pair.iloc[:, 0].nlargest(k).index)
    top_b = set(pair.iloc[:, 1].nlargest(k).index)
    overlap = len(top_a & top_b) / k
    return rho, float(overlap)


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing input: {INPUT}. Run Scripts/09_estimate_mobility_scenarios.py first.")
    df = pd.read_csv(INPUT)
    required = {"longitude_deg", "latitude_deg", "d50_scenario_mm", *METHODS.values(),
                "soulsby_combined_max_scenario_frequency_p10", "soulsby_combined_max_scenario_frequency_p90", "scenario_count"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Mobility output lacks the new directional-model fields: {sorted(missing)}. Rerun Script 09.")
    key = ["longitude_deg", "latitude_deg", "d50_scenario_mm"]
    if df.duplicated(key).any():
        raise SystemExit("Duplicate grid-node/D50 rows found; refusing ambiguous method comparison.")

    records = []
    summaries = []
    for d50, group in df.groupby("d50_scenario_mm", sort=True):
        g = group.set_index(["longitude_deg", "latitude_deg"]).sort_index()
        out = {"d50_scenario_mm": float(d50), "grid_nodes": int(len(g))}
        for label, col in METHODS.items():
            x = g[col].dropna()
            out[f"{label.lower().replace(' ', '_')}_nodes"] = int(len(x))
            out[f"{label.lower().replace(' ', '_')}_spatial_median"] = float(x.median()) if len(x) else np.nan
            out[f"{label.lower().replace(' ', '_')}_spatial_p10"] = float(x.quantile(0.10)) if len(x) else np.nan
            out[f"{label.lower().replace(' ', '_')}_spatial_p90"] = float(x.quantile(0.90)) if len(x) else np.nan

        valid = g[list(METHODS.values())].dropna()
        if len(valid):
            rho_separate, overlap_separate = rank_summary(valid[METHODS["Separate-stress screen"]], valid[METHODS["Soulsby directional"]])
            rho_aligned, overlap_aligned = rank_summary(valid[METHODS["Aligned envelope"]], valid[METHODS["Soulsby directional"]])
            k = max(1, int(np.ceil(0.25 * len(valid))))
            soulsby_top = set(valid[METHODS["Soulsby directional"]].nlargest(k).index)
            separate_top = set(valid[METHODS["Separate-stress screen"]].nlargest(k).index)
            aligned_top = set(valid[METHODS["Aligned envelope"]].nlargest(k).index)
            out.update({
                "paired_complete_nodes": int(len(valid)),
                "spearman_soulsby_vs_separate": rho_separate,
                "top_quartile_overlap_soulsby_vs_separate": overlap_separate,
                "spearman_soulsby_vs_aligned": rho_aligned,
                "top_quartile_overlap_soulsby_vs_aligned": overlap_aligned,
                "median_soulsby_minus_separate": float((valid[METHODS["Soulsby directional"]] - valid[METHODS["Separate-stress screen"]]).median()),
                "median_soulsby_minus_aligned": float((valid[METHODS["Soulsby directional"]] - valid[METHODS["Aligned envelope"]]).median()),
                "soulsby_top_quartile_nodes": len(soulsby_top),
            })
            out["soulsby_parameter_p10_p90_width_spatial_median"] = float(
                (g["soulsby_combined_max_scenario_frequency_p90"] - g["soulsby_combined_max_scenario_frequency_p10"]).median()
            )
        else:
            out.update({"paired_complete_nodes": 0})
        records.append(out)

        # Long table for a descriptive across-grid plot; spatial quantiles are
        # heterogeneity summaries and must not be interpreted as uncertainty CIs.
        for label, col in METHODS.items():
            vals = g[col].dropna()
            summaries.append({
                "d50_scenario_mm": float(d50),
                "method": label,
                "grid_nodes": int(len(vals)),
                "spatial_median": float(vals.median()) if len(vals) else np.nan,
                "spatial_p10": float(vals.quantile(0.10)) if len(vals) else np.nan,
                "spatial_p90": float(vals.quantile(0.90)) if len(vals) else np.nan,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / "mobility_method_sensitivity_by_d50.csv", index=False)
    long = pd.DataFrame(summaries)
    long.to_csv(OUT / "mobility_method_spatial_distribution.csv", index=False)

    colors = {
        "Separate-stress screen": "#2878A5",
        "Soulsby directional": "#3E8C76",
        "Aligned envelope": "#D17A32",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    d50s = sorted(long["d50_scenario_mm"].dropna().unique())
    xbase = np.arange(len(d50s), dtype=float)
    offsets = np.linspace(-0.16, 0.16, len(METHODS))
    for (label, col), offset in zip(METHODS.items(), offsets):
        sub = long.loc[long["method"].eq(label)].set_index("d50_scenario_mm").reindex(d50s)
        med = sub["spatial_median"].to_numpy(dtype=float)
        low = sub["spatial_p10"].to_numpy(dtype=float)
        high = sub["spatial_p90"].to_numpy(dtype=float)
        ax.errorbar(
            xbase + offset, med, yerr=np.vstack([med - low, high - med]),
            fmt="o-", color=colors[label], capsize=3, linewidth=1.25,
            markersize=5, label=label,
        )
    ax.set_xticks(xbase, [f"{x:g}" for x in d50s])
    ax.set_xlabel("Assumed median grain diameter, D₅₀ (mm)")
    ax.set_ylabel("Scenario exceedance frequency")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#d9dee3", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.01, -0.21,
        "Points are medians across spatial grid nodes; whiskers show the 10th–90th spatial percentiles, not confidence intervals.",
        transform=ax.transAxes, fontsize=8, color="#414a52", va="top",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"Figure_mobility_method_sensitivity.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    note = {
        "analysis": "descriptive comparison of three bed-stress formulations and their spatial ranking",
        "input_rows": int(len(df)),
        "d50_scenarios_mm": [float(v) for v in sorted(df["d50_scenario_mm"].dropna().unique())],
        "spatial_quantiles": "Summaries across forcing grid nodes describe spatial heterogeneity; they are not confidence intervals and nodes are spatially dependent.",
        "scenario_parameter_quantiles": "Soulsby P10/P90 values summarize the available prescribed depth, wave-height, and current-speed combinations at each node (maximum 27; shallow depth offsets may be omitted); they are not probabilistic intervals.",
        "scenario_count_range_by_row": {"minimum": int(df["scenario_count"].min()), "maximum": int(df["scenario_count"].max())},
        "interpretation": "No formulation is independently validated at Anza. The directional Soulsby calculation uses regional forcing and assumed bed properties; it is a sensitivity comparison, not sediment transport or surf-zone prediction.",
    }
    (OUT / "mobility_method_sensitivity_metadata.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    (OUT / "interpretation_notes_FR.txt").write_text(
        "Cet audit compare trois formulations de contrainte au fond sur les mêmes nœuds de grille et pour chaque D50. La corrélation de rang décrit la stabilité du classement spatial entre méthodes; elle ne valide pas les contraintes calculées. Le chevauchement du quartile supérieur est un indicateur de sensibilité descriptive.\n\n"
        "Les médianes et quantiles spatiaux décrivent l'hétérogénéité entre cellules; les cellules voisines ne sont pas indépendantes. Les P10–P90 des fréquences Soulsby décrivent la dispersion entre les scénarios prescrits disponibles de profondeur, de hauteur de vague et de vitesse du courant (jusqu'à 27 par cellule; certains décalages de profondeur peu profonde sont omis). Aucun intervalle ne constitue une incertitude probabiliste. Les forçages IBI restent régionaux, D50 est supposé et aucune mesure locale de flux ou de granulométrie ne valide les résultats.\n",
        encoding="utf-8",
    )
    print("Mobility formulation-sensitivity audit complete; source forcings and scenario outputs were read only.")
    print(f"D50 classes: {len(records)}; grid-node/D50 rows: {len(df)}")
    print(f"Summary: {OUT / 'mobility_method_sensitivity_by_d50.csv'}")
    print(f"Figure: {OUT / 'Figure_mobility_method_sensitivity.png'} and PDF")
    print(f"Interpretation: {OUT / 'interpretation_notes_FR.txt'}")
    print("Treat rank agreement as a method-sensitivity diagnostic, not validation or evidence of sediment flux.")


if __name__ == "__main__":
    main()
