from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "Data/Processed/ATL24/atl24_bathy_points.csv.gz"
EMODNET = ROOT / "Data/Raw/Bathymetry/emodnet_mean_2022_anza.tif"
OUT = ROOT / "Results/Bathymetry/ATL24_EMODnet_Relative_Shape"
PRIMARY_MIN_PHOTONS_PER_CELL = 3
MIN_CELLS_PER_PASS = 5
SENSITIVITY_PHOTON_THRESHOLDS = (1, 3, 5)


def corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    fn = pearsonr if kind == "pearson" else spearmanr
    return float(fn(x, y).statistic)


def summarize_pairs(pairs: pd.DataFrame, min_photons: int) -> tuple[pd.DataFrame, dict]:
    data = pairs.loc[pairs["n_photons"] >= min_photons].copy()
    if data.empty:
        return data, {"min_photons_per_cell": min_photons, "matched_cells": 0, "passes": 0}

    pass_sizes = data.groupby("granule_id").size()
    valid_passes = pass_sizes[pass_sizes >= MIN_CELLS_PER_PASS].index
    data = data[data["granule_id"].isin(valid_passes)].copy()
    if data.empty:
        return data, {"min_photons_per_cell": min_photons, "matched_cells": 0, "passes": 0}

    data["atl24_depth_anomaly_m"] = data["atl24_depth_m"] - data.groupby("granule_id")["atl24_depth_m"].transform("median")
    data["emodnet_depth_anomaly_m"] = data["emodnet_depth_m"] - data.groupby("granule_id")["emodnet_depth_m"].transform("median")
    data["relative_shape_residual_m"] = data["atl24_depth_anomaly_m"] - data["emodnet_depth_anomaly_m"]

    per_pass = []
    for granule, group in data.groupby("granule_id", sort=True):
        x = group["emodnet_depth_anomaly_m"].to_numpy(float)
        y = group["atl24_depth_anomaly_m"].to_numpy(float)
        residual = y - x
        per_pass.append({
            "granule_id": granule,
            "matched_cells": int(len(group)),
            "photons_in_matched_cells": int(group["n_photons"].sum()),
            "median_atl24_depth_m": float(group["atl24_depth_m"].median()),
            "median_emodnet_depth_LAT_m": float(group["emodnet_depth_m"].median()),
            "raw_median_depth_difference_m": float(np.median(group["atl24_depth_m"] - group["emodnet_depth_m"])),
            "pearson_relative_shape_r": corr(x, y, "pearson"),
            "spearman_relative_shape_rho": corr(x, y, "spearman"),
            "relative_shape_MAE_m": float(np.mean(np.abs(residual))),
            "relative_shape_RMSE_m": float(np.sqrt(np.mean(residual ** 2))),
            "atl24_depth_median_sigma_tvu_m": float(group["median_atl24_sigma_tvu_m"].median()),
        })
    by_pass = pd.DataFrame(per_pass)
    x = data["emodnet_depth_anomaly_m"].to_numpy(float)
    y = data["atl24_depth_anomaly_m"].to_numpy(float)
    residual = y - x
    metrics = {
        "min_photons_per_cell": min_photons,
        "matched_cells": int(len(data)),
        "passes": int(data["granule_id"].nunique()),
        "pooled_Pearson_r": corr(x, y, "pearson"),
        "pooled_Spearman_rho": corr(x, y, "spearman"),
        "pooled_relative_shape_MAE_m": float(np.mean(np.abs(residual))),
        "pooled_relative_shape_RMSE_m": float(np.sqrt(np.mean(residual ** 2))),
        "median_pass_Pearson_r": float(by_pass["pearson_relative_shape_r"].median()),
        "median_pass_Spearman_rho": float(by_pass["spearman_relative_shape_rho"].median()),
        "median_pass_relative_shape_MAE_m": float(by_pass["relative_shape_MAE_m"].median()),
        "median_pass_relative_shape_RMSE_m": float(by_pass["relative_shape_RMSE_m"].median()),
        "median_absolute_raw_pass_offset_m": float(np.median(np.abs(by_pass["raw_median_depth_difference_m"]))),
    }
    return data, {**metrics, "_pass_table": by_pass}


def main() -> None:
    for path in (POINTS, EMODNET):
        if not path.exists():
            raise SystemExit(f"Required input missing: {path}. Run the ATL24 extraction and EMODnet download stages first.")
    OUT.mkdir(parents=True, exist_ok=True)

    points = pd.read_csv(POINTS)
    required = {"granule_id", "longitude_deg", "latitude_deg", "instantaneous_depth_m", "sigma_tvu_1sigma_m"}
    missing = required.difference(points.columns)
    if missing:
        raise SystemExit(f"ATL24 point table lacks required columns: {sorted(missing)}")

    with rasterio.open(EMODNET) as src:
        if src.count < 1:
            raise SystemExit("The EMODnet raster contains no bands.")
        if src.crs is None:
            raise SystemExit("The EMODnet raster has no declared CRS; refusing an implicit coordinate match.")
        xy = zip(points["longitude_deg"].to_numpy(float), points["latitude_deg"].to_numpy(float))
        sampled_z = np.asarray([v[0] for v in src.sample(xy)], dtype="float64")
        rc = [src.index(lon, lat) for lon, lat in zip(points.longitude_deg, points.latitude_deg)]
        points["emodnet_row"] = [v[0] for v in rc]
        points["emodnet_col"] = [v[1] for v in rc]
        points["emodnet_elevation_source_sign_m"] = sampled_z
        # The downloaded EMODnet mean-depth grid stores submerged cells as negative values.
        # Convert to positive-down depth for a like-for-like relative-shape comparison only.
        points["emodnet_depth_LAT_m"] = -sampled_z
        bounds = [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)]
        resolution = [float(v) for v in src.res]
        crs = str(src.crs)

    valid = (
        np.isfinite(points["instantaneous_depth_m"])
        & np.isfinite(points["emodnet_depth_LAT_m"])
        & (points["emodnet_elevation_source_sign_m"] < 0)
        & (points["instantaneous_depth_m"] > 0)
    )
    underwater = points.loc[valid].copy()
    if underwater.empty:
        raise SystemExit("No ATL24 points overlap negative-valued underwater cells in the EMODnet raster.")

    cell_pass = underwater.groupby(["granule_id", "emodnet_row", "emodnet_col"], as_index=False).agg(
        atl24_depth_m=("instantaneous_depth_m", "median"),
        emodnet_depth_m=("emodnet_depth_LAT_m", "median"),
        n_photons=("instantaneous_depth_m", "size"),
        median_atl24_sigma_tvu_m=("sigma_tvu_1sigma_m", "median"),
        median_confidence=("confidence", "median"),
        cell_longitude_deg=("longitude_deg", "median"),
        cell_latitude_deg=("latitude_deg", "median"),
    )

    sensitivity_rows = []
    primary_data = pd.DataFrame()
    primary_passes = pd.DataFrame()
    primary_metrics: dict = {}
    for threshold in SENSITIVITY_PHOTON_THRESHOLDS:
        matched, summary = summarize_pairs(cell_pass, threshold)
        pass_table = summary.pop("_pass_table", pd.DataFrame())
        sensitivity_rows.append(summary)
        if threshold == PRIMARY_MIN_PHOTONS_PER_CELL:
            primary_data, primary_metrics, primary_passes = matched, summary, pass_table

    primary_data.to_csv(OUT / "atl24_emodnet_relative_shape_cell_pairs.csv.gz", index=False, compression="gzip")
    primary_passes.to_csv(OUT / "atl24_emodnet_relative_shape_by_granule.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "atl24_emodnet_relative_shape_threshold_sensitivity.csv", index=False)

    summary_for_json = {
        "analysis": "per-granule-median-centered relative relief comparison; no vertical shift was estimated or applied",
        "atl24_input": str(POINTS),
        "emodnet_input": str(EMODNET),
        "emodnet_crs": crs,
        "emodnet_bounds": bounds,
        "emodnet_resolution_degrees": resolution,
        "atl24_input_photons": int(len(points)),
        "photons_over_negative_emodnet_cells": int(len(underwater)),
        "granules_with_any_overlap": int(underwater.granule_id.nunique()),
        "primary_min_photons_per_cell": PRIMARY_MIN_PHOTONS_PER_CELL,
        "minimum_matched_cells_per_granule": MIN_CELLS_PER_PASS,
        "primary_summary": {k: v for k, v in primary_metrics.items() if not k.startswith("_")},
        "threshold_sensitivity": sensitivity_rows,
        "interpretation": "Descriptive relative-relief comparison only. ATL24 instantaneous depth and EMODnet LAT-referenced DTM differ in vertical reference, acquisition support, spatial resolution and source lineage. Per-pass centering removes a constant offset and therefore cannot establish a datum tie; it also does not remove spatially varying tide or product errors.",
    }
    (OUT / "atl24_emodnet_relative_shape_summary.json").write_text(json.dumps(summary_for_json, indent=2), encoding="utf-8")

    if not primary_data.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
        x = primary_data["emodnet_depth_anomaly_m"].to_numpy(float)
        y = primary_data["atl24_depth_anomaly_m"].to_numpy(float)
        axes[0].scatter(x, y, s=14, alpha=0.45, color="#287C8E", edgecolors="none", rasterized=True)
        lim = float(np.nanpercentile(np.abs(np.concatenate([x, y])), 98))
        lim = max(lim, 1.0)
        axes[0].plot([-lim, lim], [-lim, lim], color="#555555", lw=1.1, ls="--", label="1:1 reference")
        axes[0].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="EMODnet within-pass depth anomaly (m)", ylabel="ATL24 within-pass depth anomaly (m)")
        axes[0].set_title("A  Relative relief after per-pass centering", loc="left", fontweight="bold")
        axes[0].text(0.03, 0.97, f"{len(primary_data):,} pass–cell pairs; {primary_data.granule_id.nunique()} passes\nPearson r = {primary_metrics.get('pooled_Pearson_r', np.nan):.2f}; Spearman ρ = {primary_metrics.get('pooled_Spearman_rho', np.nan):.2f}", transform=axes[0].transAxes, va="top", ha="left", fontsize=8.5, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9})
        axes[0].legend(frameon=False, loc="lower right", fontsize=8)

        ps = primary_passes.dropna(subset=["spearman_relative_shape_rho"]).sort_values("matched_cells")
        axes[1].scatter(ps["matched_cells"], ps["spearman_relative_shape_rho"], s=30, alpha=0.78, color="#D17A22", edgecolors="white", linewidths=0.35)
        axes[1].axhline(0, color="#555555", lw=0.8)
        axes[1].set(xlabel="Matched EMODnet cells per ATL24 granule", ylabel="Within-granule Spearman ρ", ylim=(-1.05, 1.05))
        axes[1].set_title("B  Track-to-track consistency", loc="left", fontweight="bold")
        axes[1].text(0.03, 0.97, f"Median ρ = {ps['spearman_relative_shape_rho'].median():.2f}\nMinimum {MIN_CELLS_PER_PASS} cells; ≥{PRIMARY_MIN_PHOTONS_PER_CELL} photons per cell", transform=axes[1].transAxes, va="top", ha="left", fontsize=8.5, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9})
        for ax in axes:
            ax.grid(alpha=0.2, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Open bathymetry and ICESat-2: a relative-shape diagnostic", fontsize=13, fontweight="bold")
        fig.text(0.5, -0.025, "Descriptive comparison only. Per-pass centering removes a constant offset; it does not resolve the LAT–EGM08 datum difference or spatially varying tide.", ha="center", fontsize=8)
        fig.savefig(OUT / "Figure_ATL24_EMODnet_relative_shape.png", dpi=350, bbox_inches="tight")
        fig.savefig(OUT / "Figure_ATL24_EMODnet_relative_shape.pdf", bbox_inches="tight")
        plt.close(fig)

    notes = [
        "Diagnostic de forme relative ATL24–EMODnet",
        "",
        "Le script apparie chaque photon ATL24 à la cellule EMODnet correspondante, puis agrège les photons par cellule et par granule. L'analyse principale exige au moins trois photons par cellule et cinq cellules appariées par granule.",
        "Les profondeurs sont centrées par la médiane de chaque granule avant la comparaison. Cette opération élimine un décalage vertical constant propre à un passage; elle ne transforme pas les références verticales et ne fournit donc aucun raccordement entre LAT et EGM08.",
        "Les métriques sont descriptives. Les traces peuvent partager des cellules et les points ne sont pas des observations indépendantes. Les incertitudes ATL24 fournies sont conservées dans les tables mais ne remplacent pas une validation indépendante.",
        "Une faible concordance de forme indique que l'EMODnet utilisé n'est pas une référence locale adéquate pour évaluer l'altimétrie ATL24 à cette échelle. Une forte concordance ne suffirait pas, à elle seule, à valider les deux produits ni à établir un datum vertical.",
        "Ne pas interpréter les différences brutes par passage comme une correction de marée ou un décalage de datum: elles combinent au minimum le niveau instantané de la mer, les références verticales, la résolution et les erreurs propres aux deux produits.",
    ]
    (OUT / "interpretation_notes_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    caption = (
        "Figure. Relative-shape comparison between ICESat-2 ATL24 instantaneous depth and the EMODnet DTM 2022. "
        "ATL24 photons were aggregated by EMODnet grid cell and granule (at least three photons per cell in the primary analysis); "
        "only granules with at least five matched cells were retained. Both depth series were centered by their within-granule median, "
        "so panel A evaluates relative relief after removing a constant pass-specific offset, not absolute depth agreement. "
        "Panel B shows Spearman rank correlation for each retained granule. The products differ in temporal support, spatial resolution, "
        "source lineage, and vertical reference (ATL24 EGM08 heights/instantaneous depth versus EMODnet LAT-referenced depth); "
        "the comparison is descriptive and is neither an independent validation nor a vertical datum tie."
    )
    (OUT / "figure_caption_EN.txt").write_text(caption + "\n", encoding="utf-8")

    print("ATL24–EMODnet relative-shape audit complete; input photons and bathymetry were read only.")
    print(f"ATL24 photons: {len(points):,}; photons over underwater EMODnet cells: {len(underwater):,}; overlapping granules: {underwater.granule_id.nunique()}")
    if primary_metrics:
        print(f"Primary support: {primary_metrics['matched_cells']:,} pass-cell pairs across {primary_metrics['passes']} granules")
        print(f"Pooled relative-shape Pearson r={primary_metrics['pooled_Pearson_r']:.3f}; Spearman rho={primary_metrics['pooled_Spearman_rho']:.3f}; RMSE={primary_metrics['pooled_relative_shape_RMSE_m']:.2f} m")
        print(f"Median per-granule Spearman rho={primary_metrics['median_pass_Spearman_rho']:.3f}")
    print(f"Outputs: {OUT}")
    print("This removes only a constant per-granule offset; it is not a vertical datum tie or independent validation.")


if __name__ == "__main__":
    main()
