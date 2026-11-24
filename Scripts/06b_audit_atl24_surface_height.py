from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "Data/Processed/ATL24/atl24_bathy_points.csv.gz"
    if not source.is_file():
        raise SystemExit("Run Scripts/06_extract_atl24_points.py first.")

    data = pd.read_csv(
        source,
        usecols=[
            "granule_id", "utc_time_source_epoch", "surface_height_egm08_m",
            "longitude_deg", "latitude_deg",
        ],
    )
    data["time_utc"] = pd.to_datetime(data["utc_time_source_epoch"], utc=True, errors="coerce")
    data["surface_height_egm08_m"] = pd.to_numeric(data["surface_height_egm08_m"], errors="coerce")
    data = data.dropna(subset=["granule_id", "time_utc", "surface_height_egm08_m"])
    if data.empty:
        raise SystemExit("No finite ATL24 sea-surface-height estimates are available in the extracted point table.")

    summary = (
        data.groupby("granule_id", sort=True)
        .agg(
            n_bathymetry_photons=("surface_height_egm08_m", "size"),
            acquisition_start_utc=("time_utc", "min"),
            acquisition_end_utc=("time_utc", "max"),
            surface_height_median_egm08_m=("surface_height_egm08_m", "median"),
            surface_height_q25_egm08_m=("surface_height_egm08_m", lambda x: x.quantile(0.25)),
            surface_height_q75_egm08_m=("surface_height_egm08_m", lambda x: x.quantile(0.75)),
            longitude_median_deg=("longitude_deg", "median"),
            latitude_median_deg=("latitude_deg", "median"),
        )
        .reset_index()
    )
    summary["acquisition_midpoint_utc"] = summary["acquisition_start_utc"] + (
        summary["acquisition_end_utc"] - summary["acquisition_start_utc"]
    ) / 2
    summary["surface_height_iqr_m"] = (
        summary["surface_height_q75_egm08_m"] - summary["surface_height_q25_egm08_m"]
    )
    summary["support_ge_10_photons"] = summary["n_bathymetry_photons"] >= 10
    summary = summary.sort_values("acquisition_midpoint_utc").reset_index(drop=True)

    out = ROOT / "Results/Matchup_Audit"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "atl24_surface_height_by_granule.csv"
    summary.to_csv(csv_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

    dates = summary["acquisition_midpoint_utc"]
    med = summary["surface_height_median_egm08_m"].to_numpy(dtype=float)
    q25 = summary["surface_height_q25_egm08_m"].to_numpy(dtype=float)
    q75 = summary["surface_height_q75_egm08_m"].to_numpy(dtype=float)
    yerr = np.vstack([np.maximum(0.0, med - q25), np.maximum(0.0, q75 - med)])
    enough = summary["support_ge_10_photons"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    ax.errorbar(
        dates[enough], med[enough], yerr=yerr[:, enough], fmt="o", ms=5.3,
        capsize=2.3, color="#176B87", ecolor="#7C9EAA", alpha=0.9,
        label="At least 10 bathymetry photons per ATL24 granule",
    )
    ax.scatter(
        dates[~enough], med[~enough], marker="x", s=48, linewidth=1.5,
        color="#C05A35", label="Fewer than 10 photons; weak support", zorder=3,
    )
    ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.7)
    ax.set_title("ATL24-derived sea-surface height at coastal overpasses", loc="left", weight="bold")
    ax.set_xlabel("ATL24 acquisition time (UTC)")
    ax.set_ylabel("Median surface_h relative to EGM08 (m)")
    ax.grid(axis="y", color="#D9E0E4", linewidth=0.7)
    ax.legend(frameon=False, loc="best", fontsize=8.5)
    ax.text(
        0.01, -0.20,
        "Bars show the within-granule interquartile range, not a confidence interval. "
        "Granule-level summaries avoid treating photons from one pass as independent tide observations.",
        transform=ax.transAxes, fontsize=8.5, va="top", color="#444444",
    )
    fig_path = out / "Figure_ATL24_surface_height_pass_diagnostic.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    n_supported = int(summary["support_ge_10_photons"].sum())
    note = (
        "Diagnostic exploratoire des hauteurs de surface ATL24\n\n"
        f"Granules résumés : {len(summary)}; granules avec au moins 10 photons bathymétriques : {n_supported}.\n"
        f"Photons retenus : {len(data):,}.\n"
        f"Étendue des médianes par granule (EGM08) : "
        f"{summary['surface_height_median_egm08_m'].min():.3f} à "
        f"{summary['surface_height_median_egm08_m'].max():.3f} m.\n\n"
        "surface_h est une estimation de hauteur orthométrique de la surface marine relative à EGM08. "
        "Les valeurs sont agrégées au niveau du granule pour éviter la pseudo-réplication. "
        "L'intervalle interquartile décrit la dispersion le long du support spatial échantillonné; "
        "il ne constitue ni un intervalle de confiance ni une validation par marégraphe.\n\n"
        "Cette table constitue un contrôle satellitaire potentiel des anomalies de niveau d'eau prédites "
        "aux passages ICESat-2. Elle ne remplace pas un marégraphe, ne raccorde pas le zéro hydrographique "
        "d'Agadir au datum national et ne fournit pas, à elle seule, le niveau d'eau à l'heure Sentinel-2."
    )
    note_path = out / "atl24_surface_height_diagnostic_FR.txt"
    note_path.write_text(note, encoding="utf-8")

    print("ATL24 surface-height pass diagnostic complete; source HDF5 and extracted points were read only.")
    print(f"Granules summarized: {len(summary)}; granules with >=10 photons: {n_supported}")
    print(
        "Range of per-granule median surface height (EGM08): "
        f"{summary['surface_height_median_egm08_m'].min():.3f} to "
        f"{summary['surface_height_median_egm08_m'].max():.3f} m"
    )
    print(f"Table: {csv_path}")
    print(f"Figure: {fig_path}")
    print(f"Interpretation limits: {note_path}")
    print("This is a satellite-derived surface-height diagnostic, not tide-gauge validation or a datum tie.")


if __name__ == "__main__":
    main()