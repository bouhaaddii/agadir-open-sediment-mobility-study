from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Data/Raw/Tide_Models"


def predict_no_extrapolation(lon: np.ndarray, lat: np.ndarray, time_utc: pd.Series) -> np.ndarray:
    import pyTMD

    dates = pd.to_datetime(time_utc, utc=True, errors="coerce")
    dates = dates.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    result = pyTMD.compute.tide_elevations(
        x=np.asarray(lon, dtype="float64"),
        y=np.asarray(lat, dtype="float64"),
        delta_time=dates,
        directory=MODEL_DIR,
        model="GOT5.6",
        crs="EPSG:4326",
        type="trajectory",
        standard="datetime",
        method="linear",
        extrapolate=False,
        infer_minor=True,
    )
    values = np.asarray(result.values, dtype="float64").reshape(-1)
    if len(values) != len(dates):
        raise RuntimeError(f"Unexpected GOT5.6 result length: {len(values)} for {len(dates)} records.")
    return values


def main() -> None:
    source = ROOT / "Results/Matchup_Audit/atl24_surface_height_by_granule.csv"
    if not source.is_file():
        raise SystemExit("Run Scripts/06b_audit_atl24_surface_height.py first.")
    if not MODEL_DIR.is_dir():
        raise SystemExit("GOT5.6 model directory is missing. Run Scripts/07_download_got56.py --download first.")
    try:
        import pyTMD  # noqa: F401
    except ImportError as exc:
        raise SystemExit("pyTMD is missing. Run: .\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements.txt") from exc

    data = pd.read_csv(source)
    needed = {
        "granule_id", "n_bathymetry_photons", "acquisition_midpoint_utc",
        "surface_height_median_egm08_m", "surface_height_q25_egm08_m",
        "surface_height_q75_egm08_m", "longitude_median_deg", "latitude_median_deg",
    }
    missing = sorted(needed - set(data.columns))
    if missing:
        raise SystemExit(f"ATL24 surface-height table lacks fields: {', '.join(missing)}")

    data["acquisition_midpoint_utc"] = pd.to_datetime(data["acquisition_midpoint_utc"], utc=True, errors="coerce")
    for col in ["n_bathymetry_photons", "surface_height_median_egm08_m", "longitude_median_deg", "latitude_median_deg"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["acquisition_midpoint_utc", "surface_height_median_egm08_m", "longitude_median_deg", "latitude_median_deg"])
    data = data.loc[data["n_bathymetry_photons"] >= 10].copy().reset_index(drop=True)
    if len(data) < 3:
        raise SystemExit(f"Only {len(data)} ATL24 passes have at least 10 photons; insufficient for a model-comparison diagnostic.")

    print("Evaluating GOT5.6 tide-only predictions at ATL24 pass centroids.")
    print(f"Passes with >=10 retained bathymetry photons: {len(data)}")
    print("Spatial extrapolation is explicitly disabled; unsupported coastal predictions remain missing.")
    data["got56_tide_m"] = predict_no_extrapolation(
        data["longitude_median_deg"].to_numpy(),
        data["latitude_median_deg"].to_numpy(),
        data["acquisition_midpoint_utc"],
    )
    data["model_supported"] = np.isfinite(data["got56_tide_m"])
    supported = data.loc[data["model_supported"]].copy()
    if supported.empty:
        out = ROOT / "Results/Matchup_Audit"
        out.mkdir(parents=True, exist_ok=True)
        table_path = out / "atl24_surface_vs_got56_pass_comparison.csv"
        data.to_csv(table_path, index=False)
        raise SystemExit(
            "GOT5.6 has no valid interpolated predictions at these coastal centroids without extrapolation. "
            f"Support table saved: {table_path}. Do not enable extrapolation; test a finer coastal model next."
        )

    supported["observed_surface_anomaly_m"] = (
        supported["surface_height_median_egm08_m"] - supported["surface_height_median_egm08_m"].median()
    )
    supported["got56_tide_anomaly_m"] = supported["got56_tide_m"] - supported["got56_tide_m"].median()
    supported["centered_residual_m"] = supported["observed_surface_anomaly_m"] - supported["got56_tide_anomaly_m"]
    raw_resid = supported["surface_height_median_egm08_m"] - supported["got56_tide_m"]
    supported["raw_residual_observed_minus_model_m"] = raw_resid

    loo_errors = []
    residual_values = raw_resid.to_numpy(dtype="float64")
    for i in range(len(residual_values)):
        train = np.delete(residual_values, i)
        loo_errors.append(residual_values[i] - float(np.median(train)))
    supported["leave_one_pass_out_offset_residual_m"] = loo_errors

    observed_anom = supported["observed_surface_anomaly_m"].to_numpy(dtype="float64")
    model_anom = supported["got56_tide_anomaly_m"].to_numpy(dtype="float64")
    centered_error = observed_anom - model_anom
    corr = float(np.corrcoef(observed_anom, model_anom)[0, 1]) if len(supported) >= 3 and np.std(observed_anom) and np.std(model_anom) else float("nan")
    summary = {
        "model": "GOT5.6 standard ocean-tide constituents",
        "spatial_extrapolation": False,
        "n_atl24_passes_with_ge10_photons": int(len(data)),
        "n_model_supported_passes": int(len(supported)),
        "support_fraction": float(len(supported) / len(data)),
        "metrics_on_model_supported_passes": {
            "median_observed_minus_model_offset_m": float(np.median(residual_values)),
            "centered_anomaly_correlation": corr,
            "centered_anomaly_MAE_m": float(np.mean(np.abs(centered_error))),
            "centered_anomaly_RMSE_m": float(np.sqrt(np.mean(centered_error ** 2))),
            "leave_one_pass_out_constant_offset_RMSE_m": float(np.sqrt(np.mean(np.square(loo_errors)))),
        },
        "interpretation": (
            "Exploratory consistency check against ATL24-derived surface_h estimates. "
            "It is not tide-gauge validation: residuals include non-tidal sea-level variability, wave/setup effects, "
            "surface-height uncertainty, spatial sampling, and any vertical-reference offset. The constant offset is "
            "removed only for anomaly comparison; it is not a vertical datum transformation."
        ),
    }
    out = ROOT / "Results/Matchup_Audit"
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "atl24_surface_vs_got56_pass_comparison.csv"
    summary_path = out / "atl24_surface_vs_got56_summary.json"
    note_path = out / "atl24_surface_vs_got56_interpretation_FR.txt"
    fig_path = out / "Figure_ATL24_surface_vs_GOT56_anomalies.png"
    data.to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    note_path.write_text(
        "Comparaison exploratoire entre surface_h d'ATL24 et les anomalies de marée prédites par GOT5.6. "
        "Les résumés ATL24 sont agrégés par granule et limités aux passages disposant d'au moins 10 points. "
        "Aucune extrapolation spatiale n'est autorisée; les prédictions côtières non prises en charge restent manquantes. "
        "La médiane des résidus observé-moins-modèle sert uniquement à retirer un décalage constant pour comparer les anomalies. "
        "Les résidus incluent les variations non tidales, la houle et le setup, l'incertitude de surface_h, l'échantillonnage spatial "
        "et les écarts de référence verticale. Ce diagnostic ne vaut ni validation par marégraphe ni raccordement vertical.\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.3), constrained_layout=True)
    supported = supported.sort_values("acquisition_midpoint_utc")
    ax.plot(supported["acquisition_midpoint_utc"], supported["observed_surface_anomaly_m"], "o-", ms=4.5,
            linewidth=1.0, color="#176B87", label="ATL24 surface_h anomaly")
    ax.plot(supported["acquisition_midpoint_utc"], supported["got56_tide_anomaly_m"], "s--", ms=4.0,
            linewidth=1.0, color="#D27A2C", label="GOT5.6 astronomical tide anomaly")
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_title("ATL24 surface-height anomalies and GOT5.6 tide predictions", loc="left", weight="bold")
    ax.set_xlabel("ATL24 acquisition time (UTC)")
    ax.set_ylabel("Centered height (m)")
    ax.grid(axis="y", color="#D9E0E4", linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = summary["metrics_on_model_supported_passes"]
    print("Read-only GOT5.6/ATL24 pass-level anomaly comparison complete.")
    print(f"Model-supported passes: {len(supported)}/{len(data)}")
    print(f"Centered anomaly correlation: {metrics['centered_anomaly_correlation']:.3f}")
    print(f"Centered anomaly RMSE: {metrics['centered_anomaly_RMSE_m']:.3f} m")
    print(f"Leave-one-pass-out constant-offset RMSE: {metrics['leave_one_pass_out_constant_offset_RMSE_m']:.3f} m")
    print(f"Table: {table_path}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {fig_path}")
    print(f"Interpretation: {note_path}")
    print("Use this as a model-consistency diagnostic only; no tide-gauge validation or datum tie is claimed.")


if __name__ == "__main__":
    main()