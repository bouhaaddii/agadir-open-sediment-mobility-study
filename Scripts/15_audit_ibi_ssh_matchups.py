from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SSH_PATH = ROOT / "Data/Raw/Forcing/IBI_ssh_Anza_hourly.nc"
ATL_PATH = ROOT / "Data/Processed/ATL24/atl24_bathy_points.csv.gz"
OUT = ROOT / "Results/Bathymetry/IBI_SSH_Support_Audit"
SUPPORT_RADII_KM = (2.5, 5.0, 10.0)
PRIMARY_RADIUS_KM = 5.0


def first_present(names: tuple[str, ...], available: set[str]) -> str:
    for name in names:
        if name in available:
            return name
    raise KeyError(f"Could not find a coordinate from {names} in {sorted(available)}")


def interpolate_time(values: np.ndarray, times_ns: np.ndarray, target_ns: int) -> float:
    right = int(np.searchsorted(times_ns, target_ns, side="left"))
    if right < len(times_ns) and times_ns[right] == target_ns:
        value = values[right]
        return float(value) if np.isfinite(value) else float("nan")
    if right == 0 or right >= len(times_ns):
        return float("nan")
    left = right - 1
    v0, v1 = values[left], values[right]
    if not np.isfinite(v0) or not np.isfinite(v1):
        return float("nan")
    fraction = (target_ns - times_ns[left]) / (times_ns[right] - times_ns[left])
    return float(v0 + fraction * (v1 - v0))


def correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float((spearmanr if rank else pearsonr)(x, y).statistic)


def main() -> None:
    for path in (SSH_PATH, ATL_PATH):
        if not path.exists():
            raise SystemExit(f"Required input missing: {path}. Run the IBI SSH download and ATL24 extraction first.")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    points = pd.read_csv(ATL_PATH)
    points["utc_time_source_epoch"] = pd.to_datetime(points["utc_time_source_epoch"], utc=True, errors="coerce")
    points = points.dropna(subset=["granule_id", "utc_time_source_epoch", "longitude_deg", "latitude_deg", "surface_height_egm08_m"])
    passes = points.groupby("granule_id", as_index=False).agg(
        atl24_time_utc=("utc_time_source_epoch", "median"),
        atl24_longitude_deg=("longitude_deg", "median"),
        atl24_latitude_deg=("latitude_deg", "median"),
        atl24_surface_height_egm08_m=("surface_height_egm08_m", "median"),
        atl24_points=("surface_height_egm08_m", "size"),
    )

    with xr.open_dataset(SSH_PATH) as ds:
        variable = "zos"
        if variable not in ds:
            raise SystemExit(f"Hourly IBI SSH file lacks zos. Available variables: {list(ds.data_vars)}")
        time_name = first_present(("time", "valid_time", "TIME"), set(ds.coords) | set(ds.dims))
        lon_name = first_present(("longitude", "lon", "nav_lon"), set(ds.coords) | set(ds.data_vars))
        lat_name = first_present(("latitude", "lat", "nav_lat"), set(ds.coords) | set(ds.data_vars))
        da = ds[variable]
        spatial_dims = tuple(dim for dim in da.dims if dim != time_name)
        if len(spatial_dims) != 2:
            raise SystemExit(f"Expected two horizontal zos dimensions, got {da.dims}")
        ssh = da.transpose(time_name, *spatial_dims).values.astype("float64")
        lon_da = ds[lon_name]
        lat_da = ds[lat_name]
        if lon_da.ndim == 1 and lat_da.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon_da.values.astype(float), lat_da.values.astype(float))
        elif lon_da.ndim == 2 and lat_da.ndim == 2:
            lon_grid = lon_da.transpose(*spatial_dims).values.astype(float)
            lat_grid = lat_da.transpose(*spatial_dims).values.astype(float)
        else:
            raise SystemExit(f"Unsupported IBI coordinates: longitude dims={lon_da.dims}, latitude dims={lat_da.dims}")
        if ssh.shape[1:] != lon_grid.shape or lon_grid.shape != lat_grid.shape:
            raise SystemExit(f"Grid shape mismatch: zos={ssh.shape}, lon={lon_grid.shape}, lat={lat_grid.shape}")
        time_values = pd.to_datetime(ds[time_name].values, utc=True, errors="coerce")
        if time_values.isna().any():
            raise SystemExit("IBI SSH time coordinate contains invalid timestamps.")
        time_ns = time_values.asi8
        temporal_start = time_values.min().isoformat()
        temporal_end = time_values.max().isoformat()

    valid_nodes = np.isfinite(ssh).any(axis=0) & np.isfinite(lon_grid) & np.isfinite(lat_grid)
    valid_flat = np.flatnonzero(valid_nodes.ravel())
    if not len(valid_flat):
        raise SystemExit("No wet/valid IBI SSH grid cells are present in the downloaded subset.")
    transformer = Transformer.from_crs("EPSG:4326", cfg.get("crs_projected", "EPSG:32629"), always_xy=True)
    node_x, node_y = transformer.transform(lon_grid.ravel()[valid_flat], lat_grid.ravel()[valid_flat])
    tree = cKDTree(np.column_stack([node_x, node_y]))
    pass_x, pass_y = transformer.transform(passes.atl24_longitude_deg.to_numpy(float), passes.atl24_latitude_deg.to_numpy(float))
    distance_m, nearest = tree.query(np.column_stack([pass_x, pass_y]), k=1)
    nearest_flat = valid_flat[nearest]
    node_ids = [f"{np.unravel_index(int(i), valid_nodes.shape)[0]}_{np.unravel_index(int(i), valid_nodes.shape)[1]}" for i in nearest_flat]

    records = []
    for i, row in passes.iterrows():
        flat_index = int(nearest_flat[i])
        r, c = np.unravel_index(flat_index, valid_nodes.shape)
        model_series = ssh[:, r, c]
        target_ns = int(row.atl24_time_utc.value)
        eta = interpolate_time(model_series, time_ns, target_ns)
        records.append({
            **row.to_dict(),
            "ibi_nearest_grid_row": int(r),
            "ibi_nearest_grid_col": int(c),
            "ibi_nearest_grid_id": node_ids[i],
            "ibi_grid_longitude_deg": float(lon_grid[r, c]),
            "ibi_grid_latitude_deg": float(lat_grid[r, c]),
            "ibi_grid_distance_km": float(distance_m[i] / 1000.0),
            "ibi_zos_at_atl24_time_m": eta,
        })
    result = pd.DataFrame(records)
    result["atl24_time_utc"] = result["atl24_time_utc"].astype(str)
    result.to_csv(OUT / "atl24_pass_ibi_ssh_matchups.csv", index=False)

    metric_rows = []
    for radius in SUPPORT_RADII_KM:
        subset = result[(result.ibi_grid_distance_km <= radius) & np.isfinite(result.ibi_zos_at_atl24_time_m)].copy()
        # Compare anomalies within repeated native IBI cells to remove the cell-specific constant level.
        repeated = subset.groupby("ibi_nearest_grid_id").size()
        repeated_ids = repeated[repeated >= 2].index
        paired = subset[subset.ibi_nearest_grid_id.isin(repeated_ids)].copy()
        if not paired.empty:
            paired["atl24_surface_anomaly_m"] = paired.atl24_surface_height_egm08_m - paired.groupby("ibi_nearest_grid_id").atl24_surface_height_egm08_m.transform("median")
            paired["ibi_zos_anomaly_m"] = paired.ibi_zos_at_atl24_time_m - paired.groupby("ibi_nearest_grid_id").ibi_zos_at_atl24_time_m.transform("median")
            x = paired.ibi_zos_anomaly_m.to_numpy(float)
            y = paired.atl24_surface_anomaly_m.to_numpy(float)
            metric_rows.append({
                "maximum_grid_distance_km": radius,
                "supported_passes": int(len(subset)),
                "passes_on_repeated_grid_cells": int(len(paired)),
                "repeated_grid_cells": int(paired.ibi_nearest_grid_id.nunique()),
                "centered_Pearson_r": correlation(x, y),
                "centered_Spearman_rho": correlation(x, y, rank=True),
                "centered_RMSE_m": float(np.sqrt(np.mean((y - x) ** 2))) if len(paired) else float("nan"),
                "median_absolute_centered_difference_m": float(np.median(np.abs(y - x))) if len(paired) else float("nan"),
            })
        else:
            metric_rows.append({"maximum_grid_distance_km": radius, "supported_passes": int(len(subset)), "passes_on_repeated_grid_cells": 0, "repeated_grid_cells": 0, "centered_Pearson_r": np.nan, "centered_Spearman_rho": np.nan, "centered_RMSE_m": np.nan, "median_absolute_centered_difference_m": np.nan})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "ibi_ssh_atl24_centered_anomaly_metrics.csv", index=False)
    metadata = {
        "dataset_id": "cmems_mod_ibi_phy-ssh_my_0.027deg_PT1H-m",
        "variable": "zos (sea_surface_height_above_geoid)",
        "time_range_in_download": [temporal_start, temporal_end],
        "atl24_granules": int(len(result)),
        "valid_ibi_grid_cells": int(len(valid_flat)),
        "maximum_distance_thresholds_km": list(SUPPORT_RADII_KM),
        "primary_distance_threshold_km": PRIMARY_RADIUS_KM,
        "method": "Nearest wet IBI node to each ATL24 pass median location; hourly SSH linearly interpolated in time; pass surface and model anomalies centered within repeated native IBI nodes.",
        "caveat": "This tests relative consistency of two gridded/modelled records. It is not a local tide-gauge validation, does not establish an absolute vertical datum tie, and does not resolve surf-zone water levels.",
    }
    (OUT / "ibi_ssh_atl24_audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    notes = [
        "Audit du support IBI SSH–ATL24",
        "",
        "Le produit IBI fournit zos, une hauteur de surface de la mer au-dessus du géoïde à pas horaire. Le script apparie chaque trace ATL24 au nœud IBI marin le plus proche et interpole linéairement le niveau du modèle à l’heure médiane de la trace.",
        "Les anomalies sont centrées séparément pour chaque nœud IBI répété afin de réduire l’effet d’un niveau moyen spatial et d’un décalage vertical constant. Le nombre de nœuds répétés est donc essentiel à l’interprétation.",
        "Les rayons de 2,5, 5 et 10 km sont des seuils de support à comparer; le seuil principal est 5 km. Une grande distance indique un appariement régional et non une mesure locale.",
        "Même si les anomalies concordent, ce contrôle ne remplace pas un marégraphe et ne permet pas de raccorder LAT, EGM08 ou le zéro hydrographique d’Agadir. Le modèle régional n’est pas résolu à l’échelle du déferlement.",
    ]
    (OUT / "interpretation_notes_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("Read-only IBI hourly sea-level/ATL24 support audit complete.")
    print(f"ATL24 granules: {len(result)}; native wet IBI grid cells in subset: {len(valid_flat)}")
    print(metrics.to_string(index=False))
    print(f"Pass-level matchups: {OUT / 'atl24_pass_ibi_ssh_matchups.csv'}")
    print(f"Distance sensitivity: {OUT / 'ibi_ssh_atl24_centered_anomaly_metrics.csv'}")
    print("This is a regional model-consistency check only; no datum tie or gauge validation is claimed.")


if __name__ == "__main__":
    main()
