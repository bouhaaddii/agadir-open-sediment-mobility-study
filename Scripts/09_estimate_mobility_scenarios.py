from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]


def coord_name(ds: xr.Dataset, choices: tuple[str, ...]) -> str:
    for name in choices:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"None of the expected coordinates exist: {choices}")


def grid_forcing(ds: xr.Dataset, name: str, time_name: str, lat_name: str, lon_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    da = ds[name]
    if lat_name not in da.dims or lon_name not in da.dims or time_name not in da.dims:
        raise ValueError(f"{name} must contain time, latitude and longitude dimensions; got {da.dims}")
    arr = da.transpose(time_name, lat_name, lon_name).values.astype("float64")
    lat = np.asarray(ds[lat_name].values, dtype="float64")
    lon = np.asarray(ds[lon_name].values, dtype="float64")
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Curvilinear forcing grids need an explicit remapping step; this script accepts rectilinear grids only.")
    return arr, lat, lon


def solve_wavenumber(period_s: float, depth_m: float, gravity: float) -> float:
    omega = 2.0 * math.pi / period_s
    k = max(omega * omega / gravity, omega / math.sqrt(gravity * depth_m))
    for _ in range(40):
        kh = k * depth_m
        tanh = math.tanh(kh)
        f = gravity * k * tanh - omega * omega
        df = gravity * (tanh + kh / (math.cosh(min(kh, 50.0)) ** 2))
        nxt = k - f / max(df, 1e-12)
        if not np.isfinite(nxt) or nxt <= 0:
            break
        if abs(nxt - k) < 1e-8:
            k = nxt
            break
        k = nxt
    return k


def wave_stress(hs: np.ndarray, tp: np.ndarray, depth: float, d50_m: float, rho: float, nu: float, g: float) -> np.ndarray:
    result = np.full(hs.shape, np.nan, dtype="float64")
    ks = 2.5 * d50_m
    valid = np.isfinite(hs) & np.isfinite(tp) & (hs > 0.05) & (tp > 1.0) & (depth > 0.5)
    if not valid.any():
        return result
    h = np.clip(hs[valid], 0.05, 15.0)
    t = np.clip(tp[valid], 1.0, 30.0)
    k = np.array([solve_wavenumber(float(tt), float(depth), g) for tt in t])
    ub = math.pi * h / (t * np.sinh(np.clip(k * depth, 1e-6, 50.0)))
    excursion = ub * t / (2.0 * math.pi)
    ratio = np.maximum(excursion / ks, 1e-6)
    fw = np.where(ratio > 1.57, np.exp(5.213 * ratio ** -0.194 - 5.977), 0.30)
    result[valid] = 0.5 * rho * fw * ub * ub
    return result


def theta_critical(d50_m: float, rho_s: float, rho_w: float, nu: float, g: float) -> float:
    s = rho_s / rho_w
    dstar = d50_m * (((s - 1.0) * g) / (nu * nu)) ** (1.0 / 3.0)
    return 0.30 / (1.0 + 1.2 * dstar) + 0.055 * (1.0 - math.exp(-0.020 * dstar))


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    bathy_path = ROOT / "Data/Raw/Bathymetry/emodnet_mean_2022_anza.tif"
    wave_path = ROOT / "Data/Raw/Forcing/IBI_waves_Anza_hourly.nc"
    current_path = ROOT / "Data/Raw/Forcing/IBI_currents_Anza_daily_0_30m.nc"
    for path in (bathy_path, wave_path, current_path):
        if not path.exists():
            raise SystemExit(f"Required file missing: {path}. Run the preceding download stages.")
    with xr.open_dataset(wave_path) as wave, xr.open_dataset(current_path) as cur, rasterio.open(bathy_path) as src:
        wtime = coord_name(wave, ("time", "valid_time"))
        wlat = coord_name(wave, ("latitude", "lat"))
        wlon = coord_name(wave, ("longitude", "lon"))
        ctime = coord_name(cur, ("time", "valid_time"))
        clat = coord_name(cur, ("latitude", "lat"))
        clon = coord_name(cur, ("longitude", "lon"))
        depth_name = coord_name(cur, ("depth", "deptht", "lev"))
        hs, lat, lon = grid_forcing(wave, "VHM0", wtime, wlat, wlon)
        tp, _, _ = grid_forcing(wave, "VTPK", wtime, wlat, wlon)
        wave_from, _, _ = grid_forcing(wave, "VMDR", wtime, wlat, wlon)
        times = pd.to_datetime(wave[wtime].values, utc=True)
        cu = cur["uo"].transpose(ctime, depth_name, clat, clon).values.astype("float64")
        cv = cur["vo"].transpose(ctime, depth_name, clat, clon).values.astype("float64")
        ctimes = pd.to_datetime(cur[ctime].values, utc=True)
        cdepth = np.asarray(cur[depth_name].values, dtype="float64")
        clatv = np.asarray(cur[clat].values, dtype="float64")
        clonv = np.asarray(cur[clon].values, dtype="float64")
        if any(v.ndim != 1 for v in (lat, lon, clatv, clonv)):
            raise SystemExit("Only rectilinear IBI grids are supported; source files are unchanged.")
        lon2, lat2 = np.meshgrid(lon, lat)
        trans = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        bx, by = trans.transform(lon2.ravel(), lat2.ravel())
        z = np.array([v[0] for v in src.sample(zip(bx, by))], dtype="float64").reshape(lat2.shape)
        # EMODnet elevations are normally negative below sea level. The explicit sign is an auditable config choice.
        sign = cfg.get("emodnet_seabed_sign", "negative_elevation")
        bathy_depth = -z if sign == "negative_elevation" else z
        bathy_depth = np.where(np.isfinite(bathy_depth) & (bathy_depth > 0.5) & (bathy_depth < 6000), bathy_depth, np.nan)
        rho_w = float(cfg["water_density_kg_m3"]); rho_s = float(cfg["sediment_density_kg_m3"])
        nu = float(cfg["kinematic_viscosity_m2_s"]); g = float(cfg["gravity_m_s2"])
        cd = float(cfg["current_drag_coefficient_scenario"])
        d50s = [float(v) / 1000.0 for v in cfg["d50_scenarios_mm"]]
        dzs = [float(v) for v in cfg["water_depth_offset_scenarios_m"]]
        hscales = [float(v) for v in cfg["wave_height_scale_scenarios"]]
        cscales = [float(v) for v in cfg["current_speed_scale_scenarios"]]
        locations = []
        depth_dim_ix = {name: i for i, name in enumerate(cur["uo"].transpose(ctime, depth_name, clat, clon).dims)}[depth_name]
        # Align daily current records to hourly wave records by nearest date/hour (18-hour tolerance).
        ti = np.array([int(np.argmin(np.abs((ctimes - t).total_seconds()))) for t in times])
        gap_hours = np.array([abs((ctimes[i] - t).total_seconds()) / 3600.0 for i, t in zip(ti, times)])
        ti[gap_hours > 18] = -1
        for iy, ylat in enumerate(lat):
            for ix, xlon in enumerate(lon):
                base_depth = bathy_depth[iy, ix]
                if not np.isfinite(base_depth):
                    continue
                cy = int(np.argmin(np.abs(clatv - ylat)))
                cx = int(np.argmin(np.abs(clonv - xlon)))
                target_bottom_level = max(0.0, base_depth - 0.5)
                supported_levels = np.flatnonzero(cdepth <= target_bottom_level)
                if len(supported_levels) == 0:
                    continue
                iz = int(supported_levels[-1])
                # Save a compact forcing time series only where a bottom current layer is available.
                cur_u = np.full(len(times), np.nan); cur_v = np.full(len(times), np.nan)
                for it, ci in enumerate(ti):
                    if ci < 0:
                        continue
                    cur_u[it] = cu[ci, iz, cy, cx]
                    cur_v[it] = cv[ci, iz, cy, cx]
                speed = np.hypot(cur_u, cur_v)
                valid_current = np.isfinite(speed)
                if not valid_current.any():
                    continue
                # IBI VMDR is the direction waves come FROM; convert to propagation
                # heading. Current heading is clockwise from north, toward the flow.
                current_heading_to = np.degrees(np.arctan2(cur_u, cur_v)) % 360.0
                wave_heading_to = (wave_from[:, iy, ix] + 180.0) % 360.0
                relative_angle = (wave_heading_to - current_heading_to + 180.0) % 360.0 - 180.0
                phi = np.deg2rad(relative_angle)
                supported = (valid_current & np.isfinite(hs[:, iy, ix]) & np.isfinite(tp[:, iy, ix])
                             & (hs[:, iy, ix] > 0.05) & (tp[:, iy, ix] > 1.0))
                combined_supported = supported & np.isfinite(wave_from[:, iy, ix])
                for d50 in d50s:
                    tau_cr = theta_critical(d50, rho_s, rho_w, nu, g) * (rho_s - rho_w) * g * d50
                    lo_count = np.zeros(len(times), dtype="float64")
                    hi_count = np.zeros(len(times), dtype="float64")
                    combined_count = np.zeros(len(times), dtype="float64")
                    lo_scenario_frequencies = []
                    hi_scenario_frequencies = []
                    combined_scenario_frequencies = []
                    scenario_count = 0
                    for dz in dzs:
                        d_eff = base_depth + dz
                        if d_eff <= 0.5:
                            continue
                        tau_w_base = wave_stress(hs[:, iy, ix], tp[:, iy, ix], d_eff, d50, rho_w, nu, g)
                        for hs_scale in hscales:
                            tau_w = tau_w_base * hs_scale * hs_scale
                            for c_scale in cscales:
                                tau_c = rho_w * cd * (speed * c_scale) ** 2
                                tau_lo = np.fmax(tau_c, tau_w)
                                tau_hi = tau_c + tau_w
                                interaction_ratio = np.divide(
                                    tau_w, tau_c + tau_w,
                                    out=np.zeros_like(tau_w), where=(tau_c + tau_w) > 0,
                                )
                                tau_mean_combined = tau_c * (1.0 + 1.2 * interaction_ratio ** 3.2)
                                tau_combined_max = np.sqrt(
                                    (tau_mean_combined + tau_w * np.cos(phi)) ** 2
                                    + (tau_w * np.sin(phi)) ** 2
                                )
                                lo_count += (tau_lo >= tau_cr) & np.isfinite(tau_lo)
                                hi_count += (tau_hi >= tau_cr) & np.isfinite(tau_hi)
                                combined_count += (tau_combined_max >= tau_cr) & np.isfinite(tau_combined_max)
                                lo_scenario_frequencies.append(float(np.sum((tau_lo >= tau_cr) & supported) / max(int(supported.sum()), 1)))
                                hi_scenario_frequencies.append(float(np.sum((tau_hi >= tau_cr) & supported) / max(int(supported.sum()), 1)))
                                combined_scenario_frequencies.append(
                                    float(np.sum((tau_combined_max >= tau_cr) & combined_supported) / combined_supported.sum())
                                    if combined_supported.any() else np.nan
                                )
                                scenario_count += 1
                    if scenario_count == 0 or not supported.any():
                        continue
                    lo_freq = np.divide(lo_count, scenario_count, out=np.full_like(lo_count, np.nan), where=supported)
                    hi_freq = np.divide(hi_count, scenario_count, out=np.full_like(hi_count, np.nan), where=supported)
                    combined_supported = supported & np.isfinite(wave_from[:, iy, ix])
                    combined_freq = np.divide(
                        combined_count, scenario_count, out=np.full_like(combined_count, np.nan), where=combined_supported,
                    )
                    combined_scenario_frequencies = np.asarray(combined_scenario_frequencies, dtype="float64")
                    combined_scenario_frequencies = combined_scenario_frequencies[
                        np.isfinite(combined_scenario_frequencies)
                    ]
                    locations.append({
                        "longitude_deg": float(xlon), "latitude_deg": float(ylat),
                        "emodnet_depth_m_sign_assumed": float(base_depth),
                        "d50_scenario_mm": d50 * 1000.0,
                        "supported_hourly_records": int(supported.sum()),
                        "possible_hourly_records": int(len(times)),
                        "separate_stress_scenario_frequency_mean": float(np.nanmean(lo_freq[supported])),
                        "aligned_envelope_scenario_frequency_mean": float(np.nanmean(hi_freq[supported])),
                        "soulsby_combined_max_scenario_frequency_mean": float(np.nanmean(combined_freq[combined_supported])) if combined_supported.any() else np.nan,
                        "soulsby_combined_max_scenario_frequency_p10": float(np.nanpercentile(combined_scenario_frequencies, 10)) if len(combined_scenario_frequencies) else np.nan,
                        "soulsby_combined_max_scenario_frequency_median": float(np.nanmedian(combined_scenario_frequencies)) if len(combined_scenario_frequencies) else np.nan,
                        "soulsby_combined_max_scenario_frequency_p90": float(np.nanpercentile(combined_scenario_frequencies, 90)) if len(combined_scenario_frequencies) else np.nan,
                        "separate_stress_scenario_frequency_p10": float(np.nanpercentile(lo_scenario_frequencies, 10)),
                        "separate_stress_scenario_frequency_median": float(np.nanmedian(lo_scenario_frequencies)),
                        "separate_stress_scenario_frequency_p90": float(np.nanpercentile(lo_scenario_frequencies, 90)),
                        "aligned_envelope_scenario_frequency_p10": float(np.nanpercentile(hi_scenario_frequencies, 10)),
                        "aligned_envelope_scenario_frequency_median": float(np.nanmedian(hi_scenario_frequencies)),
                        "aligned_envelope_scenario_frequency_p90": float(np.nanpercentile(hi_scenario_frequencies, 90)),
                        "separate_stress_hours_any_scenario": int(np.sum(lo_count[supported] > 0)),
                        "aligned_envelope_hours_any_scenario": int(np.sum(hi_count[supported] > 0)),
                        "nearest_current_node_distance_km": float(6371.0 * 2.0 * math.asin(math.sqrt(
                            math.sin(math.radians(float(clatv[cy] - ylat)) / 2.0) ** 2
                            + math.cos(math.radians(float(ylat))) * math.cos(math.radians(float(clatv[cy])))
                            * math.sin(math.radians(float(clonv[cx] - xlon)) / 2.0) ** 2))),
                        "selected_current_level_m_positive_down": float(cdepth[iz]),
                        "critical_shear_stress_Pa": float(tau_cr),
                        "scenario_count": scenario_count,
                        "wave_direction_supported_hourly_records": int(combined_supported.sum()),
                    })
                print(f"Completed forcing node {iy+1}/{len(lat)}, {ix+1}/{len(lon)}")
    if not locations:
        raise SystemExit("No IBI wave/current points overlapped valid bathymetry and bottom-current levels. Check depth/sign/forcing coverage.")
    outdir = ROOT / "Results/Sediment_Mobility"
    outdir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(locations)
    output_table = outdir / "mobility_scenario_frequency_by_grid_d50_directional.csv"
    table.to_csv(output_table, index=False)
    meta = {
        "method": "scenario-envelope screen using linear wave theory, Swart wave friction, quadratic current drag, and Soulsby-Whitehouse critical Shields number",
        "current_data": "regional daily IBI; nearest wet vertical level not deeper than the nominal bottom",
        "wave_data": "hourly regional IBI; no surf-zone transformation",
        "bathymetry": str(bathy_path),
        "bathymetry_sign_assumption": sign,
        "scenario_count_per_grain_class_maximum": int(len(dzs) * len(hscales) * len(cscales)),
        "scenario_support_note": "Some grid cells omit depth-offset combinations that produce effective water depth <= 0.5 m; actual scenario_count is stored per row.",
        "separate_stress_screen": "max(current stress, wave stress); non-additive screening estimate, not a formal bound on full wave-current stress",
        "aligned_stress_envelope": "sum of current and wave stresses; aligned-forcing envelope, not a validated combined-stress solution",
        "directional_combined_stress": "Soulsby DATA2 mean stress tau_m=tau_c[1+1.2(tau_w/(tau_c+tau_w))^3.2], then tau_max=sqrt[(tau_m+tau_w cos(phi))^2+(tau_w sin(phi))^2]; wave-current relative direction is computed from VMDR (wave-from converted to propagation) and IBI eastward/northward currents",
        "directional_combined_stress_reference": "Soulsby, R.L. (1997), Dynamics of Marine Sands; Soulsby and Clarke (2005), Bed Shear-stresses Under Combined Waves and Currents on Smooth and Rough Beds, TR137",
        "scenario_frequency_quantiles": "p10/median/p90 summarize the 27 prescribed parameter combinations; they are scenario sensitivity summaries, not confidence intervals or probabilities",
        "interpretation": "scenario frequency is not a probability distribution, observed sediment flux, or measured grain-size-specific mobility",
        "water_depth_offsets_m": dzs,
        "wave_height_multipliers": hscales,
        "current_speed_multipliers": cscales,
    }
    output_metadata = outdir / "mobility_scenario_method_metadata_directional.json"
    output_metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Directional scenario table: {output_table}")
    print(f"Method metadata: {output_metadata}")
    print("Scenario frequencies are conditional screening results. The forcing does not resolve the surf zone and the grain-size/datum values are scenarios, not local measurements.")


if __name__ == "__main__":
    main()
