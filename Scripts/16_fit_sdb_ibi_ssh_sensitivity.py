from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / 'Data/Manifests/sentinel2_atl24_pm6h_selection.csv'
ATL_PATH = ROOT / 'Data/Processed/ATL24/atl24_bathy_points.csv.gz'
SSH_PATH = ROOT / 'Data/Raw/Forcing/IBI_ssh_Anza_hourly.nc'
DEFAULT_OUT = ROOT / 'Results/Bathymetry/ATL24_Sentinel2_IBI_SSH'


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


def reflectance_scale(src: rasterio.io.DatasetReader, arr: np.ndarray) -> float:
    scale = float(src.scales[0]) if src.scales else 1.0
    if 0 < scale < 0.01:
        return scale
    for key in ('scale', 'reflectance_scale', 'scale_factor'):
        try:
            value = float(src.tags(1).get(key, ''))
            if 0 < value < 0.01:
                return value
        except (TypeError, ValueError):
            pass
    return 1e-4 if np.issubdtype(arr.dtype, np.integer) and np.nanmax(arr) > 2 else 1.0


def read_ibi_grid(path: Path):
    ds = xr.open_dataset(path)
    if 'zos' not in ds:
        raise SystemExit(f'IBI file has no zos variable: {path}')
    da = ds['zos'].load()
    tname = next((n for n in ('time', 'time_counter') if n in da.dims), None)
    yname = next((n for n in ('latitude', 'lat') if n in da.dims), None)
    xname = next((n for n in ('longitude', 'lon') if n in da.dims), None)
    if not all((tname, yname, xname)):
        raise SystemExit(f'Unexpected zos dimensions: {da.dims}')
    da = da.transpose(tname, yname, xname)
    times = pd.to_datetime(ds[tname].values, utc=True)
    lons = ds[xname].values
    lats = ds[yname].values
    if lons.ndim == lats.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    elif lons.ndim == lats.ndim == 2:
        lon2d, lat2d = lons, lats
    else:
        raise SystemExit('Unsupported IBI longitude/latitude coordinate arrangement.')
    values = np.asarray(da.values, dtype='float32')
    # A cell is considered wet when the downloaded record is present for >=95% of times.
    wet = np.isfinite(values).mean(axis=0) >= 0.95
    iy, ix = np.where(wet)
    if len(iy) == 0:
        raise SystemExit('No persistent wet IBI cells found in the SSH subset.')
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32629', always_xy=True)
    gx, gy = transformer.transform(lon2d[iy, ix], lat2d[iy, ix])
    grid_xy = np.column_stack((gx, gy))
    grid_rows = np.column_stack((iy, ix))
    return ds, pd.DatetimeIndex(times), values, lon2d, lat2d, grid_xy, grid_rows


def interpolate_at_cells(times: pd.DatetimeIndex, values: np.ndarray, cell_rows: np.ndarray,
                         cell_cols: np.ndarray, query_times: pd.Series) -> np.ndarray:
    # Normalize both axes to nanoseconds: recent pandas versions may otherwise
    # retain microsecond timestamps, making valid query times appear out of range.
    q = pd.DatetimeIndex(pd.to_datetime(query_times, utc=True, errors='coerce')).as_unit('ns')
    out = np.full(len(q), np.nan, dtype='float64')
    q_ns = q.asi8
    t_ns = times.as_unit('ns').asi8
    valid_q = (~q.isna()) & (q_ns >= t_ns[0]) & (q_ns <= t_ns[-1])
    if not valid_q.any():
        return out
    # Interpolate independently at each assigned native wet cell; no spatial interpolation/extrapolation.
    for cell in np.unique(np.column_stack((cell_rows[valid_q], cell_cols[valid_q])), axis=0):
        mask = valid_q & (cell_rows == cell[0]) & (cell_cols == cell[1])
        idx = np.flatnonzero(mask)
        query = q_ns[idx]
        hi = np.searchsorted(t_ns, query, side='left')
        hi = np.clip(hi, 0, len(t_ns) - 1)
        exact = t_ns[hi] == query
        lo = np.maximum(hi - 1, 0)
        if exact.any():
            lo[exact] = hi[exact]
        y0 = values[lo, cell[0], cell[1]].astype('float64')
        y1 = values[hi, cell[0], cell[1]].astype('float64')
        denom = (t_ns[hi] - t_ns[lo]).astype('float64')
        weight = np.divide(query - t_ns[lo], denom, out=np.zeros(len(idx), dtype='float64'), where=denom != 0)
        result = y0 + weight * (y1 - y0)
        result[~(np.isfinite(y0) & np.isfinite(y1))] = np.nan
        # Refuse interpolation across missing hourly records or gaps longer than 2.1 h.
        gap_hours = (t_ns[hi] - t_ns[lo]) / 3.6e12
        result[gap_hours > 2.1] = np.nan
        out[idx] = result
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Test an IBI zos relative sea-level correction for ATL24–Sentinel-2 SDB matchup sensitivity.')
    parser.add_argument('--selection-file', default=str(DEFAULT_SELECTION))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUT))
    parser.add_argument('--primary-distance-km', type=float, default=2.5)
    args = parser.parse_args()
    selection = Path(args.selection_file)
    outdir = Path(args.output_dir)
    if not selection.is_absolute():
        selection = ROOT / selection
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    for path in (selection, ATL_PATH, SSH_PATH):
        if not path.is_file():
            raise SystemExit(f'Required input is missing: {path}')

    selected = pd.read_csv(selection, encoding='utf-8-sig')
    selected = selected.loc[selected['selected'].astype(str).str.upper().eq('YES')].copy()
    if selected.empty:
        raise SystemExit(f'No selected scenes in {selection}')
    atl = pd.read_csv(ATL_PATH)
    if "source_photon_index" not in atl.columns:
        raise SystemExit("Stable ATL24 photon identifiers are missing; rerun Scripts/06_extract_atl24_points.py before this analysis.")
    atl['utc_time_source_epoch'] = pd.to_datetime(atl['utc_time_source_epoch'], utc=True, errors='coerce')
    atl = atl.dropna(subset=['utc_time_source_epoch', 'longitude_deg', 'latitude_deg', 'instantaneous_depth_m'])
    ds, ibi_times, zos, lon2d, lat2d, grid_xy, grid_rows = read_ibi_grid(SSH_PATH)
    wet_cell_tree = cKDTree(grid_xy)
    to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32629', always_xy=True)
    max_gap = pd.Timedelta(hours=24)
    radii = sorted(set([float(args.primary_distance_km), 5.0, 10.0]))
    if args.primary_distance_km <= 0:
        raise SystemExit('--primary-distance-km must be positive.')
    point_rows, scene_rows = [], []

    for si, scene in enumerate(selected.to_dict('records'), 1):
        scene_time = pd.to_datetime(scene['acquisition_utc'], utc=True)
        pts = atl.loc[(atl['utc_time_source_epoch'] - scene_time).abs() <= max_gap].copy()
        n_time = len(pts)
        if pts.empty:
            scene_rows.append({'scene_id': scene['item_id'], 'date': scene['acquisition_date'], 'time_candidates': 0, 'clear_water_candidates': 0})
            print(f'{si}/{len(selected)} {scene["acquisition_date"]}: no ATL24 candidates within 24 h.')
            continue
        date = str(scene['acquisition_date']).replace('-', '')
        prefix = ROOT / 'Data/Processed/Sentinel2' / f'{date}_{safe_name(scene["item_id"])}'
        mask_path = Path(str(prefix) + '_watermask.tif')
        ndwi_path = Path(str(prefix) + '_NDWI.tif')
        ndti_path = Path(str(prefix) + '_NDTI.tif')
        if not all(p.is_file() for p in (mask_path, ndwi_path, ndti_path)):
            raise SystemExit(f'Missing processed S2 rasters for {scene["item_id"]}; run Scripts/07_process_sentinel2.py.')
        raw_prefix = ROOT / 'Data/Raw/Sentinel2' / f'{date}_{safe_name(scene["item_id"])}'
        with rasterio.open(mask_path) as mask_src, rasterio.open(ndwi_path) as ndwi_src, rasterio.open(ndti_path) as ndti_src:
            xy = list(zip(pts['easting_m'].to_numpy(), pts['northing_m'].to_numpy()))
            water = np.array([v[0] for v in mask_src.sample(xy)], dtype='uint8') == 1
            ndwi_values = np.array([v[0] for v in ndwi_src.sample(xy)], dtype='float32')
            ndti_values = np.array([v[0] for v in ndti_src.sample(xy)], dtype='float32')
            rc = [rasterio.transform.rowcol(mask_src.transform, x, y) for x, y in xy]
            sampled = {}
            for name in ('blue', 'green', 'red', 'nir'):
                rp = Path(str(raw_prefix) + f'_{name}.tif')
                if not rp.is_file():
                    raise SystemExit(f'Missing S2 input band: {rp}')
                with rasterio.open(rp) as src:
                    raw = src.read(1)
                    refl = raw.astype('float32') * reflectance_scale(src, raw)
                    if src.transform != mask_src.transform or src.crs != mask_src.crs or refl.shape != (mask_src.height, mask_src.width):
                        aligned = np.full((mask_src.height, mask_src.width), np.nan, dtype='float32')
                        reproject(refl, aligned, src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=mask_src.transform, dst_crs=mask_src.crs,
                                  resampling=Resampling.bilinear)
                        refl = aligned
                    sampled[name] = np.array([refl[r, c] if 0 <= r < refl.shape[0] and 0 <= c < refl.shape[1] else np.nan for r, c in rc], dtype='float32')
        keep = water & np.isfinite(ndwi_values) & np.isfinite(ndti_values)
        for v in sampled.values():
            keep &= np.isfinite(v)
        pts = pts.loc[keep].copy()
        scene_rows.append({'scene_id': scene['item_id'], 'date': scene['acquisition_date'], 'time_candidates': int(n_time), 'clear_water_candidates': int(len(pts))})
        if pts.empty:
            print(f'{si}/{len(selected)} {scene["acquisition_date"]}: zero clear-water ATL24 points.')
            continue
        pts['scene_id'] = scene['item_id']
        pts['scene_utc'] = scene['acquisition_utc']
        pts['acquisition_date'] = scene['acquisition_date']
        pts['time_gap_minutes'] = (pts['utc_time_source_epoch'] - scene_time).abs().dt.total_seconds() / 60
        pts['NDWI'] = ndwi_values[keep]
        pts['NDTI_relative'] = ndti_values[keep]
        for name, arr in sampled.items():
            pts[f'{name}_refl'] = arr[keep]
        px, py = to_utm.transform(pts['longitude_deg'].to_numpy(), pts['latitude_deg'].to_numpy())
        point_xy = np.column_stack((px, py))
        # Nearest persistent native wet IBI cell, sampled once and reused at both times.
        distance_m, nearest = wet_cell_tree.query(point_xy, k=1)
        cell_rc = grid_rows[nearest]
        eta_atl = interpolate_at_cells(ibi_times, zos, cell_rc[:, 0], cell_rc[:, 1], pts['utc_time_source_epoch'])
        eta_s2 = interpolate_at_cells(ibi_times, zos, cell_rc[:, 0], cell_rc[:, 1], pd.Series([scene_time] * len(pts)))
        pts['ibi_grid_distance_km'] = distance_m / 1000.0
        pts['ibi_grid_row'] = cell_rc[:, 0]
        pts['ibi_grid_col'] = cell_rc[:, 1]
        pts['ibi_zos_at_atl24_m'] = eta_atl
        pts['ibi_zos_at_sentinel2_m'] = eta_s2
        pts['ibi_relative_ssh_change_s2_minus_atl24_m'] = eta_s2 - eta_atl
        pts['depth_ibi_corrected_at_s2_m'] = pts['instantaneous_depth_m'].to_numpy(dtype='float64') + pts['ibi_relative_ssh_change_s2_minus_atl24_m'].to_numpy()
        pts['log_blue_green'] = np.log(np.maximum(pts['blue_refl'].to_numpy(), 1e-6) / np.maximum(pts['green_refl'].to_numpy(), 1e-6))
        pts['matchup_window_class'] = np.where(pts['time_gap_minutes'] <= 360.0, 'primary_0_6h', 'extended_6_24h')
        pts['ibi_ssh_supported'] = np.isfinite(eta_atl) & np.isfinite(eta_s2) & np.isfinite(pts['depth_ibi_corrected_at_s2_m'])
        for radius in radii:
            keep_r = pts['ibi_ssh_supported'] & (pts['ibi_grid_distance_km'] <= radius)
            sub = pts.loc[keep_r & (pts['instantaneous_depth_m'] > 0)].copy()
            sub['maximum_grid_distance_km'] = radius
            point_rows.append(sub)
        print(f'{si}/{len(selected)} {scene["acquisition_date"]}: candidates={n_time:,}; clear-water={len(pts):,}; IBI support within {args.primary_distance_km:g} km={(pts["ibi_ssh_supported"] & (pts["ibi_grid_distance_km"] <= args.primary_distance_km)).sum():,}')

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scene_rows).to_csv(outdir / 'ibi_ssh_scene_support.csv', index=False)
    if not point_rows:
        raise SystemExit('No clear-water S2–ATL24 pairs were found.')
    pairs = pd.concat(point_rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    pairs.to_csv(outdir / 's2_atl24_ibi_ssh_training_pairs_by_distance.csv.gz', index=False, compression='gzip')

    features = {
        'Log_blue_green_ratio_OLS': ['log_blue_green'],
        'Multiband_RandomForest': ['blue_refl', 'green_refl', 'red_refl', 'nir_refl', 'NDWI'],
        'Multiband_HistGradientBoosting': ['blue_refl', 'green_refl', 'red_refl', 'nir_refl', 'NDWI'],
    }
    models = {
        'Log_blue_green_ratio_OLS': make_pipeline(StandardScaler(), LinearRegression()),
        'Multiband_RandomForest': RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.8, n_jobs=-1, random_state=42),
        'Multiband_HistGradientBoosting': HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=42),
    }
    metric_rows, pred_rows = [], []
    windows = ('primary_0_6h', 'extended_6_24h', 'combined_0_24h')
    targets = {'raw_instantaneous_depth': 'instantaneous_depth_m', 'IBI_zos_relative_adjustment': 'depth_ibi_corrected_at_s2_m'}
    for radius in radii:
        for window in windows:
            subset = pairs.loc[pairs['maximum_grid_distance_km'].eq(radius)].copy()
            if window != 'combined_0_24h':
                subset = subset.loc[subset['matchup_window_class'].eq(window)].copy()
            groups = subset['granule_id'].astype(str)
            n_groups = int(groups.nunique())
            n_splits = min(5, n_groups) if n_groups >= 3 else 0
            for target_name, target_col in targets.items():
                valid_y = np.isfinite(subset[target_col].to_numpy(dtype='float64')) & (subset[target_col].to_numpy(dtype='float64') > 0)
                frame = subset.loc[valid_y].copy()
                y = frame[target_col].to_numpy(dtype='float64')
                group_ids = frame['granule_id'].astype(str)
                n_groups_y = int(group_ids.nunique())
                folds = min(5, n_groups_y) if n_groups_y >= 3 else 0
                status = 'insufficient_independent_granules_or_pairs'
                if n_groups_y >= 5 and len(frame) >= 50:
                    status = 'grouped_cross_validation_primary'
                elif n_groups_y >= 3 and len(frame) >= 50:
                    status = 'grouped_cross_validation_exploratory_small_group_count'
                for model_name, cols in features.items():
                    result = {'maximum_grid_distance_km': radius, 'window_analysis': window, 'target_scenario': target_name,
                              'model': model_name, 'n_pairs': int(len(frame)), 'independent_atl24_granules': n_groups_y,
                              'group_folds': folds, 'status': status, 'RMSE_m': np.nan, 'MAE_m': np.nan,
                              'bias_m': np.nan, 'R2': np.nan}
                    if status.startswith('grouped_cross_validation'):
                        cv = GroupKFold(n_splits=folds)
                        pred = np.full(len(frame), np.nan)
                        for tr, te in cv.split(frame[cols], y, groups=group_ids):
                            estimator = clone(models[model_name])
                            estimator.fit(frame.iloc[tr][cols], y[tr])
                            pred[te] = estimator.predict(frame.iloc[te][cols])
                        ok = np.isfinite(pred)
                        result.update({'RMSE_m': float(mean_squared_error(y[ok], pred[ok]) ** 0.5),
                                       'MAE_m': float(mean_absolute_error(y[ok], pred[ok])),
                                       'bias_m': float(np.mean(pred[ok] - y[ok])), 'R2': float(r2_score(y[ok], pred[ok]))})
                        p = frame[['scene_id', 'acquisition_date', 'granule_id', 'beam', 'utc_time_source_epoch',
                                   'time_gap_minutes', 'ibi_grid_distance_km', 'ibi_relative_ssh_change_s2_minus_atl24_m',
                                   'instantaneous_depth_m', 'depth_ibi_corrected_at_s2_m']].copy()
                        p['maximum_grid_distance_km'] = radius
                        p['window_analysis'] = window
                        p['target_scenario'] = target_name
                        p['observed_depth_m'] = y
                        p['prediction_depth_m'] = pred
                        p['model'] = model_name
                        pred_rows.append(p)
                    metric_rows.append(result)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(outdir / 'ibi_ssh_sdb_grouped_metrics.csv', index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(outdir / 'ibi_ssh_sdb_oof_predictions.csv.gz', index=False, compression='gzip')
    # Compact support/correction summary for the manuscript audit trail.
    support = pairs.groupby('maximum_grid_distance_km').agg(
        pairs=('granule_id', 'size'), independent_granules=('granule_id', 'nunique'),
        scenes=('scene_id', 'nunique'), median_time_gap_min=('time_gap_minutes', 'median'),
        median_grid_distance_km=('ibi_grid_distance_km', 'median'),
        median_abs_ssh_adjustment_m=('ibi_relative_ssh_change_s2_minus_atl24_m', lambda x: float(np.nanmedian(np.abs(x))))
    ).reset_index()
    support.to_csv(outdir / 'ibi_ssh_sdb_support_summary.csv', index=False)
    metadata = {
        'forcing': 'Copernicus Marine IBI multi-year hourly zos, sea_surface_height_above_geoid; hourly mean fields',
        'pairwise_adjustment': 'depth_S2 = ATL24 instantaneous_depth + zos(scene_time, same nearest native wet cell) - zos(ATL24_time, same cell)',
        'spatial_sampling': 'nearest persistent wet native IBI cell; no spatial interpolation or extrapolation; sensitivity radii reported',
        'time_sampling': 'linear interpolation in time only between adjacent hourly values; no temporal extrapolation; intervals >2.1 h rejected',
        'validation': 'GroupKFold by full ATL24 granule; primary status requires >=5 independent granules and >=50 pairs; 3-4 groups labeled exploratory',
        'matchup_identity': 'Retains source_photon_index inherited from ATL24 index_ph for audit of repeated photon-to-scene pairings.',
        'caveats': 'Regional hourly-mean reanalysis is not a local tide gauge, absolute vertical datum tie, surf-zone or runup model. Centered anomaly consistency is not independent validation. This analysis calibrates only sampled track locations and does not validate a continuous bathymetric map.',
        'input_selection': str(selection), 'ssh_input': str(SSH_PATH), 'selected_scenes': int(len(selected)),
        'maximum_time_gap_hours': 24, 'distance_sensitivity_km': radii,
    }
    (outdir / 'ibi_ssh_sdb_run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('IBI zos relative sea-level SDB sensitivity analysis complete; source imagery, ATL24 photons and NetCDF were read only.')
    print(f'Selected scenes: {len(selected)}; clear-water pairs retained across distance scenarios: {len(pairs):,}')
    print(support.to_string(index=False))
    print(f'Scene audit, pairs, grouped metrics, OOF predictions and metadata: {outdir}')
    print('Interpret the IBI correction as a regional-model sensitivity; use grouped metrics only where independent ATL24 granule support is adequate. No absolute datum tie, local gauge validation, SDB map, or unsampled-pixel validation is produced.')
    ds.close()


if __name__ == '__main__':
    main()


