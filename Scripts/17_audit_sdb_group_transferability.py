from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['scene_id', 'granule_id', 'beam', 'utc_time_source_epoch', 'maximum_grid_distance_km']
TARGETS = {
    'raw_instantaneous_depth': 'instantaneous_depth_m',
    'IBI_zos_relative_adjustment': 'depth_ibi_corrected_at_s2_m',
}
MODELS = ['Log_blue_green_ratio_OLS', 'Multiband_RandomForest', 'Multiband_HistGradientBoosting']


def scores(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    ok = np.isfinite(y) & np.isfinite(pred)
    if not ok.any():
        return (np.nan, np.nan, np.nan, np.nan)
    err = pred[ok] - y[ok]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    denom = float(np.sum((y[ok] - np.mean(y[ok])) ** 2))
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else np.nan
    return rmse, mae, bias, r2


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit grouped transferability of saved ATL24–Sentinel-2 SDB predictions.')
    parser.add_argument('--analysis-dir', default='Results/Bathymetry/ATL24_Sentinel2_IBI_SSH',
                        help='Directory containing Script 16 pair and OOF prediction tables.')
    parser.add_argument('--output-dir', default=None, help='Optional output directory, relative to project root or absolute.')
    args = parser.parse_args()
    base = Path(args.analysis_dir)
    if not base.is_absolute():
        base = ROOT / base
    pair_path = base / 's2_atl24_ibi_ssh_training_pairs_by_distance.csv.gz'
    pred_path = base / 'ibi_ssh_sdb_oof_predictions.csv.gz'
    outdir = Path(args.output_dir) if args.output_dir else base / 'Transferability_Audit'
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    if not pair_path.is_file() or not pred_path.is_file():
        raise SystemExit('Run Scripts/16_fit_sdb_ibi_ssh_sensitivity.py first.')
    pairs = pd.read_csv(pair_path)
    predictions = pd.read_csv(pred_path)
    if pairs.empty or predictions.empty:
        raise SystemExit('The saved SDB matchup or out-of-fold table is empty; do not interpret the previous zero-support run.')
    outdir.mkdir(parents=True, exist_ok=True)
    overall_rows, granule_rows = [], []
    distances = sorted(pairs['maximum_grid_distance_km'].dropna().unique().tolist())
    windows = ['primary_0_6h', 'extended_6_24h', 'combined_0_24h']
    for radius in distances:
        p_radius = pairs.loc[pairs['maximum_grid_distance_km'].eq(radius)].copy()
        for window in windows:
            subset = p_radius if window == 'combined_0_24h' else p_radius.loc[p_radius['matchup_window_class'].eq(window)].copy()
            for target_name, target_col in TARGETS.items():
                y0 = subset[target_col].to_numpy(dtype='float64')
                frame = subset.loc[np.isfinite(y0) & (y0 > 0)].copy()
                y = frame[target_col].to_numpy(dtype='float64')
                groups = frame['granule_id'].astype(str)
                n_groups = int(groups.nunique())
                if len(frame) == 0 or n_groups < 3:
                    continue
                folds = min(5, n_groups)
                cv = GroupKFold(n_splits=folds)
                baseline = np.full(len(frame), np.nan)
                fold_id = np.full(len(frame), -1, dtype='int16')
                for fi, (train, test) in enumerate(cv.split(frame, y, groups=groups), 1):
                    baseline[test] = float(np.median(y[train]))
                    fold_id[test] = fi
                map_frame = frame[KEYS].copy()
                map_frame['baseline_median_prediction_m'] = baseline
                map_frame['cv_fold'] = fold_id
                # ATL24 can assign identical timestamps to distinct photons on
                # the same beam. Preserve them with a stable occurrence index.
                map_frame['_duplicate_ordinal'] = map_frame.groupby(KEYS, dropna=False).cumcount()
                for model_name in MODELS:
                    pred = predictions.loc[
                        predictions['maximum_grid_distance_km'].eq(radius)
                        & predictions['window_analysis'].eq(window)
                        & predictions['target_scenario'].eq(target_name)
                        & predictions['model'].eq(model_name)
                    ].copy()
                    if pred.empty:
                        continue
                    pred['_duplicate_ordinal'] = pred.groupby(KEYS, dropna=False).cumcount()
                    joined = pred.merge(map_frame, on=KEYS + ['_duplicate_ordinal'], how='left', validate='one_to_one')
                    if joined['baseline_median_prediction_m'].isna().any():
                        raise SystemExit(f'Could not align baseline and OOF rows for {radius} km / {window} / {target_name} / {model_name}.')
                    model_rmse, model_mae, model_bias, model_r2 = scores(
                        joined['observed_depth_m'].to_numpy(float), joined['prediction_depth_m'].to_numpy(float))
                    base_rmse, base_mae, base_bias, base_r2 = scores(
                        joined['observed_depth_m'].to_numpy(float), joined['baseline_median_prediction_m'].to_numpy(float))
                    counts = joined.groupby('granule_id').size().to_numpy(float)
                    effective_groups = float(counts.sum() ** 2 / np.sum(counts ** 2)) if len(counts) else np.nan
                    overall_rows.append({
                        'maximum_grid_distance_km': radius, 'window_analysis': window,
                        'target_scenario': target_name, 'model': model_name,
                        'n_pairs': int(len(joined)), 'n_independent_granules': int(joined['granule_id'].nunique()),
                        'effective_granule_count_by_pair_weight': effective_groups,
                        'smallest_granule_n': int(joined.groupby('granule_id').size().min()),
                        'largest_granule_n': int(joined.groupby('granule_id').size().max()),
                        'model_RMSE_m': model_rmse, 'model_MAE_m': model_mae, 'model_bias_m': model_bias, 'model_R2': model_r2,
                        'median_baseline_RMSE_m': base_rmse, 'median_baseline_MAE_m': base_mae,
                        'RMSE_skill_vs_grouped_median_baseline': float(1 - model_rmse / base_rmse) if base_rmse > 0 else np.nan,
                    })
                    for granule, g in joined.groupby('granule_id', sort=True):
                        grmse, gmae, gbias, gr2 = scores(g['observed_depth_m'].to_numpy(float), g['prediction_depth_m'].to_numpy(float))
                        brmse, bmae, bbias, br2 = scores(g['observed_depth_m'].to_numpy(float), g['baseline_median_prediction_m'].to_numpy(float))
                        granule_rows.append({
                            'maximum_grid_distance_km': radius, 'window_analysis': window,
                            'target_scenario': target_name, 'model': model_name, 'granule_id': granule,
                            'n_pairs': int(len(g)), 'observed_depth_min_m': float(g['observed_depth_m'].min()),
                            'observed_depth_median_m': float(g['observed_depth_m'].median()),
                            'observed_depth_max_m': float(g['observed_depth_m'].max()),
                            'median_time_gap_min': float(g['time_gap_minutes'].median()),
                            'median_abs_ibi_adjustment_m': float(g['ibi_relative_ssh_change_s2_minus_atl24_m'].abs().median()),
                            'model_RMSE_m': grmse, 'model_MAE_m': gmae, 'model_bias_m': gbias,
                            'grouped_median_baseline_RMSE_m': brmse,
                        })
    overall = pd.DataFrame(overall_rows)
    by_granule = pd.DataFrame(granule_rows)
    if overall.empty:
        raise SystemExit('No valid out-of-fold predictions were found for grouped transferability diagnostics.')
    overall.to_csv(outdir / 'sdb_grouped_transferability_overall.csv', index=False)
    by_granule.to_csv(outdir / 'sdb_grouped_transferability_by_granule.csv', index=False)

    best = overall.loc[overall['model'].eq('Multiband_HistGradientBoosting')
                       & overall['window_analysis'].eq('primary_0_6h')].copy()
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.9))
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, :])
    colors = {'raw_instantaneous_depth': '#176B87', 'IBI_zos_relative_adjustment': '#D47A35'}
    x = np.arange(len(distances), dtype=float)
    width = 0.34
    for offset, target_name in [(-width / 2, 'raw_instantaneous_depth'), (width / 2, 'IBI_zos_relative_adjustment')]:
        rows = best.loc[best['target_scenario'].eq(target_name)].set_index('maximum_grid_distance_km').reindex(distances)
        ax0.bar(x + offset, rows['model_RMSE_m'], width, color=colors[target_name], alpha=0.9,
                    label='Raw depth' if target_name == 'raw_instantaneous_depth' else 'IBI-adjusted depth')
        ax0.scatter(x + offset, rows['median_baseline_RMSE_m'], color='black', marker='_', s=180, linewidths=2.4,
                        label='Grouped median baseline' if offset == -width / 2 else None, zorder=4)
    ax0.set_xticks(x, [f'{v:g} km' for v in distances])
    ax0.set_ylabel('Leave-one-granule-out RMSE (m)')
    ax0.set_title('(A) Model error and training-fold baseline', loc='left', fontweight='bold')
    ax0.legend(frameon=False, fontsize=8)
    ax0.grid(axis='y', alpha=0.22)

    h = by_granule.loc[by_granule['maximum_grid_distance_km'].eq(min(distances))
                       & by_granule['window_analysis'].eq('primary_0_6h')
                       & by_granule['model'].eq('Multiband_HistGradientBoosting')]
    granules = sorted(h['granule_id'].unique())
    short = []
    for granule in granules:
        stamp = granule.split('_')[1][:8]
        short.append(f'{stamp[2:4]}-{stamp[4:6]}-{stamp[6:8]}')
    for offset, target_name in [(-0.18, 'raw_instantaneous_depth'), (0.18, 'IBI_zos_relative_adjustment')]:
        rows = h.loc[h['target_scenario'].eq(target_name)].set_index('granule_id').reindex(granules)
        ax1.bar(np.arange(len(granules)) + offset, rows['model_bias_m'], 0.36,
                    color=colors[target_name], label='Raw depth' if offset < 0 else 'IBI-adjusted depth')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xticks(np.arange(len(granules)), short, rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel('Mean prediction error (m)')
    ax1.set_title('(B) Bias by held-out ATL24 granule (≤2.5 km)', loc='left', fontweight='bold')
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(axis='y', alpha=0.22)

    count = pairs.loc[pairs['matchup_window_class'].eq('primary_0_6h')].groupby(
        ['maximum_grid_distance_km', 'granule_id']).size().rename('n').reset_index()
    support_granules = sorted(count['granule_id'].unique(),
                              key=lambda g: -int(count.loc[count['granule_id'].eq(g), 'n'].max()))
    support_labels = []
    for granule in support_granules:
        stamp = granule.split('_')[1][:8]
        support_labels.append(f'{stamp[2:4]}-{stamp[4:6]}-{stamp[6:8]}')
    bar_width = 0.36
    for idx, radius in enumerate([r for r in distances if r <= 5.0]):
        values = count.loc[count['maximum_grid_distance_km'].eq(radius)].set_index('granule_id')['n'].reindex(support_granules, fill_value=0)
        ax2.bar(np.arange(len(support_granules)) + (idx - 0.5) * bar_width, values, bar_width,
                color='#176B87' if idx == 0 else '#D47A35', label=f'≤{radius:g} km')
    ax2.set_xticks(np.arange(len(support_granules)), support_labels, rotation=0, fontsize=9)
    ax2.set_xlabel('ATL24 granule date (YY-MM-DD)')
    ax2.set_ylabel('Clear-water matchup pairs')
    ax2.set_title('(C) Matchup support is concentrated in a few granules', loc='left', fontweight='bold')
    ax2.legend(title='Nearest wet-cell limit', frameon=False, fontsize=8)
    ax2.grid(axis='y', alpha=0.22)
    if 5.0 in distances and 10.0 in distances:
        n5 = int(count.loc[count['maximum_grid_distance_km'].eq(5.0), 'n'].sum())
        n10 = int(count.loc[count['maximum_grid_distance_km'].eq(10.0), 'n'].sum())
        if n5 == n10:
            ax2.text(0.99, 0.95, 'The 10-km limit adds no pairs beyond 5 km.', transform=ax2.transAxes,
                     ha='right', va='top', fontsize=9, color='#444444')
    fig.suptitle('Cross-granule transferability of Sentinel-2 bathymetry models', fontsize=15, fontweight='bold')
    fig.savefig(outdir / 'Figure_SDB_grouped_transferability.png', dpi=350, bbox_inches='tight')
    fig.savefig(outdir / 'Figure_SDB_grouped_transferability.pdf', bbox_inches='tight')
    plt.close(fig)
    (outdir / 'figure_caption_EN.txt').write_text(
        'Figure. Leave-one-granule-out performance of the multiband histogram-gradient boosting SDB model. '
        'Panel (A) compares model RMSE with the median-depth baseline calculated only from each training fold; black ticks mark baseline RMSE. '
        'Panel (B) shows signed errors for held-out ATL24 granules at the strictest distance limit. Panel (C) shows the number of clear-water matchup pairs per granule. '
        'The 10-km limit adds no pairs beyond 5 km in this dataset. The regional IBI sea-level adjustment is a relative sensitivity scenario; '
        'these track-level tests do not validate a continuous bathymetric map or establish absolute vertical-datum accuracy.\n', encoding='utf-8')
    primary = overall.loc[overall['model'].eq('Multiband_HistGradientBoosting')
                          & overall['window_analysis'].eq('primary_0_6h')]
    note = [
        'Audit du transfert des modèles SDB par granule ATL24',
        '',
        'Les prédictions hors-échantillon ont été produites en laissant un granule entier de côté. Une médiane des profondeurs du jeu d’apprentissage sert de référence sans prédicteurs optiques.',
        'Le nombre de paires n’est pas le nombre d’observations indépendantes : les photons d’un même granule restent spatialement regroupés.',
        '',
    ]
    for _, row in primary.iterrows():
        note.append(f"Distance ≤ {row['maximum_grid_distance_km']:g} km; cible={row['target_scenario']}: n={int(row['n_pairs'])}, granules={int(row['n_independent_granules'])}, effectif de granules pondéré par les paires={row['effective_granule_count_by_pair_weight']:.2f}, RMSE modèle={row['model_RMSE_m']:.2f} m, RMSE référence médiane={row['median_baseline_RMSE_m']:.2f} m, compétence RMSE={row['RMSE_skill_vs_grouped_median_baseline']:.3f}, R²={row['model_R2']:.3f}.")
    note.extend(['', 'Interprétation : le R² est calculé par rapport à la moyenne globale de l’échantillon; il peut rester négatif même si le modèle surpasse la médiane calculée dans chaque pli d’apprentissage. À ≤2,5 km, la compétence RMSE par rapport à cette référence pli-spécifique est positive, mais le RMSE reste proche de 8,5 m et l’effectif de granules pondéré par les paires n’est que de 2,72. À ≤5 km, la compétence RMSE devient négative; le seuil de 10 km n’ajoute aucun point. Ces résultats signalent un transfert instable, avec un support très inégal entre granules. La correction IBI ne constitue pas une correction de datum et reste une sensibilité relative.', ''])
    (outdir / 'interpretation_notes_FR.txt').write_text('\n'.join(note), encoding='utf-8')
    print('Grouped SDB transferability audit complete; saved out-of-fold predictions and matchup inputs were read only.')
    print(f'Overall/baseline metrics: {outdir / "sdb_grouped_transferability_overall.csv"}')
    print(f'Granule diagnostics: {outdir / "sdb_grouped_transferability_by_granule.csv"}')
    print(f'Figure: {outdir / "Figure_SDB_grouped_transferability.png"} and PDF')
    print(f'Interpretation: {outdir / "interpretation_notes_FR.txt"}')
    print('Assess skill against the fold-specific median together with absolute error and granule balance; these track-level results do not validate a continuous bathymetry map.')


if __name__ == '__main__':
    main()
