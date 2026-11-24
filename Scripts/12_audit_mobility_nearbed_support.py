from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'Results/Sediment_Mobility/mobility_scenario_frequency_by_grid_d50_directional.csv'
OUT = ROOT / 'Results/Sediment_Mobility/Forcing_Support_Audit'
TOLERANCES_M = (2.0, 5.0, 10.0)


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f'Missing directional mobility table: {INPUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    required = {'longitude_deg','latitude_deg','emodnet_depth_m_sign_assumed',
                'selected_current_level_m_positive_down','nearest_current_node_distance_km',
                'd50_scenario_mm','soulsby_combined_max_scenario_frequency_mean'}
    missing = required.difference(data.columns)
    if missing:
        raise SystemExit(f'Missing required columns: {sorted(missing)}')
    nodes = data.sort_values('d50_scenario_mm').drop_duplicates(['longitude_deg','latitude_deg']).copy()
    nodes['vertical_gap_bottom_minus_current_m'] = (
        nodes['emodnet_depth_m_sign_assumed'] - nodes['selected_current_level_m_positive_down']
    )
    nodes['current_level_bottom_support'] = np.select(
        [nodes.vertical_gap_bottom_minus_current_m <= 2,
         nodes.vertical_gap_bottom_minus_current_m <= 5,
         nodes.vertical_gap_bottom_minus_current_m <= 10],
        ['within_2m','2_to_5m','5_to_10m'], default='more_than_10m')
    nodes['bottom_depth_current_support_status'] = np.where(
        nodes.vertical_gap_bottom_minus_current_m <= 2, 'strict_support_le_2m',
        np.where(nodes.vertical_gap_bottom_minus_current_m <= 5, 'sensitivity_support_2_to_5m', 'unsupported_for_bed_stress_gt_5m'))
    node_csv = OUT / 'mobility_current_vertical_support_by_node.csv'
    nodes.to_csv(node_csv, index=False)

    records = []
    for tol in TOLERANCES_M:
        kept = nodes.loc[nodes.vertical_gap_bottom_minus_current_m <= tol]
        for d50, group in data.loc[data.set_index(['longitude_deg','latitude_deg']).index.isin(
                kept.set_index(['longitude_deg','latitude_deg']).index)].groupby('d50_scenario_mm'):
            vals = group['soulsby_combined_max_scenario_frequency_mean'].dropna().to_numpy(float)
            records.append({
                'maximum_current_level_offset_above_bed_m': tol,
                'd50_scenario_mm': float(d50),
                'supported_nodes': int(kept.shape[0]),
                'node_median_frequency_median': float(np.median(vals)) if vals.size else np.nan,
                'node_median_frequency_mean': float(np.mean(vals)) if vals.size else np.nan,
                'node_median_frequency_min': float(np.min(vals)) if vals.size else np.nan,
                'node_median_frequency_max': float(np.max(vals)) if vals.size else np.nan,
                'interpretation': 'descriptive sensitivity over nodes meeting the vertical support threshold; frequencies remain scenario conditional',
            })
    summary = pd.DataFrame(records)
    summary_path = OUT / 'mobility_frequency_by_nearbed_support_threshold.csv'
    summary.to_csv(summary_path, index=False)

    # Visualize the mismatch directly; points are model forcing nodes, not bathymetric observations.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(nodes.longitude_deg, nodes.latitude_deg,
                    c=nodes.vertical_gap_bottom_minus_current_m, s=56,
                    cmap='magma_r', vmin=0, vmax=max(70, float(nodes.vertical_gap_bottom_minus_current_m.max())),
                    edgecolors='white', linewidths=.55, zorder=3)
    ax.axhline(np.nan, color='none')
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.grid(color='#d7dee5', linewidth=.6, alpha=.8); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    cb=fig.colorbar(sc, ax=ax, pad=.025); cb.set_label('Vertical gap: bed depth − sampled current level (m)')
    n2=int((nodes.vertical_gap_bottom_minus_current_m<=2).sum())
    n5=int((nodes.vertical_gap_bottom_minus_current_m<=5).sum())
    fig.text(.5,.015,f'{n2}/{len(nodes)} nodes within 2 m; {n5}/{len(nodes)} within 5 m. Larger gaps are not near-bed current support.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ('png','pdf'):
        fig.savefig(OUT / f'Figure_mobility_current_vertical_support.{ext}', dpi=400, bbox_inches='tight')
    plt.close(fig)
    (OUT/'figure_caption_EN.txt').write_text(
        'Figure S1. Difference between the EMODnet-derived water depth and the deepest IBI current level selected by the existing mobility workflow at each paired forcing node. A small vertical gap is necessary for a bed-stress interpretation but does not establish that the regional current represents the bottom boundary layer. Only four of 44 nodes are within 2 m and nine of 44 within 5 m. Nodes outside these thresholds are excluded from near-bed summaries. The displayed mobility frequencies remain conditional on prescribed parameter scenarios and are not probabilities or observed sediment transport.\n', encoding='utf-8')
    (OUT/'interpretation_notes_FR.txt').write_text(
        'Audit vertical de soutien du calcul de mobilité. L’écart est défini comme la profondeur nominale issue de la grille EMODnet moins la profondeur positive vers le bas du niveau de courant IBI utilisé. Un grand écart signifie que le courant échantillonné est loin du fond et ne doit pas être interprété comme courant près du lit. Sur 44 nœuds, seuls 4 sont à ≤2 m du fond, 9 à ≤5 m et 11 à ≤10 m. Le champ original de mobilité mélangeait donc des niveaux de soutien vertical très différents. Les tableaux filtrés fournissent une sensibilité descriptive pour les seuils de 2, 5 et 10 m; ils ne remplacent pas une interpolation verticale validée du courant au fond. Même les nœuds qui satisfont ces seuils restent soumis à l’incertitude de datum, à la résolution régionale, à l’absence de transformation de houle en zone de surf et aux D50 hypothétiques. Aucune fréquence ne constitue une probabilité, un flux ou un transport mesuré.\n', encoding='utf-8')
    print('Mobility current vertical-support audit complete; source data and scenario outputs were read only.')
    print(f'Unique forcing nodes: {len(nodes)}; within 2 m: {n2}; within 5 m: {n5}; within 10 m: {int((nodes.vertical_gap_bottom_minus_current_m<=10).sum())}')
    print(f'Node audit: {node_csv}')
    print(f'Threshold summary: {summary_path}')
    print(f'Figure: {OUT / "Figure_mobility_current_vertical_support.png"} and PDF')

if __name__ == '__main__':
    main()
