from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Results/Figures/Publication'
BASE = ROOT / 'Results/Bathymetry/ATL24_Sentinel2_IBI_SSH_24h/Unique_Photon_Audit'
SPATIAL = BASE / 'Spatial_Block_Validation'
PAIRED = BASE / 'Paired_Target_Comparison'
MOB = ROOT / 'Results/Sediment_Mobility'
SUPPORT = MOB / 'Forcing_Support_Audit'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
                     'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
NAVY='#214E70'; BLUE='#347DA3'; ORANGE='#D47736'; TEAL='#3C8C89'; GRAY='#5B6570'; LIGHT='#EEF3F6'; LAND='#D9D6CF'


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=400, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def draw_workflow():
    fig, ax=plt.subplots(figsize=(11.4,4.4)); ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis('off')
    def box(x,y,w,h,title,body,color):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08',
            linewidth=1.2,edgecolor=color,facecolor='white')
        ax.add_patch(patch); ax.text(x+w/2,y+h-.32,title,ha='center',va='center',weight='bold',color=color,fontsize=9)
        ax.text(x+.16,y+h-.68,body,ha='left',va='top',fontsize=7.4,color='#263238',linespacing=1.45)
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,
                                     linewidth=1.1,color='#73808a'))
    box(.2,3.10,2.45,1.42,'OPEN OBSERVATIONS','ICESat-2 ATL24 photons\nSentinel-2 reflectance and SCL\nRegional IBI sea level, waves, currents',BLUE)
    box(3.10,3.10,2.45,1.42,'SUPPORT AUDIT','Time-gap classes (0–6 h; 6–24 h)\nUnique photon keys\nGrid distance and vertical offset',TEAL)
    box(6.0,3.10,2.45,1.42,'TRANSFER EVALUATION','Leave-granule-out baseline\nBuffered 1–2 km spatial blocks\nPaired raw/relative-SSH targets',ORANGE)
    box(8.95,3.10,2.75,1.42,'MOBILITY SCREEN','EMODnet depth; scenario D₅₀\nWave–current stress and Shields\nRetain only forcing support tiers',NAVY)
    arrow(2.68,3.82,3.02,3.82); arrow(5.58,3.82,5.92,3.82); arrow(8.48,3.82,8.88,3.82)
    box(1.05,.67,3.45,1.40,'REPORTABLE RESULT','Depth-specific predictive skill\nUncertainty conditional on folds\nNo validated continuous SDB map', '#4D6678')
    box(5.05,.67,3.45,1.40,'MOBILITY INTERPRETATION','Four of 44 nodes within 2 m of bed\nNine of 44 within 5 m\nNo coast-wide bed-stress inference', '#4D6678')
    box(9.05,.67,2.40,1.40,'NEXT EVIDENCE','Local control depths\nGauge/datum control\nNear-bed and grain data', '#4D6678')
    arrow(7.25,3.02,2.75,2.15); arrow(10.30,3.02,6.75,2.15); arrow(8.62,1.37,8.97,1.37)
    ax.text(6,.19,'Evidence gates prevent unsupported extrapolation from optical pixels to sediment transport.',ha='center',va='center',fontsize=8,color=GRAY)
    save(fig,'Figure_1_support_aware_workflow')


def load_bathy():
    f=ROOT/'Data/Raw/Bathymetry/emodnet_mean_2022_anza.tif'
    with rasterio.open(f) as ds:
        a=ds.read(1).astype('float32'); b=ds.bounds
        if ds.nodata is not None: a[a==ds.nodata]=np.nan
    # this selected subset has geographic bounds, increasing x and decreasing y
    a=np.where(a<0,-a,np.nan)
    return a,(b.left,b.right,b.bottom,b.top)


def draw_support_map():
    matches=pd.read_csv(BASE/'nearest_scene_unique_photon_pairs.csv.gz',usecols=['granule_id','beam','source_photon_index','longitude_deg','latitude_deg','matchup_window_class','maximum_grid_distance_km'])
    matches=matches.loc[matches.maximum_grid_distance_km.eq(5)].drop_duplicates(['granule_id','beam','source_photon_index'])
    nodes=pd.read_csv(SUPPORT/'mobility_current_vertical_support_by_node.csv')
    bathy,extent=load_bathy(); fig,axes=plt.subplots(1,2,figsize=(11.2,4.45),sharex=True,sharey=True)
    for ax in axes:
        ax.set_facecolor(LAND)
        im=ax.imshow(bathy,extent=extent,origin='upper',cmap='Blues_r',vmin=0,vmax=100,interpolation='nearest',aspect='auto')
        ax.set_xlim(-9.85,-9.55);ax.set_ylim(30.35,30.55)
        ax.set_xlabel('Longitude (°E)',fontsize=8);ax.grid(color='white',alpha=.55,linewidth=.45)
        ax.tick_params(length=2,labelsize=7)
    axes[0].scatter(matches.loc[matches.matchup_window_class.eq('primary_0_6h'),'longitude_deg'],matches.loc[matches.matchup_window_class.eq('primary_0_6h'),'latitude_deg'],s=3,c=ORANGE,alpha=.62,linewidths=0,rasterized=True,label='ATL24–Sentinel-2 pair, 0–6 h')
    axes[0].scatter(matches.loc[matches.matchup_window_class.eq('extended_6_24h'),'longitude_deg'],matches.loc[matches.matchup_window_class.eq('extended_6_24h'),'latitude_deg'],s=3,c=BLUE,alpha=.48,linewidths=0,rasterized=True,label='ATL24–Sentinel-2 pair, 6–24 h')
    axes[0].legend(loc='lower left',frameon=True,framealpha=.95,edgecolor='white',markerscale=1.8,fontsize=6.5)
    sc=axes[1].scatter(nodes.longitude_deg,nodes.latitude_deg,c=nodes.vertical_gap_bottom_minus_current_m,s=35,marker='^',cmap='magma_r',vmin=0,vmax=70,edgecolor='white',linewidth=.45,zorder=3)
    axes[1].axhline(np.nan,color='none')
    axes[0].text(.02,.97,'A',transform=axes[0].transAxes,ha='left',va='top',weight='bold',fontsize=11,color=NAVY)
    axes[1].text(.02,.97,'B',transform=axes[1].transAxes,ha='left',va='top',weight='bold',fontsize=11,color=NAVY)
    axes[0].set_ylabel('Latitude (°N)',fontsize=8)
    axes[0].set_title('ATL24–Sentinel-2 matchup support',fontsize=9,weight='bold',pad=6)
    axes[1].set_title('IBI current-level support relative to bed',fontsize=9,weight='bold',pad=6)
    # Put separate horizontal scales below each panel to avoid squeezing long labels
    # between the maps, which caused the original color bars to overlap.
    cb=fig.colorbar(im,ax=axes[0],orientation='horizontal',fraction=.065,pad=.16,aspect=32)
    cb.set_label('EMODnet depth (m; display only)',fontsize=7,labelpad=3);cb.ax.tick_params(labelsize=6.5,length=2)
    cb2=fig.colorbar(sc,ax=axes[1],orientation='horizontal',fraction=.065,pad=.16,aspect=32)
    cb2.set_label('IBI current level above nominal bed (m)',fontsize=7,labelpad=3);cb2.ax.tick_params(labelsize=6.5,length=2)
    fig.subplots_adjust(wspace=.15,bottom=.21,left=.08,right=.98,top=.91)
    save(fig,'Figure_2_spatial_and_vertical_support')


def draw_spatial_transfer():
    d=pd.read_csv(SPATIAL/'spatial_block_transfer_summary.csv')
    d=d[(d.maximum_grid_distance_km==5)&(d.model=='Multiband_RandomForest')&(d.depth_stratum_raw_ATL24=='all_depths')]
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.65),sharey=True)
    specs=[('primary_0_6h','Near-synchronous (0–6 h)'),('combined_0_24h','Combined window (0–24 h)')]
    for ax,(window,label) in zip(axes,specs):
        sub=d[d.window_analysis.eq(window)]
        for target,color,marker,lab in [('raw_instantaneous_depth',BLUE,'o','Raw ATL24 depth'),('IBI_zos_relative_adjustment',ORANGE,'s','Relative IBI SSH target')]:
            q=sub[sub.target_scenario.eq(target)].sort_values('block_size_m')
            xs=q.block_size_m.to_numpy()/1000 + (-.055 if target=='raw_instantaneous_depth' else .055)
            y=q.equal_block_skill_vs_training_median.to_numpy(); lo=q.equal_block_skill_CI95_low.to_numpy(); hi=q.equal_block_skill_CI95_high.to_numpy()
            ax.errorbar(xs,y,yerr=np.vstack((y-lo,hi-y)),marker=marker,color=color,lw=1.5,ms=5,capsize=3,label=lab)
        ax.axhline(0,color='#333333',lw=.8,ls='--');ax.set_xticks([1,2],['1 km','2 km']);ax.set_xlabel('Block width (buffer = half width)')
        ax.set_title(label,fontsize=9,weight='bold',pad=7);ax.grid(axis='y',color='#d9dee3',lw=.55);ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.text(.02,.02,'95% block-bootstrap intervals',transform=ax.transAxes,fontsize=7,color=GRAY)
    axes[0].set_ylabel('Equal-block RMSE skill vs training median')
    axes[1].legend(frameon=False,loc='upper left')
    fig.tight_layout(w_pad=1.3);save(fig,'Figure_3_spatial_transfer_skill')


def draw_paired_target():
    d=pd.read_csv(PAIRED/'paired_raw_vs_ibi_depth_skill.csv')
    d=d[(d.maximum_grid_distance_km==5)&(d.window_analysis.eq('combined_0_24h'))&(d.model.eq('Multiband_RandomForest'))].copy()
    order=['0-2 m','2-5 m','5-10 m','10-15 m','15-20 m','20-40 m'];d['depth_bin']=pd.Categorical(d.depth_bin,categories=order,ordered=True);d=d.sort_values('depth_bin')
    fig,ax=plt.subplots(figsize=(7.5,3.7));x=np.arange(len(d));y=d.paired_delta_skill_adjusted_minus_raw.to_numpy();lo=d.paired_delta_skill_CI95_low.to_numpy();hi=d.paired_delta_skill_CI95_high.to_numpy()
    colors=[TEAL if a>0 and b>0 else ORANGE if a<0 and b<0 else BLUE for a,b in zip(lo,hi)]
    for i in range(len(d)):
        ax.errorbar(x[i],y[i],yerr=[[y[i]-lo[i]],[hi[i]-y[i]]],fmt='o',ms=5,color=colors[i],capsize=3,lw=1.3)
    ax.axhline(0,color='#333333',lw=.8,ls='--');ax.set_xticks(x,[f'{v}\nn={n}' for v,n in zip(d.depth_bin.astype(str),d.common_unique_photons)])
    ax.set_ylabel('Paired Δ RMSE skill (relative-SSH target − raw depth)');ax.set_xlabel('Depth stratum defined by raw ATL24 depth')
    ax.grid(axis='y',color='#d9dee3',lw=.55);ax.set_axisbelow(True);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.text(.99,.02,'Intervals resample ATL24 granules; 20–40 m has only 27 photons / 4 granules.',transform=ax.transAxes,ha='right',fontsize=7,color=GRAY)
    fig.tight_layout();save(fig,'Figure_4_paired_relative_ssh_sensitivity')


def draw_mobility_support_sensitivity():
    d=pd.read_csv(SUPPORT/'mobility_frequency_by_nearbed_support_threshold.csv')
    fig,ax=plt.subplots(figsize=(7.2,3.7));markers={2:'o',5:'s',10:'^'};colors={2:TEAL,5:BLUE,10:ORANGE}
    for tol,g in d.groupby('maximum_current_level_offset_above_bed_m'):
        g=g.sort_values('d50_scenario_mm');x=np.arange(len(g))+(int(tol)-5)*.018
        ax.errorbar(x,g.node_median_frequency_median,yerr=[g.node_median_frequency_median-g.node_median_frequency_min,g.node_median_frequency_max-g.node_median_frequency_median],marker=markers[int(tol)],color=colors[int(tol)],capsize=3,lw=1.2,ms=4,label=f'Current level ≤{int(tol)} m above bed (n={int(g.supported_nodes.iloc[0])})')
    ax.set_xticks(range(4),[f'{v:g}' for v in sorted(d.d50_scenario_mm.unique())]);ax.set_xlabel('Assumed median grain diameter D₅₀ (mm)');ax.set_ylabel('Median scenario exceedance frequency across supported nodes')
    ax.set_ylim(0,1);ax.grid(axis='y',color='#d9dee3',lw=.55);ax.set_axisbelow(True);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.legend(frameon=False,loc='upper right')
    ax.text(.01,.02,'Bars show min–max across nodes, not confidence intervals.',transform=ax.transAxes,fontsize=7,color=GRAY)
    fig.tight_layout();save(fig,'Figure_S1_mobility_support_sensitivity')


def export_tables():
    spatial=pd.read_csv(SPATIAL/'spatial_block_transfer_summary.csv')
    keep=spatial[(spatial.maximum_grid_distance_km==5)&(spatial.model.eq('Multiband_RandomForest'))&(spatial.depth_stratum_raw_ATL24.eq('all_depths'))].copy()
    keep.to_csv(OUT/'Table_2_spatial_block_results.csv',index=False)
    paired=pd.read_csv(PAIRED/'paired_raw_vs_ibi_depth_skill.csv')
    paired=paired[(paired.maximum_grid_distance_km==5)&(paired.window_analysis.eq('combined_0_24h'))&(paired.model.eq('Multiband_RandomForest'))].copy()
    paired.to_csv(OUT/'Table_3_paired_ssh_target_results.csv',index=False)
    group=pd.read_csv(ROOT/'Results/Bathymetry/ATL24_Sentinel2_IBI_SSH_24h/Transferability_Audit/sdb_grouped_transferability_overall.csv')
    group=group[(group.maximum_grid_distance_km==5)&(group.model.eq('Multiband_RandomForest'))].copy()
    group.to_csv(OUT/'Table_S1_leave_granule_out_metrics.csv',index=False)
    mobility=pd.read_csv(SUPPORT/'mobility_frequency_by_nearbed_support_threshold.csv')
    mobility.to_csv(OUT/'Table_S2_mobility_support_tiers.csv',index=False)
    inventory=pd.DataFrame([
      ['ICESat-2 ATL24 v003','94 complete granules in downloaded inventory; matchup tables retain unique photons by granule, beam, and source index','Along-track bathymetry and sea surface; official product includes orthometric/ellipsoidal heights and instantaneous depth','Reference observations, not error-free continuous truth'],
      ['Sentinel-2 L2A','26 scene IDs represented in the ATL24–S2 matchup table','Visible/NIR reflectance and water/cloud QA','Coastal spectral retrieval; temporal gaps are separated'],
      ['IBI hourly sea-level height zos','Hourly; product resolution 0.027°; 2018-10-14 to 2026-08-25 requested','Regional relative sea-level change only','No local gauge validation or absolute datum tie'],
      ['IBI waves','Hourly; product resolution 0.027°','Significant wave height, peak period, direction','Regional forcing; not surf-zone transformed'],
      ['IBI currents','Daily; vertical levels requested 0–30 m','Eastward and northward currents','44 wave nodes; sampled-level-to-bed offset is audited'],
      ['EMODnet DTM 2022','Subset on 0.3° × 0.2° AOI; nominal raster spacing ~0.00104°','Regional depth context and mobility input','Vertical reference and local equivalence are not assumed'],
    ],columns=['dataset','study_support','role','interpretation_limit'])
    inventory.to_csv(OUT/'Table_1_data_inventory.csv',index=False)


def main():
    draw_workflow();draw_support_map();draw_spatial_transfer();draw_paired_target();draw_mobility_support_sensitivity();export_tables()
    captions={
      'Figure_1_support_aware_workflow':'A support-aware analysis sequence linking open bathymetric observations, optical matchups, blocked evaluation, relative sea-level target sensitivity and conditional sediment-mobility screening. The evidence gates distinguish valid model targets from physical claims.',
      'Figure_2_spatial_and_vertical_support':'Study-area support. (A) ATL24–Sentinel-2 photon matchups over the EMODnet DTM background, separated into 0–6 h and 6–24 h windows. (B) IBI wave/current forcing nodes colored by the vertical distance between the selected current level and the nominal EMODnet seabed. The bathymetric colors are shown for context and do not imply a verified datum tie.',
      'Figure_3_spatial_transfer_skill':'Buffered spatial-block RMSE skill of the multiband random-forest model relative to the within-fold training median at a 5 km maximum IBI-grid distance. The primary 0–6 h window is shown separately from the combined 0–24 h window. Intervals are block-bootstrap intervals for equal-block-weighted skill; the 0–24 h estimate includes 6–24 h temporal mismatches and is exploratory.',
      'Figure_4_paired_relative_ssh_sensitivity':'Paired difference in out-of-fold RMSE skill between the relative-IBI-SSH-adjusted and raw ATL24 depth targets, evaluated on the same photons for the combined 0–24 h window. Positive values favor the adjusted target; negative values favor raw depth. Intervals resample ATL24 granules. Deepest-bin support is sparse and neither target is an independently validated vertical truth.',
      'Figure_S1_mobility_support_sensitivity':'Scenario exceedance frequency summaries after restricting the mobility table to forcing nodes with the sampled IBI current level no more than 2, 5 or 10 m above the nominal EMODnet bed. Points show the median across supported nodes and bars the observed spatial range across those nodes. These are not confidence intervals, event probabilities or observed sediment transport.'
    }
    (OUT/'figure_captions_EN.txt').write_text('\n\n'.join(f'{k}: {v}' for k,v in captions.items())+'\n',encoding='utf-8')
    print('Publication figures and evidence tables written to',OUT)
    print('Figures: 4 main + 1 supplementary; data tables: 5')

if __name__=='__main__': main()
