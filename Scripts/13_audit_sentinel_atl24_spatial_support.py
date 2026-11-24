from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "Data/Manifests/sentinel2_atl24_pm6h_selection.csv"
ATL24 = ROOT / "Data/Processed/ATL24/atl24_bathy_points.csv.gz"
PROCESSED = ROOT / "Data/Processed/Sentinel2"
OUT = ROOT / "Results/Bathymetry/Sentinel2_ATL24_Spatial_Support"
TEMPORAL_WINDOW_HOURS = 24.0
NEAR_WATER_THRESHOLDS_M = (10.0, 20.0, 30.0)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    if not SELECTION.exists() or not ATL24.exists():
        raise SystemExit("Required selection or ATL24 point table is missing. Run Scripts 04d and 06 first.")
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = pd.read_csv(SELECTION, encoding="utf-8-sig")
    scenes = scenes.loc[scenes["selected"].astype(str).str.upper().eq("YES")].copy()
    if scenes.empty:
        raise SystemExit("No scenes are selected in the dedicated ATL24 matchup file.")
    points = pd.read_csv(ATL24)
    points["utc_time_source_epoch"] = pd.to_datetime(points["utc_time_source_epoch"], utc=True, errors="coerce")
    point_rows: list[dict] = []
    scene_rows: list[dict] = []
    for scene in scenes.itertuples(index=False):
        scene_time = pd.to_datetime(scene.acquisition_utc, utc=True)
        nearby = points.loc[(points["utc_time_source_epoch"] - scene_time).abs() <= pd.Timedelta(hours=TEMPORAL_WINDOW_HOURS)].copy()
        date_tag = str(scene.acquisition_date).replace("-", "")
        prefix = f"{date_tag}_{safe_name(str(scene.item_id))}"
        mask_path = PROCESSED / f"{prefix}_watermask.tif"
        if not mask_path.exists():
            scene_rows.append({"scene_id": scene.item_id, "date": scene.acquisition_date, "temporal_candidate_points": len(nearby), "status": "missing_processed_watermask"})
            continue
        with rasterio.open(mask_path) as ds:
            water = ds.read(1) == 1
            px_x = abs(float(ds.transform.a))
            px_y = abs(float(ds.transform.e))
            distance = distance_transform_edt(~water, sampling=(px_y, px_x))
            if ds.crs is None:
                raise SystemExit(f"Water-mask raster has no CRS: {mask_path}")
            if not ds.crs.is_projected:
                raise SystemExit(f"Expected a projected water-mask grid for distance calculations: {mask_path}")
            transformer = Transformer.from_crs("EPSG:32629", ds.crs, always_xy=True)
            xx, yy = transformer.transform(nearby["easting_m"].to_numpy(float), nearby["northing_m"].to_numpy(float))
            rows, cols = rasterio.transform.rowcol(ds.transform, xx, yy)
            rows, cols = np.asarray(rows), np.asarray(cols)
            inside = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
            sampled_water = np.zeros(len(nearby), dtype=bool)
            nearest_water_m = np.full(len(nearby), np.nan, dtype="float64")
            sampled_water[inside] = water[rows[inside], cols[inside]]
            nearest_water_m[inside] = distance[rows[inside], cols[inside]]

        nearby["inside_sentinel_raster"] = inside
        nearby["exact_clear_water_pixel"] = sampled_water
        nearby["distance_to_nearest_clear_water_m"] = nearest_water_m
        for granule, group in nearby.groupby("granule_id", sort=True):
            d = group["distance_to_nearest_clear_water_m"].to_numpy(float)
            valid_d = d[np.isfinite(d)]
            exact = group["exact_clear_water_pixel"].to_numpy(bool)
            row = {
                "scene_id": scene.item_id,
                "scene_date": scene.acquisition_date,
                "atl24_granule_id": granule,
                "scene_utc": scene.acquisition_utc,
                "photons_within_24h": int(len(group)),
                "points_inside_sentinel_raster": int(group["inside_sentinel_raster"].sum()),
                "exact_clear_water_pixels": int(exact.sum()),
                "median_distance_to_clear_water_m": float(np.median(valid_d)) if len(valid_d) else float("nan"),
                "p90_distance_to_clear_water_m": float(np.percentile(valid_d, 90)) if len(valid_d) else float("nan"),
                "clear_water_pixels_within_10m": int(np.sum(valid_d <= 10.0)),
                "clear_water_pixels_within_20m": int(np.sum(valid_d <= 20.0)),
                "clear_water_pixels_within_30m": int(np.sum(valid_d <= 30.0)),
                "fraction_with_exact_clear_water": float(exact.mean()) if len(exact) else float("nan"),
                "watermask_clear_water_pixel_count": int(water.sum()),
                "raster_resolution_x_m": px_x,
                "raster_resolution_y_m": px_y,
            }
            row["interpretation_class"] = (
                "exact_spatial_overlap" if row["exact_clear_water_pixels"] >= 20
                else "limited_exact_overlap" if row["exact_clear_water_pixels"] > 0
                else "no_clear_water_within_30m" if row["clear_water_pixels_within_30m"] == 0
                else "nearby_water_only_requires_registration_review"
            )
            point_rows.append(row)
        d_all = nearest_water_m[np.isfinite(nearest_water_m)]
        scene_rows.append({
            "scene_id": scene.item_id,
            "date": scene.acquisition_date,
            "temporal_candidate_points_within_24h": int(len(nearby)),
            "atl24_granules_with_temporal_candidates": int(nearby["granule_id"].nunique()),
            "points_inside_raster": int(inside.sum()),
            "exact_clear_water_points": int(sampled_water.sum()),
            "points_with_clear_water_within_10m": int(np.sum(d_all <= 10.0)),
            "points_with_clear_water_within_20m": int(np.sum(d_all <= 20.0)),
            "points_with_clear_water_within_30m": int(np.sum(d_all <= 30.0)),
            "median_distance_to_clear_water_m": float(np.median(d_all)) if len(d_all) else float("nan"),
            "p90_distance_to_clear_water_m": float(np.percentile(d_all, 90)) if len(d_all) else float("nan"),
            "clear_water_pixel_count": int(water.sum()),
            "status": "audited",
        })

    by_granule = pd.DataFrame(point_rows)
    by_scene = pd.DataFrame(scene_rows)
    by_granule.to_csv(OUT / "atl24_scene_granule_spatial_support.csv", index=False)
    by_scene.to_csv(OUT / "atl24_scene_spatial_support_summary.csv", index=False)
    summary = {
        "temporal_window_hours": TEMPORAL_WINDOW_HOURS,
        "scene_count": int(len(scenes)),
        "exact_water_support_is_required_for_primary_sdb": True,
        "near_water_distance_thresholds_m": list(NEAR_WATER_THRESHOLDS_M),
        "distance_method": "Euclidean distance to the nearest SCL-class-6 clear-water pixel on the projected Sentinel-2 grid; no point was shifted or reassigned.",
        "caution": "Near-water counts are sensitivity diagnostics only. They were not used to create training pairs; spatial reassignment requires a justified co-registration/error model and must be tested separately.",
        "outputs": ["atl24_scene_granule_spatial_support.csv", "atl24_scene_spatial_support_summary.csv", "Figure_Sentinel2_ATL24_spatial_support.png", "Figure_Sentinel2_ATL24_spatial_support.pdf"],
    }
    (OUT / "spatial_support_audit_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not by_scene.empty:
        fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
        plot = by_scene.sort_values("date").copy()
        ypos = np.arange(len(plot))
        ax.barh(ypos, plot["exact_clear_water_points"], color="#2A7F83", alpha=0.88, label="Exact clear-water pixel")
        ax.scatter(plot["points_with_clear_water_within_30m"], ypos, color="#D9862C", marker="|", s=130, linewidths=2, label="Within 30 m (sensitivity only)")
        ax.set_yticks(ypos, plot["date"])
        ax.set_xlabel("ATL24 photons spatially matched to Sentinel-2 clear water")
        ax.set_title("Spatial support of temporally matched ATL24–Sentinel-2 scenes", loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        fig.savefig(OUT / "Figure_Sentinel2_ATL24_spatial_support.png", dpi=350, bbox_inches="tight")
        fig.savefig(OUT / "Figure_Sentinel2_ATL24_spatial_support.pdf", bbox_inches="tight")
        plt.close(fig)

    notes = [
        "Contrôle de support spatial ATL24–Sentinel‑2",
        "",
        "L’appariement temporel des scènes à des photons du même intervalle de date ne garantit pas que ceux-ci se trouvent dans des pixels d’eau observés par Sentinel‑2. Le script évalue le pointage exact et la distance au pixel d’eau clair le plus proche, sans déplacer les points.",
        "Les distances de 10, 20 et 30 m sont des diagnostics de sensibilité, pas des appariements admis dans l’analyse principale. Une éventuelle recherche dans le voisinage exige un modèle de co-enregistrement et un contrôle que le pixel reste dans le même contexte marin.",
        "Une scène avec des photons distants de plusieurs kilomètres de tout pixel d’eau clair est exclue du calage SDB, même si son acquisition est proche dans le temps. Cette exclusion traduit une absence de support spatial commun, et non nécessairement la présence de nuages.",
        "Les points ATL24 et les pixels Sentinel‑2 ne sont pas traités comme des observations indépendantes. Le contrôle n’évalue ni la précision absolue des profondeurs ni la qualité du datum.",
    ]
    (OUT / "interpretation_notes_FR.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("ATL24–Sentinel-2 spatial-support audit complete; imagery and photon coordinates were read only.")
    print(f"Selected scenes: {len(scenes)}; audited scene/granule pairs: {len(by_granule)}")
    print(f"Summary: {OUT / 'atl24_scene_spatial_support_summary.csv'}")
    print(f"Per-granule table: {OUT / 'atl24_scene_granule_spatial_support.csv'}")
    print(f"Figure: {OUT / 'Figure_Sentinel2_ATL24_spatial_support.png'} and PDF")
    print("Temporal proximity is not spatial co-location; no nearby pixels were substituted for exact point samples.")


if __name__ == "__main__":
    main()
